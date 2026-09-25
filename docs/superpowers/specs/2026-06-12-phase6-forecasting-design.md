# Phase 6: Score Extrapolation Forecast — Design Spec

**Date:** 2026-06-12  
**Project:** EU-GNN-Risk-Monitor  
**Phase:** 6 of n  

---

## Goal

Extend the dashboard with a per-region 3-year score forecast (2024–2026), surfaced as a dashed line + shaded confidence band appended to the existing time series chart behind a "Show forecast" toggle.

---

## Context

The Phase 5 dashboard is a 3-panel layout:

```
[ EU Choropleth Map (~50%) ] [ Time Series Panel (~25%) ] [ Ranking / Compare Panel (~25%) ]
```

- 242 NUTS2 regions, 14 years of actual anomaly scores (2010–2023) in `risk_scores.json`
- `TimeSeriesPanel` renders a Recharts `ComposedChart` for the selected region
- All data is static JSON loaded once in `useDashboardData`
- No live API calls; all forecasts must be pre-computed Python-side

---

## Approach

**Score extrapolation via OLS linear regression.** For each region, fit a linear trend on the 14-year anomaly score series and project 3 years forward with 95% prediction intervals. Output a companion `forecast_scores.json`. The dashboard's `TimeSeriesPanel` gains a toggle to extend the chart.

This is the only defensible approach given:
- 14 data points is too few for ARIMA or any ML forecasting
- The GNN's GRU encoder has no forward-projection capability
- The anomaly score series is the real model output; projecting features through the GNN compounds uncertainty non-transparently

---

## Python Side — `export_dashboard_data.py`

### New function: `compute_forecasts`

Added after all existing export logic. Takes the already-computed `region_scores` dict (or reads from `risk_scores.json`) and writes `data/processed/forecast_scores.json`.

**Algorithm per region:**

1. Extract `years` (int array) and `scores` (float array) from the region's 14-year series.
2. Fit `sklearn.linear_model.LinearRegression` on `years.reshape(-1,1)` → `scores`.
3. Compute residual standard error: `se = sqrt(sum(residuals²) / (n - 2))` where `n = 14`.
4. For each forecast year `x_f ∈ {2024, 2025, 2026}`:
   - Point estimate: `ŷ = model.predict([[x_f]])[0]`
   - 95% prediction interval half-width:
     `margin = t₀.₉₇₅,ₙ₋₂ × se × sqrt(1 + 1/n + (x_f − x̄)² / Sxx)`
   - `t` value from `scipy.stats.t.ppf(0.975, df=n-2)` (scipy is a sklearn dependency)
   - Clamp `lower = max(0.0, ŷ - margin)`
5. Round to 4 decimal places.

### Output format

`data/processed/forecast_scores.json`:

```json
{
  "PT15": [
    {"year": 2024, "point": 0.1831, "lower": 0.0912, "upper": 0.2750},
    {"year": 2025, "point": 0.1763, "lower": 0.0742, "upper": 0.2784},
    {"year": 2026, "point": 0.1695, "lower": 0.0563, "upper": 0.2827}
  ],
  "SK01": [...]
}
```

### Python test

`tests/test_forecast.py` — one unit test: construct a perfectly linear series (score = 0.01 × year + constant), call `compute_forecasts`, assert point estimates match the exact linear extrapolation within 1e-4, assert `lower < point < upper` for all 3 years.

---

## Dashboard Types — `types.ts`

Add:

```typescript
export interface ForecastPoint {
  year: number;
  point: number;
  lower: number;
  upper: number;
}
```

Add to `DashboardData`:

```typescript
forecastScores: Record<string, ForecastPoint[]>;
```

---

## Data Loading — `useDashboardData.ts`

Add a fourth `fetch` call to `forecast_scores.json`. On 404 or parse error, default to `{}` — the toggle will simply have no data to show and remains inert.

```typescript
const forecastScores: Record<string, ForecastPoint[]> = await fetch('forecast_scores.json')
  .then(r => r.ok ? r.json() : {})
  .catch(() => ({}));
```

---

## TimeSeriesPanel Changes — `TimeSeriesPanel.tsx`

### New prop

```typescript
forecastScores: Record<string, ForecastPoint[]>;
```

### Local state

```typescript
const [showForecast, setShowForecast] = useState(false);
```

Reset `showForecast` to `false` when `selectedRegion` changes (via `useEffect` on `selectedRegion`), so switching regions doesn't leave forecast toggled on for a region with no forecast data.

### Derived value

```typescript
const hasForecast = !!selectedRegion && (forecastScores[selectedRegion]?.length ?? 0) > 0;
```

### Toggle

Small checkbox in the panel header (right side), visible only when the selected region has forecast data:

```tsx
{hasForecast && (
  <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
    <input
      type="checkbox"
      checked={showForecast}
      onChange={e => setShowForecast(e.target.checked)}
      className="accent-blue-500"
    />
    Forecast
  </label>
)}
```

### Chart data construction

Combine actual + forecast into one array for Recharts:

```typescript
type ChartRow = {
  year: number;
  score?: number;        // actual GNN score
  fPoint?: number;       // forecast point estimate
  fLower?: number;       // forecast lower bound
  fBand?: number;        // upper − lower (for stacked Area)
};
```

- Actual years: `score = anomaly_score`, `fPoint/fLower/fBand = undefined`
- Bridge point (year 2023): `score = anomaly_score`, `fPoint = anomaly_score`, `fLower = anomaly_score`, `fBand = 0` — connects the solid line to the dashed line seamlessly
- Forecast years 2024–2026: `score = undefined`, `fPoint = point`, `fLower = lower`, `fBand = upper − lower`

### Recharts additions (all inside existing `<ComposedChart>`)

```tsx
{/* Confidence band — stacked areas */}
{showForecast && (
  <>
    <Area
      dataKey="fLower"
      stroke="none"
      fill="transparent"
      stackId="band"
      isAnimationActive={false}
    />
    <Area
      dataKey="fBand"
      stroke="none"
      fill="#3b82f6"
      fillOpacity={0.15}
      stackId="band"
      isAnimationActive={false}
    />
    {/* Dashed forecast line */}
    <Line
      dataKey="fPoint"
      stroke="#3b82f6"
      strokeWidth={1.5}
      strokeDasharray="5 3"
      dot={false}
      isAnimationActive={false}
    />
    {/* Actual/forecast separator */}
    <ReferenceLine
      x={2023}
      stroke="#374151"
      strokeDasharray="2 2"
    />
  </>
)}
```

### Attribution note

Below the chart, when `showForecast` is true:

```tsx
{showForecast && (
  <p className="text-xs text-gray-500 mt-1">
    Projected 2024–2026 · OLS linear trend · 95% prediction interval
  </p>
)}
```

---

## App.tsx Changes

Pass `forecastScores` from `data` to `TimeSeriesPanel`:

```tsx
<TimeSeriesPanel
  data={data}
  selectedRegion={selectedRegion}
  activeYear={activeYear}
  forecastScores={data.forecastScores}   {/* new */}
/>
```

---

## React Tests — `TimeSeriesPanel.test.tsx`

Two new tests added to the existing test file:

1. **Toggle absent when no forecast data** — render with `forecastScores={}`, assert "Forecast" label is not in the document.
2. **Toggle present and controls chart data** — render with mock `forecastScores` for `selectedRegion`, assert "Forecast" label present; fire toggle → assert chart data now contains a row with `year=2024` and a defined `fPoint`.

---

## Files Changed

| File | Type | Change |
|---|---|---|
| `scripts/export_dashboard_data.py` | modify | Add `compute_forecasts()` function + output `forecast_scores.json` |
| `tests/test_forecast.py` | **new** | Unit test for `compute_forecasts` |
| `dashboard/src/types.ts` | modify | Add `ForecastPoint` interface + `forecastScores` field on `DashboardData` |
| `dashboard/src/hooks/useDashboardData.ts` | modify | Fetch `forecast_scores.json`, default to `{}` on failure |
| `dashboard/src/components/TimeSeriesPanel.tsx` | modify | `showForecast` toggle, bridge point logic, dashed line + band Recharts layers |
| `dashboard/src/components/TimeSeriesPanel.test.tsx` | modify | 2 new tests for forecast toggle |
| `dashboard/src/App.tsx` | modify | Pass `forecastScores` prop to `TimeSeriesPanel` |
| `data/processed/forecast_scores.json` | **new** | Pre-computed forecasts for all 242 regions |

---

## Out of Scope (Phase 6)

- Per-feature trend decomposition (what's driving the forecast direction)
- Forecast export as CSV
- Map-level forecast year extension (slider past 2023)
- Country-level aggregate forecasts
- Non-linear or ML-based forecasting methods
