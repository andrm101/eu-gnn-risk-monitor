# Phase 6: Score Extrapolation Forecast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-compute per-region OLS linear trend forecasts (2024–2026) with 95% prediction intervals in Python, then surface them in the dashboard's `TimeSeriesPanel` as a "Show forecast" toggle with a dashed line and shaded confidence band.

**Architecture:** `export_dashboard_data.py` gains a `_forecast_region()` helper and an `export_forecast_scores()` function that writes `dashboard/public/data/forecast_scores.json`. The dashboard fetches this file independently (non-blocking, defaults to `{}` on 404), stores it in `DashboardData.forecast`, and `TimeSeriesPanel` renders it as a dashed `Line` + stacked `Area` confidence band inside a `ComposedChart`.

**Tech Stack:** Python: `sklearn.linear_model.LinearRegression`, `scipy.stats.t`, `numpy`, `pandas`. TypeScript/React: Recharts (`ComposedChart`, `Line`, `Area`, `ReferenceLine`), vitest 2 + `@testing-library/react`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `scripts/export_dashboard_data.py` | Modify | Add `_forecast_region()` helper + `export_forecast_scores()` |
| `tests/test_forecast.py` | Create | Unit test for `_forecast_region` with a known linear series |
| `dashboard/public/data/forecast_scores.json` | Generated | Written by `export_forecast_scores()` — 242 region forecasts |
| `dashboard/src/types.ts` | Modify | Add `ForecastPoint` interface; add `forecast` field to `DashboardData` |
| `dashboard/src/hooks/useDashboardData.ts` | Modify | Independent non-blocking fetch for `forecast_scores.json` |
| `dashboard/src/components/TimeSeriesPanel.tsx` | Modify | `showForecast` toggle, `ComposedChart` switch, dashed line + band |
| `dashboard/src/components/TimeSeriesPanel.test.tsx` | Create | New file — toggle presence and forecast attribution text |

All paths are relative to `EU-GNN-Risk-Monitor/`.

---

### Task 1: Python forecast computation

**Files:**
- Modify: `scripts/export_dashboard_data.py`
- Create: `tests/test_forecast.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_forecast.py`:

```python
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from export_dashboard_data import _forecast_region


def test_forecast_linear_series():
    """For a perfectly linear series the point estimates must match exact extrapolation."""
    years = list(range(2010, 2024))   # 14 years — matches real data length
    slope = 0.01
    intercept = 0.05
    scores = [slope * y + intercept for y in years]

    result = _forecast_region(years, scores)

    assert len(result) == 3
    for pt in result:
        expected = slope * pt['year'] + intercept
        assert abs(pt['point'] - expected) < 1e-3, (
            f"Year {pt['year']}: expected ≈{round(expected, 4)}, got {pt['point']}"
        )
        assert pt['lower'] <= pt['point'] <= pt['upper'], (
            f"Prediction interval ordering violated at year {pt['year']}"
        )
        assert pt['lower'] >= 0.0, f"Lower bound is negative at year {pt['year']}"
        assert pt['year'] in (2024, 2025, 2026)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd EU-GNN-Risk-Monitor
python -m pytest tests/test_forecast.py -v
```

Expected: `ImportError: cannot import name '_forecast_region'`

- [ ] **Step 3: Add imports and `_forecast_region` to `export_dashboard_data.py`**

Open `scripts/export_dashboard_data.py`. After the existing `import` block at the top of the file, add:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t as t_dist


def _forecast_region(
    years: list,
    scores: list,
    horizon: list | None = None,
) -> list:
    """Fit OLS linear trend on year→score and project forward with 95% prediction intervals.

    Returns list of {year, point, lower, upper} dicts, one per horizon year.
    Lower is clamped to 0.0 (anomaly scores cannot be negative).
    """
    if horizon is None:
        horizon = [2024, 2025, 2026]
    yr = np.array(years, dtype=float)
    y = np.array(scores, dtype=float)
    n = len(yr)
    model = LinearRegression().fit(yr.reshape(-1, 1), y)
    residuals = y - model.predict(yr.reshape(-1, 1))
    se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = float(yr.mean())
    Sxx = float(np.sum((yr - x_mean) ** 2))
    t_crit = float(t_dist.ppf(0.975, df=n - 2))
    pts = []
    for fx in horizon:
        point = float(model.predict([[float(fx)]])[0])
        margin = float(t_crit * se * np.sqrt(1.0 + 1.0 / n + (fx - x_mean) ** 2 / Sxx))
        pts.append({
            'year': fx,
            'point': round(point, 4),
            'lower': round(max(0.0, point - margin), 4),
            'upper': round(point + margin, 4),
        })
    return pts
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_forecast.py -v
```

Expected: `PASSED` — 1 test.

- [ ] **Step 5: Add `export_forecast_scores` function and call it from `__main__`**

In `scripts/export_dashboard_data.py`, add after the last existing export function (before the `if __name__ == '__main__':` block):

```python
def export_forecast_scores(root: Path, out: Path) -> None:
    """Compute per-region OLS forecasts (2024–2026) and write forecast_scores.json."""
    df = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_risk_scores.parquet')
    result: dict = {}
    for code, grp in df.groupby('nuts2_code'):
        grp = grp.sort_values('year')
        years = grp['year'].tolist()
        scores = grp['anomaly_score'].tolist()
        if len(years) < 3:
            continue
        result[str(code)] = _forecast_region(years, scores)
    out_path = out / 'forecast_scores.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))
    print(f'Wrote {len(result)} region forecasts → {out_path}')
```

In the `if __name__ == '__main__':` block, add after the last existing `export_*` call:

```python
export_forecast_scores(root, out)
```

- [ ] **Step 6: Run the export script**

```bash
python scripts/export_dashboard_data.py
```

Expected output includes: `Wrote 242 region forecasts → dashboard/public/data/forecast_scores.json`

Spot-check the file:

```bash
python -c "
import json
with open('dashboard/public/data/forecast_scores.json') as f:
    d = json.load(f)
# Should have ~242 keys
print(len(d), 'regions')
# Each entry should have 3 years
first_key = next(iter(d))
print(first_key, d[first_key])
"
```

Expected: `242 regions` and a list of 3 dicts with `year`, `point`, `lower`, `upper` keys.

- [ ] **Step 7: Commit**

```bash
git add scripts/export_dashboard_data.py tests/test_forecast.py dashboard/public/data/forecast_scores.json
git commit -m "feat(forecast): _forecast_region OLS helper + export_forecast_scores + test"
```

---

### Task 2: TypeScript types

**Files:**
- Modify: `dashboard/src/types.ts`

- [ ] **Step 1: Add `ForecastPoint` interface**

Open `dashboard/src/types.ts`. After the `RegionTimeSeries` interface (before the closing of the file), add:

```typescript
export interface ForecastPoint {
  year: number;
  point: number;
  lower: number;
  upper: number;
}
```

- [ ] **Step 2: Add `forecast` field to `DashboardData`**

In the `DashboardData` interface, add as the last field:

```typescript
forecast: Record<string, ForecastPoint[]>;
```

The field is required (not optional) so TypeScript will catch every place that constructs a `DashboardData` object and is missing it — this ensures the hook and empty-state return are both updated in Task 3.

- [ ] **Step 3: Commit (types only — TS errors are expected until Task 3)**

```bash
git add dashboard/src/types.ts
git commit -m "feat(forecast): add ForecastPoint type + forecast field to DashboardData"
```

---

### Task 3: Data hook — load `forecast_scores.json`

**Files:**
- Modify: `dashboard/src/hooks/useDashboardData.ts`

Background: the hook currently fetches three files in a `Promise.all`. Adding `forecast_scores.json` to that `Promise.all` would break the entire dashboard if the file is absent during development. Instead, fetch it independently after the main `Promise.all` resolves, with an explicit `.catch(() => ({}))` fallback.

- [ ] **Step 1: Add `ForecastPoint` to the import**

Open `dashboard/src/hooks/useDashboardData.ts`. Find the import from `'../types'` (read the file to see the exact current import list) and add `ForecastPoint` to it. Example result if the file currently imports `DashboardData` and `RiskScore`:

```typescript
import type { DashboardData, RiskScore, ForecastPoint } from '../types';
```

Preserve all existing imported names — only add `ForecastPoint`.

- [ ] **Step 2: Add independent forecast fetch**

Find the async loading function inside the hook (the function that contains the `Promise.all`). After `deriveData(...)` is called and the dashboard data object is assembled, add:

```typescript
const forecast: Record<string, ForecastPoint[]> = await fetch('data/forecast_scores.json')
  .then(r => r.ok ? (r.json() as Promise<Record<string, ForecastPoint[]>>) : Promise.resolve({}))
  .catch(() => ({} as Record<string, ForecastPoint[]>));
```

Then, wherever `setData(...)` or the equivalent state setter is called with the derived data, spread `forecast` into the object:

```typescript
setData({ ...dashboardData, forecast });
```

(Replace `dashboardData` with whatever variable name holds the result of `deriveData`.)

- [ ] **Step 3: Update the empty-state return**

Find the early-return / initial-state object that is returned before data loads (look for the object literal containing `scores: []`, `byCode: {}`, etc.). Add `forecast: {}` to that object:

```typescript
forecast: {},
```

- [ ] **Step 4: TypeScript check — expect 0 errors**

```bash
cd dashboard
npx tsc --noEmit
```

Expected: 0 errors. (All `DashboardData` construction sites now include `forecast`.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/hooks/useDashboardData.ts
git commit -m "feat(forecast): non-blocking forecast_scores.json fetch in useDashboardData"
```

---

### Task 4: TimeSeriesPanel — toggle, ComposedChart, forecast layers

**Files:**
- Create: `dashboard/src/components/TimeSeriesPanel.test.tsx`
- Modify: `dashboard/src/components/TimeSeriesPanel.tsx`

Background (from scout):
- The panel has two render branches. **Branch A** (`selectedRegion === null`) renders a top-3 `LineChart` — leave it entirely untouched.
- **Branch B** (`selectedRegion` is set) renders a single-region `LineChart` with `chartData` rows keyed `{ year, [selectedRegion]: score, eu_mean: ..., [neighbour]: ... }`.
- The Branch B header currently has only text (region name + metadata). The "Show forecast" checkbox will be the first interactive control.
- Current Recharts imports: `ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, Legend`. Need to add `ComposedChart, Area` and keep `LineChart` for Branch A.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/src/components/TimeSeriesPanel.test.tsx`:

```typescript
// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import TimeSeriesPanel from './TimeSeriesPanel';
import type { DashboardData } from '../types';

afterEach(cleanup);

const baseData: DashboardData = {
  scores: [],
  neighbours: { PT15: ['PT11', 'PT16'] },
  geoJson: { type: 'FeatureCollection', features: [] } as DashboardData['geoJson'],
  regionNames: { PT15: 'Algarve' },
  byCode: {
    PT15: {
      nuts2_code: 'PT15',
      country_code: 'PT',
      peak_score: 0.28,
      peak_rank: 1,
      peak_year: 2015,
      scores: [
        { year: 2010, anomaly_score: 0.10 },
        { year: 2011, anomaly_score: 0.15 },
        { year: 2012, anomaly_score: 0.20 },
        { year: 2023, anomaly_score: 0.22 },
      ],
    },
  },
  byYear: {},
  maxScore: 0.28,
  systemPeakYear: 2015,
  topRegions: [],
  countries: ['PT'],
  forecast: {},            // no forecast data
};

const dataWithForecast: DashboardData = {
  ...baseData,
  forecast: {
    PT15: [
      { year: 2024, point: 0.21, lower: 0.11, upper: 0.31 },
      { year: 2025, point: 0.20, lower: 0.09, upper: 0.31 },
      { year: 2026, point: 0.19, lower: 0.07, upper: 0.31 },
    ],
  },
};

describe('TimeSeriesPanel', () => {
  it('hides forecast toggle when forecast data is absent for selected region', () => {
    render(
      <TimeSeriesPanel data={baseData} selectedRegion="PT15" activeYear={2015} />,
    );
    expect(screen.queryByRole('checkbox')).toBeNull();
    expect(screen.queryByText(/forecast/i)).toBeNull();
  });

  it('shows forecast toggle when forecast data exists for selected region', () => {
    render(
      <TimeSeriesPanel data={dataWithForecast} selectedRegion="PT15" activeYear={2015} />,
    );
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeInTheDocument();
    expect(screen.getByText(/forecast/i)).toBeInTheDocument();
  });

  it('shows attribution note when forecast is toggled on', () => {
    render(
      <TimeSeriesPanel data={dataWithForecast} selectedRegion="PT15" activeYear={2015} />,
    );
    const checkbox = screen.getByRole('checkbox');
    expect(screen.queryByText(/OLS linear trend/i)).toBeNull();
    fireEvent.click(checkbox);
    expect(screen.getByText(/OLS linear trend/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd dashboard
npx vitest run src/components/TimeSeriesPanel.test.tsx
```

Expected: Either `Cannot find module './TimeSeriesPanel'` (if component has a different import path) or the tests fail because the toggle doesn't exist yet. Confirm failure before proceeding.

- [ ] **Step 3: Add `ComposedChart` and `Area` to Recharts imports**

Open `dashboard/src/components/TimeSeriesPanel.tsx`. Find the recharts import line (currently: `import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, Legend } from 'recharts';`) and replace it with:

```typescript
import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
} from 'recharts';
```

- [ ] **Step 4: Add `ForecastPoint` to the types import**

Find the import from `'../types'` and add `ForecastPoint` if it isn't already there:

```typescript
import type { DashboardData, ForecastPoint } from '../types';
```

- [ ] **Step 5: Add `showForecast` state + reset effect + `hasForecast` derived value**

Inside the `TimeSeriesPanel` component function, after the existing state/variable declarations, add:

```typescript
const [showForecast, setShowForecast] = useState(false);

// Reset toggle when region changes so we don't show a stale forecast state
useEffect(() => {
  setShowForecast(false);
}, [selectedRegion]);

const hasForecast =
  !!selectedRegion && (data.forecast[selectedRegion]?.length ?? 0) > 0;
```

Make sure `useState` and `useEffect` are in the React import at the top of the file. If the file only imports `React` as a default import, add named imports:

```typescript
import { useState, useEffect } from 'react';
```

- [ ] **Step 6: Add toggle checkbox to the Branch B header**

In Branch B (the section that renders when `selectedRegion` is set), find the header area. The scout confirmed it contains only text — region name + metadata. Add the toggle as the last element in the header row (or a new row immediately below the metadata):

```tsx
{hasForecast && (
  <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer select-none">
    <input
      type="checkbox"
      checked={showForecast}
      onChange={e => setShowForecast(e.target.checked)}
      className="accent-blue-500 cursor-pointer"
    />
    Forecast
  </label>
)}
```

- [ ] **Step 7: Build the forecast chart data extension**

In Branch B, after the existing `chartData` variable is built from `ts.scores.map(...)`, add:

```typescript
// Chart row type — allows arbitrary keys for Recharts dynamic dataKey usage
type ChartRow = { year: number; fPoint?: number; fLower?: number; fBand?: number; [key: string]: unknown };

let displayData: ChartRow[] = chartData as ChartRow[];

if (showForecast && hasForecast) {
  const fPts: ForecastPoint[] = data.forecast[selectedRegion!];

  // Bridge the last actual data point into the forecast series for a seamless line join
  const lastActual = chartData[chartData.length - 1];
  const bridgeScore = lastActual[selectedRegion] as number;

  const bridgeRow: ChartRow = {
    ...(lastActual as ChartRow),
    fPoint: bridgeScore,
    fLower: bridgeScore,
    fBand: 0,
  };

  const forecastRows: ChartRow[] = fPts.map(fp => ({
    year: fp.year,
    [selectedRegion!]: undefined,
    eu_mean: null,
    fPoint: fp.point,
    fLower: fp.lower,
    fBand: fp.upper - fp.lower,
  }));

  displayData = [
    ...(chartData.slice(0, -1) as ChartRow[]),
    bridgeRow,
    ...forecastRows,
  ];
}
```

- [ ] **Step 8: Switch Branch B chart from `<LineChart>` to `<ComposedChart>` and add forecast layers**

In Branch B, replace the outer `<LineChart data={chartData} ...>` with `<ComposedChart data={displayData} ...>` (same props). Replace the closing `</LineChart>` with `</ComposedChart>`.

Inside the `<ComposedChart>`, after all existing `<Line>` elements, add the forecast layers (rendered only when `showForecast && hasForecast`):

```tsx
{showForecast && hasForecast && (
  <>
    {/* Separator between actual and projected */}
    <ReferenceLine
      x={data.byCode[selectedRegion!].scores[data.byCode[selectedRegion!].scores.length - 1].year}
      stroke="#374151"
      strokeDasharray="2 2"
      label={{ value: '2023', position: 'top', fill: '#6b7280', fontSize: 10 }}
    />
    {/* Confidence band — transparent baseline + colored band stacked */}
    <Area
      dataKey="fLower"
      fill="transparent"
      stroke="none"
      stackId="ci"
      isAnimationActive={false}
      legendType="none"
    />
    <Area
      dataKey="fBand"
      fill="#3b82f6"
      fillOpacity={0.15}
      stroke="none"
      stackId="ci"
      isAnimationActive={false}
      legendType="none"
    />
    {/* Dashed forecast line */}
    <Line
      dataKey="fPoint"
      stroke="#3b82f6"
      strokeWidth={1.5}
      strokeDasharray="5 3"
      dot={false}
      isAnimationActive={false}
      legendType="none"
    />
  </>
)}
```

- [ ] **Step 9: Add attribution note below the chart**

After the closing `</ResponsiveContainer>` tag (still inside Branch B), add:

```tsx
{showForecast && hasForecast && (
  <p className="text-xs text-gray-500 mt-1 text-center">
    Projected 2024–2026 · OLS linear trend · 95% prediction interval
  </p>
)}
```

- [ ] **Step 10: Run the tests**

```bash
cd dashboard
npx vitest run src/components/TimeSeriesPanel.test.tsx
```

Expected: 3 tests PASS.

- [ ] **Step 11: TypeScript check**

```bash
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 12: Commit**

```bash
git add dashboard/src/components/TimeSeriesPanel.tsx dashboard/src/components/TimeSeriesPanel.test.tsx
git commit -m "feat(forecast): TimeSeriesPanel show-forecast toggle + dashed line + CI band"
```

---

### Task 5: Full test suite pass

**Files:** none (verification only)

- [ ] **Step 1: Run all dashboard tests**

```bash
cd dashboard
npx vitest run
```

Expected: all tests pass (including existing `ComparePanel.test.tsx` and the new `TimeSeriesPanel.test.tsx`).

- [ ] **Step 2: TypeScript clean build**

```bash
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 3: Run Python tests**

```bash
cd ..   # back to EU-GNN-Risk-Monitor root
python -m pytest tests/ -v
```

Expected: `test_forecast.py::test_forecast_linear_series` PASS.

- [ ] **Step 4: Commit (if any fixes were needed)**

If steps 1–3 required any fixes, commit them now. If all passed on first run, nothing to commit here.

```bash
git add -p
git commit -m "fix(forecast): address full-suite issues"
```
