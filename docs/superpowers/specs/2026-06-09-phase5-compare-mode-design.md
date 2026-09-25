# Phase 5: Region Comparison Mode — Design Spec

**Date:** 2026-06-09  
**Project:** EU-GNN-Risk-Monitor  
**Phase:** 5 of n  

---

## Goal

Add a Region Comparison Mode to the existing 3-panel dashboard that lets the user compare two NUTS2 regions head-to-head: peak metrics, score trajectory gap, and shared GNN graph neighbours.

---

## Context

The Phase 4 dashboard is a 3-panel layout:

```
[ EU Choropleth Map (~50%) ] [ Time Series Panel (~25%) ] [ Ranking Panel (~25%) ]
```

- `selectedRegion` (string | null) — set by map click or ranking row click
- `activeYear` — driven by year slider / play animation
- All data is static JSON loaded once in `useDashboardData`

The right panel (Ranking) will swap to a Compare Panel when compare mode is active. No new routes, no new data files, no new dependencies.

---

## User Flow

```
1. User clicks a NUTS2 region on the map → selectedRegion = "PT15"
2. User clicks "Compare" button in RankingPanel header
   → compareMode = true
   → Region A = selectedRegion ("PT15"), compareRegion = null
   → Right panel swaps: RankingPanel → ComparePanel
   → Map shows a blue banner overlay: "Click a region to set Region B"
3. User clicks "SK01" on the map
   → compareRegion = "SK01"
   → ComparePanel populates with full stats
4. User clicks "← Back to ranking"
   → compareMode = false, compareRegion = null
   → Right panel swaps back to RankingPanel
```

**Entry guard:** The "Compare" button is disabled (dimmed, non-clickable) when `selectedRegion` is null.

---

## State Changes (App.tsx)

Two new fields added to App state:

```typescript
const [compareMode, setCompareMode] = useState(false);
const [compareRegion, setCompareRegion] = useState<string | null>(null);
```

New click handler that replaces the direct `setSelectedRegion` passed to EuropeMap:

```typescript
function handleRegionClick(code: string) {
  if (compareMode) {
    setCompareRegion(code);
  } else {
    setSelectedRegion(code);
  }
}
```

Activation handler (passed to RankingPanel):

```typescript
function handleCompareActivate() {
  setCompareMode(true);
  setCompareRegion(null);
}
```

Exit handler (passed to ComparePanel):

```typescript
function handleCompareExit() {
  setCompareMode(false);
  setCompareRegion(null);
}
```

Right panel conditional render:

```tsx
{compareMode ? (
  <ComparePanel
    data={data}
    regionA={selectedRegion}
    regionB={compareRegion}
    onExit={handleCompareExit}
  />
) : (
  <RankingPanel
    data={data}
    selectedRegion={selectedRegion}
    filterCountry={filterCountry}
    onRegionClick={setSelectedRegion}
    onRegionHover={setHoveredRegion}
    onCountryFilter={setFilterCountry}
    onCompareActivate={handleCompareActivate}
    compareDisabled={selectedRegion === null}
  />
)}
```

---

## EuropeMap Changes

### New props

```typescript
interface Props {
  data: DashboardData;
  activeYear: number;
  selectedRegion: string | null;
  compareMode: boolean;
  compareRegion: string | null;
  onRegionClick: (code: string) => void;   // routes to handleRegionClick in App
  onRegionHover: (code: string | null) => void;
}
```

### getStyle — third case

```typescript
function getStyle(feature: Feature | undefined): L.PathOptions {
  const code = (feature as NutsFeature)?.properties?.NUTS_ID ?? '';
  const score = yearScores[code] ?? 0;
  const isSelected = code === selectedRegion;
  const isCompare = code === compareRegion;
  return {
    fillColor: anomalyColor(score, data.maxScore),
    fillOpacity: 0.82,
    color: isSelected ? '#ffffff' : isCompare ? '#3b82f6' : '#1f2937',
    weight: isSelected || isCompare ? 2.5 : 0.4,
  };
}
```

### GeoJSON key — includes compareMode and compareRegion

```tsx
key={`${activeYear}-${selectedRegion ?? 'none'}-${compareRegion ?? 'none'}-${compareMode}`}
```

Must include `compareMode` as well as `compareRegion`. When compare mode activates, `compareRegion` is still `null` so the key would otherwise be unchanged — but the `onEachFeature` click handler needs to be re-bound with the updated `onRegionClick` closure (which now routes to `setCompareRegion`). Including `compareMode` in the key forces a full GeoJSON remount whenever compare mode toggles, ensuring handlers are always fresh.

### Compare mode banner overlay

EuropeMap currently returns `<MapContainer>` directly. Wrap it in a `div` with `relative h-full w-full` so the overlay can be absolutely positioned inside it:

```tsx
return (
  <div className="relative h-full w-full">
    {compareMode && !compareRegion && (
      <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[1000]
                      bg-blue-900/90 text-blue-200 text-xs px-3 py-1 rounded-full
                      pointer-events-none">
        Click a region to set Region B
      </div>
    )}
    <MapContainer
      bounds={EU_BOUNDS}
      minZoom={3}
      maxZoom={8}
      zoomControl
      style={{ height: '100%', width: '100%', background: '#030712' }}
    >
      ...
    </MapContainer>
  </div>
);
```

The overlay uses `z-[1000]` to sit above the Leaflet tile layer (which uses z-index 200–500 internally).

---

## RankingPanel Changes

Add two new props:

```typescript
onCompareActivate: () => void;
compareDisabled: boolean;
```

Add a "Compare" button to the panel header (next to the existing country filter or title):

```tsx
<button
  onClick={onCompareActivate}
  disabled={compareDisabled}
  className="text-xs px-2 py-0.5 rounded border border-blue-600 text-blue-400
             hover:bg-blue-900 disabled:opacity-30 disabled:cursor-not-allowed"
>
  Compare
</button>
```

No other changes to RankingPanel.

---

## ComparePanel (new component)

**File:** `dashboard/src/components/ComparePanel.tsx`

### Props

```typescript
interface Props {
  data: DashboardData;
  regionA: string | null;
  regionB: string | null;
  onExit: () => void;
}
```

### Derived data (computed inside component, no hook needed)

```typescript
const tsA = regionA ? data.byCode[regionA] : null;
const tsB = regionB ? data.byCode[regionB] : null;
const nameA = regionA ? (data.regionNames[regionA] ?? regionA) : '—';
const nameB = regionB ? (data.regionNames[regionB] ?? regionB) : '—';

// Avg score over all years
const avgScore = (ts: RegionTimeSeries) =>
  ts.scores.reduce((s, r) => s + r.anomaly_score, 0) / ts.scores.length;

// Max gap: year where |scoreA - scoreB| is maximised
interface GapResult { year: number; delta: number }
function maxGap(a: RegionTimeSeries, b: RegionTimeSeries): GapResult {
  let best = { year: 0, delta: 0 };
  for (const ra of a.scores) {
    const rb = b.scores.find((r) => r.year === ra.year);
    if (!rb) continue;
    const delta = Math.abs(ra.anomaly_score - rb.anomaly_score);
    if (delta > best.delta) best = { year: ra.year, delta };
  }
  return best;
}

// Neighbours
const neighboursA = new Set(regionA ? (data.neighbours[regionA] ?? []) : []);
const neighboursB = new Set(regionB ? (data.neighbours[regionB] ?? []) : []);
const shared    = [...neighboursA].filter((n) => neighboursB.has(n));
const onlyA     = [...neighboursA].filter((n) => !neighboursB.has(n));
const onlyB     = [...neighboursB].filter((n) => !neighboursA.has(n));
```

### Layout

```
┌─────────────────────────────────────────┐
│ ⚖ Compare                    [← Back]  │  header row
├──────────────┬──────────────────────────┤
│ [PT15]       │  [SK01]                  │  region chips (red / blue border)
│  Algarve     │   Bratislava             │
├──────────────┴──────────────────────────┤
│  Metric         PT15       SK01         │  stats table
│  Peak rank       #1         #2          │
│  Peak year      2015       2013         │
│  Peak score    0.2847     0.2701        │
│  Avg score     0.1923     0.1574        │
│  Max gap year   2015 (Δ 0.0428)         │  (full-width merged cell)
├─────────────────────────────────────────┤
│  GNN Neighbours                         │
│  Shared: 0 · PT15 only: 4 · SK01 only:3│
│  [PT17] [PT18] ← red                   │
│  [SK02] [AT13] ← blue                  │
└─────────────────────────────────────────┘
```

**When `regionB` is null:** Replace the Region B chip with a dimmed placeholder ("click map →") and render the stats table with "—" in the SK01 column. The neighbours section is hidden.

**When `regionA` is null:** Render an empty state message ("Select a region first"). This state should not normally occur due to the entry guard, but is handled defensively.

---

## Tests (ComparePanel.test.tsx)

Three vitest unit tests:

1. **Renders waiting state when regionB is null** — `regionA="PT15"`, `regionB={null}`, assert Region B placeholder text is present, stats column B shows "—".

2. **Renders full stats for two regions** — `regionA="PT15"`, `regionB="SK01"` with mock `data`, assert peak ranks, peak years, avg scores, and max gap row are correct.

3. **Calls onExit when back button is clicked** — fire click on "← Back to ranking", assert `onExit` mock was called once.

Mock `data` object needs: `byCode`, `neighbours`, `regionNames` for PT15 and SK01 only — 4–6 score records each.

---

## Out of Scope (Phase 5)

- Exporting the comparison as an image or CSV
- Comparing more than 2 regions simultaneously
- A gap area chart (deferred — stats table chosen over chart)
- URL-based compare state (no routing added)
- Forecasting / temporal projection (Phase 6)

---

## Files Changed

| File | Type | Change |
|---|---|---|
| `src/App.tsx` | modify | +2 state fields, conditional render, handleRegionClick, handleCompareActivate/Exit |
| `src/components/EuropeMap.tsx` | modify | compareRegion blue highlight, compare banner overlay, updated GeoJSON key |
| `src/components/RankingPanel.tsx` | modify | Compare button in header, 2 new props |
| `src/components/ComparePanel.tsx` | **new** | Full compare panel component |
| `src/components/ComparePanel.test.tsx` | **new** | 3 vitest unit tests |
