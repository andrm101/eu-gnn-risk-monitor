# Phase 5: Region Comparison Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Region Comparison Mode to the EU NUTS2 dashboard — clicking "Compare" in the ranking panel swaps the right sidebar to a head-to-head stats panel, with Region B set by clicking the map.

**Architecture:** Two new state fields (`compareMode: boolean`, `compareRegion: string | null`) in App.tsx drive everything. A new `ComparePanel` component conditionally replaces `RankingPanel` in the right slot. `EuropeMap` gains a blue border highlight for Region B and a banner overlay when waiting for a Region B click. The GeoJSON key includes `compareMode` to force click-handler rebinding when compare mode activates.

**Tech Stack:** React 18 + TypeScript + Tailwind 3, Vite 5 + vitest 2, @testing-library/react + jsdom (new dev deps for component tests)

---

## File Map

| File | Status | Purpose |
|---|---|---|
| `dashboard/src/components/ComparePanel.tsx` | **new** | Head-to-head stats panel: region chips, stats table, GNN neighbours, back button |
| `dashboard/src/components/ComparePanel.test.tsx` | **new** | 3 vitest component tests |
| `dashboard/src/components/RankingPanel.tsx` | modify | Add Compare button + 2 new props |
| `dashboard/src/components/EuropeMap.tsx` | modify | compareRegion blue highlight, banner overlay, updated GeoJSON key |
| `dashboard/src/App.tsx` | modify | compareMode/compareRegion state, routing click handler, conditional render |

---

### Task 1: Install component testing dependencies

**Files:**
- Modify: `dashboard/package.json` (via npm install)

All `npm` commands run from `EU-GNN-Risk-Monitor/dashboard/`.

- [ ] **Step 1: Install @testing-library/react and jsdom**

```bash
cd EU-GNN-Risk-Monitor/dashboard
npm install -D @testing-library/react jsdom
```

Expected: 2 packages added to devDependencies in package.json.

- [ ] **Step 2: Verify existing tests still pass**

```bash
npm test
```

Expected: `8 tests passed` (5 color + 3 format utility tests). Zero failures.

- [ ] **Step 3: Commit**

```bash
git add EU-GNN-Risk-Monitor/dashboard/package.json EU-GNN-Risk-Monitor/dashboard/package-lock.json
git commit -m "chore(dashboard): add @testing-library/react + jsdom for component tests"
```

---

### Task 2: ComparePanel — TDD (failing tests then implementation)

**Files:**
- Create: `EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.test.tsx`
- Create: `EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.tsx`

**Type context:** `RegionTimeSeries` (from `src/types.ts`) has: `nuts2_code: string`, `country_code: string`, `peak_score: number`, `peak_rank: number`, `peak_year: number`, `scores: { year: number; anomaly_score: number }[]`. `DashboardData.byCode` is `Record<string, RegionTimeSeries>`. `DashboardData.neighbours` is `Record<string, string[]>`. `DashboardData.regionNames` is `Record<string, string>`.

**Derived values for mock data:**
- PT15 scores: 2010=0.10, 2011=0.20, 2012=0.15 → avg = 0.1500, peak = 2011
- SK01 scores: 2010=0.05, 2011=0.08, 2012=0.12 → avg = 0.0833, peak = 2012
- Max gap per year: 2010=|0.10−0.05|=0.05, 2011=|0.20−0.08|=**0.12** ← max, 2012=|0.15−0.12|=0.03
- So max gap year = 2011, Δ = 0.1200 (formatScore gives "0.1200")

- [ ] **Step 1: Write the three failing tests**

Create `EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.test.tsx`:

```typescript
// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ComparePanel from './ComparePanel';
import type { DashboardData } from '../types';

const mockData: DashboardData = {
  scores: [],
  neighbours: {
    PT15: ['PT17', 'PT18'],
    SK01: ['SK02', 'AT13'],
  },
  geoJson: { type: 'FeatureCollection', features: [] },
  regionNames: { PT15: 'Algarve', SK01: 'Bratislava' },
  byCode: {
    PT15: {
      nuts2_code: 'PT15',
      country_code: 'PT',
      peak_score: 0.20,
      peak_rank: 1,
      peak_year: 2011,
      scores: [
        { year: 2010, anomaly_score: 0.10 },
        { year: 2011, anomaly_score: 0.20 },
        { year: 2012, anomaly_score: 0.15 },
      ],
    },
    SK01: {
      nuts2_code: 'SK01',
      country_code: 'SK',
      peak_score: 0.12,
      peak_rank: 2,
      peak_year: 2012,
      scores: [
        { year: 2010, anomaly_score: 0.05 },
        { year: 2011, anomaly_score: 0.08 },
        { year: 2012, anomaly_score: 0.12 },
      ],
    },
  },
  byYear: {},
  maxScore: 0.20,
  systemPeakYear: 2011,
  topRegions: [],
  countries: ['PT', 'SK'],
};

describe('ComparePanel', () => {
  it('shows placeholder when regionB is null', () => {
    render(<ComparePanel data={mockData} regionA="PT15" regionB={null} onExit={vi.fn()} />);
    expect(screen.getByText('click map →')).not.toBeNull();
  });

  it('renders head-to-head stats for two regions', () => {
    render(<ComparePanel data={mockData} regionA="PT15" regionB="SK01" onExit={vi.fn()} />);
    expect(screen.getByText('#1')).not.toBeNull();
    expect(screen.getByText('#2')).not.toBeNull();
    expect(screen.getByText('Algarve')).not.toBeNull();
    expect(screen.getByText('Bratislava')).not.toBeNull();
    // Max gap: year 2011, delta = |0.20 - 0.08| = 0.1200
    expect(screen.getByText(/Δ 0\.1200/)).not.toBeNull();
  });

  it('calls onExit when back button is clicked', () => {
    const onExit = vi.fn();
    render(<ComparePanel data={mockData} regionA="PT15" regionB="SK01" onExit={onExit} />);
    fireEvent.click(screen.getByText(/Back to ranking/));
    expect(onExit).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd EU-GNN-Risk-Monitor/dashboard && npm test
```

Expected: 3 new tests FAIL with "Cannot find module './ComparePanel'". Existing 8 tests still pass.

- [ ] **Step 3: Implement ComparePanel.tsx**

Create `EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.tsx`:

```typescript
import type { DashboardData, RegionTimeSeries } from '../types';
import { formatScore, formatRank } from '../utils/format';

interface Props {
  data: DashboardData;
  regionA: string | null;
  regionB: string | null;
  onExit: () => void;
}

function avgScore(ts: RegionTimeSeries): number {
  if (ts.scores.length === 0) return 0;
  return ts.scores.reduce((s, r) => s + r.anomaly_score, 0) / ts.scores.length;
}

interface GapResult {
  year: number;
  delta: number;
}

function maxGap(a: RegionTimeSeries, b: RegionTimeSeries): GapResult {
  let best: GapResult = { year: 0, delta: 0 };
  for (const ra of a.scores) {
    const rb = b.scores.find((r) => r.year === ra.year);
    if (!rb) continue;
    const delta = Math.abs(ra.anomaly_score - rb.anomaly_score);
    if (delta > best.delta) best = { year: ra.year, delta };
  }
  return best;
}

export default function ComparePanel({ data, regionA, regionB, onExit }: Props) {
  const tsA = regionA ? data.byCode[regionA] : null;
  const tsB = regionB ? data.byCode[regionB] : null;
  const nameA = regionA ? (data.regionNames[regionA] ?? regionA) : '—';
  const nameB = regionB ? (data.regionNames[regionB] ?? regionB) : '—';
  const gap = tsA && tsB ? maxGap(tsA, tsB) : null;

  const neighboursA = new Set(regionA ? (data.neighbours[regionA] ?? []) : []);
  const neighboursB = new Set(regionB ? (data.neighbours[regionB] ?? []) : []);
  const shared = [...neighboursA].filter((n) => neighboursB.has(n));
  const onlyA  = [...neighboursA].filter((n) => !neighboursB.has(n));
  const onlyB  = [...neighboursB].filter((n) => !neighboursA.has(n));

  const statsRows: [string, string, string][] =
    tsA && tsB
      ? [
          ['Peak rank',  formatRank(tsA.peak_rank),       formatRank(tsB.peak_rank)],
          ['Peak year',  String(tsA.peak_year),            String(tsB.peak_year)],
          ['Peak score', formatScore(tsA.peak_score),      formatScore(tsB.peak_score)],
          ['Avg score',  formatScore(avgScore(tsA)),       formatScore(avgScore(tsB))],
        ]
      : [];

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-800 shrink-0 flex items-center gap-2">
        <span className="text-xs text-gray-400 font-medium">⚖ Compare</span>
        <button
          onClick={onExit}
          className="ml-auto text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          ← Back to ranking
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">

        {/* Region chips */}
        <div className="flex gap-2">
          <div className="flex-1 rounded border border-red-600 bg-gray-900 px-2 py-1.5">
            <div className="text-red-400 text-xs font-bold">{regionA ?? '—'}</div>
            <div className="text-gray-400 text-xs truncate">{nameA}</div>
          </div>
          <div className="flex-1 rounded border border-blue-600 bg-gray-900 px-2 py-1.5">
            {regionB ? (
              <>
                <div className="text-blue-400 text-xs font-bold">{regionB}</div>
                <div className="text-gray-400 text-xs truncate">{nameB}</div>
              </>
            ) : (
              <div className="text-gray-600 text-xs italic flex items-center h-full">
                click map →
              </div>
            )}
          </div>
        </div>

        {/* Stats table — only when both regions are set */}
        {tsA && tsB && gap && (
          <div className="text-xs">
            <div className="grid grid-cols-[1fr_56px_56px] bg-gray-800 rounded-t px-2 py-1 text-gray-500 font-medium">
              <span>Metric</span>
              <span className="text-center text-red-400">{regionA}</span>
              <span className="text-center text-blue-400">{regionB}</span>
            </div>
            {statsRows.map(([label, a, b], i) => (
              <div
                key={label}
                className={`grid grid-cols-[1fr_56px_56px] px-2 py-1 ${
                  i % 2 === 1 ? 'bg-gray-900' : 'bg-gray-950'
                }`}
              >
                <span className="text-gray-500">{label}</span>
                <span className="text-center text-gray-200">{a}</span>
                <span className="text-center text-gray-200">{b}</span>
              </div>
            ))}
            <div className="bg-gray-900 rounded-b px-2 py-1">
              <span className="text-gray-500">Max gap </span>
              <span className="text-amber-400">
                {gap.year} (Δ {formatScore(gap.delta)})
              </span>
            </div>
          </div>
        )}

        {/* GNN Neighbours — only when both regions are set */}
        {regionA && regionB && (
          <div>
            <div className="text-xs text-blue-400 font-medium mb-1">GNN Neighbours</div>
            <div className="text-xs text-gray-500 mb-2">
              Shared: <span className="text-green-400">{shared.length}</span>
              {' · '}A only: {onlyA.length}
              {' · '}B only: {onlyB.length}
            </div>
            <div className="flex flex-wrap gap-1">
              {onlyA.map((n) => (
                <span key={n} className="bg-gray-800 text-red-400 text-xs px-1.5 py-0.5 rounded">
                  {n}
                </span>
              ))}
              {shared.map((n) => (
                <span key={n} className="bg-gray-800 text-green-400 text-xs px-1.5 py-0.5 rounded">
                  {n}
                </span>
              ))}
              {onlyB.map((n) => (
                <span key={n} className="bg-gray-800 text-blue-400 text-xs px-1.5 py-0.5 rounded">
                  {n}
                </span>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
npm test
```

Expected: `11 tests passed` (8 existing + 3 new). Zero failures.

- [ ] **Step 5: Commit**

```bash
git add EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.tsx \
        EU-GNN-Risk-Monitor/dashboard/src/components/ComparePanel.test.tsx
git commit -m "feat(dashboard): ComparePanel — stats table, GNN neighbours, back button"
```

---

### Task 3: RankingPanel — add Compare button

**Files:**
- Modify: `EU-GNN-Risk-Monitor/dashboard/src/components/RankingPanel.tsx`

**Context:** Current Props interface is lines 4–11. Function signature is lines 39–41. Panel header div is lines 48–61. Two new props are added: `onCompareActivate: () => void` and `compareDisabled: boolean`. The Compare button is disabled (dimmed) when no region is selected.

Note: after this task, `npm run build` will show a TypeScript error on App.tsx because the new props are not yet passed. This is expected and resolved in Task 5. Run `npm test` (not `npm run build`) to verify no regression.

- [ ] **Step 1: Replace the Props interface (lines 4–11)**

```typescript
interface Props {
  data: DashboardData;
  selectedRegion: string | null;
  filterCountry: string | null;
  onRegionClick: (code: string) => void;
  onRegionHover: (code: string | null) => void;
  onCountryFilter: (code: string | null) => void;
  onCompareActivate: () => void;
  compareDisabled: boolean;
}
```

- [ ] **Step 2: Replace the function signature (lines 39–41)**

```typescript
export default function RankingPanel({
  data, selectedRegion, filterCountry, onRegionClick, onRegionHover, onCountryFilter,
  onCompareActivate, compareDisabled,
}: Props) {
```

- [ ] **Step 3: Replace the panel header div (lines 48–61)**

```tsx
{/* Panel header */}
<div className="px-3 py-2 border-b border-gray-800 shrink-0 flex items-center gap-2">
  <span className="text-xs text-gray-400 font-medium">Peak Risk Ranking</span>
  <button
    onClick={onCompareActivate}
    disabled={compareDisabled}
    className="text-xs px-2 py-0.5 rounded border border-blue-600 text-blue-400
               hover:bg-blue-900/40 disabled:opacity-30 disabled:cursor-not-allowed
               transition-colors"
  >
    Compare
  </button>
  <select
    value={filterCountry ?? ''}
    onChange={(e) => onCountryFilter(e.target.value || null)}
    className="ml-auto text-xs bg-gray-800 border border-gray-700 text-gray-300 rounded px-1 py-0.5 cursor-pointer"
  >
    <option value="">All countries</option>
    {data.countries.map((c) => (
      <option key={c} value={c}>{c}</option>
    ))}
  </select>
</div>
```

- [ ] **Step 4: Run tests (not build) to verify no regression**

```bash
cd EU-GNN-Risk-Monitor/dashboard && npm test
```

Expected: `11 tests passed`. (A TypeScript error on App.tsx is expected at this stage — it will be fixed in Task 5.)

- [ ] **Step 5: Commit**

```bash
git add EU-GNN-Risk-Monitor/dashboard/src/components/RankingPanel.tsx
git commit -m "feat(dashboard): RankingPanel — Compare button with disabled guard"
```

---

### Task 4: EuropeMap — compareRegion highlight, banner, key update

**Files:**
- Modify: `EU-GNN-Risk-Monitor/dashboard/src/components/EuropeMap.tsx`

**Context:** Current Props interface is lines 18–24. `getStyle` is lines 32–41. The GeoJSON `key` prop is on line 87. The component currently returns `<MapContainer>` directly starting at line 77.

- [ ] **Step 1: Replace the Props interface (lines 18–24)**

```typescript
interface Props {
  data: DashboardData;
  activeYear: number;
  selectedRegion: string | null;
  compareMode: boolean;
  compareRegion: string | null;
  onRegionClick: (code: string) => void;
  onRegionHover: (code: string | null) => void;
}
```

- [ ] **Step 2: Replace the function signature (lines 26–28)**

```typescript
export default function EuropeMap({
  data, activeYear, selectedRegion, compareMode, compareRegion, onRegionClick, onRegionHover,
}: Props) {
```

- [ ] **Step 3: Replace getStyle to add compareRegion blue border (lines 32–41)**

```typescript
function getStyle(feature: Feature | undefined): L.PathOptions {
  const code = (feature as NutsFeature)?.properties?.NUTS_ID ?? '';
  const score = yearScores[code] ?? 0;
  const isSelected = code === selectedRegion;
  const isCompare  = code === compareRegion;
  return {
    fillColor: anomalyColor(score, data.maxScore),
    fillOpacity: 0.82,
    color:  isSelected ? '#ffffff' : isCompare ? '#3b82f6' : '#1f2937',
    weight: isSelected || isCompare ? 2.5 : 0.4,
  };
}
```

- [ ] **Step 4: Replace the return statement (lines 77–95) with a wrapper div + banner + updated key**

The new return wraps `<MapContainer>` in a `relative` div so the banner can be absolutely positioned inside it. The GeoJSON `key` now includes `compareMode` (forces click-handler rebinding when compare mode activates) and `compareRegion` (forces rebinding when Region B is set).

```tsx
return (
  <div className="relative h-full w-full">
    {compareMode && !compareRegion && (
      <div
        className="absolute top-2 left-1/2 -translate-x-1/2 z-[1000]
                   bg-blue-900/90 text-blue-200 text-xs px-3 py-1 rounded-full
                   pointer-events-none select-none"
      >
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
      <TileLayer url={DARK_TILE} attribution={TILE_ATTR} />
      <GeoJSON
        key={`${activeYear}-${selectedRegion ?? 'none'}-${compareRegion ?? 'none'}-${compareMode}`}
        data={data.geoJson as FeatureCollection}
        style={getStyle}
        onEachFeature={onEachFeature}
        ref={(layer) => { geoJsonRef.current = layer as unknown as L.GeoJSON; }}
      />
    </MapContainer>
  </div>
);
```

- [ ] **Step 5: Run tests**

```bash
npm test
```

Expected: `11 tests passed`.

- [ ] **Step 6: Commit**

```bash
git add EU-GNN-Risk-Monitor/dashboard/src/components/EuropeMap.tsx
git commit -m "feat(dashboard): EuropeMap — compareRegion blue border, compare banner, updated key"
```

---

### Task 5: App.tsx — wire compareMode state and conditional render

**Files:**
- Modify: `EU-GNN-Risk-Monitor/dashboard/src/App.tsx`

**Context:** This task wires all previous tasks together. The full updated `App.tsx` is shown below. Replace the entire file contents.

- [ ] **Step 1: Replace the full contents of App.tsx**

```typescript
import { useState, useEffect } from 'react';
import { useDashboardData } from './hooks/useDashboardData';
import HeaderBar from './components/HeaderBar';
import KpiBar from './components/KpiBar';
import EuropeMap from './components/EuropeMap';
import TimeSeriesPanel from './components/TimeSeriesPanel';
import RankingPanel from './components/RankingPanel';
import ComparePanel from './components/ComparePanel';

export default function App() {
  const { data, error } = useDashboardData();
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [, setHoveredRegion] = useState<string | null>(null);
  const [activeYear, setActiveYear] = useState(2023);
  const [isPlaying, setIsPlaying] = useState(false);
  const [filterCountry, setFilterCountry] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareRegion, setCompareRegion] = useState<string | null>(null);

  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => {
      setActiveYear((y) => {
        if (y >= 2023) {
          setIsPlaying(false);
          return 2010;
        }
        return y + 1;
      });
    }, 800);
    return () => clearInterval(id);
  }, [isPlaying]);

  function handleRegionClick(code: string) {
    if (compareMode) {
      setCompareRegion(code);
    } else {
      setSelectedRegion(code);
    }
  }

  function handleCompareActivate() {
    setCompareMode(true);
    setCompareRegion(null);
  }

  function handleCompareExit() {
    setCompareMode(false);
    setCompareRegion(null);
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-950 text-red-400">
        Error loading data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-950 text-gray-500">
        Loading dashboard data…
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">
      <HeaderBar
        activeYear={activeYear}
        isPlaying={isPlaying}
        onYearChange={setActiveYear}
        onPlayToggle={() => setIsPlaying((p) => !p)}
      />
      <KpiBar data={data} />
      <div className="flex flex-1 min-h-0">
        {/* Left: Map (~50%) */}
        <div className="flex-1 min-w-0">
          <EuropeMap
            data={data}
            activeYear={activeYear}
            selectedRegion={selectedRegion}
            compareMode={compareMode}
            compareRegion={compareRegion}
            onRegionClick={handleRegionClick}
            onRegionHover={setHoveredRegion}
          />
        </div>
        {/* Middle: Time Series (~25%) */}
        <div className="w-72 shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
          <TimeSeriesPanel
            selectedRegion={selectedRegion}
            data={data}
            activeYear={activeYear}
          />
        </div>
        {/* Right: Ranking or Compare (~25%) */}
        <div className="w-72 shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
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
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Run the full build to verify TypeScript**

```bash
cd EU-GNN-Risk-Monitor/dashboard && npm run build
```

Expected: Build succeeds with zero TypeScript errors. `dist/` directory is created.

- [ ] **Step 3: Run all tests**

```bash
npm test
```

Expected: `11 tests passed`. Zero failures.

- [ ] **Step 4: Commit**

```bash
git add EU-GNN-Risk-Monitor/dashboard/src/App.tsx
git commit -m "feat(dashboard): phase5 complete — compareMode state, ComparePanel wired, EuropeMap routing"
```
