# Phase 4 — React Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static React 3-panel dashboard (EU choropleth map | time-series panel | ranking panel) that visualises GNN anomaly scores for 242 EU NUTS2 regions across 2010–2023.

**Architecture:** Vite + React 18 + TypeScript scaffolded from `RO-Administrative-Reform/dashboard` (identical stack). A Python export script converts pipeline parquet outputs to static JSON files consumed by the React app. All data is static — no server required.

**Tech Stack:** Vite 5, React 18, TypeScript 5, react-leaflet 4, Recharts 2, Tailwind 3, vitest (JS tests), pytest (Python tests).

**Spec:** `docs/superpowers/specs/2026-06-07-phase4-react-dashboard-design.md`

---

## File Map

| File | Action | Task |
|---|---|---|
| `scripts/export_dashboard_data.py` | Create | 1 |
| `tests/test_export_dashboard.py` | Create | 1 |
| `dashboard/package.json` | Create (adapted from RO) | 2 |
| `dashboard/index.html` | Create (adapted from RO) | 2 |
| `dashboard/vite.config.ts` | Create (adapted from RO) | 2 |
| `dashboard/tailwind.config.ts` | Copy from RO | 2 |
| `dashboard/postcss.config.cjs` | Copy from RO | 2 |
| `dashboard/tsconfig.json` | Copy from RO | 2 |
| `dashboard/tsconfig.node.json` | Copy from RO | 2 |
| `dashboard/src/main.tsx` | Copy from RO | 2 |
| `dashboard/src/index.css` | Copy from RO | 2 |
| `dashboard/src/types.ts` | Create | 3 |
| `dashboard/src/utils/color.ts` | Create | 3 |
| `dashboard/src/utils/color.test.ts` | Create | 3 |
| `dashboard/src/utils/format.ts` | Create | 3 |
| `dashboard/src/utils/format.test.ts` | Create | 3 |
| `dashboard/src/hooks/useDashboardData.ts` | Create | 4 |
| `dashboard/src/App.tsx` | Create | 5 |
| `dashboard/src/components/HeaderBar.tsx` | Create | 5 |
| `dashboard/src/components/KpiBar.tsx` | Create | 5 |
| `dashboard/src/components/EuropeMap.tsx` | Create | 6 |
| `dashboard/src/components/TimeSeriesPanel.tsx` | Create | 7 |
| `dashboard/src/components/RankingPanel.tsx` | Create | 8 |

---

## Task 1: Python Export Script

**Context:** The React app reads three static files from `dashboard/public/data/`. This script generates them from the Phase 3 pipeline outputs. All three pipeline artefacts (`nuts2_risk_scores.parquet`, `nuts2_graph.pt`, `nuts2_node_index.parquet`) must already exist from running `train_phase3.py`.

**Files:**
- Create: `EU-GNN-Risk-Monitor/scripts/export_dashboard_data.py`
- Create: `EU-GNN-Risk-Monitor/tests/test_export_dashboard.py`

- [ ] **Step 1: Write the failing test**

Run from `EU-GNN-Risk-Monitor/` directory. Create `tests/test_export_dashboard.py`:

```python
"""Integration tests for export_dashboard_data.py.
Skipped if pipeline artefacts are not present.
"""
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / 'dashboard' / 'public' / 'data'
SCORES_PATH = ROOT / 'data' / 'processed' / 'nuts2_risk_scores.parquet'


@pytest.fixture(scope='session', autouse=False)
def run_export():
    if not SCORES_PATH.exists():
        pytest.skip('Pipeline artefacts missing — run train_phase3.py first')
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'export_dashboard_data.py')],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return OUT


def test_risk_scores_json(run_export):
    path = run_export / 'risk_scores.json'
    assert path.exists()
    records = json.loads(path.read_text())
    assert len(records) == 3388  # 242 regions × 14 years
    first = records[0]
    assert {'nuts2_code', 'country_code', 'year', 'anomaly_score', 'peak_score', 'peak_rank'} <= first.keys()
    assert isinstance(first['year'], int)
    assert isinstance(first['anomaly_score'], float)


def test_neighbours_json(run_export):
    path = run_export / 'neighbours.json'
    assert path.exists()
    neighbours = json.loads(path.read_text())
    assert len(neighbours) == 242
    # Every value is a list of strings
    for code, nbrs in neighbours.items():
        assert isinstance(code, str)
        assert isinstance(nbrs, list)
        assert all(isinstance(n, str) for n in nbrs)


def test_nuts2_geojson(run_export):
    path = run_export / 'nuts2.geojson'
    assert path.exists()
    gj = json.loads(path.read_text())
    assert gj['type'] == 'FeatureCollection'
    codes = {f['properties']['CNTR_CODE'] for f in gj['features']}
    # All 27 EU member states present
    EU27 = {'AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','GR','HU',
             'IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE'}
    assert EU27 <= codes, f"Missing countries: {EU27 - codes}"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd EU-GNN-Risk-Monitor
python -m pytest tests/test_export_dashboard.py -v
```

Expected: SKIP (artefacts present) or FAIL with "No such file or directory: scripts/export_dashboard_data.py"

- [ ] **Step 3: Create the export script**

Create `scripts/export_dashboard_data.py`:

```python
"""Export pipeline outputs to static JSON for the React dashboard."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json
import urllib.request
import pandas as pd
import torch

ROOT = Path(__file__).parents[1]
OUT = ROOT / 'dashboard' / 'public' / 'data'

EU27 = {
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
    'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
}
GISCO_URL = (
    'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/'
    'NUTS_RG_20M_2021_4326_LEVL_2.geojson'
)


def export_risk_scores(root: Path, out: Path) -> None:
    df = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_risk_scores.parquet')
    records = df.assign(
        anomaly_score=df['anomaly_score'].round(6).astype(float),
        peak_score=df['peak_score'].round(6).astype(float),
        peak_rank=df['peak_rank'].astype(int),
        year=df['year'].astype(int),
    ).to_dict(orient='records')
    (out / 'risk_scores.json').write_text(json.dumps(records, ensure_ascii=False))
    print(f'  risk_scores.json   {len(records)} records')


def export_neighbours(root: Path, out: Path) -> None:
    g = torch.load(root / 'data' / 'processed' / 'nuts2_graph.pt', weights_only=False)
    ni = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_node_index.parquet')
    idx2code = ni.set_index('node_idx')['nuts2_code'].to_dict()

    ei = g[('region', 'spatial', 'region')].edge_index.numpy()
    adj: dict[str, list[str]] = defaultdict(list)
    for s, d in zip(ei[0], ei[1]):
        adj[idx2code[int(s)]].append(idx2code[int(d)])

    (out / 'neighbours.json').write_text(json.dumps(dict(adj), ensure_ascii=False))
    print(f'  neighbours.json    {len(adj)} regions')


def export_geojson(out: Path) -> None:
    print('  nuts2.geojson      downloading from GISCO...')
    with urllib.request.urlopen(GISCO_URL) as r:
        full = json.load(r)
    features = [f for f in full['features'] if f['properties']['CNTR_CODE'] in EU27]
    geojson = {'type': 'FeatureCollection', 'features': features}
    (out / 'nuts2.geojson').write_text(json.dumps(geojson, ensure_ascii=False))
    print(f'  nuts2.geojson      {len(features)} features')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    print('Exporting dashboard data...')
    export_risk_scores(ROOT, OUT)
    export_neighbours(ROOT, OUT)
    export_geojson(OUT)
    print('Done.')
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd EU-GNN-Risk-Monitor
python -m pytest tests/test_export_dashboard.py -v
```

Expected: 3 PASSED (requires internet for GeoJSON download, ~5–15s)

- [ ] **Step 5: Commit**

```bash
git add scripts/export_dashboard_data.py tests/test_export_dashboard.py
git commit -m "feat(phase4): Python export script — risk_scores.json + neighbours.json + nuts2.geojson"
```

---

## Task 2: Dashboard Scaffold

**Context:** The React app lives in `EU-GNN-Risk-Monitor/dashboard/`. We copy the scaffold from `RO-Administrative-Reform/dashboard` (identical Vite + React 18 + TS + react-leaflet + Recharts + Tailwind stack) and adapt three files: `package.json`, `index.html`, `vite.config.ts`. All paths below are relative to `EU-GNN-Risk-Monitor/`.

**Files:** Create the entire `dashboard/` directory structure.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p dashboard/src/components dashboard/src/hooks dashboard/src/utils
mkdir -p dashboard/public/data
```

- [ ] **Step 2: Create `dashboard/package.json`**

```json
{
  "name": "eu-gnn-risk-dashboard",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "deploy": "npm run build && gh-pages -d dist"
  },
  "dependencies": {
    "leaflet": "^1.9.4",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-leaflet": "^4.2.1",
    "recharts": "^2.12.7"
  },
  "devDependencies": {
    "@types/geojson": "^7946.0.14",
    "@types/leaflet": "^1.9.12",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.19",
    "gh-pages": "^6.1.1",
    "postcss": "^8.4.39",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.2.2",
    "vite": "^5.3.1",
    "vitest": "^2.1.9"
  }
}
```

- [ ] **Step 3: Create `dashboard/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>EU NUTS2 GNN Risk Monitor</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: Create `dashboard/vite.config.ts`**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/EU-GNN-Risk-Monitor/',
})
```

- [ ] **Step 5: Copy verbatim files from RO project**

Copy the following files from `../RO-Administrative-Reform/dashboard/` into `dashboard/`:
- `tailwind.config.ts`
- `postcss.config.cjs`
- `tsconfig.json`
- `tsconfig.node.json`
- `src/main.tsx`

For `src/index.css`, create it (note: no Leaflet CSS import in `index.html` — import it via JS instead):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root {
  height: 100%;
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
}

.leaflet-container {
  height: 100%;
  width: 100%;
}

.leaflet-tooltip {
  background: #1f2937;
  border: 1px solid #374151;
  color: #f9fafb;
  font-size: 12px;
}
```

- [ ] **Step 6: Install dependencies and verify dev server starts**

```bash
cd dashboard
npm install
npm run dev
```

Expected: Vite starts on http://localhost:5173, shows blank page (no App yet — that's fine). Check for no install errors.

- [ ] **Step 7: Commit scaffold**

```bash
cd ..
git add dashboard/
git commit -m "feat(phase4): scaffold React dashboard (Vite + React18 + TS + react-leaflet + Recharts + Tailwind)"
```

---

## Task 3: TypeScript Types and Utilities

**Context:** The types file is the contract between all components. Define it first so every subsequent task can import from it without circular dependencies. Color and format utilities are pure functions — easy to unit-test with vitest.

**Files:**
- Create: `dashboard/src/types.ts`
- Create: `dashboard/src/utils/color.ts`
- Create: `dashboard/src/utils/color.test.ts`
- Create: `dashboard/src/utils/format.ts`
- Create: `dashboard/src/utils/format.test.ts`

- [ ] **Step 1: Write the failing utility tests**

Create `dashboard/src/utils/color.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { anomalyColor } from './color';

describe('anomalyColor', () => {
  it('returns dark blue for score=0', () => {
    expect(anomalyColor(0, 1)).toBe('#1e3a5f');
  });

  it('returns purple for score=maxScore', () => {
    expect(anomalyColor(1, 1)).toBe('#7c3aed');
  });

  it('returns a valid hex string for mid-range score', () => {
    const c = anomalyColor(0.5, 1);
    expect(c).toMatch(/^#[0-9a-f]{6}$/);
  });

  it('clamps score above maxScore to max stop', () => {
    expect(anomalyColor(2, 1)).toBe('#7c3aed');
  });

  it('returns dark blue when maxScore is 0', () => {
    expect(anomalyColor(0, 0)).toBe('#1e3a5f');
  });
});
```

Create `dashboard/src/utils/format.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { formatScore, formatRank } from './format';

describe('formatScore', () => {
  it('formats to 4 decimal places', () => {
    expect(formatScore(0.20712)).toBe('0.2071');
  });

  it('pads zeros to 4 decimal places', () => {
    expect(formatScore(0.1)).toBe('0.1000');
  });
});

describe('formatRank', () => {
  it('prefixes rank with #', () => {
    expect(formatRank(1)).toBe('#1');
    expect(formatRank(242)).toBe('#242');
  });
});
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd dashboard
npm test
```

Expected: FAIL — `color.ts` and `format.ts` not found.

- [ ] **Step 3: Create `dashboard/src/types.ts`**

```typescript
import type { FeatureCollection } from 'geojson';

export interface RiskScore {
  nuts2_code: string;
  country_code: string;
  year: number;
  anomaly_score: number;
  peak_score: number;
  peak_rank: number;
}

export interface RegionTimeSeries {
  nuts2_code: string;
  country_code: string;
  peak_score: number;
  peak_rank: number;
  peak_year: number;
  scores: { year: number; anomaly_score: number }[];
}

export interface DashboardData {
  scores: RiskScore[];
  neighbours: Record<string, string[]>;
  geoJson: FeatureCollection;
  regionNames: Record<string, string>;      // nuts2_code → NAME_LATN from GeoJSON
  byCode: Record<string, RegionTimeSeries>; // nuts2_code → derived time series
  byYear: Record<number, Record<string, number>>; // year → {nuts2_code → anomaly_score}
  maxScore: number;
  systemPeakYear: number;    // year with highest mean anomaly across all regions
  topRegions: RegionTimeSeries[];            // sorted ascending by peak_rank
  countries: string[];                       // sorted unique country codes
}
```

- [ ] **Step 4: Create `dashboard/src/utils/color.ts`**

```typescript
function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * Math.min(1, Math.max(0, t)));
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('');
}

// Sequential anomaly scale: dark-blue (low) → amber (mid) → red (high) → purple (extreme)
const STOPS: [number, string][] = [
  [0.00, '#1e3a5f'],
  [0.40, '#f59e0b'],
  [0.75, '#ef4444'],
  [1.00, '#7c3aed'],
];

export function anomalyColor(score: number, maxScore: number): string {
  if (maxScore === 0) return STOPS[0][1];
  const t = Math.min(score / maxScore, 1);
  for (let i = 1; i < STOPS.length; i++) {
    const [t0, c0] = STOPS[i - 1];
    const [t1, c1] = STOPS[i];
    if (t <= t1) {
      const u = (t - t0) / (t1 - t0);
      const [r0, g0, b0] = hexToRgb(c0);
      const [r1, g1, b1] = hexToRgb(c1);
      return rgbToHex(lerp(r0, r1, u), lerp(g0, g1, u), lerp(b0, b1, u));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export const SCORE_STOPS = STOPS;
```

- [ ] **Step 5: Create `dashboard/src/utils/format.ts`**

```typescript
export function formatScore(x: number): string {
  return x.toFixed(4);
}

export function formatRank(rank: number): string {
  return `#${rank}`;
}
```

- [ ] **Step 6: Run tests and confirm they pass**

```bash
cd dashboard
npm test
```

Expected: 8 PASSED (5 color + 3 format)

- [ ] **Step 7: Verify TypeScript compiles**

```bash
cd dashboard
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 8: Commit**

```bash
cd ..
git add dashboard/src/types.ts dashboard/src/utils/
git commit -m "feat(phase4): TypeScript types + color/format utilities with vitest coverage"
```

---

## Task 4: Data Hook

**Context:** `useDashboardData` fetches three JSON files and derives all the structured data the components need. The hook's derive logic is the most complex non-UI logic in the project — get it right before wiring up components.

**Files:**
- Create: `dashboard/src/hooks/useDashboardData.ts`

- [ ] **Step 1: Create `dashboard/src/hooks/useDashboardData.ts`**

```typescript
import { useEffect, useState } from 'react';
import type { FeatureCollection, Feature } from 'geojson';
import type { RiskScore, RegionTimeSeries, DashboardData } from '../types';

interface NutsProperties { NUTS_ID: string; NAME_LATN: string }

const BASE = import.meta.env.BASE_URL;

async function fetchJson<T>(name: string): Promise<T> {
  const res = await fetch(`${BASE}data/${name}`);
  if (!res.ok) throw new Error(`Failed to fetch ${name}: ${res.status}`);
  return res.json() as Promise<T>;
}

function deriveData(
  scores: RiskScore[],
  neighbours: Record<string, string[]>,
  geoJson: FeatureCollection,
): DashboardData {
  // Region names from GeoJSON
  const regionNames: Record<string, string> = {};
  for (const f of geoJson.features) {
    const p = (f as Feature & { properties: NutsProperties }).properties;
    regionNames[p.NUTS_ID] = p.NAME_LATN;
  }

  // byCode: nuts2_code → RegionTimeSeries
  const codeMap = new Map<string, RiskScore[]>();
  for (const s of scores) {
    const arr = codeMap.get(s.nuts2_code) ?? [];
    arr.push(s);
    codeMap.set(s.nuts2_code, arr);
  }

  const byCode: Record<string, RegionTimeSeries> = {};
  codeMap.forEach((rows, code) => {
    const sorted = rows.slice().sort((a, b) => a.year - b.year);
    const peakRow = sorted.reduce((a, b) => (a.anomaly_score > b.anomaly_score ? a : b));
    byCode[code] = {
      nuts2_code: code,
      country_code: rows[0].country_code,
      peak_score: rows[0].peak_score,
      peak_rank: rows[0].peak_rank,
      peak_year: peakRow.year,
      scores: sorted.map((r) => ({ year: r.year, anomaly_score: r.anomaly_score })),
    };
  });

  // byYear: year → {nuts2_code → anomaly_score}
  const byYear: Record<number, Record<string, number>> = {};
  for (const s of scores) {
    if (!byYear[s.year]) byYear[s.year] = {};
    byYear[s.year][s.nuts2_code] = s.anomaly_score;
  }

  const maxScore = Math.max(...scores.map((s) => s.anomaly_score));

  // systemPeakYear: year with highest mean anomaly
  const systemPeakYear = Object.entries(byYear)
    .map(([yr, m]) => {
      const vals = Object.values(m);
      return { year: Number(yr), mean: vals.reduce((a, b) => a + b, 0) / vals.length };
    })
    .reduce((a, b) => (a.mean > b.mean ? a : b)).year;

  const topRegions = Object.values(byCode).sort((a, b) => a.peak_rank - b.peak_rank);
  const countries = [...new Set(scores.map((s) => s.country_code))].sort();

  return { scores, neighbours, geoJson, regionNames, byCode, byYear, maxScore, systemPeakYear, topRegions, countries };
}

export function useDashboardData(): { data: DashboardData | null; error: string | null } {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchJson<RiskScore[]>('risk_scores.json'),
      fetchJson<Record<string, string[]>>('neighbours.json'),
      fetchJson<FeatureCollection>('nuts2.geojson'),
    ])
      .then(([scores, neighbours, geoJson]) => setData(deriveData(scores, neighbours, geoJson)))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return { data, error };
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd dashboard
npx tsc --noEmit
```

Expected: No errors (hook imports from types.ts which exists).

- [ ] **Step 3: Commit**

```bash
cd ..
git add dashboard/src/hooks/useDashboardData.ts
git commit -m "feat(phase4): useDashboardData hook — fetch + derive DashboardData from 3 JSON files"
```

---

## Task 5: App Root + HeaderBar + KpiBar

**Context:** `App.tsx` owns all shared state and renders the three-panel layout. `HeaderBar` and `KpiBar` are stateless display components. Build them together since they are structurally simple and wired to the same state.

**Files:**
- Create: `dashboard/src/App.tsx`
- Create: `dashboard/src/components/HeaderBar.tsx`
- Create: `dashboard/src/components/KpiBar.tsx`

- [ ] **Step 1: Create `dashboard/src/components/HeaderBar.tsx`**

```typescript
interface Props {
  activeYear: number;
  isPlaying: boolean;
  onYearChange: (year: number) => void;
  onPlayToggle: () => void;
}

export default function HeaderBar({ activeYear, isPlaying, onYearChange, onPlayToggle }: Props) {
  return (
    <header className="flex items-center px-4 h-12 bg-gray-900 border-b border-gray-800 shrink-0 gap-4">
      <span className="font-bold text-sm text-white">EU NUTS2 Risk Monitor</span>
      <span className="text-gray-500 text-xs hidden sm:block">
        GNN Anomaly Detection · 2010–2023
      </span>
      <div className="ml-auto flex items-center gap-3">
        <button
          onClick={onPlayToggle}
          className="text-gray-400 hover:text-white transition-colors text-lg w-6 text-center"
          title={isPlaying ? 'Pause animation' : 'Play animation'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <input
          type="range"
          min={2010}
          max={2023}
          value={activeYear}
          onChange={(e) => onYearChange(Number(e.target.value))}
          className="w-36 accent-blue-500 cursor-pointer"
        />
        <span className="text-blue-400 font-mono text-sm w-10 text-center">{activeYear}</span>
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Create `dashboard/src/components/KpiBar.tsx`**

```typescript
import type { DashboardData } from '../types';

interface Props { data: DashboardData }

function Chip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`font-bold text-sm ${color}`}>{value}</span>
      <span className="text-gray-500 text-xs">{label}</span>
    </div>
  );
}

export default function KpiBar({ data }: Props) {
  const top = data.topRegions[0];
  const topName = top ? (data.regionNames[top.nuts2_code] ?? top.nuts2_code) : '—';
  return (
    <div className="flex items-center gap-6 px-4 h-9 bg-gray-900 border-b border-gray-800 shrink-0">
      <Chip label="regions" value="242" color="text-blue-400" />
      <Chip label="peak risk" value={topName} color="text-red-400" />
      <Chip label="system peak year" value={String(data.systemPeakYear)} color="text-amber-400" />
      <Chip label="EU countries" value={String(data.countries.length)} color="text-green-400" />
    </div>
  );
}
```

- [ ] **Step 3: Create `dashboard/src/App.tsx`**

```typescript
import { useState, useEffect } from 'react';
import { useDashboardData } from './hooks/useDashboardData';
import HeaderBar from './components/HeaderBar';
import KpiBar from './components/KpiBar';
// NOTE: these imports will fail until Tasks 6-8 create the component files.
// Until then, run tsc only on files that exist; the full compile passes after Task 8.
import EuropeMap from './components/EuropeMap';
import TimeSeriesPanel from './components/TimeSeriesPanel';
import RankingPanel from './components/RankingPanel';

export default function App() {
  const { data, error } = useDashboardData();
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [activeYear, setActiveYear] = useState(2023);
  const [isPlaying, setIsPlaying] = useState(false);
  const [filterCountry, setFilterCountry] = useState<string | null>(null);

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
            hoveredRegion={hoveredRegion}
            onRegionClick={setSelectedRegion}
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
        {/* Right: Ranking (~25%) */}
        <div className="w-72 shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
          <RankingPanel
            data={data}
            selectedRegion={selectedRegion}
            filterCountry={filterCountry}
            onRegionClick={setSelectedRegion}
            onRegionHover={setHoveredRegion}
            onCountryFilter={setFilterCountry}
          />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify the files that exist compile cleanly**

TypeScript will error on the three missing component imports in App.tsx — that's expected and will resolve after Tasks 6–8. Verify the files created so far have no type issues by checking only those files:

```bash
cd dashboard
npx tsc --noEmit 2>&1
```

Expected: Errors only for `Cannot find module './components/EuropeMap'`, `TimeSeriesPanel`, `RankingPanel`. No other errors.

- [ ] **Step 5: Commit**

```bash
cd ..
git add dashboard/src/App.tsx dashboard/src/components/HeaderBar.tsx dashboard/src/components/KpiBar.tsx
git commit -m "feat(phase4): App layout + HeaderBar (year slider + play) + KpiBar (4 KPI chips)"
```

---

## Task 6: EuropeMap Component

**Context:** The most complex component. Uses react-leaflet to render 242 NUTS2 GeoJSON polygons with anomaly-score-based fill colors. Year changes trigger a `key` remount (full color rebuild). Hover is handled imperatively via Leaflet's `setStyle`/`resetStyle` — no remount on hover.

**Files:**
- Create: `dashboard/src/components/EuropeMap.tsx`

- [ ] **Step 1: Create `dashboard/src/components/EuropeMap.tsx`**

```typescript
import { useRef, useCallback } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Feature, FeatureCollection } from 'geojson';
import type { DashboardData } from '../types';
import { anomalyColor } from '../utils/color';
import { formatScore, formatRank } from '../utils/format';

const DARK_TILE = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_ATTR = '© <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors, © <a href="https://carto.com/">CARTO</a>';
const EU_BOUNDS: L.LatLngBoundsExpression = [[34, -25], [72, 45]];

interface NutsFeature extends Feature {
  properties: { NUTS_ID: string; NAME_LATN: string; CNTR_CODE: string };
}

interface Props {
  data: DashboardData;
  activeYear: number;
  selectedRegion: string | null;
  hoveredRegion: string | null;
  onRegionClick: (code: string) => void;
  onRegionHover: (code: string | null) => void;
}

export default function EuropeMap({
  data, activeYear, selectedRegion, onRegionClick, onRegionHover,
}: Props) {
  const geoJsonRef = useRef<L.GeoJSON | null>(null);
  const yearScores = data.byYear[activeYear] ?? {};

  function getStyle(feature: Feature | undefined): L.PathOptions {
    const code = (feature as NutsFeature)?.properties?.NUTS_ID ?? '';
    const score = yearScores[code] ?? 0;
    const isSelected = code === selectedRegion;
    return {
      fillColor: anomalyColor(score, data.maxScore),
      fillOpacity: 0.82,
      color: isSelected ? '#ffffff' : '#1f2937',
      weight: isSelected ? 2.5 : 0.4,
    };
  }

  const onEachFeature = useCallback(
    (feature: Feature, layer: L.Layer) => {
      const f = feature as NutsFeature;
      const code = f.properties.NUTS_ID;
      const name = f.properties.NAME_LATN;
      const ts = data.byCode[code];
      const score = yearScores[code];
      const scoreStr = score != null ? formatScore(score) : 'N/A';
      const rankStr = ts ? formatRank(ts.peak_rank) : '—';

      (layer as L.Path).bindTooltip(
        `<b>${name}</b> (${code})<br/>Score ${activeYear}: ${scoreStr}<br/>Peak rank: ${rankStr}`,
        { sticky: true },
      );

      layer.on({
        click: () => onRegionClick(code),
        mouseover: (e: L.LeafletMouseEvent) => {
          const target = e.target as L.Path;
          target.setStyle({ color: '#fbbf24', weight: 2 });
          target.bringToFront();
          onRegionHover(code);
        },
        mouseout: (e: L.LeafletMouseEvent) => {
          geoJsonRef.current?.resetStyle(e.target as L.Path);
          onRegionHover(null);
        },
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data.byCode, yearScores, activeYear, onRegionClick, onRegionHover],
  );

  return (
    <MapContainer
      bounds={EU_BOUNDS}
      minZoom={3}
      maxZoom={8}
      zoomControl
      style={{ height: '100%', width: '100%', background: '#030712' }}
    >
      <TileLayer url={DARK_TILE} attribution={TILE_ATTR} />
      <GeoJSON
        key={`${activeYear}-${selectedRegion ?? 'none'}`}
        data={data.geoJson as FeatureCollection}
        style={getStyle}
        onEachFeature={onEachFeature}
        ref={(layer) => { geoJsonRef.current = layer as unknown as L.GeoJSON; }}
      />
    </MapContainer>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd dashboard
npx tsc --noEmit 2>&1 | grep -v "Cannot find module"
```

Expected: No type errors in EuropeMap.tsx.

- [ ] **Step 3: Run dev server and verify the map renders**

```bash
npm run dev
```

Open http://localhost:5173. The data files must be in `dashboard/public/data/` (run `python scripts/export_dashboard_data.py` first if needed).

Expected: Dark basemap with 242 NUTS2 polygons coloured blue→amber→red, hover shows yellow border + tooltip, click selects a region (white border).

- [ ] **Step 4: Commit**

```bash
cd ..
git add dashboard/src/components/EuropeMap.tsx
git commit -m "feat(phase4): EuropeMap — NUTS2 choropleth, year-keyed remount, imperative hover"
```

---

## Task 7: TimeSeriesPanel Component

**Context:** Shows the selected region's anomaly trajectory (2010–2023) overlaid with up to 3 spatial graph neighbours and the EU mean. When no region is selected, shows the top-3 risk regions' trajectories instead.

**Files:**
- Create: `dashboard/src/components/TimeSeriesPanel.tsx`

- [ ] **Step 1: Create `dashboard/src/components/TimeSeriesPanel.tsx`**

```typescript
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, ReferenceLine, Legend,
} from 'recharts';
import type { DashboardData } from '../types';
import { anomalyColor } from '../utils/color';

interface Props {
  selectedRegion: string | null;
  data: DashboardData;
  activeYear: number;
}

// Pre-compute EU mean anomaly score per year
function euMeanByYear(data: DashboardData): Record<number, number> {
  const result: Record<number, number> = {};
  for (const [yr, scoreMap] of Object.entries(data.byYear)) {
    const vals = Object.values(scoreMap);
    result[Number(yr)] = vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  return result;
}

const NEIGHBOUR_COLORS = ['#6b7280', '#9ca3af', '#d1d5db'];
const TOP3_COLORS = ['#ef4444', '#f97316', '#f59e0b'];

export default function TimeSeriesPanel({ selectedRegion, data, activeYear }: Props) {
  const euMean = euMeanByYear(data);

  if (!selectedRegion) {
    const top3 = data.topRegions.slice(0, 3);
    const chartData = Array.from({ length: 14 }, (_, i) => {
      const year = 2010 + i;
      const point: Record<string, number | null> = { year };
      for (const r of top3) {
        point[r.nuts2_code] = r.scores.find((s) => s.year === year)?.anomaly_score ?? null;
      }
      return point;
    });

    return (
      <div className="p-4 flex flex-col h-full min-h-0">
        <p className="text-gray-500 text-xs mb-3 shrink-0">
          Click a region on the map to view its anomaly trajectory vs its GNN graph neighbours
        </p>
        <div className="text-xs text-gray-400 mb-2 font-medium shrink-0">Top 3 risk regions</div>
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
              <XAxis dataKey="year" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} width={42} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', fontSize: 11 }}
              />
              <ReferenceLine x={activeYear} stroke="#3b82f6" strokeDasharray="4 4" strokeWidth={1} />
              {top3.map((r, i) => (
                <Line
                  key={r.nuts2_code}
                  type="monotone"
                  dataKey={r.nuts2_code}
                  stroke={TOP3_COLORS[i]}
                  strokeWidth={2}
                  dot={false}
                  name={data.regionNames[r.nuts2_code] ?? r.nuts2_code}
                />
              ))}
              <Legend iconSize={8} wrapperStyle={{ fontSize: 10, color: '#9ca3af' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  }

  const ts = data.byCode[selectedRegion];
  if (!ts) return null;

  const name = data.regionNames[selectedRegion] ?? selectedRegion;
  const colour = anomalyColor(ts.peak_score, data.maxScore);
  const neighbours = (data.neighbours[selectedRegion] ?? []).slice(0, 3);

  const chartData = ts.scores.map(({ year, anomaly_score }) => {
    const point: Record<string, number | null> = {
      year,
      [selectedRegion]: anomaly_score,
      eu_mean: euMean[year] ?? null,
    };
    for (const nb of neighbours) {
      const nbTs = data.byCode[nb];
      point[nb] = nbTs?.scores.find((s) => s.year === year)?.anomaly_score ?? null;
    }
    return point;
  });

  return (
    <div className="p-4 flex flex-col h-full min-h-0">
      <div className="mb-3 shrink-0">
        <div className="font-semibold text-sm text-white truncate">{name}</div>
        <div className="text-gray-400 text-xs mt-0.5">
          {ts.country_code} · Peak rank {ts.peak_rank} · Peak year {ts.peak_year}
        </div>
        <div className="text-gray-600 text-xs mt-1">
          Anomaly score vs {neighbours.length} spatial GNN neighbours
        </div>
        <div className="text-gray-700 text-xs mt-0.5 italic">
          MSE of GNN reconstruction — higher = diverges more from graph neighbours
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
            <XAxis dataKey="year" tick={{ fontSize: 10, fill: '#6b7280' }} />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280' }}
              width={42}
              domain={[0, data.maxScore * 1.1]}
            />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151', fontSize: 11 }}
            />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10, color: '#9ca3af' }} />
            <ReferenceLine x={activeYear} stroke="#3b82f6" strokeDasharray="4 4" strokeWidth={1} />
            <Line
              type="monotone"
              dataKey={selectedRegion}
              stroke={colour}
              strokeWidth={2.5}
              dot={false}
              name={name}
            />
            <Line
              type="monotone"
              dataKey="eu_mean"
              stroke="#4b5563"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              name="EU mean"
            />
            {neighbours.map((nb, i) => (
              <Line
                key={nb}
                type="monotone"
                dataKey={nb}
                stroke={NEIGHBOUR_COLORS[i]}
                strokeWidth={1}
                strokeDasharray="2 2"
                dot={false}
                name={data.regionNames[nb] ?? nb}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd dashboard
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Verify visually in dev server**

Click a NUTS2 region on the map. Expected: time series panel updates with selected region's trajectory + up to 3 grey dashed neighbour lines + dark-grey EU mean line. Vertical blue line follows the year slider. With no selection, top-3 trajectories shown.

- [ ] **Step 4: Commit**

```bash
cd ..
git add dashboard/src/components/TimeSeriesPanel.tsx
git commit -m "feat(phase4): TimeSeriesPanel — anomaly trajectory + GNN neighbour overlay + EU mean"
```

---

## Task 8: RankingPanel Component

**Context:** Sorted list of all 242 regions by `peak_score` (descending), with inline SVG sparklines and a country filter. Clicking a row selects the region (cross-links with map and time series).

**Files:**
- Create: `dashboard/src/components/RankingPanel.tsx`

- [ ] **Step 1: Create `dashboard/src/components/RankingPanel.tsx`**

```typescript
import type { DashboardData, RegionTimeSeries } from '../types';
import { anomalyColor } from '../utils/color';

interface Props {
  data: DashboardData;
  selectedRegion: string | null;
  filterCountry: string | null;
  onRegionClick: (code: string) => void;
  onRegionHover: (code: string | null) => void;
  onCountryFilter: (code: string | null) => void;
}

function Sparkline({ scores, maxScore }: { scores: RegionTimeSeries['scores']; maxScore: number }) {
  const W = 56;
  const H = 20;
  const n = scores.length;
  if (n < 2) return null;
  const pts = scores
    .map((s, i) => {
      const x = (i / (n - 1)) * W;
      const y = H - Math.max(0, (s.anomaly_score / maxScore) * H);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={W} height={H} className="shrink-0 opacity-70">
      <polyline points={pts} fill="none" stroke="#6b7280" strokeWidth={1.2} />
    </svg>
  );
}

function rankColor(rank: number): string {
  if (rank === 1) return 'text-red-400 font-bold';
  if (rank <= 3) return 'text-orange-400 font-bold';
  if (rank <= 10) return 'text-amber-400';
  return 'text-gray-500';
}

export default function RankingPanel({
  data, selectedRegion, filterCountry, onRegionClick, onRegionHover, onCountryFilter,
}: Props) {
  const displayed = filterCountry
    ? data.topRegions.filter((r) => r.country_code === filterCountry)
    : data.topRegions;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Panel header */}
      <div className="px-3 py-2 border-b border-gray-800 shrink-0 flex items-center gap-2">
        <span className="text-xs text-gray-400 font-medium">Peak Risk Ranking</span>
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

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto">
        {displayed.map((r) => {
          const name = data.regionNames[r.nuts2_code] ?? r.nuts2_code;
          const isSelected = r.nuts2_code === selectedRegion;
          const barPct = (r.peak_score / data.maxScore) * 100;

          return (
            <div
              key={r.nuts2_code}
              onClick={() => onRegionClick(r.nuts2_code)}
              onMouseEnter={() => onRegionHover(r.nuts2_code)}
              onMouseLeave={() => onRegionHover(null)}
              className={`flex items-center gap-2 px-2 py-1.5 cursor-pointer hover:bg-gray-800 transition-colors ${
                isSelected ? 'bg-gray-700 ring-1 ring-inset ring-gray-600' : ''
              }`}
            >
              {/* Rank */}
              <span className={`w-7 text-xs font-mono text-right shrink-0 ${rankColor(r.peak_rank)}`}>
                {r.peak_rank}
              </span>

              {/* Name + score bar */}
              <div className="flex-1 min-w-0">
                <div className="text-xs text-white truncate leading-tight">{name}</div>
                <div className="flex items-center gap-1 mt-0.5">
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${barPct}%`,
                        background: anomalyColor(r.peak_score, data.maxScore),
                      }}
                    />
                  </div>
                  <span className="text-gray-500 text-xs w-11 text-right shrink-0">
                    {r.peak_score.toFixed(3)}
                  </span>
                </div>
              </div>

              {/* Sparkline */}
              <Sparkline scores={r.scores} maxScore={data.maxScore} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd dashboard
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Verify visually in dev server**

Expected:
- List shows 242 regions sorted by peak score
- Top rows are red (#1 PT15 Algarve), orange, amber
- Each row has a colour bar and a 14-point sparkline
- Country select filters the list
- Clicking a row updates the map selection and time series panel
- Hovering a row shows yellow border on map

- [ ] **Step 4: Commit**

```bash
cd ..
git add dashboard/src/components/RankingPanel.tsx
git commit -m "feat(phase4): RankingPanel — sorted peak-risk list, sparklines, country filter, cross-link"
```

---

## Task 9: Integration + Production Build

**Context:** Final integration check. Verify the full build pipeline (Python export → npm build) produces a working static site.

**Files:** No new files.

- [ ] **Step 1: Run full data export**

```bash
cd EU-GNN-Risk-Monitor
python scripts/export_dashboard_data.py
```

Expected output:
```
Exporting dashboard data...
  risk_scores.json   3388 records
  neighbours.json    242 regions
  nuts2.geojson      downloading from GISCO...
  nuts2.geojson      [N] features
Done.
```

Verify `dashboard/public/data/` contains `risk_scores.json`, `neighbours.json`, `nuts2.geojson`.

- [ ] **Step 2: Run TypeScript + vitest**

```bash
cd dashboard
npm test
npx tsc --noEmit
```

Expected: 8 vitest tests PASSED, tsc exits 0.

- [ ] **Step 3: Build the production bundle**

```bash
npm run build
```

Expected: `dist/` directory created, no TypeScript errors, no build errors.

- [ ] **Step 4: Preview the production build**

```bash
npm run preview
```

Open the URL shown (typically http://localhost:4173). Verify all 3 panels render correctly.

- [ ] **Step 5: Manual acceptance checklist**

Open the dev or preview server and verify:

- [ ] Map renders all 242 NUTS2 regions with colour variation (dark-blue to purple)
- [ ] Year slider changes the map colours
- [ ] Play button animates the choropleth year-by-year (2010→2023, then stops)
- [ ] Hovering a map region shows yellow border + tooltip with score and rank
- [ ] Clicking a map region selects it (white border) and updates the time series panel
- [ ] Time series panel shows the selected region's trajectory + neighbour lines + EU mean
- [ ] Vertical blue reference line in time series follows the year slider
- [ ] Ranking panel shows regions sorted by peak score, top rows red/orange
- [ ] Country filter in ranking panel narrows the list
- [ ] Clicking a ranking row selects the region on the map and time series
- [ ] Hovering a ranking row highlights the region on the map

- [ ] **Step 6: Final commit**

```bash
cd ..
git add dashboard/
git commit -m "feat(phase4): complete React dashboard — map + time series + ranking, production build passing"
```
