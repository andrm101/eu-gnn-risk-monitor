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

function maxGap(a: RegionTimeSeries, b: RegionTimeSeries): GapResult | null {
  let best: GapResult | null = null;
  for (const ra of a.scores) {
    const rb = b.scores.find((r) => r.year === ra.year);
    if (!rb) continue;
    const delta = Math.abs(ra.anomaly_score - rb.anomaly_score);
    if (best === null || delta > best.delta) best = { year: ra.year, delta };
  }
  return best;
}

export default function ComparePanel({ data, regionA, regionB, onExit }: Props) {
  if (!regionA) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-gray-600 px-4 text-center">
        Select a region first
      </div>
    );
  }

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
