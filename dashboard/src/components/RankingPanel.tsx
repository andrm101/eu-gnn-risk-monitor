import type { DashboardData, RegionTimeSeries } from '../types';
import { anomalyColor } from '../utils/color';

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
  onCompareActivate, compareDisabled,
}: Props) {
  const displayed = filterCountry
    ? data.topRegions.filter((r) => r.country_code === filterCountry)
    : data.topRegions;

  return (
    <div className="flex flex-col h-full min-h-0">
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

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto">
        {displayed.map((r) => {
          const name = data.regionNames[r.nuts2_code] ?? r.nuts2_code;
          const isSelected = r.nuts2_code === selectedRegion;
          const barPct = data.maxScore === 0 ? 0 : (r.peak_score / data.maxScore) * 100;

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
