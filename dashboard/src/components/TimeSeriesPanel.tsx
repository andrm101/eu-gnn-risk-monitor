import { useState, useEffect } from 'react';
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
import type { DashboardData, ForecastPoint } from '../types';
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
    result[Number(yr)] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  }
  return result;
}

const NEIGHBOUR_COLORS = ['#6b7280', '#9ca3af', '#d1d5db'];
const TOP3_COLORS = ['#ef4444', '#f97316', '#f59e0b'];

// Chart row type — allows arbitrary keys for Recharts dynamic dataKey usage
type ChartRow = { year: number; fPoint?: number; fLower?: number; fBand?: number; [key: string]: unknown };

export default function TimeSeriesPanel({ selectedRegion, data, activeYear }: Props) {
  const euMean = euMeanByYear(data);

  const [showForecast, setShowForecast] = useState(false);

  // Reset toggle when region changes so we don't show a stale forecast state
  useEffect(() => {
    setShowForecast(false);
  }, [selectedRegion]);

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

  const hasForecast =
    (data.forecast[selectedRegion]?.length ?? 0) > 0;

  const chartData: ChartRow[] = ts.scores.map(({ year, anomaly_score }) => {
    const point: ChartRow = {
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

  let displayData: ChartRow[] = chartData;

  if (showForecast && hasForecast) {
    const fPts: ForecastPoint[] = data.forecast[selectedRegion];

    // Bridge the last actual data point into the forecast series for a seamless line join
    const lastActual = chartData[chartData.length - 1];
    const bridgeScore = lastActual[selectedRegion] as number;

    const bridgeRow: ChartRow = {
      ...lastActual,
      fPoint: bridgeScore,
      fLower: bridgeScore,
      fBand: 0,
    };

    const forecastRows: ChartRow[] = fPts.map(fp => ({
      year: fp.year,
      [selectedRegion]: undefined,
      eu_mean: null,
      fPoint: fp.point,
      fLower: fp.lower,
      fBand: fp.upper - fp.lower,
    }));

    displayData = [
      ...chartData.slice(0, -1),
      bridgeRow,
      ...forecastRows,
    ];
  }

  return (
    <div className="p-4 flex flex-col h-full min-h-0">
      <div className="mb-3 shrink-0">
        <div className="flex items-center justify-between gap-2">
          <div className="font-semibold text-sm text-white truncate">{name}</div>
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
        </div>
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
          <ComposedChart data={displayData} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
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
            {showForecast && hasForecast && (
              <>
                {/* Separator between actual and projected */}
                <ReferenceLine
                  x={ts.scores[ts.scores.length - 1].year}
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
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {showForecast && hasForecast && (
        <p className="text-xs text-gray-500 mt-1 text-center">
          Projected 2024–2026 · OLS linear trend · 95% prediction interval
        </p>
      )}
    </div>
  );
}
