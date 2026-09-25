import { useEffect, useState } from 'react';
import type { FeatureCollection, Feature } from 'geojson';
import type { RiskScore, RegionTimeSeries, DashboardData, ForecastPoint } from '../types';

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
  if (scores.length === 0) {
    return { scores, neighbours, geoJson, regionNames: {}, byCode: {}, byYear: {},
             maxScore: 0, systemPeakYear: 0, topRegions: [], countries: [], forecast: {} };
  }

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

  return { scores, neighbours, geoJson, regionNames, byCode, byYear, maxScore, systemPeakYear, topRegions, countries, forecast: {} };
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
      .then(async ([scores, neighbours, geoJson]) => {
        const dashboardData = deriveData(scores, neighbours, geoJson);
        const forecast: Record<string, ForecastPoint[]> = await fetch(`${BASE}data/forecast_scores.json`)
          .then(r => r.ok ? (r.json() as Promise<Record<string, ForecastPoint[]>>) : Promise.resolve({}))
          .catch(() => ({} as Record<string, ForecastPoint[]>));
        setData({ ...dashboardData, forecast });
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return { data, error };
}
