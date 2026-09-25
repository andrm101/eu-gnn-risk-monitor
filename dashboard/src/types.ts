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

export interface ForecastPoint {
  year: number;
  point: number;
  lower: number;
  upper: number;
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
  forecast: Record<string, ForecastPoint[]>;
}
