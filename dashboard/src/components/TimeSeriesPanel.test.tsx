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
    expect(checkbox).not.toBeNull();
    expect(screen.getByText(/forecast/i)).not.toBeNull();
  });

  it('shows attribution note when forecast is toggled on', () => {
    render(
      <TimeSeriesPanel data={dataWithForecast} selectedRegion="PT15" activeYear={2015} />,
    );
    const checkbox = screen.getByRole('checkbox');
    expect(screen.queryByText(/OLS linear trend/i)).toBeNull();
    fireEvent.click(checkbox);
    expect(screen.getByText(/OLS linear trend/i)).not.toBeNull();
  });
});
