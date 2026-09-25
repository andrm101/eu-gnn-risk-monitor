// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import ComparePanel from './ComparePanel';
import type { DashboardData } from '../types';

afterEach(cleanup);

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
  forecast: {},
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

  it('shows fallback message when regionA is null', () => {
    render(<ComparePanel data={mockData} regionA={null} regionB={null} onExit={vi.fn()} />);
    expect(screen.getByText('Select a region first')).not.toBeNull();
  });
});
