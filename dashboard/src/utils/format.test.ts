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
