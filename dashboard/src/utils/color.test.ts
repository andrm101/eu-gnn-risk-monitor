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
