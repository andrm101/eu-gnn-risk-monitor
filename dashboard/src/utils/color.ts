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
