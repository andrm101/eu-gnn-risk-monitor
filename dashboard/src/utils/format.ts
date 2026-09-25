export function formatScore(x: number): string {
  return x.toFixed(4);
}

export function formatRank(rank: number): string {
  return `#${rank}`;
}
