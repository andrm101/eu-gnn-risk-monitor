import type { DashboardData } from '../types';

interface Props { data: DashboardData }

function Chip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`font-bold text-sm ${color}`}>{value}</span>
      <span className="text-gray-500 text-xs">{label}</span>
    </div>
  );
}

export default function KpiBar({ data }: Props) {
  const top = data.topRegions[0];
  const topName = top ? (data.regionNames[top.nuts2_code] ?? top.nuts2_code) : '—';
  return (
    <div className="flex items-center gap-6 px-4 h-9 bg-gray-900 border-b border-gray-800 shrink-0">
      <Chip label="regions" value="242" color="text-blue-400" />
      <Chip label="peak risk" value={topName} color="text-red-400" />
      <Chip label="system peak year" value={String(data.systemPeakYear)} color="text-amber-400" />
      <Chip label="EU countries" value={String(data.countries.length)} color="text-green-400" />
    </div>
  );
}
