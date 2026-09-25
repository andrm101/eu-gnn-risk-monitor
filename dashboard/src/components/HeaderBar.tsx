interface Props {
  activeYear: number;
  isPlaying: boolean;
  onYearChange: (year: number) => void;
  onPlayToggle: () => void;
}

export default function HeaderBar({ activeYear, isPlaying, onYearChange, onPlayToggle }: Props) {
  return (
    <header className="flex items-center px-4 h-12 bg-gray-900 border-b border-gray-800 shrink-0 gap-4">
      <span className="font-bold text-sm text-white">EU NUTS2 Risk Monitor</span>
      <span className="text-gray-500 text-xs hidden sm:block">
        GNN Anomaly Detection · 2010–2023
      </span>
      <div className="ml-auto flex items-center gap-3">
        <button
          onClick={onPlayToggle}
          className="text-gray-400 hover:text-white transition-colors text-lg w-6 text-center"
          title={isPlaying ? 'Pause animation' : 'Play animation'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <input
          type="range"
          min={2010}
          max={2023}
          value={activeYear}
          onChange={(e) => onYearChange(Number(e.target.value))}
          className="w-36 accent-blue-500 cursor-pointer"
        />
        <span className="text-blue-400 font-mono text-sm w-10 text-center">{activeYear}</span>
      </div>
    </header>
  );
}
