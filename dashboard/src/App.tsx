import { useState, useEffect } from 'react';
import { useDashboardData } from './hooks/useDashboardData';
import HeaderBar from './components/HeaderBar';
import KpiBar from './components/KpiBar';
import EuropeMap from './components/EuropeMap';
import TimeSeriesPanel from './components/TimeSeriesPanel';
import RankingPanel from './components/RankingPanel';
import ComparePanel from './components/ComparePanel';

export default function App() {
  const { data, error } = useDashboardData();
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [, setHoveredRegion] = useState<string | null>(null);
  const [activeYear, setActiveYear] = useState(2023);
  const [isPlaying, setIsPlaying] = useState(false);
  const [filterCountry, setFilterCountry] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareRegion, setCompareRegion] = useState<string | null>(null);

  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => {
      setActiveYear((y) => {
        if (y >= 2023) {
          setIsPlaying(false);
          return 2010;
        }
        return y + 1;
      });
    }, 800);
    return () => clearInterval(id);
  }, [isPlaying]);

  function handleRegionClick(code: string) {
    if (compareMode) {
      setCompareRegion(code);
    } else {
      setSelectedRegion(code);
    }
  }

  function handleCompareActivate() {
    setCompareMode(true);
    setCompareRegion(null);
  }

  function handleCompareExit() {
    setCompareMode(false);
    setCompareRegion(null);
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-950 text-red-400">
        Error loading data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-950 text-gray-500">
        Loading dashboard data…
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">
      <HeaderBar
        activeYear={activeYear}
        isPlaying={isPlaying}
        onYearChange={setActiveYear}
        onPlayToggle={() => setIsPlaying((p) => !p)}
      />
      <KpiBar data={data} />
      <div className="flex flex-1 min-h-0">
        {/* Left: Map (~50%) */}
        <div className="flex-1 min-w-0">
          <EuropeMap
            data={data}
            activeYear={activeYear}
            selectedRegion={selectedRegion}
            compareMode={compareMode}
            compareRegion={compareRegion}
            onRegionClick={handleRegionClick}
            onRegionHover={setHoveredRegion}
          />
        </div>
        {/* Middle: Time Series (~25%) */}
        <div className="w-72 shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
          <TimeSeriesPanel
            selectedRegion={selectedRegion}
            data={data}
            activeYear={activeYear}
          />
        </div>
        {/* Right: Ranking or Compare (~25%) */}
        <div className="w-72 shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
          {compareMode ? (
            <ComparePanel
              data={data}
              regionA={selectedRegion}
              regionB={compareRegion}
              onExit={handleCompareExit}
            />
          ) : (
            <RankingPanel
              data={data}
              selectedRegion={selectedRegion}
              filterCountry={filterCountry}
              onRegionClick={setSelectedRegion}
              onRegionHover={setHoveredRegion}
              onCountryFilter={setFilterCountry}
              onCompareActivate={handleCompareActivate}
              compareDisabled={selectedRegion === null}
            />
          )}
        </div>
      </div>
    </div>
  );
}
