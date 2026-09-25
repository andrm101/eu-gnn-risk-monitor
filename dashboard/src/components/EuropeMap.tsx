import { useRef, useCallback } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Feature, FeatureCollection } from 'geojson';
import type { DashboardData } from '../types';
import { anomalyColor } from '../utils/color';
import { formatScore, formatRank } from '../utils/format';

const DARK_TILE = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_ATTR = '© <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors, © <a href="https://carto.com/">CARTO</a>';
const EU_BOUNDS: L.LatLngBoundsExpression = [[34, -25], [72, 45]];

interface NutsFeature extends Feature {
  properties: { NUTS_ID: string; NAME_LATN: string; CNTR_CODE: string };
}

interface Props {
  data: DashboardData;
  activeYear: number;
  selectedRegion: string | null;
  compareMode: boolean;
  compareRegion: string | null;
  onRegionClick: (code: string) => void;
  onRegionHover: (code: string | null) => void;
}

export default function EuropeMap({
  data, activeYear, selectedRegion, compareMode, compareRegion, onRegionClick, onRegionHover,
}: Props) {
  const geoJsonRef = useRef<L.GeoJSON | null>(null);
  const yearScores = data.byYear[activeYear] ?? {};

  function getStyle(feature: Feature | undefined): L.PathOptions {
    const code = (feature as NutsFeature)?.properties?.NUTS_ID ?? '';
    const score = yearScores[code] ?? 0;
    const isSelected = code === selectedRegion;
    const isCompare  = code === compareRegion;
    return {
      fillColor: anomalyColor(score, data.maxScore),
      fillOpacity: 0.82,
      color:  isSelected ? '#ffffff' : isCompare ? '#3b82f6' : '#1f2937',
      weight: isSelected || isCompare ? 2.5 : 0.4,
    };
  }

  const onEachFeature = useCallback(
    (feature: Feature, layer: L.Layer) => {
      const f = feature as NutsFeature;
      const code = f.properties.NUTS_ID;
      const name = f.properties.NAME_LATN;
      const ts = data.byCode[code];
      const score = yearScores[code];
      const scoreStr = score != null ? formatScore(score) : 'N/A';
      const rankStr = ts ? formatRank(ts.peak_rank) : '—';

      (layer as L.Path).bindTooltip(
        `<b>${name}</b> (${code})<br/>Score ${activeYear}: ${scoreStr}<br/>Peak rank: ${rankStr}`,
        { sticky: true },
      );

      layer.on({
        click: () => onRegionClick(code),
        mouseover: (e: L.LeafletMouseEvent) => {
          const target = e.target as L.Path;
          target.setStyle({ color: '#fbbf24', weight: 2 });
          target.bringToFront();
          onRegionHover(code);
        },
        mouseout: (e: L.LeafletMouseEvent) => {
          geoJsonRef.current?.resetStyle(e.target as L.Path);
          onRegionHover(null);
        },
      });
    },
    // GeoJSON remounts on key change (year/selection/compareMode), so onEachFeature
    // is always re-bound fresh — onRegionClick in deps covers the compare routing.
    [data.byCode, yearScores, activeYear, onRegionClick, onRegionHover],
  );

  return (
    <div className="relative h-full w-full">
      {compareMode && !compareRegion && (
        <div
          className="absolute top-2 left-1/2 -translate-x-1/2 z-[1000]
                     bg-blue-900/90 text-blue-200 text-xs px-3 py-1 rounded-full
                     pointer-events-none select-none"
        >
          Click a region to set Region B
        </div>
      )}
      <MapContainer
        bounds={EU_BOUNDS}
        minZoom={3}
        maxZoom={8}
        zoomControl
        style={{ height: '100%', width: '100%', background: '#030712' }}
      >
        <TileLayer url={DARK_TILE} attribution={TILE_ATTR} />
        <GeoJSON
          key={`${activeYear}-${selectedRegion ?? 'none'}-${compareRegion ?? 'none'}-${compareMode}`}
          data={data.geoJson as FeatureCollection}
          style={getStyle}
          onEachFeature={onEachFeature}
          ref={(layer) => { geoJsonRef.current = layer as unknown as L.GeoJSON; }}
        />
      </MapContainer>
    </div>
  );
}
