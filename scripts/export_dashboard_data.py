"""Export pipeline outputs to static JSON for the React dashboard."""
from __future__ import annotations
from pathlib import Path
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from scipy.stats import t as t_dist


def _forecast_region(
    years: list,
    scores: list,
    horizon: list | None = None,
) -> list:
    """Fit OLS linear trend on year→score and project forward with 95% prediction intervals.

    Returns list of {year, point, lower, upper} dicts, one per horizon year.
    Lower is clamped to 0.0 (anomaly scores cannot be negative).
    """
    if horizon is None:
        horizon = [2024, 2025, 2026]
    yr = np.array(years, dtype=float)
    y = np.array(scores, dtype=float)
    n = len(yr)
    model = LinearRegression().fit(yr.reshape(-1, 1), y)
    residuals = y - model.predict(yr.reshape(-1, 1))
    se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_mean = float(yr.mean())
    Sxx = float(np.sum((yr - x_mean) ** 2))
    t_crit = float(t_dist.ppf(0.975, df=n - 2))
    pts = []
    for fx in horizon:
        point = float(model.predict([[float(fx)]])[0])
        margin = float(t_crit * se * np.sqrt(1.0 + 1.0 / n + (fx - x_mean) ** 2 / Sxx))
        pts.append({
            'year': fx,
            'point': round(point, 4),
            'lower': round(max(0.0, point - margin), 4),
            'upper': round(max(0.0, point + margin), 4),
        })
    return pts

ROOT = Path(__file__).parents[1]
OUT = ROOT / 'dashboard' / 'public' / 'data'

EU27 = {
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU',
    'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
}
GISCO_URL = (
    'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/'
    'NUTS_RG_20M_2021_4326_LEVL_2.geojson'
)


def export_risk_scores(root: Path, out: Path) -> None:
    df = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_risk_scores.parquet')
    records = df.assign(
        anomaly_score=df['anomaly_score'].round(6).astype(float),
        peak_score=df['peak_score'].round(6).astype(float),
        peak_rank=df['peak_rank'].astype(int),
        year=df['year'].astype(int),
    ).to_dict(orient='records')
    (out / 'risk_scores.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    print(f'  risk_scores.json   {len(records)} records')


def export_neighbours(root: Path, out: Path) -> None:
    g = torch.load(root / 'data' / 'processed' / 'nuts2_graph.pt', weights_only=False)
    ni = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_node_index.parquet')
    idx2code = ni.set_index('node_idx')['nuts2_code'].to_dict()

    # Initialise every region with an empty list so isolated nodes are included
    adj: dict[str, list[str]] = {code: [] for code in idx2code.values()}

    ei = g[('region', 'spatial', 'region')].edge_index.numpy()
    for s, d in zip(ei[0], ei[1]):
        adj[idx2code[int(s)]].append(idx2code[int(d)])

    (out / 'neighbours.json').write_text(json.dumps(adj, ensure_ascii=False), encoding='utf-8')
    print(f'  neighbours.json    {len(adj)} regions')


def export_geojson(out: Path) -> None:
    print('  nuts2.geojson      downloading from GISCO...')
    with urllib.request.urlopen(GISCO_URL, timeout=60) as r:
        full = json.load(r)
    features = [f for f in full['features'] if f['properties']['CNTR_CODE'] in EU27]
    geojson = {'type': 'FeatureCollection', 'features': features}
    # Use ensure_ascii=True so the file is pure ASCII and portable across locales
    (out / 'nuts2.geojson').write_text(json.dumps(geojson, ensure_ascii=True), encoding='utf-8')
    print(f'  nuts2.geojson      {len(features)} features')


def export_forecast_scores(root: Path, out: Path) -> None:
    """Compute per-region OLS forecasts (2024–2026) and write forecast_scores.json."""
    df = pd.read_parquet(root / 'data' / 'processed' / 'nuts2_risk_scores.parquet')
    result: dict = {}
    for code, grp in df.groupby('nuts2_code'):
        grp = grp.sort_values('year')
        years = grp['year'].tolist()
        scores = grp['anomaly_score'].tolist()
        if len(years) < 3:
            continue
        result[str(code)] = _forecast_region(years, scores)
    out_path = out / 'forecast_scores.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, separators=(',', ':'))
    print(f'  forecast_scores.json {len(result)} regions')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    print('Exporting dashboard data...')
    export_risk_scores(ROOT, OUT)
    export_neighbours(ROOT, OUT)
    export_geojson(OUT)
    export_forecast_scores(ROOT, OUT)
    print('Done.')
