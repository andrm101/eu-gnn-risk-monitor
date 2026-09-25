"""Integration tests for export_dashboard_data.py.
Skipped if pipeline artefacts are not present.
"""
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / 'dashboard' / 'public' / 'data'
SCORES_PATH = ROOT / 'data' / 'processed' / 'nuts2_risk_scores.parquet'


@pytest.fixture(scope='session', autouse=False)
def run_export():
    if not SCORES_PATH.exists():
        pytest.skip('Pipeline artefacts missing — run train_phase3.py first')
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'export_dashboard_data.py')],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return OUT


def test_risk_scores_json(run_export):
    path = run_export / 'risk_scores.json'
    assert path.exists()
    records = json.loads(path.read_text())
    assert len(records) == 3388  # 242 regions × 14 years
    first = records[0]
    assert {'nuts2_code', 'country_code', 'year', 'anomaly_score', 'peak_score', 'peak_rank'} <= first.keys()
    assert isinstance(first['year'], int)
    assert isinstance(first['anomaly_score'], float)


def test_neighbours_json(run_export):
    path = run_export / 'neighbours.json'
    assert path.exists()
    neighbours = json.loads(path.read_text())
    assert len(neighbours) == 242
    # Every value is a list of strings
    for code, nbrs in neighbours.items():
        assert isinstance(code, str)
        assert isinstance(nbrs, list)
        assert all(isinstance(n, str) for n in nbrs)


def test_nuts2_geojson(run_export):
    path = run_export / 'nuts2.geojson'
    assert path.exists()
    gj = json.loads(path.read_text())
    assert gj['type'] == 'FeatureCollection'
    codes = {f['properties']['CNTR_CODE'] for f in gj['features']}
    # All 27 EU member states present
    EU27 = {'AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','EL','HU',
             'IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE'}
    assert EU27 <= codes, f"Missing countries: {EU27 - codes}"
