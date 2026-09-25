import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="session")
def panel_df():
    path = ROOT / "data" / "processed" / "nuts2_panel.parquet"
    if not path.exists():
        pytest.skip("nuts2_panel.parquet not found — run build_nuts2_panel.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def features_df():
    path = ROOT / "data" / "processed" / "nuts2_features.parquet"
    if not path.exists():
        pytest.skip("nuts2_features.parquet not found — run build_nuts2_panel.py first")
    return pd.read_parquet(path)
