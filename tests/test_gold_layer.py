import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]

from src.data.load_gold_layer import load_gold_layer

GOLD_PATH = (
    ROOT.parent / "EU-Innovation-Panel" / "data" / "gold" / "region_profiles_gold.parquet"
)


@pytest.fixture(scope="module")
def gold_df():
    if not GOLD_PATH.exists():
        pytest.skip(f"Gold layer not found at {GOLD_PATH}")
    return load_gold_layer(GOLD_PATH)


def test_242_nuts2_present(gold_df):
    # caveman say: all EU27 regions must load
    assert len(gold_df["nuts2_code"].unique()) == 242


def test_year_range_covered(gold_df):
    # caveman say: Gold layer is 2022 vintage — year column must be 2022
    assert "year" in gold_df.columns
    assert (gold_df["year"] == 2022).all()


def test_no_duplicate_keys(gold_df):
    # caveman say: one row per region (static cross-section)
    assert not gold_df.duplicated(["nuts2_code"]).any()


def test_gdp_pc_range(gold_df):
    # caveman say: GDP per capita PPS in EUR — confirmed range 10300-127000 from Gold layer inspection
    gdp = gold_df["gdp_per_capita_pps"].dropna()
    assert gdp.between(8_000, 140_000).all(), f"Out-of-range: {gdp[~gdp.between(8_000, 140_000)]}"
