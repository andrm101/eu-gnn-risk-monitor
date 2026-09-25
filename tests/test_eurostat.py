import pytest
import pandas as pd
import numpy as np


def test_migration_column_present(panel_df):
    """Verify tgs00099 net migration joined successfully."""
    assert "net_migration_rate" in panel_df.columns


def test_migration_sign_direction(panel_df):
    """Stuttgart (DE11) is a net-in region — expect positive rates recently."""
    de11 = panel_df[(panel_df["nuts2_code"] == "DE11") & (panel_df["year"] >= 2015)]
    assert len(de11) > 0, "DE11 not found in panel"
    assert de11["net_migration_rate"].mean() > 0, "Stuttgart expected positive net migration"


def test_macro_ranges(panel_df):
    """No impossible values after join."""
    unemp = panel_df["unemployment_rate"].dropna()
    assert unemp.between(0, 40).all(), f"Unemployment out of range: {unemp[~unemp.between(0,40)]}"
    gdp = panel_df["gdp_per_capita_pps"].dropna()
    assert gdp.gt(0).all()


def test_coverage_flag_valid(panel_df):
    """Flag must be exactly one of three values."""
    assert panel_df["data_coverage"].isin(["high", "med", "low"]).all()


def test_population_accounting_consistency(panel_df):
    """EU27 population 2020 must be ~447M persons (raw counts, not thousands)."""
    eu_2020 = panel_df[panel_df["year"] == 2020]["population"].sum()
    assert 400_000_000 < eu_2020 < 500_000_000, f"EU population 2020: {eu_2020:,.0f}"
