import pytest
import pandas as pd
import numpy as np


def test_exactly_242_rows(features_df):
    # caveman say: one node per region
    assert len(features_df) == 242


def test_zscore_properties(features_df):
    # caveman say: z-score must be zero-mean, unit-variance (across 242 regions)
    z = features_df["gdp_pc_z"].dropna()
    assert abs(z.mean()) < 0.01, f"gdp_pc_z mean not ~0: {z.mean()}"
    assert abs(z.std() - 1.0) < 0.05, f"gdp_pc_z std not ~1: {z.std()}"


def test_prosperity_gap_direction(features_df):
    # caveman say: Bulgaria poorer than Luxembourg → larger prosperity gap
    bg_gap = features_df[features_df["country_code"] == "BG"]["prosperity_gap"].mean()
    lu_gap = features_df[features_df["country_code"] == "LU"]["prosperity_gap"].mean()
    assert bg_gap > lu_gap, f"BG gap {bg_gap:.3f} not > LU gap {lu_gap:.3f}"



def test_no_inf_values(features_df):
    # caveman say: division artifacts must be caught before GNN sees data
    float_cols = features_df.select_dtypes(include="float").columns
    assert not np.isinf(features_df[float_cols].values).any()


def test_panel_features_nuts2_match(panel_df, features_df):
    # caveman say: no region dropped between long and wide
    panel_codes = set(panel_df["nuts2_code"].unique())
    feature_codes = set(features_df["nuts2_code"])
    assert panel_codes == feature_codes, f"Mismatch: {panel_codes.symmetric_difference(feature_codes)}"


def test_coverage_flag_consistency(panel_df):
    # caveman say: high-coverage rows must have core macro columns non-NaN
    high = panel_df[panel_df["data_coverage"] == "high"]
    for col in ["unemployment_rate", "population"]:
        if col in high.columns:
            nulls = high[col].isna().sum()
            assert nulls == 0, f"{col} has {nulls} NaN in 'high' coverage rows"


def test_migration_plausibility_recent(panel_df):
    # caveman say: migration rates should exist for most regions after 2015
    recent = panel_df[panel_df["year"] >= 2015]
    non_null = recent["net_migration_rate"].notna()
    coverage = non_null.groupby(recent["nuts2_code"]).any().sum()
    assert coverage >= 200, f"Only {coverage} regions have any migration data post-2015"
    valid = recent.loc[non_null, "net_migration_rate"]
    # [-100, 100] per 1000: allows for exceptional years (e.g. 2022 Ukrainian refugee wave)
    assert valid.between(-100, 100).all(), f"Migration rate out of [-100,100]: {valid[~valid.between(-100,100)]}"
