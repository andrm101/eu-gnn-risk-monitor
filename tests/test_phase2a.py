"""Phase 2A integration and unit tests.

Integration tests (test_ingest_acled_*, test_no_nan_placeholders_*, etc.)
require build_phase2a.py to have been run first. They skip automatically if
the parquet is missing.

Unit tests (test_compute_*) call functions directly with synthetic data.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[1]
ACLED_PATH = ROOT / "data" / "raw" / "acled"
GISCO_PATH = ROOT / "data" / "raw" / "gisco" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
PANEL_PATH = ROOT / "data" / "processed" / "nuts2_panel.parquet"


# ── ingest_acled integration tests ────────────────────────────────────────────

@pytest.fixture(scope="session")
def acled_path():
    paths = list(ACLED_PATH.glob("*.xlsx"))
    if not paths:
        pytest.skip("No ACLED xlsx found in data/raw/acled/ — download manually")
    return paths[0]


@pytest.fixture(scope="session")
def gisco_path():
    if not GISCO_PATH.exists():
        pytest.skip("GISCO GeoJSON not found — run scripts/download_gisco_nuts2.py")
    return GISCO_PATH


@pytest.fixture(scope="session")
def panel_path():
    if not PANEL_PATH.exists():
        pytest.skip("nuts2_panel.parquet not found — run build_nuts2_panel.py first")
    return PANEL_PATH


@pytest.fixture(scope="session")
def annual_conflict(acled_path, gisco_path, panel_path):
    from src.data.ingest_acled import ingest_acled
    return ingest_acled(acled_path, gisco_path, panel_path)


def test_ingest_acled_columns(annual_conflict):
    assert set(annual_conflict.columns) >= {"nuts2_code", "year", "events_per_100k"}


def test_ingest_acled_year_range(annual_conflict):
    assert annual_conflict["year"].between(2018, 2023).all()


def test_ingest_acled_nonneg(annual_conflict):
    assert (annual_conflict["events_per_100k"] >= 0).all()


def test_ingest_acled_nuts2_count(annual_conflict):
    # Must have data for at least 200 NUTS2 regions (some may have 0 events)
    assert annual_conflict["nuts2_code"].nunique() >= 200


# ── IHPI unit tests ───────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_feat():
    """42-row synthetic feature DataFrame matching nuts2_features.parquet schema."""
    np.random.seed(42)
    n = 42
    return pd.DataFrame({
        "nuts2_code": [f"XX{i:02d}" for i in range(n)],
        "country_code": ["XX"] * n,
        "archetype_id": [0] * n,
        "gdp_per_capita_pps_latest": np.random.uniform(10_000, 80_000, n).astype("float32"),
        "unemployment_latest": np.random.uniform(2, 25, n).astype("float32"),
        "net_migration_rate": np.random.uniform(-20, 30, n).astype("float32"),
        "gdp_pc_z": np.random.randn(n).astype("float32"),
        "unemployment_z": np.random.randn(n).astype("float32"),
        "gdp_growth_3yr": np.random.uniform(-0.05, 0.1, n).astype("float32"),
        "unemp_change_3yr": np.random.uniform(-5, 5, n).astype("float32"),
        "pop_growth_3yr": np.random.uniform(-0.02, 0.05, n).astype("float32"),
        "prosperity_gap": np.random.uniform(-0.5, 0.8, n).astype("float32"),
        "stress_proxy": np.random.randn(n).astype("float32"),
        "data_coverage_frac": np.random.uniform(0.5, 1.0, n).astype("float32"),
    })


def test_compute_ihpi_range(synthetic_feat):
    from src.data.enrich_features import compute_ihpi
    ihpi = compute_ihpi(synthetic_feat)
    assert ihpi.dtype == np.float32
    assert ihpi.between(-8, 8).all(), f"IHPI out of expected range: {ihpi.describe()}"


def test_compute_ihpi_length(synthetic_feat):
    from src.data.enrich_features import compute_ihpi
    ihpi = compute_ihpi(synthetic_feat)
    assert len(ihpi) == len(synthetic_feat)


# ── schengen_member + SIS unit tests ──────────────────────────────────────────

@pytest.fixture
def eu_feat(synthetic_feat):
    """Synthetic feat with realistic EU country codes."""
    feat = synthetic_feat.copy()
    countries = ["DE", "FR", "PL", "RO", "BG", "NL", "IT", "ES", "PT", "GR",
                 "CZ", "HU", "SK", "AT", "BE", "SE", "DK", "FI", "IE", "HR",
                 "LT", "LV", "EE", "SI", "LU", "MT", "CY"]
    feat["country_code"] = (countries * 2)[:len(feat)]
    return feat


def test_schengen_member_binary(eu_feat):
    from src.data.enrich_features import compute_schengen_member
    s = compute_schengen_member(eu_feat)
    assert s.dtype == np.float32
    assert set(s.unique()).issubset({0.0, 1.0})


def test_schengen_member_ro_is_zero(eu_feat):
    from src.data.enrich_features import compute_schengen_member
    ro_rows = eu_feat[eu_feat["country_code"] == "RO"].index
    s = compute_schengen_member(eu_feat)
    assert (s.loc[ro_rows] == 0.0).all()


def test_schengen_member_de_is_one(eu_feat):
    from src.data.enrich_features import compute_schengen_member
    de_rows = eu_feat[eu_feat["country_code"] == "DE"].index
    s = compute_schengen_member(eu_feat)
    assert (s.loc[de_rows] == 1.0).all()


def test_sis_zero_for_nonschengen(eu_feat):
    from src.data.enrich_features import compute_sis
    sis = compute_sis(eu_feat)
    ro_rows = eu_feat[eu_feat["country_code"] == "RO"].index
    assert (sis.loc[ro_rows] == 0.0).all()


def test_sis_positive_for_schengen(eu_feat):
    from src.data.enrich_features import compute_sis
    sis = compute_sis(eu_feat)
    de_rows = eu_feat[eu_feat["country_code"] == "DE"].index
    assert (sis.loc[de_rows] > 0.0).all()


# ── chips_flag + SAS unit tests ───────────────────────────────────────────────

@pytest.fixture
def chips_feat(synthetic_feat):
    """Synthetic feat with a known chips region included."""
    feat = synthetic_feat.copy()
    # Plant one known chips NUTS2 code in row 0
    feat.loc[0, "nuts2_code"] = "DED2"   # Dresden
    feat.loc[0, "country_code"] = "DE"
    feat.loc[1, "nuts2_code"] = "FRK2"   # Grenoble
    feat.loc[1, "country_code"] = "FR"
    return feat


def test_chips_flag_binary(chips_feat):
    from src.data.enrich_features import compute_chips_flag
    f = compute_chips_flag(chips_feat)
    assert f.dtype == np.float32
    assert set(f.unique()).issubset({0.0, 1.0})


def test_chips_flag_known_regions(chips_feat):
    from src.data.enrich_features import compute_chips_flag
    f = compute_chips_flag(chips_feat)
    assert f.iloc[0] == 1.0, "DED2 (Dresden) should be flagged"
    assert f.iloc[1] == 1.0, "FRK2 (Grenoble) should be flagged"
    assert f.iloc[2] == 0.0, "XX02 (synthetic) should not be flagged"


def test_sas_zero_when_no_chips(chips_feat):
    from src.data.enrich_features import compute_sas
    sas = compute_sas(chips_feat)
    # All rows except 0 and 1 have synthetic nuts2_code → chips_flag=0 → sas=0
    assert (sas.iloc[2:] == 0.0).all()


def test_sas_range(chips_feat):
    from src.data.enrich_features import compute_sas
    sas = compute_sas(chips_feat)
    assert (sas >= 0.0).all()
    assert (sas <= 1.0).all()


# ── conflict + CEI integration tests ─────────────────────────────────────────
# These require the real ACLED + GISCO files.

@pytest.fixture(scope="session")
def conflict_feat(acled_path, gisco_path, panel_path, features_df):
    """features_df with conflict columns filled (calls compute_conflict_features)."""
    from src.data.enrich_features import compute_conflict_features
    feat = features_df.copy()
    return compute_conflict_features(feat, panel_path, acled_path, gisco_path)


def test_conflict_density_nonneg(conflict_feat):
    assert (conflict_feat["conflict_density"] >= 0).all()


def test_conflict_density_no_nan(conflict_feat):
    assert conflict_feat["conflict_density"].notna().all()


def test_conflict_lag_no_nan(conflict_feat):
    assert conflict_feat["conflict_lag_1"].notna().all()


def test_conflict_density_ukraine_border_elevated(conflict_feat):
    """Poland border regions should be above median after 2022 Ukraine war."""
    median = conflict_feat["conflict_density"].median()
    pl_mean = conflict_feat[conflict_feat["country_code"] == "PL"]["conflict_density"].mean()
    assert pl_mean > median, f"PL mean {pl_mean:.4f} not > median {median:.4f}"


def test_cei_nonneg(conflict_feat):
    assert (conflict_feat["cei"] >= 0).all()


def test_cei_no_nan(conflict_feat):
    assert conflict_feat["cei"].notna().all()


# ── Full integration tests (require build_phase2a.py to have run) ─────────────

def test_no_nan_placeholders(features_df):
    """All 8 Phase-2A columns must be fully populated after enrichment."""
    for col in ["ihpi", "sis", "sas", "cei", "conflict_density",
                "conflict_lag_1", "schengen_member", "chips_flag"]:
        assert features_df[col].notna().all(), f"{col} has NaN values after enrichment"


def test_ihpi_range(features_df):
    ihpi = features_df["ihpi"].dropna()
    assert ihpi.between(-6, 6).all(), f"IHPI out of range: {ihpi.describe()}"


def test_schengen_binary_real(features_df):
    vals = features_df["schengen_member"].unique()
    assert set(vals).issubset({0.0, 1.0})


def test_chips_flag_count_real(features_df):
    # At least 10 of the 14 annotated CHIPS_REGIONS exist in EU27 NUTS2
    count = int(features_df["chips_flag"].sum())
    assert 10 <= count <= 14, f"chips_flag count unexpected: {count}"


def test_sis_zero_for_ro_bg(features_df):
    """Romania and Bulgaria were not Schengen members in 2023 → SIS must be 0."""
    for cc in ["RO", "BG"]:
        sis = features_df[features_df["country_code"] == cc]["sis"]
        assert (sis == 0.0).all(), f"{cc} SIS should be 0: {sis.values}"


def test_sas_only_nonzero_in_chips_regions(features_df):
    non_chips = features_df[features_df["chips_flag"] == 0.0]
    assert (non_chips["sas"] == 0.0).all()


def test_all_float32(features_df):
    for col in ["ihpi", "sis", "sas", "cei", "conflict_density",
                "conflict_lag_1", "schengen_member", "chips_flag"]:
        assert features_df[col].dtype == np.float32, f"{col} is not float32"


def test_242_rows_still(features_df):
    assert len(features_df) == 242
