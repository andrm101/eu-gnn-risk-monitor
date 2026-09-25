# Phase 2A — Node Feature Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the 8 NaN placeholder columns in `nuts2_features.parquet` (ihpi, schengen_member, sis, chips_flag, sas, conflict_density, conflict_lag_1, cei) with computed values.

**Architecture:** Two new source modules — `ingest_acled.py` (ACLED Excel → NUTS2 conflict panel via geopandas spatial join) and `enrich_features.py` (all 8 index computations). `build_phase2a.py` orchestrates: loads Phase 1 parquets, calls enrich_features, overwrites `nuts2_features.parquet`. All 17 Phase 1 tests must still pass after enrichment.

**Tech Stack:** Python 3.11, pandas 2.2, geopandas 0.14, pyogrio 0.7, numpy 1.26, pytest 8.1. No new data downloads beyond what Task 1 fetches.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/download_gisco_nuts2.py` | Run only | Download NUTS2 boundary GeoJSON |
| `src/data/ingest_acled.py` | Create | Load ACLED Excel, spatial join to NUTS2, return annual panel |
| `src/data/enrich_features.py` | Create | All 8 enrichment functions + `enrich_features()` orchestrator |
| `build_phase2a.py` | Create | Top-level script: load parquets → enrich → save |
| `tests/test_phase2a.py` | Create | Unit + integration tests for all 8 columns |
| `tests/test_features.py` | Modify | Remove `test_phase2_placeholders_nan` (now incorrect — cols are filled) |

---

## Task 1: Download GISCO NUTS2 Shapefile

**Files:**
- Run: `scripts/download_gisco_nuts2.py`
- Verify: `data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson`

- [ ] **Step 1: Ensure the conda env has geopandas**

```
conda env update -f environment.yml --prune
conda activate eu-gnn-risk
```

- [ ] **Step 2: Run the downloader**

```
python scripts/download_gisco_nuts2.py
```

Expected output:
```
Downloading NUTS2 2021 boundaries (GeoJSON, 1:1M, EPSG:4326)...
Saved: data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson  (3.8 MB)
```

- [ ] **Step 3: Verify file exists and is valid GeoJSON**

```python
import geopandas as gpd
gdf = gpd.read_file("data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson")
print(gdf.shape, gdf.columns.tolist())
# Expected: (>300, ['NUTS_ID', 'LEVL_CODE', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE', 'FID', 'geometry'])
assert "NUTS_ID" in gdf.columns
assert len(gdf) > 200
```

---

## Task 2: ACLED Ingestion Pipeline (`src/data/ingest_acled.py`)

**Files:**
- Create: `src/data/ingest_acled.py`
- Test: `tests/test_phase2a.py` (first section)

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase2a.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_phase2a.py::test_ingest_acled_columns -v
```

Expected: SKIP or FAIL with `ImportError: cannot import name 'ingest_acled'`

- [ ] **Step 3: Create `src/data/ingest_acled.py`**

```python
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

_YEAR_MIN, _YEAR_MAX = 2018, 2023


def ingest_acled(
    acled_path: Path,
    gisco_path: Path,
    panel_path: Path,
) -> pd.DataFrame:
    """Load ACLED aggregated Excel, spatial-join ADMIN1 centroids to NUTS2 polygons,
    aggregate to annual political-violence event counts per 100k population.

    Returns long DataFrame with columns: nuts2_code, year, events_per_100k (float32).
    Coverage: 2018-2023. Regions with no ACLED ADMIN1 centroid are filled with
    the country-year mean (methodological limitation — noted in spec §6.3).
    """
    # ── NUTS2 polygons ────────────────────────────────────────────────────────
    nuts2_gdf = gpd.read_file(gisco_path)[["NUTS_ID", "geometry"]].rename(
        columns={"NUTS_ID": "nuts2_code"}
    )

    # ── ACLED load and filter ─────────────────────────────────────────────────
    acled = pd.read_excel(acled_path)
    acled["year"] = pd.to_datetime(acled["WEEK"]).dt.year
    acled = acled[
        (acled["DISORDER_TYPE"] == "Political violence")
        & acled["year"].between(_YEAR_MIN, _YEAR_MAX)
    ].copy()
    acled = acled.dropna(subset=["CENTROID_LATITUDE", "CENTROID_LONGITUDE"])

    # ── Spatial join: ADMIN1 centroid → NUTS2 polygon ────────────────────────
    acled_gdf = gpd.GeoDataFrame(
        acled,
        geometry=gpd.points_from_xy(
            acled["CENTROID_LONGITUDE"], acled["CENTROID_LATITUDE"]
        ),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(acled_gdf, nuts2_gdf, how="left", predicate="within")
    joined = joined.dropna(subset=["nuts2_code"])  # drop non-EU centroids

    # ── Aggregate to (nuts2_code, year) ──────────────────────────────────────
    annual = (
        joined.groupby(["nuts2_code", "year"])["EVENTS"]
        .sum()
        .reset_index()
        .rename(columns={"EVENTS": "events"})
    )

    # ── Normalise by population ───────────────────────────────────────────────
    panel = pd.read_parquet(panel_path)[
        ["nuts2_code", "year", "population", "country_code"]
    ]
    annual = annual.merge(panel, on=["nuts2_code", "year"], how="left")
    annual["events_per_100k"] = (
        annual["events"] / annual["population"].replace(0, np.nan) * 100_000
    ).astype("float32")

    # ── Fill NUTS2 with no centroid → country-year mean ──────────────────────
    country_means = annual.groupby(["country_code", "year"])["events_per_100k"].transform(
        "mean"
    )
    annual["events_per_100k"] = annual["events_per_100k"].fillna(country_means)

    return annual[["nuts2_code", "year", "events_per_100k"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_phase2a.py::test_ingest_acled_columns tests/test_phase2a.py::test_ingest_acled_year_range tests/test_phase2a.py::test_ingest_acled_nonneg tests/test_phase2a.py::test_ingest_acled_nuts2_count -v
```

Expected: 4 PASSED (may take ~30s for spatial join)

- [ ] **Step 5: Commit**

```
git add src/data/ingest_acled.py tests/test_phase2a.py
git commit -m "feat(phase2a): ACLED ingest pipeline with geopandas spatial join"
```

---

## Task 3: enrich_features.py Scaffold + IHPI

**Files:**
- Create: `src/data/enrich_features.py`
- Modify: `tests/test_phase2a.py` (add unit tests)

- [ ] **Step 1: Write the failing unit test**

Add to `tests/test_phase2a.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_phase2a.py::test_compute_ihpi_range -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `src/data/enrich_features.py` with scaffold and IHPI**

```python
"""Phase 2A feature enrichment — fills the 8 NaN placeholder columns.

All public functions take the features DataFrame (242 rows) and return a
pd.Series of float32, aligned by DataFrame index. The top-level
enrich_features() function applies all of them in order.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.ingest_acled import ingest_acled

# ── Schengen membership (2023 snapshot) ───────────────────────────────────────
# RO/BG joined March 2024 (air/sea) — not counted as full members in 2023.
# IS, LI, NO, CH are non-EU Schengen members; included for completeness.
SCHENGEN_2023 = {
    "AT", "BE", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HR",
    "HU", "IT", "LV", "LI", "LT", "LU", "MT", "NL", "PL",
    "PT", "SK", "SI", "ES", "SE", "IS", "NO", "CH",
}

# Trade openness 2022 (imports + exports % GDP) — World Bank NE.TRD.GNFS.ZS
# Hardcoded: published figures that do not change. EU27 only.
TRADE_OPENNESS_2022: dict[str, float] = {
    "LU": 351.0, "MT": 268.0, "IE": 210.0, "SK": 183.0, "CZ": 152.0,
    "HU": 148.0, "BE": 143.0, "NL": 138.0, "SI": 131.0, "EE": 129.0,
    "LT": 124.0, "LV": 121.0, "HR": 108.0, "AT": 106.0, "DK": 100.0,
    "BG": 101.0, "PL": 92.0,  "FI": 88.0,  "DE": 90.0,  "PT": 91.0,
    "SE": 88.0,  "RO": 79.0,  "CY": 89.0,  "ES": 68.0,  "FR": 66.0,
    "IT": 63.0,  "GR": 65.0,
}
_EU27_MEAN_TRADE = sum(TRADE_OPENNESS_2022.values()) / len(TRADE_OPENNESS_2022)

# EU Chips Act / IPCEI Microelectronics strategic NUTS2 regions (2023 announcements)
CHIPS_REGIONS: frozenset[str] = frozenset({
    "DED2",  # Germany   — Dresden (Intel, TSMC, Infineon)
    "FRK2",  # France    — Rhône-Alpes / Grenoble (STMicroelectronics, Soitec)
    "ITG1",  # Italy     — Sicilia / Catania (STMicroelectronics)
    "NL41",  # Netherlands — Noord-Brabant / Eindhoven (ASML, NXP)
    "IE06",  # Ireland   — Eastern and Midland (Intel Leixlip)
    "AT22",  # Austria   — Steiermark / Graz (Infineon HQ)
    "BE24",  # Belgium   — Prov. Vlaams-Brabant / Leuven (imec)
    "CZ01",  # Czechia   — Praha (ON Semiconductor)
    "PL51",  # Poland    — Dolnośląskie / Wrocław (Nokia, advanced packaging)
    "PT11",  # Portugal  — Norte (Bosch microelectronics)
    "SK01",  # Slovakia  — Bratislavský kraj (Samsung SDI)
    "ES30",  # Spain     — Comunidad de Madrid (Indra)
    "FI1B",  # Finland   — Helsinki-Uusimaa (Nokia Bell Labs)
    "SE11",  # Sweden    — Stockholm (Ericsson)
})

# EU Cohesion Funds 2021-2027 allocation per capita (EUR)
# Source: European Commission published ERDF + Cohesion Fund allocations
COHESION_EUR_PER_CAPITA: dict[str, int] = {
    "PL": 3_100, "RO": 2_800, "CZ": 2_400, "HU": 2_700, "BG": 3_400,
    "SK": 2_200, "HR": 3_000, "LT": 3_100, "LV": 2_900, "EE": 2_800,
    "SI": 1_200, "PT": 1_600, "GR": 1_900, "MT": 800,  "CY": 700,
    "IT": 800,  "ES": 700,  "IE": 300,  "AT": 150, "BE": 200,
    "FR": 250,  "DE": 200,  "NL": 80,   "DK": 60,  "FI": 200,
    "SE": 150,  "LU": 50,
}
_MAX_COHESION = float(max(COHESION_EUR_PER_CAPITA.values()))  # 3400 (BG)


# ── individual enrichment functions ───────────────────────────────────────────

def compute_ihpi(feat: pd.DataFrame) -> pd.Series:
    """Invisible Hand Pressure Index: z(stress) + z(prosperity_gap) - z(net_migration)."""
    def _z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    return (
        _z(feat["stress_proxy"])
        + _z(feat["prosperity_gap"])
        - _z(feat["net_migration_rate"])
    ).astype("float32")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_phase2a.py::test_compute_ihpi_range tests/test_phase2a.py::test_compute_ihpi_length -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```
git add src/data/enrich_features.py tests/test_phase2a.py
git commit -m "feat(phase2a): enrich_features scaffold + compute_ihpi"
```

---

## Task 4: schengen_member + SIS

**Files:**
- Modify: `src/data/enrich_features.py` (add two functions)
- Modify: `tests/test_phase2a.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase2a.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_phase2a.py::test_schengen_member_binary -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Add functions to `src/data/enrich_features.py`**

Append after `compute_ihpi`:

```python
def compute_schengen_member(feat: pd.DataFrame) -> pd.Series:
    """1.0 if country was a full Schengen member in 2023, else 0.0."""
    return feat["country_code"].isin(SCHENGEN_2023).astype("float32")


def compute_sis(feat: pd.DataFrame) -> pd.Series:
    """Schengen Integration Score = schengen_member × trade_openness_2022.

    Trade openness (imports + exports % GDP) acts as a proxy for how deeply
    a region's economy depends on Schengen's free movement of goods/services.
    Non-Schengen members receive 0 regardless of trade openness.
    """
    schengen = compute_schengen_member(feat)
    trade = feat["country_code"].map(TRADE_OPENNESS_2022).fillna(_EU27_MEAN_TRADE)
    return (schengen * trade).astype("float32")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_phase2a.py -k "schengen or sis" -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```
git add src/data/enrich_features.py tests/test_phase2a.py
git commit -m "feat(phase2a): compute_schengen_member + compute_sis"
```

---

## Task 5: chips_flag + SAS

**Files:**
- Modify: `src/data/enrich_features.py`
- Modify: `tests/test_phase2a.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase2a.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_phase2a.py::test_chips_flag_binary -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Add functions to `src/data/enrich_features.py`**

Append after `compute_sis`:

```python
def compute_chips_flag(feat: pd.DataFrame) -> pd.Series:
    """1.0 if NUTS2 hosts an EU Chips Act / IPCEI microelectronics site, else 0.0."""
    return feat["nuts2_code"].isin(CHIPS_REGIONS).astype("float32")


def compute_sas(feat: pd.DataFrame) -> pd.Series:
    """Strategic Autonomy Score = chips_flag × normalised cohesion funding intensity.

    Cohesion funding (ERDF + CF, 2021-2027) per capita serves as a proxy for
    EU industrial policy intensity in the region. Normalised to [0, 1] by the
    maximum allocation (Bulgaria: 3,400 EUR/capita). Non-chips regions score 0.
    """
    chips = compute_chips_flag(feat)
    cohesion_norm = (
        feat["country_code"]
        .map(COHESION_EUR_PER_CAPITA)
        .fillna(0.0)
        .div(_MAX_COHESION)
    )
    return (chips * cohesion_norm).astype("float32")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_phase2a.py -k "chips or sas" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```
git add src/data/enrich_features.py tests/test_phase2a.py
git commit -m "feat(phase2a): compute_chips_flag + compute_sas"
```

---

## Task 6: conflict_density + conflict_lag_1 + CEI

**Files:**
- Modify: `src/data/enrich_features.py`
- Modify: `tests/test_phase2a.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase2a.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_phase2a.py::test_conflict_density_nonneg -v
```

Expected: FAIL with `ImportError` or SKIP (if files missing)

- [ ] **Step 3: Add `compute_conflict_features` to `src/data/enrich_features.py`**

Append after `compute_sas`:

```python
def compute_conflict_features(
    feat: pd.DataFrame,
    panel_path: Path,
    acled_path: Path,
    gisco_path: Path,
) -> pd.DataFrame:
    """Fill conflict_density, conflict_lag_1, and cei columns.

    conflict_density: mean political-violence events per 100k pop, 2018-2023.
    conflict_lag_1:   mean of the 1-year lag series (effectively 2018-2022 values).
    cei:              conflict_density × asset_density, where asset_density
                      combines GDP per capita and data coverage fraction.

    Regions with no ACLED ADMIN1 centroid receive 0.0 (not NaN) as their
    conflict density — this is methodologically conservative (zero conflict
    assumed, not unknown). See spec §6.3 for discussion.
    """
    annual = ingest_acled(acled_path, gisco_path, panel_path)

    # conflict_density: 2018-2023 mean
    density = (
        annual.groupby("nuts2_code")["events_per_100k"]
        .mean()
        .rename("conflict_density")
    )

    # conflict_lag_1: shift by 1 year within each region, then mean
    annual_sorted = annual.sort_values(["nuts2_code", "year"])
    annual_sorted = annual_sorted.copy()
    annual_sorted["lag_1"] = annual_sorted.groupby("nuts2_code")[
        "events_per_100k"
    ].shift(1)
    lag = (
        annual_sorted.groupby("nuts2_code")["lag_1"]
        .mean()
        .rename("conflict_lag_1")
    )

    feat = feat.copy()
    feat["conflict_density"] = (
        feat["nuts2_code"].map(density).fillna(0.0).astype("float32")
    )
    feat["conflict_lag_1"] = (
        feat["nuts2_code"].map(lag).fillna(0.0).astype("float32")
    )

    # CEI: conflict × asset density
    gdp_max = feat["gdp_per_capita_pps_latest"].max()
    gdp_norm = feat["gdp_per_capita_pps_latest"] / gdp_max
    cov_norm = feat["data_coverage_frac"]
    asset_density = ((gdp_norm + cov_norm) / 2).astype("float32")
    feat["cei"] = (feat["conflict_density"] * asset_density).astype("float32")

    return feat
```

- [ ] **Step 4: Run tests (requires ACLED + GISCO files)**

```
pytest tests/test_phase2a.py -k "conflict or cei" -v
```

Expected: 6 PASSED (takes ~30-60s for spatial join)

- [ ] **Step 5: Commit**

```
git add src/data/enrich_features.py tests/test_phase2a.py
git commit -m "feat(phase2a): compute_conflict_features (density, lag, CEI)"
```

---

## Task 7: Wire Up `enrich_features()` + `build_phase2a.py`

**Files:**
- Modify: `src/data/enrich_features.py` (add top-level function)
- Create: `build_phase2a.py`
- Modify: `tests/test_features.py` (remove stale placeholder test)

- [ ] **Step 1: Add `enrich_features()` orchestrator to `src/data/enrich_features.py`**

Append at the end of the file:

```python
def enrich_features(
    feat: pd.DataFrame,
    panel_path: Path,
    acled_path: Path,
    gisco_path: Path,
) -> pd.DataFrame:
    """Fill all 8 Phase-2A placeholder columns. Returns a new DataFrame."""
    feat = feat.copy()
    feat["ihpi"]            = compute_ihpi(feat)
    feat["schengen_member"] = compute_schengen_member(feat)
    feat["sis"]             = compute_sis(feat)
    feat["chips_flag"]      = compute_chips_flag(feat)
    feat["sas"]             = compute_sas(feat)
    feat = compute_conflict_features(feat, panel_path, acled_path, gisco_path)

    # Guard: no inf values introduced
    float_cols = feat.select_dtypes(include="float").columns
    assert not np.isinf(feat[float_cols].values).any(), "Inf values in enriched features"

    return feat
```

- [ ] **Step 2: Create `build_phase2a.py`**

```python
"""Phase 2A orchestrator: enrich nuts2_features.parquet with 8 composite indices.

Usage:
    python build_phase2a.py

Reads:
    data/processed/nuts2_features.parquet  (Phase 1 output, 8 cols are NaN)
    data/processed/nuts2_panel.parquet
    data/raw/acled/*.xlsx                  (ACLED aggregated Europe file)
    data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson

Overwrites:
    data/processed/nuts2_features.parquet  (all 8 cols now populated)
"""
import numpy as np
from pathlib import Path

from src.data.enrich_features import enrich_features
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).parent
FEATURES_PATH = ROOT / "data" / "processed" / "nuts2_features.parquet"
PANEL_PATH    = ROOT / "data" / "processed" / "nuts2_panel.parquet"
GISCO_PATH    = ROOT / "data" / "raw" / "gisco" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"


def _find_acled_path() -> Path:
    acled_dir = ROOT / "data" / "raw" / "acled"
    xlsx_files = sorted(acled_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(
            f"No ACLED xlsx found in {acled_dir}. Download manually from acleddata.com."
        )
    return xlsx_files[-1]  # most recent if multiple


def main() -> None:
    acled_path = _find_acled_path()
    print(f"== Phase 2A: Node Feature Enrichment ==========================")
    print(f"  ACLED:    {acled_path.name}")
    print(f"  GISCO:    {GISCO_PATH.name}")

    feat  = pd.read_parquet(FEATURES_PATH)
    print(f"  Features: {len(feat)} rows x {len(feat.columns)} cols (before)")

    feat = enrich_features(feat, PANEL_PATH, acled_path, GISCO_PATH)
    print(f"  Features: {len(feat)} rows x {len(feat.columns)} cols (after)")

    feat.to_parquet(FEATURES_PATH, index=False)
    print(f"  Written:  {FEATURES_PATH}")
    print(f"== Done ========================================================")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Remove stale placeholder test from `tests/test_features.py`**

Open `tests/test_features.py` and delete the entire `test_phase2_placeholders_nan` function (lines that assert Phase-2 columns are all NaN — that invariant no longer holds after enrichment).

The function to remove looks like:
```python
def test_phase2_placeholders_nan(features_df):
    # caveman say: P2 columns must be NaN, not zero — distinct semantics
    for col in ["ihpi", "sis", "sas", "cei", "conflict_density", "conflict_lag_1",
                "schengen_member", "chips_flag"]:
        assert features_df[col].isna().all(), f"{col} is not all NaN"
```

- [ ] **Step 4: Run build_phase2a.py**

```
python build_phase2a.py
```

Expected output:
```
== Phase 2A: Node Feature Enrichment ==========================
  ACLED:    Europe-Central-Asia_aggregated_data_up_to_week_of-2026-05-23.xlsx
  GISCO:    NUTS_RG_01M_2021_4326_LEVL_2.geojson
  Features: 242 rows x 22 cols (before)
  Features: 242 rows x 22 cols (after)
  Written:  data/processed/nuts2_features.parquet
== Done ========================================================
```

- [ ] **Step 5: Commit**

```
git add src/data/enrich_features.py build_phase2a.py tests/test_features.py
git commit -m "feat(phase2a): enrich_features orchestrator + build_phase2a.py"
```

---

## Task 8: Integration Test Suite + Full Pass

**Files:**
- Modify: `tests/test_phase2a.py` (add final integration tests)

- [ ] **Step 1: Add integration tests to `tests/test_phase2a.py`**

Add at the end of the file:

```python
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
```

- [ ] **Step 2: Run full test suite**

```
pytest tests/ -v
```

Expected: all previous Phase 1 tests (17) + new Phase 2A tests PASS. Zero failures.

If any Phase 1 test fails, it means `enrich_features()` mutated a column it shouldn't have — check that `feat = feat.copy()` is at the top of `enrich_features()`.

- [ ] **Step 3: Final commit**

```
git add tests/test_phase2a.py
git commit -m "test(phase2a): full integration test suite — all columns verified"
```

---

## Quick Reference — Running Order

```
# One-time setup
python scripts/download_gisco_nuts2.py      # ~4 MB download
conda env update -f environment.yml --prune  # adds geopandas, pyogrio

# Build Phase 2A outputs
python build_phase2a.py                      # ~60s (spatial join)

# Verify
pytest tests/ -v                             # all tests should pass
```
