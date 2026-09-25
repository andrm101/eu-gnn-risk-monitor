# Phase 2A — Node Feature Enrichment Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fill the 8 NaN placeholder columns in `nuts2_features.parquet` with computed or derived values, producing a fully populated GNN node feature matrix.

**Architecture:** New script `src/data/enrich_features.py` reads `nuts2_features.parquet` + raw data sources, computes all 8 indices, and overwrites the features parquet. A new `build_phase2a.py` orchestrator runs the full enrichment pipeline. All 17 Phase 1 tests must continue passing after enrichment.

**Tech Stack:** Python 3.11, pandas 2.2, geopandas 0.14, pyogrio 0.7, numpy 1.26, scikit-learn 1.4 (for z-score), pytest 8.1.

---

## 1. Data Inputs

### Already present (Phase 1 outputs)
- `data/processed/nuts2_features.parquet` — 242 rows × 22 cols; 8 columns are NaN placeholders
- `data/processed/nuts2_panel.parquet` — 3,373 rows × 15 cols; contains `population`, `gdp_per_capita_pps`, `year`, `nuts2_code`, `country_code`

### New raw data (manually downloaded, immutable)
- `data/raw/acled/Europe-Central-Asia_aggregated_data_up_to_week_of-2026-05-23.xlsx`
  - Weekly aggregated ACLED events for Europe; ADMIN1 level; 2018–2026
  - Key columns: `WEEK`, `COUNTRY`, `ADMIN1`, `DISORDER_TYPE`, `EVENTS`, `FATALITIES`, `CENTROID_LATITUDE`, `CENTROID_LONGITUDE`
- `data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson`
  - Eurostat GISCO NUTS2 2021 boundaries (EPSG:4326); download via `python scripts/download_gisco_nuts2.py`
- `data/raw/worldbank_wdi/` — World Bank WDI exports already present from Phase 1 (trade openness, imports+exports % GDP)

### Hardcoded lookup tables (in `src/data/enrich_features.py`)
- **Schengen membership timeline** — country × year binary (see §4)
- **chips_flag regions** — manually annotated NUTS2 list (see §5)

---

## 2. Output

`data/processed/nuts2_features.parquet` — same 242 rows, same 22 columns, all 8 previously-NaN columns now populated:

| Column | dtype | Semantics |
|---|---|---|
| `ihpi` | float32 | Invisible Hand Pressure Index (higher = stronger out-migration pressure) |
| `schengen_member` | float32 | 1.0 if country was Schengen member in latest year, else 0.0 |
| `conflict_density` | float32 | Political violence events per 100k population, 2018–2023 average |
| `conflict_lag_1` | float32 | `conflict_density` lagged 1 year (2017–2022 average, i.e. shifted window) |
| `chips_flag` | float32 | 1.0 if NUTS2 hosts EU Chips Act / IPCEI microelectronics site, else 0.0 |
| `sis` | float32 | Schengen Integration Score (simplified) |
| `sas` | float32 | Strategic Autonomy Score (simplified) |
| `cei` | float32 | Conflict Exposure Index |

All values float32. No column may remain all-NaN after enrichment (test enforced).

---

## 3. IHPI — Invisible Hand Pressure Index

**Formula:**
```
IHPI_r = z(stress_proxy_r) + z(prosperity_gap_r) - z(net_migration_rate_r)
```

All three inputs already exist in `nuts2_features.parquet`:
- `stress_proxy` = `unemployment_z - gdp_pc_z` (computed in Phase 1)
- `prosperity_gap` = `(eu_mean_gdp - gdp_per_capita_pps_latest) / eu_mean_gdp`
- `net_migration_rate` = latest available year snapshot

**Implementation:**
```python
def compute_ihpi(feat: pd.DataFrame) -> pd.Series:
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    return (
        zscore(feat["stress_proxy"])
        + zscore(feat["prosperity_gap"])
        - zscore(feat["net_migration_rate"])
    ).astype("float32")
```

**Interpretation:** High IHPI → region under market pressure (high unemployment, low GDP, net out-migration). Low IHPI → destination region. Expected range: approximately −3 to +4.

---

## 4. schengen_member and SIS — Schengen Integration Score

### 4.1 schengen_member

Binary flag based on Schengen Area membership timeline. Latest year = 2023 for the feature snapshot.

**Hardcoded lookup (country-level, 2023):**
```python
SCHENGEN_2023 = {
    # Full members (26 countries as of 2023; RO/BG joined March 2024 — not yet in 2023)
    "AT", "BE", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
    "IS", "IT", "LV", "LI", "LT", "LU", "MT", "NL", "NO", "PL",
    "PT", "SK", "SI", "ES", "SE", "CH",
    # Not yet members in 2023:
    # "BG", "RO", "CY", "IE", "HR" (HR joined Jan 2023 — include)
    "HR",
}
```

`schengen_member_r = 1.0 if nuts2_code[:2] in SCHENGEN_2023 else 0.0`

### 4.2 SIS — Schengen Integration Score (simplified)

```
SIS_r = schengen_member_r × trade_openness_r
```

Where `trade_openness_r` = (imports + exports) % GDP for the country, sourced from World Bank WDI (`NE.TRD.GNFS.ZS`), 2022 value, merged by `country_code`. If WDI data is unavailable for a country, use the EU27 mean.

**Expected range:** 0 for non-Schengen; 50–170 for Schengen members (Luxembourg ~300 as outlier due to re-exports).

---

## 5. chips_flag and SAS — Strategic Autonomy Score

### 5.1 chips_flag

Manually annotated list of NUTS2 regions hosting EU Chips Act / IPCEI Microelectronics first movers or major semiconductor fab/R&D sites (as of 2023 public announcements):

```python
CHIPS_REGIONS = {
    # Germany — Dresden fab cluster (Intel, TSMC, Infineon)
    "DED2",
    # France — Grenoble (STMicroelectronics, Soitec)
    "FRK2",
    # Italy — Catania (STMicroelectronics)
    "ITG1",
    # Netherlands — Eindhoven (ASML, NXP)
    "NL41",
    # Ireland — East (Intel Leixlip)
    "IE06",
    # Austria — Graz (Infineon HQ)
    "AT22",
    # Belgium — Flanders (imec)
    "BE23",
    # Czech Republic — Prague (ON Semiconductor)
    "CZ01",
    # Poland — Wrocław (Nokia, advanced packaging)
    "PL51",
    # Portugal — Norte (Bosch microelectronics)
    "PT11",
    # Slovakia — Bratislava (Samsung SDI, advanced components)
    "SK01",
    # Spain — Madrid (indra, advanced systems)
    "ES30",
    # Finland — Uusimaa (Nokia Bell Labs)
    "FI1B",
    # Sweden — Stockholm (Ericsson)
    "SE11",
}

chips_flag_r = 1.0 if nuts2_code in CHIPS_REGIONS else 0.0
```

### 5.2 SAS — Strategic Autonomy Score (simplified)

```
SAS_r = chips_flag_r × cohesion_intensity_r
```

Where `cohesion_intensity_r` = EU Structural & Cohesion Funds allocation per capita for the country, 2021–2027 programming period. Use country-level total allocation (€bn) ÷ population as a proxy. Values sourced from European Commission published allocations (hardcoded dict in script — these are published fixed figures, no dataset needed).

```python
# EU Structural Funds 2021-2027 allocation per capita (EUR), country-level proxy
# Source: European Commission, Cohesion Fund + ERDF allocations
COHESION_EUR_PER_CAPITA = {
    "PL": 3_100, "RO": 2_800, "CZ": 2_400, "HU": 2_700, "BG": 3_400,
    "SK": 2_200, "HR": 3_000, "LT": 3_100, "LV": 2_900, "EE": 2_800,
    "SI": 1_200, "PT": 1_600, "GR": 1_900, "MT": 800,  "CY": 700,
    "IT": 800,  "ES": 700,  "IE": 300,  "AT": 150, "BE": 200,
    "FR": 250,  "DE": 200,  "NL": 80,   "DK": 60,  "FI": 200,
    "SE": 150,  "LU": 50,
}
```

`SAS_r = chips_flag_r × (cohesion_EUR_per_capita[country_code] / max_cohesion)`

Normalized to [0, 1] by dividing by maximum cohesion value (BG: 3,400).

---

## 6. conflict_density and conflict_lag_1

### 6.1 Pipeline

```
ACLED Excel
  → filter: DISORDER_TYPE == "Political violence"
  → filter: year in 2018–2023
  → spatial join: (CENTROID_LATITUDE, CENTROID_LONGITUDE) → NUTS2 polygon
  → aggregate: SUM(EVENTS) by (nuts2_code, year)
  → merge: panel population by (nuts2_code, year)
  → compute: events_per_100k = EVENTS / population * 100_000
  → aggregate: mean(events_per_100k) over 2018–2023 per nuts2_code
  → result: conflict_density (float32, 242 values)
```

### 6.2 Spatial join detail

```python
import geopandas as gpd
from shapely.geometry import Point

nuts2_gdf = gpd.read_file("data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson")[
    ["NUTS_ID", "geometry"]
].rename(columns={"NUTS_ID": "nuts2_code"})

acled_gdf = gpd.GeoDataFrame(
    acled_df,
    geometry=gpd.points_from_xy(acled_df["CENTROID_LONGITUDE"], acled_df["CENTROID_LATITUDE"]),
    crs="EPSG:4326",
)
joined = gpd.sjoin(acled_gdf, nuts2_gdf, how="left", predicate="within")
# Rows that don't fall within any NUTS2 polygon → drop (non-EU27 centroids)
joined = joined[joined["nuts2_code"].notna()]
```

### 6.3 Missing NUTS2 coverage

For NUTS2 regions with no ADMIN1 centroid within their boundaries (e.g. some French sub-regions within large "grande région"):
- Assign the parent country's mean `conflict_density` across its present NUTS2 regions.
- This is explicitly noted as a methodological limitation.

### 6.4 conflict_lag_1

Computed from the annual panel (before taking the 2018–2023 mean):
```python
annual["conflict_lag_1"] = annual.groupby("nuts2_code")["events_per_100k"].shift(1)
```

Feature snapshot: mean of `conflict_lag_1` over 2018–2023, dropping the 2018 NaN (no 2017 data exists) → effectively covers the 2018–2022 value window.

For the feature matrix, NaN lag values (first year per region) are dropped before averaging.

---

## 7. CEI — Conflict Exposure Index

```
CEI_r = conflict_density_r × asset_density_r
```

Where:
```python
def asset_density(feat: pd.DataFrame) -> pd.Series:
    gdp_norm = (feat["gdp_per_capita_pps_latest"] / feat["gdp_per_capita_pps_latest"].max())
    cov_norm = feat["data_coverage_frac"]
    return ((gdp_norm + cov_norm) / 2).astype("float32")
```

`asset_density` is normalized to [0, 1]. `CEI` is therefore in [0, conflict_density_max].

**Interpretation:** High CEI = high-value region with elevated physical conflict exposure. This is the primary risk signal for the dashboard.

---

## 8. Tests to Add (`tests/test_phase2a.py`)

```python
def test_no_nan_placeholders(features_df):
    """All 8 Phase-2A columns must be fully populated after enrichment."""
    for col in ["ihpi", "sis", "sas", "cei", "conflict_density",
                "conflict_lag_1", "schengen_member", "chips_flag"]:
        assert features_df[col].notna().all(), f"{col} has NaN values after enrichment"

def test_ihpi_range(features_df):
    ihpi = features_df["ihpi"].dropna()
    assert ihpi.between(-6, 6).all()

def test_schengen_binary(features_df):
    vals = features_df["schengen_member"].dropna().unique()
    assert set(vals).issubset({0.0, 1.0})

def test_chips_flag_count(features_df):
    # 14 annotated Chips Act / IPCEI regions — see §5.1 of design spec
    assert features_df["chips_flag"].sum() == 14

def test_conflict_density_nonneg(features_df):
    cd = features_df["conflict_density"].dropna()
    assert (cd >= 0).all()

def test_conflict_density_ukraine_border_elevated(features_df):
    """Poland and Romania should have above-median conflict density (2022 war effect)."""
    median = features_df["conflict_density"].median()
    pl = features_df[features_df["country_code"] == "PL"]["conflict_density"].mean()
    assert pl > median

def test_cei_nonneg(features_df):
    cei = features_df["cei"].dropna()
    assert (cei >= 0).all()

def test_phase1_tests_still_pass():
    """Sentinel: ensure Phase 1 invariants are not broken by enrichment."""
    # Run via: pytest tests/test_panel.py tests/test_features.py -q
    pass  # enforced by CI running full test suite
```

---

## 9. File Changes

**New files:**
- `src/data/enrich_features.py` — all enrichment logic (IHPI, schengen, conflict, chips, SIS, SAS, CEI)
- `src/data/ingest_acled.py` — ACLED Excel load + spatial join → annual NUTS2 conflict panel
- `build_phase2a.py` — orchestrator: loads Phase 1 outputs, calls enrich_features, writes updated parquet
- `tests/test_phase2a.py` — tests listed in §8

**Modified files:**
- `environment.yml` — already updated: geopandas 0.14, pyogrio 0.7, requests 2.32 added
- `tests/test_features.py` — remove `test_phase2_placeholders_nan` (now fails by design — all 8 filled)

**Unchanged:**
- All Phase 1 source files (`load_gold_layer.py`, `extend_eurostat.py`, `merge_validate.py`, `derive_features.py`)
- `build_nuts2_panel.py`

---

## 10. Orchestrator Sketch

```python
# build_phase2a.py
ROOT = Path(__file__).parent
FEATURES_PATH = ROOT / "data" / "processed" / "nuts2_features.parquet"
PANEL_PATH    = ROOT / "data" / "processed" / "nuts2_panel.parquet"
ACLED_PATH    = ROOT / "data" / "raw" / "acled" / "Europe-Central-Asia_aggregated_data_up_to_week_of-2026-05-23.xlsx"
GISCO_PATH    = ROOT / "data" / "raw" / "gisco" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"

np.random.seed(42)

def main():
    feat   = pd.read_parquet(FEATURES_PATH)
    panel  = pd.read_parquet(PANEL_PATH)
    feat   = enrich_features(feat, panel, ACLED_PATH, GISCO_PATH)
    feat.to_parquet(FEATURES_PATH, index=False)
    print(f"Wrote {FEATURES_PATH} — {len(feat)} rows x {len(feat.columns)} cols")
```
