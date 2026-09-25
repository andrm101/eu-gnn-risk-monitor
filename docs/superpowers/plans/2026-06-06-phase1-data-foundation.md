# Phase 1 — Data Foundation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Bronze→Silver pipeline that produces `nuts2_panel.parquet` (long panel, 242 NUTS2 × 2010–2023) and `nuts2_features.parquet` (wide GNN node feature matrix, 242 rows) for the EU-GNN-Risk-Monitor project.

**Architecture:** Single orchestrator `build_nuts2_panel.py` calls four stage functions from `src/data/`: `load_gold_layer()` (canonical NUTS2 list + 2022 metadata), `extend_eurostat()` (full time series from Eurostat TSVs), `merge_and_validate()` (join + write panel), `derive_gnn_features()` (derive features + write wide matrix). All tests run against real processed parquets or the live Gold layer — no mocks.

**Tech Stack:** Python 3.11, pandas 2.2, pyarrow 15, numpy 1.26, scikit-learn 1.4, pytest 8.1

---

## ⚠️ Prerequisites — Data Downloads (do before Task 1)

The pipeline reads pre-downloaded Eurostat bulk TSV exports. Download these files manually from Eurostat and place them in `data/raw/eurostat/`:

| File | Eurostat dataset | Eurostat navigation |
|---|---|---|
| `tgs00099.tsv` | Net migration crude rate, NUTS2 | Statistics → Regional → Population → tgs00099 |
| `tgs00010.tsv` | Unemployment rate, NUTS2 | Statistics → Regional → Labour → tgs00010 |
| `nama_10r_3gdp.tsv` | GDP at current prices, NUTS2/3 | Statistics → Regional → Economy → nama_10r_3gdp |
| `demo_r_pjangrp3.tsv` | Population by NUTS3 (aggregated to NUTS2) | Statistics → Regional → Population → demo_r_pjangrp3 |

Also download: `data/raw/gisco/nuts2_2021.geojson` from the GISCO NUTS download page (NUTS 2021, scale 1:1M, EPSG:4326).

**Eurostat TSV format note:** Bulk exports use tab-separated columns where the first column encodes dimension values as `dim1,dim2,geo\TIME_PERIOD` and subsequent columns are years. Numeric values may include flag letters (`b`, `p`, `e`, `d`) that must be stripped. The `extend_eurostat.py` module includes a shared parser for this.

**Gold layer path:** The pipeline reads `../EU-Innovation-Panel/data/gold/region_profiles_gold.parquet` as a cross-project dependency. This file already exists.

**Known Gold layer issue:** The `net_migration_rate` column in the Gold layer contains population counts (mislabeled). Do NOT use it. Migration rates come entirely from tgs00099.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `environment.yml` | Create | Conda env with all deps pinned |
| `src/__init__.py` | Create | Package marker |
| `src/data/__init__.py` | Create | Package marker |
| `src/data/load_gold_layer.py` | Create | Stage 1: canonical NUTS2 list from Gold layer |
| `src/data/extend_eurostat.py` | Create | Stage 2: time series from Eurostat TSVs |
| `src/data/merge_validate.py` | Create | Stage 3: join + validate + write nuts2_panel.parquet |
| `src/data/derive_features.py` | Create | Stage 4: derived features + write nuts2_features.parquet |
| `build_nuts2_panel.py` | Create | Orchestrator calling Stages 1–4 |
| `tests/__init__.py` | Create | Package marker |
| `tests/conftest.py` | Create | Session-scoped fixtures for processed parquets |
| `tests/test_gold_layer.py` | Create | 4 tests for load_gold_layer() |
| `tests/test_eurostat.py` | Create | 5 tests for nuts2_panel.parquet |
| `tests/test_features.py` | Create | 8 tests for nuts2_features.parquet |
| `data/raw/eurostat/README.md` | Create | Provenance: source URLs, download date, variable defs |

---

## Task 1: Scaffold Project Structure

**Files:**
- Create: `environment.yml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/raw/eurostat/README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create `environment.yml`**

```yaml
name: eu-gnn-risk
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas=2.2
  - pyarrow=15
  - numpy=1.26
  - scikit-learn=1.4
  - pytest=8.1
  - pytest-cov
  - pip
```

- [ ] **Step 2: Create package markers**

```bash
# Run from EU-GNN-Risk-Monitor/
mkdir -p src/data tests data/raw/eurostat data/raw/gisco data/processed data/external notebooks
touch src/__init__.py src/data/__init__.py tests/__init__.py
touch data/raw/gisco/.gitkeep data/external/.gitkeep data/processed/.gitkeep
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Create `data/raw/eurostat/README.md`**

```markdown
# Eurostat Raw Data — EU-GNN-Risk-Monitor

All files are immutable bulk exports from Eurostat. Never modify.

## Files

| File | Dataset code | Variable | Unit | Download date | URL |
|---|---|---|---|---|---|
| tgs00099.tsv | tgs00099 | Net migration crude rate | Per 1000 pop, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/tgs00099 |
| tgs00010.tsv | tgs00010 | Unemployment rate | %, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/tgs00010 |
| nama_10r_3gdp.tsv | nama_10r_3gdp | GDP at current prices | Millions EUR, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/nama_10r_3gdp |
| demo_r_pjangrp3.tsv | demo_r_pjangrp3 | Population by age/sex | Persons, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjangrp3 |

## Cross-project dependency

Gold layer (read-only):
../EU-Innovation-Panel/data/gold/region_profiles_gold.parquet

Upstream vintage: ref_year_cost=2022 (all regions). Gold layer provides canonical
NUTS2 list and 2022 snapshot for cross-validation only. Time series built fresh from Eurostat.

## NUTS version

All spatial codes: NUTS 2021 (Regulation EC 1059/2003 as amended by EC 2016/2066).
Geometry: GISCO NUTS 2021, EPSG:4326.
Eurostat snapshot date: fill in download date above.
```

- [ ] **Step 5: Create `.gitignore`**

```
data/processed/
data/external/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.env
```

- [ ] **Step 6: Verify structure and commit**

```bash
# Run from EU-GNN-Risk-Monitor/
find . -type f | grep -v node_modules | grep -v .git | sort
git add environment.yml src/ tests/conftest.py data/raw/eurostat/README.md .gitignore data/raw/gisco/.gitkeep data/external/.gitkeep data/processed/.gitkeep
git commit -m "feat(EU-GNN-P1): scaffold project structure and environment"
```

---

## Task 2: load_gold_layer() — TDD

**Files:**
- Create: `tests/test_gold_layer.py`
- Create: `src/data/load_gold_layer.py`

The Gold layer is a static 2022 cross-section (242 rows). `load_gold_layer()` returns a DataFrame with the canonical NUTS2 list and selected metadata. The index `nuts2_code` becomes a column.

**Gold layer column facts (confirmed by inspection):**
- `gdp_per_capita_pps`: absolute EUR per capita PPS, range ~10,300–127,000
- `population`: raw person counts (NOT thousands)
- `net_migration_rate`: **mislabeled** — equals `population`. Do NOT select this column.
- `archetype_id`: 0 or 1 (k=2 archetypes from EU-Innovation-Panel clustering)
- Index: `nuts2_code` (e.g. AT11, RO21)

- [ ] **Step 1: Write `tests/test_gold_layer.py` (failing)**

```python
import pytest
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

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
```

- [ ] **Step 2: Run — confirm all 4 tests fail**

```bash
cd EU-GNN-Risk-Monitor
pytest tests/test_gold_layer.py -v
```

Expected: `ImportError: cannot import name 'load_gold_layer'`

- [ ] **Step 3: Create `src/data/load_gold_layer.py`**

```python
from pathlib import Path
import pandas as pd

# Columns selected from Gold layer for EU-GNN use.
# Excludes net_migration_rate (mislabeled — contains population counts).
_GOLD_COLUMNS = [
    "nuts2_name",
    "country_code",
    "gdp_per_capita_pps",
    "population",
    "rd_expenditure_pct_gdp",
    "hi_tech_employment_pct",
    "archetype_id",
    "archetype_label",
    "data_quality_score",
]


def load_gold_layer(path: Path) -> pd.DataFrame:
    """Load EU-Innovation-Panel Gold layer; return canonical NUTS2 list with 2022 metadata.

    The Gold layer is a static 2022 cross-section (242 rows). The index is nuts2_code.
    Returns a flat DataFrame with nuts2_code as a regular column and year=2022 added.
    """
    df = pd.read_parquet(path)

    # Confirm index is nuts2_code
    if df.index.name == "nuts2_code":
        df = df.reset_index()
    elif "nuts2_code" not in df.columns:
        raise ValueError("Gold layer has no nuts2_code index or column")

    # Select available columns (tolerate Gold layer schema evolution)
    available = ["nuts2_code"] + [c for c in _GOLD_COLUMNS if c in df.columns]
    df = df[available].copy()

    # Derive country_code if missing
    if "country_code" not in df.columns:
        df["country_code"] = df["nuts2_code"].str[:2]

    # Derive nuts2_name if missing
    if "nuts2_name" not in df.columns:
        df["nuts2_name"] = df["nuts2_code"]

    # Tag vintage year
    df["year"] = 2022

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run — confirm all 4 tests pass**

```bash
pytest tests/test_gold_layer.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/data/load_gold_layer.py tests/test_gold_layer.py
git commit -m "feat(EU-GNN-P1): load_gold_layer — canonical NUTS2 list from Gold layer"
```

---

## Task 3: extend_eurostat() — TDD

**Files:**
- Create: `src/data/extend_eurostat.py`

Reads four Eurostat TSV files and returns a long-format DataFrame (nuts2_code × year, 2010–2023) with macro variables and a `data_coverage` flag. The function does not touch the Gold layer.

**Eurostat TSV format:**
```
unit,geo\TIME_PERIOD	2000	2001	...	2023
PC,AT11	3.9	4.1	...	4.5 p
PC,BG31	8.2	9.1	...	7.8 b
```
First column: comma-separated dimensions, last dimension is geo code. Year columns contain numeric values optionally followed by a space and a flag letter (`b` break, `p` provisional, `e` estimated). Values of `:` mean missing.

- [ ] **Step 1: Create `src/data/extend_eurostat.py`**

```python
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Mapping from our column names to Eurostat TSV files + filter criteria.
# Each entry: (tsv_filename, geo_dimension_position, filter_dict)
# geo_dimension_position: which comma-split part of the first column is the geo code.
_SOURCES = {
    "net_migration_rate": ("tgs00099.tsv", -1, {}),
    "unemployment_rate": ("tgs00010.tsv", -1, {}),
    "population_count": ("demo_r_pjangrp3.tsv", -1, {"sex": "T", "age": "TOTAL"}),
    "gdp_millions_eur": ("nama_10r_3gdp.tsv", -1, {"unit": "MIO_EUR"}),
}

_NUTS2_PATTERN = re.compile(r"^[A-Z]{2}\d{2}$")
_YEAR_RANGE = range(2010, 2024)  # 2010 inclusive, 2024 exclusive → 2010–2023


def _strip_flags(val: str) -> float:
    """Strip Eurostat flag letters and return float; NaN if missing."""
    if not isinstance(val, str):
        return float(val) if pd.notna(val) else np.nan
    val = val.strip()
    if val in (":", "", "-"):
        return np.nan
    return float(re.sub(r"[a-z ]+$", "", val.strip()))


def _parse_tsv(path: Path, geo_pos: int, filter_dict: dict) -> pd.DataFrame:
    """Parse a single Eurostat bulk-download TSV into a long (geo, year, value) frame."""
    raw = pd.read_csv(path, sep="\t", dtype=str)

    # First column: dimension codes, e.g. "unit,sex,age,geo\TIME_PERIOD"
    dim_col = raw.columns[0]
    dim_names = re.split(r"[,\\]", dim_col.replace("TIME_PERIOD", "").rstrip(",\\"))
    dim_names = [d.strip() for d in dim_names if d.strip()]

    # Split first column into dimension values
    dims_df = raw[dim_col].str.split(",", expand=True)
    dims_df.columns = dim_names[: len(dims_df.columns)]

    # Apply dimension filters (e.g. sex=T, age=TOTAL)
    mask = pd.Series([True] * len(dims_df), index=dims_df.index)
    for k, v in filter_dict.items():
        if k in dims_df.columns:
            mask &= dims_df[k].str.strip() == v

    # Extract geo code
    geo_col = dims_df.columns[geo_pos]
    dims_df["geo"] = dims_df[geo_col].str.strip()

    # Keep only NUTS2 rows (4-char: 2 letters + 2 digits)
    nuts2_mask = dims_df["geo"].str.match(_NUTS2_PATTERN, na=False)
    mask &= nuts2_mask

    dims_filtered = dims_df[mask].copy()
    raw_filtered = raw[mask].copy()

    # Year columns
    year_cols = {
        int(c.strip()): c
        for c in raw.columns[1:]
        if c.strip().isdigit() and int(c.strip()) in _YEAR_RANGE
    }

    rows = []
    for year, col in year_cols.items():
        vals = raw_filtered[col].apply(_strip_flags)
        for geo, val in zip(dims_filtered["geo"], vals):
            rows.append({"nuts2_code": geo, "year": year, "_value": val})

    return pd.DataFrame(rows)


def _data_coverage_flag(row: pd.Series, core_cols: list[str]) -> str:
    """Classify row coverage: high (all core non-NaN), med (≥50%), low (<50%)."""
    present = sum(pd.notna(row[c]) for c in core_cols if c in row.index)
    frac = present / len(core_cols) if core_cols else 0
    if frac == 1.0:
        return "high"
    if frac >= 0.5:
        return "med"
    return "low"


_CORE_COLS = ["net_migration_rate", "unemployment_rate", "population_count", "gdp_millions_eur"]


def extend_eurostat(raw_dir: Path) -> pd.DataFrame:
    """Build long-format panel (nuts2_code × year) from Eurostat TSV files.

    Returns DataFrame with columns:
      nuts2_code, year, net_migration_rate, unemployment_rate,
      population_count, gdp_millions_eur, data_coverage
    """
    frames: dict[str, pd.DataFrame] = {}
    for col, (fname, geo_pos, flt) in _SOURCES.items():
        tsv_path = raw_dir / fname
        if not tsv_path.exists():
            raise FileNotFoundError(f"Eurostat file missing: {tsv_path}")
        df = _parse_tsv(tsv_path, geo_pos, flt)
        df = df.rename(columns={"_value": col})
        frames[col] = df

    # Outer join on nuts2_code × year across all sources
    base = frames["net_migration_rate"]
    for col, df in frames.items():
        if col == "net_migration_rate":
            continue
        base = base.merge(df, on=["nuts2_code", "year"], how="outer")

    # Filter to NUTS2 pattern and year range
    base = base[base["nuts2_code"].str.match(_NUTS2_PATTERN, na=False)].copy()
    base = base[base["year"].isin(_YEAR_RANGE)].copy()

    # Compute GDP per capita in EUR (millions EUR × 1e6 / population).
    # Named gdp_per_capita_pps to match spec — this is EUR-equivalent PPS approximated at
    # NUTS2 level; true PPS deflation (prc_ppp_ind) is only available at national level.
    base["gdp_per_capita_pps"] = (
        base["gdp_millions_eur"] * 1_000_000 / base["population_count"].replace(0, np.nan)
    )

    # Coverage flag
    base["data_coverage"] = base.apply(
        lambda r: _data_coverage_flag(r, _CORE_COLS), axis=1
    )

    return base.reset_index(drop=True)
```

- [ ] **Step 2: Verify the module imports without error**

```bash
cd EU-GNN-Risk-Monitor
python -c "from src.data.extend_eurostat import extend_eurostat; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke-test against real Eurostat files (manual check)**

```bash
python -c "
from pathlib import Path
from src.data.extend_eurostat import extend_eurostat
df = extend_eurostat(Path('data/raw/eurostat'))
print(df.shape)
print(df.dtypes)
print(df.head())
print('NaN rates:')
print(df.isna().mean())
"
```

Expected: shape roughly `(242 × 14 years, 8)` = ~3,400 rows, sensible values

- [ ] **Step 4: Commit**

```bash
git add src/data/extend_eurostat.py
git commit -m "feat(EU-GNN-P1): extend_eurostat — parse Eurostat TSVs into long panel"
```

---

## Task 4: merge_and_validate() — write nuts2_panel.parquet

**Files:**
- Create: `src/data/merge_validate.py`
- Test: `tests/test_eurostat.py`

Joins the Gold layer metadata (242-row cross-section) with the Eurostat panel, validates invariants, and writes `data/processed/nuts2_panel.parquet`.

- [ ] **Step 1: Write `tests/test_eurostat.py` (all skip until parquet exists)**

```python
import pytest
import pandas as pd
import numpy as np


def test_migration_column_present(panel_df):
    # caveman say: tgs00099 must join successfully
    assert "net_migration_rate" in panel_df.columns


def test_migration_sign_direction(panel_df):
    # caveman say: Stuttgart (DE11) is a net-in region — expect positive rates recently
    de11 = panel_df[(panel_df["nuts2_code"] == "DE11") & (panel_df["year"] >= 2015)]
    assert len(de11) > 0, "DE11 not found in panel"
    assert de11["net_migration_rate"].mean() > 0, "Stuttgart expected positive net migration"


def test_macro_ranges(panel_df):
    # caveman say: no impossible values after join
    unemp = panel_df["unemployment_rate"].dropna()
    assert unemp.between(0, 40).all(), f"Unemployment out of range: {unemp[~unemp.between(0,40)]}"
    gdp = panel_df["gdp_per_capita_pps"].dropna()
    assert gdp.gt(0).all()


def test_coverage_flag_valid(panel_df):
    # caveman say: flag must be exactly one of three values
    assert panel_df["data_coverage"].isin(["high", "med", "low"]).all()


def test_population_accounting_consistency(panel_df):
    # caveman say: EU27 population 2020 must be ~447M persons (raw counts, not thousands)
    eu_2020 = panel_df[panel_df["year"] == 2020]["population"].sum()
    assert 400_000_000 < eu_2020 < 500_000_000, f"EU population 2020: {eu_2020:,.0f}"
```

- [ ] **Step 2: Run — confirm all tests skip (parquet not yet generated)**

```bash
pytest tests/test_eurostat.py -v
```

Expected: `5 skipped`

- [ ] **Step 3: Create `src/data/merge_validate.py`**

```python
from pathlib import Path
import pandas as pd
import numpy as np

_EXPECTED_NUTS2 = 242
_YEAR_MIN, _YEAR_MAX = 2010, 2023


def merge_and_validate(gold_df: pd.DataFrame, eurostat_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Join Gold metadata with Eurostat time panel, validate, write nuts2_panel.parquet.

    gold_df: 242-row cross-section from load_gold_layer()
    eurostat_df: long panel from extend_eurostat()
    out_path: path to write nuts2_panel.parquet
    """
    # Merge: left join Gold canonical list × eurostat panel
    # Gold provides nuts2_name, country_code, archetype_id etc. as region-level attributes.
    # Eurostat provides time-varying variables.
    # Gold layer provides static 2022 metadata columns.
    # rd_expenditure_pct_gdp and hi_tech_employment_pct are 2022-vintage only (no Eurostat
    # time series in Phase 1); they appear as a single value per region, same for all years.
    gold_meta_cols = ["nuts2_code", "nuts2_name", "country_code", "archetype_id",
                      "archetype_label", "data_quality_score"]
    for col in ["rd_expenditure_pct_gdp", "hi_tech_employment_pct"]:
        if col in gold_df.columns:
            gold_meta_cols.append(col)

    panel = eurostat_df.merge(gold_df[gold_meta_cols], on="nuts2_code", how="left")

    # Rename internal column names to match spec output contract
    panel = panel.rename(columns={"population_count": "population"})

    # Cast dtypes
    panel["year"] = panel["year"].astype("int16")
    for col in ["net_migration_rate", "unemployment_rate", "gdp_per_capita_pps",
                "population", "gdp_millions_eur", "rd_expenditure_pct_gdp",
                "hi_tech_employment_pct"]:
        if col in panel.columns:
            panel[col] = panel[col].astype("float32")

    # Validate invariants
    n_regions = panel["nuts2_code"].nunique()
    assert n_regions == _EXPECTED_NUTS2, f"Expected 242 NUTS2, got {n_regions}"

    dupes = panel.duplicated(["nuts2_code", "year"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (nuts2_code, year) rows"

    unemp = panel["unemployment_rate"].dropna()
    if len(unemp) > 0:
        assert unemp.between(0, 40).all(), "Unemployment out of plausible range"

    assert panel["data_coverage"].isin(["high", "med", "low"]).all()

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} — {len(panel):,} rows × {len(panel.columns)} cols")

    return panel
```

- [ ] **Step 4: Run pipeline stages 1–3 via a temporary script to generate the parquet**

```bash
python - <<'EOF'
from pathlib import Path
from src.data.load_gold_layer import load_gold_layer
from src.data.extend_eurostat import extend_eurostat
from src.data.merge_validate import merge_and_validate

ROOT = Path(".")
GOLD = ROOT.parent / "EU-Innovation-Panel" / "data" / "gold" / "region_profiles_gold.parquet"

gold_df = load_gold_layer(GOLD)
eurostat_df = extend_eurostat(ROOT / "data" / "raw" / "eurostat")
merge_and_validate(gold_df, eurostat_df, ROOT / "data" / "processed" / "nuts2_panel.parquet")
EOF
```

- [ ] **Step 5: Run test_eurostat.py — confirm all 5 pass**

```bash
pytest tests/test_eurostat.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/data/merge_validate.py tests/test_eurostat.py
git commit -m "feat(EU-GNN-P1): merge_and_validate — join Gold + Eurostat, write nuts2_panel.parquet"
```

---

## Task 5: derive_gnn_features() — write nuts2_features.parquet

**Files:**
- Create: `src/data/derive_features.py`
- Create: `tests/test_features.py`

Reads `nuts2_panel.parquet`, derives 10 Phase-1 feature columns plus 8 NaN Phase-2 placeholders, writes `nuts2_features.parquet` (242 rows × 19 cols).

- [ ] **Step 1: Write `tests/test_features.py` (all skip until parquet exists)**

```python
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


def test_phase2_placeholders_nan(features_df):
    # caveman say: P2 columns must be NaN, not zero — distinct semantics
    for col in ["ihpi", "sis", "sas", "cei", "conflict_density", "conflict_lag_1",
                "schengen_member", "chips_flag"]:
        assert features_df[col].isna().all(), f"{col} is not all NaN"


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
    for col in ["unemployment_rate", "population_count"]:
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
    assert valid.between(-50, 50).all(), f"Migration rate out of [-50,50]: {valid[~valid.between(-50,50)]}"
```

- [ ] **Step 2: Run — confirm all tests skip**

```bash
pytest tests/test_features.py -v
```

Expected: `7 skipped, 1 skipped or needs panel_df too`

- [ ] **Step 3: Create `src/data/derive_features.py`**

```python
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

_LATEST_YEAR = 2023  # fall back to most recent available


def _latest_year_snapshot(panel: pd.DataFrame, col: str) -> pd.Series:
    """For each nuts2_code, return value for latest non-NaN year ≤ _LATEST_YEAR."""
    sub = panel[panel["year"] <= _LATEST_YEAR][["nuts2_code", "year", col]].dropna(subset=[col])
    idx = sub.groupby("nuts2_code")["year"].idxmax()
    return sub.loc[idx].set_index("nuts2_code")[col]


def _cagr(start: pd.Series, end: pd.Series, n: int) -> pd.Series:
    """Compound annual growth rate; NaN where start is zero or missing."""
    ratio = end / start.replace(0, np.nan)
    return ratio ** (1.0 / n) - 1.0


def derive_gnn_features(panel_path: Path, out_path: Path) -> pd.DataFrame:
    """Derive Phase-1 GNN node features from nuts2_panel.parquet.

    Writes nuts2_features.parquet (242 rows × 19 cols).
    Phase-2 placeholder columns are all NaN.
    """
    panel = pd.read_parquet(panel_path)

    # ── canonical index ──────────────────────────────────────────────────────
    nuts2_codes = sorted(panel["nuts2_code"].unique())
    feat = pd.DataFrame({"nuts2_code": nuts2_codes})

    # Bring in country_code and archetype_id from panel (region-level attributes)
    meta = panel[["nuts2_code", "country_code", "archetype_id"]].drop_duplicates("nuts2_code")
    feat = feat.merge(meta, on="nuts2_code", how="left")

    # ── levels (latest available year) ───────────────────────────────────────
    gdp_latest = _latest_year_snapshot(panel, "gdp_per_capita_pps")
    unemp_latest = _latest_year_snapshot(panel, "unemployment_rate")
    mig_latest = _latest_year_snapshot(panel, "net_migration_rate")

    feat["gdp_per_capita_pps_latest"] = feat["nuts2_code"].map(gdp_latest).astype("float32")
    feat["unemployment_latest"] = feat["nuts2_code"].map(unemp_latest).astype("float32")
    feat["net_migration_rate"] = feat["nuts2_code"].map(mig_latest).astype("float32")

    # ── z-scores (cross-sectional, across 242 regions) ───────────────────────
    gdp_vals = feat["gdp_per_capita_pps_latest"]
    unemp_vals = feat["unemployment_latest"]
    feat["gdp_pc_z"] = ((gdp_vals - gdp_vals.mean()) / gdp_vals.std()).astype("float32")
    feat["unemployment_z"] = ((unemp_vals - unemp_vals.mean()) / unemp_vals.std()).astype("float32")

    # ── 3-year CAGRs ─────────────────────────────────────────────────────────
    def year_snap(col: str, yr: int) -> pd.Series:
        sub = panel[panel["year"] == yr][["nuts2_code", col]].set_index("nuts2_code")[col]
        return feat["nuts2_code"].map(sub).astype("float32")

    gdp_t = year_snap("gdp_per_capita_pps", 2023)
    gdp_t3 = year_snap("gdp_per_capita_pps", 2020)
    unemp_t = year_snap("unemployment_rate", 2023)
    unemp_t3 = year_snap("unemployment_rate", 2020)
    pop_t = year_snap("population_count", 2023)
    pop_t3 = year_snap("population_count", 2020)

    feat["gdp_growth_3yr"] = _cagr(gdp_t3, gdp_t, 3).astype("float32")
    feat["unemp_change_3yr"] = (unemp_t - unemp_t3).astype("float32")
    feat["pop_growth_3yr"] = _cagr(pop_t3, pop_t, 3).astype("float32")

    # ── composite proxies ─────────────────────────────────────────────────────
    eu_mean_gdp = gdp_vals.mean()
    feat["prosperity_gap"] = ((eu_mean_gdp - gdp_vals) / eu_mean_gdp).astype("float32")
    feat["stress_proxy"] = (feat["unemployment_z"] - feat["gdp_pc_z"]).astype("float32")

    # ── coverage fraction (fraction of panel years with full macro data) ──────
    core = ["net_migration_rate", "unemployment_rate", "population", "gdp_millions_eur"]
    core_present = [c for c in core if c in panel.columns]
    coverage_frac = (
        panel.groupby("nuts2_code")[core_present]
        .apply(lambda g: g.notna().all(axis=1).mean())
        .rename("data_coverage_frac")
    )
    feat["data_coverage_frac"] = feat["nuts2_code"].map(coverage_frac).astype("float32")

    # ── Phase-2 placeholder columns (all NaN) ─────────────────────────────────
    for col in ["conflict_density", "conflict_lag_1", "schengen_member", "chips_flag",
                "ihpi", "sis", "sas", "cei"]:
        feat[col] = np.nan

    # ── guard: no inf values ──────────────────────────────────────────────────
    float_cols = feat.select_dtypes(include="float").columns
    assert not np.isinf(feat[float_cols].values).any(), "Inf values found in features"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} — {len(feat)} rows × {len(feat.columns)} cols")

    return feat
```

- [ ] **Step 4: Run derive_features manually to generate the parquet**

```bash
python - <<'EOF'
from pathlib import Path
from src.data.derive_features import derive_gnn_features

ROOT = Path(".")
derive_gnn_features(
    ROOT / "data" / "processed" / "nuts2_panel.parquet",
    ROOT / "data" / "processed" / "nuts2_features.parquet",
)
EOF
```

- [ ] **Step 5: Run all feature tests — confirm pass**

```bash
pytest tests/test_features.py -v
```

Expected: `8 passed` (the 2 tests that also need `panel_df` use the fixture from conftest.py)

- [ ] **Step 6: Commit**

```bash
git add src/data/derive_features.py tests/test_features.py
git commit -m "feat(EU-GNN-P1): derive_gnn_features — GNN node feature matrix with Phase-2 placeholders"
```

---

## Task 6: Orchestrator — build_nuts2_panel.py

**Files:**
- Create: `build_nuts2_panel.py`

Wires all four stages into a single end-to-end run.

- [ ] **Step 1: Create `build_nuts2_panel.py`**

```python
"""Phase 1 orchestrator: Bronze → Silver pipeline for EU-GNN-Risk-Monitor.

Usage:
    python build_nuts2_panel.py

Writes:
    data/processed/nuts2_panel.parquet    (long panel, ~3400 rows)
    data/processed/nuts2_features.parquet (wide GNN features, 242 rows)
"""
import numpy as np
from pathlib import Path

from src.data.load_gold_layer import load_gold_layer
from src.data.extend_eurostat import extend_eurostat
from src.data.merge_validate import merge_and_validate
from src.data.derive_features import derive_gnn_features

np.random.seed(42)

ROOT = Path(__file__).parent
GOLD_PATH = ROOT.parent / "EU-Innovation-Panel" / "data" / "gold" / "region_profiles_gold.parquet"
RAW_EUROSTAT = ROOT / "data" / "raw" / "eurostat"
PANEL_OUT = ROOT / "data" / "processed" / "nuts2_panel.parquet"
FEATURES_OUT = ROOT / "data" / "processed" / "nuts2_features.parquet"


def main() -> None:
    print("── Stage 1: load_gold_layer ──────────────────────────")
    gold_df = load_gold_layer(GOLD_PATH)
    print(f"  Gold layer: {len(gold_df)} regions loaded")

    print("── Stage 2: extend_eurostat ──────────────────────────")
    eurostat_df = extend_eurostat(RAW_EUROSTAT)
    print(f"  Eurostat panel: {len(eurostat_df):,} rows")

    print("── Stage 3: merge_and_validate ───────────────────────")
    panel = merge_and_validate(gold_df, eurostat_df, PANEL_OUT)
    print(f"  Panel: {len(panel):,} rows × {len(panel.columns)} cols")

    print("── Stage 4: derive_gnn_features ──────────────────────")
    features = derive_gnn_features(PANEL_OUT, FEATURES_OUT)
    print(f"  Features: {len(features)} rows × {len(features.columns)} cols")

    print("── Done ──────────────────────────────────────────────")
    print(f"  {PANEL_OUT}")
    print(f"  {FEATURES_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end**

```bash
cd EU-GNN-Risk-Monitor
python build_nuts2_panel.py
```

Expected output:
```
── Stage 1: load_gold_layer ──────────────────────────
  Gold layer: 242 regions loaded
── Stage 2: extend_eurostat ──────────────────────────
  Eurostat panel: ~3,388 rows
── Stage 3: merge_and_validate ───────────────────────
  Panel: ~3,388 rows × 12 cols
── Stage 4: derive_gnn_features ──────────────────────
  Features: 242 rows × 19 cols
── Done ──────────────────────────────────────────────
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: `16 passed, 0 failed` (or some skipped if Eurostat files not yet downloaded)

- [ ] **Step 4: Commit**

```bash
git add build_nuts2_panel.py
git commit -m "feat(EU-GNN-P1): orchestrator — end-to-end Bronze→Silver pipeline"
```

---

## Self-Review Checklist

Before marking Phase 1 complete, verify:

- [ ] `pytest tests/ -v` → all 16 tests pass (or skip cleanly if data not downloaded)
- [ ] `python build_nuts2_panel.py` completes without errors
- [ ] `data/processed/nuts2_panel.parquet` — 242 unique NUTS2 codes, years 2010–2023
- [ ] `data/processed/nuts2_features.parquet` — exactly 242 rows, no inf values
- [ ] Phase-2 placeholder columns all NaN in features file
- [ ] `data/raw/eurostat/README.md` has actual download dates filled in
- [ ] No absolute paths in any `.py` file
- [ ] `git log --oneline` shows 5+ commits (one per task)
