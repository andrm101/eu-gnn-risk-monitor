# Phase 1 — Data Foundation: Design Spec

**Project:** EU-GNN-Risk-Monitor  
**Phase:** 1 of 5 (Data Foundation)  
**Date:** 2026-06-06  
**Status:** Approved — ready for implementation planning

---

## Goal

Build the Bronze → Silver data pipeline that produces two artefacts consumed by all downstream phases:
- `nuts2_panel.parquet` — long-format source of truth for all 242 EU-27 NUTS2 regions, 2010–2023
- `nuts2_features.parquet` — wide-format GNN node feature matrix (242 rows), Phase 2 placeholder columns declared

**Canonical-base invariant:** Phase 1 ensures all future node features can be derived from `nuts2_panel.parquet` without re-touching raw sources. Raw files are never re-ingested for downstream phases — they extend the features file only.

---

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Geographic scope | Full EU-27 (242 NUTS2), staged ingestion | GNN requires full EU graph; specialist sources (ACLED, Schengen) deferred to Phase 2 |
| Starting point | Innovation Panel Gold layer as Bronze input | Avoids re-ingesting GDP/unemployment; Gold layer read-only, never modified |
| Year range | 2010–2023 (or latest available) | ACLED reliable post-2010; post-enlargement Eurostat solid for all EU-27; COVID included |
| Temporal structure | Full panel stored; derived snapshot+lags fed to GNN v1 | Panel enables future temporal GNN; Phase 1 keeps GNN architecture simple |
| Pipeline architecture | Single orchestrator + four stage functions as importable modules | Matches `I0_build_infra_layer.py` pattern; each stage independently testable |
| Project folder name | `EU-GNN-Risk-Monitor/` | Sandbox naming convention |

---

## Architecture

`build_nuts2_panel.py` (root orchestrator) calls four stage functions in order:

```
load_gold_layer()
    └─ reads EU-Innovation-Panel/data/gold/region_profiles_gold.parquet (read-only)
    └─ selects GNN-relevant columns, filters 2010–2022 × 242 NUTS2
    ↓
extend_eurostat()
    └─ adds net_migration_rate from tgs00099
    └─ extends GDP, unemployment, population to 2023 via nama_10r_3gdp, tgs00010, demo_r_pjangrp3
    └─ flags gaps as data_coverage (high/med/low) — NaN retained, never dropped
    ↓
merge_and_validate()
    └─ joins on nuts2_code × year
    └─ asserts 242 NUTS2 present, no duplicates, range checks
    └─ writes data/processed/nuts2_panel.parquet
    ↓
derive_gnn_features()
    └─ computes z-scores, 3yr CAGRs, prosperity_gap, stress_proxy
    └─ declares Phase 2 placeholder columns as NaN
    └─ writes data/processed/nuts2_features.parquet
```

---

## File Structure

```
EU-GNN-Risk-Monitor/
├── environment.yml
├── build_nuts2_panel.py               # Phase 1 orchestrator
├── data/
│   ├── raw/
│   │   ├── eurostat/
│   │   │   ├── tgs00099_net_migration.csv
│   │   │   ├── tgs00010_unemployment.csv
│   │   │   ├── nama_10r_3gdp.csv
│   │   │   └── README.md             # provenance: source URL, download date, variable defs
│   │   └── gisco/
│   │       └── nuts2_2021.geojson    # GISCO boundary file — geometry only
│   └── processed/
│       ├── nuts2_panel.parquet       # long: ~3,400 rows × 10+ cols
│       └── nuts2_features.parquet   # wide: 242 rows × 19 cols (11 filled + 8 NaN placeholders)
├── data/
│   └── external/                        # future: DEMIFER, ACLED preprocessed — empty in Phase 1
├── src/
│   └── data/
│       ├── load_gold_layer.py
│       ├── extend_eurostat.py
│       ├── merge_validate.py
│       └── derive_features.py
├── tests/
│   ├── test_gold_layer.py
│   ├── test_eurostat.py
│   └── test_features.py
├── notebooks/                        # EDA only — not part of pipeline
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

**Cross-project dependency:** Gold layer is referenced as `../EU-Innovation-Panel/data/gold/region_profiles_gold.parquet` — never copied. Documented in `data/raw/eurostat/README.md`.

**Phase 2+ additions (not scaffolded in Phase 1):**
- `src/data/fetch_acled.py`
- `src/features/build_node_features.py`
- `src/features/build_edge_features.py`
- `src/models/graphsage_model.py`
- `src/visualization/dashboard_app.py`

---

## Data Contract

### `nuts2_panel.parquet` — long format, ~3,388 rows

| column | dtype | source | notes |
|---|---|---|---|
| nuts2_code | str | Gold | ISO format e.g. DE11, RO21 — PK part |
| nuts2_name | str | Gold | Human-readable region name |
| country_code | str | Gold | ISO-2 derived from nuts2_code[:2] |
| year | int16 | Gold/Eurostat | 2010–2023 — PK part |
| gdp_per_capita_pps | float32 | Gold + nama_10r_3gdp | EUR per inhabitant, purchasing power standard — plausible range [5,000, 150,000]; confirmed against Gold layer at implementation |
| unemployment_rate | float32 | Gold + tgs00010 | % of active population |
| population | float32 | Gold + demo_r_pjangrp3 | Thousands |
| net_migration_rate | float32 | tgs00099 | Per 1,000 inhabitants — new variable |
| rd_expenditure_pct_gdp | float32 | Gold | % GDP — may have sparse years |
| high_tech_employment_pct | float32 | Gold | % total employment |
| data_coverage | str | derived | "high" / "med" / "low" per row |

**Invariants:** unique on `(nuts2_code, year)`; NaN retained and flagged — never silently dropped.

### `nuts2_features.parquet` — wide format, 242 rows

**Phase 1 columns (filled):**

| column | dtype | derivation |
|---|---|---|
| nuts2_code | str | Primary key — matches geometry, NUTS 2021 codes |
| country_code | str | ISO-2 derived from nuts2_code[:2] — retained for country-level grouping without re-join |
| gdp_pc_z | float32 | Z-score across 242 NUTS2, latest year |
| unemployment_z | float32 | Z-score across 242 NUTS2, latest year |
| net_migration_rate | float32 | Raw value, latest year |
| gdp_growth_3yr | float32 | CAGR: (gdp_t / gdp_t-3)^(1/3) − 1 |
| unemp_change_3yr | float32 | Δ unemployment rate over 3 years (pp) |
| pop_growth_3yr | float32 | CAGR population over 3 years |
| prosperity_gap | float32 | (EU27_mean_gdp_pc − region_gdp_pc) / EU27_mean_gdp_pc |
| stress_proxy | float32 | unemployment_z − gdp_pc_z |
| data_coverage_frac | float32 | Fraction of panel years where all core macro cols are non-NaN — Phase 2 can use to weight sparse regions |

**Phase 2 placeholder columns (all NaN in Phase 1):**
`conflict_density`, `conflict_lag_1`, `schengen_member`, `chips_flag`, `ihpi`, `sis`, `sas`, `cei`

---

## Test Coverage

### `tests/test_gold_layer.py`
- `test_242_nuts2_present` — `len(df.nuts2_code.unique()) == 242`
- `test_year_range_covered` — `df.year.min() <= 2010` and `df.year.max() >= 2022`
- `test_no_duplicate_keys` — no duplicate `(nuts2_code, year)` pairs
- `test_gdp_pc_range` — all values in `[1000, 150000]`

### `tests/test_eurostat.py`
- `test_migration_column_present` — `net_migration_rate` in columns
- `test_migration_sign_direction` — DE11 (Stuttgart) shows positive net migration
- `test_macro_ranges` — unemployment in `[0, 40]`, GDP > 0
- `test_coverage_flag_valid` — `data_coverage` values in `{"high", "med", "low"}`
- `test_population_accounting_consistency` — EU27 population 2020 in `[440_000, 460_000]` (thousands)

### `tests/test_features.py`
- `test_exactly_242_rows` — exactly 242 rows
- `test_zscore_properties` — `gdp_pc_z` mean ≈ 0 (tol 0.01), std ≈ 1 (tol 0.05)
- `test_prosperity_gap_direction` — Bulgaria mean gap > Luxembourg mean gap
- `test_phase2_placeholders_nan` — `ihpi`, `sis`, `sas`, `cei`, `conflict_density` all NaN
- `test_no_inf_values` — no `np.inf` in any float32 column
- `test_panel_features_nuts2_match` — `set(panel.nuts2_code.unique()) == set(features.nuts2_code)` — no region dropped between long and wide
- `test_coverage_flag_consistency` — for all rows where `data_coverage == "high"`, core macro columns (`gdp_per_capita_pps`, `unemployment_rate`, `population`) must be non-NaN
- `test_migration_plausibility_recent` — for years ≥ 2015, `net_migration_rate` is non-NaN for ≥ 200 of 242 NUTS2 and lies in `[-50, 50]` per 1,000

---

## Invariants

- Raw CSVs in `data/raw/` are never modified — immutable
- Gold layer is read-only — never written back
- All joins on `nuts2_code` (ISO NUTS2 format)
- NaN → explicit `data_coverage` flag, never silently dropped or zero-filled
- NUTS2 geometry kept separate in `data/raw/gisco/` — not merged into panel
- Phase 2 placeholder columns declared as `NaN`, not `0` (semantically distinct)
- No absolute paths — all paths relative to project root via `pathlib.Path`
- Random seed: `np.random.seed(42)` in any stochastic step
- All spatial operations use NUTS 2021 boundary codes (GISCO 2021 release), CRS EPSG:4326 — never mix with NUTS 2016 codes
- All Eurostat series pulled from a single reference snapshot (download date documented in `data/raw/eurostat/README.md`) — no mixing of vintages across variables

---

## Out of Scope for Phase 1

- ACLED conflict data ingestion
- Schengen/CHIPS binary flags
- IHPI, SIS, SAS, CEI composite index construction
- Graph object construction (PyTorch Geometric)
- GNN model training
- Dashboard / visualisation
- Gamification layer

---

## Literature Anchors

- Composite indicator methodology: Nardo et al. (OECD/JRC 2008) — applied in Phase 2 for IHPI/SIS/SAS/CEI
- Migration data: Eurostat tgs00099 methodological notes (Eurostat 2023)
- NUTS2 classification: Eurostat NUTS 2021 (Regulation EC 1059/2003 as amended)
