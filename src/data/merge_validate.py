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

    Returns: merged panel DataFrame
    """
    # Gold layer provides static 2022 metadata columns.
    # rd_expenditure_pct_gdp and hi_tech_employment_pct are 2022-vintage only (no Eurostat
    # time series in Phase 1); they appear as a single value per region, same for all years.
    gold_meta_cols = ["nuts2_code", "nuts2_name", "country_code", "archetype_id",
                      "archetype_label", "data_quality_score"]
    for col in ["rd_expenditure_pct_gdp", "hi_tech_employment_pct"]:
        if col in gold_df.columns:
            gold_meta_cols.append(col)

    # Merge: keep all Eurostat rows, join Gold metadata left
    panel = eurostat_df.merge(gold_df[gold_meta_cols], on="nuts2_code", how="left")

    # Restrict to Gold's canonical 242 NUTS2 — drops any non-EU27 codes that slip
    # through Eurostat (e.g. UK/NO regions that match [A-Z]{2}\d{2} pattern).
    valid_codes = set(gold_df["nuts2_code"])
    panel = panel[panel["nuts2_code"].isin(valid_codes)].copy()

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

    assert panel["year"].between(_YEAR_MIN, _YEAR_MAX).all(), f"Year outside [{_YEAR_MIN}, {_YEAR_MAX}]"

    unemp = panel["unemployment_rate"].dropna()
    if len(unemp) > 0:
        assert unemp.between(0, 40).all(), "Unemployment out of plausible range"

    assert panel["data_coverage"].isin(["high", "med", "low"]).all()

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} — {len(panel):,} rows × {len(panel.columns)} cols")

    return panel
