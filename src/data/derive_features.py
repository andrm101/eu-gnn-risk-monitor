from pathlib import Path
import pandas as pd
import numpy as np

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
    pop_t = year_snap("population", 2023)
    pop_t3 = year_snap("population", 2020)

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
        feat[col] = pd.array([np.nan] * len(feat), dtype="float32")

    # ── guard: no inf values ──────────────────────────────────────────────────
    float_cols = feat.select_dtypes(include="float").columns
    assert not np.isinf(feat[float_cols].values).any(), "Inf values found in features"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} — {len(feat)} rows × {len(feat.columns)} cols")

    return feat
