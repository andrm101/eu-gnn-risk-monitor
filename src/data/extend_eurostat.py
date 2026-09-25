from pathlib import Path
import pandas as pd
import numpy as np
import re

# Mapping from our column names to Eurostat TSV files + filter criteria.
# Each entry: (tsv_filename, geo_dimension_position, filter_dict)
# geo_dimension_position: which comma-split part of the first column is the geo code.
_SOURCES = {
    "net_migration_rate": ("tgs00099.tsv", -1, {"indic_de": "CNMIGRATRT"}),
    "unemployment_rate": ("tgs00010.tsv", -1, {"sex": "T", "isced11": "TOTAL"}),
    "population_count": ("demo_r_pjangrp3.tsv", -1, {"sex": "T", "age": "TOTAL"}),
    "gdp_millions_eur": ("nama_10r_3gdp.tsv", -1, {"unit": "MIO_EUR"}),
}

_NUTS2_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{2}$")
_YEAR_RANGE = range(2010, 2024)  # 2010 inclusive, 2024 exclusive → 2010–2023


def _strip_flags(val: object) -> float:
    """Strip Eurostat flag letters and return float; NaN if missing."""
    if not isinstance(val, str):
        return float(val) if pd.notna(val) else np.nan
    val = val.strip()
    if val in (":", "", "-"):
        return np.nan
    cleaned = re.sub(r"[a-z ]+$", "", val.strip())
    if not cleaned or cleaned == ":":  # handles ": p", ": b", ": e"
        return np.nan
    return float(cleaned)


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
      population_count, gdp_millions_eur, gdp_per_capita_pps, data_coverage
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
