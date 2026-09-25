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
