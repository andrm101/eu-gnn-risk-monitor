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
    print("== Stage 1: load_gold_layer ==================================")
    gold_df = load_gold_layer(GOLD_PATH)
    print(f"  Gold layer: {len(gold_df)} regions loaded")

    print("== Stage 2: extend_eurostat ==================================")
    eurostat_df = extend_eurostat(RAW_EUROSTAT)
    print(f"  Eurostat panel: {len(eurostat_df):,} rows")

    print("== Stage 3: merge_and_validate ===============================")
    panel = merge_and_validate(gold_df, eurostat_df, PANEL_OUT)
    print(f"  Panel: {len(panel):,} rows x {len(panel.columns)} cols")

    print("== Stage 4: derive_gnn_features ==============================")
    features = derive_gnn_features(PANEL_OUT, FEATURES_OUT)
    print(f"  Features: {len(features)} rows x {len(features.columns)} cols")

    print("== Done =======================================================")
    print(f"  {PANEL_OUT}")
    print(f"  {FEATURES_OUT}")


if __name__ == "__main__":
    main()
