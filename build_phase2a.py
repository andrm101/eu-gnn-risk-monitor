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
