"""HeteroData assembly and temporal feature tensor for EU NUTS2 graph."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from src.graph.edge_builders import FEATURE_COLS

# Panel column → FEATURE_COLS column: the three time-varying raw indicators
_PANEL_FEAT_MAP: dict[str, str] = {
    "gdp_per_capita_pps": "gdp_per_capita_pps_latest",
    "unemployment_rate":  "unemployment_latest",
    "net_migration_rate": "net_migration_rate",
}


def build_hetero_graph(
    features_path: Path,
    node_index: pd.DataFrame,
    spatial_ei: torch.Tensor,
    economic_ei: torch.Tensor,
    schengen_ei: torch.Tensor,
) -> HeteroData:
    """Assemble tri-relational HeteroData with 2022-snapshot node features."""
    feat = pd.read_parquet(features_path)
    code_to_idx = node_index.set_index("nuts2_code")["node_idx"].to_dict()
    feat["node_idx"] = feat["nuts2_code"].map(code_to_idx)
    feat = feat.sort_values("node_idx").reset_index(drop=True)

    raw = feat[FEATURE_COLS].values.astype(np.float32)
    col_medians = np.nanmedian(raw, axis=0)
    nan_mask = np.isnan(raw)
    raw[nan_mask] = np.broadcast_to(col_medians, raw.shape)[nan_mask]
    x = torch.tensor(raw, dtype=torch.float32)

    data = HeteroData()
    data["region"].x = x
    data["region"].nuts2_code = feat["nuts2_code"].tolist()
    data["region"].country_code = feat["country_code"].tolist()

    data["region", "spatial",  "region"].edge_index = spatial_ei
    data["region", "economic", "region"].edge_index = economic_ei
    data["region", "schengen", "region"].edge_index = schengen_ei

    return data


def build_temporal_tensor(
    panel_path: Path,
    features_path: Path,
    node_index: pd.DataFrame,
) -> torch.Tensor:
    """Build (14, 242, 19) float32 tensor; GDP/unemployment panel data available from 2014 only — years 2010-2013 fall back to the 2023 snapshot values."""
    panel = pd.read_parquet(panel_path)
    feat_base = pd.read_parquet(features_path)

    code_to_idx = node_index.set_index("nuts2_code")["node_idx"].to_dict()
    feat_base = feat_base.copy()
    feat_base["node_idx"] = feat_base["nuts2_code"].map(code_to_idx)
    feat_base = feat_base.sort_values("node_idx").reset_index(drop=True)

    base = feat_base[FEATURE_COLS].values.astype(np.float32)  # (242, 19)
    years = list(range(2010, 2024))  # 14 years
    n_nodes = len(node_index)
    n_feats = len(FEATURE_COLS)

    temporal = np.zeros((len(years), n_nodes, n_feats), dtype=np.float32)

    for t, year in enumerate(years):
        matrix = base.copy()
        year_df = panel[panel["year"] == year].copy()
        year_df["node_idx"] = year_df["nuts2_code"].map(code_to_idx)
        year_df = year_df.dropna(subset=["node_idx"])
        year_df["node_idx"] = year_df["node_idx"].astype(int)

        for panel_col, feat_col in _PANEL_FEAT_MAP.items():
            if panel_col not in year_df.columns:
                continue
            f_idx = FEATURE_COLS.index(feat_col)
            valid = year_df[["node_idx", panel_col]].dropna()
            matrix[valid["node_idx"].values, f_idx] = valid[panel_col].values.astype(np.float32)

        temporal[t] = matrix

    # Fill residual NaN with per-feature median across all time × node cells
    for f in range(n_feats):
        col = temporal[:, :, f]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = float(np.nanmedian(col))
            temporal[:, :, f] = col

    return torch.tensor(temporal, dtype=torch.float32)
