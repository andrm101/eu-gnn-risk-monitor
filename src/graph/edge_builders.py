"""Edge builders for EU NUTS2 tri-relational heterogeneous graph."""
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from src.data.enrich_features import SCHENGEN_2023

FEATURE_COLS: list[str] = [
    "gdp_per_capita_pps_latest", "unemployment_latest", "net_migration_rate",
    "gdp_pc_z", "unemployment_z", "gdp_growth_3yr", "unemp_change_3yr",
    "pop_growth_3yr", "prosperity_gap", "stress_proxy", "data_coverage_frac",
    "ihpi", "schengen_member", "conflict_density", "conflict_lag_1",
    "chips_flag", "sis", "sas", "cei",
]


def build_node_index(features_path: Path) -> pd.DataFrame:
    """Map parquet row order to integer node_idx + nuts2_code + country_code."""
    feat = pd.read_parquet(features_path, columns=["nuts2_code", "country_code"])
    feat = feat.reset_index(drop=True)
    feat.index.name = "node_idx"
    return feat.reset_index()


def build_spatial_edges(gisco_path: Path, node_index: pd.DataFrame) -> torch.Tensor:
    """Return edge_index (2, E) for shared-border adjacency, both directions stored."""
    gdf = gpd.read_file(gisco_path, engine="pyogrio")[["NUTS_ID", "geometry"]]
    valid = set(node_index["nuts2_code"])
    gdf = gdf[gdf["NUTS_ID"].isin(valid)].reset_index(drop=True)

    code_to_idx = node_index.set_index("nuts2_code")["node_idx"].to_dict()
    gdf["node_idx"] = gdf["NUTS_ID"].map(code_to_idx)

    # Rename to avoid column-suffix ambiguity in sjoin result
    src_gdf = gdf[["node_idx", "geometry"]].rename(columns={"node_idx": "src_idx"})
    tgt_gdf = gdf[["node_idx", "geometry"]].rename(columns={"node_idx": "tgt_idx"})

    # predicate="touches" finds pairs sharing a border; sjoin with same GDF
    # produces both (A→B) and (B→A) because A touches B implies B touches A
    joined = gpd.sjoin(src_gdf, tgt_gdf, how="inner", predicate="touches")
    joined = joined[joined["src_idx"] != joined["tgt_idx"]]
    joined = joined.drop_duplicates(subset=["src_idx", "tgt_idx"])

    src = torch.tensor(joined["src_idx"].values, dtype=torch.long)
    tgt = torch.tensor(joined["tgt_idx"].values, dtype=torch.long)
    return torch.stack([src, tgt], dim=0).contiguous()


def build_economic_edges(features_path: Path, k: int = 4) -> torch.Tensor:
    """Return edge_index (2, E) for k=4 economic kNN, symmetrised."""
    feat = pd.read_parquet(features_path)
    X = feat[FEATURE_COLS].values.astype(np.float64)
    # Impute NaN with column median (consistent with build_graph.py)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.tile(col_medians, (X.shape[0], 1))[nan_mask]

    X = StandardScaler().fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)
    # indices[:, 0] is self (distance == 0); skip it

    # Collect unique undirected pairs, then emit both directions
    undirected: set[tuple[int, int]] = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            undirected.add((min(i, j), max(i, j)))

    src_list: list[int] = []
    tgt_list: list[int] = []
    for a, b in undirected:
        src_list.extend([a, b])
        tgt_list.extend([b, a])

    return torch.tensor([src_list, tgt_list], dtype=torch.long).contiguous()


def build_schengen_edges(
    spatial_edge_index: torch.Tensor,
    node_index: pd.DataFrame,
) -> torch.Tensor:
    """Filter spatial edges to pairs where both country_codes are in SCHENGEN_2023."""
    idx_to_country = node_index.set_index("node_idx")["country_code"].to_dict()

    src = spatial_edge_index[0].tolist()
    tgt = spatial_edge_index[1].tolist()
    mask = torch.tensor(
        [
            idx_to_country.get(i, "") in SCHENGEN_2023
            and idx_to_country.get(j, "") in SCHENGEN_2023
            for i, j in zip(src, tgt)
        ],
        dtype=torch.bool,
    )
    return torch.stack(
        [spatial_edge_index[0][mask], spatial_edge_index[1][mask]], dim=0
    ).contiguous()
