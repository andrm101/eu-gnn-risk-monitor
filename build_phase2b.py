"""Phase 2B: build tri-relational PyG HeteroData graph for 242 EU NUTS2 regions."""
import numpy as np
import torch
from pathlib import Path

from src.graph.edge_builders import (
    build_node_index,
    build_spatial_edges,
    build_economic_edges,
    build_schengen_edges,
)
from src.graph.build_graph import build_hetero_graph, build_temporal_tensor

np.random.seed(42)
torch.manual_seed(42)

ROOT          = Path(__file__).parent
FEATURES_PATH = ROOT / "data" / "processed" / "nuts2_features.parquet"
PANEL_PATH    = ROOT / "data" / "processed" / "nuts2_panel.parquet"
GISCO_PATH    = ROOT / "data" / "raw" / "gisco" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
GRAPH_PATH    = ROOT / "data" / "processed" / "nuts2_graph.pt"
TEMPORAL_PATH = ROOT / "data" / "processed" / "nuts2_temporal_features.pt"
NODE_IDX_PATH = ROOT / "data" / "processed" / "nuts2_node_index.parquet"


def main() -> None:
    print("== Phase 2B: Graph Construction ===========================")

    node_index = build_node_index(FEATURES_PATH)
    node_index.to_parquet(NODE_IDX_PATH, index=False)
    print(f"  node index : {len(node_index)} nodes -> {NODE_IDX_PATH.name}")

    print("  building spatial edges (GISCO .touches()) ...")
    spatial_ei = build_spatial_edges(GISCO_PATH, node_index)
    print(f"  spatial    : {spatial_ei.shape[1]} directed edges")

    print("  building economic kNN edges (k=4) ...")
    economic_ei = build_economic_edges(FEATURES_PATH)
    print(f"  economic   : {economic_ei.shape[1]} directed edges")

    print("  building Schengen edges ...")
    schengen_ei = build_schengen_edges(spatial_ei, node_index)
    print(f"  schengen   : {schengen_ei.shape[1]} directed edges")

    print("  assembling HeteroData ...")
    graph = build_hetero_graph(FEATURES_PATH, node_index, spatial_ei, economic_ei, schengen_ei)
    torch.save(graph, GRAPH_PATH)
    print(f"  saved      : {GRAPH_PATH.name}")

    print("  building temporal tensor (14 × 242 × 19) ...")
    temporal = build_temporal_tensor(PANEL_PATH, FEATURES_PATH, node_index)
    torch.save(temporal, TEMPORAL_PATH)
    print(f"  saved      : {TEMPORAL_PATH.name}")

    print("== Done =====================================================")


if __name__ == "__main__":
    main()
