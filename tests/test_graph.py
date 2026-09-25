import pytest
import torch
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "nuts2_features.parquet"
GISCO_PATH    = ROOT / "data" / "raw" / "gisco" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
PANEL_PATH    = ROOT / "data" / "processed" / "nuts2_panel.parquet"
GRAPH_PATH    = ROOT / "data" / "processed" / "nuts2_graph.pt"
TEMPORAL_PATH = ROOT / "data" / "processed" / "nuts2_temporal_features.pt"
NODE_IDX_PATH = ROOT / "data" / "processed" / "nuts2_node_index.parquet"


# ── Unit fixtures (build from raw files, no orchestrator needed) ──────────────

@pytest.fixture(scope="session")
def node_index_raw():
    if not FEATURES_PATH.exists():
        pytest.skip("nuts2_features.parquet not found")
    from src.graph.edge_builders import build_node_index
    return build_node_index(FEATURES_PATH)


@pytest.fixture(scope="session")
def spatial_ei(node_index_raw):
    if not GISCO_PATH.exists():
        pytest.skip("GISCO GeoJSON not found")
    from src.graph.edge_builders import build_spatial_edges
    return build_spatial_edges(GISCO_PATH, node_index_raw)


@pytest.fixture(scope="session")
def economic_ei():
    if not FEATURES_PATH.exists():
        pytest.skip("nuts2_features.parquet not found")
    from src.graph.edge_builders import build_economic_edges
    return build_economic_edges(FEATURES_PATH)


@pytest.fixture(scope="session")
def schengen_ei(spatial_ei, node_index_raw):
    from src.graph.edge_builders import build_schengen_edges
    return build_schengen_edges(spatial_ei, node_index_raw)


@pytest.fixture(scope="session")
def built_graph(spatial_ei, economic_ei, schengen_ei, node_index_raw):
    if not FEATURES_PATH.exists():
        pytest.skip("nuts2_features.parquet not found")
    from src.graph.build_graph import build_hetero_graph
    return build_hetero_graph(FEATURES_PATH, node_index_raw, spatial_ei, economic_ei, schengen_ei)


@pytest.fixture(scope="session")
def built_temporal(node_index_raw):
    if not PANEL_PATH.exists() or not FEATURES_PATH.exists():
        pytest.skip("panel or features parquet not found")
    from src.graph.build_graph import build_temporal_tensor
    return build_temporal_tensor(PANEL_PATH, FEATURES_PATH, node_index_raw)


# ── Integration fixtures (load from .pt files — require build_phase2b.py) ────

@pytest.fixture(scope="session")
def graph():
    if not GRAPH_PATH.exists():
        pytest.skip("nuts2_graph.pt not found — run build_phase2b.py first")
    return torch.load(GRAPH_PATH, weights_only=False)


@pytest.fixture(scope="session")
def temporal():
    if not TEMPORAL_PATH.exists():
        pytest.skip("nuts2_temporal_features.pt not found — run build_phase2b.py first")
    return torch.load(TEMPORAL_PATH, weights_only=False)


@pytest.fixture(scope="session")
def node_index():
    if not NODE_IDX_PATH.exists():
        pytest.skip("nuts2_node_index.parquet not found — run build_phase2b.py first")
    return pd.read_parquet(NODE_IDX_PATH)


# ── Task 2: node index + spatial edge unit tests ──────────────────────────────

def test_node_index_has_242_rows(node_index_raw):
    assert len(node_index_raw) == 242


def test_node_index_columns(node_index_raw):
    assert {"node_idx", "nuts2_code", "country_code"}.issubset(node_index_raw.columns)


def test_node_idx_unique_and_contiguous(node_index_raw):
    idxs = sorted(node_index_raw["node_idx"].tolist())
    assert idxs == list(range(242))


def test_spatial_edge_tensor_shape(spatial_ei):
    assert spatial_ei.ndim == 2 and spatial_ei.shape[0] == 2


def test_spatial_edge_count_range(spatial_ei):
    E = spatial_ei.shape[1]
    assert 800 <= E <= 1400, f"Spatial edges {E} outside [800, 1400]"


def test_spatial_no_self_loops(spatial_ei):
    assert (spatial_ei[0] != spatial_ei[1]).all()


def test_spatial_bidirectional(spatial_ei):
    fwd = set(zip(spatial_ei[0].tolist(), spatial_ei[1].tolist()))
    rev = set(zip(spatial_ei[1].tolist(), spatial_ei[0].tolist()))
    assert fwd == rev, "Spatial edge set is not symmetric"


def test_spatial_indices_in_range(spatial_ei):
    assert spatial_ei.min() >= 0 and spatial_ei.max() <= 241


# ── Task 3: economic kNN edge unit tests ──────────────────────────────────────

def test_economic_edge_tensor_shape(economic_ei):
    assert economic_ei.ndim == 2 and economic_ei.shape[0] == 2


def test_economic_edge_count_range(economic_ei):
    E = economic_ei.shape[1]
    assert 968 <= E <= 1936, f"Economic edges {E} outside [968, 1936]"


def test_economic_no_self_loops(economic_ei):
    assert (economic_ei[0] != economic_ei[1]).all()


def test_economic_bidirectional(economic_ei):
    fwd = set(zip(economic_ei[0].tolist(), economic_ei[1].tolist()))
    rev = set(zip(economic_ei[1].tolist(), economic_ei[0].tolist()))
    assert fwd == rev, "Economic edge set is not symmetric"


def test_economic_indices_in_range(economic_ei):
    assert economic_ei.min() >= 0 and economic_ei.max() <= 241


# ── Task 4: Schengen edge unit tests ─────────────────────────────────────────

def test_schengen_subset_of_spatial_unit(spatial_ei, schengen_ei):
    assert schengen_ei.shape[1] <= spatial_ei.shape[1]


def test_schengen_no_self_loops(schengen_ei):
    assert (schengen_ei[0] != schengen_ei[1]).all()


def test_schengen_bidirectional(schengen_ei):
    fwd = set(zip(schengen_ei[0].tolist(), schengen_ei[1].tolist()))
    rev = set(zip(schengen_ei[1].tolist(), schengen_ei[0].tolist()))
    assert fwd == rev, "Schengen edge set is not symmetric"


def test_schengen_de_fr_cross_unit(schengen_ei, node_index_raw):
    """DE and FR share a border and are both Schengen — at least one DE↔FR edge."""
    de_idxs = set(node_index_raw[node_index_raw["country_code"] == "DE"]["node_idx"])
    fr_idxs = set(node_index_raw[node_index_raw["country_code"] == "FR"]["node_idx"])
    pairs = set(zip(schengen_ei[0].tolist(), schengen_ei[1].tolist()))
    cross = [(i, j) for i in de_idxs for j in fr_idxs if (i, j) in pairs]
    assert len(cross) > 0, "No DE↔FR edges in Schengen edge set"


def test_schengen_non_member_excluded(schengen_ei, node_index_raw):
    """RO is not in SCHENGEN_2023 — no RO node should appear in any Schengen edge."""
    ro_idxs = set(node_index_raw[node_index_raw["country_code"] == "RO"]["node_idx"])
    src = set(schengen_ei[0].tolist())
    tgt = set(schengen_ei[1].tolist())
    ro_as_src = ro_idxs & src
    ro_as_tgt = ro_idxs & tgt
    assert not ro_as_src and not ro_as_tgt, f"RO nodes found in Schengen edges"


# ── Task 5: HeteroData unit tests ─────────────────────────────────────────────

def test_built_node_count(built_graph):
    assert built_graph["region"].x.shape[0] == 242


def test_built_feature_dim(built_graph):
    assert built_graph["region"].x.shape[1] == 19


def test_built_no_nan_features(built_graph):
    assert not built_graph["region"].x.isnan().any()


def test_built_has_three_edge_types(built_graph):
    types = {et for et in built_graph.edge_types}
    assert ("region", "spatial",  "region") in types
    assert ("region", "economic", "region") in types
    assert ("region", "schengen", "region") in types


def test_built_nuts2_code_list_length(built_graph):
    assert len(built_graph["region"].nuts2_code) == 242


# ── Task 6: temporal tensor unit tests ───────────────────────────────────────

def test_built_temporal_shape(built_temporal):
    assert built_temporal.shape == (14, 242, 19)


def test_built_temporal_dtype(built_temporal):
    assert built_temporal.dtype == torch.float32


def test_built_temporal_no_nan(built_temporal):
    assert not built_temporal.isnan().any()


def test_built_temporal_year_axis(built_temporal):
    # GDP panel data starts 2014 — 2010-2013 fall back to 2023 snapshot (bit-identical to t[13])
    # axis-0 index 4 = 2014, index 13 = 2023; FEATURE_COLS[0] = gdp_per_capita_pps_latest
    gdp_2014 = built_temporal[4, :, 0]
    gdp_2023 = built_temporal[13, :, 0]
    assert not torch.allclose(gdp_2014, gdp_2023), "Temporal tensor appears static"


# ── Task 7: integration tests (require build_phase2b.py to have been run) ─────

def test_node_count(graph):
    assert graph["region"].x.shape[0] == 242


def test_feature_dim(graph):
    assert graph["region"].x.shape[1] == 19


def test_no_nan_features(graph):
    assert not graph["region"].x.isnan().any()


def test_spatial_edge_count(graph):
    E = graph["region", "spatial", "region"].edge_index.shape[1]
    assert 800 <= E <= 1400, f"Spatial edge count {E} out of expected range"


def test_economic_edge_count(graph):
    E = graph["region", "economic", "region"].edge_index.shape[1]
    assert 968 <= E <= 1936, f"Economic edge count {E} out of expected range"


def test_schengen_subset_of_spatial(graph):
    E_sc = graph["region", "schengen", "region"].edge_index.shape[1]
    E_s  = graph["region", "spatial",  "region"].edge_index.shape[1]
    assert E_sc <= E_s


def test_no_self_loops(graph):
    for rel in [
        ("region", "spatial",  "region"),
        ("region", "economic", "region"),
        ("region", "schengen", "region"),
    ]:
        ei = graph[rel].edge_index
        assert (ei[0] != ei[1]).all(), f"Self-loop found in {rel}"


def test_edge_indices_in_range(graph):
    for rel in [
        ("region", "spatial",  "region"),
        ("region", "economic", "region"),
        ("region", "schengen", "region"),
    ]:
        ei = graph[rel].edge_index
        assert ei.max() <= 241 and ei.min() >= 0


def test_de_fr_schengen_edge(graph, node_index):
    de = set(node_index[node_index["country_code"] == "DE"]["node_idx"])
    fr = set(node_index[node_index["country_code"] == "FR"]["node_idx"])
    ei = graph["region", "schengen", "region"].edge_index
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    cross = [(i, j) for i in de for j in fr if (i, j) in pairs or (j, i) in pairs]
    assert len(cross) > 0, "No DE↔FR Schengen edges found"


def test_temporal_tensor_shape(temporal):
    assert temporal.shape == (14, 242, 19)


def test_temporal_no_nan(temporal):
    assert not temporal.isnan().any()


def test_node_index_length(node_index):
    assert len(node_index) == 242
