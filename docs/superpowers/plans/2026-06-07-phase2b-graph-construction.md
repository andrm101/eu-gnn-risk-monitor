# Phase 2B — Graph Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tri-relational PyTorch Geometric `HeteroData` graph over 242 EU NUTS2 regions with spatial, economic, and Schengen edge types, plus a (14, 242, 19) temporal feature tensor for Phase 3 GNN models.

**Architecture:** Two new source modules — `src/graph/edge_builders.py` (FEATURE_COLS constant + three edge-builder functions) and `src/graph/build_graph.py` (node index + HeteroData assembly + temporal tensor). `build_phase2b.py` at project root orchestrates: loads Phase 2A outputs, builds graph, writes three artefacts. All 45 Phase 2A tests must continue passing.

**Tech Stack:** Python 3.11, pandas 2.2, geopandas 0.14 (shapely 2.x backend), scikit-learn 1.4, torch 2.3.0+cpu, torch-geometric 2.5.3, pytest 8.1.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `environment.yml` | Modify | Add `pip:` block for torch + torch-geometric |
| `src/graph/__init__.py` | Create | Empty package marker |
| `src/graph/edge_builders.py` | Create | `FEATURE_COLS` constant; `build_spatial_edges`, `build_economic_edges`, `build_schengen_edges` |
| `src/graph/build_graph.py` | Create | `build_node_index`, `build_hetero_graph`, `build_temporal_tensor` |
| `build_phase2b.py` | Create | Orchestrator: node index → edges → HeteroData → temporal → save |
| `tests/test_graph.py` | Create | Unit tests for each function + integration tests from spec §5 |

---

## Task 1: Install torch + torch-geometric

**Files:** `environment.yml`

- [ ] **Step 1: Add pip block to environment.yml**

Replace the last two lines of `environment.yml` (currently `- pip`) with:

```yaml
name: eu-gnn-risk
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas=2.2
  - pyarrow=15
  - numpy=1.26
  - scikit-learn=1.4
  - pytest=8.1
  - pytest-cov
  - jupyter
  - ipykernel
  - requests=2.32
  - geopandas=0.14
  - pyogrio=0.7
  - pip
  - pip:
    - torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
    - torch-geometric==2.5.3
```

- [ ] **Step 2: Install the packages**

Run from the project root (with `eu-gnn-risk` conda env active):

```bash
pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "import torch; from torch_geometric.data import HeteroData; print('torch', torch.__version__); h = HeteroData(); print('HeteroData OK')"
```

Expected output:
```
torch 2.3.0+cpu
HeteroData OK
```

- [ ] **Step 4: Commit**

```bash
git add environment.yml
git commit -m "feat(phase2b): add torch 2.3.0+cpu and torch-geometric 2.5.3 to environment"
```

---

## Task 2: Package scaffold + node index + spatial edges

**Files:**
- Create: `src/graph/__init__.py`
- Create: `src/graph/edge_builders.py` (FEATURE_COLS + build_node_index stub + build_spatial_edges)
- Create: `tests/test_graph.py` (fixtures + spatial edge unit tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_graph.py::test_node_index_has_242_rows -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'src.graph'`

- [ ] **Step 3: Create package marker**

Create `src/graph/__init__.py` as an empty file.

- [ ] **Step 4: Implement `edge_builders.py`**

Create `src/graph/edge_builders.py`:

```python
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

    # predicate="touches" finds pairs sharing a border; sjoin produces both directions
    # because src and tgt are the same regions (A touches B → entry for (A,B) and (B,A))
    joined = gpd.sjoin(src_gdf, tgt_gdf, how="inner", predicate="touches")
    joined = joined[joined["src_idx"] != joined["tgt_idx"]]

    src = torch.tensor(joined["src_idx"].values, dtype=torch.long)
    tgt = torch.tensor(joined["tgt_idx"].values, dtype=torch.long)
    return torch.stack([src, tgt], dim=0).contiguous()


def build_economic_edges(features_path: Path, k: int = 4) -> torch.Tensor:
    """Return edge_index (2, E) for k=4 economic kNN, symmetrised."""
    feat = pd.read_parquet(features_path)
    X = feat[FEATURE_COLS].values.astype(np.float64)
    X = StandardScaler().fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)
    # indices[:, 0] is self (distance == 0); skip it

    src_list: list[int] = []
    tgt_list: list[int] = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            if i < j:
                src_list.extend([i, j])
                tgt_list.extend([j, i])

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
```

- [ ] **Step 5: Run the Task 2 tests**

```bash
pytest tests/test_graph.py -k "node_index or spatial" -v
```

Expected: 7 PASSED (the 4 node_index + 4 spatial tests; the GISCO load takes ~15s on first run).

- [ ] **Step 6: Commit**

```bash
git add src/graph/__init__.py src/graph/edge_builders.py tests/test_graph.py
git commit -m "feat(phase2b): scaffold src/graph package, FEATURE_COLS, build_node_index, build_spatial_edges"
```

---

## Task 3: Economic kNN edges

**Files:** `tests/test_graph.py` (add economic tests; `build_economic_edges` already exists in edge_builders.py from Task 2)

- [ ] **Step 1: Add failing tests to `tests/test_graph.py`**

Append after the Task 2 test block:

```python
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
```

- [ ] **Step 2: Run to confirm they pass (build_economic_edges already implemented)**

```bash
pytest tests/test_graph.py -k "economic" -v
```

Expected: 5 PASSED. (If `FEATURES_PATH` is present, no skip.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph.py
git commit -m "test(phase2b): economic kNN edge unit tests — 5 PASSED"
```

---

## Task 4: Schengen edges

**Files:** `tests/test_graph.py` (add Schengen unit tests; `build_schengen_edges` already implemented)

- [ ] **Step 1: Add failing tests to `tests/test_graph.py`**

Append after the Task 3 block:

```python
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
    """RO is not in SCHENGEN_2023 — no RO node should appear on both sides of any edge."""
    ro_idxs = set(node_index_raw[node_index_raw["country_code"] == "RO"]["node_idx"])
    src = set(schengen_ei[0].tolist())
    tgt = set(schengen_ei[1].tolist())
    # A RO node can appear as src only if tgt is Schengen, which is impossible by construction
    ro_as_src = ro_idxs & src
    ro_as_tgt = ro_idxs & tgt
    assert not ro_as_src and not ro_as_tgt, f"RO nodes found in Schengen edges"
```

- [ ] **Step 2: Run to confirm they pass**

```bash
pytest tests/test_graph.py -k "schengen" -v
```

Expected: 5 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph.py
git commit -m "test(phase2b): Schengen edge unit tests — 5 PASSED"
```

---

## Task 5: HeteroData assembly

**Files:**
- Create: `src/graph/build_graph.py`
- Modify: `tests/test_graph.py` (add HeteroData unit tests)

- [ ] **Step 1: Add failing tests to `tests/test_graph.py`**

Append after the Task 4 block:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_graph.py -k "built" -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'src.graph.build_graph'`

- [ ] **Step 3: Implement `build_graph.py`**

Create `src/graph/build_graph.py`:

```python
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


def build_node_index(features_path: Path) -> pd.DataFrame:
    """Map parquet row order to integer node_idx + nuts2_code + country_code."""
    feat = pd.read_parquet(features_path, columns=["nuts2_code", "country_code"])
    feat = feat.reset_index(drop=True)
    feat.index.name = "node_idx"
    return feat.reset_index()


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

    x = torch.tensor(feat[FEATURE_COLS].values, dtype=torch.float32)

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
    """Build (14, 242, 19) temporal feature tensor.

    GDP/unemployment/migration update per year from the panel.
    All other features held at 2022 snapshot values for every year.
    Residual NaN filled with per-feature median across all time-node cells.
    """
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
```

- [ ] **Step 4: Run the Task 5 tests**

```bash
pytest tests/test_graph.py -k "built" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/graph/build_graph.py tests/test_graph.py
git commit -m "feat(phase2b): build_graph.py — build_hetero_graph + build_temporal_tensor"
```

---

## Task 6: Temporal tensor

**Files:** `tests/test_graph.py` (add temporal unit tests; `build_temporal_tensor` already implemented)

- [ ] **Step 1: Add failing tests to `tests/test_graph.py`**

Append after the Task 5 block:

```python
# ── Task 6: temporal tensor unit tests ───────────────────────────────────────

def test_built_temporal_shape(built_temporal):
    assert built_temporal.shape == (14, 242, 19)


def test_built_temporal_dtype(built_temporal):
    assert built_temporal.dtype == torch.float32


def test_built_temporal_no_nan(built_temporal):
    assert not built_temporal.isnan().any()


def test_built_temporal_year_axis(built_temporal, built_graph):
    """GDP values in year 0 (2010) should differ from year 13 (2023)."""
    # Feature index 0 is gdp_per_capita_pps_latest — should vary across years
    gdp_2010 = built_temporal[0, :, 0]
    gdp_2023 = built_temporal[13, :, 0]
    assert not torch.allclose(gdp_2010, gdp_2023), "Temporal tensor appears static"
```

- [ ] **Step 2: Run to confirm they pass**

```bash
pytest tests/test_graph.py -k "built_temporal" -v
```

Expected: 4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph.py
git commit -m "test(phase2b): temporal tensor unit tests — 4 PASSED"
```

---

## Task 7: Orchestrator + full integration tests

**Files:**
- Create: `build_phase2b.py`
- Modify: `tests/test_graph.py` (add 12 integration tests from spec §5)

- [ ] **Step 1: Add integration tests to `tests/test_graph.py`**

Append after the Task 6 block:

```python
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
```

- [ ] **Step 2: Run to confirm integration tests skip (artefacts not yet built)**

```bash
pytest tests/test_graph.py -k "test_node_count or test_feature_dim or test_temporal_tensor_shape" -v
```

Expected: 3 SKIPPED (nuts2_graph.pt not found).

- [ ] **Step 3: Create `build_phase2b.py`**

Create at project root:

```python
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
    print(f"  node index : {len(node_index)} nodes → {NODE_IDX_PATH.name}")

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
```

- [ ] **Step 4: Run the orchestrator**

```bash
python build_phase2b.py
```

Expected output (edge counts will vary slightly):
```
== Phase 2B: Graph Construction ===========================
  node index : 242 nodes → nuts2_node_index.parquet
  building spatial edges (GISCO .touches()) ...
  spatial    : 1054 directed edges
  building economic kNN edges (k=4) ...
  economic   : 1148 directed edges
  building Schengen edges ...
  schengen   : 712 directed edges
  assembling HeteroData ...
  saved      : nuts2_graph.pt
  building temporal tensor (14 × 242 × 19) ...
  saved      : nuts2_temporal_features.pt
== Done =====================================================
```

Spatial count must fall in [800, 1400]; economic in [968, 1936].

- [ ] **Step 5: Run the full integration test suite**

```bash
pytest tests/test_graph.py -v
```

Expected: all unit tests PASSED, all 12 integration tests PASSED (none skipped now that .pt files exist).

- [ ] **Step 6: Run the full Phase 2A + Phase 2B suite**

```bash
pytest tests/ -v
```

Expected: 45 Phase 2A + 31 Phase 2B = 76 tests, all PASSED, 0 FAILED.

- [ ] **Step 7: Commit**

```bash
git add build_phase2b.py tests/test_graph.py
git commit -m "feat(phase2b): orchestrator build_phase2b.py + 12 integration tests — 76 tests PASSED"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Covered by |
|---|---|
| 242 nodes, integer index 0–241 | Task 2 `build_node_index`, tests `test_node_index_*` |
| `data['region'].x` shape (242, 19), float32 | Task 5 `build_hetero_graph`, test `test_built_feature_dim` |
| `nuts2_node_index.parquet` artefact | Task 7 orchestrator, test `test_node_index_length` |
| Spatial edges from GISCO `.touches()`, both directions | Task 2 `build_spatial_edges`, test `test_spatial_bidirectional` |
| Spatial edge count 800–1400 | Task 2 test `test_spatial_edge_count_range` |
| Economic kNN k=4, StandardScaler, symmetrised | Task 3 `build_economic_edges`, test `test_economic_edge_count_range` |
| Economic edge count 968–1936 | Task 3 test |
| Schengen edges = subset of spatial where both ∈ SCHENGEN_2023 | Task 4 `build_schengen_edges` |
| Schengen ≤ spatial edge count | Task 4 + Task 7 spec tests |
| DE↔FR Schengen edge present | Task 4 test `test_schengen_de_fr_cross_unit`, Task 7 spec test |
| RO not in Schengen edges | Task 4 test `test_schengen_non_member_excluded` |
| Overseas NUTS2 (FRY1–5, ES70, PT20, PT30): zero spatial edges | Implicit — sjoin only finds touches, islands don't touch continent |
| `nuts2_graph.pt` artefact | Task 7 orchestrator |
| Temporal tensor (14, 242, 19), float32, no NaN | Task 6 + Task 7 spec tests |
| Temporal tensor axis 0 = years 2010–2023 | Task 6 `build_temporal_tensor` uses `range(2010, 2024)` |
| Panel columns (gdp, unemployment, migration) vary per year | Task 6 `_PANEL_FEAT_MAP` loop |
| Composite indices static at 2022 snapshot | Task 6 — only 3 panel columns overridden |
| Missing cells filled with column median | Task 6 post-loop NaN fill |
| `nuts2_temporal_features.pt` artefact | Task 7 orchestrator |
| `environment.yml` pip block for torch + pyg | Task 1 |
| All 45 Phase 2A tests still passing | Task 7 Step 6 full suite run |

### Placeholder scan

No TBD/TODO in any code block. All function signatures, imports, and test assertions are fully specified.

### Type consistency

- `build_node_index` defined in `src/graph/edge_builders.py` (Task 2 code block) and also in `src/graph/build_graph.py` (Task 5 code block). **Fix:** `build_node_index` lives in `edge_builders.py` only. `build_graph.py` imports it:

```python
from src.graph.edge_builders import FEATURE_COLS, build_node_index
```

And `build_phase2b.py` imports `build_node_index` from `edge_builders`:

```python
from src.graph.edge_builders import (
    build_node_index,
    build_spatial_edges,
    build_economic_edges,
    build_schengen_edges,
)
```

The `build_graph.py` source in Task 5 Step 3 must remove its own `build_node_index` definition and import it instead. The corrected `build_graph.py` header:

```python
from src.graph.edge_builders import FEATURE_COLS, build_node_index
```

And delete the duplicate `build_node_index` function body from `build_graph.py`. The `HeteroData` assembly and temporal tensor functions remain as shown.

**Implementer note:** In Task 5 Step 3, when creating `build_graph.py`, omit the `build_node_index` function entirely from that file — it already exists in `edge_builders.py` from Task 2. The `build_graph.py` imports it via `from src.graph.edge_builders import FEATURE_COLS, build_node_index`. The `build_phase2b.py` also imports `build_node_index` from `edge_builders`.

- `built_temporal` fixture depends on `built_graph` in `test_built_temporal_year_axis` — remove that dependency; `built_temporal` is independent of `built_graph`.

Corrected test:

```python
def test_built_temporal_year_axis(built_temporal):
    gdp_2010 = built_temporal[0, :, 0]
    gdp_2023 = built_temporal[13, :, 0]
    assert not torch.allclose(gdp_2010, gdp_2023), "Temporal tensor appears static"
```
