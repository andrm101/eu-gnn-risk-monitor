# Phase 2B — Graph Construction Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construct a heterogeneous, temporally-aware graph over 242 EU NUTS2 regions and serialize it as a PyTorch Geometric `HeteroData` object for direct consumption by Phase 3 GNN models.

**Architecture:** Two new source modules — `src/graph/edge_builders.py` (one function per edge type) and `src/graph/build_graph.py` (assembles the `HeteroData` object). `build_phase2b.py` orchestrates: loads Phase 2A outputs, builds graph, writes two artefacts. All 45 Phase 2A tests must continue passing.

**Tech Stack:** Python 3.11, pandas 2.2, geopandas 0.14, numpy 1.26, scikit-learn 1.4, torch 2.3.0+cpu, torch-geometric 2.5.3, pytest 8.1.

---

## 1. Graph Definition

### 1.1 Node type

Single node type `'region'`. One node per NUTS2 code in `nuts2_features.parquet` — exactly 242 nodes. Integer node index 0–241 maps to `nuts2_code` via `nuts2_node_index.parquet`.

### 1.2 Node features

The 19 numeric columns from `nuts2_features.parquet` (all `float32`; excludes `nuts2_code`, `country_code`, `archetype_id`):

| # | Column | Semantics |
|---|---|---|
| 0 | `gdp_per_capita_pps_latest` | Latest-year GDP per capita (EUR, PPS approx.) |
| 1 | `unemployment_latest` | Latest-year unemployment rate (%) |
| 2 | `net_migration_rate` | Latest-year net migration per 1,000 pop |
| 3 | `gdp_pc_z` | GDP per capita z-score (cross-sectional) |
| 4 | `unemployment_z` | Unemployment z-score |
| 5 | `gdp_growth_3yr` | 3-year GDP CAGR |
| 6 | `unemp_change_3yr` | 3-year unemployment change (pp) |
| 7 | `pop_growth_3yr` | 3-year population CAGR |
| 8 | `prosperity_gap` | (EU mean GDP − region GDP) / EU mean GDP |
| 9 | `stress_proxy` | unemployment_z − gdp_pc_z |
| 10 | `data_coverage_frac` | Fraction of panel years with data |
| 11 | `ihpi` | Invisible Hand Pressure Index |
| 12 | `schengen_member` | 1.0 if Schengen member in 2023 |
| 13 | `conflict_density` | Political violence events / 100k pop (2018–2023 mean) |
| 14 | `conflict_lag_1` | conflict_density lagged 1 year |
| 15 | `chips_flag` | 1.0 if EU Chips Act / IPCEI site |
| 16 | `sis` | Schengen Integration Score |
| 17 | `sas` | Strategic Autonomy Score |
| 18 | `cei` | Conflict Exposure Index |

Node feature tensor: `data['region'].x` — shape `(242, 19)`, float32 — snapshot from the latest available year (2022 for most regions).

### 1.3 Temporal node features

The graph topology is **fixed** (borders and economic similarity do not change year-to-year). Node features **evolve** over 2010–2023 (14 years). A separate temporal tensor is produced from `nuts2_panel.parquet` joined with the composite indices, providing the full time series for Phase 3 temporal GNN models.

Temporal tensor: `nuts2_temporal_features.pt` — shape `(14, 242, 19)`, float32 — year axis ordered 2010, 2011, …, 2023. Years or regions with no panel data are filled with the column median for that year.

### 1.4 Edge types

Three undirected relation types, all stored as `edge_index` tensors of shape `(2, E)`:

All edge types store both (i,j) and (j,i) in `edge_index` (standard PyG undirected convention).

| Relation key | Definition | Expected \|E\| (directed, both ways) |
|---|---|---|
| `('region', 'spatial', 'region')` | Shared land border in GISCO NUTS2 2021 shapefile | 800–1400 |
| `('region', 'economic', 'region')` | k=4 nearest neighbours in StandardScaled 19-feature space | 968–1936 |
| `('region', 'schengen', 'region')` | Spatial neighbours where both `country_code` ∈ `SCHENGEN_2023` | ≤ spatial \|E\| |

No edge features. Edge type label is the sole relational signal.

---

## 2. Edge Construction

### 2.1 Spatial edges (`src/graph/edge_builders.py::build_spatial_edges`)

```
GISCO GeoJSON (already present: data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson)
  → filter to 242 canonical NUTS_ID codes
  → geopandas .touches() predicate on all pairs
  → deduplicate to undirected (keep i < j pairs only, then add reverse)
  → map nuts2_code → integer node index
  → return edge_index tensor (2, E_spatial)
```

Island / overseas NUTS2 regions (FRY1–FRY5, ES70, PT20, PT30) have no continental neighbours — they get zero spatial edges. They are still graph nodes and receive economic kNN and (if applicable) Schengen edges.

### 2.2 Economic kNN edges (`build_economic_edges`)

```
nuts2_features.parquet (19 numeric columns)
  → StandardScaler (fit on 242 regions)
  → pairwise Euclidean distance (242 × 242 matrix)
  → for each node: select k=4 nearest neighbours (exclude self)
  → symmetrise: add reverse edges (B→A if A→B)
  → deduplicate
  → return edge_index tensor (2, E_economic)
```

kNN is computed in scaled feature space; ties broken by lower node index.

### 2.3 Schengen edges (`build_schengen_edges`)

```
spatial edge_index (from §2.1)
  → for each (i, j): look up country_code[i], country_code[j]
  → keep edge if both country codes ∈ SCHENGEN_2023
  → return edge_index tensor (2, E_schengen)
```

`SCHENGEN_2023` is imported from `src.data.enrich_features` (already defined in Phase 2A). No new data required.

---

## 3. Output Artefacts

### 3.1 `data/processed/nuts2_graph.pt`

PyG `HeteroData` object serialised with `torch.save`. Schema:

```python
data['region'].x            # (242, 19) float32 — 2022 snapshot node features
data['region'].nuts2_code   # list[str] len 242 — ordered node identifiers
data['region'].country_code # list[str] len 242 — ISO 2-letter country codes

data['region', 'spatial',  'region'].edge_index  # (2, E_s)
data['region', 'economic', 'region'].edge_index  # (2, E_e)
data['region', 'schengen', 'region'].edge_index  # (2, E_sc)
```

### 3.2 `data/processed/nuts2_temporal_features.pt`

`torch.Tensor` shape `(14, 242, 19)`, float32. Axis 0 = years 2010–2023 in order. Axis 1 = node index matching `nuts2_graph.pt`. Axis 2 = same 19 features in same column order.

### 3.3 `data/processed/nuts2_node_index.parquet`

Two-column DataFrame (`node_idx` int, `nuts2_code` str, `country_code` str) for interpretability in Phase 3 outputs and dashboards.

---

## 4. File Map

| File | Action | Responsibility |
|---|---|---|
| `src/graph/__init__.py` | Create | Empty package marker |
| `src/graph/edge_builders.py` | Create | `build_spatial_edges`, `build_economic_edges`, `build_schengen_edges` |
| `src/graph/build_graph.py` | Create | `build_hetero_graph`, `build_temporal_tensor` |
| `build_phase2b.py` | Create | Orchestrator: load → build → save |
| `tests/test_graph.py` | Create | Graph invariant tests |
| `environment.yml` | Modify | Add `pip:` block with torch + torch-geometric |

---

## 5. Tests (`tests/test_graph.py`)

All tests load from `data/processed/nuts2_graph.pt` and skip if the file is missing (same pattern as Phase 1/2A).

```python
def test_node_count(graph):
    assert graph['region'].x.shape[0] == 242

def test_feature_dim(graph):
    assert graph['region'].x.shape[1] == 19

def test_no_nan_features(graph):
    assert not graph['region'].x.isnan().any()

def test_spatial_edge_count(graph):
    # Both (i,j) and (j,i) stored — ~500 undirected borders × 2
    E = graph['region', 'spatial', 'region'].edge_index.shape[1]
    assert 800 <= E <= 1400, f"Spatial edge count {E} out of expected range"

def test_economic_edge_count(graph):
    # 242 × k=4 directed kNN, symmetrised → between 968 (all mutual) and 1936 (none mutual)
    E = graph['region', 'economic', 'region'].edge_index.shape[1]
    assert 968 <= E <= 1936, f"Economic edge count {E} out of expected range"

def test_schengen_subset_of_spatial(graph):
    E_sc = graph['region', 'schengen', 'region'].edge_index.shape[1]
    E_s  = graph['region', 'spatial',  'region'].edge_index.shape[1]
    assert E_sc <= E_s

def test_no_self_loops(graph):
    for rel in [('region','spatial','region'), ('region','economic','region'),
                ('region','schengen','region')]:
        ei = graph[rel].edge_index
        assert (ei[0] != ei[1]).all(), f"Self-loop found in {rel}"

def test_edge_indices_in_range(graph):
    for rel in [('region','spatial','region'), ('region','economic','region'),
                ('region','schengen','region')]:
        ei = graph[rel].edge_index
        assert ei.max() <= 241 and ei.min() >= 0

def test_de_fr_schengen_edge(graph, node_index):
    """DE and FR are Schengen neighbours — at least one DE↔FR edge must exist."""
    de = set(node_index[node_index['country_code'] == 'DE']['node_idx'])
    fr = set(node_index[node_index['country_code'] == 'FR']['node_idx'])
    ei = graph['region', 'schengen', 'region'].edge_index
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

---

## 6. Dependencies

Add to `environment.yml` under `pip:`:

```yaml
  - pip:
    - torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
    - torch-geometric==2.5.3
```

`scikit-learn` (StandardScaler) is already present. `geopandas` (spatial join) is already present. No other new dependencies.

---

## 7. Methodological Notes

- **Overseas NUTS2 regions** (FRY1–5, ES70, PT20, PT30): no spatial or Schengen edges. Connected to the graph solely via economic kNN. This is acceptable — their risk profiles are structurally distinct from continental neighbours anyway.
- **Temporal coverage gaps**: not all 242 NUTS2 regions have panel data for all 14 years. Missing node-year cells in `nuts2_temporal_features.pt` are filled with the column median for that year. Flag this as a methodological limitation in the Phase 3 report.
- **Schengen as a graph feature vs. node feature**: `schengen_member` is already encoded as a node feature (SIS). The Schengen edge type encodes it as a *structural relationship* — allowing the GNN to learn that spillovers across open borders differ from spillovers across controlled borders. These are complementary, not redundant.
- **Static graph justification**: NUTS2 boundaries changed in 2016 and 2021; the 2021 vintage (GISCO) is used throughout. Economic kNN is computed once on the 2022 feature snapshot. Using a fixed topology is standard in NUTS-level GNN literature and avoids label alignment problems across temporal editions.
