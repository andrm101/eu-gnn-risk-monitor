# Phase 3 — GNN Anomaly Detection Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a spatio-temporal GNN autoencoder over the 242-node EU NUTS2 heterogeneous graph to produce per-(region, year) anomaly scores covering 2010–2023, saved as `nuts2_risk_scores.parquet` for downstream interface phases.

**Architecture:** `SpatioTemporalEncoder` (HeteroConv + GRU) → `FeatureDecoder` (MLP) → reconstruction loss → anomaly score = per-cell MSE. Full-graph training on CPU, no time-split, 500 epochs.

**Tech Stack:** Python 3.11, torch 2.9.0+cpu, torch-geometric 2.5.3, pandas 2.2, numpy 1.26, pytest 8.1.

---

## 1. Inputs

All inputs are the serialised artefacts produced by Phase 2B. No new data ingestion.

| Artefact | Path | Usage |
|---|---|---|
| Graph topology + snapshot features | `data/processed/nuts2_graph.pt` | Edge indices for all 3 relation types; 2022 snapshot `x` (not used in training — temporal tensor replaces it) |
| Temporal feature tensor | `data/processed/nuts2_temporal_features.pt` | Shape `(14, 242, 19)` float32; axis-0 = years 2010–2023 |
| Node index | `data/processed/nuts2_node_index.parquet` | Maps integer `node_idx` → `nuts2_code`, `country_code` |

---

## 2. Model Architecture

### 2.1 `SpatioTemporalEncoder`

**Step 1 — Per-timestep spatial aggregation (HeteroConv):**

For each of the 14 timesteps `t`, apply one shared `HeteroConv` layer containing three `SAGEConv` instances (one per relation type: `spatial`, `economic`, `schengen`):

```
X_t ∈ R^{242 × 19}  →  HeteroConv(edge_indices_spatial, edge_indices_economic, edge_indices_schengen)  →  H_t ∈ R^{242 × 64}
```

The `HeteroConv` output for node `n` at time `t` is the sum of the three relation-specific SAGEConv outputs. ReLU applied after aggregation.

**Step 2 — Temporal sequence (GRU):**

Stack `H_0 … H_13` → shape `(14, 242, 64)`. Treat nodes as the batch dimension and years as the sequence dimension, so GRU sees input `(seq_len=14, batch=242, input_size=64)`:

```
GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=False)
→  Z ∈ R^{14 × 242 × 32}
```

`Z[t, n, :]` is the latent embedding for region `n` at year `t`.

### 2.2 `FeatureDecoder`

Two-layer MLP, applied identically to every `(t, n)` cell:

```
Linear(32 → 64) → ReLU → Linear(64 → 19)
→  X̂_t ∈ R^{242 × 19}
```

No graph structure in the decoder.

### 2.3 `GNNAutoencoder`

Thin wrapper combining encoder and decoder:

```python
class GNNAutoencoder(torch.nn.Module):
    def __init__(self, in_channels=19, hidden_channels=64, latent_dim=32): ...
    def encode(self, x_seq, edge_index_dict) -> torch.Tensor: ...   # (14, 242, 32)
    def decode(self, z) -> torch.Tensor: ...                        # (14, 242, 19)
    def forward(self, x_seq, edge_index_dict) -> torch.Tensor: ...  # (14, 242, 19)
```

`edge_index_dict` matches the PyG `HeteroData` key format:
```python
{
    ('region', 'spatial',  'region'): edge_index_spatial,   # (2, E_s)
    ('region', 'economic', 'region'): edge_index_economic,  # (2, E_e)
    ('region', 'schengen', 'region'): edge_index_schengen,  # (2, E_sc)
}
```

---

## 3. Training

### 3.1 Loss

Mean squared error over all (timestep, node, feature) cells:

```
L = MSE(X̂, X_seq)   where X_seq ∈ R^{14 × 242 × 19}
```

No regularisation term. If training loss plateaus above 0.5 after 200 epochs, reduce lr to 5e-4.

### 3.2 Hyperparameters

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Epochs | 500 |
| Batch | Full graph (no mini-batching) |
| Hidden channels | 64 |
| Latent dim | 32 |
| Random seed | `torch.manual_seed(42)`, `np.random.seed(42)` |

### 3.3 Train/validation split

None. 14 years is too short to hold out a temporal slice without degrading the GRU's sequence modelling. The reconstruction error at convergence is the anomaly signal — no validation loss is needed.

### 3.4 Checkpoint

Save `gnn_model.pt` every 100 epochs and at the end of training. Contents:

```python
{
    "epoch": int,
    "model_state_dict": model.state_dict(),
    "hyperparams": {"in_channels": 19, "hidden_channels": 64, "latent_dim": 32},
    "final_loss": float,
}
```

Only the most recent checkpoint is kept; overwrite each time.

---

## 4. Anomaly Scoring

After training, run one forward pass (no gradient) to get `X̂ ∈ R^{14 × 242 × 19}`.

**Per-(node, year) anomaly score:**

```
score[n, t] = mean_f((X̂[t, n, f] - X[t, n, f])^2)   for f in 0..18
```

Shape: `(242, 14)`.

**Per-node peak score:**

```
peak_score[n] = max_t(score[n, t])
```

**Peak rank:** rank regions by `peak_score` descending; rank 1 = highest risk.

---

## 5. Output Artefacts

### 5.1 `data/processed/gnn_model.pt`

Trained model checkpoint (see §3.4).

### 5.2 `data/processed/nuts2_risk_scores.parquet`

Long-format DataFrame, one row per (region, year):

| Column | Type | Description |
|---|---|---|
| `nuts2_code` | str | NUTS2 identifier |
| `country_code` | str | ISO 2-letter country code |
| `year` | int | 2010–2023 |
| `anomaly_score` | float32 | Per-(node, year) MSE across 19 features |
| `peak_score` | float32 | Max anomaly score for this region across all years |
| `peak_rank` | int | Region rank by peak_score (1 = highest risk) |

Total rows: 242 × 14 = 3,388.

---

## 6. File Map

| File | Action | Responsibility |
|---|---|---|
| `src/models/__init__.py` | Create | Empty package marker |
| `src/models/gnn_autoencoder.py` | Create | `SpatioTemporalEncoder`, `FeatureDecoder`, `GNNAutoencoder` |
| `src/models/train.py` | Create | Training loop; saves checkpoint to `data/processed/gnn_model.pt` |
| `src/models/score.py` | Create | Load checkpoint, run forward pass, compute score matrix, write parquet |
| `train_phase3.py` | Create | Orchestrator: load artefacts → train → score → save |
| `tests/test_models.py` | Create | Unit + integration tests |

---

## 7. Tests (`tests/test_models.py`)

All tests use `torch.manual_seed(0)` for determinism. Heavy fixtures (full training run) are skipped if `data/processed/nuts2_graph.pt` is missing.

### Unit tests (no file I/O)

```python
def test_encoder_output_shape():
    # Smoke-test encoder with random input — no real graph needed
    model = GNNAutoencoder()
    x_seq = torch.randn(14, 10, 19)   # 10 fake nodes
    edge_index = torch.randint(0, 10, (2, 20))
    edge_index_dict = {
        ('region', 'spatial',  'region'): edge_index,
        ('region', 'economic', 'region'): edge_index,
        ('region', 'schengen', 'region'): edge_index,
    }
    z = model.encode(x_seq, edge_index_dict)
    assert z.shape == (14, 10, 32)

def test_decoder_output_shape():
    model = GNNAutoencoder()
    z = torch.randn(14, 10, 32)
    x_hat = model.decode(z)
    assert x_hat.shape == (14, 10, 19)

def test_forward_no_nan():
    model = GNNAutoencoder()
    x_seq = torch.randn(14, 10, 19)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_index_dict = {
        ('region', 'spatial',  'region'): edge_index,
        ('region', 'economic', 'region'): edge_index,
        ('region', 'schengen', 'region'): edge_index,
    }
    out = model(x_seq, edge_index_dict)
    assert not out.isnan().any()

def test_loss_decreases():
    # 5-step training loop on tiny random graph — loss must fall
    torch.manual_seed(0)
    model = GNNAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_seq = torch.randn(14, 10, 19)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_index_dict = {
        ('region', 'spatial',  'region'): edge_index,
        ('region', 'economic', 'region'): edge_index,
        ('region', 'schengen', 'region'): edge_index,
    }
    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = model(x_seq, edge_index_dict)
        loss = torch.nn.functional.mse_loss(out, x_seq)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "Loss did not decrease over 5 steps"
```

### Integration tests (require `nuts2_graph.pt` + `nuts2_temporal_features.pt`)

```python
def test_score_parquet_shape(scores):
    assert len(scores) == 3388   # 242 × 14

def test_score_columns(scores):
    assert {"nuts2_code", "country_code", "year", "anomaly_score",
            "peak_score", "peak_rank"}.issubset(scores.columns)

def test_anomaly_score_non_negative(scores):
    assert (scores["anomaly_score"] >= 0).all()

def test_peak_rank_range(scores):
    ranks = scores["peak_rank"].unique()
    assert ranks.min() == 1 and ranks.max() == 242

def test_years_covered(scores):
    assert set(scores["year"].unique()) == set(range(2010, 2024))

def test_no_nan_scores(scores):
    assert not scores["anomaly_score"].isna().any()
```

---

## 8. Methodological Notes

- **No temporal split**: with only 14 observations per node, holding out years for validation would leave the GRU with too little sequence context. Reconstruction error at convergence is the anomaly signal; this is standard practice in unsupervised graph anomaly detection (DOMINANT, Fan et al., 2020).
- **Shared HeteroConv weights across timesteps**: the encoder applies the same HeteroConv weights at every year. This enforces that the spatial message-passing function is stationary, which is reasonable given fixed graph topology.
- **Anomaly score interpretation**: high MSE indicates the model cannot reconstruct a region's features from its neighbourhood context — i.e., the region behaves unlike its graph neighbours. This captures both structural outliers (persistently anomalous) and change-point events (sudden divergence in a given year).
- **Schengen edge type in GNN**: separate SAGEConv for Schengen edges allows the model to learn distinct spillover dynamics across open vs. controlled borders, complementing the `schengen_member` node feature already present.
- **Limitation**: anomaly scores are relative to the training distribution (2010–2023). A region that is consistently extreme but stable will score lower than one that suddenly diverges from its neighbours. Report this nuance in the Phase 4 interface.
