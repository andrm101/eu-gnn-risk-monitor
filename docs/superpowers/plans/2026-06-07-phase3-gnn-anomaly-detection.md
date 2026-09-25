# Phase 3 — GNN Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a spatio-temporal GNN autoencoder on the 242-node EU NUTS2 heterogeneous graph and produce `nuts2_risk_scores.parquet` containing per-(region, year) anomaly scores for 2010–2023.

**Architecture:** `HeteroConv` (three `SAGEConv` modules, one per edge relation) aggregates spatial context per timestep; a single-layer `GRU` captures temporal dynamics across the 14-year sequence; a two-layer MLP decoder reconstructs the 19 node features; reconstruction MSE is the anomaly signal.

**Tech Stack:** Python 3.11, torch 2.9.0+cpu, torch-geometric 2.5.3, pandas 2.2, numpy 1.26, pytest 8.1.

---

## Context for subagents

Working directory: `EU-GNN-Risk-Monitor/` (project root).

**Inputs already on disk (Phase 2B outputs):**
- `data/processed/nuts2_graph.pt` — PyG `HeteroData`, 242 nodes, 3 edge types
- `data/processed/nuts2_temporal_features.pt` — `(14, 242, 19)` float32 tensor, years 2010–2023
- `data/processed/nuts2_node_index.parquet` — columns: `node_idx` (int), `nuts2_code` (str), `country_code` (str)

**Edge type keys** used throughout (dict keys in `edge_index_dict`):
```python
('region', 'spatial',  'region')   # 1020 directed edges
('region', 'economic', 'region')   # 1382 directed edges
('region', 'schengen', 'region')   # 922  directed edges
```

**Existing test suite:** `tests/test_graph.py` — 84 tests, all PASS. Do not modify it.

**Code conventions (CLAUDE.md):**
- No `inplace=True` in pandas — always reassign
- Type hints on all function signatures
- One-line docstrings only
- No magic numbers — use named constants

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/models/__init__.py` | Create | Empty package marker |
| `src/models/gnn_autoencoder.py` | Create | `SpatioTemporalEncoder`, `FeatureDecoder`, `GNNAutoencoder` |
| `src/models/train.py` | Create | Training loop + checkpoint saving |
| `src/models/score.py` | Create | Anomaly score computation + parquet serialisation |
| `train_phase3.py` | Create | Orchestrator: load → train → score → save |
| `tests/test_models.py` | Create | 4 unit tests + 6 integration tests |

---

## Task 1: Unit test scaffold + package marker

**Files:**
- Create: `src/models/__init__.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Create the package marker**

```python
# src/models/__init__.py
# (empty)
```

- [ ] **Step 2: Write the full test file**

```python
# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.gnn_autoencoder import GNNAutoencoder

ROOT          = Path(__file__).parents[1]
SCORES_PATH   = ROOT / "data" / "processed" / "nuts2_risk_scores.parquet"
GRAPH_PATH    = ROOT / "data" / "processed" / "nuts2_graph.pt"
TEMPORAL_PATH = ROOT / "data" / "processed" / "nuts2_temporal_features.pt"
NODE_IDX_PATH = ROOT / "data" / "processed" / "nuts2_node_index.parquet"

EDGE_TYPES = [
    ("region", "spatial",  "region"),
    ("region", "economic", "region"),
    ("region", "schengen", "region"),
]


def _tiny_edge_index_dict(n_nodes: int = 10) -> dict:
    edge_index = torch.randint(0, n_nodes, (2, 20))
    return {et: edge_index for et in EDGE_TYPES}


# ── Unit tests (no file I/O) ─────────────────────────────────────────────────

def test_encoder_output_shape():
    torch.manual_seed(0)
    model = GNNAutoencoder()
    x_seq = torch.randn(14, 10, 19)
    z = model.encode(x_seq, _tiny_edge_index_dict())
    assert z.shape == (14, 10, 32)


def test_decoder_output_shape():
    torch.manual_seed(0)
    model = GNNAutoencoder()
    z = torch.randn(14, 10, 32)
    x_hat = model.decode(z)
    assert x_hat.shape == (14, 10, 19)


def test_forward_no_nan():
    torch.manual_seed(0)
    model = GNNAutoencoder()
    x_seq = torch.randn(14, 10, 19)
    out = model(x_seq, _tiny_edge_index_dict())
    assert not out.isnan().any()


def test_loss_decreases():
    torch.manual_seed(0)
    model = GNNAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_seq = torch.randn(14, 10, 19)
    ei_dict = _tiny_edge_index_dict()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = F.mse_loss(model(x_seq, ei_dict), x_seq)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "Loss did not decrease over 5 steps"


# ── Integration fixture (requires train_phase3.py to have been run) ──────────

@pytest.fixture(scope="session")
def scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        pytest.skip("nuts2_risk_scores.parquet not found — run train_phase3.py first")
    return pd.read_parquet(SCORES_PATH)


# ── Integration tests ────────────────────────────────────────────────────────

def test_score_parquet_shape(scores: pd.DataFrame):
    assert len(scores) == 3388  # 242 × 14


def test_score_columns(scores: pd.DataFrame):
    assert {"nuts2_code", "country_code", "year", "anomaly_score",
            "peak_score", "peak_rank"}.issubset(scores.columns)


def test_anomaly_score_non_negative(scores: pd.DataFrame):
    assert (scores["anomaly_score"] >= 0).all()


def test_peak_rank_range(scores: pd.DataFrame):
    ranks = scores["peak_rank"].unique()
    assert ranks.min() == 1 and ranks.max() == 242


def test_years_covered(scores: pd.DataFrame):
    assert set(scores["year"].unique()) == set(range(2010, 2024))


def test_no_nan_scores(scores: pd.DataFrame):
    assert not scores["anomaly_score"].isna().any()
```

- [ ] **Step 3: Run tests to confirm expected failure**

```
pytest tests/test_models.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'src.models.gnn_autoencoder'`

This is correct. Do not fix it yet.

- [ ] **Step 4: Commit the scaffold**

```
git add src/models/__init__.py tests/test_models.py
git commit -m "test(phase3): unit + integration test scaffold for GNN autoencoder"
```

---

## Task 2: GNNAutoencoder model

**Files:**
- Create: `src/models/gnn_autoencoder.py`
- Test: `tests/test_models.py` (existing — 4 unit tests must PASS after this task)

- [ ] **Step 1: Create the model module**

```python
# src/models/gnn_autoencoder.py
"""Spatio-temporal GNN autoencoder for EU NUTS2 anomaly detection."""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

EDGE_TYPES: list[tuple[str, str, str]] = [
    ("region", "spatial",  "region"),
    ("region", "economic", "region"),
    ("region", "schengen", "region"),
]


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 19, hidden_channels: int = 64, latent_dim: int = 32):
        super().__init__()
        self.conv = HeteroConv(
            {et: SAGEConv(in_channels, hidden_channels) for et in EDGE_TYPES},
            aggr="sum",
        )
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=False,
        )

    def forward(self, x_seq: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        """Return latent embeddings (T, N, latent_dim) for all timesteps."""
        T = x_seq.shape[0]
        h_list = []
        for t in range(T):
            h_dict = self.conv({"region": x_seq[t]}, edge_index_dict)
            h_list.append(self.relu(h_dict["region"]))
        h_seq = torch.stack(h_list, dim=0)  # (T, N, hidden_channels)
        z, _ = self.gru(h_seq)              # (T, N, latent_dim)
        return z


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_channels: int = 64, out_channels: int = 19):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed features (T, N, out_channels)."""
        return self.mlp(z)


class GNNAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 19, hidden_channels: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = FeatureDecoder(latent_dim, hidden_channels, in_channels)
        self.hyperparams = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "latent_dim": latent_dim,
        }

    def encode(self, x_seq: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        """Return (T, N, latent_dim) latent embeddings."""
        return self.encoder(x_seq, edge_index_dict)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return (T, N, in_channels) reconstructed features."""
        return self.decoder(z)

    def forward(self, x_seq: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        """Return (T, N, in_channels) reconstruction."""
        return self.decode(self.encode(x_seq, edge_index_dict))
```

- [ ] **Step 2: Run the 4 unit tests**

```
pytest tests/test_models.py::test_encoder_output_shape tests/test_models.py::test_decoder_output_shape tests/test_models.py::test_forward_no_nan tests/test_models.py::test_loss_decreases -v
```

Expected: `4 passed` (integration tests SKIP — parquet not yet on disk, which is correct)

- [ ] **Step 3: Run the full suite to confirm no regression in Phase 2B tests**

```
pytest tests/ -v --tb=short
```

Expected: `84 passed` from `test_graph.py` + `4 passed` + `6 skipped` from `test_models.py`

- [ ] **Step 4: Commit**

```
git add src/models/gnn_autoencoder.py
git commit -m "feat(phase3): GNNAutoencoder — SpatioTemporalEncoder + FeatureDecoder"
```

---

## Task 3: Training and scoring modules

**Files:**
- Create: `src/models/train.py`
- Create: `src/models/score.py`

These modules contain no new tests — they are exercised end-to-end by the orchestrator in Task 4.

- [ ] **Step 1: Create the training module**

```python
# src/models/train.py
"""Training loop for GNNAutoencoder."""
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.gnn_autoencoder import GNNAutoencoder

PRINT_EVERY = 100


def train(
    model: GNNAutoencoder,
    x_seq: torch.Tensor,
    edge_index_dict: dict,
    checkpoint_path: Path,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> float:
    """Train model in-place, save checkpoint every PRINT_EVERY epochs + final. Returns final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    final_loss = float("inf")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat = model(x_seq, edge_index_dict)
        loss = F.mse_loss(x_hat, x_seq)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if epoch % PRINT_EVERY == 0 or epoch == epochs:
            print(f"  epoch {epoch:>4d}/{epochs}  loss={final_loss:.6f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "hyperparams": model.hyperparams,
                    "final_loss": final_loss,
                },
                checkpoint_path,
            )
    return final_loss
```

- [ ] **Step 2: Create the scoring module**

```python
# src/models/score.py
"""Compute per-(region, year) anomaly scores from a trained GNNAutoencoder."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.gnn_autoencoder import GNNAutoencoder

YEARS: list[int] = list(range(2010, 2024))


def compute_scores(
    model: GNNAutoencoder,
    x_seq: torch.Tensor,
    edge_index_dict: dict,
) -> np.ndarray:
    """Return anomaly score matrix, shape (N, T) — mean-squared reconstruction error per cell."""
    model.eval()
    with torch.no_grad():
        x_hat = model(x_seq, edge_index_dict)  # (T, N, 19)
    err = (x_hat - x_seq) ** 2                 # (T, N, 19)
    scores_TN = err.mean(dim=2).numpy()         # (T, N)
    return scores_TN.T                          # (N, T)


def build_scores_df(
    scores: np.ndarray,
    node_index: pd.DataFrame,
    years: list[int] = YEARS,
) -> pd.DataFrame:
    """Convert (N, T) score matrix to long-format DataFrame with peak_score and peak_rank."""
    N, T = scores.shape
    node_idxs = np.repeat(np.arange(N), T)
    year_vals = np.tile(years, N)
    score_flat = scores.flatten().astype(np.float32)

    code_ser = node_index.set_index("node_idx")["nuts2_code"]
    country_ser = node_index.set_index("node_idx")["country_code"]

    df = pd.DataFrame({
        "nuts2_code":    code_ser.loc[node_idxs].values,
        "country_code":  country_ser.loc[node_idxs].values,
        "year":          year_vals,
        "anomaly_score": score_flat,
    })

    peak = (
        df.groupby("nuts2_code")["anomaly_score"]
        .max()
        .rename("peak_score")
        .astype(np.float32)
    )
    df = df.join(peak, on="nuts2_code")

    rank = peak.rank(ascending=False, method="min").astype(int).rename("peak_rank")
    df = df.join(rank, on="nuts2_code")

    return df
```

- [ ] **Step 3: Run existing tests to confirm no regression**

```
pytest tests/test_models.py -v --tb=short
```

Expected: `4 passed`, `6 skipped` (unchanged from Task 2).

- [ ] **Step 4: Commit**

```
git add src/models/train.py src/models/score.py
git commit -m "feat(phase3): training loop + anomaly scoring modules"
```

---

## Task 4: Orchestrator + full pipeline run + integration tests

**Files:**
- Create: `train_phase3.py`

After running the orchestrator, `nuts2_risk_scores.parquet` will exist and the 6 integration tests in `tests/test_models.py` will unblock.

- [ ] **Step 1: Create the orchestrator**

```python
# train_phase3.py
"""Phase 3: train GNN autoencoder and score 242 EU NUTS2 regions for 2010–2023."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.gnn_autoencoder import GNNAutoencoder
from src.models.train import train
from src.models.score import compute_scores, build_scores_df

np.random.seed(42)
torch.manual_seed(42)

ROOT          = Path(__file__).parent
GRAPH_PATH    = ROOT / "data" / "processed" / "nuts2_graph.pt"
TEMPORAL_PATH = ROOT / "data" / "processed" / "nuts2_temporal_features.pt"
NODE_IDX_PATH = ROOT / "data" / "processed" / "nuts2_node_index.parquet"
MODEL_PATH    = ROOT / "data" / "processed" / "gnn_model.pt"
SCORES_PATH   = ROOT / "data" / "processed" / "nuts2_risk_scores.parquet"

YEARS: list[int] = list(range(2010, 2024))


def main() -> None:
    print("== Phase 3: GNN Anomaly Detection ==========================")

    graph      = torch.load(GRAPH_PATH, weights_only=False)
    x_seq      = torch.load(TEMPORAL_PATH, weights_only=False)  # (14, 242, 19)
    node_index = pd.read_parquet(NODE_IDX_PATH)

    edge_index_dict = {
        ("region", "spatial",  "region"): graph["region", "spatial",  "region"].edge_index,
        ("region", "economic", "region"): graph["region", "economic", "region"].edge_index,
        ("region", "schengen", "region"): graph["region", "schengen", "region"].edge_index,
    }

    print(f"  graph      : {x_seq.shape[1]} nodes, {x_seq.shape[0]} years, {x_seq.shape[2]} features")

    model = GNNAutoencoder(in_channels=19, hidden_channels=64, latent_dim=32)
    print(f"  training   : 500 epochs, Adam lr=1e-3 ...")
    final_loss = train(model, x_seq, edge_index_dict, MODEL_PATH, epochs=500)
    print(f"  final loss : {final_loss:.6f}")
    print(f"  checkpoint : {MODEL_PATH.name}")

    scores = compute_scores(model, x_seq, edge_index_dict)  # (242, 14)
    df = build_scores_df(scores, node_index, YEARS)
    df.to_parquet(SCORES_PATH, index=False)
    print(f"  scores     : {len(df)} rows -> {SCORES_PATH.name}")
    print("== Done =====================================================")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the orchestrator**

```
python train_phase3.py
```

Expected output (approx):
```
== Phase 3: GNN Anomaly Detection ==========================
  graph      : 242 nodes, 14 years, 19 features
  training   : 500 epochs, Adam lr=1e-3 ...
  epoch  100/500  loss=0.xxxxxx
  epoch  200/500  loss=0.xxxxxx
  epoch  300/500  loss=0.xxxxxx
  epoch  400/500  loss=0.xxxxxx
  epoch  500/500  loss=0.xxxxxx
  final loss : 0.xxxxxx
  checkpoint : gnn_model.pt
  scores     : 3388 rows -> nuts2_risk_scores.parquet
== Done =====================================================
```

Expected files created:
- `data/processed/gnn_model.pt`
- `data/processed/nuts2_risk_scores.parquet`

If the run fails with an import or tensor-shape error, stop and investigate before continuing.

- [ ] **Step 3: Run the full test suite**

```
pytest tests/ -v --tb=short
```

Expected: `84 passed` (Phase 2B) + `4 passed` + `6 passed` (Phase 3) = **94 passed, 0 skipped**.

If any integration test fails, check the parquet schema with:
```python
import pandas as pd
df = pd.read_parquet("data/processed/nuts2_risk_scores.parquet")
print(df.dtypes, df.shape, df.head())
```

- [ ] **Step 4: Commit**

```
git add train_phase3.py
git commit -m "feat(phase3): GNN autoencoder orchestrator — 94 tests PASS, risk scores produced"
```

---

## §5 — Integration test reference

Expected shape of `nuts2_risk_scores.parquet` after a successful run:

| Column | Dtype | Constraint |
|---|---|---|
| `nuts2_code` | object | 242 unique values |
| `country_code` | object | EU ISO codes |
| `year` | int64 | 2010–2023 (14 distinct) |
| `anomaly_score` | float32 | ≥ 0 |
| `peak_score` | float32 | ≥ 0 |
| `peak_rank` | int64 | 1–242 |

Total rows: 3,388 (242 × 14). All tests in §7 of the spec map directly to these constraints.
