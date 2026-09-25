import pytest
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.gnn_autoencoder import GNNAutoencoder, EDGE_TYPES

ROOT        = Path(__file__).parents[1]
SCORES_PATH = ROOT / "data" / "processed" / "nuts2_risk_scores.parquet"
SCALER_PATH = ROOT / "data" / "processed" / "nuts2_feature_scaler.pkl"


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


def test_scaler_file_exists():
    if not SCALER_PATH.exists():
        pytest.skip("nuts2_feature_scaler.pkl not found — run train_phase3.py first")
    import pickle
    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)
    assert hasattr(scaler, "mean_") and scaler.mean_.shape == (19,)


def test_anomaly_scores_in_scaled_range(scores: pd.DataFrame):
    # Scores are MSE over standardised features — should be O(1), not O(10^7)
    assert scores["anomaly_score"].max() < 100, (
        f"Peak anomaly score {scores['anomaly_score'].max():.2f} suggests un-normalised features"
    )
