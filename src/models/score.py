"""Compute per-(region, year) anomaly scores from a trained GNNAutoencoder."""
import numpy as np
import pandas as pd
import torch

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
