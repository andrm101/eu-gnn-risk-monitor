"""Phase 3: train GNN autoencoder and score 242 EU NUTS2 regions for 2010–2023."""
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.models.gnn_autoencoder import GNNAutoencoder, EDGE_TYPES
from src.models.train import train
from src.models.score import compute_scores, build_scores_df, YEARS

np.random.seed(42)
torch.manual_seed(42)

ROOT          = Path(__file__).parent
GRAPH_PATH    = ROOT / "data" / "processed" / "nuts2_graph.pt"
TEMPORAL_PATH = ROOT / "data" / "processed" / "nuts2_temporal_features.pt"
NODE_IDX_PATH = ROOT / "data" / "processed" / "nuts2_node_index.parquet"
MODEL_PATH    = ROOT / "data" / "processed" / "gnn_model.pt"
SCALER_PATH   = ROOT / "data" / "processed" / "nuts2_feature_scaler.pkl"
SCORES_PATH   = ROOT / "data" / "processed" / "nuts2_risk_scores.parquet"


def main() -> None:
    print("== Phase 3: GNN Anomaly Detection ==========================")

    graph      = torch.load(GRAPH_PATH, weights_only=False)
    x_seq      = torch.load(TEMPORAL_PATH, weights_only=False)  # (14, 242, 19)
    node_index = pd.read_parquet(NODE_IDX_PATH)

    edge_index_dict = {et: graph[et].edge_index for et in EDGE_TYPES}

    T, N, F = x_seq.shape
    print(f"  graph      : {N} nodes, {T} years, {F} features")

    # Fit per-feature StandardScaler across all (T×N) observations so all 19
    # features contribute equally to the MSE loss (raw GDP ≈30k otherwise dominates).
    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(x_seq.numpy().reshape(-1, F)).reshape(T, N, F)
    x_seq_scaled = torch.tensor(x_scaled_np, dtype=torch.float32)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  scaler     : fitted on {T*N} obs, saved -> {SCALER_PATH.name}")

    model = GNNAutoencoder(in_channels=19, hidden_channels=64, latent_dim=32)
    print(f"  training   : 500 epochs, Adam lr=1e-3 ...")
    final_loss = train(model, x_seq_scaled, edge_index_dict, MODEL_PATH, epochs=500)
    print(f"  final loss : {final_loss:.6f}")
    print(f"  checkpoint : {MODEL_PATH.name}")

    scores = compute_scores(model, x_seq_scaled, edge_index_dict)  # (242, 14)
    df = build_scores_df(scores, node_index, YEARS)
    df.to_parquet(SCORES_PATH, index=False)
    print(f"  scores     : {len(df)} rows -> {SCORES_PATH.name}")
    print("== Done =====================================================")


if __name__ == "__main__":
    main()
