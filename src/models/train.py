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
