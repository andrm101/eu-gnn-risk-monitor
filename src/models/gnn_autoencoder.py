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
