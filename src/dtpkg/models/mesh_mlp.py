"""MeSH-only multilayer perceptron for symmetric drug-pair features."""
import torch
import torch.nn as nn

class MLP_DDI(nn.Module):
    """
    A simple, flexible Multi-Layer Perceptron for DDI prediction.
    Input: symmetric [a+b, |a-b|] pair features (256 dimensions for 128-D drugs).
    Output: probability of interaction (0–1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        use_batchnorm=True,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final layer: binary classification
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x).squeeze(1)
        return torch.sigmoid(logits)
