"""Scalar topology weighting on the existing standardized raw pair features.

The input is the symmetric raw-concat pair vector after the *common*, frozen
StandardScaler fitted for the unweighted raw-concat control. Its layout is
``[MeSH sum, topology sum, MeSH absolute difference, topology absolute
difference]``. Applying alpha here prevents a separately fitted scaler from
canceling positive fixed weights.
"""

import math

import torch
from torch import nn

from dtpkg.models.latent_gate import Classifier


class WeightedPairHead(nn.Module):
    """Classify standardized raw pairs after scalar topology weighting.

    ``learned=False`` stores alpha as a fixed buffer. ``learned=True`` makes it
    one unconstrained parameter shared by all topology coordinates and pairs.
    The returned logits retain shape ``(batch, 1)`` for the baseline trainer's
    subsequent squeeze, including when the last batch contains one pair.
    """

    def __init__(self, bio_dim=128, topo_dim=12, *, alpha_init=4.0,
                 learned=False, hidden=(256, 128), dropout=0.1):
        super().__init__()
        if bio_dim <= 0 or topo_dim <= 0:
            raise ValueError("bio_dim and topo_dim must be positive")
        if not math.isfinite(alpha_init):
            raise ValueError("alpha_init must be finite")
        self.bio_dim = int(bio_dim)
        self.topo_dim = int(topo_dim)
        self.learned = bool(learned)
        self._init_kwargs = dict(bio_dim=bio_dim, topo_dim=topo_dim,
                                 alpha_init=float(alpha_init), learned=bool(learned),
                                 hidden=tuple(hidden), dropout=dropout)
        alpha = torch.tensor(float(alpha_init), dtype=torch.float32)
        if self.learned:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.classifier = Classifier(2 * (bio_dim + topo_dim),
                                     hidden=hidden, dropout=dropout)

    @property
    def alpha_value(self):
        """Current scalar value for checkpoint metadata and reporting."""
        return float(self.alpha.detach().cpu())

    def new_instance(self):
        """Create fresh classifier weights and restore the declared initial alpha."""
        return type(self)(**self._init_kwargs)

    def weight_pair(self, values):
        """Multiply only topology sum and absolute-difference coordinates."""
        expected = 2 * (self.bio_dim + self.topo_dim)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"values must have shape (batch, {expected}); got {tuple(values.shape)}")
        split = self.bio_dim + self.topo_dim
        return torch.cat((
            values[:, :self.bio_dim],
            self.alpha * values[:, self.bio_dim:split],
            values[:, split:split + self.bio_dim],
            self.alpha * values[:, split + self.bio_dim:],
        ), dim=1)

    def forward(self, values):
        return self.classifier(self.weight_pair(values)).reshape(-1, 1)
