"""Input-level fusion controls, separate from the saved latent-fusion models.

Both models use the fusion trainer's pairwise call signature. Neither has a
learned modality gate, so the second return value is always ``(None, None)``.
"""

import torch
from torch import nn

from dtpkg.models.latent_gate import Classifier, Encoder


def _check_features(values, width, name):
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (batch, {width}); got {tuple(values.shape)}")


def _symmetric_pair(first, second):
    if first.shape != second.shape:
        raise ValueError("The two drug representations must have the same shape")
    return torch.cat([first + second, torch.abs(first - second)], dim=1)


def _modality_dropout(values, probability, training):
    # Match LatentGateModel's modality dropout: one decision for this branch
    # and endpoint batch, rather than independent elementwise dropout.
    if training and probability > 0.0 and torch.rand(()) < probability:
        return torch.zeros_like(values)
    return values


class RawConcatModel(nn.Module):
    """Concatenate raw modalities per drug and classify their symmetric pair.

    There is no modality encoder, latent projection, or gate. Hidden layers
    belong only to the pair classifier. Optional fixed standardization acts
    after symmetric pairing; its statistics must be fitted by the caller on
    training data. With ``topo_dim=0``, topology is ignored for a raw MeSH-only
    control with the same training interface.
    """

    def __init__(self, bio_dim, topo_dim, head_hidden=(256, 128), dropout=0.1,
                 pair_mean=None, pair_scale=None):
        super().__init__()
        if bio_dim <= 0 or topo_dim < 0:
            raise ValueError("bio_dim must be positive and topo_dim nonnegative")
        self.bio_dim = bio_dim
        self.topo_dim = topo_dim
        self.mode = "sym"
        self._init_kwargs = dict(bio_dim=bio_dim, topo_dim=topo_dim,
                                 head_hidden=tuple(head_hidden), dropout=dropout)
        pair_dim = 2 * (bio_dim + topo_dim)
        mean = (torch.zeros(pair_dim) if pair_mean is None else
                torch.as_tensor(pair_mean).detach().to(device="cpu", dtype=torch.float32).clone())
        scale = (torch.ones(pair_dim) if pair_scale is None else
                 torch.as_tensor(pair_scale).detach().to(device="cpu", dtype=torch.float32).clone())
        if mean.shape != (pair_dim,) or scale.shape != (pair_dim,):
            raise ValueError(f"pair_mean and pair_scale must have shape ({pair_dim},)")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("pair_mean and pair_scale must be finite")
        if not (scale > 0).all():
            raise ValueError("pair_scale must be strictly positive")
        self.register_buffer("pair_mean", mean)
        self.register_buffer("pair_scale", scale)
        self.classifier = Classifier(pair_dim,
                                     hidden=head_hidden, dropout=dropout)

    def new_instance(self):
        """Create fresh weights while retaining fixed pair normalization."""
        return type(self)(**self._init_kwargs,
                          pair_mean=self.pair_mean, pair_scale=self.pair_scale)

    def fuse_one(self, bio, topo, drop_bio_prob=0.0, drop_topo_prob=0.0):
        _check_features(bio, self.bio_dim, "bio")
        if self.topo_dim == 0:
            return _modality_dropout(bio, drop_bio_prob, self.training), None
        _check_features(topo, self.topo_dim, "topo")
        if bio.shape[0] != topo.shape[0]:
            raise ValueError("bio and topo must have the same batch size")
        bio = _modality_dropout(bio, drop_bio_prob, self.training)
        topo = _modality_dropout(topo, drop_topo_prob, self.training)
        return torch.cat([bio, topo], dim=1), None

    def make_pair(self, first, second):
        return (_symmetric_pair(first, second) - self.pair_mean) / self.pair_scale

    def forward(self, bio1, topo1, bio2, topo2,
                drop_bio_prob=0.0, drop_topo_prob=0.0):
        first, _ = self.fuse_one(bio1, topo1, drop_bio_prob, drop_topo_prob)
        second, _ = self.fuse_one(bio2, topo2, drop_bio_prob, drop_topo_prob)
        return self.classifier(self.make_pair(first, second)).view(-1), (None, None)


class EncodedMeshModel(nn.Module):
    """MeSH-only control with the fusion model's encoder and classifier widths.

    Topology arguments are accepted for trainer compatibility and ignored.
    The ``bio_encoder`` and ``classifier`` state dictionaries have the same
    keys and shapes as a LatentGateModel with matching settings, allowing a
    caller to copy their common initialization explicitly.
    """

    def __init__(self, bio_dim, latent_dim=128, enc_hidden=(256,),
                 head_hidden=(256, 128), dropout=0.1):
        super().__init__()
        if bio_dim <= 0 or latent_dim <= 0:
            raise ValueError("bio_dim and latent_dim must be positive")
        self.bio_dim = bio_dim
        self.latent_dim = latent_dim
        self.mode = "sym"
        self._init_kwargs = dict(bio_dim=bio_dim, latent_dim=latent_dim,
                                 enc_hidden=tuple(enc_hidden),
                                 head_hidden=tuple(head_hidden), dropout=dropout)
        self.bio_encoder = Encoder(bio_dim, latent_dim, hidden=enc_hidden, dropout=dropout)
        self.classifier = Classifier(2 * latent_dim, hidden=head_hidden, dropout=dropout)

    def new_instance(self):
        """Create fresh parameters; matching a reference is the caller's job."""
        return type(self)(**self._init_kwargs)

    def fuse_one(self, bio, topo, drop_bio_prob=0.0, drop_topo_prob=0.0):
        _check_features(bio, self.bio_dim, "bio")
        encoded = self.bio_encoder(bio)
        return _modality_dropout(encoded, drop_bio_prob, self.training), None

    def make_pair(self, first, second):
        return _symmetric_pair(first, second)

    def forward(self, bio1, topo1, bio2, topo2,
                drop_bio_prob=0.0, drop_topo_prob=0.0):
        first, _ = self.fuse_one(bio1, topo1, drop_bio_prob, drop_topo_prob)
        second, _ = self.fuse_one(bio2, topo2, drop_bio_prob, drop_topo_prob)
        return self.classifier(self.make_pair(first, second)).view(-1), (None, None)
