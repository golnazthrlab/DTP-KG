"""Biological/topological latent gating with symmetric drug-pair features."""

import torch
import torch.nn as nn


# ---------------------------
# Encoder network for bio/topo features
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(256,), dropout=0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # Map features to latent space
        return self.norm(self.net(x))


# ---------------------------
# Simple binary classifier head
# ---------------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Output raw logit (not passed through sigmoid)
        return self.net(x).squeeze(-1)


# ---------------------------
# Main model: fuse bio + topo features per drug, predict DDI per pair
# ---------------------------
class LatentGateModel(nn.Module):
    def __init__(
        self,
        bio_dim,
        topo_dim,
        latent_dim=128,
        fusion_mode="sym",
        enc_hidden=(256,),
        head_hidden=(256, 128),
        dropout=0.1,
        gate_scalar=False,
        gate_bias_init=0.7,
        gate_mode=None,
    ):
        super().__init__()

        # ``gate_scalar`` is retained for older callers/checkpoints. Explicit
        # gate_mode selects the within-drug fusion; fusion_mode still selects
        # the separate representation of the two drugs in a pair.
        if gate_mode is None:
            gate_mode = "adaptive_scalar" if gate_scalar else "adaptive_vector"
        if gate_mode not in {"adaptive_vector", "adaptive_scalar", "fixed_half", "concat_projection"}:
            raise ValueError(f"Unknown gate_mode: {gate_mode!r}")
        if gate_scalar and gate_mode != "adaptive_scalar":
            raise ValueError("gate_scalar=True conflicts with the explicit gate_mode")
        self.gate_mode = gate_mode
        self._init_kwargs = dict(bio_dim=bio_dim, topo_dim=topo_dim, latent_dim=latent_dim,
            fusion_mode=fusion_mode, enc_hidden=enc_hidden, head_hidden=head_hidden,
            dropout=dropout, gate_scalar=gate_scalar, gate_bias_init=gate_bias_init,
            gate_mode=gate_mode)

        # "sym"    : [z1 + z2, |z1 - z2|]  -- permutation-invariant (default)
        # "concat" : [z1, z2]              -- ordered; kept only as the ablation baseline
        assert fusion_mode in ("concat", "sym"), fusion_mode

        # Encoders for biological and topological features
        self.bio_encoder = Encoder(bio_dim, latent_dim, hidden=enc_hidden, dropout=dropout)
        self.topo_encoder = Encoder(topo_dim, latent_dim, hidden=enc_hidden, dropout=dropout)

        if gate_mode.startswith("adaptive_"):
            gate_out_dim = 1 if gate_mode == "adaptive_scalar" else latent_dim
            self.gate = nn.Linear(2 * latent_dim, gate_out_dim)
            with torch.no_grad():
                self.gate.bias.fill_(torch.logit(torch.tensor(gate_bias_init)))
        elif gate_mode == "concat_projection":
            # Project per-drug concatenated encodings back to the common width,
            # keeping symmetric pairing and the classifier identical in shape.
            self.fusion_projection = nn.Linear(2 * latent_dim, latent_dim)

        # Classifier head (takes combined drug pair); both modes yield 2 * latent_dim
        pair_input_dim = 2 * latent_dim
        self.classifier = Classifier(pair_input_dim, hidden=head_hidden, dropout=dropout)

        self.mode = fusion_mode

    def new_instance(self):
        """Construct under the caller's seed; preserve the declared gate-bias prior."""
        return type(self)(**self._init_kwargs)

    # ---------------------------
    # Fuse bio + topo features for one drug
    # ---------------------------
    def fuse_one(self, bio, topo, drop_bio_prob=0.0, drop_topo_prob=0.0):
        z_bio = self.bio_encoder(bio)
        z_topo = self.topo_encoder(topo)

        if self.training and drop_bio_prob > 0.0 and torch.rand(()) < drop_bio_prob:
            z_bio = torch.zeros_like(z_bio)
        if self.training and drop_topo_prob > 0.0 and torch.rand(()) < drop_topo_prob:
            z_topo = torch.zeros_like(z_topo)

        # Old full-module torch checkpoints do not have ``gate_mode``. Their
        # stored gate already has the correct scalar/vector output dimension.
        gate_mode = getattr(self, "gate_mode", "adaptive_vector")
        if gate_mode == "concat_projection":
            return self.fusion_projection(torch.cat([z_bio, z_topo], dim=1)), None
        if gate_mode == "fixed_half":
            g = torch.full_like(z_bio, 0.5)
        else:
            g = torch.sigmoid(self.gate(torch.cat([z_bio, z_topo], dim=1)))
        if g.shape[1] == 1:
            g = g.expand_as(z_bio)

        z = (1 - g) * z_bio + g * z_topo
        return z, g


    # ---------------------------
    # Combine the two per-drug latents into one pair vector
    # ---------------------------
    def make_pair(self, z1, z2):
        if self.mode == "concat":
            # Ordered: swapping the drugs changes the input, so the head can
            # (and does) learn a different function for (A, B) than for (B, A).
            return torch.cat([z1, z2], dim=1)
        # Symmetric concat: a DDI is an unordered pair, so the pair vector must
        # be identical for (A, B) and (B, A). The element-wise sum and absolute
        # difference are both commutative. This is NOT a lossless encoding of
        # unordered vector pairs: coordinatewise maxima/minima do not identify
        # which coordinates belonged to the same original vector.
        # The head input stays 2 * latent_dim, so nothing downstream changes.
        return torch.cat([z1 + z2, torch.abs(z1 - z2)], dim=1)


    def forward(self, bio1, topo1, bio2, topo2, drop_bio_prob=0.0, drop_topo_prob=0.0):
        # embed modalities to a common latent and fuse them for each drug
        # drop_topo_prob was accepted here but never forwarded to fuse_one, so
        # modality dropout on the topological branch never fired. The auxiliary
        # topology-removal ablation sets it to 0.2; its saved outputs were
        # therefore produced with topo dropout effectively 0.
        z1, g1 = self.fuse_one(bio1, topo1, drop_bio_prob, drop_topo_prob)
        z2, g2 = self.fuse_one(bio2, topo2, drop_bio_prob, drop_topo_prob)
        # build pair representation and classify
        pair_vec = self.make_pair(z1, z2)
        logit = self.classifier(pair_vec)
        return logit.view(-1), (g1, g2)
