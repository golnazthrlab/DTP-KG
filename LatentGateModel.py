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
        fusion_mode="concat",
        enc_hidden=(256,),
        head_hidden=(256, 128),
        dropout=0.1,
        gate_scalar=False,
        gate_bias_init=0.7,
    ):
        super().__init__()

        print('init gate_scalar: ', gate_scalar)

        assert fusion_mode in ("concat", "sym4")

        # Encoders for biological and topological features
        self.bio_encoder = Encoder(bio_dim, latent_dim, hidden=enc_hidden, dropout=dropout)
        self.topo_encoder = Encoder(topo_dim, latent_dim, hidden=enc_hidden, dropout=dropout)

        # Learnable gate (decides how much to use topo vs bio)
        gate_out_dim = 1 if gate_scalar else latent_dim
        self.gate = nn.Linear(2 * latent_dim, gate_out_dim)
        with torch.no_grad():
            self.gate.bias.fill_(torch.logit(torch.tensor(gate_bias_init)))

        # Classifier head (takes combined drug pair)
        pair_input_dim = 2 * latent_dim if fusion_mode == "concat" else 4 * latent_dim
        self.classifier = Classifier(pair_input_dim, hidden=head_hidden, dropout=dropout)

        self.mode = fusion_mode

    # ---------------------------
    # Fuse bio + topo features for one drug
    # ---------------------------
    def fuse_one(self, bio, topo, drop_bio_prob=0.0):
        if self.training and drop_bio_prob > 0.0 and torch.rand(()) < drop_bio_prob:
            bio = torch.zeros_like(bio)

        z_bio = self.bio_encoder(bio)
        z_topo = self.topo_encoder(topo)

        g = torch.sigmoid(self.gate(torch.cat([z_bio, z_topo], dim=1)))
        if g.shape[1] == 1:
            g = g.expand_as(z_bio)

        # Weighted combination: z = (1-g)*bio + g*topo
        z = (1 - g) * z_bio + g * z_topo
        return z, g


    def make_pair(self, z1, z2):
        if self.mode == "concat":
            return torch.cat([z1, z2], dim=1)
        else:
            return torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)


    def forward(self, bio1, topo1, bio2, topo2, drop_bio_prob=0.0):
        # embed modalities to a common latent and fuse them for each drug
        z1, g1 = self.fuse_one(bio1, topo1, drop_bio_prob)
        z2, g2 = self.fuse_one(bio2, topo2, drop_bio_prob)
        # build pair representation and classify
        pair_vec = self.make_pair(z1, z2)
        logit = self.classifier(pair_vec)
        return logit.view(-1), (g1, g2)