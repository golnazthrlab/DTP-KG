"""Deterministic training-positive sampling with fixed held-out pairs.

Proposed in the 2026-09-17 data review as an alternative to freezing one small
positive sample for the whole fit: the negative set is small (3,343 vetted
pairs) while the eligible positive pool is ~500k, so a fixed balanced sample
sees <1% of the documented interactions. Drawing ``len(N_train)`` fresh
positives every epoch keeps the class-balanced objective (half the mean loss
over the positive pool plus half over the negatives) while exposing the model
to far more positives over training.

FusionTrainer consumes this schedule in both the MeSH and fusion training loops.
``mode='fixed'`` reuses epoch-zero membership; both modes reshuffle each epoch.

Contract per transductive fold:

1. Validation/test pairs are fixed and never sampled into either class.
2. ``P_train`` = eligible positive pool minus every held-out pair;
   ``N_train`` = the fold's inner-training negatives (used once per epoch).
3. Each epoch draws ``len(N_train)`` *distinct* positives uniformly from
   ``P_train`` with a seed keyed by (seed, fold, repetition, epoch); the same
   key gives the same draw on every graph arm, so arms stay paired.
4. Disjointness is asserted on unordered pair identities, not row indices.
5. Sampling a pair never touches the fold graph mask.

The endpoint union of a declared schedule is what a topology cache must cover.
"""
import numpy as np
import pandas as pd

from dtpkg.ddi_labels import canonicalize_pairs, QUARANTINED_PAIRS


def _keys(df):
    canon = canonicalize_pairs(df[["drug1", "drug2"]])
    return list(zip(canon["drug1"], canon["drug2"]))


def epoch_seed(seed, fold, repetition, epoch):
    """Deterministic, collision-free for fold < 1000, repetition < 100, epoch < 10000."""
    if any(int(x) != x for x in (seed, fold, repetition, epoch)):
        raise ValueError("seed coordinates must be integers")
    if seed < 0 or not 0 <= fold < 1000 or not 0 <= repetition < 100 or not 0 <= epoch < 10000:
        raise ValueError("invalid seed coordinates: seed>=0, fold<1000, repetition<100, epoch<10000")
    return int(seed) * 1_000_000_000 + int(fold) * 1_000_000 + int(repetition) * 10_000 + int(epoch)


class EpochPositiveSampler:
    def __init__(self, positive_pool, train_negatives, heldout_pairs,
                 seed=42, fold=0, repetition=0, mode="per_epoch",
                 positive_to_negative_ratio=1.0, excluded_drugs=()):
        if mode not in ("fixed", "per_epoch"):
            raise ValueError("mode must be 'fixed' or 'per_epoch'")
        ratio = float(positive_to_negative_ratio)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("positive_to_negative_ratio must be finite and > 0")
        if "label" in positive_pool and not positive_pool.label.eq(1).all():
            raise ValueError("positive_pool contains non-positive labels")
        if "label" in train_negatives and not train_negatives.label.eq(0).all():
            raise ValueError("train_negatives contains non-negative labels")
        pool = set(_keys(positive_pool))
        if pool & set(QUARANTINED_PAIRS):
            raise ValueError("positive_pool contains quarantined pairs; use the guarded loader")
        heldout = set(_keys(heldout_pairs)) if heldout_pairs is not None else set()
        negatives = _keys(train_negatives)
        neg_set = set(negatives)
        if neg_set & set(QUARANTINED_PAIRS):
            raise ValueError("train_negatives contains quarantined pairs; use the guarded loader")
        excluded = set(map(str, excluded_drugs))
        if any(a in excluded or b in excluded for a, b in neg_set):
            raise ValueError("training negatives contain an excluded drug")
        pool = {(a, b) for a, b in pool if a not in excluded and b not in excluded}
        if len(neg_set) != len(negatives):
            raise ValueError("train_negatives contains duplicate unordered pairs")
        if neg_set & heldout:
            raise ValueError(f"{len(neg_set & heldout)} training negatives are held-out pairs")
        train_pool = pool - heldout
        if train_pool & neg_set:
            raise ValueError(f"{len(train_pool & neg_set)} pairs are both positive and negative")
        n_positive = int(np.floor(ratio * len(neg_set)))
        if not neg_set or n_positive < 1:
            raise ValueError("an epoch requires at least one pair of each class")
        if len(train_pool) < n_positive:
            raise ValueError("positive pool smaller than the negative set; cannot draw "
                             "distinct positives for a balanced epoch")
        self.mode, self.positive_to_negative_ratio = mode, ratio
        self.n_positive = n_positive
        self.positives = np.array(sorted(train_pool), dtype=object)
        self.negatives = pd.DataFrame(sorted(neg_set), columns=["drug1", "drug2"])
        self.heldout = heldout
        self.excluded_heldout_positives = len(pool & heldout)
        self.seed, self.fold, self.repetition = int(seed), int(fold), int(repetition)

    @property
    def epoch_size(self):
        return self.n_positive + len(self.negatives)

    def epoch(self, epoch_idx):
        """Shuffled frame with distinct positives and each negative once."""
        draw_epoch = 0 if self.mode == "fixed" else epoch_idx
        draw_rng = np.random.default_rng(epoch_seed(self.seed, self.fold, self.repetition, draw_epoch))
        draw = draw_rng.choice(len(self.positives), size=self.n_positive, replace=False)
        pos = pd.DataFrame(list(self.positives[np.sort(draw)]), columns=["drug1", "drug2"])
        pos["label"] = 1
        neg = self.negatives.copy()
        neg["label"] = 0
        frame = pd.concat([pos, neg], ignore_index=True)
        # Sampling membership and SGD ordering are independent random streams.
        # Both policies shuffle every epoch; only positive membership differs.
        order_rng = np.random.default_rng(np.random.SeedSequence(
            [epoch_seed(self.seed, self.fold, self.repetition, epoch_idx), 1]))
        order = order_rng.permutation(len(frame))
        return frame.iloc[order].reset_index(drop=True)

    def schedule(self, n_epochs):
        return [self.epoch(e) for e in range(int(n_epochs))]

    def endpoint_union(self, n_epochs):
        """Every drug a topology table must cover for this schedule (training side)."""
        drugs = set()
        for frame in (self.epoch(e) for e in range(int(n_epochs))):
            drugs.update(frame["drug1"]); drugs.update(frame["drug2"])
        return drugs

    def unique_positives(self, n_epochs):
        seen = set()
        for frame in (self.epoch(e) for e in range(int(n_epochs))):
            seen.update(zip(frame.loc[frame.label == 1, "drug1"],
                            frame.loc[frame.label == 1, "drug2"]))
        return len(seen)

    def expected_unique_positives(self, n_epochs):
        """E[#distinct] for independent draws of m distinct from N: N(1-(1-m/N)^E)."""
        n, m = len(self.positives), self.n_positive
        if self.mode == "fixed":
            return m if int(n_epochs) > 0 else 0
        return n * (1.0 - (1.0 - m / n) ** int(n_epochs))

    def describe(self, n_epochs=None):
        out = {
            "positive_sampling": self.mode,
            "positive_to_negative_ratio": self.positive_to_negative_ratio,
            "training_positive_pool": int(len(self.positives)),
            "heldout_positives_excluded_from_pool": int(self.excluded_heldout_positives),
            "train_negatives": int(len(self.negatives)),
            "epoch_size": int(self.epoch_size),
            "seed": self.seed, "fold": self.fold, "repetition": self.repetition,
        }
        if n_epochs is not None:
            out["epochs"] = int(n_epochs)
            out["expected_unique_positives"] = float(self.expected_unique_positives(n_epochs))
        return out
