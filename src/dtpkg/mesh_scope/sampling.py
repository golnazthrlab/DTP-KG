"""Pair-sampling loaders extracted from the manuscript fusion trainer.

Only the sampling and training-only scaling behavior is needed by this experiment.
"""
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dtpkg.data_loaders import DDIDataset, build_pair_features
from dtpkg.ddi_sampling import EpochPositiveSampler


class SamplingLoaders:
    """Construct paired sampling schedules and freeze training-only scalers."""

    def __init__(self, *, positive_pool, seed=42, batch_size=128,
                 positive_sampling="per_epoch", positive_to_negative_ratio=1.0):
        self.positive_pool = positive_pool
        self.seed = seed
        self.batch_size = batch_size
        self.positive_sampling = positive_sampling
        self.positive_to_negative_ratio = positive_to_negative_ratio

    def make_sampler(self, train_pairs, heldout_pairs, emb_df, fold=0, repetition=0,
                     excluded_drugs=()):
        """Construct a schedule before feature extraction; fail rather than
        silently claim epoch resampling from a previously downsampled cohort."""
        if self.positive_pool is None:
            if self.positive_sampling == "per_epoch":
                raise ValueError("per_epoch requires positive_pool: pass the complete eligible "
                                 "positive table to SamplingLoaders, or select positive_sampling='fixed'")
            if self.positive_to_negative_ratio != 1.0:
                raise ValueError("a non-default ratio requires positive_pool")
            return None  # explicit legacy fixed-cohort fit
        pool = self.positive_pool
        ids = set(emb_df.index.astype(str))
        pool = pool[pool.drug1.isin(ids) & pool.drug2.isin(ids)]
        return EpochPositiveSampler(
            pool, train_pairs[train_pairs.label == 0], heldout_pairs,
            seed=self.seed, fold=fold, repetition=repetition,
            mode=self.positive_sampling,
            positive_to_negative_ratio=self.positive_to_negative_ratio,
            excluded_drugs=excluded_drugs)

    def sampled_loaders(self, sampler, emb_df, val_pairs, test_pairs=None, *, scaler_epochs=None):
        """Fit the MeSH scaler on epoch-zero TRAINING pairs, then freeze it.
        Both sampling modes share the same epoch-zero draw and scaling policy.

        ``scaler_epochs=n`` fits the scaler incrementally on the whole n-epoch
        training schedule instead (still training pairs only). A feature that
        is near-constant in one 3,818-pair draw but not in the population gets
        a scale of ~1e-6 from epoch zero alone, and pairs drawn in later epochs
        then enter the network standardized to 1e5 -- which corrupts the
        BatchNorm running statistics used at evaluation time (Intermediate
        MeSH scope on the common-coverage cohort, 2026-09-18)."""
        if scaler_epochs is not None and (int(scaler_epochs) != scaler_epochs or scaler_epochs < 1):
            raise ValueError("scaler_epochs must be a positive integer")
        X0, _, p0 = build_pair_features(emb_df, sampler.epoch(0))
        if len(p0) != sampler.epoch_size:
            raise ValueError("sampling schedule contains drugs without embeddings")
        scaler = StandardScaler()
        if scaler_epochs is None:
            scaler.fit(X0)
        else:
            scaler.partial_fit(X0)
            for epoch in range(1, int(scaler_epochs)):
                Xe, _, pe = build_pair_features(emb_df, sampler.epoch(epoch))
                if len(pe) != sampler.epoch_size:
                    raise ValueError("sampling schedule contains drugs without embeddings")
                scaler.partial_fit(Xe)

        def loader(frame):
            X, y, kept = build_pair_features(emb_df, frame)
            if len(kept) != len(frame):
                raise ValueError("evaluation/sampled pairs lack embeddings")
            return DataLoader(DDIDataset(scaler.transform(X), y),
                              batch_size=self.batch_size, shuffle=False)

        def epoch_loader(epoch):
            return loader(sampler.epoch(epoch))

        return (epoch_loader(0), loader(val_pairs),
                None if test_pairs is None else loader(test_pairs), epoch_loader, scaler)
