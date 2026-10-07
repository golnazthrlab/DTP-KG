"""What is fitted for one drug split and training seed — the hashed part of a fit.

This module, together with the shared training code in ``CODE_FILES``, is part of
every fit identity: changing anything here invalidates saved fits, which is the
intended guard. Orchestration (identity bookkeeping, reuse, recovery and
exports) lives in ``inductive_experiments.py`` and is deliberately NOT
hashed, so IO or reporting fixes never strand a completed study.
"""
from pathlib import Path

from dtpkg.fusion.trainer import FusionTrainer
from dtpkg.models.latent_gate import Classifier, LatentGateModel
from dtpkg.data_loaders import build_pair_features
from dtpkg.evaluation_inputs import pair_drugs

DEFAULT_SETTINGS = dict(epochs=100, patience=10, batch_size=128, lr=.001, weight_decay=.00001,
                        drop_bio_prob=.3, drop_topo_prob=0., val_split=.15,
                        positive_sampling="per_epoch", positive_to_negative_ratio=1.0,
                        scaling_policy="planned_training_schedule")

# Paths are relative to the installed dtpkg package. Hashes describe this release.
CODE_FILES = (
    "fusion/trainer.py", "models/latent_gate.py", "fusion/topology_baseline.py",
    "data_loaders.py", "topology.py", "ddi_sampling.py", "metrics.py",
    "fusion/graph_baselines.py", "inductive/inductive_fit.py",
    "inductive/inductive_data.py", "inductive/validation.py", "inductive/graph_access.py",
    "evaluation_inputs.py", "topology_config.py", "mesh_scopes.py", "network/graph_variants.py",
)

def fit_split_seed(manifest, emb, frames, test, depth_map, train_extractor, heldout_extractor,
                   graph_factory, options, training_seed, destination, signature, device, fit_baseline):
    """One fit: MeSH-only (optional), topology-only and fusion, scoring both scenarios."""
    trainer = FusionTrainer(
        LatentGateModel(emb.shape[1], train_extractor.num_of_topo_feats, latent_dim=128, fusion_mode="sym",
                        enc_hidden=(256,), head_hidden=(256, 128), dropout=.1, gate_bias_init=.7),
        baseline_cls=Classifier(emb.shape[1] * 2, hidden=(256, 128), dropout=.1),
        topo_extractor=train_extractor, device=device, seed=int(training_seed),
        out_dir=destination, positive_pool=frames['positive_pool'],
        include_topology_baseline=True, include_graph_baselines=False,
        **{k: v for k, v in options.items() if k not in ('val_split', 'scaling_policy')})
    required_heldout = sorted(pair_drugs(test) & set(manifest['heldout_drugs']))

    def heldout_features(validation_pairs):
        return heldout_extractor.compute_for_fold(required_heldout, eval_pairs=validation_pairs)

    X, y, pairs = build_pair_features(emb, frames['development'])
    trainer.run_inductive_eval(X, y, pairs, emb, None, test, depth_map,
        num_experiments=1, val_split=options['val_split'],
        inductive_drugs=manifest['heldout_drugs'], heldout_topology_factory=heldout_features,
        checkpoint_dir=Path(destination) / 'checkpoints', checkpoint_identity=signature,
        scaling_policy=options['scaling_policy'], fit_baseline=fit_baseline,
        graph_baseline_factory=graph_factory)
    return trainer.inductive_predictions_df
