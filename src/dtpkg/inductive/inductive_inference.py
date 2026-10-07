"""Recover record-level outputs from trusted private inductive checkpoints."""
from pathlib import Path

import pandas as pd
import torch

from dtpkg.fusion.trainer import FusionTrainer

def recover_ancillary_exports(bundle_path, destination):
    """Rewrite the per-fit gate and per-bin metric CSVs from a saved bundle."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    written = []
    for key, name in (("gates", "inductive_drug_gates.csv"), ("fold_metrics", "inductive_fold_metrics.csv")):
        if key in bundle:
            frame = bundle[key].copy()
            if key == "fold_metrics":
                frame["inference_protocol"] = "inductive_fixed_holdout"
            frame.to_csv(Path(destination) / name, index=False)
            written.append(name)
    return written


def score_saved_fit(path, device="cpu", expected_identity=None):
    """Re-score a trusted LOCAL bundle without fitting a model or transform.

    The checkpoint includes full modules/pandas/sklearn objects. Never load a
    checkpoint from an untrusted source. Saved test pairs/features are fixed;
    introducing new partners requires extracting their permitted topology first.
    """
    from dtpkg.metrics import evaluate_baseline_inductive, evaluate_by_pair_bins_fusion
    bundle = torch.load(path, map_location=device, weights_only=False)
    if bundle.get("protocol") != "inductive_fixed_holdout":
        raise ValueError("not an inductive checkpoint")
    if expected_identity is not None and bundle.get('fit_identity') != expected_identity:
        raise ValueError("checkpoint does not belong to the requested fit identity")
    frames = []
    for name, model in bundle["models"].items():
        model = model.to(device)
        if name == "baseline":
            _, p = evaluate_baseline_inductive(model, bundle["test_pairs"], bundle["embeddings"],
                bundle["drug_to_cat"], device, scaler=bundle["baseline_scaler"], return_predictions=True)
        elif name == "fusion":
            _, p = evaluate_by_pair_bins_fusion(model, bundle["test_pairs"], bundle["embeddings"],
                bundle["eval_topology"], bundle["drug_to_cat"], device, return_predictions=True)
        elif name == "topo_only":
            # This evaluator uses only these two attributes; avoid creating files.
            trainer = object.__new__(FusionTrainer)
            trainer.device, trainer.drug_to_cat = device, bundle["drug_to_cat"]
            mu, sigma, columns = bundle["topo_only_normalization"]
            trainer._evaluate_bins_topo_only(model, bundle["eval_topology"], bundle["test_pairs"], mu, sigma, columns)
            p = trainer.last_predictions
        else:
            raise ValueError(f"unsupported checkpoint model {name}")
        p = p.rename(columns={"pred": "score", "pair_bin": "category"}).assign(model=name)
        p["threshold"] = .5
        p["score_kind"] = "neural_probability"
        frames.append(p)
    for name, p in bundle.get("graph_baseline_predictions", {}).items():
        p = p.rename(columns={"pred": "score", "pair_bin": "category"}).assign(model=name)
        p["score_kind"] = "raw_graph_score"
        frames.append(p)
    out = pd.concat(frames, ignore_index=True)
    out["drug1_heldout"] = out.drug1.isin(bundle["heldout_drugs"])
    out["drug2_heldout"] = out.drug2.isin(bundle["heldout_drugs"])
    return out.assign(**bundle["metadata"])
