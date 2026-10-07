"""The published inductive protocol: no DDI edges and no held-out training nodes."""
from copy import copy

from dtpkg.fusion.graph_baselines import shared_target_scores


def _verify_no_ddi(graph):
    drug = {n for n, data in graph.nodes(data=True) if str(data.get("type", "")).lower() == "drug"}
    if any(a in drug and b in drug for a, b in graph.edges):
        raise ValueError("The biological-only inductive protocol excludes every DDI edge")


def build_split_graphs(base_extractor, heldout_drugs):
    """Retain biological relations at inference; remove held-out nodes in training."""
    _verify_no_ddi(base_extractor.G)
    heldout = set(map(str, heldout_drugs))
    if not heldout.issubset(base_extractor.G):
        raise ValueError("Held-out drugs must be present in the source graph")
    inference = copy(base_extractor)
    inference.G = base_extractor.G.copy()
    training = copy(base_extractor)
    training.G = base_extractor.G.copy()
    training.remove_inductive_drugs(heldout)
    if heldout & set(training.G):
        raise ValueError("Held-out drugs remain in the training graph")
    _verify_no_ddi(training.G)
    checks = dict(inference=dict(policy="none", n_heldout_incident_ddi=0),
                  training=dict(n_nodes=len(training.G), n_edges=training.G.number_of_edges()))

    def graph_factory(masked_pairs):
        # These counts use only biological neighbours; there are no DDIs to mask.
        return {"shared_targets": dict(
            validation=lambda pairs: shared_target_scores(training.G, pairs),
            test=lambda pairs: shared_target_scores(inference.G, pairs))}

    return training, inference, graph_factory, checks
