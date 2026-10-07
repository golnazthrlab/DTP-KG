"""Two fixed-direction DDI heuristics, evaluated on the same masked graph."""
import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_curve


def masked_ddi_graph(graph, heldout_pairs):
    """Use drug--drug edges only. Remove validation AND test positives on a copy."""
    drugs = [node for node, data in graph.nodes(data=True)
             if str(data.get("type", "")).lower() == "drug"]
    out = nx.Graph(graph.subgraph(drugs))
    out.remove_edges_from((r.drug1, r.drug2) for r in heldout_pairs.itertuples(index=False)
                          if r.label == 1)
    out.remove_edges_from(nx.selfloop_edges(out))
    if out.number_of_edges() == 0:
        raise ValueError("No DDI edges remain in the masked drug graph; disable graph baselines "
                         "for a no_ddi graph instead of reporting constant scores")
    return out


def graph_scores(graph, pairs, method):
    if method not in ("common_neighbors", "degree_product"):
        raise ValueError("unknown graph baseline")
    def neighbors(drug):
        return set(graph[drug]) if drug in graph else set()
    adjacency = {d: neighbors(d) for d in set(pairs.drug1) | set(pairs.drug2)}
    return np.array([len(adjacency[r.drug1] & adjacency[r.drug2])
                     if method == "common_neighbors" else
                     len(adjacency[r.drug1]) * len(adjacency[r.drug2])
                     for r in pairs.itertuples(index=False)], dtype=float)


def shared_target_scores(graph, pairs):
    """Number of common NON-drug neighbours (targets/proteins) of the two drugs.

    A biological baseline that survives the removal of every DDI edge, so it is
    available in both inductive arms; it never reads a drug--drug edge.
    """
    def biological_neighbors(drug):
        if drug not in graph:
            return set()
        return {v for v in graph[drug] if str(graph.nodes[v].get("type", "")).lower() != "drug"}
    adjacency = {d: biological_neighbors(d) for d in set(pairs.drug1) | set(pairs.drug2)}
    return np.array([len(adjacency[r.drug1] & adjacency[r.drug2])
                     for r in pairs.itertuples(index=False)], dtype=float)


def validation_f1_threshold(labels, scores):
    """Choose on inner validation only; never reverse the heuristic's ranking.
    Ties choose the highest threshold. Raw graph scores are not probabilities.
    """
    y, score = np.asarray(labels), np.asarray(scores, float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(score).all():
        raise ValueError("threshold selection requires finite scores and both validation classes")
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
