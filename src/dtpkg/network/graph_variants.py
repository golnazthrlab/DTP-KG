"""Edge-type variants and content hashes for one canonical DTP-KG.

Variants retain the same node universe and partition the base graph's edges,
allowing subsequent ablations to compare DDI-only, non-DDI, and full topology.
No graph data are included in this module.
"""

import hashlib

import networkx as nx

DRUG = "drug"
VARIANTS = ("full", "no_ddi", "ddi_only")


def _is_drug(graph, node):
    return str(graph.nodes[node].get("type", "")).lower() == DRUG


def ddi_edges(graph):
    """Every drug--drug edge, as an unordered-pair list."""
    return [(u, v) for u, v in graph.edges()
            if _is_drug(graph, u) and _is_drug(graph, v)]


def build_variant(graph, variant, keep_node_universe=True):
    """Return a new graph holding `variant`'s edge subset of `graph`.

    `keep_node_universe` retains every node of the base graph in every variant, so
    the drug cohort, the node set and the ``n_nodes_subgraph`` denominators stay
    comparable across conditions. Under ``ddi_only`` this leaves target/protein
    nodes as isolates; they are unreachable from any drug's ego and therefore do
    not enter any descriptor. Set it to False only to record a variant whose node
    set is deliberately different, and say so in the results.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("variants are defined for a simple undirected graph")

    out = nx.Graph()
    out.add_nodes_from((n, dict(d)) for n, d in graph.nodes(data=True))
    drug_drug = set(map(frozenset, ddi_edges(graph)))
    for u, v, data in graph.edges(data=True):
        is_ddi = frozenset((u, v)) in drug_drug
        if variant == "full" or (variant == "ddi_only") == is_ddi:
            out.add_edge(u, v, **dict(data))

    if not keep_node_universe:
        out.remove_nodes_from([n for n in list(out) if out.degree(n) == 0])
    return out


def graph_content_hash(graph):
    """Stable digest of node types and the undirected edge set.

    Used as the graph component of a feature-cache key so a table is never reused
    across a different mask or a rebuilt graph. Ignores edge attributes, which do
    not enter the unweighted descriptors.
    """
    digest = hashlib.sha256()
    for node in sorted(graph):
        digest.update(f"{node}\x1f{graph.nodes[node].get('type', '')}\x1e".encode())
    digest.update(b"\x1d")
    for u, v in sorted(tuple(sorted((str(u), str(v)))) for u, v in graph.edges()):
        digest.update(f"{u}\x1f{v}\x1e".encode())
    return digest.hexdigest()


def variant_provenance(graph, variant, base_hash=None):
    """Counts and hashes to record alongside any result computed on `graph`."""
    types = nx.get_node_attributes(graph, "type")
    counts = {}
    for u, v in graph.edges():
        key = "-".join(sorted((str(types.get(u, "?")).lower(),
                               str(types.get(v, "?")).lower())))
        counts[key] = counts.get(key, 0) + 1
    drugs = [n for n in graph if _is_drug(graph, n)]
    return {
        "variant": variant,
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "drug_nodes": len(drugs),
        "isolated_drug_nodes": sum(1 for n in drugs if graph.degree(n) == 0),
        "edges_by_endpoint_type": counts,
        "graph_content_hash": graph_content_hash(graph),
        "base_graph_content_hash": base_hash,
    }
