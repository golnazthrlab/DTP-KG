"""Verify graph DDIs against the positive table and report aggregate provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from dtpkg.ddi_labels import load_dpi_drugs, pair_set
from dtpkg.project_paths import DATA_DIR, NETWORK_DIR


def graph_content_hash(graph):
    """Study digest of node IDs/types and undirected edges, excluding attributes."""
    digest = hashlib.sha256()
    for node in sorted(graph):
        digest.update(f"{node}\x1f{graph.nodes[node].get('type', '')}\x1e".encode())
    digest.update(b"\x1d")
    for u, v in sorted(tuple(sorted((str(u), str(v)))) for u, v in graph.edges()):
        digest.update(f"{u}\x1f{v}\x1e".encode())
    return digest.hexdigest()


def verify_graph(graph, positives, dpi_drugs):
    """Compare unordered DDI pairs without exposing pair identities in the report."""
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("expected a simple undirected graph")
    types = {n: str(d.get("type", "")).lower() for n, d in graph.nodes(data=True)}
    drugs = {n for n, kind in types.items() if kind == "drug"}
    edges = {tuple(sorted((str(u), str(v)))) for u, v in graph.edges()
             if u in drugs and v in drugs}
    positive_keys = pair_set(positives)
    counts = {}
    for u, v in graph.edges():
        kind = "-".join(sorted((types[u], types[v])))
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "graph_ddi_edges": len(edges), "positive_pairs": len(positive_keys),
        "edges_not_in_positives": len(edges - positive_keys),
        "positives_not_in_graph": len(positive_keys - edges),
        "exact_match": edges == positive_keys,
        "drug_nodes": len(drugs),
        "drug_nodes_without_targets": len(drugs - set(dpi_drugs)),
        "provenance": {
            "variant": "full", "nodes": len(graph), "edges": graph.number_of_edges(),
            "drug_nodes": len(drugs),
            "isolated_drug_nodes": sum(graph.degree(d) == 0 for d in drugs),
            "edges_by_endpoint_type": counts,
            "graph_content_hash": graph_content_hash(graph),
            "base_graph_content_hash": None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=NETWORK_DIR / "unweighted_dppi_PubMedBERT.graphml")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--positives", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    positives = args.positives or args.data_dir / "interactions" / "DDI_positive_pairs.csv"
    graph = nx.read_graphml(args.graph)
    report = verify_graph(graph, pd.read_csv(positives, dtype=str),
                          load_dpi_drugs(args.data_dir / "interactions" / "DPI_enriched.csv"))
    report.update(graph=str(args.graph), positives=str(positives))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
