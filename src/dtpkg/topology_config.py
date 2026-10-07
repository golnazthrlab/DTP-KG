"""Declare the graph and feature settings used by a topology experiment."""

import networkx as nx

from dtpkg.network.graph_variants import (
    VARIANTS, build_variant, graph_content_hash, variant_provenance,
)
from dtpkg.project_paths import NETWORK_DIR
from dtpkg.topology import TopoFeatExtractor


CANONICAL_GRAPH = NETWORK_DIR / "unweighted_dppi_PubMedBERT.graphml"

# Retained for explicit reproduction of the earlier sparse-graph definition.
# The transductive fusion results in this release use DENSE_GRAPH_SETTINGS.
PUBLISHED_GRAPH = NETWORK_DIR / "simon_network.graphml"
PAPER_SETTINGS = dict(closeness_max_nodes=200, katz_alpha_mode="fixed")

# Dense DDI graphs require spectral Katz scaling for convergence. Removing the
# ego-size cutoff computes closeness even when an ego has 200 or more nodes.
DENSE_GRAPH_SETTINGS = dict(closeness_max_nodes=None, katz_alpha_mode="spectral")


def load_graph(path=CANONICAL_GRAPH):
    """Load a private GraphML with node ``type`` attributes."""
    return nx.read_graphml(str(path))


def make_extractor(variant="full", graph=None, graph_path=CANONICAL_GRAPH,
                   settings=None, backend="optimized", n_jobs=1,
                   cache_dir=None, **extractor_kwargs):
    """Return an extractor and aggregate graph/feature provenance.

    Node types are ``drug``, ``target``, or ``protein``. DDI edges are identified
    by their drug endpoints. The study uses a simple undirected graph, unweighted
    descriptors, and radius-two egos. Callers must pass all validation and test
    positive pairs to ``compute_for_fold`` so those targets are excluded from
    the graph before any fold features are computed.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    base = load_graph(graph_path) if graph is None else graph
    base_hash = graph_content_hash(base)
    chosen = base if variant == "full" else build_variant(base, variant)
    provenance = variant_provenance(chosen, variant, base_hash=base_hash)
    provenance["base_graph_path"] = str(graph_path) if graph is None else None
    resolved = dict(settings if settings is not None else DENSE_GRAPH_SETTINGS)
    resolved.update(extractor_kwargs)
    extractor = TopoFeatExtractor(
        graph=chosen, backend=backend, n_jobs=n_jobs, cache_dir=cache_dir,
        **resolved,
    )
    provenance["feature_settings"] = extractor.feature_settings(
        radius=2, katz_alpha=0.005,
    )
    provenance["n_jobs"] = n_jobs
    return extractor, provenance
