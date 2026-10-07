"""Export a post hoc, size-matched positive/negative neighborhood illustration.

Saved topology-only scores and two-edge drug-rooted biological neighborhoods
are used. No fit or correlation is rerun, and no original snapshot is changed.
"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import networkx as nx
import pandas as pd

from .context import _context, _edges


FIT = "inductive_split_00_seed_101"
RULE = ("Post hoc illustrative selection: fixed seen-unseen split_00/seed101; "
        "reference-positive, score >= 0.9, recorded targets for both drugs, "
        "zero shared direct targets, at least two shared biological proteins, "
        "and at most 100 proteins in the union so all can be displayed. "
        "Choose highest protein Jaccard, then greatest shared-protein count, "
        "then highest score, then lexicographic pair ID. "
        "This overlap-informed choice is an illustration, not independent "
        "evidence for a population-level association or pathway mechanism.")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def biological_graph(path):
    """Read node types and biological edges, omitting unused text embeddings."""
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    graph = nx.Graph()
    type_key = None
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == ns + "key":
            if element.get("for") == "node" and element.get("attr.name") == "type":
                type_key = element.get("id")
            element.clear()
        elif element.tag == ns + "node":
            data = {item.get("key"): item.text for item in element.findall(ns + "data")}
            kind = data.get(type_key, "").lower()
            if kind not in {"drug", "protein", "target"}:
                raise ValueError(f"unexpected node type: {element.get('id')}/{kind}")
            graph.add_node(element.get("id"), type=kind)
            element.clear()
        elif element.tag == ns + "edge":
            a, b = element.get("source"), element.get("target")
            if a == b or element.get("directed", "false").lower() == "true":
                raise ValueError("biological contexts require undirected edges without self loops")
            if a not in graph or b not in graph:
                raise ValueError("GraphML edge encountered before its node definitions")
            if not (graph.nodes[a]["type"] == graph.nodes[b]["type"] == "drug"):
                graph.add_edge(a, b)
            element.clear()
        elif element.tag == ns + "graph" and element.get("edgedefault") != "undirected":
            raise ValueError("biological contexts require an undirected graph")
    return graph


NEGATIVE_RULE = ("Same seen-unseen split_00/seed101; reference reliable-negative, "
                 "score <= 0.1, both endpoints with recorded targets, no shared "
                 "direct targets, protein union <= 100, and fewer shared proteins "
                 "than the selected positive. Minimize the sum of absolute "
                 "differences between sorted endpoint neighborhood sizes and "
                 "the positive's sorted sizes; break ties by pair ID. This is "
                 "post hoc illustrative contrast selection, not a controlled "
                 "test of the score-overlap relationship.")


def export_pair(row, biological, training, heldout, names):
    if row.scenario != "seen_unseen" or sum(bool(row[name]) for name in ("drug1_heldout", "drug2_heldout")) != 1:
        raise ValueError("the illustration requires one seen and one held-out endpoint")
    # Keep the seen neighborhood on the blue/left side in either source order.
    if bool(row.drug1_heldout):
        row = row.copy()
        for first, second in (("drug1", "drug2"), ("drug1_heldout", "drug2_heldout"),
                              ("n_targets_a", "n_targets_b"), ("n_proteins_a", "n_proteins_b")):
            row[first], row[second] = row[second], row[first]
    drugs = [row.drug1, row.drug2]
    graphs = [biological if drug in heldout else training for drug in drugs]
    if any(graph.has_edge(*drugs) for graph in graphs):
        raise ValueError("evaluated DDI present in the endpoint graphs")
    for drug, flag in zip(drugs, [row.drug1_heldout, row.drug2_heldout]):
        if (drug in heldout) != bool(flag):
            raise ValueError("endpoint holdout status differs from the saved fit")
    a, b = [_context(graph, drug) for graph, drug in zip(graphs, drugs)]
    shared, union = a["proteins"] & b["proteins"], a["proteins"] | b["proteins"]
    counts = dict(n_targets_a=len(a["direct"]), n_targets_b=len(b["direct"]),
                  n_proteins_a=len(a["proteins"]), n_proteins_b=len(b["proteins"]),
                  shared_protein_count=len(shared), shared_target_count=len(a["direct"] & b["direct"]),
                  protein_union_size=len(union), biological_shared_protein_count=len(shared))
    for key, value in counts.items():
        if int(row[key]) != value:
            raise ValueError(f"reconstructed {key} differs from the saved row")
    if not math.isclose(float(row.protein_jaccard), len(shared) / len(union), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("reconstructed protein Jaccard differs from the saved row")
    edges = {}
    for graph, context in zip(graphs, [a, b]):
        edges.update(_edges(graph, context))
    if "DDI" in edges.values():
        raise ValueError("DDI found in biological example")
    nodes = [dict(id=n, type=biological.nodes[n]["type"], is_drug1=n == drugs[0],
                  is_drug2=n == drugs[1], direct_target_a=n in a["direct"],
                  direct_target_b=n in b["direct"], neighborhood_a=n in a["proteins"],
                  neighborhood_b=n in b["proteins"], shared=n in shared)
             for n in sorted(union | set(drugs))]
    return dict(drug1=drugs[0], drug2=drugs[1], name1=str(names.loc[drugs[0], "name"]),
                name2=str(names.loc[drugs[1], "name"]), score=float(row.score), label=int(row.label),
                scenario=row.scenario, fit_id=row.fit_id, drug1_heldout=bool(row.drug1_heldout),
                drug2_heldout=bool(row.drug2_heldout), **counts,
                protein_jaccard=float(row.protein_jaccard), nodes=nodes,
                edges=[dict(source=u, target=v, relation=relation) for (u, v), relation in sorted(edges.items())],
                shared_paths=[dict(protein=n, path_a=list(a["paths"][n]), path_b=list(b["paths"][n]))
                              for n in sorted(shared)],
                displayed_proteins=len(union), omitted_proteins=0)


def select_examples(predictions, *, fit_id=FIT):
    """Apply the fixed post hoc positive rule and size-matched negative rule."""
    required = {"fit_id", "scenario", "drug1", "drug2", "pair_id", "label", "score",
                "n_targets_a", "n_targets_b", "shared_target_count", "biological_shared_protein_count",
                "shared_protein_count", "protein_union_size", "protein_jaccard", "n_proteins_a", "n_proteins_b"}
    if missing := required - set(predictions):
        raise ValueError(f"missing example-selection columns: {sorted(missing)}")
    cohort = predictions[(predictions.fit_id == fit_id) & (predictions.scenario == "seen_unseen")]
    covered = cohort[(cohort.n_targets_a > 0) & (cohort.n_targets_b > 0) & (cohort.shared_target_count == 0)]
    positive_candidates = covered[(covered.label == 1) & (covered.score >= .9)
                                  & (covered.biological_shared_protein_count >= 2)].copy()
    compact = positive_candidates[positive_candidates.protein_union_size <= 100].sort_values(
        ["protein_jaccard", "shared_protein_count", "score", "pair_id"],
        ascending=[False, False, False, True], kind="stable")
    if compact.empty:
        raise ValueError("no positive candidate meets the documented rule")
    positive = compact.iloc[0]
    negative_candidates = covered[(covered.label == 0) & (covered.score <= .1)
                                  & (covered.protein_union_size <= 100)
                                  & (covered.biological_shared_protein_count < positive.biological_shared_protein_count)].copy()
    small, large = sorted([positive.n_proteins_a, positive.n_proteins_b])
    negative_candidates["neighborhood_size_distance"] = (
        (negative_candidates[["n_proteins_a", "n_proteins_b"]].min(axis=1) - small).abs()
        + (negative_candidates[["n_proteins_a", "n_proteins_b"]].max(axis=1) - large).abs())
    negative_candidates = negative_candidates.sort_values(["neighborhood_size_distance", "pair_id"], kind="stable")
    if negative_candidates.empty:
        raise ValueError("no negative candidate meets the documented rule")
    negative = negative_candidates.iloc[0]
    counts = dict(n_source_pairs=len(cohort), n_positive_eligible=len(positive_candidates),
                  n_positive_compact=len(compact), n_negative_eligible=len(negative_candidates),
                  neighborhood_size_distance=int(negative.neighborhood_size_distance))
    return positive, negative, counts


def _load_verified_results(results):
    if (results / "snapshot_manifest.json").is_file():
        from .saved_results import load_saved_results
        loaded = load_saved_results("inductive", results)
        if "inductive/source_provenance.json" not in loaded["snapshot_manifest"]["retained_files"]:
            raise ValueError("source provenance is absent from the verified snapshot inventory")
    else:
        from .workflow import load_results
        loaded = load_results("inductive", results)
    return loaded["pair_overlap_scores"]


def export_examples(results_dir, graph_path, manifest_path, drug_names_csv, output_dir, *, fit_id=FIT):
    """Write a private examples.json with both complete biological neighborhoods.

    Results are verified saved case-study analyses, not training directories.
    The JSON contains licensed graph records and drug identities and is not a
    public release artifact. Both paths and selection parameters are explicit.
    """
    results, graph_path, split_path, name_path, out = map(
        Path, (results_dir, graph_path, manifest_path, drug_names_csv, output_dir))
    predictions = _load_verified_results(results)
    positive, negative, counts = select_examples(predictions, fit_id=fit_id)
    provenance_rows = json.loads((results / "inductive/source_provenance.json").read_text())
    matching = [record for record in provenance_rows if record["fit_id"] == fit_id]
    if len(matching) != 1:
        raise ValueError("require exactly one provenance record for the selected fit")
    provenance = matching[0]
    if digest(graph_path) != provenance["graph_sha256"] or digest(split_path) != provenance["prepared_manifest_sha256"]:
        raise ValueError("graph or holdout manifest differs from the verified fit")
    heldout = set(json.loads(split_path.read_text())["heldout_drugs"])
    biological = biological_graph(graph_path)
    training = nx.subgraph_view(biological, filter_node=lambda node: node not in heldout)
    names = pd.read_csv(name_path, dtype={"drugbank_id": str}).set_index("drugbank_id")
    if not names.index.is_unique or "name" not in names or names.name.isna().any():
        raise ValueError("drug names must have unique identifiers and nonmissing names")
    examples = {kind: export_pair(row, biological, training, heldout, names)
                for kind, row in [("positive", positive), ("negative", negative)]}
    for data in examples.values():
        if any(node["shared"] and (node["direct_target_a"] or node["direct_target_b"]) for node in data["nodes"]):
            raise ValueError("selected shared protein is a focal direct target")
        if any(len(item[key]) != 3 for item in data["shared_paths"] for key in ["path_a", "path_b"]):
            raise ValueError("selected overlap is not two-edge reach from both drugs")
    rules = dict(positive=RULE, negative=NEGATIVE_RULE)
    if fit_id != FIT:
        rules = {name: rule.replace("split_00/seed101", fit_id) for name, rule in rules.items()}
    bundle = dict(**examples, selection_rules=rules, **counts,
                  graph_sha256=provenance["graph_sha256"], split_sha256=digest(split_path),
                  name_source_sha256=digest(name_path), train_graph_id=provenance["train_graph_id"],
                  inference_graph_id=provenance["inference_graph_id"],
                  validation="source graph, split and complete neighborhood counts verified; saved predictions reused, not replayed")
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
