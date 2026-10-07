"""Complete drug-rooted protein contexts and deterministic witness paths."""

PROTEIN_TYPES = {"target", "protein"}


def _type(graph, node):
    return str(graph.nodes[node].get("type", "")).lower()


def _context(graph, drug):
    """Complete protein membership, with one deterministic shortest witness.

    Direct neighbors are resolved first. Among two-step routes, protein
    intermediates precede drug intermediates, then identifiers break ties.
    This makes the selected route biological whenever that is possible.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("case-study views require simple undirected graphs")
    if drug not in graph or _type(graph, drug) != "drug":
        raise ValueError(f"focal endpoint {drug!r} must be a drug in its feature graph")
    first = set(graph[drug])
    direct = {n for n in first if _type(graph, n) in PROTEIN_TYPES}
    paths = {n: (drug, n) for n in sorted(direct, key=str)}
    for middle in sorted(first, key=lambda n: (_type(graph, n) not in PROTEIN_TYPES, str(n))):
        for protein in graph[middle]:
            if _type(graph, protein) in PROTEIN_TYPES and protein not in paths:
                paths[protein] = (drug, middle, protein)
    return {"drug": drug, "proteins": set(paths), "direct": direct, "paths": paths}


def _relation(graph, u, v):
    a, b = _type(graph, u), _type(graph, v)
    if a == b == "drug":
        return "DDI"
    if a in PROTEIN_TYPES and b in PROTEIN_TYPES:
        return "PP"
    if "drug" in (a, b) and (a in PROTEIN_TYPES or b in PROTEIN_TYPES):
        return "DT"
    raise ValueError(f"unsupported edge types in protein view: {a}, {b}")


def _edges(graph, context):
    """Selected complete witness paths, plus induced P-P context edges."""
    result = {}
    for path in context["paths"].values():
        for u, v in zip(path, path[1:]):
            edge = tuple(sorted((u, v), key=str))
            result[edge] = _relation(graph, *edge)
    proteins = context["proteins"]
    for u in sorted(proteins, key=str):
        for v in graph[u]:
            if v in proteins and str(u) < str(v):
                result[(u, v)] = "PP"
    return result
