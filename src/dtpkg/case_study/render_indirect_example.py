"""Render comparative protein neighborhoods with concise node identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np


BLUE = "#2184B3"
ORANGE = "#DE982E"
BLUE_LIGHT = "#A6CDDF"
ORANGE_LIGHT = "#ECCD9F"
RED = "#C3324B"
DT_COLOR = "#5D7180"
PP_COLOR = "#BAC2C8"


def _validate(data):
    records = {node["id"]: node for node in data["nodes"]}
    if len(records) != len(data["nodes"]):
        raise ValueError("duplicate saved node")
    graph = nx.Graph()
    graph.add_nodes_from(records)
    for edge in data["edges"]:
        a, b = edge["source"], edge["target"]
        if a not in records or b not in records or a == b:
            raise ValueError("invalid edge endpoints")
        if edge["relation"] not in ("DT", "PP"):
            raise ValueError("the illustration requires DDI-free DT/PP edges")
        if graph.has_edge(a, b):
            raise ValueError("duplicate saved edge")
        graph.add_edge(a, b, relation=edge["relation"],
                       weight=2.3 if edge["relation"] == "DT" else 1.)
    if any(drug not in records for drug in (data["drug1"], data["drug2"])):
        raise ValueError("missing focal drug")
    if (data["drug1"] == data["drug2"] or data["scenario"] != "seen_unseen"
            or data["drug1_heldout"] or not data["drug2_heldout"]):
        raise ValueError("panels require the seen drug first and the held-out drug second")
    for node in records.values():
        if (bool(node["is_drug1"]) != (node["id"] == data["drug1"])
                or bool(node["is_drug2"]) != (node["id"] == data["drug2"])):
            raise ValueError("focal drug flags disagree with node identities")
        focal = node["is_drug1"] or node["is_drug2"]
        if node["type"] not in ({"drug"} if focal else {"protein", "target"}):
            raise ValueError("node type disagrees with its drug/protein role")
    for a, b, attributes in graph.edges(data=True):
        types = {records[a]["type"], records[b]["type"]}
        expected = "DT" if "drug" in types else "PP"
        if (records[a]["type"] == records[b]["type"] == "drug"
                or attributes["relation"] != expected):
            raise ValueError("edge relation disagrees with endpoint types")
    proteins = [node for node in records.values() if not node["is_drug1"] and not node["is_drug2"]]
    if len(proteins) != data["protein_union_size"]:
        raise ValueError("saved protein union does not match node inventory")
    if data["displayed_proteins"] != len(proteins) or data["omitted_proteins"] != 0:
        raise ValueError("the illustration must display the complete protein union")
    checks = {"n_targets_a": "direct_target_a", "n_targets_b": "direct_target_b",
              "n_proteins_a": "neighborhood_a", "n_proteins_b": "neighborhood_b",
              "shared_protein_count": "shared"}
    for count, flag in checks.items():
        if sum(bool(node[flag]) for node in proteins) != data[count]:
            raise ValueError(f"saved {count} differs from node inventory")
    if sum(node["direct_target_a"] and node["direct_target_b"] for node in proteins) != data["shared_target_count"]:
        raise ValueError("saved shared direct-target count differs from node inventory")
    for node in proteins:
        if not (node["neighborhood_a"] or node["neighborhood_b"]):
            raise ValueError("protein outside both endpoint neighborhoods")
        if bool(node["shared"]) != bool(node["neighborhood_a"] and node["neighborhood_b"]):
            raise ValueError("shared membership differs from endpoint neighborhoods")
    from .context import _context
    nx.set_node_attributes(graph, {node: record["type"] for node, record in records.items()}, "type")
    for suffix, drug in (("a", data["drug1"]), ("b", data["drug2"])):
        context = _context(graph, drug)
        if (context["proteins"] != {node["id"] for node in proteins if node[f"neighborhood_{suffix}"]}
                or context["direct"] != {node["id"] for node in proteins if node[f"direct_target_{suffix}"]}):
            raise ValueError("saved endpoint membership differs from reconstructed two-edge context")
    paths = data["shared_paths"]
    if {entry["protein"] for entry in paths} != {node["id"] for node in proteins if node["shared"]}:
        raise ValueError("shared paths do not cover precisely the shared proteins")
    for entry in paths:
        for field, drug in (("path_a", data["drug1"]), ("path_b", data["drug2"])):
            path = entry[field]
            if not 2 <= len(path) <= 3 or path[0] != drug or path[-1] != entry["protein"]:
                raise ValueError("invalid drug-rooted two-edge witness path")
            if any(not graph.has_edge(a, b) for a, b in zip(path, path[1:])):
                raise ValueError("a witness path contains an unsaved edge")
    return records, graph


def _color(node):
    if node["shared"]:
        return RED
    if node["is_drug1"] or node["direct_target_a"]:
        return BLUE
    if node["is_drug2"] or node["direct_target_b"]:
        return ORANGE
    return BLUE_LIGHT if node["neighborhood_a"] else ORANGE_LIGHT


def _shape(node):
    if node["is_drug1"] or node["is_drug2"]:
        return "s"
    return "D" if node["direct_target_a"] or node["direct_target_b"] else "o"


def _layout(data, records, graph):
    """Use common geometric anchors and force parameters, without rescaling."""
    rng = np.random.default_rng(42)
    positions = {}
    for node in sorted(graph):
        side = -.65 if records[node]["neighborhood_a"] else .65
        positions[node] = np.array([side, 0.]) + rng.normal(0, .32, size=2)
    drug1, drug2 = data["drug1"], data["drug2"]
    positions[drug1], positions[drug2] = np.array([-1.6, 0.]), np.array([1.6, 0.])
    shared = sorted(node for node in graph if records[node]["shared"])
    for i, node in enumerate(shared):
        positions[node] = np.array([0., .28 * ((len(shared) - 1) / 2 - i)])
    return nx.spring_layout(graph, pos=positions, fixed=[drug1, drug2, *shared],
                            seed=42, k=.23, iterations=400, weight="weight")


def _draw(ax, data, records, graph, positions):
    for relation, color, width, alpha in (("PP", PP_COLOR, .65, .48),
                                           ("DT", DT_COLOR, 1.55, .85)):
        edges = [(a, b) for a, b, attributes in graph.edges(data=True)
                 if attributes["relation"] == relation]
        nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=edges,
                               edge_color=color, width=width, alpha=alpha)
    for drug, color in ((data["drug1"], BLUE), (data["drug2"], ORANGE)):
        edges = [(a, b) for a, b, attributes in graph.edges(data=True)
                 if attributes["relation"] == "DT" and drug in (a, b)]
        lines = nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=edges,
                                       edge_color=color, width=1.8, alpha=.9)
        lines.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    # Shared nodes and focal drugs are drawn last to retain their visibility.
    for shared in (False, True):
        for shape in ("o", "D", "s"):
            ids = [node for node in graph if bool(records[node]["shared"]) == shared
                   and _shape(records[node]) == shape]
            if not ids:
                continue
            size = 290 if shape == "s" else 96 if shape == "D" else 96 if shared else 39
            nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=ids,
                                   node_color=[_color(records[node]) for node in ids],
                                   node_size=size, node_shape=shape, edgecolors="white",
                                   linewidths=.95 if shared or shape != "o" else .65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _annotate_nodes(ax, data, records, positions):
    """Label the drugs, shared proteins and targets on their witness routes.

    Short leaders join labels to their actual nodes; they are annotation marks,
    not new graph edges. Placement avoids previously placed labels and markers.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    points = {node: ax.transData.transform(position) for node, position in positions.items()}
    bounds = ax.get_window_extent()
    used = []
    drugs = {data["drug1"]: data["name1"], data["drug2"]: data["name2"]}
    intermediates = {entry[key][1] for entry in data["shared_paths"]
                     for key in ("path_a", "path_b") if len(entry[key]) == 3}
    shared = sorted(node for node in records if records[node]["shared"])
    ordered = list(drugs) + shared + sorted(intermediates - set(shared) - set(drugs))
    for node in ordered:
        is_drug = node in drugs
        text = f"{drugs[node]}\n{node}" if is_drug else node
        fontsize = 14 if is_drug else 12
        font = FontProperties(family="DejaVu Sans", size=fontsize, weight="bold")
        lines = text.splitlines()
        width = max(renderer.get_text_width_height_descent(line, font, False)[0] for line in lines) + 5 * scale
        height = fontsize * 1.25 * len(lines) * scale + 4 * scale
        candidates = []
        for radius in ((27, 37, 50, 65, 80, 100) if is_drug else (17, 25, 36, 49, 65, 80, 100)):
            for dx, dy in ((radius, radius/2), (-radius, radius/2),
                           (radius, -radius/2), (-radius, -radius/2),
                           (0, radius), (0, -radius), (radius, 0), (-radius, 0)):
                cx, cy = points[node] + np.array([dx, dy]) * scale
                box = (cx-width/2, cy-height/2, cx+width/2, cy+height/2)
                collisions = sum(not (box[2] < other[0] or box[0] > other[2]
                                      or box[3] < other[1] or box[1] > other[3]) for other in used)
                for other, (px, py) in points.items():
                    radius_px = (9 if other in drugs else 5) * scale
                    collisions += (box[0]-radius_px <= px <= box[2]+radius_px
                                   and box[1]-radius_px <= py <= box[3]+radius_px)
                outside = box[0] < bounds.x0 or box[2] > bounds.x1 or box[1] < bounds.y0 or box[3] > bounds.y1
                cost = 10000 * outside + 1000 * collisions + np.hypot(dx, dy)
                candidates.append((cost, dx, dy, box))
        _, dx, dy, box = min(candidates, key=lambda item: item[0])
        used.append(box)
        ax.annotate(text, positions[node], xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=fontsize, fontweight="bold",
                    color=_color(records[node]),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.94, pad=1),
                    arrowprops=dict(arrowstyle="-", color="#8E9AA3", lw=.55,
                                    shrinkA=2, shrinkB=8 if is_drug else 5),
                    zorder=8)
    return len(ordered)


def render_example(data_file, output_prefix):
    """Write positive (left) and negative (right) views as PNG and vector PDF."""
    bundle = json.loads(Path(data_file).read_text())
    panels = []
    for name in ("positive", "negative"):
        data = bundle[name]
        if data["label"] != (1 if name == "positive" else 0):
            raise ValueError("positive/negative panel labels are reversed")
        records, graph = _validate(data)
        panels.append((data, records, graph, _layout(data, records, graph)))
    coords = np.concatenate([np.array(list(panel[3].values())) for panel in panels])
    xlimit = float(np.abs(coords[:, 0]).max()) + .18
    ylimit = float(np.abs(coords[:, 1]).max()) + .18
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), facecolor="white")
    fig.subplots_adjust(left=.022, right=.978, top=.985, bottom=.18, wspace=.035)
    for ax, (data, records, graph, positions) in zip(axes, panels):
        _draw(ax, data, records, graph, positions)
        ax.set_xlim(-xlimit, xlimit)
        ax.set_ylim(-ylimit, ylimit)
    label_counts = [_annotate_nodes(ax, data, records, positions)
                    for ax, (data, records, _, positions) in zip(axes, panels)]
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="Seen-drug neighborhood"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=7, label="Held-out-drug neighborhood"),
        Line2D([], [], marker="o", linestyle="", color=RED, markersize=8, label="Shared protein"),
        Line2D([], [], marker="s", linestyle="", color="#52616B", markersize=9, label="Drug"),
        Line2D([], [], marker="D", linestyle="", color="#52616B", markersize=7, label="Direct target"),
        Line2D([], [], marker="o", linestyle="", color="#52616B", markersize=6, label="Other protein"),
        Line2D([], [], color=DT_COLOR, linewidth=1.55, label="Drug–target edge"),
        Line2D([], [], color=PP_COLOR, linewidth=.9, label="Protein–protein edge"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=4, frameon=False, fontsize=12.5, columnspacing=2.0,
               handletextpad=.8, labelspacing=1.3)
    # Only node identities and the legend belong in the image. Explanatory
    # paragraphs, scores and counts remain in the separate caption.
    if fig.texts or any(len(ax.texts) != count for ax, count in zip(axes, label_counts)):
        raise ValueError("unexpected explanatory text in the comparative figure")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {extension: prefix.with_suffix("." + extension) for extension in ("png", "pdf")}
    for extension, path in outputs.items():
        fig.savefig(path, dpi=240 if extension == "png" else 100, facecolor="white")
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    for kind, path in render_example(args.data_file, args.output_prefix).items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    main()
