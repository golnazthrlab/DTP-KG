"""Prepare shared private tables from explicit local inputs.

DrugBank/MeSH XML and the negative supplement are parsed locally. DPI and PPI
must already be prepared interaction tables; this module does not reconstruct
their upstream extraction. Network construction, MeSH embeddings and inductive
holdout-specific representations belong to their subsequent experiment tasks.
"""
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from dtpkg import ddi_labels
from dtpkg.drugbank.preparation import (
    prepare_modality_coverage, extract_approved_drug_mesh,
    parse_mesh_hierarchy, extend_drug_mesh_annotations,
)
from dtpkg.network import preparation as network_preparation


MANIFEST = "preparation_manifest.json"
SUMMARY = "preparation_summary.json"
CODE_FILES = (
    "preparation.py", "drugbank/preparation.py", "mesh_scopes.py",
    "network/preparation.py", "negative_inputs.py", "ddi_labels.py",
)


def _output_files():
    outputs = dict(
        coverage="drugbank/coverage_analysis_cache.csv",
        approved_mesh="mesh/approved_drugbank_mesh.csv",
        mesh_hierarchy="mesh/mesh_hierarchy.csv",
        annotations="mesh/extended_drug_info.csv",
    )
    outputs.update({name: relative for name, relative in network_preparation.OUTPUTS.items()
                    if name != "summary"})
    outputs["label_summary"] = network_preparation.OUTPUTS["summary"]
    outputs["network_manifest"] = network_preparation.MANIFEST
    outputs["summary"] = SUMMARY
    return outputs


def _hashes(paths):
    return {name: ddi_labels.sha256(path) for name, path in paths.items()}


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n")


def _row_count(path):
    return sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=100_000))


def prepare_data(data_dir, *, drugbank_xml, mesh_xml, negative_source, dpi_csv,
                 ppi_csv):
    """Return paths to the prepared shared tables, their summary and manifest.

    ``negative_source`` accepts the publisher's Additional file 2 XLSX (Table
    S6 seeds) or a prepared ``drug1,drug2`` CSV. ``dpi_csv`` and ``ppi_csv`` are
    prepared CSVs with ``db_id,targets`` and ``uniprot_id,interactions`` columns.
    Downloads can remain outside ``data_dir``; raw XML is never copied.

    Matching completed outputs are reused after checking input, source-code,
    dependency and output hashes. Changed inputs or outputs require a new data directory.
    All generation and destination checks finish before any final file is added.
    """
    root = Path(data_dir).expanduser().resolve()
    sources = {name: Path(path).expanduser().resolve() for name, path in dict(
        drugbank_xml=drugbank_xml, mesh_xml=mesh_xml, negative_source=negative_source,
        dpi_csv=dpi_csv, ppi_csv=ppi_csv).items()}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    package = Path(__file__).resolve().parent
    code_paths = {name: package / name for name in CODE_FILES}
    identity = dict(format_version=1, kind="shared_data_preparation", sources=_hashes(sources),
                    code=_hashes(code_paths),
                    dependencies={name: version(name) for name in ("numpy", "pandas")})
    relative = _output_files()
    paths = {name: root / filename for name, filename in relative.items()}
    paths["manifest"] = root / MANIFEST
    if paths["manifest"].exists():
        saved = json.loads(paths["manifest"].read_text())
        if saved.get("identity") != identity or set(saved.get("outputs", {})) != set(relative.values()):
            raise ValueError("preparation inputs, code or dependencies changed; use a new data_dir")
        for filename, expected in saved["outputs"].items():
            path = root / filename
            if not path.is_file() or ddi_labels.sha256(path) != expected:
                raise ValueError(f"prepared output changed: {path}; use a new data_dir")
        return paths

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".data_preparation_", dir=root) as temporary:
        stage = Path(temporary)
        outputs = {name: stage / filename for name, filename in relative.items()}
        coverage = prepare_modality_coverage(sources["drugbank_xml"], output_csv=outputs["coverage"])
        approved = extract_approved_drug_mesh(sources["drugbank_xml"], output_csv=outputs["approved_mesh"])
        hierarchy = parse_mesh_hierarchy(sources["mesh_xml"], output_csv=outputs["mesh_hierarchy"])
        annotations = extend_drug_mesh_annotations(approved, hierarchy, output_csv=outputs["annotations"])
        network_preparation.prepare_network_inputs(stage, **{name: sources[name] for name in (
            "drugbank_xml", "negative_source", "dpi_csv", "ppi_csv")})

        # The inner preparer used the staging root. Final reports must name the
        # durable output paths, and its manifest must hash that rewritten report.
        label_summary = network_preparation._rewrite_paths(
            json.loads(outputs["label_summary"].read_text()), stage, root)
        _write_json(outputs["label_summary"], label_summary)
        network_manifest = network_preparation._rewrite_paths(
            json.loads(outputs["network_manifest"].read_text()), stage, root)
        network_manifest["outputs"] = {
            filename: ddi_labels.sha256(stage / filename)
            for filename in network_preparation.OUTPUTS.values()}
        _write_json(outputs["network_manifest"], network_manifest)

        counts = dict(coverage=len(coverage), approved_mesh=len(approved),
                      mesh_hierarchy=len(hierarchy), annotations=len(annotations),
                      negative_input=label_summary["negatives"]["rows"],
                      positives=label_summary["after"]["positives"],
                      negatives=label_summary["after"]["negatives"])
        datasets = []
        for name, filename in relative.items():
            if not filename.endswith(".csv"):
                continue
            row = dict(dataset=name, rows=counts[name] if name in counts else _row_count(outputs[name]),
                       path=filename)
            if name == "annotations":
                row["drugs"] = int(annotations.drugbank_id.nunique())
            datasets.append(row)
        export_info = label_summary.get("drugbank_export", {})
        _write_json(outputs["summary"], dict(
            datasets=datasets,
            drugbank_export={name: export_info.get(name) for name in ("version", "exported_on")},
            interaction_inputs="supplied prepared DPI/PPI tables",
        ))

        if _hashes(sources) != identity["sources"] or _hashes(code_paths) != identity["code"]:
            raise ValueError("preparation inputs or code changed during execution; rerun preparation")
        hashes = {filename: ddi_labels.sha256(stage / filename) for filename in relative.values()}
        for filename, digest in hashes.items():
            target = root / filename
            if target.exists() and (not target.is_file() or ddi_labels.sha256(target) != digest):
                raise FileExistsError(f"different output already exists: {target}; use a new data_dir")
        manifest = dict(identity=identity, source_paths={name: str(path) for name, path in sources.items()},
                        outputs=hashes)
        _write_json(stage / MANIFEST, manifest)
        # Same-filesystem hard links publish complete files without a second
        # copy or overwriting existing files; the completion manifest is last.
        for filename in (*relative.values(), MANIFEST):
            target = root / filename
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(stage / filename, target)
    return paths
