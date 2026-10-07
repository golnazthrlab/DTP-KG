"""Prepare graph inputs from explicit local files without changing their sources.

DPI/PPI are prepared interaction CSVs: their original upstream selection rules
are unavailable, so this module validates and copies them rather than inferring
new DrugBank/UniProt filters. All generated records remain in ``data_dir``.
"""
import ast
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from dtpkg import ddi_labels


OUTPUTS = {
    "dpi_csv": "interactions/DPI_enriched.csv",
    "ppi_csv": "interactions/PPI_enriched.csv",
    "negative_input": "interactions/unresolved/DDI_negative_pairs.csv",
    "release_ddi": "drugbank/drugbank_ddi_release.csv",
    "release_drugs": "drugbank/drugbank_drugs_release.csv",
    "positives": "interactions/DDI_positive_pairs.csv",
    "negatives": "interactions/DDI_negative_pairs.csv",
    "summary": "interactions/label_audit/label_audit_summary.json",
}
MANIFEST = "interactions/network_preparation.json"


def _table(path, columns):
    if not set(columns).issubset(pd.read_csv(path, nrows=0).columns):
        raise ValueError(f"{path}: CSV with columns {columns} required")
    frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"{path}: nonempty CSV with columns {columns} required")
    for name in columns:
        if frame[name].str.strip().eq("").any():
            raise ValueError(f"{path}: empty {name} value")
    return frame


def _interaction_table(path, id_column, list_column):
    frame = _table(path, [id_column, list_column])
    if frame[id_column].duplicated().any() or not frame[id_column].eq(frame[id_column].str.strip()).all():
        raise ValueError(f"{path}: unique, whitespace-free {id_column} values required")
    def check_list(value):
        if not isinstance(value, list):
            raise ValueError(f"{path}: {list_column} must serialize a list of accessions")
        for item in value:
            if isinstance(item, list):
                check_list(item)
            elif not isinstance(item, str) or not item.strip() or item != item.strip():
                raise ValueError(f"{path}: {list_column} contains an invalid accession")
    for value in frame[list_column]:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"{path}: malformed serialized {list_column} list") from error
        check_list(parsed)


def _rewrite_paths(value, temporary, destination):
    if isinstance(value, dict):
        return {_rewrite_paths(key, temporary, destination): _rewrite_paths(item, temporary, destination)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, temporary, destination) for item in value]
    if isinstance(value, str) and value.startswith(str(temporary) + "/"):
        return str(destination) + value[len(str(temporary)):]
    return value


def prepare_network_inputs(data_dir, *, drugbank_xml, negative_source, dpi_csv, ppi_csv):
    """Create the local inputs consumed by network construction and training.

    ``negative_source`` is either Zheng Additional file 2 (XLSX, Table S6), or
    a comma-separated CSV with DrugBank-accession ``drug1,drug2`` columns.
    XLSX seeds are restricted to the supplied DPI drug roster; CSV rows are
    used as supplied. Both then receive the study's canonicalization, XML
    conflict screening, and quarantine policy. Positives use all XML DDIs.

    ``dpi_csv`` requires ``db_id,targets``; ``ppi_csv`` requires
    ``uniprot_id,interactions``. Interaction fields serialize lists, optionally
    nested, of UniProt accessions. Additional enrichment columns are retained
    but not used by graph construction. These must already be prepared tables.

    Matching completed outputs are reused. Different existing outputs or
    source/code changes are refused; choose a new data directory in that case.
    Returns paths to the eight outputs and the preparation manifest.
    """
    root = Path(data_dir).expanduser().resolve()
    sources = {name: Path(path).expanduser().resolve() for name, path in dict(
        drugbank_xml=drugbank_xml, negative_source=negative_source, dpi_csv=dpi_csv, ppi_csv=ppi_csv).items()}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sources["negative_source"].suffix.lower() not in (".csv", ".xlsx"):
        raise ValueError("negative_source must be a drug1/drug2 CSV or the publisher XLSX")
    source_hashes = {name: ddi_labels.sha256(path) for name, path in sources.items()}
    package = Path(__file__).resolve().parents[1]
    code = {name: ddi_labels.sha256(package / name)
            for name in ("network/preparation.py", "negative_inputs.py", "ddi_labels.py")}
    identity = dict(format_version=1, kind="network_input_preparation", sources=source_hashes, code=code)
    paths = {name: root / relative for name, relative in OUTPUTS.items()}
    paths["manifest"] = root / MANIFEST
    if paths["manifest"].exists():
        saved = json.loads(paths["manifest"].read_text())
        if saved.get("identity") != identity or set(saved.get("outputs", {})) != set(OUTPUTS.values()):
            raise ValueError("preparation inputs or code changed; use a new data_dir")
        for relative, expected in saved["outputs"].items():
            path = root / relative
            if not path.is_file() or ddi_labels.sha256(path) != expected:
                raise ValueError(f"prepared output changed: {path}; use a new data_dir")
        return paths
    _interaction_table(sources["dpi_csv"], "db_id", "targets")
    _interaction_table(sources["ppi_csv"], "uniprot_id", "interactions")
    if sources["negative_source"].suffix.lower() == ".csv":
        _table(sources["negative_source"], ["drug1", "drug2"])
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".network_preparation_", dir=root) as temporary:
        stage = Path(temporary)
        outputs = {name: stage / relative for name, relative in OUTPUTS.items()}
        for path in outputs.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for name in ("dpi_csv", "ppi_csv"):
            shutil.copyfile(sources[name], outputs[name])
        if sources["negative_source"].suffix.lower() == ".xlsx":
            from dtpkg.negative_inputs import prepare_seed_negatives
            prepare_seed_negatives(sources["negative_source"], outputs["dpi_csv"], outputs["negative_input"])
        else:
            shutil.copyfile(sources["negative_source"], outputs["negative_input"])
        ddi_labels.extract_release_ddis(sources["drugbank_xml"], outputs["release_ddi"], outputs["release_drugs"])
        _table(outputs["release_ddi"], ["drug1", "drug2"])
        _table(outputs["release_drugs"], ["drugbank_id", "type"])
        report = ddi_labels.build_resolved_datasets(raw_neg=outputs["negative_input"],
            release_ddi=outputs["release_ddi"], release_drugs=outputs["release_drugs"],
            dpi_path=outputs["dpi_csv"], xml_path=sources["drugbank_xml"],
            out_pos=outputs["positives"], out_neg=outputs["negatives"], summary=outputs["summary"],
            positive_source="xml", restrict_to_dpi=False, write_audit=False)
        outputs["summary"].write_text(json.dumps(_rewrite_paths(report, stage, root), indent=2) + "\n")
        for name, source in sources.items():
            if ddi_labels.sha256(source) != source_hashes[name]:
                raise ValueError(f"source changed during preparation: {source}")
        if any(ddi_labels.sha256(package / name) != digest for name, digest in code.items()):
            raise ValueError("preparation code changed during execution; rerun preparation")
        hashes = {relative: ddi_labels.sha256(stage / relative) for relative in OUTPUTS.values()}
        # Check every destination before publishing any artifact.
        for relative, digest in hashes.items():
            target = root / relative
            if target.exists() and (not target.is_file() or ddi_labels.sha256(target) != digest):
                raise FileExistsError(f"different output already exists: {target}; use a new data_dir")
        for relative in OUTPUTS.values():
            target = root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle, (stage / relative).open("rb") as source:
                    shutil.copyfileobj(source, handle)
        with paths["manifest"].open("x") as handle:
            json.dump(dict(identity=identity, source_paths={name: str(path) for name, path in sources.items()},
                           outputs=hashes), handle, indent=2)
            handle.write("\n")
    return paths
