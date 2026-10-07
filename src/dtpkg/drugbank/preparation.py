"""Prepare private inputs for the DrugBank exploration figures.

These functions reconstruct the analysis inputs from separately obtained
DrugBank and MeSH XML. They do not download or redistribute data. Coverage uses
all small-molecule and biotech records; MeSH annotations use approved drugs,
branch D tree numbers, and their ancestors. No embeddings are built here.

Call these functions after the notebook's path setup cell. All functions
return DataFrames; ``output_csv`` optionally saves a private CSV.
Existing outputs are protected unless ``overwrite=True`` is explicit.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from dtpkg.mesh_scopes import knowledge_category


# Field-presence definitions from the study's coverage analysis. Multi-valued
# fields require a child, text fields require nonblank text, properties require
# a nonblank value, and external IDs require the named resource. In particular,
# the historical sequence flag denotes a <sequence> element of any type.
TEXT_FIELDS = (
    "description", "indication", "mechanism-of-action", "pharmacodynamics",
    "toxicity", "metabolism", "absorption", "half-life", "protein-binding",
    "route-of-elimination", "volume-of-distribution", "clearance", "state",
    "cas-number", "unii", "synthesis-reference",
)
MULTI_FIELDS = {
    "targets": "targets/target",
    "enzymes": "enzymes/enzyme",
    "transporters": "transporters/transporter",
    "carriers": "carriers/carrier",
    "pathways": "pathways/pathway",
    "reactions": "reactions/reaction",
    "mesh-categories": "categories/category",
    "atc-codes": "atc-codes/atc-code",
    "drug-interactions": "drug-interactions/drug-interaction",
    "food-interactions": "food-interactions/food-interaction",
    "dosages": "dosages/dosage",
    "synonyms": "synonyms/synonym",
    "aa-sequence": "sequences/sequence",
    "affected-organisms": "affected-organisms/affected-organism",
    "pdb-entries": "pdb-entries/pdb-entry",
    "snp-effects": "snp-effects/effect",
    "snp-adverse-drug-reactions": "snp-adverse-drug-reactions/reaction",
    "patents": "patents/patent",
    "ahfs-codes": "ahfs-codes/ahfs-code",
    "external-links": "external-links/external-link",
    "salts": "salts/salt",
    "products": "products/product",
}
PROPERTY_FIELDS = {
    "smiles": "SMILES", "inchikey": "InChIKey",
    "molecular-formula": "Molecular Formula",
}
EXTERNAL_ID_FIELDS = {
    "pubchem-cid": "PubChem Compound", "chembl-id": "ChEMBL",
    "uniprot-id": "UniProtKB", "wikipedia": "Wikipedia",
    "kegg-drug": "KEGG Drug", "pharmgkb-id": "PharmGKB",
    "ttd-id": "Therapeutic Targets Database", "pdb-extid": "PDB",
    "chebi-id": "ChEBI", "rxcui-id": "RxCUI",
}
COVERAGE_FEATURES = (
    *TEXT_FIELDS, *MULTI_FIELDS, *PROPERTY_FIELDS, *EXTERNAL_ID_FIELDS,
)
APPROVED_COLUMNS = ["drugbank_id", "name", "drug_type", "mesh_terms", "mesh_ids"]
HIERARCHY_COLUMNS = ["mesh_id", "mesh_name", "tree_number", "level"]
ANNOTATION_COLUMNS = [
    "drugbank_id", "tree_number", "level", "deepest_level", "knowledge_category",
]
TREE_NUMBER = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9]{3})*")


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _prefix(element: ET.Element) -> str:
    return element.tag.split("}", 1)[0] + "}" if "}" in element.tag else ""


def _xml_path(prefix: str, path: str) -> str:
    return "/".join(prefix + part for part in path.split("/"))


def _text(element: ET.Element, prefix: str, path: str) -> str:
    return (element.findtext(_xml_path(prefix, path)) or "").strip()


def _records(xml_path: str | Path, root_name: str, record_name: str) -> Iterator[ET.Element]:
    """Stream only direct children; nested <drug> references are not records."""
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input XML does not exist: {path}")
    with path.open("rb") as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        _, root = next(context)
        if _tag_name(root.tag) != root_name:
            raise ValueError(f"Expected XML root <{root_name}>")
        depth = 1
        for event, element in context:
            if event == "start":
                depth += 1
                continue
            if depth == 2:
                if _tag_name(element.tag) == record_name:
                    yield element
                root.remove(element)
                element.clear()
            depth -= 1


def _primary_id(drug: ET.Element, prefix: str) -> str | None:
    element = drug.find(f'{prefix}drugbank-id[@primary="true"]')
    if element is None:
        return None
    value = (element.text or "").strip()
    if not value:
        raise ValueError("DrugBank record contains an empty primary ID")
    return value


def _check_output(output_csv, inputs=(), *, overwrite=False) -> None:
    if output_csv is None:
        return
    output = Path(output_csv)
    if any(output.resolve() == Path(path).resolve() for path in inputs):
        raise ValueError("Output path must differ from every input path")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use overwrite=True to replace: {output}")


def _save(frame: pd.DataFrame, output_csv, *, overwrite=False) -> pd.DataFrame:
    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, mode="w" if overwrite else "x")
    return frame


def prepare_modality_coverage(drugbank_xml, output_csv=None, *, overwrite=False) -> pd.DataFrame:
    """Return ``db_id, type, approved`` and the 51 study field-presence flags.

    All small-molecule and biotech entries with a primary ID are included,
    regardless of approval status. Unused drug-name and heuristic subtype
    columns from the exploratory cache are intentionally omitted.
    """
    _check_output(output_csv, [drugbank_xml], overwrite=overwrite)
    rows = []
    for drug in _records(drugbank_xml, "drugbank", "drug"):
        prefix = _prefix(drug)
        drug_id = _primary_id(drug, prefix)
        drug_type = drug.get("type", "unknown")
        if drug_id is None or drug_type not in ("small molecule", "biotech"):
            continue
        groups = [g.text for g in drug.findall(_xml_path(prefix, "groups/group"))]
        row = {"db_id": drug_id, "type": drug_type, "approved": "approved" in groups}
        for name in TEXT_FIELDS:
            row[name] = bool(_text(drug, prefix, name))
        for name, path in MULTI_FIELDS.items():
            row[name] = drug.find(_xml_path(prefix, path)) is not None
        properties = {
            _text(prop, prefix, "kind")
            for prop in drug.findall(_xml_path(prefix, "calculated-properties/property"))
            if _text(prop, prefix, "value")
        }
        for name, kind in PROPERTY_FIELDS.items():
            row[name] = kind in properties
        resources = {
            _text(ext, prefix, "resource")
            for ext in drug.findall(_xml_path(prefix, "external-identifiers/external-identifier"))
        }
        for name, resource in EXTERNAL_ID_FIELDS.items():
            row[name] = resource in resources
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["db_id", "type", "approved", *COVERAGE_FEATURES])
    if frame.empty:
        raise ValueError("No small-molecule or biotech records with primary IDs were found")
    if frame["db_id"].duplicated().any():
        raise ValueError("Duplicate DrugBank primary IDs in coverage input")
    return _save(frame, output_csv, overwrite=overwrite)


def extract_approved_drug_mesh(drugbank_xml, output_csv=None, *, overwrite=False) -> pd.DataFrame:
    """Extract approved records, including drugs without MeSH assignments.

    ``mesh_terms`` and ``mesh_ids`` retain the source category order, joined by
    ``|``. They are extracted independently, as in the original preparation;
    do not assume their positions form name/ID pairs when a category lacks an ID.
    """
    _check_output(output_csv, [drugbank_xml], overwrite=overwrite)
    rows = []
    for drug in _records(drugbank_xml, "drugbank", "drug"):
        prefix = _prefix(drug)
        groups = [g.text for g in drug.findall(_xml_path(prefix, "groups/group"))]
        if "approved" not in groups:
            continue
        drug_id = _primary_id(drug, prefix)
        if drug_id is None:
            continue
        terms, ids = [], []
        for category in drug.findall(_xml_path(prefix, "categories/category")):
            term = _text(category, prefix, "category")
            mesh_id = _text(category, prefix, "mesh-id")
            if term:
                terms.append(term)
            if mesh_id:
                ids.append(mesh_id)
        rows.append({
            "drugbank_id": drug_id, "name": _text(drug, prefix, "name") or None,
            "drug_type": drug.get("type", "unknown"),
            "mesh_terms": "|".join(terms) or None,
            "mesh_ids": "|".join(ids) or None,
        })
    frame = pd.DataFrame(rows, columns=APPROVED_COLUMNS)
    if frame.empty:
        raise ValueError("No approved DrugBank records with primary IDs were found")
    if frame["drugbank_id"].duplicated().any():
        raise ValueError("Duplicate DrugBank primary IDs in approved-drug input")
    return _save(frame, output_csv, overwrite=overwrite)


def parse_mesh_hierarchy(mesh_xml, output_csv=None, *, overwrite=False) -> pd.DataFrame:
    """Extract all descriptor tree positions from the NLM descriptor XML.

    A descriptor can have multiple tree numbers. Depth is the number of
    dot-separated components, including the first component (e.g. D01 is 1).
    All branches are retained; branch D is selected in annotation expansion.
    """
    _check_output(output_csv, [mesh_xml], overwrite=overwrite)
    rows = []
    for record in _records(mesh_xml, "DescriptorRecordSet", "DescriptorRecord"):
        prefix = _prefix(record)
        mesh_id = _text(record, prefix, "DescriptorUI")
        mesh_name = _text(record, prefix, "DescriptorName/String")
        for element in record.findall(_xml_path(prefix, "TreeNumberList/TreeNumber")):
            tree_number = (element.text or "").strip()
            if not mesh_id or not mesh_name or not TREE_NUMBER.fullmatch(tree_number):
                raise ValueError("MeSH descriptor has a missing ID/name or invalid tree number")
            rows.append((mesh_id, mesh_name, tree_number, tree_number.count(".") + 1))
    frame = pd.DataFrame(rows, columns=HIERARCHY_COLUMNS)
    if frame.empty:
        raise ValueError("No descriptor tree numbers were found in MeSH XML")
    _validate_hierarchy(frame)
    return _save(frame, output_csv, overwrite=overwrite)


def _read_frame(table) -> pd.DataFrame:
    return table.copy() if isinstance(table, pd.DataFrame) else pd.read_csv(table)


def _require_columns(frame, columns, label) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _validate_hierarchy(hierarchy) -> None:
    _require_columns(hierarchy, ["mesh_id", "tree_number", "level"], "MeSH hierarchy")
    if hierarchy[["mesh_id", "tree_number", "level"]].isna().any().any():
        raise ValueError("MeSH hierarchy contains missing IDs, tree numbers, or levels")
    valid_trees = hierarchy["tree_number"].astype(str).str.fullmatch(TREE_NUMBER)
    if not valid_trees.all():
        raise ValueError("MeSH hierarchy contains invalid tree numbers")
    levels = pd.to_numeric(hierarchy["level"], errors="raise")
    expected = hierarchy["tree_number"].str.count(r"\.") + 1
    if not levels.eq(expected).all():
        raise ValueError("MeSH hierarchy levels do not match tree-number depths")
    if hierarchy["tree_number"].duplicated().any():
        raise ValueError("MeSH hierarchy contains duplicate tree numbers")


def extend_drug_mesh_annotations(approved, hierarchy, output_csv=None, *, overwrite=False) -> pd.DataFrame:
    """Map approved drugs to branch D and add every tree-number prefix.

    Inputs may be CSV paths or DataFrames. Unmapped IDs and drugs without a
    branch D assignment do not enter the annotation population. Duplicate
    assignments are collapsed. Ancestors are syntactic tree-number prefixes,
    including prefixes without a separate descriptor record, matching the
    study workflow. The per-drug deepest level sets its knowledge category:
    Low <= 5, Mid 6–7, Deep 8–10. These categories differ from term-depth scopes.
    """
    paths = [value for value in (approved, hierarchy) if not isinstance(value, pd.DataFrame)]
    _check_output(output_csv, paths, overwrite=overwrite)
    drugs, tree = _read_frame(approved), _read_frame(hierarchy)
    _require_columns(drugs, ["drugbank_id", "mesh_ids"], "Approved drug table")
    if drugs["drugbank_id"].isna().any() or drugs["drugbank_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Approved drug table contains missing drug IDs")
    _validate_hierarchy(tree)
    drugs["drugbank_id"] = drugs["drugbank_id"].astype(str)
    expanded = drugs[["drugbank_id", "mesh_ids"]].copy()
    expanded["mesh_id"] = expanded["mesh_ids"].fillna("").str.split("|")
    expanded = expanded.explode("mesh_id")
    expanded["mesh_id"] = expanded["mesh_id"].str.strip()
    branch_d = tree[tree["tree_number"].str.startswith("D")]
    mapped = expanded.merge(branch_d[["mesh_id", "tree_number"]], on="mesh_id", how="inner")
    pairs = set()
    for drug_id, tree_number in mapped[["drugbank_id", "tree_number"]].itertuples(index=False, name=None):
        parts = tree_number.split(".")
        for depth in range(1, len(parts) + 1):
            pairs.add((drug_id, ".".join(parts[:depth]), depth))
    if not pairs:
        raise ValueError("No approved-drug MeSH IDs map to branch D in this hierarchy")
    result = pd.DataFrame(sorted(pairs), columns=["drugbank_id", "tree_number", "level"])
    result["deepest_level"] = result.groupby("drugbank_id")["level"].transform("max")
    result["knowledge_category"] = result["deepest_level"].map(knowledge_category)
    return _save(result, output_csv, overwrite=overwrite)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description, input_flag in (
        ("coverage", "DrugBank modality-presence cache", "drugbank-xml"),
        ("approved-mesh", "Approved DrugBank MeSH assignments", "drugbank-xml"),
        ("hierarchy", "NLM MeSH descriptor tree positions", "mesh-xml"),
        ("annotations", "Branch-D mapping and ancestor expansion", None),
    ):
        command = commands.add_parser(name, help=description)
        if input_flag:
            command.add_argument(f"--{input_flag}", type=Path, required=True)
        else:
            command.add_argument("--approved-csv", type=Path, required=True)
            command.add_argument("--hierarchy-csv", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True, help="Private output CSV path")
        command.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    args = parser.parse_args(argv)
    options = {"output_csv": args.output, "overwrite": args.overwrite}
    if args.command == "coverage":
        frame = prepare_modality_coverage(args.drugbank_xml, **options)
    elif args.command == "approved-mesh":
        frame = extract_approved_drug_mesh(args.drugbank_xml, **options)
    elif args.command == "hierarchy":
        frame = parse_mesh_hierarchy(args.mesh_xml, **options)
    else:
        frame = extend_drug_mesh_annotations(args.approved_csv, args.hierarchy_csv, **options)
    print(f"Saved {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()
