"""Read the publisher's Table S6 seed negatives into a private input CSV.

The study starts from the seed negatives in Additional file 2 of Zheng et al.
(2019), restricted to drugs in the prepared drug–target table. Duplicate rows,
endpoint order and the first data row are preserved. ``dtpkg.ddi_labels`` then
screens these candidates against DrugBank positives and the study quarantine.
XLSX parsing uses the Python standard library; no Excel reader is required.
"""
from __future__ import annotations

import csv
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


_DRUG_ID = re.compile(r"DB[0-9]{5}\Z")
_ADDRESS = re.compile(r"([A-Za-z]+)([1-9][0-9]*)\Z")
_COLUMNS = ["drug1", "drug2"]


def _part_path(relationship):
    """Resolve a workbook-relative ZIP member without following external links."""
    target = relationship.get("Target", "")
    uri = urlsplit(target)
    if (relationship.get("TargetMode", "").casefold() == "external"
            or uri.scheme or uri.netloc or uri.query or uri.fragment):
        raise ValueError("Table S6 and shared strings must be internal XLSX parts")
    target = unquote(uri.path)
    if not target or "\\" in target:
        raise ValueError("Invalid XLSX relationship target")
    member = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                 else posixpath.join("xl", target))
    if member in (".", "..") or member.startswith("../"):
        raise ValueError("XLSX relationship escapes the workbook archive")
    return member


def _text(element):
    if element is None:
        return ""
    # Rich-text runs contribute their text, not formatting or phonetic hints.
    pieces = []
    for child in element:
        if child.tag.rsplit("}", 1)[-1] == "t":
            pieces.append(child.text or "")
        elif child.tag.rsplit("}", 1)[-1] == "r":
            pieces.extend(node.text or "" for node in child.findall("{*}t"))
    return "".join(pieces)


def _cell_text(cell, strings):
    if cell.find("{*}f") is not None:
        raise ValueError("Table S6 must contain saved identifiers, not formulas")
    kind = cell.get("t", "n")
    value = cell.findtext("{*}v", "")
    if kind == "s":
        if not value.isdigit() or int(value) >= len(strings):
            raise ValueError("Invalid shared-string reference in Table S6")
        return strings[int(value)].strip()
    if kind == "inlineStr":
        return _text(cell.find("{*}is")).strip()
    if kind in ("n", "str"):
        return value.strip()
    if not value:
        return ""
    raise ValueError(f"Unsupported Table S6 cell type: {kind}")


def _seed_pairs(supplement_path):
    with zipfile.ZipFile(supplement_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        mapping = {entry.get("Id"): entry for entry in relations}
        if len(mapping) != len(relations) or None in mapping:
            raise ValueError("Workbook relationship identifiers must be unique")
        sheets = [sheet for sheet in workbook.findall("{*}sheets/{*}sheet")
                  if " ".join(sheet.get("name", "").split()).casefold() == "table s6"]
        if len(sheets) != 1:
            raise ValueError("Additional file 2 must contain exactly one sheet named Table S6")
        relation_ids = [value for key, value in sheets[0].attrib.items()
                        if key.rsplit("}", 1)[-1] == "id"]
        relation = mapping.get(relation_ids[0]) if len(relation_ids) == 1 else None
        if relation is None or not relation.get("Type", "").endswith("/worksheet"):
            raise ValueError("Table S6 has no valid worksheet relationship")
        sheet = ET.fromstring(archive.read(_part_path(relation)))
        string_parts = [entry for entry in relations if entry.get("Type", "").endswith("/sharedStrings")]
        if len(string_parts) > 1:
            raise ValueError("Workbook has multiple shared-string tables")
        strings = ([_text(item) for item in ET.fromstring(archive.read(_part_path(string_parts[0])))]
                   if string_parts else [])

        columns, pairs = None, []
        for row_number, row in enumerate(sheet.findall("{*}sheetData/{*}row"), start=1):
            cells, next_column = {}, 0
            for cell in row.findall("{*}c"):
                address = cell.get("r")
                if address:
                    match = _ADDRESS.fullmatch(address)
                    if match is None or (row.get("r") and int(match[2]) != int(row.get("r"))):
                        raise ValueError("Invalid or inconsistent cell address in Table S6")
                    column = 0
                    for letter in match[1].upper():
                        column = 26 * column + ord(letter) - ord("A") + 1
                    column -= 1
                else:
                    column = next_column
                if column in cells:
                    raise ValueError("Duplicate cell address in Table S6")
                cells[column] = _cell_text(cell, strings)
                next_column = column + 1
            if not any(cells.values()):
                continue
            if columns is None:
                header = {name: [index for index, value in cells.items()
                                 if "".join(value.split()).casefold() == name]
                          for name in _COLUMNS}
                if all(len(header[name]) == 1 for name in _COLUMNS):
                    columns = [header[name][0] for name in _COLUMNS]
                continue
            pair = tuple(cells.get(column, "") for column in columns)
            if not all(_DRUG_ID.fullmatch(value) for value in pair):
                raise ValueError(f"Table S6 row {row.get('r', row_number)} has a missing or invalid DrugBank ID")
            pairs.append(pair)
        if columns is None:
            raise ValueError("Table S6 needs a header containing Drug1 and Drug2")
        if not pairs:
            raise ValueError("Table S6 contains no seed-negative rows")
        return pairs


def _dpi_drugs(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or reader.fieldnames.count("db_id") != 1:
            raise ValueError("The drug–target CSV must contain one db_id column")
        drugs = set()
        for row in reader:
            value = (row.get("db_id") or "").strip()
            if not _DRUG_ID.fullmatch(value):
                raise ValueError(f"Invalid DrugBank ID in drug–target CSV row {reader.line_num}")
            drugs.add(value)
    if not drugs:
        raise ValueError("The drug–target CSV contains no drug IDs")
    return drugs


def prepare_seed_negatives(supplement_path, dpi_path, output_csv, *, overwrite=False):
    """Save DPI-filtered Table S6 rows and return ``drug1, drug2`` candidates.

    ``supplement_path`` is the publisher's Additional file 2 XLSX; ``dpi_path``
    is the prepared drug–target CSV with a ``db_id`` column. Existing identical
    outputs are reused. Replacing different output requires ``overwrite=True``.
    Inputs are never overwritten. This step does not resolve negative labels.
    """
    supplement, dpi, output = [Path(path).expanduser().resolve()
                               for path in (supplement_path, dpi_path, output_csv)]
    for source in (supplement, dpi):
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        if output == source or (output.exists() and output.samefile(source)):
            raise ValueError("Negative output must be separate from both input files")
    drugs = _dpi_drugs(dpi)
    pairs = [pair for pair in _seed_pairs(supplement) if pair[0] in drugs and pair[1] in drugs]
    if not pairs:
        raise ValueError("No Table S6 seed negatives have both endpoints in the drug–target CSV")
    frame = pd.DataFrame(pairs, columns=_COLUMNS)
    if output.exists() and not overwrite:
        try:
            existing = pd.read_csv(output, dtype=str, keep_default_na=False)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise FileExistsError(f"Existing negative output cannot be reused: {output}") from error
        if not existing.equals(frame):
            raise FileExistsError(f"Existing negative output differs; use overwrite=True to replace it: {output}")
        return frame
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        frame.to_csv(stream, index=False)
    return frame
