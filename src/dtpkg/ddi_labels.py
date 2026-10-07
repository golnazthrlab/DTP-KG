"""DDI pair identity and the study's DrugBank XML label policy.

Production positives are all canonical, deduplicated, non-self interaction
pairs in the supplied DrugBank export. Reliable negatives are canonicalized,
then screened against that export and the four author-defined quarantined
pairs. Historical positives are optional audit input, never production labels.

The study used DrugBank 5.1.13 (XML exported 2025-01-02). Input tables and all
pair-level outputs stay private; obtain the underlying data separately.
"""
import csv
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from dtpkg.project_paths import DATA_DIR

INTERACTIONS_DIR = DATA_DIR / "interactions"
#: Original reliable-negative input and optional historical positives.
#: Experiments must read the resolved outputs, never these input tables.
UNRESOLVED_DIR = INTERACTIONS_DIR / "unresolved"
RAW_POSITIVES = UNRESOLVED_DIR / "DDI_positive_pairs.csv"
RAW_NEGATIVES = UNRESOLVED_DIR / "DDI_negative_pairs.csv"
#: What every notebook and the graph builder read: canonical, deduplicated,
#: conflicts resolved under POLICY below.
RESOLVED_POSITIVES = INTERACTIONS_DIR / "DDI_positive_pairs.csv"
RESOLVED_NEGATIVES = INTERACTIONS_DIR / "DDI_negative_pairs.csv"
#: Complete interaction list of the DrugBank export (directed, with description)
#: and its drug table; both are derived from data/drugbank/full_database.xml.
DRUGBANK_XML = DATA_DIR / "drugbank" / "full_database.xml"
RELEASE_DDI = DATA_DIR / "drugbank" / "drugbank_ddi_release.csv"
RELEASE_DRUGS = DATA_DIR / "drugbank" / "drugbank_drugs_release.csv"
AUDIT_DIR = INTERACTIONS_DIR / "label_audit"
CONFLICT_TABLE = AUDIT_DIR / "label_conflicts.csv"
AUDIT_SUMMARY = AUDIT_DIR / "label_audit_summary.json"

POLICIES = ("positive", "exclude", "keep")
POLICY = "positive"

#: Where positive labels (and graph DDI edges) come from.
POSITIVE_SOURCES = ("xml", "historical+xml")
POSITIVE_SOURCE = "xml"
#: Negative pairs that only the retired historical positive CSV listed as
#: interactions. Absent from the export, so not documented positives; flagged
#: by a source we no longer trust, so not vetted negatives either. Excluded from
#: both classes pending provenance (independent review, 2026-09-17).
QUARANTINED_PAIRS = frozenset({
    ("DB00541", "DB00820"), ("DB00659", "DB00927"),
    ("DB00704", "DB01229"), ("DB00795", "DB01045"),
})
#: Drugs with at least one recorded DrugBank target (the drug--target table).
DPI_PATH = INTERACTIONS_DIR / "DPI_enriched.csv"
#: Production default: every export pair, regardless of recorded targets.
RESTRICT_TO_DPI = False
#: Outputs of the DPI-restricted sensitivity mode live here, never in the
#: production paths above.
DPI_SENSITIVITY_DIR = INTERACTIONS_DIR / "dpi_restricted"
HISTORICAL_ONLY_POSITIVES = AUDIT_DIR / "historical_only_positives.csv"
QUARANTINE_TABLE = AUDIT_DIR / "quarantined_pairs.csv"

_NS = "{http://www.drugbank.ca}"


# ---------------------------------------------------------------------------
# Unordered-pair identity
# ---------------------------------------------------------------------------
def canonicalize_pairs(df, drop_self=True):
    """Return ``df`` with string IDs, ``drug1 <= drug2``, one row per unordered pair.

    Other columns are kept (first occurrence wins). Self-pairs are dropped unless
    ``drop_self`` is False.
    """
    out = df.copy()
    out["drug1"] = out["drug1"].astype(str).str.strip()
    out["drug2"] = out["drug2"].astype(str).str.strip()
    swap = out["drug1"] > out["drug2"]
    out.loc[swap, ["drug1", "drug2"]] = out.loc[swap, ["drug2", "drug1"]].values
    if drop_self:
        out = out[out["drug1"] != out["drug2"]]
    out = out.drop_duplicates(subset=["drug1", "drug2"])
    return out.sort_values(["drug1", "drug2"]).reset_index(drop=True)


def pair_stats(df):
    """Row/duplicate/self-pair/unique-pair/drug counts of a raw pair table."""
    ids = df[["drug1", "drug2"]].astype(str)
    exact = int(ids.duplicated().sum())
    canon = canonicalize_pairs(ids, drop_self=False)
    self_pairs = int((canon["drug1"] == canon["drug2"]).sum())
    return {
        "rows": int(len(ids)),
        "exact_duplicate_rows": exact,
        "reverse_orientation_duplicate_rows": int(len(ids) - exact - len(canon)),
        "self_pairs": self_pairs,
        "unique_unordered_pairs": int(len(canon) - self_pairs),
        "drugs": int(pd.unique(ids.values.ravel()).size),
    }


def pair_set(df):
    """Set of canonical ``(drug1, drug2)`` tuples."""
    canon = canonicalize_pairs(df[["drug1", "drug2"]])
    return set(zip(canon["drug1"], canon["drug2"]))


# ---------------------------------------------------------------------------
# Conflict detection and resolution
# ---------------------------------------------------------------------------
def find_conflicts(pos_df, neg_df, reference_df=None):
    """Negative pairs that are also positive somewhere.

    Returns a canonical table with boolean columns ``in_pipeline_positives`` and
    ``in_reference`` (all False in the latter when no reference is given).
    """
    neg = canonicalize_pairs(neg_df[["drug1", "drug2"]])
    pos = pair_set(pos_df)
    ref = pair_set(reference_df) if reference_df is not None else set()
    keys = list(zip(neg["drug1"], neg["drug2"]))
    neg["in_pipeline_positives"] = [k in pos for k in keys]
    neg["in_reference"] = [k in ref for k in keys]
    return neg[neg["in_pipeline_positives"] | neg["in_reference"]].reset_index(drop=True)


def resolve_labels(pos_df, neg_df, reference_df=None, policy=POLICY):
    """Canonicalize both sets and apply ``policy`` to the conflicting pairs.

    Returns ``(pos, neg, report)``; ``pos``/``neg`` carry only ``drug1``/``drug2``.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    before = {"positives": pair_stats(pos_df), "negatives": pair_stats(neg_df)}
    pos = canonicalize_pairs(pos_df[["drug1", "drug2"]])
    neg = canonicalize_pairs(neg_df[["drug1", "drug2"]])
    conflicts = find_conflicts(pos, neg, reference_df)
    conflict_keys = set(zip(conflicts["drug1"], conflicts["drug2"]))
    is_conflict = [k in conflict_keys for k in zip(neg["drug1"], neg["drug2"])]

    if policy == "positive":
        neg = neg[[not c for c in is_conflict]]
        # Every conflict is a documented interaction in at least one source, so
        # it joins the positives; pairs already there are not duplicated.
        pos = canonicalize_pairs(pd.concat([pos, conflicts[["drug1", "drug2"]]]))
    elif policy == "exclude":
        neg = neg[[not c for c in is_conflict]]
        pos = pos[[k not in conflict_keys for k in zip(pos["drug1"], pos["drug2"])]]
    pos = pos.reset_index(drop=True)
    neg = neg.reset_index(drop=True)

    report = {
        "policy": policy,
        "reference_used": reference_df is not None,
        "before": before,
        "conflicts": {
            "total": int(len(conflicts)),
            "in_pipeline_positives": int(conflicts["in_pipeline_positives"].sum()),
            "in_reference": int(conflicts["in_reference"].sum()),
            "in_both": int((conflicts["in_pipeline_positives"] & conflicts["in_reference"]).sum()),
            "reference_only": int((~conflicts["in_pipeline_positives"] & conflicts["in_reference"]).sum()),
            "pipeline_only": int((conflicts["in_pipeline_positives"] & ~conflicts["in_reference"]).sum()),
            "negative_rows_affected": int(sum(
                k in conflict_keys for k in pair_keys_with_duplicates(neg_df))),
        },
        "after": {"positives": int(len(pos)), "negatives": int(len(neg))},
    }
    return pos, neg, report


def pair_keys_with_duplicates(df):
    """Canonical key per *row* (duplicates retained), for row-level counts."""
    a = df["drug1"].astype(str).str.strip()
    b = df["drug2"].astype(str).str.strip()
    return list(zip(a.where(a <= b, b), b.where(a <= b, a)))


def format_report(report):
    b, c, a = report["before"], report["conflicts"], report["after"]
    ref = " + DrugBank release" if report["reference_used"] else ""
    lines = [
        f"[ddi_labels] policy={report['policy']} (conflicts checked against pipeline positives{ref})",
        f"   positives: {b['positives']['rows']:,} rows -> {b['positives']['unique_unordered_pairs']:,} unique pairs",
        f"   negatives: {b['negatives']['rows']:,} rows -> {b['negatives']['unique_unordered_pairs']:,} unique pairs",
        f"   conflicts: {c['total']:,} negative pairs also listed as positive"
        f" (pipeline {c['in_pipeline_positives']:,}, release {c['in_reference']:,})",
        f"   after:     {a['positives']:,} positives, {a['negatives']:,} negatives",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DrugBank release
# ---------------------------------------------------------------------------
def drugbank_export_info(xml_path=DRUGBANK_XML):
    """``version`` and ``exported-on`` attributes of the XML root, without parsing it."""
    with open(xml_path, "rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
    for _, el in ET.iterparse(xml_path, events=("start",)):
        return {"version": el.get("version"), "exported_on": el.get("exported-on"),
                "header": head.split("\n", 1)[0]}


def extract_release_ddis(xml_path=DRUGBANK_XML, ddi_out=RELEASE_DDI,
                         drugs_out=RELEASE_DRUGS, verbose=True):
    """Stream the XML once and write every listed drug-drug interaction.

    ``ddi_out``: ``drug1, drug2, description`` — one row per ``<drug-interaction>``
    as listed under ``drug1`` (DrugBank lists nearly every pair from both sides).
    ``drugs_out``: ``drugbank_id, secondary_ids, name, type, groups``.
    Takes about a minute for the 1.6 GB 5.1.13 export.
    """
    ddi_out, drugs_out = Path(ddi_out), Path(drugs_out)
    ddi_out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_drugs = n_ddi = 0
    with open(ddi_out, "w", newline="") as f_ddi, open(drugs_out, "w", newline="") as f_dr:
        w_ddi = csv.writer(f_ddi)
        w_ddi.writerow(["drug1", "drug2", "description"])
        w_dr = csv.writer(f_dr)
        w_dr.writerow(["drugbank_id", "secondary_ids", "name", "type", "groups"])
        for _, el in ET.iterparse(xml_path, events=("end",)):
            # Top-level <drug> elements carry a type attribute; the nested
            # <drug-interaction> entries are a different tag.
            if el.tag != _NS + "drug" or el.get("type") is None:
                continue
            ids = el.findall(_NS + "drugbank-id")
            primary = next((i.text for i in ids if i.get("primary") == "true"), ids[0].text)
            secondary = [i.text for i in ids if i.get("primary") != "true"]
            groups = [g.text for g in el.findall(f"{_NS}groups/{_NS}group")]
            w_dr.writerow([primary, "|".join(secondary), el.findtext(_NS + "name"),
                           el.get("type"), "|".join(groups)])
            n_drugs += 1
            for di in el.findall(f"{_NS}drug-interactions/{_NS}drug-interaction"):
                w_ddi.writerow([primary, di.findtext(_NS + "drugbank-id"),
                                di.findtext(_NS + "description")])
                n_ddi += 1
            el.clear()
    if verbose:
        print(f"[ddi_labels] {n_drugs:,} drugs, {n_ddi:,} directed interaction rows "
              f"({time.time() - started:.0f}s) -> {ddi_out}")
    return n_drugs, n_ddi


def load_release_pairs(path=RELEASE_DDI):
    """Canonical unique unordered pairs of the release (two columns)."""
    return canonicalize_pairs(pd.read_csv(path, dtype=str, usecols=["drug1", "drug2"]))


def load_release_drugs(path=RELEASE_DRUGS):
    return pd.read_csv(path, dtype=str).fillna("").set_index("drugbank_id")


def release_descriptions(path=RELEASE_DDI):
    """One description per canonical pair (the first listed orientation)."""
    rel = pd.read_csv(path, dtype=str)
    return canonicalize_pairs(rel).set_index(["drug1", "drug2"])["description"]


# ---------------------------------------------------------------------------
# XML-only supervision sets
# ---------------------------------------------------------------------------
def load_dpi_drugs(path=DPI_PATH):
    """DrugBank IDs of every drug in the drug--target table."""
    return set(pd.read_csv(path, dtype=str, usecols=["db_id"])["db_id"].str.strip())


def restrict_to_universe(df, universe):
    """Keep pairs whose two drugs are both in ``universe`` (a set of IDs)."""
    if universe is None:
        return df.reset_index(drop=True)
    keep = df["drug1"].isin(universe) & df["drug2"].isin(universe)
    return df[keep].reset_index(drop=True)


def build_supervision_sets(release_pairs, raw_neg, historical_pos=None,
                           quarantine=QUARANTINED_PAIRS, universe=None):
    """Positive/negative pair tables under the XML-only source.

    ``release_pairs``: the canonical master table extracted from the export.
    ``raw_neg``: the reliable negatives as supplied (any orientation/duplicates).
    ``historical_pos``: optional migrated positive CSV, used only to report
    which of its pairs the export does not contain.
    ``universe``: optional drug-ID set; pairs with an endpoint outside it are
    dropped from both classes (see module docstring).

    Returns ``(pos, neg, report)``. ``pos`` and ``neg`` are disjoint by
    construction and the report records every count needed for the paper.
    """
    pos = canonicalize_pairs(release_pairs[["drug1", "drug2"]])
    neg = canonicalize_pairs(raw_neg[["drug1", "drug2"]])
    pos_keys = set(zip(pos["drug1"], pos["drug2"]))
    neg_keys = list(zip(neg["drug1"], neg["drug2"]))
    quarantine = set(map(tuple, quarantine or ()))

    listed = [k in pos_keys for k in neg_keys]
    quarantined = [k in quarantine for k in neg_keys]
    neg_screened = neg[[not (a or b) for a, b in zip(listed, quarantined)]]

    report = {
        "positive_source": "xml",
        "release_pairs": int(len(pos)),
        "negatives": {
            **pair_stats(raw_neg),
            "listed_in_release": int(sum(listed)),
            "quarantined": int(sum(quarantined)),
            "quarantined_pairs": sorted(quarantine),
            "after_screen": int(len(neg_screened)),
        },
    }
    if historical_pos is not None:
        hist = canonicalize_pairs(historical_pos[["drug1", "drug2"]])
        hist_keys = set(zip(hist["drug1"], hist["drug2"]))
        hist_only = hist_keys - pos_keys
        hist_drugs = {d for k in hist_keys for d in k}
        release_among_hist = {k for k in pos_keys
                              if k[0] in hist_drugs and k[1] in hist_drugs}
        report["historical_positives"] = {
            **pair_stats(historical_pos),
            "absent_from_release": int(len(hist_only)),
            "release_pairs_among_historical_drugs_missing": int(
                len(release_among_hist - hist_keys)),
            "negatives_listed_only_here": sorted(
                k for k in neg_keys if k in hist_keys and k not in pos_keys),
        }
        report["_historical_only_pairs"] = sorted(hist_only)

    before_universe = {"positives": int(len(pos)), "negatives": int(len(neg_screened))}
    if universe is not None:
        pos = restrict_to_universe(pos, universe)
        neg_screened = restrict_to_universe(neg_screened, universe)
    report["universe"] = {
        "restricted": universe is not None,
        "drugs": int(len(universe)) if universe is not None else None,
        "before": before_universe,
        "dropped_positives": before_universe["positives"] - int(len(pos)),
        "dropped_negatives": before_universe["negatives"] - int(len(neg_screened)),
    }
    pos = pos.reset_index(drop=True)
    neg = neg_screened.reset_index(drop=True)
    overlap = set(zip(pos["drug1"], pos["drug2"])) & set(zip(neg["drug1"], neg["drug2"]))
    if overlap:  # cannot happen by construction; guard against future edits
        raise AssertionError(f"{len(overlap)} pairs carry both labels")
    report["after"] = {"positives": int(len(pos)), "negatives": int(len(neg)),
                       "positive_negative_overlap": 0}
    return pos, neg, report


def check_supervision_files(pos_df, neg_df, quarantine=QUARANTINED_PAIRS,
                            reference_df=None):
    """Guard run by the loaders: disjoint labels, no quarantined pair, optional
    positives-subset-of-reference. Returns a dict of violations (empty = OK)."""
    pos = pair_set(pos_df)
    neg = pair_set(neg_df)
    problems = {}
    both = pos & neg
    if both:
        problems["pairs_with_both_labels"] = len(both)
    hit = (pos | neg) & set(map(tuple, quarantine or ()))
    if hit:
        problems["quarantined_pairs_present"] = sorted(hit)
    if reference_df is not None:
        outside = pos - pair_set(reference_df)
        if outside:
            problems["positives_absent_from_reference"] = len(outside)
    return problems


def recorded_target_coverage(pos, neg, dpi_path=DPI_PATH):
    """Per class: pairs / drugs whose members have a recorded target in the DPI table.

    "No recorded target" means absence from ``DPI_enriched.csv``; it is not a
    claim that the drug has no biological target.
    """
    dpi = load_dpi_drugs(dpi_path)

    def one(df):
        d1 = df["drug1"].isin(dpi); d2 = df["drug2"].isin(dpi)
        drugs = set(df["drug1"]) | set(df["drug2"])
        return {
            "pairs": int(len(df)),
            "pairs_both_drugs_with_recorded_target": int((d1 & d2).sum()),
            "pairs_with_at_least_one_drug_lacking_recorded_target": int((~(d1 & d2)).sum()),
            "drugs": int(len(drugs)),
            "drugs_with_recorded_target": int(len(drugs & dpi)),
            "drugs_without_recorded_target": int(len(drugs - dpi)),
        }
    return {"positives": one(pos), "negatives": one(neg), "dpi_table_drugs": int(len(dpi))}


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _conflict_table(pos_raw, neg_raw, reference, release_drugs, release_ddi):
    """Every negative pair that any positive source lists, annotated for the audit."""
    conflicts = find_conflicts(pos_raw, neg_raw, reference)
    drugs = load_release_drugs(release_drugs)
    desc = release_descriptions(release_ddi)
    keys = list(zip(conflicts["drug1"], conflicts["drug2"]))
    row_counts = pd.Series(pair_keys_with_duplicates(neg_raw)).value_counts()
    return conflicts.rename(columns={"in_reference": "in_drugbank_release"}).assign(
        name1=[drugs["name"].get(d, "") for d in conflicts["drug1"]],
        name2=[drugs["name"].get(d, "") for d in conflicts["drug2"]],
        type1=[drugs["type"].get(d, "") for d in conflicts["drug1"]],
        type2=[drugs["type"].get(d, "") for d in conflicts["drug2"]],
        negative_rows_in_source=[int(row_counts.get(k, 0)) for k in keys],
        release_description=[desc.get(k, "") for k in keys],
    )


def build_resolved_datasets(raw_pos=None, raw_neg=RAW_NEGATIVES,
                            release_ddi=RELEASE_DDI, release_drugs=RELEASE_DRUGS,
                            positive_source=POSITIVE_SOURCE, policy=POLICY,
                            restrict_to_dpi=RESTRICT_TO_DPI, dpi_path=DPI_PATH,
                            quarantine=QUARANTINED_PAIRS, xml_path=DRUGBANK_XML,
                            out_pos=RESOLVED_POSITIVES, out_neg=RESOLVED_NEGATIVES,
                            conflict_table=CONFLICT_TABLE, summary=AUDIT_SUMMARY,
                            historical_only_out=HISTORICAL_ONLY_POSITIVES,
                            quarantine_out=QUARANTINE_TABLE, verbose=True, write_audit=False):
    """Write production pairs and a summary with input/output hashes.

    Under the default XML policy, ``raw_pos`` is optional historical audit
    input. Its presence never changes the production labels. ``write_audit``
    additionally writes private pair-level audit tables; drug names and
    descriptions are required only for this optional audit.

    The explicit ``historical+xml`` source retains the legacy audit API and
    requires ``raw_pos``. It is not exposed by the production CLI.
    """
    if positive_source not in POSITIVE_SOURCES:
        raise ValueError(f"positive_source must be one of {POSITIVE_SOURCES}, "
                         f"got {positive_source!r}")
    if restrict_to_dpi:
        # Sensitivity mode: same file names under a separate directory, so the
        # restricted tables can never be mistaken for the production ones.
        defaults = {"out_pos": RESOLVED_POSITIVES, "out_neg": RESOLVED_NEGATIVES,
                    "conflict_table": CONFLICT_TABLE, "summary": AUDIT_SUMMARY,
                    "historical_only_out": HISTORICAL_ONLY_POSITIVES,
                    "quarantine_out": QUARANTINE_TABLE}
        given = {"out_pos": out_pos, "out_neg": out_neg, "conflict_table": conflict_table,
                 "summary": summary, "historical_only_out": historical_only_out,
                 "quarantine_out": quarantine_out}
        for key, default in defaults.items():
            if Path(given[key]) == Path(default):
                given[key] = DPI_SENSITIVITY_DIR / Path(default).relative_to(INTERACTIONS_DIR)
        out_pos, out_neg, conflict_table = given["out_pos"], given["out_neg"], given["conflict_table"]
        summary, historical_only_out, quarantine_out = (given["summary"],
                                                        given["historical_only_out"],
                                                        given["quarantine_out"])
    if positive_source == "historical+xml" and raw_pos is None:
        raise ValueError("historical+xml requires explicit historical positives")
    pos_raw = pd.read_csv(raw_pos, dtype=str) if raw_pos is not None else None
    neg_raw = pd.read_csv(raw_neg, dtype=str)
    reference = load_release_pairs(release_ddi)
    universe = load_dpi_drugs(dpi_path) if restrict_to_dpi else None

    if positive_source == "xml":
        pos, neg, report = build_supervision_sets(
            reference, neg_raw, historical_pos=pos_raw,
            quarantine=quarantine, universe=universe)
        hist_only = (pd.DataFrame(report.pop("_historical_only_pairs"),
                                  columns=["drug1", "drug2"])
                     if pos_raw is not None else None)
    else:
        print("WARNING: rebuilding the SUPERSEDED historical+xml union; "
              "these files are for audit comparison only.")
        pos, neg, report = resolve_labels(pos_raw, neg_raw, reference, policy=policy)
        report["positive_source"] = "historical+xml"
        report["universe"] = {"restricted": False}
        hist_only = None

    outputs = [out_pos, out_neg]
    for path in outputs + [summary]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pos.to_csv(out_pos, index=False)
    neg.to_csv(out_neg, index=False)
    if write_audit:
        # Without historical input the audit checks negatives against XML alone.
        audit_pos = pos_raw if pos_raw is not None else reference.iloc[:0]
        table = _conflict_table(audit_pos, neg_raw, reference, release_drugs, release_ddi)
        Path(conflict_table).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(conflict_table, index=False)
        outputs.append(conflict_table)
        if hist_only is not None:
            Path(historical_only_out).parent.mkdir(parents=True, exist_ok=True)
            hist_only.to_csv(historical_only_out, index=False)
            outputs.append(historical_only_out)
        quarantine_rows = pd.DataFrame(sorted(map(tuple, quarantine or ())),
                                       columns=["drug1", "drug2"])
        quarantine_rows["reason"] = (
            "listed as positive only by the retired historical CSV; absent from "
            "the study's DrugBank export; excluded pending provenance")
        Path(quarantine_out).parent.mkdir(parents=True, exist_ok=True)
        quarantine_rows.to_csv(quarantine_out, index=False)
        outputs.append(quarantine_out)

    input_paths = [raw_neg, release_ddi]
    if raw_pos is not None:
        input_paths.append(raw_pos)
    report["inputs"] = {str(p): sha256(p) for p in input_paths}
    report["audit_tables_written"] = bool(write_audit)
    if Path(xml_path).exists():
        report["inputs"][str(xml_path)] = sha256(xml_path)
        report["drugbank_export"] = drugbank_export_info(xml_path)
    report["outputs"] = {str(p): sha256(p) for p in outputs}
    report["release"] = {"pairs": int(len(reference)), "source": str(release_ddi)}
    report["node_universe"] = {
        "restricted_to_dpi": bool(restrict_to_dpi),
        "mode": "dpi_restricted_sensitivity" if restrict_to_dpi else "production_all_export_pairs",
        "dpi_path": str(dpi_path) if restrict_to_dpi else None}
    report["recorded_target_coverage"] = recorded_target_coverage(pos, neg, dpi_path)
    with open(summary, "w") as fh:
        json.dump(report, fh, indent=2)
    if verbose:
        if positive_source == "xml":
            print(format_xml_report(report))
        else:
            print(format_report(report))
        print("   wrote " + ", ".join(str(p) for p in outputs + [summary]))
    return report


def format_xml_report(report):
    n, u, a = report["negatives"], report["universe"], report["after"]
    lines = [
        f"[ddi_labels] positive_source=xml  (release pairs: {report['release_pairs']:,})",
        f"   negatives: {n['rows']:,} rows -> {n['unique_unordered_pairs']:,} unique pairs; "
        f"{n['listed_in_release']:,} listed as interactions in the export, "
        f"{n['quarantined']} quarantined -> {n['after_screen']:,}",
    ]
    if "historical_positives" in report:
        h = report["historical_positives"]
        lines.append(f"   historical positive CSV: {h['unique_unordered_pairs']:,} pairs, "
                     f"{h['absent_from_release']:,} absent from the export (archived, not used)")
    if u["restricted"]:
        lines.append(f"   universe: {u['drugs']:,} DPI drugs; dropped "
                     f"{u['dropped_positives']:,} positives, {u['dropped_negatives']:,} negatives")
    lines.append(f"   after:     {a['positives']:,} positives, {a['negatives']:,} negatives "
                 f"(overlap {a['positive_negative_overlap']})")
    return "\n".join(lines)
