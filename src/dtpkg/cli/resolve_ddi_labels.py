"""Extract DrugBank interactions and apply the study's production label policy.

Private outputs are canonical positive/negative pairs and a count/hash summary.
Detailed pair-level audit tables are written only with --audit. Historical
positives are optional audit input and do not affect production labels.
"""
import argparse
from pathlib import Path

from dtpkg import ddi_labels
from dtpkg.project_paths import DATA_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--xml", type=Path, help="DrugBank XML; defaults under data-dir")
    parser.add_argument("--negatives", type=Path,
                        help="original reliable-negative drug1/drug2 CSV")
    parser.add_argument("--dpi", type=Path, help="prepared DPI table for target-coverage counts")
    parser.add_argument("--out-dir", type=Path,
                        help="output directory; defaults to data-dir/interactions")
    parser.add_argument("--extract", action="store_true",
                        help="replace cached XML extraction")
    parser.add_argument("--historical-positives", type=Path,
                        help="optional retired positive table for audit only")
    parser.add_argument("--audit", action="store_true",
                        help="also write private pair-level audit tables")
    args = parser.parse_args(argv)
    interactions = args.data_dir / "interactions"
    xml = args.xml or args.data_dir / "drugbank" / "full_database.xml"
    negatives = args.negatives or interactions / "unresolved" / "DDI_negative_pairs.csv"
    dpi = args.dpi or interactions / "DPI_enriched.csv"
    release_ddi = args.data_dir / "drugbank" / "drugbank_ddi_release.csv"
    release_drugs = args.data_dir / "drugbank" / "drugbank_drugs_release.csv"
    out = args.out_dir or interactions
    required = [negatives, dpi]
    if args.historical_positives is not None:
        required.append(args.historical_positives)
    for path in required:
        if not path.is_file():
            parser.error(f"missing input: {path}; see data/README.md")
    if args.extract or not release_ddi.is_file() or (args.audit and not release_drugs.is_file()):
        if not xml.is_file():
            parser.error(f"missing XML needed for extraction: {xml}")
        info = ddi_labels.drugbank_export_info(xml)
        print(f"DrugBank XML version {info['version']}, exported {info['exported_on']}")
        ddi_labels.extract_release_ddis(xml, release_ddi, release_drugs)
    ddi_labels.build_resolved_datasets(
        raw_pos=args.historical_positives, raw_neg=negatives,
        release_ddi=release_ddi, release_drugs=release_drugs,
        positive_source="xml", restrict_to_dpi=False, dpi_path=dpi, xml_path=xml,
        out_pos=out / "DDI_positive_pairs.csv", out_neg=out / "DDI_negative_pairs.csv",
        conflict_table=out / "label_audit" / "label_conflicts.csv",
        summary=out / "label_audit" / "label_audit_summary.json",
        historical_only_out=out / "label_audit" / "historical_only_positives.csv",
        quarantine_out=out / "label_audit" / "quarantined_pairs.csv",
        write_audit=args.audit)


if __name__ == "__main__":
    main()
