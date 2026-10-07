"""Report-only notebook/CLI workflow; uses completed fits and never trains."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from dtpkg.case_study.analysis import (annotate_pairs, correlations, partial_correlations, coverage_partial_correlations,
    summarize_correlations, select_examples)


TABLES = ("pair_overlap_scores", "correlations_by_fit", "correlations_by_split",
          "correlation_summary", "selected_examples", "example_selection", "coverage", "paper_table")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: file_hash(folder / name)
            for name in ("analysis.py", "sources.py", "workflow.py")}


def load_results(setting, results_root):
    """Load complete saved tables after checking code, input and output hashes."""
    if setting not in ("transductive", "inductive"):
        raise ValueError("choose transductive or inductive")
    folder = Path(results_root) / setting
    manifest = json.loads((folder / "analysis_manifest.json").read_text())
    if (manifest.get("format_version") != 4 or manifest.get("kind") != "case_study_analysis"
            or manifest["setting"] != setting or manifest.get("complete") is not True):
        raise ValueError("incomplete or mismatched case-study report")
    if manifest["analysis_code"] != _code_hashes():
        raise ValueError("analysis code changed; recompute this case study")
    for path, digest in manifest["source_files"].items():
        if file_hash(path) != digest:
            raise ValueError(f"case-study source changed: {path}; recompute")
    required = {f"{name}.csv" for name in TABLES} | {"source_provenance.json"}
    if not required.issubset(manifest["outputs"]):
        raise ValueError("analysis manifest does not cover every required table")
    for name, digest in manifest["outputs"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid report artifact path")
        if file_hash(folder / name) != digest:
            raise ValueError(f"case-study output changed: {folder / name}; recompute")
    return {**{name: pd.read_csv(folder / f"{name}.csv") for name in TABLES},
            "folder": folder, "manifest": manifest}


def _validate_output_location(results_root, options):
    output = Path(results_root).expanduser().resolve()
    for key in ("results_dir", "prepared_dir", "data_dir", "graph_path"):
        if key not in options:
            continue
        source = Path(options[key]).expanduser().resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("case-study results must not overlap their source inputs")


def paper_table(summary):
    """Definition-matched biology first; DDI-containing context is a diagnostic.

    Preserve adjustment encoding, reference class and subset in every row. The
    raw historical result is retained, not retroactively relabelled biological.
    """
    parts = []
    for feature, subset, role in (
        ("biological_shared_protein_count", "all_pairs", "definition_matched_biology"),
        ("biological_shared_protein_count", "both_have_targets", "biology_with_target_coverage"),
        ("shared_target_count", "all_pairs", "direct_targets_all_pairs"),
        ("shared_target_count", "both_have_targets", "direct_targets_with_target_coverage"),
        ("shared_protein_count", "all_pairs", "full_graph_context_diagnostic"),
    ):
        keep = summary[(summary.feature == feature) & (summary.subset == subset)
                       & (summary.label_group == "overall")].copy()
        keep["reporting_role"] = role
        keep["_order"] = keep.statistic.map({"spearman": 0, "partial_spearman": 1})
        parts.append(keep.sort_values(["scenario", "_order", "control_encoding"]).drop(columns="_order"))
    labelled = summary[(summary.feature == "shared_protein_count") & (summary.subset == "all_pairs")
                       & (summary.label_group != "overall") & (summary.statistic == "spearman")].copy()
    labelled["reporting_role"] = "reference_class_heterogeneity"
    parts.append(labelled)
    return pd.concat(parts, ignore_index=True)


def run_analysis(setting, results_root, *, source_options,
                 positive_cutoff=.9, negative_cutoff=.1):
    """Recompute overlap associations from completed, verified release fits.

    ``source_options`` supplies explicit input paths for the corresponding
    source iterator. All declared repeats/holdouts are used unless a subset is
    requested. Grid size is recorded, never assumed to match the historical
    25-CV/15-inductive-fit snapshot. This function performs no model fitting or
    figure rendering. Record-level outputs belong in a local results directory.
    """
    from dtpkg.case_study.sources import iter_transductive_fits, iter_inductive_fits
    if setting not in ("transductive", "inductive"):
        raise ValueError("choose transductive or inductive")
    if not source_options:
        raise ValueError("source_options must supply explicit input paths")
    _validate_output_location(results_root, source_options)
    folder = Path(results_root) / setting
    if folder.exists() and any(folder.iterdir()) and not (folder / "analysis_manifest.json").is_file():
        raise FileExistsError(f"unrecognized nonempty analysis directory: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    # Remove only files owned by the previous generated report.
    # Invalidate completion first so interruptions cannot expose a mixed report.
    previous_manifest = folder / "analysis_manifest.json"
    owned = []
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text())
        if previous.get("kind") != "case_study_analysis" or previous.get("setting") != setting:
            raise ValueError("unrecognized previous case-study report")
        owned = list(previous.get("outputs", {}))
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in owned):
            raise ValueError("invalid previous report artifact path")
    generated_names = [*(f"{table}.csv" for table in TABLES), "source_provenance.json"]
    # An incomplete marker also makes an interrupted analysis recognizable on
    # rerun. Loading it still fails until every output and source is verified.
    previous_manifest.write_text(json.dumps(dict(format_version=4, kind="case_study_analysis",
        setting=setting, complete=False, outputs={name: None for name in generated_names}), indent=2) + "\n")
    for name in owned:
        (folder / name).unlink(missing_ok=True)
    iterator = iter_transductive_fits if setting == "transductive" else iter_inductive_fits
    frames, stats, sources = [], [], []
    for fit in iterator(**source_options):
        print(f"Annotating {fit['fit_id']} ({len(fit['predictions']):,} held-out pairs)", flush=True)
        frame = annotate_pairs(fit["predictions"], fit["train_graph"],
                               fit["inference_graph"], fit["heldout_drugs"])
        provenance = fit["provenance"]
        for suffix, flag in (("a", "drug1_heldout"), ("b", "drug2_heldout")):
            frame[f"endpoint_graph_id_{suffix}"] = [
                provenance["inference_graph_id" if held else "train_graph_id"] for held in frame[flag]]
        frames.append(frame)
        stats.append(pd.concat([correlations(frame), partial_correlations(frame),
                                coverage_partial_correlations(frame)], ignore_index=True))
        sources.append(provenance)
    if not frames:
        raise ValueError("no completed fits available")
    pairs = pd.concat(frames, ignore_index=True)
    by_fit = pd.concat(stats, ignore_index=True)
    by_split, summary = summarize_correlations(by_fit)
    examples, selection = select_examples(pairs, positive_cutoff, negative_cutoff)
    # Preserve columns even if a strict selection rule finds no examples.
    if examples.empty:
        examples = pairs.iloc[:0].assign(example_kind=pd.Series(dtype=str))
    coverage = []
    for scenario, group in pairs.groupby("scenario", sort=True):
        coverage.append(dict(setting=setting, scenario=scenario,
            n_prediction_rows=len(group), n_unique_pairs=group.pair_id.nunique(),
            n_unique_drugs=len(set(group.drug1) | set(group.drug2)), n_fits=group.fit_id.nunique(),
            n_positive_rows=int(group.label.eq(1).sum()), n_negative_rows=int(group.label.eq(0).sum()),
            n_zero_overlap_rows=int(group.shared_protein_count.eq(0).sum()),
            n_both_target_rows=int(((group.n_targets_a > 0) & (group.n_targets_b > 0)).sum())))
    tables = dict(zip(TABLES, (pairs, by_fit, by_split, summary, examples, selection,
                              pd.DataFrame(coverage), paper_table(summary))))
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    sources_path = folder / "source_provenance.json"
    sources_path.write_text(json.dumps(sources, indent=2, default=str) + "\n")
    source_files = {}
    for source in sources:
        if not source.get("source_files"):
            raise ValueError("source loader must provide file hashes for report reuse")
        for path, digest in source["source_files"].items():
            if path in source_files and source_files[path] != digest:
                raise ValueError("source changed during analysis")
            source_files[path] = digest
    for path, digest in source_files.items():
        if file_hash(path) != digest:
            raise ValueError(f"source changed during case-study analysis: {path}")
    outputs = {name: file_hash(folder / name) for name in generated_names}
    manifest = dict(format_version=4, kind="case_study_analysis", setting=setting, complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), model="topo_only", training_performed=False,
        source_files=source_files, analysis_code=_code_hashes(), outputs=outputs,
        n_fits=len(sources), n_prediction_rows=len(pairs), n_unique_pairs=pairs.pair_id.nunique(),
        primary_feature="biological_shared_protein_count", original_primary_feature="shared_protein_count",
        protocol_amendment="2026-09-21: exploratory size and zero-biological-size adjustment; coverage reporting; visible direct-target routes; unchanged pair selection",
        radius_from_drug=2,
        protein_node_types=["target", "protein"], graph_paths="all split-permitted paths",
        biological_sensitivity_paths="drug-target then at most one protein-protein edge",
        score_kind="saved neural probability; ties retained; not calibrated confidence",
        statistics="within-fit raw and exploratory rank-residual partial Spearman; seed/fold means then split means; no p-values/CIs",
        partial_controls="matching endpoint neighborhood/target sizes; average tied ranks and OLS intercept",
        partial_control_encodings=["endpoint_columns", "unordered_minmax"],
        coverage_control_encodings=["endpoint_columns_plus_zero", "unordered_minmax_plus_zero"],
        selection=dict(positive_cutoff=positive_cutoff, negative_cutoff=negative_cutoff,
                       both_targets_required=True, first_prespecified_fit=True),
        fit_ids=[source["fit_id"] for source in sources],
        split_repeats=sorted(map(int, pairs.split_repeat.unique())),
        training_seeds=sorted(map(int, pairs.training_seed.unique())),
        scenarios=sorted(pairs.scenario.unique()),
        source_options={key: value for key, value in source_options.items()},
        figures=[])
    (folder / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"Saved {setting}: {len(sources)} fits, {len(pairs):,} rows, {pairs.pair_id.nunique():,} unique pairs", flush=True)
    return {**tables, "folder": folder, "manifest": manifest}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("transductive", "inductive"), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    options = dict(results_dir=args.source_results, graph_path=args.graph_path)
    if args.setting == "inductive":
        if args.prepared_dir is None or args.data_dir is None:
            parser.error("inductive analysis requires --prepared-dir and --data-dir")
        options.update(prepared_dir=args.prepared_dir, data_dir=args.data_dir)
    run_analysis(args.setting, args.results_root, source_options=options)


if __name__ == "__main__":
    main()
