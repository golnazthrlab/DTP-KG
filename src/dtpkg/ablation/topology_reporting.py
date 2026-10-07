"""Table-only topology contribution report from aggregate matched-fold results.

The first table compares raw concatenation with MeSH only; the second compares
three input-level fusion methods with the latent gate. Each table retains its
own six-test Holm family across the three methods and AUROC/F1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtpkg.ablation.artifacts import file_hash, write_json
from dtpkg.evaluation_stats import comparison_table, performance_summary

MESH = "mesh_only_pairwise"
CONCAT = "raw_concat_pairwise"
FIXED = "raw_fixed4_pairwise"
LEARNED = "raw_learned_pairwise"
GATE = "adaptive_vector"
RAW = (CONCAT, FIXED, LEARNED)
MODELS = (MESH, *RAW, GATE)
METRICS = ("auc", "f1")
METRIC_LABEL = {"auc": "AUROC", "f1": "F1"}
LABEL = {MESH: "MeSH only", CONCAT: "Raw concat", FIXED: "Fixed α=4",
         LEARNED: "Learned α", GATE: "Latent gate"}
CONTRASTS = tuple((model, MESH) for model in RAW)
FUSION_CONTRASTS = tuple((model, GATE) for model in RAW)
COLUMNS = ("model", "split_repeat", "fold", "training_seed", "inference_protocol",
           "n_outer_train", "n_outer_test", "category", "n", "n_pos", "n_neg", "auc", "f1")
CSV_FILES = {"fold_metrics": "fold_metrics.csv", "summary": "summary_statistics.csv",
             "comparisons": "topology_statistics.csv", "numeric_table": "topology_contribution.csv",
             "paper_table": "topology_contribution_paper.csv",
             "fusion_comparisons": "fusion_statistics.csv", "fusion_numeric_table": "fusion_methods.csv",
             "fusion_paper_table": "fusion_methods_paper.csv"}
PUBLIC_PROTOCOL = {
    "study": "topology_contribution", "models": list(MODELS), "metrics": list(METRICS),
    "split_repeats": 5, "folds_per_repeat": 5, "optimizer_batch_size": 1,
    "displayed_models": [MESH, CONCAT], "confidence": 0.95, "f1_threshold": 0.5,
    "contrasts": [list(pair) for pair in CONTRASTS], "family": "raw_topology_contribution",
    "family_size": 6, "summary": "equal-fold mean and sample SD across 25 matched folds",
    "fusion_displayed_models": [*RAW, GATE],
    "fusion_contrasts": [list(pair) for pair in FUSION_CONTRASTS],
    "fusion_family": "raw_vs_latent_fusion", "fusion_family_size": 6,
    "fusion_difference_direction": "model minus latent gate",
    "inference": "pointwise approximate corrected-CV intervals and paired tests; Holm separately within each six-test family",
    "correction": "SE^2 = (1/25 + mean(n_outer_test/n_outer_train)) * sample variance; df=24",
    "graph_policy": "outer-test and inner-validation DDI edges masked before topology extraction",
    "raw_scaling": "symmetric pair features standardized using epoch-zero inner-training pairs only",
    "weighted_scaling": "fixed alpha=4 or learned alpha initialized at 2 applied after the same frozen pair scaler",
    "limitations": "Training-overlap correction does not fully account for shared-drug/network dependence; follow-up on a previously inspected cohort.",
}


def _validate_fold_metrics(frame):
    if set(frame.columns) != set(COLUMNS):
        raise ValueError("Topology metrics require only the documented aggregate columns")
    expected = {(model, repeat, fold) for model in MODELS for repeat in range(5) for fold in range(5)}
    keys = ["model", "split_repeat", "fold"]
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Require five models with 25 complete unique matched folds each")
    if not frame.category.eq("overall").all() or not frame.inference_protocol.eq("cv").all():
        raise ValueError("Require overall corrected-CV metrics")
    integer_columns = ["split_repeat", "fold", "training_seed", "n_outer_train", "n_outer_test", "n", "n_pos", "n_neg"]
    numbers = frame[integer_columns].to_numpy(float)
    if not np.isfinite(numbers).all() or not np.equal(numbers, np.floor(numbers)).all():
        raise ValueError("Fold identifiers and support must be finite integers")
    if (frame[["n_outer_train", "n_outer_test", "n_pos", "n_neg"]] <= 0).any().any():
        raise ValueError("Require positive partition and class support")
    if not frame.n.eq(frame.n_outer_test).all() or not frame.n.eq(frame.n_pos + frame.n_neg).all():
        raise ValueError("Test support disagrees with outer partition sizes")
    values = frame[list(METRICS)].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("AUROC and F1 must be defined and lie in [0,1]")
    metadata = ["training_seed", "n_outer_train", "n_outer_test", "n", "n_pos", "n_neg"]
    if (frame.groupby(["split_repeat", "fold"])[metadata].nunique() != 1).any().any():
        raise ValueError("Models must have identical fold support and training seeds")
    return frame.loc[:, COLUMNS].copy()


def summarize_topology(frame):
    """Recompute both paper tables with separate complete six-test families."""
    frame = _validate_fold_metrics(frame)
    summary = performance_summary(frame, METRICS, confidence=0.95)
    comparisons = comparison_table(frame, CONTRASTS, METRICS, confidence=0.95,
                                   family="raw_topology_contribution", categories=["overall"])
    if len(comparisons) != 6 or not comparisons.family_size.eq(6).all():
        raise ValueError("Require the complete six-test Holm family")
    fusion = comparison_table(frame, FUSION_CONTRASTS, METRICS, confidence=0.95,
                              family="raw_vs_latent_fusion", categories=["overall"])
    if len(fusion) != 6 or not fusion.family_size.eq(6).all():
        raise ValueError("Require the complete six-test fusion Holm family")
    numeric = _numeric_table((MESH, CONCAT), summary, comparisons, MESH, "model_minus_reference")
    fusion_numeric = _numeric_table((*RAW, GATE), summary, fusion, GATE, "model_minus_reference")
    return dict(fold_metrics=frame, summary=summary, comparisons=comparisons,
                numeric_table=numeric, paper_table=_paper_table(numeric),
                fusion_comparisons=fusion, fusion_numeric_table=fusion_numeric,
                fusion_paper_table=_paper_table(fusion_numeric))

def _summary_value(summary, model, metric):
    rows = summary[(summary.model == model) & (summary.metric == metric)]
    if len(rows) != 1:
        raise ValueError(f"Missing summary for {model}/{metric}")
    return rows.iloc[0]

def _contrast_value(frame, model_a, model_b, metric):
    rows = frame[(frame.model_a == model_a) & (frame.model_b == model_b) & (frame.metric == metric)]
    if len(rows) != 1:
        raise ValueError(f"Missing comparison for {model_a}/{model_b}/{metric}")
    return rows.iloc[0]

def _numeric_table(models, summary, contrasts, reference, direction):
    """One paper row per arm, with performance and its paired comparison."""
    rows = []
    for model in models:
        record = {"model": model, "label": LABEL[model], "reference": LABEL[reference],
                  "difference_direction": direction}
        for metric in METRICS:
            score = _summary_value(summary, model, metric)
            record.update({f"{metric}_mean": float(score.estimate), f"{metric}_sd": float(score.sd),
                           f"{metric}_ci_low": float(score.ci_low), f"{metric}_ci_high": float(score.ci_high)})
            if model == reference or (direction == "reference_minus_model" and model == MESH):
                values = (np.nan,) * 4 + ("",)
            else:
                diff = (_contrast_value(contrasts, model, reference, metric) if direction == "model_minus_reference"
                        else _contrast_value(contrasts, reference, model, metric))
                values = (float(diff.estimate), float(diff.ci_low), float(diff.ci_high),
                          float(diff.p_holm), diff.stars if isinstance(diff.stars, str) else "")
            keys = ("difference", "difference_ci_low", "difference_ci_high", "holm_p", "stars")
            record.update({f"{metric}_{key}": value for key, value in zip(keys, values)})
        rows.append(record)
    return pd.DataFrame(rows)

def _p(value):
    if not np.isfinite(value):
        return "—"
    return "<0.0001" if value < .0001 else f"{value:.4f}"

def _paper_table(table):
    if table.difference_direction.nunique() != 1 or table.reference.nunique() != 1:
        raise ValueError("Paper table mixes difference directions or references")
    direction = table.difference_direction.iloc[0]
    reference = table.reference.iloc[0]
    difference_meaning = (f"{reference} − model" if direction == "reference_minus_model"
                          else f"model − {reference}")
    rows = []
    for row in table.itertuples(index=False):
        record = {"Model": LABEL[row.model]}
        for metric in METRICS:
            name = METRIC_LABEL[metric]
            mean, sd = getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sd")
            delta = getattr(row, f"{metric}_difference")
            stars = getattr(row, f"{metric}_stars")
            record[f"{name} mean ± SD"] = f"{mean:.4f} ± {sd:.4f}"
            record[f"Δ{name} ({difference_meaning})"] = "—" if not np.isfinite(delta) else f"{delta:+.5f}{stars}"
            record[f"Holm p ({name})"] = _p(getattr(row, f"{metric}_holm_p"))
        rows.append(record)
    return pd.DataFrame(rows)

def _compact_difference(contrasts, first, second, metric):
    row = _contrast_value(contrasts, first, second, metric)
    stars = row.stars if isinstance(row.stars, str) else ""
    marker = f"^{{{stars}}}" if stars else ""
    return (r"\shortstack{$" + f"{row.estimate:+.5f}{marker}" +
            r"$\\{\scriptsize $(" + _p(row.p_holm) + r")$}}")

def _compact_score(summary, model, metric, bold=False):
    row = _summary_value(summary, model, metric)
    value = f"{row.estimate:.4f}\\pm{row.sd:.4f}"
    return "$" + (r"\mathbf{" + value + "}" if bold else value) + "$"


def _write_compact_paper_tables(tables, destination):
    """Write the topology table and five-column fusion-method comparison."""
    destination.mkdir(parents=True, exist_ok=True)
    score = lambda arm, metric: _compact_score(tables["summary"], arm, metric)
    topology_delta = lambda metric: _compact_difference(
        tables["comparisons"], CONCAT, MESH, metric)
    common = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method or contrast & AUROC & F1 \\",
        r"\midrule",
    ]
    topology = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Adding topology by direct raw-input concatenation. Scores "
         r"are mean $\pm$ SD across 25 matched folds. The difference is raw "
         r"concat minus MeSH only; parentheses give Holm-adjusted $p$-values "
         r"from approximate corrected-CV paired tests. Adjustment retains the "
         r"original six-test family of three raw methods versus MeSH only across "
         r"AUROC and F1, although only direct concat is shown. $^{**}$ indicates "
         r"adjusted $p<0.01$; F1 uses threshold $0.5$.}"),
        r"\label{tab:raw-concat-topology}",
        *common,
        f"MeSH only & {score(MESH, 'auc')} & {score(MESH, 'f1')} " + r"\\",
        f"Raw concat & {score(CONCAT, 'auc')} & {score(CONCAT, 'f1')} " + r"\\",
        r"\midrule",
        (f"Raw concat $-$ MeSH only & {topology_delta('auc')} & "
         f"{topology_delta('f1')} " + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (destination / "topology_contribution_compact.tex").write_text(
        "\n".join(topology) + "\n")
    fusion = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Input-level fusion methods compared with latent gating. "
         r"Scores are mean $\pm$ SD across 25 matched folds. Differences are "
         r"method minus latent gate; parentheses give Holm-adjusted $p$-values "
         r"from approximate corrected-CV paired tests across the separate "
         r"six-test fusion family (three methods by AUROC and F1). "
         r"$^{**}$ indicates adjusted $p<0.01$; F1 uses threshold $0.5$.}"),
        r"\label{tab:fusion-methods}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\toprule",
        r"Method & AUROC & $\Delta$AUROC & F1 & $\Delta$F1 \\",
        r"\midrule",
    ]
    names = {CONCAT: "Raw concat", FIXED: r"Fixed $\alpha=4$", LEARNED: r"Learned $\alpha$",
             GATE: r"\textbf{Latent gate}"}
    for index, model in enumerate((*RAW, GATE)):
        if index:
            fusion.append(r"\cmidrule(lr){1-5}")
        cells = [names[model]]
        for metric in METRICS:
            cells.append(_compact_score(tables["summary"], model, metric, bold=model == GATE))
            cells.append("---" if model == GATE else _compact_difference(
                tables["fusion_comparisons"], model, GATE, metric))
        fusion.append(" & ".join(cells) + r" \\")
    fusion.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    (destination / "fusion_methods.tex").write_text("\n".join(fusion) + "\n")


def _check_saved(actual, expected, filename):
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"Saved {filename} structure differs from recomputed results")
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            try:
                np.testing.assert_allclose(actual[column].to_numpy(float), expected[column].to_numpy(float),
                                           rtol=1e-12, atol=1e-14, equal_nan=True)
            except (AssertionError, ValueError) as exc:
                raise ValueError(f"Saved {filename}/{column} differs from recomputed results") from exc
        elif not actual[column].fillna("").astype(str).equals(expected[column].fillna("").astype(str)):
            raise ValueError(f"Saved {filename}/{column} differs from recomputed results")


def load_topology_tables(source_dir):
    """Read and verify public aggregate tables; private inputs are unnecessary."""
    source = Path(source_dir)
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol != PUBLIC_PROTOCOL:
        raise ValueError("Topology report protocol differs from the declared study")
    tables = summarize_topology(pd.read_csv(source / CSV_FILES["fold_metrics"], float_precision="round_trip"))
    for key, filename in CSV_FILES.items():
        if key != "fold_metrics":
            _check_saved(pd.read_csv(source / filename, float_precision="round_trip"), tables[key], filename)
    tables["protocol"] = protocol
    return tables


def _separate_output(output_dir, *source_dirs):
    output = Path(output_dir).expanduser().resolve()
    for root in source_dirs:
        source = Path(root).expanduser().resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Table output must be separate from its source directories")
    return output


def _write_tables(tables, output, provenance):
    output.mkdir(parents=True, exist_ok=True)
    for key, filename in CSV_FILES.items():
        tables[key].to_csv(output / filename, index=False)
    _write_compact_paper_tables(tables, output)
    write_json(output / "protocol.json", PUBLIC_PROTOCOL)
    write_json(output / "provenance.json", provenance)
    return tables


def write_topology_tables(source_dir, output_dir):
    """Recreate CSV and compact LaTeX tables from published fold aggregates."""
    output = _separate_output(output_dir, source_dir)
    tables = load_topology_tables(source_dir)
    source = Path(source_dir)
    provenance = json.loads((source / "provenance.json").read_text()) if (source / "provenance.json").is_file() else {}
    return _write_tables(tables, output, provenance)


def _fit_map(fits):
    result = {}
    for fit in fits:
        info = fit["metadata"]
        key = (info["model"], info["split_repeat"], info["fold"])
        if key in result:
            raise ValueError(f"Duplicate saved fit {key}")
        result[key] = fit
    return result


def export_topology_tables(input_dir, weighted_dir, output_dir):
    """Verify private fit artifacts and export both aggregate-only paper tables.

    Both completed extension studies must come from the same source folds.
    Checkpoints and pair-level predictions are never written to the output.
    """
    from dtpkg.ablation.reporting import _load
    output = _separate_output(output_dir, input_dir, weighted_dir)
    input_dir, weighted_dir = Path(input_dir), Path(weighted_dir)
    original, original_fits, missing = _load(input_dir, allow_partial=False)
    weighted, weighted_fits, weighted_missing = _load(weighted_dir, allow_partial=False)
    if missing or weighted_missing or original["study"] != "input_fusion" or weighted["study"] != "raw_weighted_fusion":
        raise ValueError("Require complete input-fusion and raw-weighted studies")
    for name in ("data_identity", "split_manifest_hash"):
        if original[name] != weighted[name]:
            raise ValueError(f"Input and weighted studies differ in {name}")
    if original.get("confidence", .95) != .95 or weighted.get("confidence", .95) != .95:
        raise ValueError("The reported protocol uses confidence 0.95")
    old, new = _fit_map(original_fits), _fit_map(weighted_fits)
    coordinates = {(repeat, fold) for repeat in range(5) for fold in range(5)}
    columns = ["drug1", "drug2", "label", "pair_bin", "training_seed", "has_target1", "has_target2",
               "in_training_negative_roster1", "in_training_negative_roster2", "isolated1", "isolated2"]
    selected, sources = [], []
    for model in MODELS:
        mapping, root, label = (old, input_dir, "input_fusion") if model in (MESH, CONCAT, GATE) else (new, weighted_dir, "raw_weighted_fusion")
        if {(rep, fold) for arm, rep, fold in mapping if arm == model} != coordinates:
            raise ValueError(f"Missing matched folds for {model}")
        for rep, fold in sorted(coordinates):
            fit = mapping[(model, rep, fold)]
            reference = old[(CONCAT, rep, fold)]
            for phase in ("test", "validation"):
                if not fit[phase][columns].equals(reference[phase][columns]):
                    raise ValueError(f"Mismatched {phase} pairs or annotations for {model}/{rep}/{fold}")
            selected.append(fit["metrics"].loc[fit["metrics"].category.eq("overall"), COLUMNS])
            relative = Path("fits") / model / f"repeat_{rep:02d}_fold_{fold:02d}"
            sources.append({"study": label, "model": model, "split_repeat": rep, "fold": fold,
                            "metrics_sha256": file_hash(root / relative / "fold_metrics.csv"),
                            "completion_sha256": file_hash(root / relative / "completed.json")})
    tables = summarize_topology(pd.concat(selected, ignore_index=True))
    provenance = {"format_version": 1, "source_protocols": {
        "input_fusion": file_hash(input_dir / "protocol.json"),
        "raw_weighted_fusion": file_hash(weighted_dir / "protocol.json")},
        "source_fits": sources, "exported_rows": len(tables["fold_metrics"]),
        "record_scope": "Aggregate overall metrics only; no drug identifiers, individual pairs or predictions."}
    return _write_tables(tables, output, provenance)
