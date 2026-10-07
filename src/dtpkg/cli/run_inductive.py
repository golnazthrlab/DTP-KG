"""Fit or reuse the biological-only inductive models on locally prepared drug splits."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, help="Private immutable prepared drug splits")
    parser.add_argument("--out-dir", required=True, help="Private results root; matching completed fits can be resumed")
    parser.add_argument("--data-dir", help="Local data root containing interactions/ and mesh/")
    parser.add_argument("--graph-path", help="Local canonical DDI/DPI/PPI GraphML (DDIs removed for this study)")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--split-ids", nargs="+", help="Optional prepared split IDs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", help="PyTorch device; default CUDA if available, otherwise CPU")
    parser.add_argument("--topology-jobs", type=int, default=16)
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread limit")
    parser.add_argument("--reuse-only", action="store_true", help="Reuse or recover checkpoints; never fit missing models")
    parser.add_argument("--export-report", action="store_true", help="Export the aggregate seen–unseen report after a complete study")
    args = parser.parse_args(argv)
    from dtpkg.inductive.inductive_experiments import (
        run_prepared_splits, validate_output_location, claim_output_directory,
    )
    from dtpkg.inductive.code_snapshot import snapshot_code
    from dtpkg.inductive.validation import validate_experiment
    from dtpkg.project_paths import DATA_DIR
    from dtpkg.topology_config import CANONICAL_GRAPH
    import dtpkg
    import os
    data = Path(DATA_DIR if args.data_dir is None else args.data_dir).expanduser().resolve()
    graph = args.graph_path
    if graph is None:
        graph = (Path(os.environ["DTP_KG_NETWORK_DIR"]).expanduser() / CANONICAL_GRAPH.name
                 if os.environ.get("DTP_KG_NETWORK_DIR") else
                 CANONICAL_GRAPH if args.data_dir is None else data / "networks" / CANONICAL_GRAPH.name)
    out = validate_output_location(args.prepared_dir, args.out_dir, data, graph)
    config = validate_experiment(args.prepared_dir, graph, data_dir=data)
    claim_output_directory(out, dict(schema_version=1, kind="inductive_analysis",
                                    experiment_id=config["experiment_id"], arm="biological_only"))
    snapshot_code(Path(dtpkg.__file__).parent, out)
    predictions = run_prepared_splits(args.prepared_dir, out / "fits",
        split_ids=args.split_ids, training_seeds=args.training_seeds,
        settings=dict(epochs=args.epochs, patience=args.patience),
        data_dir=data, graph_path=graph, n_jobs=args.topology_jobs,
        device=args.device, fit_missing=not args.reuse_only, threads=args.threads)
    if args.export_report:
        from dtpkg.inductive.evaluation import scenario_metrics
        import pandas as pd
        from dtpkg.inductive.reporting import write_report, load_report, METRIC_COLUMNS, CELL_KEYS
        metrics = scenario_metrics(predictions)
        metrics = metrics[metrics.scenario.eq("seen_unseen") & metrics.model.isin(["baseline", "topo_only", "fusion"])]
        report = out / "figure_data"
        if report.exists():
            saved = load_report(report)["metrics"]
            keys = [*CELL_KEYS, "split_id", "training_seed", "model"]
            pd.testing.assert_frame_equal(saved[METRIC_COLUMNS].set_index(keys).sort_index(),
                metrics[METRIC_COLUMNS].set_index(keys).sort_index(), check_dtype=False,
                rtol=0, atol=1e-12)
        else:
            write_report(metrics, report)
    print(f"Completed private inductive results: {out}")


if __name__ == "__main__":
    main()
