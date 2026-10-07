"""Run the paired MeSH scope experiment with explicitly supplied private inputs."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="New empty private run directory")
    parser.add_argument("--data-dir", help="Input data root (defaults to DTP_KG_DATA_DIR or data/)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args(argv))
    from dtpkg.mesh_scope.experiment import run_scope_comparison
    run_scope_comparison(**args)


if __name__ == "__main__":
    main()
