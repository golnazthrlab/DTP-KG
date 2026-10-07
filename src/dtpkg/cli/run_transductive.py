"""Command-line entry point for the local transductive fusion experiment."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Empty directory for private results and checkpoints")
    parser.add_argument("--data-dir", help="Local data root containing interactions/ and mesh/")
    parser.add_argument("--graph-path", help="Local canonical full DDI/DPI/PPI GraphML")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device; default CUDA if available, otherwise CPU")
    parser.add_argument("--topology-jobs", type=int, default=16)
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread limit")
    args = parser.parse_args(argv)
    from dtpkg.transductive.experiment import run_transductive
    path = run_transductive(**vars(args))
    print(f"Completed local transductive run: {path}")


if __name__ == "__main__":
    main()
