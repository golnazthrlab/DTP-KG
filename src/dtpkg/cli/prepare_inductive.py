"""Prepare the count-supported inductive drug holdouts from licensed local inputs."""
import argparse
from pathlib import Path

from dtpkg.inductive.prepare_seen_unseen import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="New private output folder; existing files are never overwritten.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--graph-path", type=Path)
    parser.add_argument("--n-holdouts", type=int, default=3)
    parser.add_argument("--proposal-seed", type=int, default=30000)
    parser.add_argument("--max-proposals", type=int, default=200)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-heldout", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=.15)
    parser.add_argument("--n-components", type=int, default=128)
    args = parser.parse_args(argv)
    prepare(args.out_dir, data_dir=args.data_dir, graph_path=args.graph_path,
            n_holdouts=args.n_holdouts, proposal_seed=args.proposal_seed,
            max_proposals=args.max_proposals, training_seeds=tuple(args.training_seeds),
            min_pairs=args.min_pairs, min_heldout=args.min_heldout,
            fraction=args.fraction, n_components=args.n_components)


if __name__ == "__main__":
    main()
