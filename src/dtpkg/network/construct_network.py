"""Construct the canonical graph from three prepared interaction tables."""
import argparse
from pathlib import Path

from dtpkg.project_paths import DATA_DIR, NETWORK_DIR
from dtpkg.network.biological_network import BiologicalNetwork


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="data root containing interactions/")
    parser.add_argument("--out", type=Path,
                        default=NETWORK_DIR / "unweighted_dppi_PubMedBERT.graphml")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output graph")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} already exists; pass --overwrite to replace it")
    inputs = args.data_dir / "interactions"
    paths = [inputs / name for name in
             ("DDI_positive_pairs.csv", "DPI_enriched.csv", "PPI_enriched.csv")]
    for path in paths:
        if not path.is_file():
            parser.error(f"missing prepared input: {path}; see experiments/02_network_construction/README.md")
    net = BiologicalNetwork(*paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(args.out)
    print(f"Nodes: {len(net.graph):,}; edges: {net.graph.number_of_edges():,}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
