# Network construction

[NetworkConstruction.ipynb](NetworkConstruction.ipynb) builds and verifies the
drug–target–protein network from the outputs of
[data preparation](../../data/DataPreparation.ipynb).
Fill in its path placeholders and run the setup cell before the snippet below.

## Build and verify

```python
from dtpkg.network.construct_network import main as construct_network
from dtpkg.cli.verify_graph_source import main as verify_graph_source

DATA_DIR = ...  # The output directory from data preparation
GRAPH_PATH = ...  # Output GraphML file

construct_network(["--data-dir", str(DATA_DIR), "--out", str(GRAPH_PATH)])
verify_graph_source(["--data-dir", str(DATA_DIR), "--graph", str(GRAPH_PATH)])
```

The builder uses these prepared files under `DATA_DIR / "interactions"`:

| File | Required columns |
|---|---|
| `DDI_positive_pairs.csv` | `drug1`, `drug2`: unique, unordered, non-self pairs |
| `DPI_enriched.csv` | `db_id`, `targets`: drug ID and serialized list of target UniProt IDs |
| `PPI_enriched.csv` | `uniprot_id`, `interactions`: protein ID and serialized list of interacting protein IDs |

The graph is undirected and unweighted. Despite its historical filename,
`unweighted_dppi_PubMedBERT.graphml`, it uses no PubMedBERT embeddings.

Reliable-negative pairs are supervision labels and are never network edges.
Input preparation is described in the [data guide](../../data/README.md).

## Source code

[Graph construction](../../src/dtpkg/network/) and
[verification](../../src/dtpkg/cli/verify_graph_source.py).

## Reproduction limits

DrugBank inputs and derived networks are omitted because of licensing
restrictions.
