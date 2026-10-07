# Transductive fusion

[TransductiveFusion.ipynb](TransductiveFusion.ipynb) presents AUROC, F1, and
Fusion–MeSH differences by annotation-depth category.
Fill in the notebook's path placeholders and run its setup cell before the snippets below.

## Redraw the figure

Use the included [aggregate tables](tables/).

```python
from pathlib import Path
from dtpkg.transductive.plot_three_panel import render

PROJECT_ROOT = ...  # Path to your DTP-KG checkout
PROJECT_ROOT = Path(PROJECT_ROOT).expanduser().resolve()
EXPERIMENT = PROJECT_ROOT / "experiments" / "04_transductive_fusion"

render(
    source_dir=EXPERIMENT / "tables",
    out_dir=EXPERIMENT / "figures" / "reproduced",
)
```

## Regenerate the results

Use the [prepared inputs](../../data/README.md) from
[data exploration](../01_data_exploration/README.md),
[network construction](../02_network_construction/README.md), and
[MeSH scope](../03_mesh_scope/README.md). `PRIVATE_RUN` must be new or empty.

```python
from pathlib import Path
from dtpkg.transductive.experiment import run_transductive
from dtpkg.transductive.plot_three_panel import render

PROJECT_ROOT = ...  # Path to your DTP-KG checkout
DATA_DIR = ...  # Root of your prepared private data
GRAPH_PATH = ...  # Complete path to the canonical GraphML network

PROJECT_ROOT = Path(PROJECT_ROOT).expanduser().resolve()
DATA_DIR = Path(DATA_DIR).expanduser().resolve()
GRAPH_PATH = Path(GRAPH_PATH).expanduser().resolve()
EXPERIMENT = PROJECT_ROOT / "experiments" / "04_transductive_fusion"
PRIVATE_RUN = EXPERIMENT / "results" / "transductive_fusion"

run_transductive(
    out_dir=PRIVATE_RUN,
    data_dir=DATA_DIR,
    graph_path=GRAPH_PATH,
    topology_jobs=16,
)
render(
    source_dir=PRIVATE_RUN,
    out_dir=EXPERIMENT / "figures" / "reproduced" / "new_run",
)
```

## Source code

The experiment and plotting functions are in
[`src/dtpkg/transductive/`](../../src/dtpkg/transductive/), with training in
[`fusion/`](../../src/dtpkg/fusion/), models in
[`models/latent_gate.py`](../../src/dtpkg/models/latent_gate.py), and features in
[`topology.py`](../../src/dtpkg/topology.py).

## Study protocol

Five repeated five-fold partitions compare three neural models and two graph
baselines. Drugs can occur in both training and test pairs; held-out positive
DDI edges are removed before feature extraction. Checkpoints use validation loss,
and the figure reports the exploratory 96-comparison Holm family.
See the notebook for details and [snapshot notes](tables/snapshot_provenance.json)
for the saved results.

## Reproduction limits

DrugBank data and record-level derivatives are not distributed because of licensing restrictions.
