# Inductive fusion

[InductiveEvaluation.ipynb](InductiveEvaluation.ipynb) presents the selected
seen–unseen [AUROC](figures/seen_unseen_auroc.png), each showing overall and MeSH-depth
results. The notebook runs from the public aggregate tables in `tables/`.

## Study protocol

Three supported drug holdouts, each evaluated with three training seeds, give
nine split–seed runs and 27 neural model fits. MeSH transforms are fitted on
development drugs only. All DDI edges are removed; held-out nodes are also
removed from training graphs, while their available non-DDI connections can be
used at inference. Seen partners must occur in the first training epoch.

Scores are averaged over seeds within each holdout, then over the three
holdouts. Figure bars show between-holdout sample SD. The paired figures retain
Holm adjustment across four overall and 60 depth comparisons, including metrics
not displayed. Approximate overlap-corrected tests use three holdout differences
(df = 2). The notebook explains the selection, training, and inference details.

## Redraw the figures

Run these snippets after the notebook's setup cell. The required libraries are
listed in the [main setup instructions](../../README.md#run-the-notebooks).

```python
from dtpkg.inductive.plot_paired import render_figures

TABLES = ...  # This experiment's public tables directory
OUTPUT_DIR = ...
render_figures(TABLES, OUTPUT_DIR)
```

## Regenerate the results

Generate the Intermediate embedding in the
[MeSH-scope notebook](../03_mesh_scope/MeSHScope.ipynb) with `RUN_EMBEDDINGS = True`
first. Its drug IDs define the available population; this experiment fits new
representations on development drugs.

Fill in the local input and output paths in the notebook, then enable
`PREPARE_SPLITS` and `RUN_TRAINING`. Python prepares the drug holdouts and
split-specific MeSH transforms, trains the models, and exports the results.
The [data guide](../../data/README.md) describes the inputs.

```python
from dtpkg.cli.prepare_inductive import main as prepare_inductive
from dtpkg.cli.run_inductive import main as run_inductive

DATA_DIR = ...
GRAPH_PATH = ...
PREPARED_DIR = ...
RUN_DIR = ...

prepare_inductive([
    "--data-dir", str(DATA_DIR), "--graph-path", str(GRAPH_PATH),
    "--out-dir", str(PREPARED_DIR),
])
run_inductive([
    "--data-dir", str(DATA_DIR), "--graph-path", str(GRAPH_PATH),
    "--prepared-dir", str(PREPARED_DIR), "--out-dir", str(RUN_DIR),
    "--export-report",
])
```

Source: [`src/dtpkg/inductive/`](../../src/dtpkg/inductive/).
DrugBank inputs, split memberships, transforms, predictions, and checkpoints
remain private.
