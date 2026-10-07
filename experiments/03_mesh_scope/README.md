# MeSH ontology scope

[MeSHScope.ipynb](MeSHScope.ipynb) presents the scope comparison.
Fill in the notebook's path placeholders and run its setup cell before the snippets below.

## Redraw the figures

Use the included [aggregate tables](tables/) to redraw the selection figure and heatmap.

```python
from pathlib import Path

from dtpkg.mesh_scope.plot_selection import plot_scope_selection
from dtpkg.mesh_scope.plot_heatmap import render_scope_heatmap

PROJECT_ROOT = ...  # Path to your DTP-KG checkout.
PROJECT_ROOT = Path(PROJECT_ROOT).expanduser().resolve()

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "03_mesh_scope"
TABLES_DIR = EXPERIMENT_DIR / "tables"
FIGURE_DIR = EXPERIMENT_DIR / "figures" / "reproduced"

plot_scope_selection(TABLES_DIR, FIGURE_DIR)
render_scope_heatmap(TABLES_DIR, FIGURE_DIR)  # Optional notebook heatmap.
```

## Regenerate the results

Use the annotations and DDI labels saved by
[data preparation](../../data/DataPreparation.ipynb). This experiment generates
the three 128-dimensional MeSH embeddings before comparing the scopes. Set
`RUN_EMBEDDINGS = True` in the notebook to generate the embeddings without model
training, or `RUN_TRAINING = True` for the full experiment. Existing embeddings
are reused. Set the same `DATA_DIR` and use a new or empty `RUN_DIR`.

```python
from pathlib import Path

from dtpkg.mesh_scopes import SCOPES
from dtpkg.mesh_scope.embeddings import build_embeddings
from dtpkg.mesh_scope.experiment import run_scope_comparison
from dtpkg.mesh_scope.verify import verify
from dtpkg.mesh_scope.plot_selection import plot_scope_selection
from dtpkg.mesh_scope.plot_heatmap import render_scope_heatmap

PROJECT_ROOT = ...  # Path to your DTP-KG checkout.
DATA_DIR = ...  # Private input root, following data/README.md.
PROJECT_ROOT = Path(PROJECT_ROOT).expanduser().resolve()
DATA_DIR = Path(DATA_DIR).expanduser().resolve()

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "03_mesh_scope"
RUN_DIR = EXPERIMENT_DIR / "results" / "mesh_scope_run_01"
FIGURE_DIR = EXPERIMENT_DIR / "figures" / "reproduced" / "mesh_scope_run_01"

missing_scopes = [scope for scope in SCOPES
                  if not (DATA_DIR / "mesh" / f"MeSH_{scope.level_key}_tfidf_svd128.csv").is_file()]
if missing_scopes:
    build_embeddings(
        DATA_DIR / "mesh" / "extended_drug_info.csv",
        DATA_DIR / "mesh",
        scopes=missing_scopes,
        n_components=128,
    )
run_scope_comparison(RUN_DIR, data_dir=DATA_DIR)
verify(RUN_DIR, data_dir=DATA_DIR)
plot_scope_selection(RUN_DIR, FIGURE_DIR)
render_scope_heatmap(RUN_DIR, FIGURE_DIR)
```

## Source code

Preparation, training, verification, and plotting are in
[`src/dtpkg/mesh_scope/`](../../src/dtpkg/mesh_scope/). Shared definitions are in
[`mesh_scopes.py`](../../src/dtpkg/mesh_scopes.py), and the model is in
[`mesh_mlp.py`](../../src/dtpkg/models/mesh_mlp.py).

## Study protocol

| Representation | MeSH depths | Drugs per tree position |
| --- | --- | --- |
| Shallow | 1–5 | 2–35 |
| Intermediate | 1–7 | 2–63 |
| Full | 1–10 | 2–52 |

Each scope uses TF-IDF and 128-component SVD embeddings. Three repeated five-fold
partitions give 45 matched fits. Mean validation AUROC selects the Intermediate
scope; the heatmap shows exploratory test comparisons. Further details are in the notebook.

### Saved-result provenance

See the saved [protocol](tables/protocol.json) and [snapshot notes](tables/snapshot_provenance.json).

## Reproduction limits

DrugBank data and record-level derivatives are not distributed because of licensing restrictions.
