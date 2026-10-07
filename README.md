# DTP-KG

Code and experiment notebooks for the DTP-KG drug–drug interaction study.
The publication checkout contains data preparation, exploration, network
construction, MeSH scope, transductive fusion, inductive evaluation, ablation,
and case-study workflows. Download the required source files, prepare the shared
datasets with the supplied Python functions, then use those datasets throughout
the experiments.

| Directory | Contents |
| --- | --- |
| [`data/`](data/README.md) | Source guide, input schemas, and parsing notebook; no datasets |
| `src/dtpkg/` | Importable Python implementation shared by the experiment notebooks |
| [`experiments/`](experiments/README.md) | Task notebooks, instructions, and publication figures and tables |

## Code and experiment layout

```text
data/
├── README.md              # source downloads and required input formats
└── DataPreparation.ipynb  # shared input parsing and dataset preparation
src/dtpkg/
├── preparation.py         # shared data preparation entry point
├── project_paths.py       # fallback locations for CLI and library calls
├── ddi_labels.py          # extraction and label preparation
├── mesh_scopes.py         # shared MeSH definitions and coverage calculations
├── drugbank/              # preparation and exploration plotting functions
├── network/               # graph constructor
├── mesh_scope/            # embeddings, training, evaluation, and scope figures
├── transductive/          # transductive experiment and three-panel figure
├── inductive/             # drug holdouts, preparation, training, and evaluation
├── ablation/              # topology, feature-group, and edge-type comparisons
├── case_study/            # protein-neighborhood examples and overlap analysis
├── fusion/                # fusion training and comparator methods
├── topology.py            # topology feature extraction
├── models/                # model definitions
└── cli/                   # preparation, verification, and experiment commands
experiments/
├── 01_data_exploration/   # exploration notebook and figures
├── 02_network_construction/ # notebook and graph construction instructions
├── 03_mesh_scope/         # notebook, selection figure, heatmap, aggregate scores
├── 04_transductive_fusion/ # notebook, paper figure, aggregate scores
├── 05_inductive_fusion/   # notebook, seen–unseen AUROC/F1 figures, aggregate scores
├── 06_ablation/           # notebooks, publication tables, aggregate scores
│   ├── topology_contribution/
│   ├── topology_feature_groups/
│   └── graph_edge_types/
└── 07_case_study/         # notebook, two paper figures, aggregate correlations
```

Experiment notebooks contain the explanation, configuration, and calls to the
Python source code. Python implementations live in `src/dtpkg/`; experiments
share preparation, models, training, evaluation, and plotting functions. 
Selected figures belong beside their experiment notebooks; private inputs and
generated networks stay outside the tracked code.

Fill the `...` path placeholders in each notebook's first code cell before
running it, replacing them with quoted filesystem paths or `Path(...)` values.
The cell also contains output directories and run options; no configuration
file or environment-variable setup is needed.

## Run the notebooks

Use Python 3.12. From the checkout folder, install the required libraries once
and open the notebooks:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab data/DataPreparation.ipynb
```

### Prepare the data

The study uses DrugBank for drug information, drug-level MeSH annotations,
DDIs and drug–protein interactions (DPI); NLM MeSH for the descriptor hierarchy;
UniProt for protein–protein interactions (PPI); and Zheng et al.’s supplement
for reliable-negative pairs. Source files and download instructions are listed
in the [data guide](data/README.md).

In [DataPreparation.ipynb](data/DataPreparation.ipynb),
set the input paths and `DATA_DIR = ...`, then enable `RUN_PREPARATION`.
The notebook calls [`prepare_data()`](src/dtpkg/preparation.py) to parse and validate
the inputs, resolve DDI labels, and generate coverage and annotation tables.
It saves these datasets and the validated interaction tables under `DATA_DIR`. All later notebooks use those saved files; source-file paths
are supplied only in the data preparation notebook.

### Run the experiments

Use the same `DATA_DIR` in the later notebooks and run the tasks in order.
Task 02 constructs the network from the prepared interaction tables. Task 03
builds the MeSH embeddings and compares their scopes. Later model experiments
reuse the saved labels, annotations, and selected embeddings. The inductive task
creates its own holdouts and fits representations on development drugs.

Each notebook loads the Python implementation from `src/`. Fill in its remaining
path placeholders and enable the relevant preparation or training options.

The exploration notebook displays the four supplied paper figures. Recomputing
them requires local licensed/prepared inputs; see the
[exploration instructions](experiments/01_data_exploration/README.md).

The [network guide](experiments/02_network_construction/README.md)
describes graph construction. The [MeSH-scope notebook](experiments/03_mesh_scope/MeSHScope.ipynb) presents the paper's scope-selection figure and an additional heatmap. Both scope figures can be redrawn from public aggregate tables without private data;
see the [scope experiment guide](experiments/03_mesh_scope/README.md) for training.

The [transductive-fusion notebook](experiments/04_transductive_fusion/TransductiveFusion.ipynb) presents the five-method comparison; its three-panel figure can also be redrawn
from public aggregate scores. The [fusion guide](experiments/04_transductive_fusion/README.md) provides the training command.

The [inductive-evaluation notebook](experiments/05_inductive_fusion/InductiveEvaluation.ipynb) presents the seen–unseen AUROC and F1 figures; its [guide](experiments/05_inductive_fusion/README.md) documents drug-holdout preparation and training.

The [ablation notebooks](experiments/06_ablation/README.md) present tables for
topology contribution, feature groups, and graph edge types, with code to rerun each study.

The [case-study notebook](experiments/07_case_study/CaseStudy.ipynb) presents the
protein-neighborhood illustration and adjusted overlap correlations, with code
to redraw the figures and repeat the analysis from local completed fits.

## Private data and generated networks

Set `PROJECT_ROOT = ...` to your checkout and `DATA_DIR = ...` to your private
data root, following the layout in [`data/README.md`](data/README.md).
The fusion notebooks also require a `GRAPH_PATH = ...` network location for
preparation or training. Review the notebook's output paths before enabling
those steps. Command-line users can pass the path flags shown in each
experiment guide.

DrugBank records and derived record-level tables/networks are not distributed.
To reproduce the analyses, obtain appropriate DrugBank access and provide the
locally prepared inputs described in the data guide.
