# Topology contribution

[TopologyContribution.ipynb](TopologyContribution.ipynb) presents both tables:
MeSH-only versus raw concat, and input-level fusion methods versus latent gating.
Fill in the notebook's path placeholders and run its setup cell before the snippets below.

## Recreate the tables

```python
from dtpkg.ablation.topology_reporting import load_topology_tables

TABLES = ...  # This experiment's tables directory
report = load_topology_tables(TABLES)
print(report["paper_table"].to_string(index=False))
print(report["fusion_paper_table"].to_string(index=False))
```

The second table uses method-minus-gate differences. CSV and LaTeX exports are in [tables/](tables/).

## Regenerate the results

Use the [prepared inputs](../../../data/README.md) and a completed
[transductive run](../../04_transductive_fusion/README.md).
Each table retains its own six-test Holm family.

```python
from pathlib import Path
from dtpkg.ablation.workflow import prepare_study
from dtpkg.ablation.input_fusion import prepare_input_fusion, run_input_fusion
from dtpkg.ablation.raw_weighted_fusion import prepare_raw_weighted, run_raw_weighted
from dtpkg.ablation.topology_reporting import export_topology_tables

DATA_DIR = ...
GRAPH_PATH = ...
REFERENCE_DIR = ...
RESULTS_ROOT = ...
OUTPUT_DIR = ...  # Separate directory for the aggregate tables
RESULTS_ROOT = Path(RESULTS_ROOT).expanduser().resolve()

fusion = prepare_study(
    study="fusion_methods", data_dir=DATA_DIR, graph_path=GRAPH_PATH,
    reference_dir=REFERENCE_DIR, results_root=RESULTS_ROOT,
)
fusion.run()
input_run = RESULTS_ROOT / "input_fusion"
weighted_run = RESULTS_ROOT / "raw_weighted_fusion"
prepare_input_fusion(fusion.output_dir, input_run)
run_input_fusion(input_run)
prepare_raw_weighted(input_run, weighted_run)
run_raw_weighted(weighted_run)
export_topology_tables(input_run, weighted_run, OUTPUT_DIR)
```

## Source code

[Shared training](../../../src/dtpkg/ablation/workflow.py),
[input-fusion controls](../../../src/dtpkg/ablation/input_fusion.py),
[weighting controls](../../../src/dtpkg/ablation/raw_weighted_fusion.py), and
[table reporting](../../../src/dtpkg/ablation/topology_reporting.py).

## Reproduction limits

DrugBank data and record-level derivatives are not distributed because of licensing restrictions.
