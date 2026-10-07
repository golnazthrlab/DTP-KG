# Topological feature groups

[TopologyFeatureGroups.ipynb](TopologyFeatureGroups.ipynb) presents the feature-group comparison.
Fill in the notebook's path placeholders and run its setup cell before the snippets below.

## Recreate the tables

```python
from dtpkg.ablation.public_tables import load_report, performance_table, contrast_table

TABLES = ...  # This experiment's tables directory
report = load_report(TABLES, study="topological_features")
print(performance_table(report).to_string(index=False))
print(contrast_table(report).to_string(index=False))
```

## Regenerate the results

Use the [prepared inputs](../../../data/README.md) and a completed
[transductive run](../../04_transductive_fusion/README.md) as the reference.

```python
from dtpkg.ablation.workflow import prepare_study
from dtpkg.ablation.public_tables import export_tables

DATA_DIR = ...
GRAPH_PATH = ...
REFERENCE_DIR = ...
RESULTS_ROOT = ...
OUTPUT_DIR = ...  # Separate directory for the aggregate tables

study = prepare_study(
    study="topological_features", data_dir=DATA_DIR, graph_path=GRAPH_PATH,
    reference_dir=REFERENCE_DIR, results_root=RESULTS_ROOT,
)
study.run()
export_tables(study.output_dir, OUTPUT_DIR, study="topological_features")
```

## Source code

[Training](../../../src/dtpkg/ablation/workflow.py),
[study definitions](../../../src/dtpkg/ablation/specification.py), and
[table reporting](../../../src/dtpkg/ablation/public_tables.py).

## Reproduction limits

DrugBank data and record-level derivatives are not distributed because of licensing restrictions.
