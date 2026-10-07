# Case study

[CaseStudy.ipynb](CaseStudy.ipynb) presents the
[protein-neighborhood illustration](figures/seen_unseen_indirect_overlap_example.png)
and [adjusted overlap correlations](figures/adjusted_overlap_correlations.png).
Fill in the notebook's path placeholders and run its setup cell before the snippets below.

## Redraw the figures

The correlation panel uses the included aggregate tables:

```python
from pathlib import Path
from dtpkg.case_study.adjusted_overlap_figure import plot_adjusted_overlap

TABLES = ...  # This experiment's tables directory
OUTPUT_DIR = ...
OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()
plot_adjusted_overlap(TABLES, OUTPUT_DIR)
```

The network illustration uses privately exported neighborhoods:

```python
from dtpkg.case_study.render_indirect_example import render_example

EXAMPLES_JSON = ...  # Local comparative_overlap_examples/examples.json
render_example(EXAMPLES_JSON, OUTPUT_DIR / "seen_unseen_indirect_overlap_example")
```

## Regenerate the results

Use local [prepared inputs](../../data/README.md) and completed topology-only
fits. The paper snapshot contains 25 transductive fits and 15 DDI-free inductive
fits from five holdouts and three seeds. Experiment 05 uses different holdouts;
analysing those fits produces a new case-study result.

```python
from pathlib import Path
from dtpkg.case_study.workflow import run_analysis
from dtpkg.case_study.reporting import export_tables
from dtpkg.case_study.examples import export_examples

DATA_DIR = ...
GRAPH_PATH = ...
TRANSDUCTIVE_RUN = ...
INDUCTIVE_RUN = ...
PREPARED_DIR = ...  # Holdouts used by INDUCTIVE_RUN
RESULTS_DIR = ...  # Private case-study output directory

run_analysis("transductive", RESULTS_DIR, source_options={
    "results_dir": TRANSDUCTIVE_RUN, "graph_path": GRAPH_PATH,
})
run_analysis("inductive", RESULTS_DIR, source_options={
    "results_dir": INDUCTIVE_RUN, "prepared_dir": PREPARED_DIR,
    "graph_path": GRAPH_PATH, "data_dir": DATA_DIR,
})
tables = Path(RESULTS_DIR) / "public_tables"
export_tables(RESULTS_DIR, tables)
plot_adjusted_overlap(tables, OUTPUT_DIR)

examples = Path(RESULTS_DIR) / "comparative_overlap_examples"
export_examples(
    RESULTS_DIR, GRAPH_PATH, Path(PREPARED_DIR) / "split_00" / "manifest.json",
    Path(DATA_DIR) / "drugbank" / "drugbank_drugs_release.csv", examples,
)
render_example(examples / "examples.json", OUTPUT_DIR / "seen_unseen_indirect_overlap_example")
```

The illustration is selected post hoc. Correlations adjust for endpoint
neighborhood sizes; the plotted split variation is descriptive.

Implementation: [`src/dtpkg/case_study/`](../../src/dtpkg/case_study/).
DrugBank data, predictions, checkpoints, and neighborhood graphs are not
distributed because of licensing restrictions.
