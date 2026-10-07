# Study experiments

Each task keeps its notebook, instructions, and selected figures and tables here. Reusable
Python code lives in [`src/dtpkg/`](../src/dtpkg/); the notebooks configure and
load that code directly after their path setup cell.

Start with [data preparation](../data/DataPreparation.ipynb), then run the
experiments below using its saved datasets. Fill in the path placeholders in
each notebook before running it.

| Task | Status |
| --- | --- |
| [01 — Data exploration](01_data_exploration/README.md) | Exploration notebook and four figures |
| [02 — Network construction](02_network_construction/README.md) | Notebook, graph construction, and verification code |
| [03 — MeSH ontology scope](03_mesh_scope/README.md) | Reviewed embeddings/training code, notebook, selection figure, heatmap, and aggregate scores |
| [04 — Transductive fusion](04_transductive_fusion/README.md) | Reviewed topology/model/training code, notebook, three-panel paper figure, and aggregate scores |
| [05 — Inductive fusion](05_inductive_fusion/README.md) | Reviewed split preparation/training code, notebook, seen–unseen AUROC/F1 figures, and aggregate scores |
| [06 — Ablation](06_ablation/README.md) | Three notebooks, publication tables, aggregate scores, and training code |
| [07 — Case study](07_case_study/README.md) | Notebook, two paper figures, aggregate correlations, and analysis code |

<!-- Guide style: redraw code, regenerate code, then source links. Keep prose brief and use ... path placeholders. -->
