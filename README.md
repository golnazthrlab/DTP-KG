# README

## Overview
This repository contains the full pipeline for drug–drug interaction (DDI) prediction using a biologically infused Drug–Target–Protein Knowledge Graph (DTP-KG) network. The project integrates:

- Biological features (MeSH-based drug embeddings)
- Topological features extracted from the DTP-KG network
- A Latent Gated Fusion architecture that learns to merge both modalities inside a shared latent space

The repo includes multiple fusion strategies, statistical significance tests (DeLong), grouped topological ablations, and a mechanistic case study.

---

## Repository Structure (Short Descriptions)

**BiologicalNetwork.py**  
Constructs the DTP-KG network.

**TopoFeatExtractor.py**  
Leak-free extraction of topological features. Its compute_for_fold method removes DDI edges from validation/test to avoid information leakage.

**LatentGateModel.py**  
The latent-space gating architecture that fuses biological and topological signals.

**FusionTrainer.py**  
Training logic for LatentGateModel, including cross-validation and DeLong statistical comparison.

**TopoAblation.py**  
Implements grouped ablation for topological feature subsets.

**MeSHDataLoaders.py**  
Utilities for MeSH-based feature loading with configurable specificity levels.

**utils.py**  
Shared evaluation + visualization helpers.

---

## Folders

### PoC/
Contains all proof-of-concept fusion attempts:
- Simple concatenation
- Weighted concatenation
- Learnable-weight concatenation
- Latent-gated fusion

Each notebook reports CV performance for baseline (Bio only) and fusion (Bio+Topo) and includes DeLong test results.

### PoC_MeSH/
Experiments using different MeSH hierarchy levels across six drug categories. Includes all accuracy/AUC/F1/precision/recall plots and bin-level results.

### results/
Contains all experiment outputs:
- Cross-validation performance by category
- Fusion vs baseline comparisons
- DeLong stat significance plots
- Grouped ablations (topological features only)
- Inductive evaluation figures
- Case study neighborhood visualizations (interacting vs non-interacting)

---

## Notebooks

**LatentGate_MLP.ipynb**  
Main notebook: loads data, trains baseline and fusion models using CV, computes DeLong p-values.

**run_ablation_topo_only.ipynb**  
Runs grouped ablation experiments using topo-only models, producing summary plots across metrics.

**case_study.ipynb**  
Splits a held-out test set, trains a topo-only model with early stopping, selects confident positive/negative predictions, and visualizes 2-hop mechanistic protein neighborhoods.
