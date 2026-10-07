# Input data

This directory documents the inputs; it does not distribute the source datasets,
record-level derived tables, embeddings, or built networks. Obtain the source
data under the applicable terms and prepare the files locally. 

## Data sources

| Source | Records used |
| --- | --- |
| [DrugBank 5.1.13](https://go.drugbank.com/releases/5-1-13) | Drug information, documented DDIs, drug–target interactions (DPI), and drug-level MeSH assignments |
| [UniProt 2025_03 archive](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2025_03/) | Protein–protein interaction (PPI) records, identified by UniProt accessions |
| [NLM MeSH 2025](https://nlmpubs.nlm.nih.gov/projects/mesh/2025/xmlmesh/) | Descriptor hierarchy, supplied as `desc2025.xml` |
| [Zheng et al., Additional file 2](https://doi.org/10.1186/s12859-019-3214-6) | Reliable-negative DDI reference pairs |

## Input formats

Download the source records above. Supply the DrugBank drug–target associations
and UniProt protein interactions as CSVs with the columns below. You may choose
any filenames and locations; set `DPI_CSV` and `PPI_CSV` to those files.

| Notebook variable | Required input |
| --- | --- |
| `DRUGBANK_XML` | Extracted DrugBank all-drug XML |
| `MESH_XML` | Extracted MeSH descriptor XML, `desc2025.xml` |
| `NEGATIVE_SOURCE` | Additional file 2 as an `.xlsx` workbook, or a `drug1,drug2` CSV of candidate negatives |
| `DPI_CSV` | CSV with `db_id,targets`: one DrugBank primary accession and its list of target UniProt accessions per row |
| `PPI_CSV` | CSV with `uniprot_id,interactions`: one UniProt accession and its list of interaction partners per row |

Group links by drug or protein so that the first-column identifiers are unique.
List columns contain quoted accession strings inside a serialized list; use `[]`
for an empty list. Identifiers must be nonempty and have no surrounding spaces.
Additional columns are optional. Synthetic examples of the required formatting:

```csv
db_id,targets
DB00001,"['P12345', 'Q67890']"
DB00002,"['Q67890']"
```

```csv
uniprot_id,interactions
P12345,"['Q67890', 'O12345']"
Q67890,"['P12345']"
```

## Parse and save the inputs

Open [DataPreparation.ipynb](DataPreparation.ipynb),
set these five input paths, `PROJECT_ROOT`, and `DATA_DIR`, then enable
`RUN_PREPARATION`. The notebook parses the XML/workbook inputs, validates the interaction
CSVs, and saves the shared datasets under `DATA_DIR`. Input files can stay in
their original locations. The DPI/PPI output filenames are assigned by the workflow:
`interactions/DPI_enriched.csv` and `interactions/PPI_enriched.csv`.

After the notebook's path cell, the shared workflow is called with:

```python
from dtpkg.preparation import prepare_data

outputs = prepare_data(
    DATA_DIR,
    drugbank_xml=DRUGBANK_XML,
    mesh_xml=MESH_XML,
    negative_source=NEGATIVE_SOURCE,
    dpi_csv=DPI_CSV,
    ppi_csv=PPI_CSV,
)
```

Later tasks read the saved datasets from the same `DATA_DIR`. Graph construction follows in task 02, and split-specific processing remains within the relevant experiment.

When using Zheng et al.'s workbook, the code reads candidate negative pairs from
Table S6 and keeps only pairs where both drugs appear in the supplied drug–target
(DPI) table. It then removes pairs that DrugBank lists as interacting and any
pairs on the study's four-pair exclusion list. For the study inputs, this leaves
3,343 negative pairs.


## Local layout

Replace `DATA_DIR = ...` in the notebook's first code cell with the quoted path
to your own data directory, which can be outside the checkout. The following
layout applies below that root:

```text
data/
├── preparation_summary.json
├── preparation_manifest.json
├── drugbank/
│   ├── coverage_analysis_cache.csv       # prepared locally
│   ├── drugbank_ddi_release.csv          # extracted locally
│   └── drugbank_drugs_release.csv        # extracted locally
├── mesh/
│   ├── approved_drugbank_mesh.csv        # prepared locally
│   ├── mesh_hierarchy.csv                # prepared locally
│   ├── extended_drug_info.csv            # prepared locally
│   ├── MeSH_low_level_tfidf_svd128.csv   # generated in experiment 03
│   ├── MeSH_mid_level_tfidf_svd128.csv
│   └── MeSH_deep_level_tfidf_svd128.csv
├── networks/
│   └── unweighted_dppi_PubMedBERT.graphml # constructed locally in experiment 02
├── splits/
│   └── inductive_seen_unseen_v1/         # prepared locally for experiment 05
└── interactions/
    ├── unresolved/
    │   └── DDI_negative_pairs.csv        # candidates before screening
    ├── DDI_positive_pairs.csv            # resolved XML-derived positives
    ├── DDI_negative_pairs.csv            # resolved negatives
    ├── DPI_enriched.csv                  # prepared drug–target table
    └── PPI_enriched.csv                  # prepared protein–protein table
```


## Preparation and analyses

[DataPreparation.ipynb](DataPreparation.ipynb) calls the
[shared preparation workflow](../src/dtpkg/preparation.py) for DrugBank/MeSH parsing,
label screening and interaction-table validation.
The [exploration notebook](../experiments/01_data_exploration/DrugBankExplore.ipynb)
redraws the four study figures from those saved tables.

The [network guide](../experiments/02_network_construction/README.md) constructs
the graph from positive DDIs, drug–protein links, and protein–protein links.
Negative pairs are supervision labels and are never network edges.
The [MeSH-scope guide](../experiments/03_mesh_scope/README.md) builds the three
128-dimensional TF-IDF/SVD embeddings from the saved annotations, then trains
and compares scopes. Later experiments reuse the selected Intermediate embedding.

The [transductive-fusion guide](../experiments/04_transductive_fusion/README.md)
uses the Intermediate embedding, drug-depth annotations, resolved DDI labels,
and canonical network. Supply a separate network location with the notebook's
`GRAPH_PATH` setting or the CLI's `--graph-path` flag. Generated predictions,
gates, topology caches, and checkpoints remain local.

The [inductive-evaluation guide](../experiments/05_inductive_fusion/README.md)
adds locally prepared drug holdouts. Preparation uses the Intermediate embedding
file's row IDs for candidate availability, then fits new MeSH vocabulary,
TF-IDF, and SVD transforms on each holdout's development drugs. Global embedding
values are not reused. Split memberships, fitted transforms (`.joblib`), and
split-specific embeddings remain private under `splits/`.

The [ablation studies](../experiments/06_ablation/README.md) use the transductive
inputs and a completed transductive run for matched splits and reference fits.
Set `REFERENCE_DIR` to that local run. Ablation tables can be reconstructed from
the included aggregate scores; checkpoints and other record-level outputs stay local.

| Prepared input | Required fields |
| --- | --- |
| Positive and negative DDI tables | `drug1`, `drug2`: DrugBank primary accessions; unordered pairs are canonicalized and deduplicated |
| `DPI_enriched.csv` | `db_id`, `targets`: DrugBank accession and a serialized list (possibly nested) of UniProt accessions |
| `PPI_enriched.csv` | `uniprot_id`, `interactions`: UniProt accession and a serialized list of partner accessions |
| `extended_drug_info.csv` | Unique `drugbank_id` / `tree_number` rows, with `level`, `deepest_level`, and `knowledge_category`; produced by the preparation module |
| `coverage_analysis_cache.csv` | Drug type and boolean modality-presence flags; produced by the preparation module |
| `MeSH_*_tfidf_svd128.csv` | Unique DrugBank accessions in the first CSV column (index), followed by 128 numeric `svd_1`–`svd_128` columns; produced in experiment 03 by `dtpkg.mesh_scope.embeddings` |

Descriptions and other enrichment columns in the DPI/PPI files are
not used by the current graph constructor.


## Reproduction limits

DrugBank data and derived record-level datasets and networks are not included because of licensing restrictions. To reproduce the analyses, researchers must obtain appropriate DrugBank access and provide the locally prepared inputs described in this README.
