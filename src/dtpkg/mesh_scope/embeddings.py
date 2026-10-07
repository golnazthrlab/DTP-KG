"""Create the three private MeSH embeddings used by the scope experiment.

This extracts the study's preparation notebook: binary drug/tree-number
features, default TF-IDF weighting, then randomized SVD with seed 42. Drugs
and terms are sorted before fitting; drugs with no retained terms are omitted.
The scopes therefore have different drug populations. Training compares their
common eligible population separately.

Inputs and generated embeddings contain licensed record-level information and
must remain local. This module does not download data or run on import.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from dtpkg.mesh_scopes import (
    SCOPES, SCOPE_BY_KEY, Scope, load_annotations, retained_terms, scope_table,
)


N_COMPONENTS = 128
RANDOM_STATE = 42


def build_feature_matrix(annotations: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    """Return sorted binary drug/term features after depth/frequency filtering.

    Frequency means distinct drugs per tree number within the depth range, and
    both frequency bounds are inclusive. The integer matrix and unnamed index
    match the original ``create_mesh_datasets.ipynb`` preparation procedure.
    """
    required = {"drugbank_id", "tree_number", "level"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    if annotations.empty or annotations[list(required)].isna().any().any():
        raise ValueError("Annotations must be nonempty with no missing IDs, terms, or levels")
    if annotations.duplicated(["drugbank_id", "tree_number"]).any():
        raise ValueError("Annotation table has duplicate (drug, tree_number) rows")
    if not annotations["level"].between(1, max(s.depth_max for s in SCOPES)).all():
        raise ValueError("MeSH levels must be within the study's supported depth range 1–10")

    subset = scope_table(annotations, scope)
    filtered = subset[subset["tree_number"].isin(retained_terms(annotations, scope))]
    drug_to_terms = filtered.groupby("drugbank_id")["tree_number"].apply(set).to_dict()
    all_terms = sorted(set(filtered["tree_number"]))
    if not all_terms:
        raise ValueError(f"No MeSH terms survive filtering for {scope.name}")
    all_drugs = sorted(annotations["drugbank_id"].unique())
    binary = MultiLabelBinarizer(classes=all_terms)
    matrix = binary.fit_transform([drug_to_terms.get(drug, set()) for drug in all_drugs])
    features = pd.DataFrame(matrix, index=all_drugs, columns=binary.classes_)
    return features.loc[features.sum(axis=1) != 0]


def build_embedding(
    annotations: pd.DataFrame,
    scope: Scope,
    *,
    n_components: int = N_COMPONENTS,
) -> pd.DataFrame:
    """Fit the study's TF-IDF/SVD embedding, retaining float64 precision.

    The manuscript uses 128 components. ``n_components`` is exposed for small
    synthetic examples; changing it creates a different experiment. Scikit-learn's
    default TF-IDF and randomized-SVD options are deliberately retained, as in the
    original notebook. Reproduction requires the recorded library versions.
    """
    features = build_feature_matrix(annotations, scope)
    if n_components < 1 or n_components > min(features.shape):
        raise ValueError(
            f"{n_components} SVD components require at least that many drugs and terms; "
            f"{scope.name} has shape {features.shape}"
        )
    tfidf = TfidfTransformer().fit_transform(features.values)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(tfidf)
    return pd.DataFrame(
        reduced,
        index=features.index,
        columns=[f"svd_{i + 1}" for i in range(reduced.shape[1])],
    )


def build_embeddings(
    annotation_csv: str | Path,
    output_dir: str | Path,
    *,
    scopes: Sequence[Scope] = SCOPES,
    n_components: int = N_COMPONENTS,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write private embedding CSVs to an explicit directory.

    Every destination is checked before fitting. Existing files are refused unless
    ``overwrite=True`` is explicit; write to a new directory when comparing a
    reconstruction with the embeddings used for the published experiment.
    """
    if not scopes or len({scope.level_key for scope in scopes}) != len(scopes):
        raise ValueError("Provide at least one scope, with no duplicate scope keys")
    output_dir = Path(output_dir)
    paths = {
        scope.level_key: output_dir / f"MeSH_{scope.level_key}_tfidf_svd{n_components}.csv"
        for scope in scopes
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Embedding output already exists: {existing[0]}")
    annotations = load_annotations(annotation_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scope in scopes:
        embedding = build_embedding(annotations, scope, n_components=n_components)
        path = paths[scope.level_key]
        embedding.to_csv(path, mode="w" if overwrite else "x")
        print(f"{scope.name}: {len(embedding):,} drugs × {embedding.shape[1]} features → {path}")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Private extended_drug_info.csv from data/DataPreparation.ipynb")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Explicit local directory for the three private embedding CSVs")
    parser.add_argument("--scope", action="append", choices=list(SCOPE_BY_KEY),
                        help="Scope to build; repeat to select several (default: all three)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacing existing generated embeddings")
    args = parser.parse_args(argv)
    scopes = tuple(SCOPE_BY_KEY[key] for key in args.scope) if args.scope else SCOPES
    build_embeddings(args.annotations, args.output_dir, scopes=scopes, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
