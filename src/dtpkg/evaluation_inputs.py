"""Fail early on ambiguous depth labels and incomplete evaluation features."""
import numpy as np
import pandas as pd

DEPTHS = frozenset(("low_level", "mid_level", "deep_level"))


def pair_drugs(pairs):
    return set(pairs.drug1) | set(pairs.drug2)


def validate_drug_depths(drug_to_cat, drugs):
    invalid = sorted(str(d) for d in drugs if drug_to_cat.get(d) not in DEPTHS)
    if invalid:
        raise ValueError(f"Missing or invalid Low/Mid/Deep depth for {len(invalid)} drugs: {invalid[:10]}")


def load_drug_depths(path):
    """Collapse identical repeated annotations; never silently choose a conflict."""
    frame = pd.read_csv(path, usecols=["drugbank_id", "knowledge_category"])
    if frame.isna().any().any():
        raise ValueError("Null drug ID or depth in drug annotations")
    counts = frame.groupby("drugbank_id").knowledge_category.nunique()
    conflicts = counts[counts > 1].index.tolist()
    if conflicts:
        raise ValueError(f"Conflicting depth annotations for drugs: {conflicts[:10]}")
    mapping = frame.drop_duplicates().set_index("drugbank_id").knowledge_category.to_dict()
    validate_drug_depths(mapping, mapping)
    return mapping


def validate_feature_coverage(frame, drugs, name):
    """A missing drug is an input error, not a zero/mean-imputed feature vector."""
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"{name} requires unique drug IDs and feature columns")
    missing = sorted(set(drugs) - set(frame.index))
    if missing:
        raise ValueError(f"Missing {name} features for {len(missing)} drugs: {missing[:10]}")
    if not np.isfinite(frame.loc[list(drugs)].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite {name} features in evaluated drugs")
