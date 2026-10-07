"""MeSH embedding scopes and drug knowledge categories — one definition, used by
the figure scripts, the eligibility/coverage exporter and the tests.

Two different things share the words Low/Mid/Deep in this project and must not
be conflated:

* **Embedding scope** (Shallow / Intermediate / Full): which *term depths* enter
  the drug × term matrix and which *term frequency* band is retained. Defined
  over the whole annotation table. These are the three ``MeSH_*_tfidf_svd128``
  files. Operative values (defining vocabulary sizes 1,416 / 2,128 / 2,183 and the
  embedding drug counts 2,694 / 3,125 / 2,994 exactly):

      Shallow       term depth 1–5,  2 ≤ drugs/term ≤ 35
      Intermediate  term depth 1–7,  2 ≤ drugs/term ≤ 63
      Full          term depth 1–10, 2 ≤ drugs/term ≤ 52

* **Drug knowledge category** (Low / Mid / Deep): a property of each *drug*, its
  deepest assigned term level, used to stratify evaluation pairs into the six
  Low–Low … Deep–Deep bins. The ``knowledge_category`` column of
  ``extended_drug_info.csv`` uses **Low ≤ 5, Mid 6–7,
  Deep 8–10**. These drug categories are distinct from the term-depth scopes.

Term identity is the MeSH tree number; frequency is the number of distinct drugs
annotated with that tree number *after* the scope's depth filter and *before*
its frequency filter, exactly as in the preparation notebook.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dtpkg.project_paths import DATA_DIR

ANNOTATIONS = DATA_DIR / "mesh" / "extended_drug_info.csv"


@dataclass(frozen=True)
class Scope:
    name: str            # manuscript name
    level_key: str       # file/loader key
    depth_min: int
    depth_max: int
    freq_min: int        # inclusive
    freq_max: int        # inclusive
    color: str

    @property
    def embedding_path(self):
        return DATA_DIR / "mesh" / f"MeSH_{self.level_key}_tfidf_svd128.csv"

    @property
    def depth_label(self):
        return f"depth {self.depth_min}–{self.depth_max}"

    @property
    def freq_label(self):
        return f"{self.freq_min} ≤ drugs/term ≤ {self.freq_max}"


SCOPES = (
    Scope("Shallow", "low_level", 1, 5, 2, 35, "#2a9d8f"),
    Scope("Intermediate", "mid_level", 1, 7, 2, 63, "#e76f51"),
    Scope("Full", "deep_level", 1, 10, 2, 52, "#264653"),
)
SCOPE_BY_KEY = {s.level_key: s for s in SCOPES}
SCOPE_BY_NAME = {s.name: s for s in SCOPES}

#: Drug knowledge categories by deepest assigned level (inclusive upper bounds).
KNOWLEDGE_CATEGORIES = (("Low", "low_level", 1, 5), ("Mid", "mid_level", 6, 7),
                        ("Deep", "deep_level", 8, 10))


def knowledge_category(deepest_level):
    for name, key, lo, hi in KNOWLEDGE_CATEGORIES:
        if lo <= deepest_level <= hi:
            return key
    raise ValueError(f"deepest level {deepest_level} outside 1–10")


def load_annotations(path=ANNOTATIONS):
    """Ancestor-closed (drug, tree_number, level) table; one row per pair."""
    df = pd.read_csv(path, dtype={"drugbank_id": str, "tree_number": str})
    if df.duplicated(["drugbank_id", "tree_number"]).any():
        raise ValueError("annotation table has duplicate (drug, tree_number) rows")
    return df


def scope_table(annotations, scope):
    """Rows within the scope's depth range — the population frequencies are counted on."""
    return annotations[(annotations["level"] >= scope.depth_min)
                       & (annotations["level"] <= scope.depth_max)]


def term_frequencies(annotations, scope):
    """Distinct drugs per tree number in the scope population, before frequency filtering."""
    return scope_table(annotations, scope).groupby("tree_number")["drugbank_id"].nunique()


def retained_terms(annotations, scope):
    f = term_frequencies(annotations, scope)
    return f[(f >= scope.freq_min) & (f <= scope.freq_max)].index


def embedding_drugs(annotations, scope):
    """Drugs with at least one retained term — the rows of the embedding file."""
    sub = scope_table(annotations, scope)
    return set(sub[sub["tree_number"].isin(retained_terms(annotations, scope))]["drugbank_id"])


def coverage_curve(annotations, scope):
    """Cumulative fraction of the scope population covered as terms are added by
    decreasing frequency.

    numerator   distinct drugs annotated with at least one of the first k terms
    denominator drugs with at least one term in the scope's depth range
                (the population *before* frequency filtering)
    Returns (rank, fraction, is_retained) arrays, rank starting at 1.
    """
    import numpy as np
    sub = scope_table(annotations, scope)
    population = sub["drugbank_id"].nunique()
    term_drugs = sub.groupby("tree_number")["drugbank_id"].apply(set)
    freq = term_drugs.map(len)
    order = freq.sort_values(ascending=False, kind="stable").index
    seen, cov, kept = set(), [], []
    for t in order:
        seen |= term_drugs[t]
        cov.append(len(seen) / population)
        kept.append(scope.freq_min <= freq[t] <= scope.freq_max)
    return np.arange(1, len(cov) + 1), np.array(cov), np.array(kept)


def retained_coverage_curve(annotations, scope):
    """Cumulative coverage by the RETAINED vocabulary only, terms added by
    decreasing frequency.

    numerator   distinct drugs annotated with at least one of the first k
                retained terms
    denominator drugs with at least one term in the scope's depth range
                (3,342 for every scope)
    The last value is embedding_drugs / population, i.e. the drug coverage the
    stored embedding actually has. Returns (rank, fraction) with rank from 1.
    """
    import numpy as np
    sub = scope_table(annotations, scope)
    population = sub["drugbank_id"].nunique()
    kept = set(retained_terms(annotations, scope))
    term_drugs = sub[sub["tree_number"].isin(kept)].groupby("tree_number")["drugbank_id"].apply(set)
    order = term_drugs.map(len).sort_values(ascending=False, kind="stable").index
    seen, cov = set(), []
    for t in order:
        seen |= term_drugs[t]
        cov.append(len(seen) / population)
    return np.arange(1, len(cov) + 1), np.array(cov)


def scope_counts(annotations, scope, *, check_stored=True):
    """Summarize a scope, optionally comparing locally stored embedding IDs."""
    f = term_frequencies(annotations, scope)
    kept = retained_terms(annotations, scope)
    population = scope_table(annotations, scope)["drugbank_id"].nunique()
    emb = embedding_drugs(annotations, scope)
    stored = None
    if check_stored and scope.embedding_path.exists():
        stored = set(pd.read_csv(scope.embedding_path, index_col=0, usecols=[0]).index.astype(str))
    return {
        "scope": scope.name, "level_key": scope.level_key,
        "term_depth": [scope.depth_min, scope.depth_max],
        "term_frequency_inclusive": [scope.freq_min, scope.freq_max],
        "terms_before_frequency_filter": int(len(f)),
        "terms_retained": int(len(kept)),
        "terms_dropped_below_min": int((f < scope.freq_min).sum()),
        "terms_dropped_above_max": int((f > scope.freq_max).sum()),
        "population_drugs_with_any_term_in_depth_range": int(population),
        "annotation_table_drugs": int(annotations["drugbank_id"].nunique()),
        "embedding_drugs_with_a_retained_term": int(len(emb)),
        "drugs_lost_to_frequency_filter": int(population - len(emb)),
        "stored_embedding_rows": None if stored is None else int(len(stored)),
        "stored_embedding_matches_reconstruction": None if stored is None else stored == emb,
    }
