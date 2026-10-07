"""Scope experiment labels and declared comparison family (no training imports)."""
from dtpkg.mesh_scopes import SCOPES

LEVELS = tuple(s.level_key for s in SCOPES)
NAMES = {s.level_key: s.name for s in SCOPES}
CATEGORIES = ("low-low", "low-mid", "low-deep", "mid-mid", "mid-deep", "deep-deep")
CONTRASTS = (("mid_level", "low_level"), ("deep_level", "low_level"),
             ("deep_level", "mid_level"))
METRICS = ("auc", "f1", "ap", "mcc")
