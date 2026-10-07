"""The experiment choices, kept outside notebook presentation and fitting code."""
from dataclasses import asdict, dataclass
from pathlib import Path



@dataclass(frozen=True)
class Arm:
    name: str
    model_kind: str = "fusion"
    graph_variant: str = "full"
    gate_mode: str = "adaptive_vector"
    feature_group: str = "all"
    positive_sampling: str = "per_epoch"
    topology_scaling: str = "raw"
    reference_model: str | None = None
    reference_only: bool = False


@dataclass(frozen=True)
class StudyConfig:
    study: str
    data_dir: Path
    results_root: Path
    reference_dir: Path
    repetitions: tuple | None = None
    folds: tuple | None = None
    epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    seed: int = 42
    n_jobs: int = 16
    device: str = "cpu"
    reuse_reference: bool = True
    graph_path: Path | None = None
    positive_path: Path | None = None
    negative_path: Path | None = None
    embedding_path: Path | None = None
    depth_path: Path | None = None
    split_manifest_path: Path | None = None
    latent_dim: int = 128
    enc_hidden: tuple = (256,)
    head_hidden: tuple = (256, 128)
    topo_hidden: tuple = (64, 32)
    dropout: float = 0.1
    confidence: float = 0.95

    def __post_init__(self):
        study_design(self.study)
        for name in ("data_dir", "results_root", "reference_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        defaults = {
            "graph_path": "networks/unweighted_dppi_PubMedBERT.graphml",
            "positive_path": "interactions/DDI_positive_pairs.csv",
            "negative_path": "interactions/DDI_negative_pairs.csv",
            "embedding_path": "mesh/MeSH_mid_level_tfidf_svd128.csv",
            "depth_path": "mesh/extended_drug_info.csv",
        }
        for name, relative in defaults.items():
            value = getattr(self, name)
            object.__setattr__(self, name, Path(value if value is not None else self.data_dir / relative).expanduser().resolve())
        if self.split_manifest_path is not None:
            object.__setattr__(self, "split_manifest_path", Path(self.split_manifest_path).expanduser().resolve())
        for name in ("epochs", "patience", "batch_size", "n_jobs", "latent_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid seed or optimizer settings")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        for name in ("repetitions", "folds"):
            value = getattr(self, name)
            if value is not None:
                if not value or len(set(value)) != len(value) or any(int(x) != x or x < 0 for x in value):
                    raise ValueError(f"{name} must contain distinct nonnegative indices")

    @property
    def manifest_path(self):
        return Path(self.split_manifest_path or Path(self.reference_dir) / "split_manifest.json")

    @property
    def output_dir(self):
        root = Path(self.results_root) / self.study
        if self.repetitions is not None or self.folds is not None:
            names = ["all" if value is None else "-".join(map(str, sorted(value)))
                     for value in (self.repetitions, self.folds)]
            root = root / "subsets" / f"repeats_{names[0]}__folds_{names[1]}"
        return root


FEATURE_GROUPS = {
    "degree_size": ("deg_total", "deg_drug", "deg_prot", "deg_min", "deg_max",
                    "deg_mean", "deg_std", "n_nodes_subgraph"),
    "clustering_boundary": ("clust_local", "boundary_ratio"),
    "centrality": ("close_local", "katz_local"),
}
FEATURE_COLUMNS = tuple(c for group in FEATURE_GROUPS.values() for c in group)


def reference_arms():
    return [Arm("mesh_only", model_kind="mesh", reference_model="baseline", reference_only=True),
            *[Arm(name, model_kind=name, reference_only=True)
              for name in ("common_neighbors", "degree_product", "negative_count")]]


def study_design(study):
    """Prespecified arms and contrasts; no selection from test results."""
    if study == "fusion_methods":
        arms = [Arm("adaptive_vector", reference_model="fusion"),
                Arm("fixed_half", gate_mode="fixed_half"),
                Arm("adaptive_scalar", gate_mode="adaptive_scalar"),
                Arm("concat_projection", gate_mode="concat_projection"), *reference_arms()]
        contrasts = [("adaptive_vector", name) for name in ("fixed_half", "adaptive_scalar", "concat_projection")]
        contrasts += [("adaptive_scalar", "fixed_half")]
    elif study == "edge_types":
        arms = [Arm(f"{graph}_{kind}", model_kind=kind, graph_variant=graph,
                    reference_model=("fusion" if kind == "fusion" else "topo_only") if graph == "full" else None)
                for graph in ("full", "no_ddi", "ddi_only") for kind in ("fusion", "topo_only")]
        arms += reference_arms()
        contrasts = [(f"full_{kind}", f"{graph}_{kind}")
                     for kind in ("fusion", "topo_only") for graph in ("no_ddi", "ddi_only")]
        contrasts += [("no_ddi_fusion", "mesh_only")]
    elif study == "topological_features":
        arms = [Arm("all_descriptors", model_kind="topo_only", reference_model="topo_only")]
        arms += [Arm(f"without_{group}", model_kind="topo_only", feature_group=group)
                 for group in FEATURE_GROUPS]
        contrasts = [("all_descriptors", arm.name) for arm in arms[1:]]
    elif study == "sampling_policy":
        arms = [Arm(f"{policy}_{kind}", model_kind=kind, positive_sampling=policy,
                    reference_model={"mesh": "baseline", "fusion": "fusion", "topo_only": "topo_only"}[kind]
                    if policy == "per_epoch" else None)
                for kind in ("mesh", "fusion", "topo_only") for policy in ("per_epoch", "fixed")]
        contrasts = [(f"per_epoch_{kind}", f"fixed_{kind}") for kind in ("mesh", "fusion", "topo_only")]
    elif study == "topology_scaling":
        arms = [Arm("raw_topology", reference_model="fusion"),
                Arm("standardized_topology", topology_scaling="train_standardized")]
        contrasts = [("standardized_topology", "raw_topology")]
    else:
        raise ValueError(f"Unknown study {study!r}")
    return arms, contrasts


def scientific_config(config):
    excluded = {"data_dir", "results_root", "reference_dir", "repetitions", "folds", "n_jobs", "device",
                "reuse_reference", "split_manifest_path", "graph_path", "positive_path", "negative_path",
                "embedding_path", "depth_path"}
    return {k: v for k, v in asdict(config).items() if k not in excluded}
