"""Atomic artifacts and explicit identities for resumable ablation fits."""
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIT_CODE = (
    "ablation/workflow.py", "ablation/specification.py", "ablation/artifacts.py",
    "models/latent_gate.py", "fusion/trainer.py", "fusion/topology_baseline.py", "topology.py",
    "data_loaders.py", "ddi_sampling.py", "ddi_labels.py", "evaluation_inputs.py",
    "metrics.py", "topology_config.py", "network/graph_variants.py", "fusion/graph_baselines.py",
)


def code_hashes(root=PACKAGE_ROOT):
    """Hash the installed package's fitting code, independent of notebook paths."""
    return {name: file_hash(Path(root) / name) for name in FIT_CODE}


def snapshot(root, destination, hashes):
    """Archive exact fitting code; a fit also rechecks hashes before completion."""
    destination = Path(destination)
    if destination.exists():
        return
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in hashes.items():
            data = (Path(root) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("Fitting code changed during preparation; restart the kernel.")
            archive.writestr(name, data)
    temporary.replace(destination)


def validate_completed(folder, identity):
    path = Path(folder) / "completed.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    for key, value in identity.items():
        if record.get(key) != value:
            raise ValueError(f"Completed fit identity mismatch: {folder}: {key}")
    for name, expected in record.get("artifact_hashes", {}).items():
        artifact = Path(folder) / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise ValueError(f"Completed fit has a missing/changed artifact: {artifact}")
    if not record.get("artifact_hashes"):
        raise ValueError(f"Completed fit has no artifact hashes: {folder}")
    return True
