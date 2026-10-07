"""Archive the installed study source when launching a new private run."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import sys
import uuid
import zipfile


def snapshot_code(package_root, results_root, label=None):
    """Write a unique source archive and manifest beneath the run directory.

    ``package_root`` is the installed ``dtpkg`` directory. Data, notebooks,
    cached bytecode, and model artifacts are never collected.
    """
    package_root, results_root = Path(package_root), Path(results_root)
    if not (package_root / "__init__.py").is_file():
        raise ValueError("package_root must point to the installed dtpkg package")
    if label is not None and (not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label)):
        raise ValueError("snapshot label may contain only letters, digits, underscore and hyphen")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{uuid.uuid4().hex[:12]}" + (f"_{label}" if label else "")
    folder = results_root / "code_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{stem}.zip"
    files = {}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(package_root.rglob("*.py")):
            if path.is_symlink():
                raise ValueError("source snapshot may not follow symlinks")
            name = f"dtpkg/{path.relative_to(package_root).as_posix()}"
            stream.write(path, arcname=name)
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    import torch
    manifest = dict(created_utc=stamp, label=label, archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
        files=files, missing=[])
    (folder / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
