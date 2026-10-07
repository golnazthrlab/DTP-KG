"""Fallback locations for CLI/library calls that omit explicit paths.

Experiment notebooks define their own editable paths and pass them directly to
the library. Environment overrides here remain available to command-line users.
"""
import os
from pathlib import Path


def find_project_root():
    """Locate this checkout from the working directory or package path.

    A standalone installed package can use DTP_KG_PROJECT_ROOT to select a
    checkout. If no checkout is available, paths are relative to the working
    directory; explicit data/network environment variables still take precedence.
    """
    configured = os.environ.get("DTP_KG_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents, *Path(__file__).resolve().parents):
        if ((candidate / "experiments").is_dir()
                and (candidate / "src" / "dtpkg" / "__init__.py").is_file()):
            return candidate
    return cwd


PROJECT_ROOT = find_project_root()
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DATA_DIR = Path(os.environ.get("DTP_KG_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
NETWORK_DIR = Path(os.environ.get("DTP_KG_NETWORK_DIR", str(DATA_DIR / "networks"))).expanduser().resolve()
