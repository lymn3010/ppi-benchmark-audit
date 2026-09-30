"""Check that the paper's promised source materials are present."""
from pathlib import Path

from .config import MANIFEST, REPO_ROOT


def check_materials(root: Path = REPO_ROOT) -> None:
    missing = [name for name in MANIFEST['required_materials'] if not (root / name).is_file()]
    if missing:
        raise RuntimeError('Missing paper materials:\n' + '\n'.join(missing))
