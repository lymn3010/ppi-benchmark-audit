"""Explicit experiment sources; never fall back to a latest local run."""
from __future__ import annotations
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name('experiment.json')
MANIFEST = json.loads(MANIFEST_PATH.read_text())
SOURCE_PAPER_DIR = REPO_ROOT / MANIFEST['paper_dir']
PAPER_DIR = Path(os.environ.get('PPI_REPRODUCTION_PAPER', REPO_ROOT / 'output/reproduction/paper'))
CORPORA = MANIFEST['corpora']
CANONICAL_RUN_IDS = MANIFEST['canonical_runs']
SHIPPED_LEDGER_ROOT = SOURCE_PAPER_DIR / 'ledgers'


def _db_paths() -> dict[str, Path]:
    explicit = os.environ.get('PPI_REPRODUCTION_RUNS')
    if explicit:
        paths = json.loads(Path(explicit).read_text())
        if set(paths) != set(CORPORA):
            raise ValueError('Run manifest must identify exactly the five paper corpora')
        return {name: Path(paths[name]) for name in CORPORA}
    root = Path(os.environ.get('PPI_PAPER_OUTPUT', REPO_ROOT / 'output'))
    return {name: root / name / 'full' / run / 'db/events.db' for name, run in CANONICAL_RUN_IDS.items()}


DB_PATHS = _db_paths()


def stats_dir(corpus: str) -> Path:
    if corpus not in CORPORA:
        raise ValueError(f'Unknown paper corpus: {corpus}')
    override = os.environ.get('PPI_PAPER_LEDGER_ROOT')
    path = (Path(override) if override else SHIPPED_LEDGER_ROOT) / corpus
    if not path.is_dir():
        raise FileNotFoundError(f'Missing fixed observation ledgers: {path}')
    return path
