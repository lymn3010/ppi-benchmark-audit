"""Dataset paths: data/datasets/{Name}/{split}.json."""
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent / "data" / "datasets"

# 'name': {'split': 'Folder/split.json'}; 'full' combines all splits.

DATASETS = {
    "aimed": {
        "train": "AIMed/train.json",
        "test":  "AIMed/test.json",
        "full":  "AIMed/full.json",
    },
    "bioinfer": {
        "train": "BioInfer/train.json",
        "test":  "BioInfer/test.json",
        "full":  "BioInfer/full.json",
    },
    "hprd50": {
        "train": "HPRD50/train.json",
        "test":  "HPRD50/test.json",
        "full":  "HPRD50/full.json",
    },
    "iepa": {
        "train": "IEPA/train.json",
        "test":  "IEPA/test.json",
        "full":  "IEPA/full.json",
    },
    "lll": {
        "train": "LLL/train.json",
        "test":  "LLL/test.json",
        "full":  "LLL/full.json",
    },

}

DEFAULT_SPLIT = "train"

def get_dataset_path(name: str, split: str | None = None) -> Path:
    """Return the JSON path for dataset *name* and *split*; raise FileNotFoundError if absent."""
    split = split or DEFAULT_SPLIT
    key = name.lower()

    if key not in DATASETS:
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Available: {list(DATASETS)}"
        )
    splits = DATASETS[key]
    if split not in splits:
        raise FileNotFoundError(
            f"Split '{split}' not available for '{name}'. Available: {list(splits)}"
        )
    return BASE_PATH / splits[split]

def list_datasets() -> dict:
    """Return all registered datasets with their available splits."""
    return {name: list(splits) for name, splits in DATASETS.items()}


def print_datasets():
    print("\nAvailable datasets:")
    print("=" * 50)
    for name, splits in sorted(DATASETS.items()):
        print(f"  {name:<12} [{', '.join(splits)}]")
    print("=" * 50)
