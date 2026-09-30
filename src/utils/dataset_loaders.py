"""Load PPI datasets and gold relations."""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import Pipeline


def load_dataset_json(dataset_path: Path) -> Dict:
    """Load a dataset JSON (``{"data": ...}`` or direct); raise ValueError if invalid."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            json_obj = json.load(file)
    except (FileNotFoundError, IOError) as e:
        raise ValueError(f"Could not load JSON from {dataset_path}: {e}")
    
    if not isinstance(json_obj, dict):
        raise ValueError("Dataset JSON must be a dictionary.")
    
    # Handle both formats: {"data": {...}} or direct dict
    data = json_obj.get('data', json_obj)
    
    if not isinstance(data, dict):
        raise ValueError("Dataset must contain dictionary of entries.")
    
    return data


def load_prediction_entries(path: Path | str) -> Dict[str, dict]:
    """Load and preflight masked prediction JSON or JSONL input."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        data: dict[str, dict] = {}
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Prediction row {line_number} must be an object")
                entry_id = str(row.get("id") or "").strip()
                if not entry_id:
                    raise ValueError(f"Prediction row {line_number} requires a unique id")
                if entry_id in data:
                    raise ValueError(f"Duplicate prediction id: {entry_id}")
                data[entry_id] = row
    elif path.suffix.lower() == ".json":
        data = load_dataset_json(path)
    else:
        raise ValueError("Prediction input must be .json or .jsonl")

    validated: dict[str, dict] = {}
    for entry_id, row in data.items():
        if not isinstance(row, dict):
            raise ValueError(f"Prediction entry {entry_id!r} must be an object")
        text = row.get("text") if "text" in row else row.get("sentence")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prediction entry {entry_id!r} requires text or sentence")
        if not re.search(r"\bPROTEIN\d+\b", text):
            raise ValueError(f"Prediction entry {entry_id!r} contains no masked PROTEIN index")
        validated[str(entry_id)] = {
            "text": text,
            "proteins": row.get("proteins") or [],
        }
    return validated


def parse_relation_keys(relations: Dict[str, str]) -> Dict[Tuple[int, int], str]:
    """Convert ``{"[0, 1]": label}`` keys to ``{(0, 1): label}``."""
    parsed_relations = {}
    
    for key, value in relations.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Relation key and value must be strings: {key} → {value}")
        
        try:
            pair = ast.literal_eval(key)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Cannot parse relation key '{key}': {e}")
        
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Relation key must evaluate to a 2-tuple: {pair}")
        if not all(isinstance(i, int) and i >= 0 for i in pair):
            raise ValueError(f"Relation indices must be non-negative integers: {pair}")
        
        parsed_relations[tuple(pair)] = value
    
    return parsed_relations


def _normalise_gold_pair(first: object, second: object) -> Tuple[int, int] | None:
    """Return a sorted valid protein-index pair, preserving index ``0``."""
    if not isinstance(first, int) or isinstance(first, bool):
        return None
    if not isinstance(second, int) or isinstance(second, bool):
        return None
    if first < 0 or second < 0:
        return None
    return tuple(sorted((first, second)))


def _first_present(mapping: Dict, primary: str, alias: str) -> object | None:
    """Select by key presence; truthiness would incorrectly discard index 0."""
    return mapping[primary] if primary in mapping else mapping.get(alias)


def parse_gold_pairs(relations: Union[Dict, List]) -> Set[Tuple[int, int]]:
    """Return sorted gold pairs from dict, list-of-dict or list-of-pair relations."""
    pairs = set()
    
    # Handle dict format ({"[0, 4]": "Negative_Regulation", ...})
    if isinstance(relations, dict):
        for key in relations.keys():
            if isinstance(key, str):
                try:
                    pair = ast.literal_eval(key)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    normalised = _normalise_gold_pair(pair[0], pair[1])
                    if normalised is not None:
                        pairs.add(normalised)
            elif isinstance(key, (list, tuple)) and len(key) >= 2:
                # Key is already a tuple
                normalised = _normalise_gold_pair(key[0], key[1])
                if normalised is not None:
                    pairs.add(normalised)
        return pairs
    
    # Handle list format
    for rel in relations:
        if isinstance(rel, dict):
            p1 = _first_present(rel, 'protein_a', 'arg1')
            p2 = _first_present(rel, 'protein_b', 'arg2')
        elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
            p1, p2 = rel[0], rel[1]
        else:
            continue
            
        normalised = _normalise_gold_pair(p1, p2)
        if normalised is not None:
            pairs.add(normalised)
    
    return pairs


def extract_gold_pairs(relations: Dict[Tuple[int, int], str], 
                       exclude_missing: bool = True) -> List[Tuple[int, int]]:
    """Return sorted gold pairs; optionally skip ``MISSING_LABEL`` relations."""
    gold_pairs = []
    
    for key, value in relations.items():
        # Skip manually added missing labels from experiments
        if exclude_missing and value.startswith("MISSING_LABEL"):
            continue
        gold_pairs.append(tuple(sorted(key)))
    
    return gold_pairs
