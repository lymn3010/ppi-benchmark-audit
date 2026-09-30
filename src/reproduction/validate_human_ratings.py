"""Strict, comment-free interchange format for the released human ratings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.reproduction.config import SOURCE_PAPER_DIR as EXPERIMENT_DIR

VERSION = EXPERIMENT_DIR
PACKAGE = VERSION / 'validation'
QUESTIONS = ('relation_asserted', 'trigger_correct', 'carrier_correct', 'endpoints_correct')
FIELDS = ('item_id', 'annotator_id', *QUESTIONS)
ANSWERS = {'是': 'yes', '否': 'no', '不確定': 'uncertain', 'yes': 'yes', 'no': 'no', 'uncertain': 'uncertain'}
RATINGS = PACKAGE / 'validation-ratings-2026-09-12.csv'
ITEMS_PATH = PACKAGE / 'validation-items.csv'
ITEM_FIELDS = ('item_id', 'corpus', 'sentence_id', 'sentence', 'marked_pair',
               'reader_says_trigger', 'reader_says_carriers', 'carrier_class',
               'cell_significant')


def public_items(package: Path = PACKAGE) -> list[dict]:
    """Read the released item table and require its exact public schema."""
    with (package / 'validation-items.csv').open(encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != list(ITEM_FIELDS):
            raise ValueError('Validation items use an unexpected schema')
        rows = list(reader)
    if len({row['item_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate item IDs in validation items')
    return rows


def normalize(rows: list[dict], expected_ids: set[str]) -> list[dict]:
    """Whitelist answer fields; never propagate free text or private identity."""
    result = []
    seen = set()
    for row in rows:
        iid = row['item_id']
        if iid in seen or iid not in expected_ids:
            raise ValueError(f'Unknown or duplicate item ID: {iid}')
        seen.add(iid)
        answers = {}
        for field in QUESTIONS:
            if row.get(field) not in ANSWERS:
                raise ValueError(f'Invalid or missing answer: {iid}/{field}')
            answers[field] = ANSWERS[row[field]]
        result.append({'item_id': iid, 'annotator_id': 'annotator_01', **answers})
    if seen != expected_ids:
        raise ValueError('Rating IDs do not match the fixed sample')
    return sorted(result, key=lambda row: row['item_id'])


def load_public(path: Path = RATINGS, package: Path = PACKAGE) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('Validation ratings use an unexpected schema')
        rows = list(reader)
    ids = {row['item_id'] for row in public_items(package)}
    clean = normalize(rows, ids)
    if rows != clean or any(set(row) != set(FIELDS) for row in rows):
        raise ValueError('Public ratings must use the exact anonymous, comment-free schema')
    return rows


def export_csv(rows: list[dict], path: Path, fields=FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate the released CSV files without writing')
    args = parser.parse_args()
    rows = load_public()
    if not args.check:
        parser.error('the public package is canonical; use --check')
    print(f'Public annotations validated: {len(rows)} rows; four answers; no comments or private identities')


if __name__ == '__main__':
    main()
