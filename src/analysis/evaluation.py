from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models import Record


def _canon(pairs) -> set:
    """Canonicalize pairs to sorted tuples so (i, j) and (j, i) map to the same key."""
    return {tuple(sorted(p)) for p in pairs}


def compute_metrics(results: list["Record"]) -> dict[str, Any]:
    """Compute sentence-level and pair-level evaluation metrics."""

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    pred_labels: list[int] = []
    gold_labels: list[int] = []
    for record in results:
        pred_set = _canon(record.predicted_pairs)
        gold_set = _canon(record.gold_pairs) if record.gold_pairs is not None else set()
        for pair in combinations(range(record.target_count), 2):
            pred_labels.append(1 if pair in pred_set else 0)
            gold_labels.append(1 if pair in gold_set else 0)

    if not pred_labels:
        cm_dict = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        precision = recall = f1 = accuracy = 0.0
    else:
        cm = confusion_matrix(gold_labels, pred_labels, labels=[0, 1])
        cm_dict = {
            "tp": int(cm[1][1]),
            "fp": int(cm[0][1]),
            "tn": int(cm[0][0]),
            "fn": int(cm[1][0]),
        }
        precision = float(precision_score(gold_labels, pred_labels, zero_division=0))
        recall = float(recall_score(gold_labels, pred_labels, zero_division=0))
        f1 = float(f1_score(gold_labels, pred_labels, zero_division=0))
        accuracy = float(accuracy_score(gold_labels, pred_labels))

    correct = too_many = too_few = wrong = 0
    for record in results:
        if record.gold_pairs is None:
            continue

        pred_set = _canon(record.predicted_pairs)
        gold_set = _canon(record.gold_pairs)
        if pred_set == gold_set:
            correct += 1
        elif len(pred_set) > len(gold_set):
            too_many += 1
        elif len(pred_set) < len(gold_set):
            too_few += 1
        else:
            wrong += 1

    tp_by_type: dict[str, int] = defaultdict(int)
    fp_by_type: dict[str, int] = defaultdict(int)
    for record in results:
        if record.gold_pairs is None or not record.events:
            continue

        gold_set = _canon(record.gold_pairs)
        for event in record.events:
            event_type = event.extraction_detail or type(event).__name__
            for pair in event.pred_relations or []:
                target = tp_by_type if tuple(sorted(pair)) in gold_set else fp_by_type
                target[event_type] += 1

    event_type_performance: dict[str, dict[str, int | float]] = {}
    for event_type in sorted(set(tp_by_type) | set(fp_by_type)):
        tp = tp_by_type[event_type]
        fp = fp_by_type[event_type]
        total = tp + fp
        event_type_performance[event_type] = {
            "tp": tp,
            "fp": fp,
            "total": total,
            "precision": round(tp / total, 4) if total else 0.0,
        }

    return {
        "confusion_matrix": cm_dict,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total_sentences": len(results),
        "predicted_pairs": sum(len(record.predicted_pairs) for record in results),
        "gold_pairs": sum(len(record.gold_pairs) for record in results if record.gold_pairs),
        "error_summary": {
            "correct": correct,
            "too_many": too_many,
            "too_few": too_few,
            "wrong_pairs": wrong,
        },
        "event_type_performance": event_type_performance,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print a compact metrics summary to the terminal."""

    confusion = metrics["confusion_matrix"]
    errors = metrics["error_summary"]

    print(f"\n{'=' * 60}")
    print(
        f"  Precision: {metrics['precision']:.4f}    "
        f"Recall: {metrics['recall']:.4f}    "
        f"F1: {metrics['f1']:.4f}"
    )
    print(
        f"  TP={confusion['tp']}  FP={confusion['fp']}  "
        f"TN={confusion['tn']}  FN={confusion['fn']}"
    )
    print(
        f"  Correct={errors['correct']}  TooMany={errors['too_many']}  "
        f"TooFew={errors['too_few']}  Wrong={errors['wrong_pairs']}"
    )

    if metrics["event_type_performance"]:
        print(f"\n  {'Event Type':<35} {'TP':>5} {'FP':>5} {'Prec':>7}")
        print(f"  {'-' * 55}")
        for event_type, data in sorted(
            metrics["event_type_performance"].items(),
            key=lambda item: item[1]["total"],
            reverse=True,
        ):
            print(
                f"  {event_type:<35} "
                f"{data['tp']:>5} {data['fp']:>5} {data['precision']:>7.3f}"
            )

    print(f"{'=' * 60}\n")
