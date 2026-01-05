import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.phase3_modeling import (  # noqa: E402 - internal import
    ATTEMPT_PATH,
    CONSTRAINT_PATH,
    DEFAULT_SEED,
    PlanKey,
    _load_attempts,
    _load_family_scores,
    _sort_attempts,
    _split_plans,
)
from sklearn.metrics import balanced_accuracy_score

OUTPUT_PATH = ROOT_DIR / "data" / "derived" / "phase3_baselines.json"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _macro_average(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def run_phase3_baselines(
    attempt_path: Path = ATTEMPT_PATH,
    constraint_path: Path = CONSTRAINT_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
    min_plans_per_protocol: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Dict[str, Any]:
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing {attempt_path}")

    per_protocol, attempt_index_map, _feature_map = _load_attempts(
        attempt_path, verbose=False
    )
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]] = {}
    if constraint_path.exists():
        family_stats = _load_family_scores(
            constraint_path, attempt_index_map, verbose=False
        )

    task1_records = []
    task2_records = []
    task3_records = []

    for protocol, plans in per_protocol.items():
        plan_keys = list(plans.keys())
        if len(plan_keys) < min_plans_per_protocol:
            continue

        splits = _split_plans(plan_keys, seed, train_ratio, val_ratio)

        # Task 1 baseline: mean_percentile increases across adjacent attempts.
        total_pairs = 0
        correct_pairs = 0
        for plan_key in splits["test"]:
            attempts = plans.get(plan_key, [])
            attempts_sorted = _sort_attempts(attempts)
            for idx in range(len(attempts_sorted) - 1):
                current = attempts_sorted[idx]
                nxt = attempts_sorted[idx + 1]
                current_val = _safe_float(current.get("mean_percentile"))
                next_val = _safe_float(nxt.get("mean_percentile"))
                if current_val is None or next_val is None:
                    continue
                delta = next_val - current_val
                total_pairs += 1
                if delta > 0:
                    correct_pairs += 1
        task1_accuracy = (
            correct_pairs / total_pairs if total_pairs > 0 else None
        )
        task1_records.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "pair_count": total_pairs,
                "metrics": {"accuracy": task1_accuracy},
            }
        )

        # Task 2 baseline: predict mean remaining steps from training plans.
        train_labels = []
        test_labels = []
        for plan_key, attempts in plans.items():
            attempts_sorted = _sort_attempts(attempts)
            total = len(attempts_sorted)
            if total == 0:
                continue
            for idx, attempt in enumerate(attempts_sorted, start=1):
                attempt_index = attempt.get("attempt_index") or idx
                attempt_count = attempt.get("attempt_count") or total
                label = max(float(attempt_count - attempt_index), 0.0)
                if plan_key in splits["train"]:
                    train_labels.append(label)
                elif plan_key in splits["test"]:
                    test_labels.append(label)
        task2_mae = None
        if train_labels and test_labels:
            baseline = sum(train_labels) / len(train_labels)
            task2_mae = sum(abs(label - baseline) for label in test_labels) / len(test_labels)
        task2_records.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "attempt_count": len(test_labels),
                "metrics": {"mae": task2_mae},
            }
        )

        # Task 3 baseline: top-k most common families from training.
        train_labels = []
        test_labels = []
        for plan_key, attempts in plans.items():
            attempts_sorted = _sort_attempts(attempts)
            if len(attempts_sorted) < 2:
                continue
            for idx in range(len(attempts_sorted) - 1):
                current = attempts_sorted[idx]
                nxt = attempts_sorted[idx + 1]
                cur_key = (protocol, plan_key.patient_id, plan_key.plan_id, current["attempt_index"])
                next_key = (protocol, plan_key.patient_id, plan_key.plan_id, nxt["attempt_index"])
                cur_families = family_stats.get(cur_key, {}).get("mean", {})
                next_families = family_stats.get(next_key, {}).get("mean", {})
                if not cur_families or not next_families:
                    continue
                best_family = None
                best_delta = None
                for family in set(cur_families) & set(next_families):
                    delta = next_families[family] - cur_families[family]
                    if best_delta is None or delta > best_delta:
                        best_delta = delta
                        best_family = family
                if best_family is None:
                    continue
                if plan_key in splits["train"]:
                    train_labels.append(best_family)
                elif plan_key in splits["test"]:
                    test_labels.append(best_family)

        task3_accuracy = None
        task3_balanced = None
        task3_top3 = None
        task3_top5 = None
        task3_topk = {}
        if train_labels and test_labels:
            label_counts = Counter(train_labels)
            ranked_labels = [label for label, _count in label_counts.most_common()]
            mode_label = ranked_labels[0]
            correct = sum(1 for label in test_labels if label == mode_label)
            task3_accuracy = correct / len(test_labels) if test_labels else None
            try:
                task3_balanced = float(
                    balanced_accuracy_score(
                        test_labels, [mode_label for _ in test_labels]
                    )
                )
            except ValueError:
                task3_balanced = None
            for k in (1, 2, 3, 5):
                top_labels = set(ranked_labels[:k])
                hits = sum(1 for label in test_labels if label in top_labels)
                task3_topk[k] = hits / len(test_labels) if test_labels else None
            task3_top3 = task3_topk.get(3)
            task3_top5 = task3_topk.get(5)
        task3_records.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "pair_count": len(test_labels),
                "metrics": {
                    "accuracy": task3_accuracy,
                    "balanced_accuracy": task3_balanced,
                    "top3_accuracy": task3_top3,
                    "top5_accuracy": task3_top5,
                    "topk_accuracy": task3_topk,
                },
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "min_plans_per_protocol": min_plans_per_protocol,
        "splits": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1 - train_ratio - val_ratio, 3),
            "seed": seed,
        },
        "task1": {
            "macro": {"accuracy": _macro_average(r["metrics"]["accuracy"] for r in task1_records)},
            "protocols": task1_records,
        },
        "task2": {
            "macro": {"mae": _macro_average(r["metrics"]["mae"] for r in task2_records)},
            "protocols": task2_records,
        },
        "task3": {
            "macro": {
                "accuracy": _macro_average(r["metrics"]["accuracy"] for r in task3_records),
                "balanced_accuracy": _macro_average(
                    r["metrics"]["balanced_accuracy"] for r in task3_records
                ),
                "top3_accuracy": _macro_average(
                    r["metrics"]["top3_accuracy"] for r in task3_records
                ),
                "top5_accuracy": _macro_average(
                    r["metrics"]["top5_accuracy"] for r in task3_records
                ),
            },
            "protocols": task3_records,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Phase 3 baselines written to {output_path}")
    print("Task 1 baseline accuracy:", payload["task1"]["macro"]["accuracy"])
    print("Task 2 baseline MAE:", payload["task2"]["macro"]["mae"])
    print("Task 3 baseline accuracy:", payload["task3"]["macro"]["accuracy"])
    print("Task 3 baseline balanced accuracy:", payload["task3"]["macro"]["balanced_accuracy"])
    print("Task 3 baseline top-3 accuracy:", payload["task3"]["macro"]["top3_accuracy"])
    print("Task 3 baseline top-5 accuracy:", payload["task3"]["macro"]["top5_accuracy"])

    return payload


if __name__ == "__main__":
    run_phase3_baselines()
