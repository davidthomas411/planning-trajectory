import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.phase3_modeling import (  # noqa: E402
    ATTEMPT_PATH,
    CONSTRAINT_PATH,
    DEFAULT_SEED,
    PlanKey,
    _load_attempts,
    _load_family_scores,
    _sort_attempts,
)

OUTPUT_PATH = ROOT_DIR / "data" / "derived" / "phase3_analysis.json"


def _entropy(counts: Dict[str, int]) -> Optional[float]:
    total = sum(counts.values())
    if total == 0:
        return None
    entropy = 0.0
    for value in counts.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log(p, 2)
    return entropy


def _label_from_pair(
    protocol: str,
    plan_key: PlanKey,
    current_attempt: Dict[str, Any],
    next_attempt: Dict[str, Any],
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]],
) -> Optional[str]:
    cur_key = (protocol, plan_key.patient_id, plan_key.plan_id, current_attempt["attempt_index"])
    next_key = (protocol, plan_key.patient_id, plan_key.plan_id, next_attempt["attempt_index"])
    cur_families = family_stats.get(cur_key, {}).get("mean", {})
    next_families = family_stats.get(next_key, {}).get("mean", {})
    if not cur_families or not next_families:
        return None
    best_family = None
    best_delta = None
    for family in set(cur_families) & set(next_families):
        delta = next_families[family] - cur_families[family]
        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_family = family
    return best_family


def run_phase3_analysis(
    attempt_path: Path = ATTEMPT_PATH,
    constraint_path: Path = CONSTRAINT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing {attempt_path}")

    per_protocol, attempt_index_map, _ = _load_attempts(
        attempt_path, verbose=False
    )
    family_stats = {}
    if constraint_path.exists():
        family_stats = _load_family_scores(
            constraint_path, attempt_index_map, verbose=False
        )

    protocol_records: List[Dict[str, Any]] = []
    for protocol, plans in per_protocol.items():
        label_counts: Counter = Counter()
        pair_count = 0
        for plan_key, attempts in plans.items():
            attempts_sorted = _sort_attempts(attempts)
            if len(attempts_sorted) < 2:
                continue
            for idx in range(len(attempts_sorted) - 1):
                label = _label_from_pair(
                    protocol,
                    plan_key,
                    attempts_sorted[idx],
                    attempts_sorted[idx + 1],
                    family_stats,
                )
                if label is None:
                    continue
                label_counts[label] += 1
                pair_count += 1

        if pair_count == 0:
            continue

        top_label, top_count = label_counts.most_common(1)[0]
        top_share = top_count / pair_count
        protocol_records.append(
            {
                "protocol_name": protocol,
                "pair_count": pair_count,
                "family_count": len(label_counts),
                "top_family": top_label,
                "top_share": top_share,
                "entropy": _entropy(label_counts),
                "label_counts": dict(label_counts),
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "protocols": protocol_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Phase 3 analysis written to {output_path}")
    return payload


if __name__ == "__main__":
    run_phase3_analysis()
