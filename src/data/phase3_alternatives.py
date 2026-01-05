import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.phase3_modeling import (  # noqa: E402
    ATTEMPT_PATH,
    CONSTRAINT_PATH,
    DEFAULT_SEED,
    FEATURE_FIELDS,
    PlanKey,
    _build_feature_map,
    _load_attempts,
    _load_family_scores,
    _sort_attempts,
    _split_plans,
)

OUTPUT_PATH = ROOT_DIR / "data" / "derived" / "phase3_alternatives.json"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _macro(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _rankdata(values: List[float]) -> List[float]:
    sorted_idx = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = rank
        i = j + 1
    return ranks


def _spearman(scores: List[float], targets: List[int]) -> Optional[float]:
    if len(scores) < 2:
        return None
    score_ranks = _rankdata(scores)
    target_ranks = _rankdata([float(value) for value in targets])
    n = len(scores)
    diff_sq = sum((score_ranks[i] - target_ranks[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * diff_sq) / (n * (n**2 - 1))


def _remaining_steps(attempt: Dict[str, Any]) -> int:
    count = attempt.get("attempt_count") or 0
    index = attempt.get("attempt_index") or 0
    return max(int(count) - int(index), 0)


def _bin_remaining_steps(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value <= 3:
        return 2
    return 3


def _collect_attempts(
    plans: Dict[PlanKey, List[Dict[str, Any]]],
    plan_subset: set,
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    protocol: str,
    label_getter,
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for plan_key in plan_subset:
        attempts = plans.get(plan_key, [])
        for attempt in _sort_attempts(attempts):
            key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
            vec = feature_map.get(key)
            if vec is None:
                continue
            label = label_getter(attempt)
            if label is None:
                continue
            features.append(vec)
            labels.append(label)
    if not features:
        return np.empty((0, len(FEATURE_FIELDS))), np.array([])
    return np.array(features, dtype=float), np.array(labels)


def _next_focus_pairs(
    plans: Dict[PlanKey, List[Dict[str, Any]]],
    plan_subset: set,
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]],
    protocol: str,
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for plan_key in plan_subset:
        attempts = _sort_attempts(plans.get(plan_key, []))
        if len(attempts) < 2:
            continue
        for idx in range(len(attempts) - 1):
            current = attempts[idx]
            nxt = attempts[idx + 1]
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
            vec = feature_map.get(cur_key)
            if vec is None:
                continue
            features.append(vec)
            labels.append(best_family)
    if not features:
        return np.empty((0, len(FEATURE_FIELDS))), np.array([])
    return np.array(features, dtype=float), np.array(labels)


def _stop_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
    metrics = {"accuracy": None, "balanced_accuracy": None, "auc": None}
    if len(y_true) == 0:
        return metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        metrics["balanced_accuracy"] = None
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    if len(y_true) == 0:
        return metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        metrics["balanced_accuracy"] = None
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    return metrics


def _next_focus_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"accuracy": None, "top3": None, "top5": None, "mrr": None}
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    hits1 = hits3 = hits5 = 0
    rr_total = 0.0
    for idx, label in enumerate(y_true):
        if label not in class_to_idx:
            continue
        label_idx = class_to_idx[label]
        order = np.argsort(probs[idx])[::-1]
        rank = int(np.where(order == label_idx)[0][0]) + 1
        if rank == 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        if rank <= 5:
            hits5 += 1
        rr_total += 1.0 / rank
    total = len(y_true)
    return {
        "accuracy": hits1 / total,
        "top3": hits3 / total,
        "top5": hits5 / total,
        "mrr": rr_total / total,
    }


def _baseline_topk(labels: List[str], k: int) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    ranked = [label for label, _count in counts.most_common()]
    top = set(ranked[:k])
    hits = sum(1 for label in labels if label in top)
    return hits / len(labels)


def run_phase3_alternatives(
    attempt_path: Path = ATTEMPT_PATH,
    constraint_path: Path = CONSTRAINT_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
    min_plans_per_protocol: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    include_family_features: bool = False,
) -> Dict[str, Any]:
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing {attempt_path}")

    per_protocol, attempt_index_map, base_feature_map = _load_attempts(
        attempt_path, verbose=False
    )
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]] = {}
    if constraint_path.exists():
        family_stats = _load_family_scores(
            constraint_path, attempt_index_map, verbose=False
        )
    feature_map, feature_names = _build_feature_map(
        base_feature_map, family_stats, include_family_features
    )

    eligible_protocols = {
        protocol: plans
        for protocol, plans in per_protocol.items()
        if len(plans) >= min_plans_per_protocol
    }

    stop_protocols = []
    ordinal_protocols = []
    rank_protocols = []
    focus_protocols = []

    for protocol, plans in eligible_protocols.items():
        plan_keys = list(plans.keys())
        splits = _split_plans(plan_keys, seed, train_ratio, val_ratio)

        # Stop / Continue
        X_train, y_train = _collect_attempts(
            plans, splits["train"], feature_map, protocol, lambda attempt: int(bool(attempt.get("label_stop")))
        )
        X_test, y_test = _collect_attempts(
            plans, splits["test"], feature_map, protocol, lambda attempt: int(bool(attempt.get("label_stop")))
        )
        stop_metrics = {"accuracy": None, "balanced_accuracy": None, "auc": None}
        stop_baseline = {"accuracy": None, "balanced_accuracy": None, "auc": None}
        if len(y_train) > 0 and len(y_test) > 0:
            mode = Counter(y_train).most_common(1)[0][0]
            base_pred = np.full_like(y_test, mode)
            stop_baseline = _stop_metrics(y_test, base_pred, None)
            if len(set(y_train)) > 1:
                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                    ]
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if len(set(y_train)) > 1 else None
                stop_metrics = _stop_metrics(y_test, y_pred, y_prob)

        stop_protocols.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "attempt_count": len(y_test),
                "metrics": stop_metrics,
                "baseline": stop_baseline,
            }
        )

        # Ordinal progression (remaining steps bins)
        X_train, y_train = _collect_attempts(
            plans, splits["train"], feature_map, protocol, lambda attempt: _bin_remaining_steps(_remaining_steps(attempt))
        )
        X_test, y_test = _collect_attempts(
            plans, splits["test"], feature_map, protocol, lambda attempt: _bin_remaining_steps(_remaining_steps(attempt))
        )
        ordinal_metrics = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None}
        ordinal_baseline = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None}
        if len(y_train) > 0 and len(y_test) > 0:
            mode = Counter(y_train).most_common(1)[0][0]
            base_pred = np.full_like(y_test, mode)
            ordinal_baseline = _ordinal_metrics(y_test, base_pred)
            if len(set(y_train)) > 1:
                clf = GradientBoostingClassifier(random_state=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                ordinal_metrics = _ordinal_metrics(y_test, y_pred)

        ordinal_protocols.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "attempt_count": len(y_test),
                "metrics": ordinal_metrics,
                "baseline": ordinal_baseline,
            }
        )

        # Ranking correlation
        X_train_pairs = []
        y_train_pairs = []
        for plan_key in splits["train"]:
            attempts = _sort_attempts(plans.get(plan_key, []))
            for idx in range(len(attempts) - 1):
                a = attempts[idx]
                b = attempts[idx + 1]
                key_a = (protocol, plan_key.patient_id, plan_key.plan_id, a["attempt_index"])
                key_b = (protocol, plan_key.patient_id, plan_key.plan_id, b["attempt_index"])
                vec_a = feature_map.get(key_a)
                vec_b = feature_map.get(key_b)
                if vec_a is None or vec_b is None:
                    continue
                diff = np.array(vec_b) - np.array(vec_a)
                X_train_pairs.append(diff)
                y_train_pairs.append(1)
                X_train_pairs.append(-diff)
                y_train_pairs.append(0)

        rank_metrics = {"spearman": None}
        rank_baseline = {"spearman": None}
        if X_train_pairs:
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            )
            clf.fit(np.array(X_train_pairs), np.array(y_train_pairs))
            spearman_scores = []
            baseline_scores = []
            for plan_key in splits["test"]:
                attempts = _sort_attempts(plans.get(plan_key, []))
                if len(attempts) < 2:
                    continue
                scores = []
                baseline_vals = []
                order = []
                for attempt in attempts:
                    key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
                    vec = feature_map.get(key)
                    if vec is None:
                        continue
                    score = clf.decision_function([vec])[0]
                    scores.append(score)
                    baseline_vals.append(_safe_float(attempt.get("mean_percentile")) or 0.0)
                    order.append(int(attempt["attempt_index"]))
                if len(scores) < 2:
                    continue
                spearman = _spearman(scores, order)
                baseline = _spearman(baseline_vals, order)
                if spearman is not None:
                    spearman_scores.append(spearman)
                if baseline is not None:
                    baseline_scores.append(baseline)
            if spearman_scores:
                rank_metrics["spearman"] = float(sum(spearman_scores) / len(spearman_scores))
            if baseline_scores:
                rank_baseline["spearman"] = float(sum(baseline_scores) / len(baseline_scores))

        rank_protocols.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "metrics": rank_metrics,
                "baseline": rank_baseline,
            }
        )

        # Next-focus recommender
        X_train, y_train = _next_focus_pairs(
            plans, splits["train"], feature_map, family_stats, protocol
        )
        X_test, y_test = _next_focus_pairs(
            plans, splits["test"], feature_map, family_stats, protocol
        )
        focus_metrics = {"accuracy": None, "top3": None, "top5": None, "mrr": None}
        focus_baseline = {"accuracy": None, "top3": None, "top5": None, "mrr": None}
        if len(y_train) > 0 and len(y_test) > 0 and len(set(y_train)) > 1:
            clf = GradientBoostingClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            focus_metrics = _next_focus_metrics(y_test, probs, list(clf.classes_))

            focus_baseline = {
                "accuracy": _baseline_topk(list(y_test), 1),
                "top3": _baseline_topk(list(y_test), 3),
                "top5": _baseline_topk(list(y_test), 5),
                "mrr": None,
            }
            ranked = [label for label, _count in Counter(y_train).most_common()]
            if ranked:
                rr = []
                for label in y_test:
                    if label in ranked:
                        rr.append(1.0 / (ranked.index(label) + 1))
                focus_baseline["mrr"] = sum(rr) / len(rr) if rr else None

        focus_protocols.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "pair_count": len(y_test),
                "metrics": focus_metrics,
                "baseline": focus_baseline,
            }
        )

    def macro_from(records: List[Dict[str, Any]], key: str) -> Optional[float]:
        return _macro(record["metrics"].get(key) for record in records)

    def baseline_macro(records: List[Dict[str, Any]], key: str) -> Optional[float]:
        return _macro(record["baseline"].get(key) for record in records)

    # Global models across all eligible protocols.
    global_plans = []
    for protocol, plans in eligible_protocols.items():
        global_plans.extend([(protocol, plan_key) for plan_key in plans.keys()])
    global_plan_keys = [PlanKey(item[1].patient_id, item[1].plan_id) for item in global_plans]
    global_splits = _split_plans(global_plan_keys, seed, train_ratio, val_ratio)
    global_protocol_map = {plan_key: protocol for protocol, plan_key in global_plans}

    def global_subset(split_keys: set) -> List[Tuple[str, PlanKey]]:
        return [(global_protocol_map[key], key) for key in split_keys if key in global_protocol_map]

    # Build global stop/continue.
    global_stop_metrics = {"accuracy": None, "balanced_accuracy": None, "auc": None}
    global_stop_baseline = {"accuracy": None, "balanced_accuracy": None, "auc": None}
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for protocol, plan_key in global_subset(global_splits["train"]):
        for attempt in _sort_attempts(eligible_protocols[protocol].get(plan_key, [])):
            key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
            vec = feature_map.get(key)
            if vec is None:
                continue
            X_train.append(vec)
            y_train.append(int(bool(attempt.get("label_stop"))))
    for protocol, plan_key in global_subset(global_splits["test"]):
        for attempt in _sort_attempts(eligible_protocols[protocol].get(plan_key, [])):
            key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
            vec = feature_map.get(key)
            if vec is None:
                continue
            X_test.append(vec)
            y_test.append(int(bool(attempt.get("label_stop"))))
    if X_train and X_test:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        mode = Counter(y_train).most_common(1)[0][0]
        base_pred = np.array([mode for _ in y_test])
        global_stop_baseline = _stop_metrics(np.array(y_test), base_pred, None)
        if len(set(y_train)) > 1:
            clf.fit(np.array(X_train), np.array(y_train))
            y_pred = clf.predict(np.array(X_test))
            y_prob = (
                clf.predict_proba(np.array(X_test))[:, 1] if len(set(y_train)) > 1 else None
            )
            global_stop_metrics = _stop_metrics(np.array(y_test), y_pred, y_prob)

    # Global ordinal.
    global_ord_metrics = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    global_ord_baseline = {"accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for protocol, plan_key in global_subset(global_splits["train"]):
        for attempt in _sort_attempts(eligible_protocols[protocol].get(plan_key, [])):
            key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
            vec = feature_map.get(key)
            if vec is None:
                continue
            X_train.append(vec)
            y_train.append(_bin_remaining_steps(_remaining_steps(attempt)))
    for protocol, plan_key in global_subset(global_splits["test"]):
        for attempt in _sort_attempts(eligible_protocols[protocol].get(plan_key, [])):
            key = (protocol, plan_key.patient_id, plan_key.plan_id, attempt["attempt_index"])
            vec = feature_map.get(key)
            if vec is None:
                continue
            X_test.append(vec)
            y_test.append(_bin_remaining_steps(_remaining_steps(attempt)))
    if X_train and X_test:
        mode = Counter(y_train).most_common(1)[0][0]
        base_pred = np.array([mode for _ in y_test])
        global_ord_baseline = _ordinal_metrics(np.array(y_test), base_pred)
        if len(set(y_train)) > 1:
            clf = GradientBoostingClassifier(random_state=seed)
            clf.fit(np.array(X_train), np.array(y_train))
            y_pred = clf.predict(np.array(X_test))
            global_ord_metrics = _ordinal_metrics(np.array(y_test), y_pred)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "settings": {
            "min_plans_per_protocol": min_plans_per_protocol,
            "seed": seed,
            "split": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
            "include_family_features": include_family_features,
            "ordinal_bins": ["0=final", "1=1 step", "2=2-3 steps", "3=4+ steps"],
        },
        "stop_continue": {
            "macro": {
                "accuracy": macro_from(stop_protocols, "accuracy"),
                "balanced_accuracy": macro_from(stop_protocols, "balanced_accuracy"),
                "auc": macro_from(stop_protocols, "auc"),
            },
            "baseline": {
                "accuracy": baseline_macro(stop_protocols, "accuracy"),
                "balanced_accuracy": baseline_macro(stop_protocols, "balanced_accuracy"),
            },
            "global": {"metrics": global_stop_metrics, "baseline": global_stop_baseline},
            "protocols": stop_protocols,
        },
        "ordinal_progress": {
            "macro": {
                "accuracy": macro_from(ordinal_protocols, "accuracy"),
                "balanced_accuracy": macro_from(ordinal_protocols, "balanced_accuracy"),
                "macro_f1": macro_from(ordinal_protocols, "macro_f1"),
            },
            "baseline": {
                "accuracy": baseline_macro(ordinal_protocols, "accuracy"),
                "balanced_accuracy": baseline_macro(ordinal_protocols, "balanced_accuracy"),
                "macro_f1": baseline_macro(ordinal_protocols, "macro_f1"),
            },
            "global": {"metrics": global_ord_metrics, "baseline": global_ord_baseline},
            "protocols": ordinal_protocols,
        },
        "ranking_correlation": {
            "macro": {"spearman": macro_from(rank_protocols, "spearman")},
            "baseline": {"spearman": baseline_macro(rank_protocols, "spearman")},
            "protocols": rank_protocols,
        },
        "next_focus_recommender": {
            "macro": {
                "accuracy": macro_from(focus_protocols, "accuracy"),
                "top3": macro_from(focus_protocols, "top3"),
                "top5": macro_from(focus_protocols, "top5"),
                "mrr": macro_from(focus_protocols, "mrr"),
            },
            "baseline": {
                "accuracy": baseline_macro(focus_protocols, "accuracy"),
                "top3": baseline_macro(focus_protocols, "top3"),
                "top5": baseline_macro(focus_protocols, "top5"),
                "mrr": baseline_macro(focus_protocols, "mrr"),
            },
            "protocols": focus_protocols,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Phase 3 alternatives written to {output_path}")
    return payload


if __name__ == "__main__":
    run_phase3_alternatives()
