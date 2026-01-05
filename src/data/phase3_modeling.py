import json
import math
import random
import re
import time
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT_DIR / "data" / "derived"
ATTEMPT_PATH = PHASE2_DIR / "plan_attempt_features.jsonl"
CONSTRAINT_PATH = PHASE2_DIR / "constraint_features.jsonl"
OUTPUT_PATH = PHASE2_DIR / "phase3_metrics.json"
SWEEP_PATH = PHASE2_DIR / "phase3_sweep.json"

DEFAULT_SEED = 42

FEATURE_FIELDS = [
    "constraints_total",
    "constraints_matched",
    "coverage_pct",
    "constraints_pass",
    "constraints_fail",
    "constraints_unknown",
    "near_limit_count",
    "worst_margin",
    "worst_normalized_margin",
    "min_percentile",
    "p10_percentile",
    "p50_percentile",
    "p90_percentile",
    "mean_percentile",
    "pass_rate",
    "fail_rate",
    "unknown_rate",
    "near_limit_rate",
]

EXCLUDED_FIELDS = [
    "plan_score",
    "attempt_index",
    "attempt_count",
    "attempt_progress",
    "created_at",
    "patient_id",
    "plan_id",
    "protocol_name",
]

FAMILY_PATTERNS = OrderedDict(
    [
        ("Target", [r"ptv", r"gtv", r"ctv", r"itv", r"boost", r"ring", r"eval"]),
        ("Lung", [r"lung", r"pulm"]),
        ("Heart", [r"heart", r"cardiac", r"pericard", r"atrium", r"ventricle", r"greatves", r"aorta", r"vessel"]),
        ("Cord", [r"spinal", r"cord", r"thecal", r"cauda"]),
        ("Brainstem", [r"brainstem", r"medulla", r"pons", r"midbrain"]),
        ("Brain", [r"brain", r"cerebell", r"temporal", r"frontal", r"parietal", r"occipital"]),
        ("Optic", [r"optic", r"chiasm", r"retina", r"lens", r"eye", r"lacrimal", r"orbit"]),
        ("Cochlea", [r"cochlea"]),
        ("Salivary", [r"parotid", r"submand", r"subling", r"saliv"]),
        ("OralCavity", [r"oral", r"cavity", r"tongue", r"lips", r"floor"]),
        ("Esophagus", [r"esoph", r"oesoph"]),
        ("Bronchus", [r"bronch", r"airway", r"trachea"]),
        ("ChestWall", [r"chest wall", r"chestwall", r"rib", r"infram"]),
        ("Breast", [r"breast", r"mamm"]),
        ("Bowel", [r"bowel", r"duod", r"jejun", r"ileum", r"colon", r"stomach", r"stom"]),
        ("Bladder", [r"bladder"]),
        ("Rectum", [r"rectum", r"rectal"]),
        ("Prostate", [r"prostate"]),
        ("Genitourinary", [r"penile", r"genital", r"test", r"ovary", r"uterus", r"vagina", r"urethra"]),
        ("Kidney", [r"kidney", r"renal"]),
        ("Liver", [r"liver"]),
        ("Bone", [r"mandible", r"femur", r"pelvis", r"sacrum", r"iliac", r"hip", r"bone"]),
        ("BrachialPlex", [r"brachial", r"plex"]),
        ("Skin", [r"skin", r"gluteal", r"cleft"]),
        ("LymphNodes", [r"\\bln\\b", r"lymph", r"sclav", r"ax", r"imn"]),
        ("Thyroid", [r"thyroid"]),
    ]
)

FAMILY_ORDER = list(FAMILY_PATTERNS.keys())
_STRUCTURE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
FAMILY_REGEX = OrderedDict(
    (family, [re.compile(pattern) for pattern in patterns])
    for family, patterns in FAMILY_PATTERNS.items()
)


@dataclass(frozen=True)
class PlanKey:
    patient_id: Any
    plan_id: Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_structure_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _STRUCTURE_NORMALIZE_RE.sub(" ", text)
    return text.strip()


def _sort_attempts(attempts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    attempts = list(attempts)
    if not attempts:
        return []
    if all(attempt.get("attempt_index") is not None for attempt in attempts):
        return sorted(attempts, key=lambda attempt: attempt["attempt_index"])
    if all(attempt.get("attempt_number") is not None for attempt in attempts):
        return sorted(attempts, key=lambda attempt: attempt["attempt_number"])
    if all(attempt.get("created_at") is not None for attempt in attempts):
        return sorted(attempts, key=lambda attempt: attempt["created_at"])
    return attempts


def _extract_features(record: Dict[str, Any], fields: List[str]) -> List[float]:
    values: List[float] = []
    for field in fields:
        numeric = _safe_float(record.get(field))
        values.append(numeric if numeric is not None else 0.0)
    return values


def _split_plans(
    plan_keys: List[PlanKey],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, set]:
    rng = random.Random(seed)
    shuffled = list(plan_keys)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)
    return {
        "train": set(shuffled[:train_cut]),
        "val": set(shuffled[train_cut:val_cut]),
        "test": set(shuffled[val_cut:]),
    }


def _structure_family(name: Any) -> str:
    text = _normalize_structure_text(name)
    if not text:
        return "Other"
    for family, patterns in FAMILY_REGEX.items():
        for pattern in patterns:
            if pattern.search(text):
                return family
    return "Other"


def _load_attempts(
    path: Path,
    log_every: int = 5000,
    verbose: bool = True,
) -> Tuple[
    Dict[str, Dict[PlanKey, List[Dict[str, Any]]]],
    Dict[Tuple[str, Any, Any, Any], int],
    Dict[Tuple[str, Any, Any, int], List[float]],
]:
    per_protocol: Dict[str, Dict[PlanKey, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    attempt_index_map: Dict[Tuple[str, Any, Any, Any], int] = {}
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]] = {}

    start = time.time()
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            protocol = record.get("protocol_name") or "Unknown"
            patient_id = record.get("patient_id")
            plan_id = record.get("plan_id")
            matched = _safe_float(record.get("constraints_matched")) or 0.0
            if matched > 0:
                record["pass_rate"] = (_safe_float(record.get("constraints_pass")) or 0.0) / matched
                record["fail_rate"] = (_safe_float(record.get("constraints_fail")) or 0.0) / matched
                record["unknown_rate"] = (_safe_float(record.get("constraints_unknown")) or 0.0) / matched
                record["near_limit_rate"] = (_safe_float(record.get("near_limit_count")) or 0.0) / matched
            else:
                record["pass_rate"] = 0.0
                record["fail_rate"] = 0.0
                record["unknown_rate"] = 0.0
                record["near_limit_rate"] = 0.0
            plan_key = PlanKey(patient_id, plan_id)
            per_protocol[protocol][plan_key].append(record)
            if verbose and log_every and idx % log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[Phase3] Loaded {idx} attempts in {elapsed:.1f}s "
                    f"({len(per_protocol)} protocols)"
                )

    for protocol, plan_map in per_protocol.items():
        for plan_key, attempts in plan_map.items():
            attempts_sorted = _sort_attempts(attempts)
            for idx, attempt in enumerate(attempts_sorted, start=1):
                attempt_index = attempt.get("attempt_index") or idx
                attempt_number = attempt.get("attempt_number")
                if attempt_number is not None:
                    attempt_index_map[
                        (protocol, plan_key.patient_id, plan_key.plan_id, attempt_number)
                    ] = int(attempt_index)
                feature_map[
                    (protocol, plan_key.patient_id, plan_key.plan_id, int(attempt_index))
                ] = _extract_features(attempt, FEATURE_FIELDS)
                attempt["attempt_index"] = int(attempt_index)
                attempt["attempt_count"] = attempt.get("attempt_count") or len(attempts_sorted)

    if verbose:
        elapsed = time.time() - start
        print(
            f"[Phase3] Finished attempts load: {len(per_protocol)} protocols "
            f"in {elapsed:.1f}s"
        )

    return per_protocol, attempt_index_map, feature_map


def _load_family_scores(
    path: Path,
    attempt_index_map: Dict[Tuple[str, Any, Any, Any], int],
    log_every: int = 20000,
    verbose: bool = True,
) -> Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]]:
    per_attempt: Dict[Tuple[str, Any, Any, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    start = time.time()
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            protocol = record.get("protocol_name") or "Unknown"
            patient_id = record.get("patient_id")
            plan_id = record.get("plan_id")
            attempt_number = record.get("attempt_number")
            if attempt_number is None:
                continue
            attempt_index = attempt_index_map.get(
                (protocol, patient_id, plan_id, attempt_number)
            )
            if attempt_index is None:
                continue
            percentile = _safe_float(record.get("percentile"))
            if percentile is None:
                continue
            structure_name = record.get("structure_tg263") or record.get("structure")
            family = _structure_family(structure_name)
            key = (protocol, patient_id, plan_id, int(attempt_index))
            per_attempt[key][family].append(percentile)
            if verbose and log_every and idx % log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[Phase3] Loaded {idx} constraint rows in {elapsed:.1f}s "
                    f"({len(per_attempt)} attempts with family data)"
                )

    family_scores: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]] = {}
    for key, family_values in per_attempt.items():
        means = {}
        mins = {}
        counts = {}
        for family, values in family_values.items():
            if not values:
                continue
            means[family] = float(sum(values) / len(values))
            mins[family] = float(min(values))
            counts[family] = float(len(values))
        family_scores[key] = {"mean": means, "min": mins, "count": counts}

    if verbose:
        elapsed = time.time() - start
        print(
            f"[Phase3] Finished constraint load: {len(family_scores)} attempts "
            f"in {elapsed:.1f}s"
        )
    return family_scores


def _build_feature_map(
    base_feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]],
    include_family_features: bool,
) -> Tuple[Dict[Tuple[str, Any, Any, int], List[float]], List[str]]:
    if not include_family_features:
        return base_feature_map, list(FEATURE_FIELDS)

    feature_names = list(FEATURE_FIELDS)
    for family in FAMILY_ORDER:
        feature_names.extend(
            [f"{family}_mean_pct", f"{family}_min_pct", f"{family}_count"]
        )

    feature_map: Dict[Tuple[str, Any, Any, int], List[float]] = {}
    for key, base_vec in base_feature_map.items():
        stats = family_stats.get(key, {})
        means = stats.get("mean", {})
        mins = stats.get("min", {})
        counts = stats.get("count", {})
        extras: List[float] = []
        for family in FAMILY_ORDER:
            extras.append(float(means.get(family, 0.0)))
            extras.append(float(mins.get(family, 0.0)))
            extras.append(float(counts.get(family, 0.0)))
        feature_map[key] = list(base_vec) + extras

    return feature_map, feature_names


def _build_task1_pairs(
    attempts_by_plan: Dict[PlanKey, List[Dict[str, Any]]],
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    protocol: str,
    plan_subset: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    labels: List[int] = []
    for plan_key, attempts in attempts_by_plan.items():
        if plan_subset is not None and plan_key not in plan_subset:
            continue
        attempts_sorted = _sort_attempts(attempts)
        if len(attempts_sorted) < 2:
            continue
        for idx in range(len(attempts_sorted) - 1):
            left = attempts_sorted[idx]
            right = attempts_sorted[idx + 1]
            left_key = (protocol, plan_key.patient_id, plan_key.plan_id, left["attempt_index"])
            right_key = (protocol, plan_key.patient_id, plan_key.plan_id, right["attempt_index"])
            left_vec = feature_map.get(left_key)
            right_vec = feature_map.get(right_key)
            if left_vec is None or right_vec is None:
                continue
            diff = np.array(right_vec) - np.array(left_vec)
            features.append(diff.tolist())
            labels.append(1)
            features.append((-diff).tolist())
            labels.append(0)
    if not features:
        return np.empty((0, len(FEATURE_FIELDS))), np.array([])
    return np.array(features, dtype=float), np.array(labels, dtype=int)


def _build_task2_rows(
    attempts_by_plan: Dict[PlanKey, List[Dict[str, Any]]],
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    protocol: str,
    plan_subset: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    labels: List[float] = []
    for plan_key, attempts in attempts_by_plan.items():
        if plan_subset is not None and plan_key not in plan_subset:
            continue
        attempts_sorted = _sort_attempts(attempts)
        total = len(attempts_sorted)
        if total == 0:
            continue
        for idx, attempt in enumerate(attempts_sorted, start=1):
            attempt_index = attempt.get("attempt_index") or idx
            attempt_count = attempt.get("attempt_count") or total
            label = max(float(attempt_count - attempt_index), 0.0)
            key = (protocol, plan_key.patient_id, plan_key.plan_id, int(attempt_index))
            vec = feature_map.get(key)
            if vec is None:
                continue
            features.append(vec)
            labels.append(label)
    if not features:
        return np.empty((0, len(FEATURE_FIELDS))), np.array([])
    return np.array(features, dtype=float), np.array(labels, dtype=float)


def _build_task3_rows(
    attempts_by_plan: Dict[PlanKey, List[Dict[str, Any]]],
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]],
    protocol: str,
    plan_subset: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[List[float]] = []
    labels: List[str] = []
    for plan_key, attempts in attempts_by_plan.items():
        if plan_subset is not None and plan_key not in plan_subset:
            continue
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
            candidates = []
            for family in set(cur_families) & set(next_families):
                delta = next_families[family] - cur_families[family]
                candidates.append((delta, family))
            if not candidates:
                continue
            candidates.sort(reverse=True, key=lambda item: item[0])
            label = candidates[0][1]
            vec = feature_map.get(cur_key)
            if vec is None:
                continue
            features.append(vec)
            labels.append(label)
    if not features:
        return np.empty((0, len(FEATURE_FIELDS))), np.array([]), []
    return np.array(features, dtype=float), np.array(labels, dtype=str), sorted(set(labels))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _compute_phase3_metrics(
    per_protocol: Dict[str, Dict[PlanKey, List[Dict[str, Any]]]],
    feature_map: Dict[Tuple[str, Any, Any, int], List[float]],
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]],
    seed: int,
    min_plans_per_protocol: int,
    train_ratio: float,
    val_ratio: float,
    verbose: bool,
) -> Dict[str, Any]:
    task1_results = []
    task2_results = []
    task3_results = []
    task3_true: List[str] = []
    task3_pred: List[str] = []
    task3_labels: set = set()

    for protocol, plans in per_protocol.items():
        plan_keys = list(plans.keys())
        if len(plan_keys) < min_plans_per_protocol:
            continue

        splits = _split_plans(plan_keys, seed, train_ratio, val_ratio)
        if verbose:
            print(
                f"[Phase3] Protocol {protocol}: {len(plan_keys)} plans "
                f"(train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])})"
            )

        # Task 1: pairwise ranking
        X_train, y_train = _build_task1_pairs(plans, feature_map, protocol, splits["train"])
        X_test, y_test = _build_task1_pairs(plans, feature_map, protocol, splits["test"])
        task1_metrics = {"accuracy": None, "auc": None}
        if len(y_train) >= 2 and len(y_test) >= 2:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            task1_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            task1_metrics["auc"] = _safe_auc(y_test, y_prob)

        task1_results.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "pair_count": len(y_test),
                "metrics": task1_metrics,
            }
        )

        # Task 2: progress regression
        X_train, y_train = _build_task2_rows(plans, feature_map, protocol, splits["train"])
        X_test, y_test = _build_task2_rows(plans, feature_map, protocol, splits["test"])
        task2_metrics = {"mae": None}
        if len(y_train) >= 2 and len(y_test) >= 2:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", GradientBoostingRegressor(random_state=seed)),
                ]
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            task2_metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

        task2_results.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "attempt_count": len(y_test),
                "metrics": task2_metrics,
            }
        )

        # Task 3: next-improvement prediction
        X_train, y_train, _train_labels = _build_task3_rows(
            plans, feature_map, family_stats, protocol, splits["train"]
        )
        X_test, y_test, _test_labels = _build_task3_rows(
            plans, feature_map, family_stats, protocol, splits["test"]
        )
        task3_metrics = {"accuracy": None, "labels": []}
        if len(y_train) >= 2 and len(set(y_train)) >= 2 and len(y_test) >= 2:
            classifier = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", GradientBoostingClassifier(random_state=seed)),
                ]
            )
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            task3_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            try:
                task3_metrics["balanced_accuracy"] = float(
                    balanced_accuracy_score(y_test, y_pred)
                )
            except ValueError:
                task3_metrics["balanced_accuracy"] = None

            probs = classifier.predict_proba(X_test)
            topk_results: Dict[int, float] = {}
            for k in (1, 2, 3, 5):
                top_k = min(k, len(classifier.classes_))
                if top_k <= 0:
                    topk_results[k] = None
                    continue
                hits = 0
                for idx, label in enumerate(y_test):
                    top_idx = np.argsort(probs[idx])[-top_k:]
                    top_labels = set(classifier.classes_[top_idx])
                    if label in top_labels:
                        hits += 1
                topk_results[k] = hits / len(y_test)

            task3_metrics["top3_accuracy"] = topk_results.get(3)
            task3_metrics["top5_accuracy"] = topk_results.get(5)
            task3_metrics["topk_accuracy"] = topk_results

            task3_metrics["labels"] = sorted(set(y_train) | set(y_test))
            task3_true.extend(y_test.tolist())
            task3_pred.extend(y_pred.tolist())
            task3_labels.update(task3_metrics["labels"])

        task3_results.append(
            {
                "protocol_name": protocol,
                "plan_count": len(plan_keys),
                "pair_count": len(y_test),
                "metrics": task3_metrics,
            }
        )

    def macro_average(records: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [
            record["metrics"].get(key)
            for record in records
            if record["metrics"].get(key) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    task3_confusion = []
    task3_label_list = sorted(task3_labels)
    if task3_true and task3_pred and task3_label_list:
        task3_confusion = confusion_matrix(
            task3_true, task3_pred, labels=task3_label_list
        ).tolist()

    return {
        "task1": {
            "macro": {
                "accuracy": macro_average(task1_results, "accuracy"),
                "auc": macro_average(task1_results, "auc"),
            },
            "protocols": task1_results,
        },
        "task2": {
            "macro": {"mae": macro_average(task2_results, "mae")},
            "protocols": task2_results,
        },
        "task3": {
            "macro": {
                "accuracy": macro_average(task3_results, "accuracy"),
                "balanced_accuracy": macro_average(task3_results, "balanced_accuracy"),
                "top3_accuracy": macro_average(task3_results, "top3_accuracy"),
                "top5_accuracy": macro_average(task3_results, "top5_accuracy"),
            },
            "protocols": task3_results,
            "labels": task3_label_list,
            "confusion_matrix": task3_confusion,
        },
        "protocol_count": len(task1_results),
    }

def run_phase3_modeling(
    attempt_path: Path = ATTEMPT_PATH,
    constraint_path: Path = CONSTRAINT_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
    min_plans_per_protocol: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    verbose: bool = True,
    include_family_features: bool = False,
    write_output: bool = True,
) -> Dict[str, Any]:
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing {attempt_path}")

    if verbose:
        print(f"[Phase3] Loading attempts from {attempt_path}")
    per_protocol, attempt_index_map, base_feature_map = _load_attempts(
        attempt_path, verbose=verbose
    )
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]] = {}
    if constraint_path.exists():
        if verbose:
            print(f"[Phase3] Loading constraint rows from {constraint_path}")
        family_stats = _load_family_scores(
            constraint_path, attempt_index_map, verbose=verbose
        )
    elif verbose:
        print(f"[Phase3] Constraint features missing at {constraint_path}; Task 3 will be limited.")

    feature_map, feature_names = _build_feature_map(
        base_feature_map, family_stats, include_family_features
    )

    metrics = _compute_phase3_metrics(
        per_protocol,
        feature_map,
        family_stats,
        seed,
        min_plans_per_protocol,
        train_ratio,
        val_ratio,
        verbose,
    )
    metrics.update(
        {
            "generated_at": _utc_now_iso(),
            "min_plans_per_protocol": min_plans_per_protocol,
            "splits": {
                "train": train_ratio,
                "val": val_ratio,
                "test": round(1 - train_ratio - val_ratio, 3),
                "seed": seed,
            },
            "features": feature_names,
            "excluded_features": EXCLUDED_FIELDS,
            "include_family_features": include_family_features,
        }
    )

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Phase 3 metrics written.")
    print("Protocols:", len(metrics.get("task1", {}).get("protocols", [])))
    print("Task 1 macro accuracy:", metrics["task1"]["macro"]["accuracy"])
    print("Task 2 macro MAE:", metrics["task2"]["macro"]["mae"])
    print("Task 3 macro accuracy:", metrics["task3"]["macro"]["accuracy"])
    if write_output:
        print("Output:", output_path)

    return metrics


def run_phase3_sweep(
    thresholds: List[int],
    attempt_path: Path = ATTEMPT_PATH,
    constraint_path: Path = CONSTRAINT_PATH,
    output_path: Path = SWEEP_PATH,
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    include_family_features: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing {attempt_path}")

    if verbose:
        print(f"[Phase3] Sweep loading attempts from {attempt_path}")
    per_protocol, attempt_index_map, base_feature_map = _load_attempts(
        attempt_path, verbose=verbose
    )
    family_stats: Dict[Tuple[str, Any, Any, int], Dict[str, Dict[str, float]]] = {}
    if constraint_path.exists():
        if verbose:
            print(f"[Phase3] Sweep loading constraint rows from {constraint_path}")
        family_stats = _load_family_scores(
            constraint_path, attempt_index_map, verbose=verbose
        )

    feature_map, feature_names = _build_feature_map(
        base_feature_map, family_stats, include_family_features
    )

    results: List[Dict[str, Any]] = []
    for threshold in thresholds:
        if verbose:
            print(f"[Phase3] Sweep min_plans_per_protocol={threshold}")
        metrics = _compute_phase3_metrics(
            per_protocol,
            feature_map,
            family_stats,
            seed,
            threshold,
            train_ratio,
            val_ratio,
            verbose=False,
        )
        results.append(
            {
                "min_plans_per_protocol": threshold,
                "protocols": metrics.get("protocol_count", 0),
                "task1_accuracy": metrics["task1"]["macro"]["accuracy"],
                "task1_auc": metrics["task1"]["macro"]["auc"],
                "task2_mae": metrics["task2"]["macro"]["mae"],
                "task3_accuracy": metrics["task3"]["macro"]["accuracy"],
            }
        )

    payload = {
        "generated_at": _utc_now_iso(),
        "splits": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1 - train_ratio - val_ratio, 3),
            "seed": seed,
        },
        "include_family_features": include_family_features,
        "features": feature_names,
        "thresholds": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Phase 3 sweep written.")
    print("Output:", output_path)
    return payload


if __name__ == "__main__":
    run_phase3_modeling()
