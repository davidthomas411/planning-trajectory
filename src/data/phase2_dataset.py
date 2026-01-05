import json
import math
import os
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymongo import MongoClient

try:
    from src.data.trajectory_builder import DEFAULT_MONGODB_URI, _sort_attempts
except ModuleNotFoundError:  # Allow running as a script without package context.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.trajectory_builder import DEFAULT_MONGODB_URI, _sort_attempts

_STRUCTURE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_METRIC_NORMALIZE_RE = re.compile(r"\s+")
_PROTOCOL_NORMALIZE_RE = re.compile(r"\s+")
_STRUCTURE_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_STRUCTURE_STOPWORDS = {
    "left",
    "right",
    "lt",
    "rt",
    "prox",
    "dist",
    "ant",
    "post",
    "sup",
    "inf",
    "upper",
    "lower",
    "mid",
    "med",
    "lat",
    "tree",
}


def _normalize_structure_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return _STRUCTURE_NORMALIZE_RE.sub("", value.strip().lower())


def _normalize_metric_key(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = _METRIC_NORMALIZE_RE.sub("", normalized)
    normalized = re.sub(r"^d(\d+(?:\.\d+)?)cc", r"d\1", normalized)
    return normalized


def _normalize_protocol_key(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = _PROTOCOL_NORMALIZE_RE.sub(" ", value.strip().lower())
    return normalized


def _structure_tokens(value: Optional[str]) -> set:
    if not value:
        return set()
    raw_tokens = _STRUCTURE_TOKEN_SPLIT_RE.split(value.strip().lower())
    tokens: set = set()
    for token in raw_tokens:
        if not token:
            continue
        if token in _STRUCTURE_STOPWORDS:
            continue
        tokens.add(token)
        for suffix in ("us", "um", "ae", "i", "a", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                tokens.add(token[: -len(suffix)])
    return {token for token in tokens if len(token) >= 4}


def _normalize_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_operator(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def _goal_direction(goal_operator: Any) -> Optional[str]:
    if goal_operator is None:
        return None
    op = str(goal_operator).strip().lower()
    if op in {">", ">=", "ge", "gte", "≥"}:
        return "higher"
    if op in {"<", "<=", "le", "lte", "≤"}:
        return "lower"
    return None


def _priority_weight(priority: Any) -> float:
    if priority is None:
        return 1.0
    try:
        value = int(float(priority))
    except (TypeError, ValueError):
        return 1.0
    if value == 1:
        return 2.0
    if value == 2:
        return 1.0
    return 1.0


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    return float(values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f))


def _percentile_rank(value: float, sorted_values: List[float]) -> float:
    if not sorted_values:
        return 0.0
    index = bisect_left(sorted_values, value)
    if index <= 0:
        return 0.0
    if index >= len(sorted_values):
        return 100.0
    return (index / len(sorted_values)) * 100.0


def _directional_percentile(
    value: float, sorted_values: List[float], direction: Optional[str]
) -> Optional[float]:
    if direction not in {"higher", "lower"}:
        return None
    percentile = _percentile_rank(value, sorted_values)
    if direction == "lower":
        return max(0.0, min(100.0, 100.0 - percentile))
    return max(0.0, min(100.0, percentile))


def _constraint_key(
    structure: str,
    metric_label: str,
    goal: Optional[Dict[str, Any]],
    variation: Optional[Dict[str, Any]],
    priority: Any = None,
    include_thresholds: bool = True,
) -> Tuple[Any, ...]:
    if not include_thresholds:
        goal = None
        variation = None
    goal_op = _normalize_operator(goal.get("operator")) if isinstance(goal, dict) else None
    goal_val = _normalize_numeric(goal.get("value")) if isinstance(goal, dict) else None
    var_op = _normalize_operator(variation.get("operator")) if isinstance(variation, dict) else None
    var_val = _normalize_numeric(variation.get("value")) if isinstance(variation, dict) else None
    priority_val = _normalize_numeric(priority)
    return (
        _normalize_structure_key(structure),
        _normalize_metric_key(metric_label),
        goal_op,
        goal_val,
        var_op,
        var_val,
        priority_val,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _extract_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        if "value" in value:
            return _normalize_numeric(value.get("value"))
    return None


def _normalize_protocol_name(standard_protocol: Optional[str], protocol_name: Optional[str]) -> str:
    name = (standard_protocol or "").strip()
    if not name:
        name = (protocol_name or "").strip()
    return name if name else "Unknown"


def _resolve_standard_protocol_name(
    raw_name: str,
    template_to_standard: Dict[str, str],
    standard_name_map: Dict[str, str],
) -> str:
    if not raw_name:
        return "Unknown"
    raw_name = raw_name.strip()
    key = _normalize_protocol_key(raw_name)
    if not key:
        return "Unknown"
    mapped = template_to_standard.get(key) or standard_name_map.get(key)
    return mapped or raw_name


def _load_protocol_name_map(db) -> Tuple[Dict[str, str], Dict[str, str]]:
    standard_coll = db.get_collection("standard_protocols")
    protocols_coll = db.get_collection("protocols")

    standard_name_by_id: Dict[Any, str] = {}
    standard_name_map: Dict[str, str] = {}
    for doc in standard_coll.find({}, {"protocol_name": 1}):
        name = doc.get("protocol_name")
        if not name:
            continue
        standard_key = _normalize_protocol_key(name)
        if standard_key:
            standard_name_map[standard_key] = name
        standard_id = doc.get("_id")
        if standard_id is not None:
            standard_name_by_id[standard_id] = name
        alt_id = doc.get("standard_id")
        if alt_id is not None:
            standard_name_by_id.setdefault(alt_id, name)

    template_to_standard: Dict[str, str] = {}
    for doc in protocols_coll.find(
        {"standard_ref.standard_id": {"$exists": True}},
        {"protocol_name": 1, "standard_ref": 1},
    ):
        template_name = doc.get("protocol_name")
        if not template_name:
            continue
        template_key = _normalize_protocol_key(template_name)
        if not template_key:
            continue
        standard_ref = doc.get("standard_ref") or {}
        standard_id = standard_ref.get("standard_id")
        standard_name = standard_ref.get("standard_name")
        mapped = standard_name_by_id.get(standard_id) or standard_name
        if mapped:
            template_to_standard[template_key] = mapped
    return template_to_standard, standard_name_map


def _load_structure_aliases(db) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    coll = db.get_collection("custom_structure_aliases")
    for doc in coll.find({}, {"canonical": 1, "aliases": 1}):
        canonical = doc.get("canonical")
        if not canonical:
            continue
        canonical_key = _normalize_structure_key(canonical)
        if canonical_key:
            alias_map[canonical_key] = canonical
        for alias in doc.get("aliases") or []:
            alias_key = _normalize_structure_key(alias)
            if alias_key:
                alias_map[alias_key] = canonical
    return alias_map


def _select_canonical_structure(
    structure_name: str,
    alias_map: Dict[str, str],
    canonical_by_key: Dict[str, str],
    canonical_tokens: Dict[str, set],
) -> str:
    structure_key = _normalize_structure_key(structure_name)
    canonical = alias_map.get(structure_key) or canonical_by_key.get(structure_key)
    if canonical:
        return canonical

    tokens = _structure_tokens(structure_name)
    if not tokens:
        return structure_name

    best_key = ""
    best_score = (0, 0)
    for key, token_set in canonical_tokens.items():
        intersection = tokens & token_set
        if not intersection:
            continue
        score = (len(intersection), max(len(token) for token in intersection))
        if score > best_score:
            best_score = score
            best_key = key

    return canonical_by_key.get(best_key, structure_name)


def _build_protocol_specs(
    db,
    protocol_names: Iterable[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    standard_coll = db.get_collection("standard_protocols")
    protocols_coll = db.get_collection("protocols")
    alias_map = _load_structure_aliases(db)

    protocol_specs_by_key: Dict[str, Dict[str, Any]] = {}
    remaining = {name for name in protocol_names if name}

    def build_spec(
        name: str, constraints: List[Dict[str, Any]], source: str
    ) -> Dict[str, Any]:
        include_thresholds = source == "protocols"
        constraint_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for constraint in constraints:
            structure = _stringify(constraint.get("structure") or "Unknown structure")
            priority = constraint.get("priority")
            direction = None
            goal = None
            variation = None
            metric_display = ""

            if source == "protocols":
                metric = constraint.get("metric") or {}
                metric_display = _stringify(metric.get("display") or constraint.get("objective"))
                goal = (
                    constraint.get("goal")
                    if isinstance(constraint.get("goal"), dict)
                    else None
                )
                variation = (
                    constraint.get("variation")
                    if isinstance(constraint.get("variation"), dict)
                    else None
                )
                direction = _goal_direction(
                    (goal or {}).get("operator") if isinstance(goal, dict) else None
                )
            else:
                metric_display = _stringify(constraint.get("objective"))
                goal_operator = constraint.get("goal_operator")
                goal = {
                    "operator": goal_operator,
                    "value": constraint.get("goal_value"),
                }
                variation = {
                    "operator": constraint.get("variation_operator"),
                    "value": constraint.get("variation_value"),
                }
                direction = _goal_direction(goal_operator)

            key = _constraint_key(
                structure,
                metric_display,
                goal,
                variation,
                priority,
                include_thresholds,
            )
            existing = constraint_map.get(key)
            if existing:
                if existing.get("direction") is None and direction is not None:
                    existing["direction"] = direction
                continue

            constraint_map[key] = {
                "structure": structure,
                "metric": metric_display,
                "priority": priority,
                "direction": direction,
                "weight": _priority_weight(priority),
                "values": [],
                "sorted_values": [],
            }

        canonical_by_key: Dict[str, str] = {}
        canonical_tokens: Dict[str, set] = {}
        for constraint in constraint_map.values():
            structure = constraint["structure"]
            structure_key = _normalize_structure_key(structure)
            if not structure_key or structure_key in canonical_by_key:
                continue
            canonical_by_key[structure_key] = structure
            canonical_tokens[structure_key] = _structure_tokens(structure)

        return {
            "name": name,
            "constraint_map": constraint_map,
            "include_thresholds": include_thresholds,
            "canonical_by_key": canonical_by_key,
            "canonical_tokens": canonical_tokens,
            "constraint_source": source,
        }

    if remaining:
        for doc in standard_coll.find(
            {"protocol_name": {"$in": list(remaining)}},
            {"protocol_name": 1, "constraints": 1},
        ):
            name = doc.get("protocol_name")
            if not name:
                continue
            constraints = doc.get("constraints") or []
            spec = build_spec(name, constraints, "standard_protocols")
            protocol_specs_by_key[_normalize_protocol_key(name)] = spec
            remaining.discard(name)

    if remaining:
        fallback_docs: Dict[str, Dict[str, Any]] = {}
        for doc in protocols_coll.find(
            {"standard_ref.standard_name": {"$in": list(remaining)}},
            {"protocol_name": 1, "constraints": 1, "standard_ref": 1},
        ):
            standard_ref = doc.get("standard_ref") or {}
            standard_name = standard_ref.get("standard_name")
            if not standard_name or standard_name not in remaining:
                continue
            is_primary = bool(standard_ref.get("is_primary"))
            if standard_name not in fallback_docs or is_primary:
                fallback_docs[standard_name] = doc
        for name, doc in fallback_docs.items():
            constraints = doc.get("constraints") or []
            spec = build_spec(name, constraints, "protocols")
            protocol_specs_by_key[_normalize_protocol_key(name)] = spec

    return protocol_specs_by_key, alias_map


def _load_qualified_plans(
    coll,
    min_attempts: int,
) -> List[Dict[str, Any]]:
    pipeline = [
        {
            "$sort": {
                "patient.patient_id": 1,
                "plan_id": 1,
                "attempt_number": 1,
                "created_at": 1,
            }
        },
        {
            "$group": {
                "_id": {"patient_id": "$patient.patient_id", "plan_id": "$plan_id"},
                "attempt_count": {"$sum": 1},
                "last_is_approved": {"$last": "$approval.is_approved"},
                "standard_protocol": {"$last": "$standard_protocol"},
                "protocol_name": {"$last": "$protocol.protocol_name"},
            }
        },
        {
            "$match": {
                "attempt_count": {"$gte": min_attempts},
                "last_is_approved": True,
            }
        },
    ]
    return list(coll.aggregate(pipeline, allowDiskUse=True))


def build_phase2_dataset(
    mongo_uri: Optional[str] = None,
    db_name: str = "planeval",
    collection_name: str = "evaluations",
    min_attempts: int = 2,
    min_coverage_pct: float = 0.7,
    plateau_delta: float = 2.0,
    near_limit_pct: float = 0.02,
    output_dir: str = "data/derived",
    write_constraint_features: bool = True,
    print_summary: bool = True,
) -> Dict[str, Any]:
    uri = mongo_uri or DEFAULT_MONGODB_URI
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    coll = db[collection_name]

    qualified_plans = _load_qualified_plans(coll, min_attempts=min_attempts)
    qualified_keys: set = set()
    plan_attempt_counts: Dict[Tuple[Any, Any], int] = {}
    protocol_names: set = set()
    for doc in qualified_plans:
        patient_id = doc["_id"].get("patient_id")
        plan_id = doc["_id"].get("plan_id")
        if patient_id is None or plan_id is None:
            continue
        qualified_keys.add((patient_id, plan_id))
        plan_attempt_counts[(patient_id, plan_id)] = int(doc.get("attempt_count", 0))
        protocol_raw = _normalize_protocol_name(
            doc.get("standard_protocol"), doc.get("protocol_name")
        )
        protocol_names.add(protocol_raw)

    template_to_standard, standard_name_map = _load_protocol_name_map(db)
    standard_protocol_names = {
        _resolve_standard_protocol_name(name, template_to_standard, standard_name_map)
        for name in protocol_names
    }
    protocol_specs_by_key, alias_map = _build_protocol_specs(
        db, standard_protocol_names
    )

    # Build CPDs from final approved evaluations that meet coverage threshold.
    cpd_included_plans = Counter()
    pipeline = [
        {
            "$sort": {
                "patient.patient_id": 1,
                "plan_id": 1,
                "attempt_number": 1,
                "created_at": 1,
            }
        },
        {
            "$group": {
                "_id": {"patient_id": "$patient.patient_id", "plan_id": "$plan_id"},
                "attempt_count": {"$sum": 1},
                "last_is_approved": {"$last": "$approval.is_approved"},
                "standard_protocol": {"$last": "$standard_protocol"},
                "protocol_name": {"$last": "$protocol.protocol_name"},
                "last_results": {"$last": "$results"},
            }
        },
        {
            "$match": {
                "attempt_count": {"$gte": min_attempts},
                "last_is_approved": True,
            }
        },
    ]

    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        protocol_raw = _normalize_protocol_name(
            doc.get("standard_protocol"), doc.get("protocol_name")
        )
        standard_name = _resolve_standard_protocol_name(
            protocol_raw, template_to_standard, standard_name_map
        )
        spec = protocol_specs_by_key.get(_normalize_protocol_key(standard_name))
        if not spec:
            continue
        constraint_map = spec["constraint_map"]
        if not constraint_map:
            continue

        matched_constraints = 0
        seen_keys: set = set()
        for result in doc.get("last_results", []) or []:
            structure_raw = (
                result.get("structure") or result.get("structure_tg263") or "Unknown"
            )
            structure_name = _select_canonical_structure(
                _stringify(structure_raw) or "Unknown",
                alias_map,
                spec["canonical_by_key"],
                spec["canonical_tokens"],
            )
            metric = result.get("metric") or {}
            metric_display = _stringify(metric.get("display") or result.get("objective"))
            goal = result.get("goal") if isinstance(result.get("goal"), dict) else None
            variation = (
                result.get("variation")
                if isinstance(result.get("variation"), dict)
                else None
            )
            key = _constraint_key(
                structure_name,
                metric_display,
                goal,
                variation,
                result.get("priority"),
                spec["include_thresholds"],
            )
            if key in seen_keys:
                continue
            constraint = constraint_map.get(key)
            if not constraint:
                continue
            numeric_value = _extract_numeric(result.get("achieved"))
            if numeric_value is None:
                continue
            seen_keys.add(key)
            constraint["values"].append(numeric_value)
            matched_constraints += 1

        coverage_pct = matched_constraints / max(len(constraint_map), 1)
        if coverage_pct < min_coverage_pct:
            continue
        cpd_included_plans[standard_name] += 1

    for spec in protocol_specs_by_key.values():
        for constraint in spec["constraint_map"].values():
            values = constraint.get("values", [])
            constraint["sorted_values"] = sorted(values)

    os.makedirs(output_dir, exist_ok=True)
    attempt_path = os.path.join(output_dir, "plan_attempt_features.jsonl")
    constraint_path = os.path.join(output_dir, "constraint_features.jsonl")

    attempt_records: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    constraint_file = open(constraint_path, "w", encoding="utf-8") if write_constraint_features else None

    projection = {
        "patient.patient_id": 1,
        "plan_id": 1,
        "attempt_number": 1,
        "created_at": 1,
        "standard_protocol": 1,
        "protocol.protocol_name": 1,
        "results": 1,
    }

    for doc in coll.find({}, projection):
        patient_id = doc.get("patient", {}).get("patient_id")
        plan_id = doc.get("plan_id")
        if (patient_id, plan_id) not in qualified_keys:
            continue

        protocol_raw = _normalize_protocol_name(
            doc.get("standard_protocol"), doc.get("protocol", {}).get("protocol_name")
        )
        standard_name = _resolve_standard_protocol_name(
            protocol_raw, template_to_standard, standard_name_map
        )
        spec = protocol_specs_by_key.get(_normalize_protocol_key(standard_name))
        if not spec:
            continue
        constraint_map = spec["constraint_map"]
        if not constraint_map:
            continue

        total_constraints = len(constraint_map)
        matched_constraints = 0
        pass_count = 0
        fail_count = 0
        unknown_count = 0
        near_limit_count = 0
        margins: List[float] = []
        normalized_margins: List[float] = []
        percentiles: List[float] = []
        total_weight = 0.0
        total_score = 0.0
        seen_keys: set = set()

        for result in doc.get("results", []) or []:
            structure_raw = (
                result.get("structure") or result.get("structure_tg263") or "Unknown"
            )
            structure_name = _select_canonical_structure(
                _stringify(structure_raw) or "Unknown",
                alias_map,
                spec["canonical_by_key"],
                spec["canonical_tokens"],
            )
            metric = result.get("metric") or {}
            metric_display = _stringify(metric.get("display") or result.get("objective"))
            goal = result.get("goal") if isinstance(result.get("goal"), dict) else None
            variation = (
                result.get("variation")
                if isinstance(result.get("variation"), dict)
                else None
            )
            key = _constraint_key(
                structure_name,
                metric_display,
                goal,
                variation,
                result.get("priority"),
                spec["include_thresholds"],
            )
            if key in seen_keys:
                continue
            constraint = constraint_map.get(key)
            if not constraint:
                continue
            numeric_value = _extract_numeric(result.get("achieved"))
            if numeric_value is None:
                continue
            seen_keys.add(key)
            matched_constraints += 1

            goal_operator = (goal or {}).get("operator") if isinstance(goal, dict) else None
            goal_value = (goal or {}).get("value") if isinstance(goal, dict) else None
            goal_value_numeric = _normalize_numeric(goal_value)

            pass_fail = None
            margin = None
            normalized_margin = None
            if goal_value_numeric is not None and goal_operator is not None:
                op = str(goal_operator).strip()
                if op in ("<", "<=", "≤"):
                    pass_fail = numeric_value <= goal_value_numeric
                    margin = goal_value_numeric - numeric_value
                elif op in (">", ">=", "≥"):
                    pass_fail = numeric_value >= goal_value_numeric
                    margin = numeric_value - goal_value_numeric

            if pass_fail is True:
                pass_count += 1
            elif pass_fail is False:
                fail_count += 1
            else:
                unknown_count += 1

            if margin is not None:
                margins.append(margin)
                if goal_value_numeric not in (None, 0):
                    normalized_margin = margin / abs(goal_value_numeric)
                    normalized_margins.append(normalized_margin)
                    if abs(normalized_margin) <= near_limit_pct:
                        near_limit_count += 1

            percentile = None
            sorted_values = constraint.get("sorted_values") or []
            direction = constraint.get("direction")
            if sorted_values and direction is not None:
                percentile = _directional_percentile(
                    numeric_value, sorted_values, direction
                )
                if percentile is not None:
                    percentiles.append(percentile)
                    weight = float(constraint.get("weight", 1.0))
                    total_weight += weight
                    total_score += weight * percentile

            if constraint_file:
                constraint_record = {
                    "protocol_name": standard_name,
                    "patient_id": patient_id,
                    "plan_id": plan_id,
                    "attempt_number": doc.get("attempt_number"),
                    "structure": structure_name,
                    "structure_tg263": result.get("structure_tg263"),
                    "metric_display": metric_display,
                    "metric_type": (metric or {}).get("type"),
                    "metric_subtype": (metric or {}).get("subtype"),
                    "priority": result.get("priority"),
                    "goal_operator": goal_operator,
                    "goal_value": goal_value_numeric,
                    "achieved_value": numeric_value,
                    "pass": pass_fail,
                    "margin": margin,
                    "normalized_margin": normalized_margin,
                    "near_limit": (
                        abs(normalized_margin) <= near_limit_pct
                        if normalized_margin is not None
                        else None
                    ),
                    "percentile": percentile,
                }
                constraint_file.write(json.dumps(constraint_record) + "\n")

        coverage_pct = matched_constraints / max(total_constraints, 1)
        if coverage_pct < min_coverage_pct:
            continue

        plan_score = total_score / total_weight if total_weight > 0 else None
        if plan_score is None:
            continue

        attempt_record = {
            "protocol_name": standard_name,
            "patient_id": patient_id,
            "plan_id": plan_id,
            "attempt_number": doc.get("attempt_number"),
            "created_at": doc.get("created_at"),
            "attempt_count": plan_attempt_counts.get((patient_id, plan_id)),
            "constraints_total": total_constraints,
            "constraints_matched": matched_constraints,
            "coverage_pct": coverage_pct,
            "constraints_pass": pass_count,
            "constraints_fail": fail_count,
            "constraints_unknown": unknown_count,
            "near_limit_count": near_limit_count,
            "worst_margin": min(margins) if margins else None,
            "worst_normalized_margin": min(normalized_margins) if normalized_margins else None,
            "min_percentile": min(percentiles) if percentiles else None,
            "p10_percentile": _percentile(percentiles, 10.0) if percentiles else None,
            "p50_percentile": _percentile(percentiles, 50.0) if percentiles else None,
            "p90_percentile": _percentile(percentiles, 90.0) if percentiles else None,
            "mean_percentile": (sum(percentiles) / len(percentiles)) if percentiles else None,
            "plan_score": plan_score,
        }
        attempt_records[(patient_id, plan_id)].append(attempt_record)

    if constraint_file:
        constraint_file.close()

    with open(attempt_path, "w", encoding="utf-8") as attempt_file:
        attempt_count = 0
        label_counts = Counter()
        for key, attempts in attempt_records.items():
            attempts_sorted = _sort_attempts(attempts)
            scores = [attempt["plan_score"] for attempt in attempts_sorted]
            for idx, attempt in enumerate(attempts_sorted):
                future_scores = scores[idx + 1 :]
                max_future = max(future_scores) if future_scores else attempt["plan_score"]
                delta = max_future - attempt["plan_score"]
                label_stop = delta < plateau_delta
                attempt["label_stop"] = label_stop
                attempt["future_best_delta"] = delta
                attempt["attempt_index"] = idx + 1
                attempt["attempt_progress"] = (
                    (idx) / (len(attempts_sorted) - 1)
                    if len(attempts_sorted) > 1
                    else 1.0
                )
                label_counts["stop" if label_stop else "continue"] += 1
                attempt_file.write(json.dumps(attempt, default=str) + "\n")
                attempt_count += 1

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "min_attempts": min_attempts,
        "min_coverage_pct": min_coverage_pct,
        "plateau_delta": plateau_delta,
        "near_limit_pct": near_limit_pct,
        "qualified_plans": len(qualified_keys),
        "attempts_written": attempt_count,
        "protocols": len(protocol_specs_by_key),
        "cpd_plans_used": dict(cpd_included_plans),
    }

    summary_path = os.path.join(output_dir, "phase2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(json.dumps(summary, indent=2))

    client.close()

    if print_summary:
        print("Phase 2 dataset written.")
        print("Attempts:", attempt_count)
        print("Protocols:", len(protocol_specs_by_key))
        print("Coverage threshold:", min_coverage_pct)
        print("Plateau delta:", plateau_delta)
        print("Output dir:", output_dir)

    return summary


if __name__ == "__main__":
    build_phase2_dataset()
