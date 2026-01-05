import html
import json
import math
import mimetypes
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from bisect import bisect_left
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

from pymongo import MongoClient

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.trajectory_builder import DEFAULT_MONGODB_URI


ASSET_CANDIDATES = [
    ("jefferson-university-2.svg", "image/svg+xml"),
    ("TJU_logo.jpg", "image/jpeg"),
]

DEFAULT_PORT = 8000
DEFAULT_MIN_PROTOCOL_PLANS = 20

_METRICS_CACHE: Dict[str, Any] = {}
_METRICS_ERROR: str = ""

PHASE2_DIR = ROOT_DIR / "data" / "derived"
PHASE2_ATTEMPTS_PATH = PHASE2_DIR / "plan_attempt_features.jsonl"
PHASE2_SUMMARY_PATH = PHASE2_DIR / "phase2_summary.json"
PHASE2_CONSTRAINTS_PATH = PHASE2_DIR / "constraint_features.jsonl"

_PHASE2_CACHE: Dict[str, Any] = {}
_PHASE2_ERROR: str = ""

PHASE3_METRICS_PATH = PHASE2_DIR / "phase3_metrics.json"
_PHASE3_CACHE: Dict[str, Any] = {}
_PHASE3_ERROR: str = ""

PHASE3_SWEEP_PATH = PHASE2_DIR / "phase3_sweep.json"
_PHASE3_SWEEP_CACHE: Dict[str, Any] = {}
_PHASE3_SWEEP_ERROR: str = ""

PHASE3_BASELINE_PATH = PHASE2_DIR / "phase3_baselines.json"
_PHASE3_BASELINE_CACHE: Dict[str, Any] = {}
_PHASE3_BASELINE_ERROR: str = ""

PHASE3_ANALYSIS_PATH = PHASE2_DIR / "phase3_analysis.json"
_PHASE3_ANALYSIS_CACHE: Dict[str, Any] = {}
_PHASE3_ANALYSIS_ERROR: str = ""

PHASE3_ALTERNATIVES_PATH = PHASE2_DIR / "phase3_alternatives.json"
_PHASE3_ALTERNATIVES_CACHE: Dict[str, Any] = {}
_PHASE3_ALTERNATIVES_ERROR: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _mongo_client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def _parse_int(value: Optional[str], default: int) -> int:
    try:
        if value is None:
            return default
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _single_aggregate(coll, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs = list(coll.aggregate(pipeline, allowDiskUse=True))
    return docs[0] if docs else {}


def _normalize_protocol_name(standard_protocol: Optional[str], protocol_name: Optional[str]) -> str:
    name = (standard_protocol or "").strip()
    if not name:
        name = (protocol_name or "").strip()
    return name if name else "Unknown"


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

FAMILY_REGEX = OrderedDict(
    (family, [re.compile(pattern) for pattern in patterns])
    for family, patterns in FAMILY_PATTERNS.items()
)


def _normalize_family_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _STRUCTURE_NORMALIZE_RE.sub(" ", text)
    return text.strip()


def _structure_family(value: Any) -> str:
    text = _normalize_family_text(value)
    if not text:
        return "Other"
    for family, patterns in FAMILY_REGEX.items():
        for pattern in patterns:
            if pattern.search(text):
                return family
    return "Other"


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


def _normalize_operator(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def _normalize_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_priority(value: Any) -> Optional[float | str]:
    if value is None:
        return None
    numeric = _normalize_numeric(value)
    if numeric is not None:
        return numeric
    return _normalize_operator(value)


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
    priority_val = _normalize_priority(priority)
    return (
        _normalize_structure_key(structure),
        _normalize_metric_key(metric_label),
        goal_op,
        goal_val,
        var_op,
        var_val,
        priority_val,
    )


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


def _key_component(value: Any) -> Any:
    if value is None:
        return None
    return _stringify(value)


def _extract_numeric(value: Any) -> Tuple[Optional[float], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        try:
            return float(value), None
        except ValueError:
            return None, None
    if isinstance(value, dict):
        if "value" in value:
            try:
                return float(value.get("value")), value.get("unit")
            except (TypeError, ValueError):
                return None, value.get("unit")
    return None, None


def _format_threshold(threshold: Optional[Dict[str, Any]]) -> str:
    if not isinstance(threshold, dict):
        return ""
    operator = threshold.get("operator")
    value = threshold.get("value")
    unit = threshold.get("unit")
    if operator is None and value is None:
        return ""
    operator_text = str(operator).strip() if operator is not None else ""
    if value is None:
        return operator_text
    value_text = _format_number(value)
    unit_text = f"{unit}" if unit else ""
    if operator_text:
        return f"{operator_text} {value_text}{unit_text}"
    return f"{value_text}{unit_text}"


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


def _fetch_qualified_plans(coll, min_attempts: int) -> List[Dict[str, Any]]:
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


def _build_plan_index(
    qualified_plans: Iterable[Dict[str, Any]],
    template_to_standard: Optional[Dict[str, str]] = None,
    standard_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[set, Counter, Dict[str, Dict[str, Any]], int, set]:
    qualified_keys: set = set()
    attempt_distribution: Counter = Counter()
    patient_ids: set = set()
    protocol_stats: Dict[str, Dict[str, Any]] = {}
    total_attempts = 0

    for doc in qualified_plans:
        patient_id = doc["_id"].get("patient_id")
        plan_id = doc["_id"].get("plan_id")
        attempt_count = int(doc.get("attempt_count", 0))
        if patient_id is None or plan_id is None:
            continue

        qualified_keys.add((patient_id, plan_id))
        patient_ids.add(patient_id)
        attempt_distribution[attempt_count] += 1
        total_attempts += attempt_count

        protocol_name = _normalize_protocol_name(
            doc.get("standard_protocol"), doc.get("protocol_name")
        )
        if template_to_standard is not None or standard_name_map is not None:
            protocol_name = _resolve_standard_protocol_name(
                protocol_name,
                template_to_standard or {},
                standard_name_map or {},
            )
        stats = protocol_stats.setdefault(
            protocol_name,
            {"plan_count": 0, "patient_ids": set()},
        )
        stats["plan_count"] += 1
        stats["patient_ids"].add(patient_id)

    return qualified_keys, attempt_distribution, protocol_stats, total_attempts, patient_ids


def _iter_qualified_evaluations(
    coll,
    qualified_keys: set,
) -> Tuple[
    Counter,
    int,
    set,
    Dict[int, List[float]],
    int,
    Optional[datetime],
    Optional[datetime],
]:
    approval_counts: Counter = Counter()
    total_constraints = 0
    structures: set = set()
    scores_by_attempt: Dict[int, List[float]] = defaultdict(list)
    evaluation_count = 0
    min_created: Optional[datetime] = None
    max_created: Optional[datetime] = None

    projection = {
        "patient.patient_id": 1,
        "plan_id": 1,
        "attempt_number": 1,
        "summary.score": 1,
        "approval.status": 1,
        "results.structure": 1,
        "created_at": 1,
    }

    cursor = coll.find({}, projection)
    for doc in cursor:
        patient = doc.get("patient", {})
        patient_id = patient.get("patient_id")
        plan_id = doc.get("plan_id")
        if (patient_id, plan_id) not in qualified_keys:
            continue

        evaluation_count += 1
        approval_status = doc.get("approval", {}).get("status") or "Unknown"
        approval_counts[approval_status] += 1

        created_at = doc.get("created_at")
        if created_at:
            if min_created is None or created_at < min_created:
                min_created = created_at
            if max_created is None or created_at > max_created:
                max_created = created_at

        results = doc.get("results") or []
        total_constraints += len(results)
        for result in results:
            structure = result.get("structure")
            if structure:
                structures.add(structure)

        attempt_number = doc.get("attempt_number")
        score_value = doc.get("summary", {}).get("score")
        if score_value is not None and attempt_number is not None:
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                score = None
            if score is not None:
                scores_by_attempt[int(attempt_number)].append(score)

    return (
        approval_counts,
        total_constraints,
        structures,
        scores_by_attempt,
        evaluation_count,
        min_created,
        max_created,
    )


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


def _compute_plan_score_stats(
    db,
    coll,
    qualified_keys: set,
    plan_attempt_counts: Dict[Tuple[Any, Any], int],
    protocol_stats: Dict[str, Dict[str, Any]],
    template_to_standard: Dict[str, str],
    standard_name_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    protocol_specs_by_key, alias_map = _build_protocol_specs(db, protocol_stats.keys())

    # Build CPDs from final approved evaluations per plan
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
                "attempt_count": {"$gte": 2},
                "last_is_approved": True,
            }
        },
        {
            "$project": {
                "standard_protocol": 1,
                "protocol_name": 1,
                "last_results": 1,
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
            constraint = spec["constraint_map"].get(key)
            if not constraint:
                continue
            numeric_value, _unit = _extract_numeric(result.get("achieved"))
            if numeric_value is None:
                continue
            constraint["values"].append(numeric_value)

    for spec in protocol_specs_by_key.values():
        for constraint in spec["constraint_map"].values():
            values = constraint.get("values", [])
            constraint["sorted_values"] = sorted(values)

    bins = 10
    protocol_attempt_scores: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_attempt_scores: Dict[int, List[float]] = defaultdict(list)

    projection = {
        "patient.patient_id": 1,
        "plan_id": 1,
        "attempt_number": 1,
        "standard_protocol": 1,
        "protocol.protocol_name": 1,
        "results": 1,
    }
    cursor = coll.find({}, projection)
    for doc in cursor:
        patient_id = doc.get("patient", {}).get("patient_id")
        plan_id = doc.get("plan_id")
        if (patient_id, plan_id) not in qualified_keys:
            continue
        attempt_number = doc.get("attempt_number")
        if attempt_number is None:
            continue
        attempt_idx = int(attempt_number)

        protocol_raw = _normalize_protocol_name(
            doc.get("standard_protocol"), doc.get("protocol", {}).get("protocol_name")
        )
        standard_name = _resolve_standard_protocol_name(
            protocol_raw, template_to_standard, standard_name_map
        )
        spec = protocol_specs_by_key.get(_normalize_protocol_key(standard_name))
        if not spec:
            continue

        total_weight = 0.0
        total_score = 0.0
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
            constraint = spec["constraint_map"].get(key)
            if not constraint:
                continue
            sorted_values = constraint.get("sorted_values") or []
            direction = constraint.get("direction")
            if not sorted_values or direction is None:
                continue
            numeric_value, _unit = _extract_numeric(result.get("achieved"))
            if numeric_value is None:
                continue
            percentile = _directional_percentile(
                numeric_value, sorted_values, direction
            )
            if percentile is None:
                continue
            weight = float(constraint.get("weight", 1.0))
            total_score += weight * percentile
            total_weight += weight

        if total_weight == 0:
            continue
        plan_score = total_score / total_weight
        attempt_count = plan_attempt_counts.get((patient_id, plan_id)) or 0
        if attempt_count <= 1:
            progress = 1.0
        else:
            progress = (attempt_idx - 1) / (attempt_count - 1)
        bin_idx = int(progress * bins)
        if bin_idx >= bins:
            bin_idx = bins - 1
        if bin_idx < 0:
            bin_idx = 0
        protocol_attempt_scores[standard_name][bin_idx].append(plan_score)
        all_attempt_scores[attempt_idx].append(plan_score)

    protocol_score_trends: List[Dict[str, Any]] = []
    for protocol_name, attempt_map in protocol_attempt_scores.items():
        attempt_stats = []
        for bin_idx in range(bins):
            values = attempt_map.get(bin_idx, [])
            attempt_stats.append(
                {
                    "attempt": bin_idx,
                    "label": f"{bin_idx * int(100 / bins)}-{(bin_idx + 1) * int(100 / bins)}%",
                    "p25": _percentile(values, 25.0) if values else None,
                    "p50": _percentile(values, 50.0) if values else None,
                    "p75": _percentile(values, 75.0) if values else None,
                    "count": len(values),
                }
            )
        plan_count = protocol_stats.get(protocol_name, {}).get("plan_count", 0)
        protocol_score_trends.append(
            {
                "name": protocol_name,
                "plan_count": plan_count,
                "attempt_stats": attempt_stats,
            }
        )

    protocol_score_trends.sort(
        key=lambda item: (item.get("plan_count", 0), item.get("name", "")), reverse=True
    )

    score_stats: List[Dict[str, Any]] = []
    all_scores: List[float] = []
    for attempt_number in sorted(all_attempt_scores.keys()):
        values = all_attempt_scores[attempt_number]
        if not values:
            continue
        score_stats.append(
            {
                "attempt": attempt_number,
                "p25": _percentile(values, 25.0),
                "p50": _percentile(values, 50.0),
                "p75": _percentile(values, 75.0),
                "count": len(values),
            }
        )
        all_scores.extend(values)

    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 1.0
    if score_min == score_max:
        score_max = score_min + 1.0

    return protocol_score_trends, score_stats, {"min": score_min, "max": score_max}


def compute_metrics(uri: str, db_name: str, collection_name: str) -> Dict[str, Any]:
    client = _mongo_client(uri)
    db = client[db_name]
    coll = db[collection_name]

    qualified_plans = _fetch_qualified_plans(coll, min_attempts=2)
    plan_attempt_counts: Dict[Tuple[Any, Any], int] = {}
    for doc in qualified_plans:
        patient_id = doc["_id"].get("patient_id")
        plan_id = doc["_id"].get("plan_id")
        if patient_id is None or plan_id is None:
            continue
        plan_attempt_counts[(patient_id, plan_id)] = int(doc.get("attempt_count", 0))
    template_to_standard, standard_name_map = _load_protocol_name_map(db)
    qualified_keys, attempt_distribution, protocol_stats, total_attempts, patient_ids = (
        _build_plan_index(qualified_plans, template_to_standard, standard_name_map)
    )

    (
        approval_counts,
        total_constraints,
        structures,
        _scores_by_attempt,
        evaluation_count,
        min_created,
        max_created,
    ) = _iter_qualified_evaluations(coll, qualified_keys)

    protocol_score_trends, score_stats, score_range = _compute_plan_score_stats(
        db,
        coll,
        qualified_keys,
        plan_attempt_counts,
        protocol_stats,
        template_to_standard,
        standard_name_map,
    )

    protocol_rows: List[Dict[str, Any]] = []
    for name, stats in protocol_stats.items():
        protocol_rows.append(
            {
                "name": name,
                "plan_count": stats["plan_count"],
                "patient_count": len(stats["patient_ids"]),
            }
        )
    protocol_rows.sort(key=lambda item: item["plan_count"], reverse=True)

    attempt_histogram = OrderedDict()
    overflow_attempts = 0
    for attempt_count, count in sorted(attempt_distribution.items()):
        if attempt_count > 10:
            overflow_attempts += count
            continue
        attempt_histogram[int(attempt_count)] = int(count)

    client.close()

    avg_attempts = round(total_attempts / len(qualified_plans), 2) if qualified_plans else 0.0
    avg_constraints = round(total_constraints / evaluation_count, 2) if evaluation_count else 0.0

    date_range = {
        "min": min_created.strftime("%Y-%m-%d") if min_created else "n/a",
        "max": max_created.strftime("%Y-%m-%d") if max_created else "n/a",
    }

    return {
        "generated_at": _utc_now_iso(),
        "totals": {
            "patients": len(patient_ids),
            "plans": len(qualified_plans),
            "evaluations": evaluation_count,
            "avg_attempts": avg_attempts,
            "total_constraints": total_constraints,
            "avg_constraints": avg_constraints,
            "unique_structures": len(structures),
        },
        "attempt_histogram": attempt_histogram,
        "attempt_overflow": overflow_attempts,
        "approval_distribution": OrderedDict(
            sorted(approval_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "score_stats": score_stats,
        "score_range": score_range,
        "protocol_score_trends": protocol_score_trends,
        "protocols": protocol_rows,
        "date_range": date_range,
    }


def compute_protocol_detail(
    uri: str,
    db_name: str,
    collection_name: str,
    protocol_name: str,
    selected_plan_idx: Optional[int] = None,
) -> Dict[str, Any]:
    protocol_name = protocol_name.strip()
    client = _mongo_client(uri)
    db = client[db_name]
    coll = db[collection_name]
    protocols_coll = db.get_collection("protocols")
    standard_coll = db.get_collection("standard_protocols")

    template_to_standard, standard_name_map = _load_protocol_name_map(db)
    resolved_name = _resolve_standard_protocol_name(
        protocol_name, template_to_standard, standard_name_map
    )
    standard_lookup_name = standard_name_map.get(_normalize_protocol_key(resolved_name), resolved_name)

    standard_doc = standard_coll.find_one({"protocol_name": standard_lookup_name})
    protocol_doc = None
    standard_id = None
    standard_name = standard_lookup_name or protocol_name
    if standard_doc:
        standard_name = standard_doc.get("protocol_name") or protocol_name
        standard_id = standard_doc.get("_id") or standard_doc.get("standard_id")
    else:
        protocol_doc = protocols_coll.find_one({"protocol_name": protocol_name})
        if not protocol_doc and resolved_name != protocol_name:
            protocol_doc = protocols_coll.find_one({"protocol_name": resolved_name})
        if not protocol_doc:
            protocol_doc = protocols_coll.find_one(
                {"protocol_name": {"$regex": f"^{re.escape(protocol_name)}$", "$options": "i"}}
            )
        if protocol_doc:
            standard_ref = protocol_doc.get("standard_ref") or {}
            standard_name = standard_ref.get("standard_name") or protocol_name
            standard_id = standard_ref.get("standard_id")
            if standard_id is not None:
                standard_doc = standard_coll.find_one({"_id": standard_id})
                if standard_doc and standard_doc.get("protocol_name"):
                    standard_name = standard_doc.get("protocol_name")

    related_protocols: List[Dict[str, Any]] = []
    if standard_id is not None:
        related_protocols = list(
            protocols_coll.find({"standard_ref.standard_id": standard_id})
        )
    if not related_protocols and standard_name:
        related_protocols = list(
            protocols_coll.find(
                {
                    "standard_ref.standard_name": {
                        "$regex": f"^{re.escape(standard_name)}$",
                        "$options": "i",
                    }
                }
            )
        )
    if not related_protocols and protocol_doc:
        related_protocols = [protocol_doc]

    related_template_names: List[str] = []
    for doc in related_protocols:
        name = doc.get("protocol_name")
        if name:
            related_template_names.append(name)
    if standard_name:
        related_template_names.append(standard_name)
    if protocol_name:
        related_template_names.append(protocol_name)
    if resolved_name and resolved_name not in related_template_names:
        related_template_names.append(resolved_name)
    related_template_names = list(OrderedDict.fromkeys(related_template_names))
    related_template_keys = [
        key for name in related_template_names if (key := _normalize_protocol_key(name))
    ]
    related_template_keys = list(OrderedDict.fromkeys(related_template_keys))

    constraint_source = "standard_protocols"
    canonical_constraints: List[Dict[str, Any]] = []
    if standard_doc:
        canonical_constraints = standard_doc.get("constraints") or []
    if not canonical_constraints:
        constraint_source = "protocols"
        primary_protocol = None
        for doc in related_protocols:
            standard_ref = doc.get("standard_ref") or {}
            if standard_ref.get("is_primary"):
                primary_protocol = doc
                break
        if not primary_protocol:
            for doc in related_protocols:
                if doc.get("protocol_name") == standard_name:
                    primary_protocol = doc
                    break
        if not primary_protocol and related_protocols:
            primary_protocol = related_protocols[0]
        if primary_protocol:
            canonical_constraints = primary_protocol.get("constraints") or []
    if not canonical_constraints:
        constraint_source = "unknown"

    alias_map = _load_structure_aliases(db)

    def clean_threshold(operator: Any, value: Any, unit: Any = None) -> Optional[Dict[str, Any]]:
        if operator is None and value is None and unit is None:
            return None
        return {"operator": operator, "value": value, "unit": unit}

    constraint_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for constraint in canonical_constraints:
        structure_raw = constraint.get("structure") or "Unknown structure"
        structure = _stringify(structure_raw) or "Unknown structure"
        if constraint_source == "protocols":
            metric = constraint.get("metric") or {}
            metric_display = _stringify(metric.get("display") or constraint.get("objective"))
            goal = constraint.get("goal") if isinstance(constraint.get("goal"), dict) else None
            variation = (
                constraint.get("variation")
                if isinstance(constraint.get("variation"), dict)
                else None
            )
            priority = constraint.get("priority")
            unit = metric.get("unit") if isinstance(metric, dict) else None
            if not unit and isinstance(goal, dict):
                unit = goal.get("unit")
            direction = _goal_direction(goal.get("operator") if isinstance(goal, dict) else None)
        else:
            metric_display = _stringify(constraint.get("objective"))
            goal = clean_threshold(
                constraint.get("goal_operator"),
                constraint.get("goal_value"),
            )
            variation = clean_threshold(
                constraint.get("variation_operator"),
                constraint.get("variation_value"),
            )
            priority = constraint.get("priority")
            unit = None
            direction = _goal_direction(constraint.get("goal_operator"))

        goal_label = _format_threshold(goal)
        variation_label = _format_threshold(variation)
        title_parts = [part for part in [metric_display, goal_label] if part]
        if variation_label:
            title_parts.append(f"Var {variation_label}")
        title = " | ".join(title_parts) or "Constraint"

        include_thresholds = constraint_source == "protocols"
        key = _constraint_key(
            structure,
            metric_display,
            goal,
            variation,
            priority,
            include_thresholds,
        )
        if key in constraint_map:
            existing = constraint_map[key]
            if goal_label:
                existing["goal_variants"].add(goal_label)
            if variation_label:
                existing["variation_variants"].add(variation_label)
            continue
        constraint_map[key] = {
            "structure": structure,
            "title": title,
            "priority": priority,
            "metric": metric_display,
            "goal": goal,
            "variation": variation,
            "unit": unit,
            "direction": direction,
            "weight": _priority_weight(priority),
            "values": [],
            "goal_variants": {goal_label} if goal_label else set(),
            "variation_variants": {variation_label} if variation_label else set(),
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
            "$addFields": {
                "protocol_normalized": {
                    "$cond": [
                        {
                            "$and": [
                                {"$ne": ["$standard_protocol", None]},
                                {"$ne": ["$standard_protocol", ""]},
                            ]
                        },
                        "$standard_protocol",
                        "$protocol_name",
                    ]
                }
            },
        },
        {
            "$addFields": {
                "protocol_key": {
                    "$toLower": {
                        "$trim": {"input": "$protocol_normalized"}
                    }
                }
            }
        },
        {
            "$match": {
                "attempt_count": {"$gte": 2},
                "last_is_approved": True,
                "protocol_key": {"$in": related_template_keys}
                if related_template_keys
                else _normalize_protocol_key(protocol_name),
            }
        },
        {
            "$project": {
                "patient_id": "$_id.patient_id",
                "plan_id": "$_id.plan_id",
                "last_results": 1,
            }
        },
    ]

    plan_count = 0
    matched_plans = 0
    matched_constraint_keys: set = set()
    plan_records: List[Dict[str, Any]] = []

    for doc in coll.aggregate(pipeline, allowDiskUse=True):
        plan_count += 1
        plan_records.append(
            {
                "patient_id": doc.get("patient_id"),
                "plan_id": doc.get("plan_id"),
                "results": doc.get("last_results", []) or [],
            }
        )
        plan_matched = False
        for result in doc.get("last_results", []) or []:
            structure_raw = (
                result.get("structure") or result.get("structure_tg263") or "Unknown"
            )
            structure_name = _stringify(structure_raw) or "Unknown"
            structure_name = _select_canonical_structure(
                structure_name, alias_map, canonical_by_key, canonical_tokens
            )

            metric = result.get("metric") or {}
            metric_display = _stringify(metric.get("display") or result.get("objective"))
            goal = result.get("goal") if isinstance(result.get("goal"), dict) else None
            variation = (
                result.get("variation")
                if isinstance(result.get("variation"), dict)
                else None
            )
            include_thresholds = constraint_source == "protocols"
            key = _constraint_key(
                structure_name,
                metric_display,
                goal,
                variation,
                result.get("priority"),
                include_thresholds,
            )
            constraint = constraint_map.get(key)
            if not constraint:
                continue

            achieved_value = result.get("achieved")
            numeric_value, unit = _extract_numeric(achieved_value)
            if numeric_value is not None:
                constraint["values"].append(numeric_value)
                if unit and not constraint.get("unit"):
                    constraint["unit"] = unit
                plan_matched = True
                matched_constraint_keys.add(key)
        if plan_matched:
            matched_plans += 1

    client.close()

    for constraint in constraint_map.values():
        values = constraint["values"]
        constraint["sorted_values"] = sorted(values)

    def compute_plan_score(
        results: List[Dict[str, Any]],
        include_details: bool = False,
    ) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        total_weight = 0.0
        total_score = 0.0
        details: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for result in results:
            structure_raw = (
                result.get("structure") or result.get("structure_tg263") or "Unknown"
            )
            structure_name = _stringify(structure_raw) or "Unknown"
            structure_name = _select_canonical_structure(
                structure_name, alias_map, canonical_by_key, canonical_tokens
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
                constraint_source == "protocols",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            constraint = constraint_map.get(key)
            if not constraint:
                continue
            sorted_values = constraint.get("sorted_values") or []
            direction = constraint.get("direction")
            if not sorted_values or direction is None:
                continue
            numeric_value, _unit = _extract_numeric(result.get("achieved"))
            if numeric_value is None:
                continue
            percentile = _directional_percentile(
                numeric_value, sorted_values, direction
            )
            if percentile is None:
                continue
            weight = float(constraint.get("weight", 1.0))
            total_score += weight * percentile
            total_weight += weight
            if include_details:
                goal_value = None
                goal = constraint.get("goal")
                if isinstance(goal, dict):
                    goal_value = goal.get("value")
                requested_percentile = None
                if goal_value is not None:
                    goal_numeric = _normalize_numeric(goal_value)
                    if goal_numeric is not None:
                        requested_percentile = _directional_percentile(
                            goal_numeric, sorted_values, direction
                        )
                details.append(
                    {
                        "structure": constraint.get("structure"),
                        "metric": constraint.get("metric"),
                        "percentile": percentile,
                        "requested": requested_percentile,
                        "priority": constraint.get("priority"),
                        "value": numeric_value,
                        "unit": constraint.get("unit"),
                    }
                )

        if total_weight == 0:
            return None, details
        return total_score / total_weight, details

    plan_records.sort(
        key=lambda record: (
            _stringify(record.get("patient_id")),
            _stringify(record.get("plan_id")),
        )
    )
    scored_plans: List[Dict[str, Any]] = []
    for record in plan_records:
        score, _details = compute_plan_score(record.get("results", []))
        if score is None:
            continue
        scored_plans.append(
            {
                "patient_id": record.get("patient_id"),
                "plan_id": record.get("plan_id"),
                "score": score,
                "results": record.get("results", []),
            }
        )

    distribution = [
        {"index": idx, "score": plan["score"]}
        for idx, plan in enumerate(scored_plans)
    ]
    scores_only = [item["score"] for item in distribution]
    distribution_stats = {
        "count": len(scores_only),
        "min": min(scores_only) if scores_only else None,
        "max": max(scores_only) if scores_only else None,
        "median": _percentile(scores_only, 50.0) if scores_only else None,
    }

    selected_plan = None
    if selected_plan_idx is not None:
        if 0 <= selected_plan_idx < len(scored_plans):
            chosen = scored_plans[selected_plan_idx]
            score, details = compute_plan_score(
                chosen.get("results", []), include_details=True
            )
            if score is not None:
                details.sort(
                    key=lambda item: (
                        _stringify(item.get("structure")),
                        _stringify(item.get("metric")),
                    )
                )
                selected_plan = {
                    "index": selected_plan_idx,
                    "score": score,
                    "daisy_points": details,
                }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for constraint in constraint_map.values():
        values = constraint["values"]
        constraint["count"] = len(values)
        constraint["min"] = min(values) if values else None
        constraint["max"] = max(values) if values else None
        constraint["median"] = _percentile(values, 50.0) if values else None
        grouped[constraint["structure"]].append(constraint)

    def priority_sort_key(priority_value: Any) -> Tuple[int, Any]:
        if priority_value is None:
            return (3, "")
        if isinstance(priority_value, str):
            normalized = priority_value.strip().lower()
            if normalized in {"high", "h"}:
                return (0, 1)
            if normalized in {"medium", "m"}:
                return (1, 2)
            if normalized in {"low", "l"}:
                return (2, 3)
            try:
                return (0, int(normalized))
            except ValueError:
                return (2, normalized)
        try:
            return (0, int(priority_value))
        except (TypeError, ValueError):
            return (2, str(priority_value))

    constraint_groups: List[Dict[str, Any]] = []
    for structure in sorted(grouped.keys()):
        constraints = sorted(
            grouped[structure],
            key=lambda item: (priority_sort_key(item["priority"]), item["title"]),
        )
        constraint_groups.append(
            {"structure": structure, "constraints": constraints}
        )

    return {
        "protocol_name": standard_name or protocol_name,
        "plan_count": plan_count,
        "matched_plans": matched_plans,
        "constraint_groups": constraint_groups,
        "constraint_count": len(constraint_map),
        "matched_constraint_count": len(matched_constraint_keys),
        "related_templates": related_template_names,
        "constraint_source": constraint_source,
        "plan_score_distribution": distribution,
        "plan_score_stats": distribution_stats,
        "selected_plan": selected_plan,
    }


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Optional[float], scale: float = 1.0, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * scale:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_signed_percent(value: Optional[float], scale: float = 1.0, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        scaled = float(value) * scale
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if scaled >= 0 else ""
    return f"{sign}{scaled:.{decimals}f}%"


def _delta_value(model: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if model is None or baseline is None:
        return None
    return model - baseline


def _render_histogram(data: OrderedDict, overflow_count: int) -> str:
    if not data:
        return '<div class="empty">No data</div>'

    max_count = max(data.values()) or 1
    bars = []
    for attempt, count in data.items():
        height_pct = (count / max_count) * 100
        bars.append(
            f"""
            <div class="hist-bar" title="Attempt {attempt}: {count}">
              <div class="hist-fill" style="height: {height_pct:.1f}%"></div>
              <div class="hist-label">{attempt}</div>
              <div class="hist-count">{_format_int(count)}</div>
            </div>
            """
        )

    note = ""
    if overflow_count:
        note = f'<div class="hist-note">Not shown (>10 attempts): {_format_int(overflow_count)} plans</div>'

    return f"""
    <div class="histogram">
      {"".join(bars)}
    </div>
    {note}
    """


def _render_bar_rows(data: OrderedDict) -> str:
    if not data:
        return '<div class="empty">No data</div>'

    max_count = max(data.values()) or 1
    rows = []
    for label, count in data.items():
        width_pct = (count / max_count) * 100
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(str(label))}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {width_pct:.1f}%"></div>
              </div>
              <div class="bar-value">{_format_int(count)}</div>
            </div>
            """
        )
    return "\n".join(rows)


def _render_score_plot(score_stats: List[Dict[str, Any]], score_range: Dict[str, float]) -> str:
    if not score_stats:
        return '<div class="empty">No plan scores available</div>'

    display_stats = [stat for stat in score_stats if stat["attempt"] <= 10]
    if not display_stats:
        return '<div class="empty">No attempts in the first 10 iterations</div>'

    width = 620
    height = 180
    pad_x = 30
    pad_top = 20
    pad_bottom = 30
    chart_height = height - pad_top - pad_bottom
    chart_width = width - (2 * pad_x)

    score_min = score_range.get("min", 0.0)
    score_max = score_range.get("max", 1.0)
    if score_min == score_max:
        score_max = score_min + 1.0

    def y_for_score(value: float) -> float:
        return pad_top + ((score_max - value) / (score_max - score_min)) * chart_height

    x_step = chart_width / max(len(display_stats), 1)

    elements = []
    axis_y = pad_top + chart_height
    elements.append(
        f'<line x1="{pad_x}" y1="{axis_y:.1f}" x2="{width - pad_x}" y2="{axis_y:.1f}" stroke="#dfe1df" stroke-width="1" />'
    )
    max_label = _format_number(score_max)
    min_label = _format_number(score_min)
    if max_label != "n/a":
        max_label = f"{max_label}%"
    if min_label != "n/a":
        min_label = f"{min_label}%"
    elements.append(
        f'<text x="{pad_x - 6}" y="{pad_top + 6:.1f}" text-anchor="end" font-size="10" fill="#8e9089">{max_label}</text>'
    )
    elements.append(
        f'<text x="{pad_x - 6}" y="{axis_y:.1f}" text-anchor="end" font-size="10" fill="#8e9089">{min_label}</text>'
    )

    for idx, stat in enumerate(display_stats):
        attempt = stat["attempt"]
        p25 = stat["p25"]
        p50 = stat["p50"]
        p75 = stat["p75"]
        count = stat["count"]
        if p25 is None or p50 is None or p75 is None:
            continue

        x_center = pad_x + (idx * x_step) + (x_step / 2)
        y_p25 = y_for_score(p25)
        y_p75 = y_for_score(p75)
        y_med = y_for_score(p50)

        elements.append(
            f'<line x1="{x_center:.1f}" y1="{y_p75:.1f}" x2="{x_center:.1f}" y2="{y_p25:.1f}" stroke="#59b7df" stroke-width="6" stroke-linecap="round" />'
        )
        elements.append(
            f'<circle cx="{x_center:.1f}" cy="{y_med:.1f}" r="4" fill="#152456" />'
        )
        elements.append(
            f'<text x="{x_center:.1f}" y="{axis_y + 16:.1f}" text-anchor="middle" font-size="10" fill="#4e5259">{attempt}</text>'
        )
        elements.append(
            f'<text x="{x_center:.1f}" y="{axis_y + 28:.1f}" text-anchor="middle" font-size="9" fill="#8e9089">n={count}</text>'
        )

    return f"""
    <svg class="score-plot" viewBox="0 0 {width} {height}" role="img" aria-label="Plan score distribution by attempt">
      {"".join(elements)}
    </svg>
    """


def _render_protocol_score_sparkline(attempt_stats: List[Dict[str, Any]]) -> str:
    if not attempt_stats:
        return '<div class="empty">No scores</div>'

    width = 280
    height = 120
    pad_x = 44
    pad_top = 12
    pad_bottom = 22
    chart_height = height - pad_top - pad_bottom
    chart_width = width - (2 * pad_x)

    values = []
    for stat in attempt_stats:
        for key in ("p25", "p50", "p75"):
            val = stat.get(key)
            if val is not None:
                values.append(float(val))
    if not values:
        return '<div class="empty">No scores</div>'

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0

    def y_for_score(value: float) -> float:
        return pad_top + ((max_val - value) / (max_val - min_val)) * chart_height

    n = len(attempt_stats)
    if n == 1:
        x_positions = [pad_x + chart_width / 2]
    else:
        step = chart_width / (n - 1)
        x_positions = [pad_x + idx * step for idx in range(n)]

    median_points = []
    upper_points = []
    lower_points = []
    for idx, stat in enumerate(attempt_stats):
        p50 = stat.get("p50")
        p75 = stat.get("p75")
        p25 = stat.get("p25")
        if p50 is None:
            continue
        x = x_positions[idx]
        median_points.append((x, y_for_score(p50)))
        if p75 is not None and p25 is not None:
            upper_points.append((x, y_for_score(p75)))
            lower_points.append((x, y_for_score(p25)))

    median_path = " ".join(
        f"{'M' if idx == 0 else 'L'}{x:.1f},{y:.1f}"
        for idx, (x, y) in enumerate(median_points)
    )

    band_path = ""
    if upper_points and lower_points and len(upper_points) == len(lower_points):
        polygon_points = upper_points + list(reversed(lower_points))
        band_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon_points)

    band_element = (
        f'<polygon points="{band_path}" fill="rgba(89, 183, 223, 0.2)" />'
        if band_path
        else ""
    )
    median_element = (
        f'<path d="{median_path}" fill="none" stroke="#152456" stroke-width="2" />'
        if median_path
        else ""
    )

    max_label = _format_number(max_val)
    min_label = _format_number(min_val)
    if max_label != "n/a":
        max_label = f"{max_label}%"
    if min_label != "n/a":
        min_label = f"{min_label}%"

    return f"""
    <svg viewBox="0 0 {width} {height}" class="score-sparkline" role="img" aria-label="Plan score trend">
      <line x1="{pad_x}" y1="{pad_top + chart_height:.1f}" x2="{width - pad_x}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" stroke-width="1" />
      <line x1="{pad_x}" y1="{pad_top:.1f}" x2="{pad_x}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" stroke-width="1" />
      <text x="{pad_x - 6}" y="{pad_top + 6:.1f}" text-anchor="end" font-size="9" fill="#8e9089">{max_label}</text>
      <text x="{pad_x - 6}" y="{pad_top + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">{min_label}</text>
      <text x="{pad_x:.1f}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">0%</text>
      <text x="{width - pad_x:.1f}" y="{height - 6}" text-anchor="end" font-size="9" fill="#8e9089">100%</text>
      {band_element}
      {median_element}
    </svg>
    """


def _render_protocol_score_grid(
    protocol_score_trends: List[Dict[str, Any]],
    min_count: int,
) -> str:
    filtered = [
        item for item in protocol_score_trends if item.get("plan_count", 0) >= min_count
    ]
    if not filtered:
        return '<div class="empty">No protocols meet the minimum plan count for scoring.</div>'

    cards = []
    for item in filtered:
        name = html.escape(item.get("name", "Unknown"))
        plan_count = _format_int(int(item.get("plan_count", 0)))
        attempt_stats = item.get("attempt_stats", [])
        chart = _render_protocol_score_sparkline(attempt_stats)
        cards.append(
            f"""
            <div class="score-card">
              <h4>{name}</h4>
              <div class="score-meta">Qualified plans: {plan_count}</div>
              {chart}
            </div>
            """
        )

    return f"""
    <div class="score-grid">
      {"".join(cards)}
    </div>
    """


def _jitter(index: int, spread: float) -> float:
    seed = (index * 9301 + 49297) % 233280
    return ((seed / 233280.0) - 0.5) * 2 * spread


def _render_plan_score_distribution(
    distribution: List[Dict[str, Any]],
    stats: Dict[str, Any],
    protocol_name: str,
    min_protocol_plans: int,
    selected_plan: Optional[Dict[str, Any]],
) -> str:
    if not distribution:
        return '<div class="empty">No plan scores available for this protocol.</div>'

    width = 280
    height = 220
    pad_left = 40
    pad_right = 16
    pad_top = 16
    pad_bottom = 28
    chart_height = height - pad_top - pad_bottom
    chart_width = width - pad_left - pad_right
    center_x = pad_left + (chart_width / 2)

    scale_min = stats.get("min", 0.0)
    scale_max = stats.get("max", 100.0)
    if scale_min is None or scale_max is None:
        scale_min, scale_max = 0.0, 100.0
    if scale_min == scale_max:
        scale_min -= 1.0
        scale_max += 1.0

    sorted_items = sorted(distribution, key=lambda item: item["score"])
    points = []
    path_points = []
    for rank, item in enumerate(sorted_items):
        idx = item["index"]
        score = item["score"]
        if len(sorted_items) > 1:
            x = pad_left + (rank / (len(sorted_items) - 1)) * chart_width
        else:
            x = center_x
        y = pad_top + ((scale_max - score) / (scale_max - scale_min)) * chart_height
        radius = 3 if selected_plan and idx == selected_plan.get("index") else 2
        fill = "#152456" if selected_plan and idx == selected_plan.get("index") else "#59b7df"
        link = (
            f"/?protocol={quote(protocol_name)}&plan_idx={idx}"
            f"&min_protocol_plans={min_protocol_plans}#population"
        )
        path_points.append((x, y))
        points.append(
            f"""
            <a href="{link}">
              <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}">
                <title>Rank {rank + 1} — Score {score:.1f}</title>
              </circle>
            </a>
            """
        )

    count = stats.get("count", 0)
    median = _format_number(stats.get("median"))
    if median != "n/a":
        median = f"{median}%"
    min_val = _format_number(scale_min)
    max_val = _format_number(scale_max)
    if min_val != "n/a":
        min_val = f"{min_val}%"
    if max_val != "n/a":
        max_val = f"{max_val}%"

    path = ""
    if path_points:
        path = " ".join(
            f"{'M' if idx == 0 else 'L'}{x:.1f},{y:.1f}"
            for idx, (x, y) in enumerate(path_points)
        )
        path = f'<path d="{path}" fill="none" stroke="rgba(89, 183, 223, 0.45)" stroke-width="2" />'

    return f"""
    <div class="score-distribution">
      <div class="score-distribution-meta">
        <span>n={_format_int(int(count))}</span>
        <span>median {median}</span>
        <span>min {min_val}</span>
        <span>max {max_val}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Plan score distribution">
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" stroke-width="1" />
        <line x1="{pad_left}" y1="{pad_top + chart_height:.1f}" x2="{width - pad_right}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" stroke-width="1" />
        <text x="{pad_left - 6}" y="{pad_top + 6}" text-anchor="end" font-size="9" fill="#8e9089">{max_val}</text>
        <text x="{pad_left - 6}" y="{pad_top + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">{min_val}</text>
        <text x="{pad_left}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">0% ranked</text>
        <text x="{pad_left + 70}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">Ranked plans (low to high)</text>
        <text x="{width - pad_right}" y="{height - 6}" text-anchor="end" font-size="9" fill="#8e9089">100%</text>
        {path}
        {"".join(points)}
      </svg>
      <div class="hint">Click a dot to view the plan scorecard (scores sorted low to high).</div>
    </div>
    """


def _percentile_color(percentile: float) -> str:
    if percentile < 20:
        return "#dc2626"
    if percentile < 40:
        return "#f87171"
    if percentile < 60:
        return "#fde047"
    if percentile < 80:
        return "#bbf7d0"
    if percentile < 90:
        return "#4ade80"
    if percentile < 95:
        return "#16a34a"
    return "#065f46"


def _priority_opacity(priority: Any) -> float:
    try:
        value = int(float(priority))
    except (TypeError, ValueError):
        value = None
    if value == 1:
        return 1.0
    if value == 2:
        return 0.7
    if value == 3:
        return 0.4
    return 0.4


def _plan_score_color(score: float) -> str:
    if score < 35:
        return "#dc2626"
    if score < 75:
        return "#d97706"
    return "#16a34a"


def _estimate_text_width(text: str, font_size: int) -> float:
    return (len(text) * font_size * 0.6) + 18


def _arc_path(
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    center_x: float,
    center_y: float,
) -> str:
    def polar(radius: float, angle: float) -> Tuple[float, float]:
        return (center_x + radius * math.sin(angle), center_y - radius * math.cos(angle))

    sx, sy = polar(outer_radius, start_angle)
    ex, ey = polar(outer_radius, end_angle)
    sx2, sy2 = polar(inner_radius, end_angle)
    ex2, ey2 = polar(inner_radius, start_angle)
    large_arc = 1 if (end_angle - start_angle) > math.pi else 0
    return (
        f"M {sx:.1f},{sy:.1f} "
        f"A {outer_radius:.1f},{outer_radius:.1f} 0 {large_arc} 1 {ex:.1f},{ey:.1f} "
        f"L {sx2:.1f},{sy2:.1f} "
        f"A {inner_radius:.1f},{inner_radius:.1f} 0 {large_arc} 0 {ex2:.1f},{ey2:.1f} Z"
    )


def _render_daisy_plot(selected_plan: Optional[Dict[str, Any]]) -> str:
    if not selected_plan:
        return '<div class="empty">Select a plan from the distribution to view its scorecard.</div>'

    points = selected_plan.get("daisy_points", [])
    if not points:
        return '<div class="empty">No constraint percentiles available for this plan.</div>'

    metrics = []
    for item in points:
        structure = _stringify(item.get("structure")).strip()
        metric = _stringify(item.get("metric")).strip()
        label = f"{structure} {metric}".strip()
        metrics.append(
            {
                "label": label,
                "percentile": float(item.get("percentile", 0.0)),
                "requested": item.get("requested"),
                "priority": item.get("priority"),
                "value": item.get("value"),
                "unit": item.get("unit"),
            }
        )
    def _priority_sort_value(item: Dict[str, Any]) -> Tuple[float, str]:
        value = _normalize_numeric(item.get("priority"))
        return (value if value is not None else 99.0, item["label"])

    metrics.sort(key=_priority_sort_value)

    width = 520
    height = 520
    cx = width / 2
    cy = height / 2
    inner_radius = 90
    outer_radius = 180

    rings = []
    for pct in (25, 50, 75, 100):
        radius = inner_radius + (outer_radius - inner_radius) * (pct / 100.0)
        rings.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius:.1f}" fill="none" stroke="#ddd" stroke-width="1" stroke-dasharray="2,2" />'
        )
        rings.append(
            f'<text x="{cx + 6:.1f}" y="{cy - radius + 6:.1f}" font-size="10" fill="#999">{pct}%</text>'
        )

    count = max(len(metrics), 1)
    angle_step = (2 * math.pi) / count
    start_angle = -math.pi / 4
    bar_width = angle_step * 0.7

    sectors = []
    labels = []
    dots = []
    tooltips = []
    for idx, metric in enumerate(metrics):
        angle = start_angle + idx * angle_step
        percentile = metric["percentile"]
        bar_height = (outer_radius - inner_radius) * (percentile / 100.0)
        arc = _arc_path(
            inner_radius,
            inner_radius + bar_height,
            angle - bar_width / 2,
            angle + bar_width / 2,
            cx,
            cy,
        )
        base_color = _percentile_color(percentile)
        opacity = _priority_opacity(metric.get("priority"))
        sectors.append(
            f'<path d="{arc}" fill="{base_color}" opacity="{opacity:.2f}" stroke="#000000" stroke-width="1" />'
        )

        dot_radius = 5
        dot_x = cx + (inner_radius + bar_height) * math.sin(angle)
        dot_y = cy - (inner_radius + bar_height) * math.cos(angle)
        dots.append(
            f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="{dot_radius}" fill="black" />'
        )

        requested = metric.get("requested")
        if requested is not None:
            try:
                requested_val = float(requested)
            except (TypeError, ValueError):
                requested_val = None
            if requested_val is not None:
                req_height = (outer_radius - inner_radius) * (requested_val / 100.0)
                req_x = cx + (inner_radius + req_height) * math.sin(angle)
                req_y = cy - (inner_radius + req_height) * math.cos(angle)
                dots.append(
                    f'<circle cx="{req_x:.1f}" cy="{req_y:.1f}" r="{dot_radius}" fill="#888" opacity="0.7" />'
                )

        label_radius = outer_radius + 18
        label_x = cx + label_radius * math.sin(angle)
        label_y = cy - label_radius * math.cos(angle)
        font_size = 11
        label_text = metric["label"]
        label_width = _estimate_text_width(label_text, font_size)
        label_height = 20
        labels.append(
            f'<g transform="translate({label_x:.1f},{label_y:.1f})">'
            f'<rect x="{-(label_width / 2):.1f}" y="{-(label_height / 2):.1f}" '
            f'width="{label_width:.1f}" height="{label_height:.1f}" fill="white" opacity="0.9" '
            f'rx="4" stroke="#ccc" stroke-width="1" />'
            f'<text text-anchor="middle" dominant-baseline="middle" font-size="{font_size}" fill="black">{html.escape(label_text)}</text>'
            "</g>"
        )

        value_text = _format_number(metric.get("value"))
        unit = metric.get("unit") or ""
        if unit:
            unit_suffix = unit if unit.startswith("%") else f" {unit}"
        else:
            unit_suffix = ""
        tooltip_text = f"{percentile:.1f}%, {value_text}{unit_suffix}"
        tip_x = cx + (inner_radius + bar_height) * math.sin(angle)
        tip_y = cy - (inner_radius + bar_height) * math.cos(angle)
        tooltip_width = _estimate_text_width(tooltip_text, 10)
        tooltips.append(
            f'<g transform="translate({tip_x:.1f},{tip_y:.1f})">'
            f'<rect x="{-(tooltip_width / 2):.1f}" y="-12" width="{tooltip_width:.1f}" height="22" '
            f'fill="white" opacity="0.9" rx="4" stroke="#ccc" stroke-width="1" />'
            f'<text text-anchor="middle" dominant-baseline="middle" font-size="10" fill="black">{html.escape(tooltip_text)}</text>'
            "</g>"
        )

    plan_score = float(selected_plan.get("score") or 0.0)
    plan_score_text = _format_number(plan_score)
    plan_score_color = _plan_score_color(plan_score)
    plan_label = f"Plan #{int(selected_plan.get('index', 0)) + 1}"

    return f"""
    <div class="scorecard">
      <div class="scorecard-header">
        <h4>{html.escape(plan_label)}</h4>
        <div class="scorecard-score">Plan score {plan_score_text}%</div>
      </div>
      <svg class="daisy-plot" viewBox="0 0 {width} {height}" role="img" aria-label="Daisy plot of constraint percentiles">
        {"".join(rings)}
        {"".join(sectors)}
        {"".join(dots)}
        {"".join(tooltips)}
        {"".join(labels)}
        <text x="{cx}" y="{cy}" text-anchor="middle" font-size="36" font-weight="bold" fill="{plan_score_color}">{plan_score_text}%</text>
        <text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="11" fill="#8e9089">Plan score</text>
      </svg>
      <div class="scorecard-legend">
        <span><span class="legend-dot achieved"></span> Achieved</span>
        <span><span class="legend-dot requested"></span> Requested</span>
      </div>
      <div class="scorecard-legend">
        <span><span class="legend-swatch p1"></span> Priority 1</span>
        <span><span class="legend-swatch p2"></span> Priority 2</span>
        <span><span class="legend-swatch p3"></span> Priority 3</span>
      </div>
    </div>
    """


def _render_violin(values: List[float]) -> str:
    if not values:
        return '<div class="violin-empty">No numeric values</div>'

    width = 140
    height = 120
    pad = 8
    inner_height = height - (2 * pad)
    center_x = width / 2
    max_half_width = width * 0.35

    if len(values) == 1:
        y = height / 2
        return f"""
        <svg class="violin-plot" viewBox="0 0 {width} {height}" role="img" aria-label="Single value distribution">
          <line x1="{center_x:.1f}" y1="{pad}" x2="{center_x:.1f}" y2="{height - pad}" stroke="#59b7df" stroke-width="4" />
          <circle cx="{center_x:.1f}" cy="{y:.1f}" r="4" fill="#152456" />
        </svg>
        """

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        y = height / 2
        return f"""
        <svg class="violin-plot" viewBox="0 0 {width} {height}" role="img" aria-label="Single value distribution">
          <line x1="{center_x:.1f}" y1="{pad}" x2="{center_x:.1f}" y2="{height - pad}" stroke="#59b7df" stroke-width="4" />
          <circle cx="{center_x:.1f}" cy="{y:.1f}" r="4" fill="#152456" />
        </svg>
        """

    bins = min(24, max(10, int(len(values) / 4)))
    step = (max_val - min_val) / bins
    counts = [0] * bins
    for value in values:
        idx = int((value - min_val) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    smoothed = []
    for idx, count in enumerate(counts):
        left = counts[idx - 1] if idx > 0 else count
        right = counts[idx + 1] if idx < bins - 1 else count
        smoothed.append((count * 0.5) + (left * 0.25) + (right * 0.25))

    max_density = max(smoothed) if smoothed else 1
    if max_density == 0:
        max_density = 1

    right_points = []
    left_points = []
    for idx, density in enumerate(smoothed):
        center_val = min_val + (idx + 0.5) * step
        y = pad + (1 - (center_val - min_val) / (max_val - min_val)) * inner_height
        half_width = (density / max_density) * max_half_width
        right_points.append((center_x + half_width, y))
        left_points.append((center_x - half_width, y))

    polygon_points = right_points + list(reversed(left_points))
    points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon_points)
    median_val = _percentile(values, 50.0)
    median_y = (
        pad + (1 - (median_val - min_val) / (max_val - min_val)) * inner_height
        if median_val is not None
        else height / 2
    )

    return f"""
    <svg class="violin-plot" viewBox="0 0 {width} {height}" role="img" aria-label="Distribution violin plot">
      <polygon points="{points_str}" fill="rgba(89, 183, 223, 0.35)" stroke="#59b7df" stroke-width="1" />
      <line x1="{center_x - max_half_width:.1f}" y1="{median_y:.1f}" x2="{center_x + max_half_width:.1f}" y2="{median_y:.1f}" stroke="#152456" stroke-width="2" />
    </svg>
    """


def _render_protocol_detail(
    detail: Optional[Dict[str, Any]],
    min_protocol_plans: int,
) -> str:
    if not detail:
        return ""

    protocol_name_raw = detail.get("protocol_name", "")
    protocol_name = html.escape(protocol_name_raw)
    plan_count = detail.get("plan_count", 0)
    matched_plans = detail.get("matched_plans", 0)
    constraint_count = detail.get("constraint_count", 0)
    matched_constraint_count = detail.get("matched_constraint_count", 0)
    related_templates = detail.get("related_templates", [])
    constraint_source = detail.get("constraint_source", "unknown")
    groups = detail.get("constraint_groups", [])
    distribution = detail.get("plan_score_distribution", [])
    distribution_stats = detail.get("plan_score_stats", {})
    selected_plan = detail.get("selected_plan")

    if plan_count == 0:
        return """
        <div class="protocol-detail">
          <div class="empty">No qualified plans found for this protocol.</div>
        </div>
        """

    group_blocks = []
    numeric_found = False
    for idx, group in enumerate(groups):
        structure = html.escape(group.get("structure", "Unknown"))
        constraints = group.get("constraints", [])
        cards = []
        for constraint in constraints:
            title = html.escape(constraint.get("title", "Constraint"))
            priority = constraint.get("priority")
            count = constraint.get("count", 0)
            min_val = _format_number(constraint.get("min"))
            max_val = _format_number(constraint.get("max"))
            median_val = _format_number(constraint.get("median"))
            values = constraint.get("values", [])
            unit = constraint.get("unit")
            if values:
                numeric_found = True

            meta_bits = [f"n={count}"]
            if priority not in (None, ""):
                meta_bits.insert(0, f"Priority {priority}")
            goal_variants = constraint.get("goal_variants") or set()
            variation_variants = constraint.get("variation_variants") or set()
            if len(goal_variants) > 1:
                meta_bits.append(f"goals {len(goal_variants)}")
            if len(variation_variants) > 1:
                meta_bits.append(f"vars {len(variation_variants)}")
            unit_suffix = f" {unit}" if unit else ""
            meta_bits.append(f"median {median_val}{unit_suffix}")
            meta_bits.append(f"range {min_val}-{max_val}{unit_suffix}")
            meta = html.escape(" | ".join(meta_bits))

            cards.append(
                f"""
                <div class="constraint-card">
                  <div class="constraint-title">{title}</div>
                  <div class="constraint-meta">{meta}</div>
                  {_render_violin(values)}
                </div>
                """
            )

        open_attr = " open" if idx < 2 else ""
        group_blocks.append(
            f"""
            <details class="structure-group"{open_attr}>
              <summary>{structure} ({len(constraints)} constraints)</summary>
              <div class="constraint-grid">
                {"".join(cards)}
              </div>
            </details>
            """
        )

    numeric_note = ""
    if not numeric_found:
        numeric_note = (
            '<p class="hint">No numeric achieved values detected. '
            'Expected results.achieved.value to be numeric for plotting.</p>'
        )

    source_label = {
        "protocols": "Protocol template constraints",
        "standard_protocols": "Standard protocol constraints",
        "unknown": "Unknown constraint source",
    }.get(constraint_source, "Constraint source")

    matching_note = ""
    if constraint_source == "standard_protocols":
        matching_note = (
            '<p class="hint">Matching uses structure + metric + priority; '
            "goal/variation differences across templates are shown but not required. "
            "Structure aliases use custom aliases plus token matching.</p>"
        )

    related_templates_html = ""
    if related_templates:
        items = "".join(
            f"<li>{html.escape(name)}</li>" for name in related_templates if name
        )
        related_templates_html = f"""
        <details class="related-templates">
          <summary>Related templates ({len(related_templates)})</summary>
          <ul>{items}</ul>
        </details>
        """

    distribution_html = _render_plan_score_distribution(
        distribution,
        distribution_stats,
        protocol_name_raw,
        min_protocol_plans,
        selected_plan,
    )
    daisy_html = _render_daisy_plot(selected_plan)

    return f"""
    <div class="protocol-detail">
      <div class="protocol-detail-header">
        <h3>{protocol_name}</h3>
        <p>Qualified plans: {_format_int(int(plan_count))} | Matched plans: {_format_int(int(matched_plans))}</p>
        <p>Canonical constraints: {_format_int(int(constraint_count))} | Constraints with data: {_format_int(int(matched_constraint_count))}</p>
        <p>{source_label}. Values shown: results.achieved.value from final approved evaluations.</p>
        {related_templates_html}
        {matching_note}
        {numeric_note}
      </div>
      <div class="scorecard-grid">
        <div>
          <h4>Final Approved Plan Scores</h4>
          {distribution_html}
        </div>
        <div>
          <h4>Plan Scorecard</h4>
          {daisy_html}
        </div>
      </div>
      {"".join(group_blocks)}
    </div>
    """


def _render_protocol_table(
    protocols: List[Dict[str, Any]],
    min_count: int,
    selected_protocol: Optional[str],
) -> str:
    filtered = [p for p in protocols if p["plan_count"] >= min_count]
    if not filtered:
        return '<div class="empty">No protocols meet the minimum plan count yet.</div>'

    rows = []
    for item in filtered:
        name = item["name"]
        name_safe = html.escape(name)
        plan_count = _format_int(item["plan_count"])
        patient_count = _format_int(item["patient_count"])
        link = f"/?protocol={quote(name)}&min_protocol_plans={min_count}#population"
        selected_attr = " class=\"selected\"" if selected_protocol == name else ""
        rows.append(
            f"""
            <tr{selected_attr}>
              <td><a class="protocol-link" href="{link}">{name_safe}</a></td>
              <td>{plan_count}</td>
              <td>{patient_count}</td>
            </tr>
            """
        )

    return f"""
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Standard Protocol</th>
          <th>Qualified Plans</th>
          <th>Patients</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _build_bin_counts(
    values: List[float],
    bins: List[Tuple[float, float]],
    labels: List[str],
) -> Tuple[OrderedDict, int]:
    counts = OrderedDict((label, 0) for label in labels)
    overflow = 0
    for value in values:
        placed = False
        for idx, (low, high) in enumerate(bins):
            if value < low:
                continue
            if value < high or (idx == len(bins) - 1 and value <= high):
                counts[labels[idx]] += 1
                placed = True
                break
        if not placed:
            overflow += 1
    return counts, overflow


def _render_phase2_protocol_table(protocol_rows: List[Dict[str, Any]]) -> str:
    if not protocol_rows:
        return '<div class="empty">No Phase 2 protocol data found.</div>'

    display_rows = protocol_rows[:30]
    rows = []
    for item in display_rows:
        name = html.escape(item.get("protocol_name", "Unknown"))
        plans = _format_int(int(item.get("plan_count", 0)))
        attempts = _format_int(int(item.get("attempt_count", 0)))
        stop_rate = _format_percent(item.get("stop_rate"), scale=100, decimals=1)
        median_score = _format_percent(item.get("median_score"), scale=1.0, decimals=1)
        median_coverage = _format_percent(item.get("median_coverage"), scale=100, decimals=1)
        rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{plans}</td>
              <td>{attempts}</td>
              <td>{stop_rate}</td>
              <td>{median_score}</td>
              <td>{median_coverage}</td>
            </tr>
            """
        )

    note = ""
    if len(protocol_rows) > len(display_rows):
        note = f"<div class=\"hint\">Showing top {len(display_rows)} protocols by plan count.</div>"

    return f"""
    {note}
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Protocol</th>
          <th>Plans</th>
          <th>Attempts</th>
          <th>Stop Rate</th>
          <th>Median Score</th>
          <th>Median Coverage</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _render_model_protocol_table(
    records: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    scale: Optional[float] = None,
    value_suffix: str = "",
    top_n: int = 20,
) -> str:
    filtered = [
        record
        for record in records
        if record.get("metrics", {}).get(metric_key) is not None
    ]
    if not filtered:
        return '<div class="empty">No metrics available.</div>'

    filtered.sort(
        key=lambda item: item.get("metrics", {}).get(metric_key) or 0.0, reverse=True
    )
    display_rows = filtered[:top_n]
    rows = []
    for item in display_rows:
        name = html.escape(item.get("protocol_name", "Unknown"))
        plan_count = _format_int(int(item.get("plan_count", 0)))
        metric_value = item.get("metrics", {}).get(metric_key)
        if scale is not None:
            metric_display = _format_percent(metric_value, scale=scale, decimals=1)
        else:
            metric_display = _format_number(metric_value)
        if metric_display != "n/a" and value_suffix:
            metric_display = f"{metric_display}{value_suffix}"
        rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{plan_count}</td>
              <td>{metric_display}</td>
            </tr>
            """
        )

    return f"""
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Protocol</th>
          <th>Plans</th>
          <th>{metric_label}</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _select_best_worst(
    records: List[Dict[str, Any]],
    metric_key: str,
    top_n: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered = [
        record
        for record in records
        if record.get("metrics", {}).get(metric_key) is not None
    ]
    filtered.sort(
        key=lambda item: item.get("metrics", {}).get(metric_key) or 0.0, reverse=True
    )
    if not filtered:
        return [], []
    top = filtered[:top_n]
    bottom = filtered[-top_n:] if len(filtered) >= top_n else filtered
    return top, bottom


def _render_metric_table(
    records: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    scale: Optional[float] = None,
) -> str:
    if not records:
        return '<div class="empty">No protocols available.</div>'
    rows = []
    for record in records:
        name = html.escape(str(record.get("protocol_name", "Unknown")))
        plans = _format_int(int(record.get("plan_count", 0)))
        metric_value = record.get("metrics", {}).get(metric_key)
        metric_display = (
            _format_percent(metric_value, scale=scale or 1.0, decimals=1)
            if scale is not None
            else _format_number(metric_value)
        )
        rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{plans}</td>
              <td>{metric_display}</td>
            </tr>
            """
        )
    return f"""
    <table class="protocol-table compact">
      <thead>
        <tr>
          <th>Protocol</th>
          <th>Plans</th>
          <th>{metric_label}</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _render_metric_bars(
    records: List[Dict[str, Any]],
    metric_key: str,
    scale: Optional[float] = None,
) -> str:
    if not records:
        return '<div class="empty">No protocols available.</div>'
    values = [
        record.get("metrics", {}).get(metric_key)
        for record in records
        if record.get("metrics", {}).get(metric_key) is not None
    ]
    if not values:
        return '<div class="empty">No numeric metrics.</div>'
    max_val = 1.0
    if scale is None:
        max_val = max(values) or 1.0
    rows = []
    for record in records:
        name = html.escape(str(record.get("protocol_name", "Unknown")))
        metric_value = record.get("metrics", {}).get(metric_key)
        if metric_value is None:
            continue
        if scale is not None:
            width_pct = max(0.0, min(metric_value * scale, 100.0))
        else:
            width_pct = (metric_value / max_val) * 100 if max_val else 0.0
        metric_display = (
            _format_percent(metric_value, scale=scale or 1.0, decimals=1)
            if scale is not None
            else _format_number(metric_value)
        )
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{name}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {width_pct:.1f}%"></div>
              </div>
              <div class="bar-value">{metric_display}</div>
            </div>
            """
        )
    axis = ""
    if scale is not None:
        axis = """
        <div class="bar-axis">
          <span>0%</span>
          <span>100%</span>
        </div>
        """
    return f"{''.join(rows)}{axis}"


def _render_progress_band_chart(
    points: List[Dict[str, Any]],
    y_label: str = "Plan score",
    x_label: str = "Iteration %",
) -> str:
    if not points:
        return '<div class="empty">No trajectory data.</div>'

    usable = [point for point in points if point.get("mean") is not None]
    if len(usable) < 2:
        return '<div class="empty">Not enough data.</div>'

    width = 240
    height = 140
    pad_x = 26
    pad_y = 18
    chart_width = width - pad_x * 2
    chart_height = height - pad_y * 2

    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _y(val: float) -> float:
        return pad_y + ((100.0 - _clamp(val, 0.0, 100.0)) / 100.0) * chart_height

    xs = []
    means = []
    uppers = []
    lowers = []
    for point in usable:
        center = float(point.get("center", 0.0))
        mean = float(point.get("mean", 0.0))
        std = float(point.get("std", 0.0) or 0.0)
        x = pad_x + center * chart_width
        xs.append(x)
        means.append(_y(mean))
        uppers.append(_y(mean + std))
        lowers.append(_y(mean - std))

    upper_path = " ".join(
        f"{'M' if idx == 0 else 'L'}{xs[idx]:.1f},{uppers[idx]:.1f}"
        for idx in range(len(xs))
    )
    lower_path = " ".join(
        f"{'L'}{xs[idx]:.1f},{lowers[idx]:.1f}"
        for idx in range(len(xs) - 1, -1, -1)
    )
    band_path = f"{upper_path} {lower_path} Z"
    mean_path = " ".join(
        f"{'M' if idx == 0 else 'L'}{xs[idx]:.1f},{means[idx]:.1f}"
        for idx in range(len(xs))
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="score-sparkline" role="img" aria-label="Plan score trajectory">
      <line x1="{pad_x}" y1="{pad_y + chart_height:.1f}" x2="{width - pad_x}" y2="{pad_y + chart_height:.1f}" stroke="#dfe1df" />
      <line x1="{pad_x}" y1="{pad_y:.1f}" x2="{pad_x}" y2="{pad_y + chart_height:.1f}" stroke="#dfe1df" />
      <text x="{pad_x - 6}" y="{pad_y + 6:.1f}" text-anchor="end" font-size="9" fill="#8e9089">100</text>
      <text x="{pad_x - 6}" y="{pad_y + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">0</text>
      <text x="{pad_x}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">0%</text>
      <text x="{width - pad_x}" y="{height - 6}" text-anchor="end" font-size="9" fill="#8e9089">100%</text>
      <text x="{pad_x + 2}" y="{pad_y - 4}" font-size="9" fill="#8e9089">{html.escape(y_label)}</text>
      <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle" font-size="9" fill="#8e9089">{html.escape(x_label)}</text>
      <path d="{band_path}" fill="rgba(89, 183, 223, 0.2)" stroke="none" />
      <path d="{mean_path}" fill="none" stroke="#152456" stroke-width="2" />
    </svg>
    """


def _render_family_bar_chart(label_counts: Dict[str, int], max_families: int = 5) -> str:
    if not label_counts:
        return '<div class="empty">No family data.</div>'

    items = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    items = items[:max_families]
    total = sum(label_counts.values()) or 1

    width = 240
    height = 140
    pad_x = 26
    pad_y = 22
    chart_width = width - pad_x * 2
    chart_height = height - pad_y * 2
    slot_width = chart_width / max(len(items), 1)
    bar_width = slot_width * 0.6

    bars = []
    labels = []
    for idx, (family, count) in enumerate(items):
        pct = (count / total) * 100.0 if total else 0.0
        bar_height = chart_height * (pct / 100.0)
        x = pad_x + idx * slot_width + (slot_width - bar_width) / 2
        y = pad_y + chart_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="#59b7df" />'
        )
        label_x = x + bar_width / 2
        label_y = height - 6
        labels.append(
            f'<text x="{label_x:.1f}" y="{label_y}" font-size="8" fill="#8e9089" '
            f'text-anchor="end" transform="rotate(-30 {label_x:.1f} {label_y})">{html.escape(str(family))}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="score-sparkline" role="img" aria-label="Structure family distribution">
      <line x1="{pad_x}" y1="{pad_y + chart_height:.1f}" x2="{width - pad_x}" y2="{pad_y + chart_height:.1f}" stroke="#dfe1df" />
      <line x1="{pad_x}" y1="{pad_y:.1f}" x2="{pad_x}" y2="{pad_y + chart_height:.1f}" stroke="#dfe1df" />
      <text x="{pad_x - 6}" y="{pad_y + 6:.1f}" text-anchor="end" font-size="9" fill="#8e9089">100</text>
      <text x="{pad_x - 6}" y="{pad_y + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">0</text>
      <text x="{pad_x + 2}" y="{pad_y - 4}" font-size="9" fill="#8e9089">Share %</text>
      <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle" font-size="9" fill="#8e9089">Structure family</text>
      {"".join(bars)}
      {"".join(labels)}
    </svg>
    """


def _render_family_distribution_cards(
    records: List[Dict[str, Any]],
    label_counts_map: Dict[str, Dict[str, int]],
    subtitle: str,
    metric_key: Optional[str] = None,
    metric_label: str = "",
    scale: Optional[float] = None,
) -> str:
    if not records:
        return '<div class="empty">No protocols available.</div>'
    cards = []
    for record in records:
        name = str(record.get("protocol_name", "Unknown"))
        label_counts = label_counts_map.get(name, {})
        chart = _render_family_bar_chart(label_counts)
        metric_display = ""
        if metric_key:
            metric_val = record.get("metrics", {}).get(metric_key)
            if scale is not None:
                metric_display = _format_percent(metric_val, scale=scale, decimals=1)
            else:
                metric_display = _format_number(metric_val)
        metric_line = (
            f"{metric_label} {metric_display} · {subtitle}"
            if metric_display
            else subtitle
        )
        cards.append(
            f"""
            <div class="score-card">
              <h4>{html.escape(name)}</h4>
              <div class="score-meta">{html.escape(metric_line)}</div>
              {chart}
            </div>
            """
        )
    return f"<div class=\"progress-grid\">{''.join(cards)}</div>"


def _mean_metric(records: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    values = [
        record.get("metrics", {}).get(metric_key)
        for record in records
        if record.get("metrics", {}).get(metric_key) is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _render_progress_cards(
    records: List[Dict[str, Any]],
    progress_map: Dict[str, List[Dict[str, Any]]],
    subtitle: str,
    y_label: str = "Plan score",
    metric_key: Optional[str] = None,
    metric_label: str = "",
    scale: Optional[float] = None,
) -> str:
    if not records:
        return '<div class="empty">No protocols available.</div>'
    cards = []
    for record in records:
        name = str(record.get("protocol_name", "Unknown"))
        series = progress_map.get(name, [])
        chart = _render_progress_band_chart(series, y_label=y_label)
        metric_display = ""
        if metric_key:
            metric_val = record.get("metrics", {}).get(metric_key)
            if scale is not None:
                metric_display = _format_percent(metric_val, scale=scale, decimals=1)
            else:
                metric_display = _format_number(metric_val)
        metric_line = (
            f"{metric_label} {metric_display} · {subtitle}"
            if metric_display
            else subtitle
        )
        cards.append(
            f"""
            <div class="score-card">
              <h4>{html.escape(name)}</h4>
              <div class="score-meta">{html.escape(metric_line)}</div>
              {chart}
            </div>
            """
        )
    return f"<div class=\"progress-grid\">{''.join(cards)}</div>"


def _render_family_progress_cards(
    records: List[Dict[str, Any]],
    family_progress: Dict[str, Dict[str, List[Dict[str, Any]]]],
    family_lookup: Dict[str, str],
    subtitle: str,
) -> str:
    if not records:
        return '<div class="empty">No protocols available.</div>'
    cards = []
    for record in records:
        name = str(record.get("protocol_name", "Unknown"))
        family = family_lookup.get(name) or "Other"
        series = family_progress.get(name, {}).get(family, [])
        chart = _render_progress_band_chart(series, y_label="Percentile")
        cards.append(
            f"""
            <div class="score-card">
              <h4>{html.escape(name)}</h4>
              <div class="score-meta">{html.escape(subtitle)} · Family: {html.escape(family)}</div>
              {chart}
            </div>
            """
        )
    return f"<div class=\"progress-grid\">{''.join(cards)}</div>"


def _render_confusion_matrix(labels: List[str], matrix: List[List[int]]) -> str:
    if not labels or not matrix:
        return '<div class="empty">No confusion matrix available.</div>'

    header = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    rows = []
    for label, row in zip(labels, matrix):
        cells = "".join(f"<td>{_format_int(int(value))}</td>" for value in row)
        rows.append(
            f"""
            <tr>
              <th>{html.escape(label)}</th>
              {cells}
            </tr>
            """
        )

    return f"""
    <div class="confusion-matrix">
      <table class="protocol-table">
        <thead>
          <tr>
            <th></th>
            {header}
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </div>
    """


def _render_phase3_sweep_table(thresholds: List[Dict[str, Any]]) -> str:
    if not thresholds:
        return '<div class="empty">No sweep results available.</div>'

    rows = []
    for item in thresholds:
        min_plans = _format_int(int(item.get("min_plans_per_protocol", 0)))
        protocols = _format_int(int(item.get("protocols", 0)))
        task1 = _format_percent(item.get("task1_accuracy"), scale=100, decimals=1)
        task1_auc = _format_percent(item.get("task1_auc"), scale=100, decimals=1)
        task2 = _format_number(item.get("task2_mae"))
        task3 = _format_percent(item.get("task3_accuracy"), scale=100, decimals=1)
        rows.append(
            f"""
            <tr>
              <td>{min_plans}</td>
              <td>{protocols}</td>
              <td>{task1}</td>
              <td>{task1_auc}</td>
              <td>{task2}</td>
              <td>{task3}</td>
            </tr>
            """
        )

    return f"""
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Min Plans</th>
          <th>Protocols</th>
          <th>Pref Acc</th>
          <th>Pref AUC</th>
          <th>Remaining MAE</th>
          <th>Next-Focus Acc</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _build_protocol_metric_lookup(
    records: List[Dict[str, Any]], metric_key: str
) -> Dict[str, Optional[float]]:
    lookup: Dict[str, Optional[float]] = {}
    for record in records:
        name = record.get("protocol_name")
        if not name:
            continue
        lookup[name] = record.get("metrics", {}).get(metric_key)
    return lookup


def _render_protocol_delta_table(
    records: List[Dict[str, Any]],
    baseline_lookup: Dict[str, Optional[float]],
    metric_key: str,
    metric_label: str,
    scale: Optional[float] = None,
    top_n: int = 12,
    reverse: bool = True,
) -> str:
    rows_data = []
    for record in records:
        name = record.get("protocol_name", "Unknown")
        model_val = record.get("metrics", {}).get(metric_key)
        base_val = baseline_lookup.get(name)
        if model_val is None or base_val is None:
            continue
        delta = model_val - base_val
        rows_data.append((name, model_val, base_val, delta))

    if not rows_data:
        return '<div class="empty">No per-protocol comparison data.</div>'

    rows_data.sort(key=lambda item: item[3], reverse=reverse)
    rows_data = rows_data[:top_n]

    rows = []
    for name, model_val, base_val, delta in rows_data:
        name_safe = html.escape(str(name))
        if scale is not None:
            model_disp = _format_percent(model_val, scale=scale, decimals=1)
            base_disp = _format_percent(base_val, scale=scale, decimals=1)
            delta_disp = _format_signed_percent(delta, scale=scale, decimals=1)
        else:
            model_disp = _format_number(model_val)
            base_disp = _format_number(base_val)
            delta_disp = _format_number(delta)
        rows.append(
            f"""
            <tr>
              <td>{name_safe}</td>
              <td>{model_disp}</td>
              <td>{base_disp}</td>
              <td>{delta_disp}</td>
            </tr>
            """
        )

    return f"""
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Protocol</th>
          <th>Model {metric_label}</th>
          <th>Baseline</th>
          <th>Δ</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _render_delta_histogram(values: List[float]) -> str:
    if not values:
        return '<div class="empty">No delta values available.</div>'

    bins = [(-1.0, -0.2), (-0.2, -0.1), (-0.1, -0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]
    labels = ["<-0.2", "-0.2..-0.1", "-0.1..-0.05", "-0.05..0", "0..0.05", "0.05..0.1", "0.1..0.2", ">0.2"]
    counts = OrderedDict((label, 0) for label in labels)
    for value in values:
        for idx, (low, high) in enumerate(bins):
            if value < low:
                continue
            if value < high or (idx == len(bins) - 1 and value <= high):
                counts[labels[idx]] += 1
                break

    return _render_histogram(counts, 0)


def _render_scatter(
    points: List[Tuple[float, float]],
    x_label: str,
    y_label: str,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> str:
    if not points:
        return '<div class="empty">No scatter data.</div>'

    width = 320
    height = 220
    pad_left = 36
    pad_right = 18
    pad_top = 18
    pad_bottom = 34
    chart_width = width - pad_left - pad_right
    chart_height = height - pad_top - pad_bottom

    y_values = [point[1] for point in points]
    if y_min is None:
        y_min = min(y_values)
    if y_max is None:
        y_max = max(y_values)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05

    def x_for(value: float) -> float:
        return pad_left + ((value - x_min) / (x_max - x_min)) * chart_width

    def y_for(value: float) -> float:
        return pad_top + ((y_max - value) / (y_max - y_min)) * chart_height

    points_svg = []
    for x, y in points:
        points_svg.append(
            f'<circle cx="{x_for(x):.1f}" cy="{y_for(y):.1f}" r="3" fill="#59b7df" opacity="0.85" />'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="score-plot" role="img" aria-label="Scatter plot">
      <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" />
      <line x1="{pad_left}" y1="{pad_top + chart_height:.1f}" x2="{width - pad_right}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" />
      <text x="{pad_left - 6}" y="{pad_top + 6}" text-anchor="end" font-size="9" fill="#8e9089">{_format_percent(y_max, scale=100, decimals=1)}</text>
      <text x="{pad_left - 6}" y="{pad_top + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">{_format_percent(y_min, scale=100, decimals=1)}</text>
      <text x="{pad_left}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">{x_label}</text>
      <text x="{width - pad_right}" y="{height - 6}" text-anchor="end" font-size="9" fill="#8e9089">{y_label}</text>
      {"".join(points_svg)}
    </svg>
    """


def _render_imbalance_table(
    records: List[Dict[str, Any]],
    sort_key: str,
    title_label: str,
    top_n: int = 12,
    reverse: bool = True,
) -> str:
    if not records:
        return '<div class="empty">No imbalance data available.</div>'

    sorted_records = sorted(
        records,
        key=lambda item: item.get(sort_key) if item.get(sort_key) is not None else -1,
        reverse=reverse,
    )[:top_n]

    rows = []
    for record in sorted_records:
        name = html.escape(str(record.get("protocol_name", "Unknown")))
        top_family = html.escape(str(record.get("top_family", "n/a")))
        top_share = _format_percent(record.get("top_share"), scale=100, decimals=1)
        entropy = _format_number(record.get("entropy"))
        family_count = _format_int(int(record.get("family_count", 0)))
        pair_count = _format_int(int(record.get("pair_count", 0)))
        rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{top_family}</td>
              <td>{top_share}</td>
              <td>{entropy}</td>
              <td>{family_count}</td>
              <td>{pair_count}</td>
            </tr>
            """
        )

    return f"""
    <table class="protocol-table">
      <thead>
        <tr>
          <th>Protocol</th>
          <th>Top Family</th>
          <th>{title_label}</th>
          <th>Entropy</th>
          <th>Families</th>
          <th>Pairs</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def _render_topk_sparkline(topk: Dict[str, Any]) -> str:
    if not topk:
        return '<div class="empty">No top-k data.</div>'

    ks = [1, 2, 3, 5]
    points = [(k, topk.get(str(k)) or topk.get(k)) for k in ks]
    points = [(k, v) for k, v in points if isinstance(v, (int, float))]
    if not points:
        return '<div class="empty">No top-k data.</div>'

    width = 240
    height = 120
    pad_x = 26
    pad_top = 14
    pad_bottom = 22
    chart_width = width - (2 * pad_x)
    chart_height = height - pad_top - pad_bottom

    xs = []
    ys = []
    for k, v in points:
        x = pad_x + ((k - 1) / (5 - 1)) * chart_width
        y = pad_top + ((1.0 - v) * chart_height)
        xs.append(x)
        ys.append(y)

    path = " ".join(
        f"{'M' if idx == 0 else 'L'}{x:.1f},{y:.1f}"
        for idx, (x, y) in enumerate(zip(xs, ys))
    )

    circles = [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#59b7df" />'
        for x, y in zip(xs, ys)
    ]

    return f"""
    <svg viewBox="0 0 {width} {height}" class="score-sparkline" role="img" aria-label="Top-k accuracy">
      <line x1="{pad_x}" y1="{pad_top + chart_height:.1f}" x2="{width - pad_x}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" />
      <line x1="{pad_x}" y1="{pad_top:.1f}" x2="{pad_x}" y2="{pad_top + chart_height:.1f}" stroke="#dfe1df" />
      <text x="{pad_x - 6}" y="{pad_top + 6:.1f}" text-anchor="end" font-size="9" fill="#8e9089">100%</text>
      <text x="{pad_x - 6}" y="{pad_top + chart_height:.1f}" text-anchor="end" font-size="9" fill="#8e9089">0%</text>
      <text x="{pad_x}" y="{height - 6}" text-anchor="start" font-size="9" fill="#8e9089">k=1</text>
      <text x="{width - pad_x}" y="{height - 6}" text-anchor="end" font-size="9" fill="#8e9089">k=5</text>
      <path d="{path}" fill="none" stroke="#152456" stroke-width="2" />
      {"".join(circles)}
    </svg>
    """


def _render_dashboard(
    metrics: Dict[str, Any],
    phase2_metrics: Dict[str, Any],
    phase3_metrics: Dict[str, Any],
    phase3_sweep: Dict[str, Any],
    phase3_baselines: Dict[str, Any],
    phase3_analysis: Dict[str, Any],
    phase3_alternatives: Dict[str, Any],
    min_protocol_plans: int,
    logo_uri: str,
    selected_protocol: Optional[str],
    protocol_detail_html: str,
) -> str:
    totals = metrics.get("totals", {})
    attempt_histogram = metrics.get("attempt_histogram", OrderedDict())
    attempt_overflow = metrics.get("attempt_overflow", 0)
    approval_distribution = metrics.get("approval_distribution", OrderedDict())
    score_stats = metrics.get("score_stats", [])
    score_range = metrics.get("score_range", {})
    protocol_score_trends = metrics.get("protocol_score_trends", [])
    protocols = metrics.get("protocols", [])
    date_range = metrics.get("date_range", {})

    phase2_summary = phase2_metrics.get("summary", {})
    phase2_attempts = int(phase2_metrics.get("attempts", 0))
    phase2_plans = int(phase2_metrics.get("plans", 0))
    phase2_protocols = int(phase2_metrics.get("protocols", 0))
    phase2_label_counts = phase2_metrics.get("label_counts", OrderedDict())
    phase2_score_bins = phase2_metrics.get("score_bins", OrderedDict())
    phase2_coverage_bins = phase2_metrics.get("coverage_bins", OrderedDict())
    phase2_protocol_rows = phase2_metrics.get("protocol_rows", [])
    phase2_score_stats = phase2_metrics.get("score_stats", {})
    phase2_coverage_stats = phase2_metrics.get("coverage_stats", {})
    phase2_progress_map = phase2_metrics.get("protocol_progress", {})
    phase2_family_progress = phase2_metrics.get("family_progress", {})
    phase2_constraint_count = phase2_metrics.get("constraint_count")
    phase2_unique_structures = phase2_metrics.get("unique_structures")
    phase2_error = phase2_metrics.get("error")

    phase3_summary = phase3_metrics.get("summary", {})
    phase3_task1 = phase3_metrics.get("task1", {})
    phase3_task2 = phase3_metrics.get("task2", {})
    phase3_task3 = phase3_metrics.get("task3", {})
    phase3_error = phase3_metrics.get("error")

    phase3_sweep_thresholds = phase3_sweep.get("thresholds", [])
    phase3_sweep_generated_at = phase3_sweep.get("generated_at") or "n/a"
    phase3_sweep_error = phase3_sweep.get("error")

    phase3_baseline_task1 = phase3_baselines.get("task1", {}).get("macro", {})
    phase3_baseline_task2 = phase3_baselines.get("task2", {}).get("macro", {})
    phase3_baseline_task3 = phase3_baselines.get("task3", {}).get("macro", {})
    phase3_baseline_error = phase3_baselines.get("error")

    baseline_task1_lookup = _build_protocol_metric_lookup(
        phase3_baselines.get("task1", {}).get("protocols", []), "accuracy"
    )
    baseline_task3_bal_lookup = _build_protocol_metric_lookup(
        phase3_baselines.get("task3", {}).get("protocols", []), "balanced_accuracy"
    )
    baseline_task3_top3_lookup = _build_protocol_metric_lookup(
        phase3_baselines.get("task3", {}).get("protocols", []), "top3_accuracy"
    )

    phase3_analysis_records = phase3_analysis.get("protocols", [])
    phase3_analysis_error = phase3_analysis.get("error")

    phase3_alt_generated_at = phase3_alternatives.get("generated_at") or "n/a"
    phase3_alt_settings = phase3_alternatives.get("settings", {})
    phase3_alt_stop = phase3_alternatives.get("stop_continue", {})
    phase3_alt_ordinal = phase3_alternatives.get("ordinal_progress", {})
    phase3_alt_rank = phase3_alternatives.get("ranking_correlation", {})
    phase3_alt_focus = phase3_alternatives.get("next_focus_recommender", {})
    phase3_alt_error = phase3_alternatives.get("error")

    phase2_ready = phase2_attempts > 0
    phase2_status_line = (
        "Phase 2 dataset loaded from data/derived."
        if phase2_ready
        else "Phase 2 dataset not found. Run src/data/phase2_dataset.py."
    )
    phase2_progress = "100%" if phase2_ready else "0%"

    histogram_html = _render_histogram(attempt_histogram, attempt_overflow)
    approval_rows = _render_bar_rows(approval_distribution)
    score_plot = _render_score_plot(score_stats, score_range)
    protocol_score_grid = _render_protocol_score_grid(
        protocol_score_trends, min_protocol_plans
    )
    protocol_table = _render_protocol_table(
        protocols, min_protocol_plans, selected_protocol
    )
    phase2_label_rows = _render_bar_rows(phase2_label_counts)
    phase2_score_rows = _render_bar_rows(phase2_score_bins)
    phase2_coverage_rows = _render_bar_rows(phase2_coverage_bins)
    phase2_protocol_table = _render_phase2_protocol_table(phase2_protocol_rows)

    phase3_task1_rows = _render_model_protocol_table(
        phase3_task1.get("protocols", []),
        "accuracy",
        "Preference Acc.",
        scale=100.0,
    )
    phase3_task2_rows = _render_model_protocol_table(
        phase3_task2.get("protocols", []),
        "mae",
        "Remaining MAE",
    )
    phase3_task3_rows = _render_model_protocol_table(
        phase3_task3.get("protocols", []),
        "accuracy",
        "Next-Focus Acc.",
        scale=100.0,
    )
    phase3_confusion = _render_confusion_matrix(
        phase3_task3.get("labels", []), phase3_task3.get("confusion_matrix", [])
    )
    phase3_sweep_table = _render_phase3_sweep_table(phase3_sweep_thresholds)

    task1_gain_table = _render_protocol_delta_table(
        phase3_task1.get("protocols", []),
        baseline_task1_lookup,
        "accuracy",
        "Accuracy",
        scale=100.0,
        top_n=12,
        reverse=True,
    )
    task1_loss_table = _render_protocol_delta_table(
        phase3_task1.get("protocols", []),
        baseline_task1_lookup,
        "accuracy",
        "Accuracy",
        scale=100.0,
        top_n=12,
        reverse=False,
    )
    task3_bal_gain_table = _render_protocol_delta_table(
        phase3_task3.get("protocols", []),
        baseline_task3_bal_lookup,
        "balanced_accuracy",
        "Balanced Acc",
        scale=100.0,
        top_n=12,
        reverse=True,
    )
    task3_top3_gain_table = _render_protocol_delta_table(
        phase3_task3.get("protocols", []),
        baseline_task3_top3_lookup,
        "top3_accuracy",
        "Top-3 Acc",
        scale=100.0,
        top_n=12,
        reverse=True,
    )

    analysis_lookup = {item.get("protocol_name"): item for item in phase3_analysis_records}
    family_label_counts = {
        name: record.get("label_counts", {})
        for name, record in analysis_lookup.items()
        if name
    }
    task1_delta_values: List[float] = []
    task2_delta_values: List[float] = []
    task3_bal_delta_values: List[float] = []
    scatter_points: List[Tuple[float, float]] = []
    useful_protocols: List[str] = []

    model_task1_lookup = _build_protocol_metric_lookup(
        phase3_task1.get("protocols", []), "accuracy"
    )
    model_task2_lookup = _build_protocol_metric_lookup(
        phase3_task2.get("protocols", []), "mae"
    )
    model_task3_bal_lookup = _build_protocol_metric_lookup(
        phase3_task3.get("protocols", []), "balanced_accuracy"
    )
    model_task3_top3_lookup = _build_protocol_metric_lookup(
        phase3_task3.get("protocols", []), "top3_accuracy"
    )

    for name, model_acc in model_task1_lookup.items():
        base_acc = baseline_task1_lookup.get(name)
        if model_acc is None or base_acc is None:
            continue
        delta = model_acc - base_acc
        task1_delta_values.append(delta)

    if not hasattr(_render_dashboard, "_task2_lookup"):  # type: ignore[attr-defined]
        _render_dashboard._task2_lookup = _build_protocol_metric_lookup(  # type: ignore[attr-defined]
            phase3_baselines.get("task2", {}).get("protocols", []), "mae"
        )
    base_task2_lookup = _render_dashboard._task2_lookup  # type: ignore[attr-defined]

    for name, model_mae in model_task2_lookup.items():
        base_val = base_task2_lookup.get(name)
        if model_mae is None or base_val is None:
            continue
        task2_delta_values.append(base_val - model_mae)

    for name, model_bal in model_task3_bal_lookup.items():
        base_bal = baseline_task3_bal_lookup.get(name)
        if model_bal is None or base_bal is None:
            continue
        delta = model_bal - base_bal
        task3_bal_delta_values.append(delta)
        analysis = analysis_lookup.get(name, {})
        top_share = analysis.get("top_share")
        if isinstance(top_share, (int, float)):
            scatter_points.append((float(top_share), delta))

    for name, model_acc in model_task1_lookup.items():
        base_task1 = baseline_task1_lookup.get(name)
        model_bal = model_task3_bal_lookup.get(name)
        base_bal = baseline_task3_bal_lookup.get(name)
        if (
            model_acc is None
            or base_task1 is None
            or model_bal is None
            or base_bal is None
        ):
            continue
        if (model_acc - base_task1) >= 0.15 and (model_bal - base_bal) >= 0.05:
            useful_protocols.append(str(name))

    task1_histogram = _render_delta_histogram(task1_delta_values)
    task2_histogram = _render_delta_histogram(task2_delta_values)
    task3_bal_histogram = _render_delta_histogram(task3_bal_delta_values)
    task3_scatter = _render_scatter(
        scatter_points, "Top-family share", "Δ balanced acc", y_min=-0.2, y_max=0.2
    )

    phase2_label_total = sum(phase2_label_counts.values()) or 0
    phase2_stop_rate = (
        phase2_label_counts.get("stop", 0) / phase2_label_total
        if phase2_label_total
        else None
    )
    phase2_coverage_threshold = phase2_summary.get("min_coverage_pct")
    phase2_plateau_delta = phase2_summary.get("plateau_delta")
    phase2_generated_at = phase2_summary.get("generated_at") or "n/a"
    phase2_coverage_label = _format_percent(phase2_coverage_threshold, scale=100, decimals=1)
    phase2_plateau_label = _format_number(phase2_plateau_delta)

    score_min = phase2_score_stats.get("min")
    score_max = phase2_score_stats.get("max")
    score_median = phase2_score_stats.get("median")
    coverage_min = phase2_coverage_stats.get("min")
    coverage_max = phase2_coverage_stats.get("max")
    coverage_median = phase2_coverage_stats.get("median")

    score_summary = (
        f"min {_format_percent(score_min, scale=1.0)} · "
        f"median {_format_percent(score_median, scale=1.0)} · "
        f"max {_format_percent(score_max, scale=1.0)}"
    )
    coverage_summary = (
        f"min {_format_percent(coverage_min, scale=100)} · "
        f"median {_format_percent(coverage_median, scale=100)} · "
        f"max {_format_percent(coverage_max, scale=100)}"
    )

    phase2_error_html = ""
    if phase2_error:
        phase2_error_html = (
            f"<div class=\"empty\">Phase 2 load error: {html.escape(str(phase2_error))}</div>"
        )

    phase3_ready = bool(phase3_task1.get("protocols"))
    phase3_status_line = (
        "Phase 3 metrics loaded from data/derived."
        if phase3_ready
        else "Phase 3 metrics not found. Run src/data/phase3_modeling.py."
    )
    phase3_progress = "100%" if phase3_ready else "0%"

    phase3_generated_at = phase3_summary.get("generated_at") or "n/a"
    phase3_min_plans = phase3_summary.get("min_plans_per_protocol")
    phase3_splits = phase3_summary.get("splits", {})
    phase3_features = phase3_summary.get("features", [])
    phase3_excluded = phase3_summary.get("excluded_features", [])
    phase3_include_family = phase3_summary.get("include_family_features")

    phase3_task1_acc = phase3_task1.get("macro", {}).get("accuracy")
    phase3_task1_auc = phase3_task1.get("macro", {}).get("auc")
    phase3_task2_mae = phase3_task2.get("macro", {}).get("mae")
    phase3_task3_acc = phase3_task3.get("macro", {}).get("accuracy")
    phase3_task3_bal = phase3_task3.get("macro", {}).get("balanced_accuracy")
    phase3_task3_top3 = phase3_task3.get("macro", {}).get("top3_accuracy")
    phase3_task3_top5 = phase3_task3.get("macro", {}).get("top5_accuracy")

    phase3_split_label = (
        f"{phase3_splits.get('train', 'n/a')}/"
        f"{phase3_splits.get('val', 'n/a')}/"
        f"{phase3_splits.get('test', 'n/a')}"
    )
    phase3_features_label = ", ".join(phase3_features) if phase3_features else "n/a"
    phase3_excluded_label = ", ".join(phase3_excluded) if phase3_excluded else "n/a"
    phase3_min_plans_display = (
        _format_int(int(phase3_min_plans))
        if isinstance(phase3_min_plans, (int, float))
        else "n/a"
    )
    phase3_family_label = (
        "yes" if phase3_include_family else "no"
        if phase3_include_family is not None
        else "n/a"
    )

    phase2_qualified_plans = phase2_summary.get("qualified_plans")
    phase2_attempts_written = phase2_summary.get("attempts_written")
    phase2_protocol_total = phase2_summary.get("protocols")
    phase2_min_attempts = phase2_summary.get("min_attempts")
    phase2_min_coverage = phase2_summary.get("min_coverage_pct")

    abstract_plan_count = (
        _format_int(int(phase2_qualified_plans))
        if isinstance(phase2_qualified_plans, (int, float))
        else "n/a"
    )
    abstract_attempts_count = (
        _format_int(int(phase2_attempts_written))
        if isinstance(phase2_attempts_written, (int, float))
        else "n/a"
    )
    abstract_protocol_count = (
        _format_int(int(phase2_protocol_total))
        if isinstance(phase2_protocol_total, (int, float))
        else "n/a"
    )
    abstract_min_attempts = (
        _format_int(int(phase2_min_attempts))
        if isinstance(phase2_min_attempts, (int, float))
        else "n/a"
    )
    abstract_min_coverage = _format_percent(phase2_min_coverage, scale=100, decimals=0)
    abstract_total_constraints = (
        _format_int(int(phase2_constraint_count))
        if isinstance(phase2_constraint_count, (int, float))
        else "n/a"
    )
    abstract_unique_structures = (
        _format_int(int(phase2_unique_structures))
        if isinstance(phase2_unique_structures, (int, float))
        else "n/a"
    )

    abstract_model_protocols = _format_int(len(phase3_task1.get("protocols", [])))
    abstract_pref_acc = _format_percent(phase3_task1_acc, scale=100, decimals=1)
    abstract_pref_auc = _format_number(phase3_task1_auc)
    abstract_pref_base = _format_percent(
        phase3_baseline_task1.get("accuracy"), scale=100, decimals=1
    )
    abstract_rem_mae = _format_number(phase3_task2_mae)
    abstract_rem_base = _format_number(phase3_baseline_task2.get("mae"))
    abstract_focus_top3 = _format_percent(phase3_task3_top3, scale=100, decimals=1)
    abstract_focus_top5 = _format_percent(phase3_task3_top5, scale=100, decimals=1)
    abstract_focus_bal = _format_percent(phase3_task3_bal, scale=100, decimals=1)
    abstract_focus_top3_base = _format_percent(
        phase3_baseline_task3.get("top3_accuracy"), scale=100, decimals=1
    )
    abstract_focus_bal_base = _format_percent(
        phase3_baseline_task3.get("balanced_accuracy"), scale=100, decimals=1
    )

    phase3_alt_min_plans = phase3_alt_settings.get("min_plans_per_protocol")
    phase3_alt_min_plans_display = (
        _format_int(int(phase3_alt_min_plans))
        if isinstance(phase3_alt_min_plans, (int, float))
        else "n/a"
    )
    phase3_alt_family = phase3_alt_settings.get("include_family_features")
    phase3_alt_family_label = (
        "yes" if phase3_alt_family else "no"
        if phase3_alt_family is not None
        else "n/a"
    )

    alt_stop_macro = phase3_alt_stop.get("macro", {})
    alt_stop_base = phase3_alt_stop.get("baseline", {})
    alt_stop_global = phase3_alt_stop.get("global", {}).get("metrics", {})
    alt_stop_global_base = phase3_alt_stop.get("global", {}).get("baseline", {})
    alt_ord_macro = phase3_alt_ordinal.get("macro", {})
    alt_ord_base = phase3_alt_ordinal.get("baseline", {})
    alt_ord_global = phase3_alt_ordinal.get("global", {}).get("metrics", {})
    alt_ord_global_base = phase3_alt_ordinal.get("global", {}).get("baseline", {})
    alt_rank_macro = phase3_alt_rank.get("macro", {})
    alt_rank_base = phase3_alt_rank.get("baseline", {})
    alt_focus_macro = phase3_alt_focus.get("macro", {})
    alt_focus_base = phase3_alt_focus.get("baseline", {})

    abstract_stop_bal = _format_percent(
        alt_stop_macro.get("balanced_accuracy"), scale=100, decimals=1
    )
    abstract_stop_base = _format_percent(
        alt_stop_base.get("balanced_accuracy"), scale=100, decimals=1
    )
    abstract_stop_auc = _format_number(alt_stop_macro.get("auc"))
    abstract_ord_f1 = _format_percent(
        alt_ord_macro.get("macro_f1"), scale=100, decimals=1
    )
    abstract_ord_base = _format_percent(
        alt_ord_base.get("macro_f1"), scale=100, decimals=1
    )
    abstract_rank_spearman = _format_number(alt_rank_macro.get("spearman"))
    abstract_rank_base = _format_number(alt_rank_base.get("spearman"))

    stop_bal_delta = _delta_value(
        alt_stop_macro.get("balanced_accuracy"),
        alt_stop_base.get("balanced_accuracy"),
    )
    ord_f1_delta = _delta_value(
        alt_ord_macro.get("macro_f1"),
        alt_ord_base.get("macro_f1"),
    )
    rank_spearman_delta = _delta_value(
        alt_rank_macro.get("spearman"),
        alt_rank_base.get("spearman"),
    )
    focus_top3_delta = _delta_value(
        alt_focus_macro.get("top3"),
        alt_focus_base.get("top3"),
    )

    task1_improved = sum(1 for delta in task1_delta_values if delta > 0)
    task2_improved = sum(1 for delta in task2_delta_values if delta > 0)
    task3_bal_improved = sum(1 for delta in task3_bal_delta_values if delta > 0)
    protocol_count = len(model_task1_lookup)
    useful_protocols_display = ", ".join(useful_protocols[:6]) if useful_protocols else "n/a"

    top_share_values = [
        item.get("top_share")
        for item in phase3_analysis_records
        if isinstance(item.get("top_share"), (int, float))
    ]
    if top_share_values:
        top_share_median = _percentile(top_share_values, 50.0)
        top_share_median_display = _format_percent(top_share_median, scale=100, decimals=1)
    else:
        top_share_median_display = "n/a"

    imbalance_high_table = _render_imbalance_table(
        phase3_analysis_records, "top_share", "Top-share", top_n=10, reverse=True
    )
    imbalance_low_table = _render_imbalance_table(
        phase3_analysis_records, "top_share", "Top-share", top_n=10, reverse=False
    )

    topk_cards = []
    task3_protocols = phase3_task3.get("protocols", [])
    task3_protocols_sorted = sorted(
        task3_protocols,
        key=lambda item: item.get("plan_count", 0),
        reverse=True,
    )
    for record in task3_protocols_sorted[:12]:
        name = html.escape(str(record.get("protocol_name", "Unknown")))
        topk = record.get("metrics", {}).get("topk_accuracy", {})
        chart = _render_topk_sparkline(topk)
        topk_cards.append(
            f"""
            <div class="score-card">
              <h4>{name}</h4>
              <div class="score-meta">Top-k accuracy</div>
              {chart}
            </div>
            """
        )
    topk_grid = (
        f"<div class=\"score-grid\">{''.join(topk_cards)}</div>"
        if topk_cards
        else '<div class="empty">No top-k data.</div>'
    )

    def _alt_improvement_summary(
        records: List[Dict[str, Any]], metric_key: str
    ) -> Tuple[int, int, Optional[str], Optional[float]]:
        deltas: List[float] = []
        best_name = None
        best_delta = None
        for record in records:
            model_val = record.get("metrics", {}).get(metric_key)
            base_val = record.get("baseline", {}).get(metric_key)
            if model_val is None or base_val is None:
                continue
            delta = model_val - base_val
            deltas.append(delta)
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_name = record.get("protocol_name")
        improved = sum(1 for value in deltas if value > 0)
        return improved, len(deltas), best_name, best_delta

    stop_improved, stop_total, stop_best_name, stop_best_delta = _alt_improvement_summary(
        phase3_alt_stop.get("protocols", []), "balanced_accuracy"
    )
    ord_improved, ord_total, ord_best_name, ord_best_delta = _alt_improvement_summary(
        phase3_alt_ordinal.get("protocols", []), "macro_f1"
    )
    rank_improved, rank_total, rank_best_name, rank_best_delta = _alt_improvement_summary(
        phase3_alt_rank.get("protocols", []), "spearman"
    )
    focus_improved, focus_total, focus_best_name, focus_best_delta = _alt_improvement_summary(
        phase3_alt_focus.get("protocols", []), "top3"
    )

    stop_best_label = (
        f"{html.escape(str(stop_best_name))} ({_format_signed_percent(stop_best_delta, scale=100, decimals=1)})"
        if stop_best_name and stop_best_delta is not None
        else "n/a"
    )
    ord_best_label = (
        f"{html.escape(str(ord_best_name))} ({_format_signed_percent(ord_best_delta, scale=100, decimals=1)})"
        if ord_best_name and ord_best_delta is not None
        else "n/a"
    )
    rank_best_label = (
        f"{html.escape(str(rank_best_name))} ({_format_number(rank_best_delta)})"
        if rank_best_name and rank_best_delta is not None
        else "n/a"
    )
    focus_best_label = (
        f"{html.escape(str(focus_best_name))} ({_format_signed_percent(focus_best_delta, scale=100, decimals=1)})"
        if focus_best_name and focus_best_delta is not None
        else "n/a"
    )

    alt_summary_rows = []
    alt_summary_rows.append(
        {
            "approach": "Stop / Continue",
            "metric": "Balanced Acc",
            "model": _format_percent(alt_stop_macro.get("balanced_accuracy"), scale=100, decimals=1),
            "baseline": _format_percent(alt_stop_base.get("balanced_accuracy"), scale=100, decimals=1),
            "delta": _format_signed_percent(stop_bal_delta, scale=100, decimals=1),
            "improved": f"{stop_improved}/{stop_total}" if stop_total else "n/a",
        }
    )
    alt_summary_rows.append(
        {
            "approach": "Ordinal Progress",
            "metric": "Macro F1",
            "model": _format_percent(alt_ord_macro.get("macro_f1"), scale=100, decimals=1),
            "baseline": _format_percent(alt_ord_base.get("macro_f1"), scale=100, decimals=1),
            "delta": _format_signed_percent(ord_f1_delta, scale=100, decimals=1),
            "improved": f"{ord_improved}/{ord_total}" if ord_total else "n/a",
        }
    )
    alt_summary_rows.append(
        {
            "approach": "Ranking Correlation",
            "metric": "Spearman",
            "model": _format_number(alt_rank_macro.get("spearman")),
            "baseline": _format_number(alt_rank_base.get("spearman")),
            "delta": _format_number(rank_spearman_delta),
            "improved": f"{rank_improved}/{rank_total}" if rank_total else "n/a",
        }
    )
    alt_summary_rows.append(
        {
            "approach": "Next-Focus Recommender",
            "metric": "Top-3",
            "model": _format_percent(alt_focus_macro.get("top3"), scale=100, decimals=1),
            "baseline": _format_percent(alt_focus_base.get("top3"), scale=100, decimals=1),
            "delta": _format_signed_percent(focus_top3_delta, scale=100, decimals=1),
            "improved": f"{focus_improved}/{focus_total}" if focus_total else "n/a",
        }
    )

    alt_summary_table = ""
    if alt_summary_rows:
        rows = []
        for row in alt_summary_rows:
            rows.append(
                f"""
                <tr>
                  <td>{row['approach']}</td>
                  <td>{row['metric']}</td>
                  <td>{row['model']}</td>
                  <td>{row['baseline']}</td>
                  <td>{row['delta']}</td>
                  <td>{row['improved']}</td>
                </tr>
                """
            )
        alt_summary_table = f"""
        <table class="protocol-table">
          <thead>
            <tr>
              <th>Approach</th>
              <th>Metric</th>
              <th>Model</th>
              <th>Baseline</th>
              <th>Δ</th>
              <th>Improved</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """
    else:
        alt_summary_table = '<div class="empty">No alternative metrics available.</div>'

    alt_global_rows = [
        {
            "approach": "Stop / Continue",
            "metric": "Balanced Acc",
            "model": _format_percent(alt_stop_global.get("balanced_accuracy"), scale=100, decimals=1),
            "baseline": _format_percent(alt_stop_global_base.get("balanced_accuracy"), scale=100, decimals=1),
            "delta": _format_signed_percent(
                _delta_value(
                    alt_stop_global.get("balanced_accuracy"),
                    alt_stop_global_base.get("balanced_accuracy"),
                ),
                scale=100,
                decimals=1,
            ),
        },
        {
            "approach": "Ordinal Progress",
            "metric": "Macro F1",
            "model": _format_percent(alt_ord_global.get("macro_f1"), scale=100, decimals=1),
            "baseline": _format_percent(alt_ord_global_base.get("macro_f1"), scale=100, decimals=1),
            "delta": _format_signed_percent(
                _delta_value(
                    alt_ord_global.get("macro_f1"),
                    alt_ord_global_base.get("macro_f1"),
                ),
                scale=100,
                decimals=1,
            ),
        },
    ]
    alt_global_table_rows = []
    for row in alt_global_rows:
        alt_global_table_rows.append(
            f"""
            <tr>
              <td>{row['approach']}</td>
              <td>{row['metric']}</td>
              <td>{row['model']}</td>
              <td>{row['baseline']}</td>
              <td>{row['delta']}</td>
            </tr>
            """
        )
    alt_global_table = (
        f"""
        <table class="protocol-table">
          <thead>
            <tr>
              <th>Approach</th>
              <th>Metric</th>
              <th>Model</th>
              <th>Baseline</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            {''.join(alt_global_table_rows)}
          </tbody>
        </table>
        """
        if alt_global_table_rows
        else '<div class="empty">No global metrics available.</div>'
    )

    pref_best, pref_worst = _select_best_worst(
        phase3_task1.get("protocols", []), "accuracy", top_n=5
    )
    stop_best, stop_worst = _select_best_worst(
        phase3_alt_stop.get("protocols", []), "balanced_accuracy", top_n=5
    )
    focus_best, focus_worst = _select_best_worst(
        phase3_task3.get("protocols", []), "top3_accuracy", top_n=5
    )

    pref_best_progress = _render_progress_cards(
        pref_best,
        phase2_progress_map,
        "Mean ± SD plan score",
        metric_key="accuracy",
        metric_label="Acc",
        scale=100,
    )
    pref_worst_progress = _render_progress_cards(
        pref_worst,
        phase2_progress_map,
        "Mean ± SD plan score",
        metric_key="accuracy",
        metric_label="Acc",
        scale=100,
    )
    pref_best_mean = _mean_metric(pref_best, "accuracy")
    pref_worst_mean = _mean_metric(pref_worst, "accuracy")

    stop_best_progress = _render_progress_cards(
        stop_best,
        phase2_progress_map,
        "Mean ± SD plan score",
        metric_key="balanced_accuracy",
        metric_label="Bal Acc",
        scale=100,
    )
    stop_worst_progress = _render_progress_cards(
        stop_worst,
        phase2_progress_map,
        "Mean ± SD plan score",
        metric_key="balanced_accuracy",
        metric_label="Bal Acc",
        scale=100,
    )
    stop_best_mean = _mean_metric(stop_best, "balanced_accuracy")
    stop_worst_mean = _mean_metric(stop_worst, "balanced_accuracy")

    focus_best_progress = _render_family_distribution_cards(
        focus_best,
        family_label_counts,
        "Top families by next-focus labels",
        metric_key="top3_accuracy",
        metric_label="Top-3",
        scale=100,
    )
    focus_worst_progress = _render_family_distribution_cards(
        focus_worst,
        family_label_counts,
        "Top families by next-focus labels",
        metric_key="top3_accuracy",
        metric_label="Top-3",
        scale=100,
    )
    focus_best_mean = _mean_metric(focus_best, "top3_accuracy")
    focus_worst_mean = _mean_metric(focus_worst, "top3_accuracy")

    pref_best_mean_display = _format_percent(pref_best_mean, scale=100, decimals=1)
    pref_worst_mean_display = _format_percent(pref_worst_mean, scale=100, decimals=1)
    stop_best_mean_display = _format_percent(stop_best_mean, scale=100, decimals=1)
    stop_worst_mean_display = _format_percent(stop_worst_mean, scale=100, decimals=1)
    focus_best_mean_display = _format_percent(focus_best_mean, scale=100, decimals=1)
    focus_worst_mean_display = _format_percent(focus_worst_mean, scale=100, decimals=1)

    stop_bal_delta_display = _format_signed_percent(stop_bal_delta, scale=100, decimals=1)
    ord_f1_delta_display = _format_signed_percent(ord_f1_delta, scale=100, decimals=1)
    rank_delta_display = _format_number(rank_spearman_delta)
    focus_top3_display = _format_percent(alt_focus_macro.get("top3"), scale=100, decimals=1)
    focus_top3_base_display = _format_percent(alt_focus_base.get("top3"), scale=100, decimals=1)

    phase3_error_html = ""
    if phase3_error:
        phase3_error_html = (
            f"<div class=\"empty\">Phase 3 load error: {html.escape(str(phase3_error))}</div>"
        )

    phase3_sweep_error_html = ""
    if phase3_sweep_error:
        phase3_sweep_error_html = (
            f"<div class=\"empty\">Sweep load error: {html.escape(str(phase3_sweep_error))}</div>"
        )

    phase3_baseline_error_html = ""
    if phase3_baseline_error:
        phase3_baseline_error_html = (
            f"<div class=\"empty\">Baseline load error: {html.escape(str(phase3_baseline_error))}</div>"
        )

    phase3_analysis_error_html = ""
    if phase3_analysis_error:
        phase3_analysis_error_html = (
            f"<div class=\"empty\">Analysis load error: {html.escape(str(phase3_analysis_error))}</div>"
        )

    phase3_alt_error_html = ""
    if phase3_alt_error:
        phase3_alt_error_html = (
            f"<div class=\"empty\">Alternative load error: {html.escape(str(phase3_alt_error))}</div>"
        )

    abstract_html = f"""
      <section id="abstract" class="grid">
        <div class="panel abstract-panel">
          <h2>Draft Abstract</h2>
          <p class="hint">Plain-language summary for clinical review. Auto-generated from Phase 2/3 outputs.</p>
          <div class="abstract-text">
            <p><strong>Title</strong> Protocol-specific planning trajectories for decision support in radiotherapy plan optimization.</p>
            <p><strong>Purpose</strong> During treatment planning, planners perform repeated DVH-based evaluations to judge whether a plan is improving, identify remaining tradeoffs, and decide when acceptable quality has been reached. This study evaluates whether DVH evaluations collected during real iterative planning contain sufficient signal to predict expert planning decisions at the next iteration, enabling protocol-specific plan quality prediction and decision support.</p>
            <p><strong>Methods</strong> DVH evaluation data were retrospectively curated from an institutional plan evaluation system, capturing the iterative assessments performed by planners during treatment planning. A total of {abstract_plan_count} clinically approved plans ({abstract_attempts_count} evaluation attempts) spanning {abstract_protocol_count} protocols were included, requiring at least {abstract_min_attempts} evaluations per plan and a final approved state with minimum coverage of {abstract_min_coverage}. Across all iterations, {abstract_total_constraints} constraint evaluations covering {abstract_unique_structures} unique structures were analyzed. A composite plan score was derived exclusively from approved plan evaluations and used only as a reference for defining protocol-specific quality targets. Intermediate DVH evaluations from earlier planning iterations were used to construct per-iteration feature representations summarizing constraint satisfaction, coverage, and violation severity. For protocols with at least 20 plans ({abstract_model_protocols} protocols), plan-level 70/10/20 train/validation/test splits were used to train models to answer three planning questions: (1) whether the next planner iteration would improve plan quality relative to the current iteration, using chronological order as a proxy label; (2) which structure family was most likely to improve next; and (3) whether additional iterations were likely to yield meaningful improvement (stop versus continue). A secondary analysis estimated remaining iterations to approval.</p>
            <p><strong>Results</strong> The improvement-direction model correctly identified the next iteration as better or worse with {abstract_pref_acc} accuracy versus a {abstract_pref_base} baseline (AUC {abstract_pref_auc}); top-5 protocols averaged {pref_best_mean_display} vs {pref_worst_mean_display} for the bottom-5. The stop/continue model achieved {abstract_stop_bal} balanced accuracy versus {abstract_stop_base} baseline (AUC {abstract_stop_auc}); top-5 protocols averaged {stop_best_mean_display} vs {stop_worst_mean_display} for the bottom-5. Mean absolute error for remaining-iteration prediction was {abstract_rem_mae} (baseline {abstract_rem_base}). Next-focus prediction included the observed improvement within the top three suggested structure families in {abstract_focus_top3} of cases (top five: {abstract_focus_top5}); top-5 protocols averaged {focus_best_mean_display} vs {focus_worst_mean_display} for the bottom-5, with lower balanced accuracy ({abstract_focus_bal} vs {abstract_focus_bal_base}) due to label imbalance. Highest-signal protocols included brain SRT (3–5 fx), head and neck oral cavity, breast tangents, and lung SBRT 54Gy/3fx, while rectal 54Gy and lung 45–60Gy protocols showed lower predictability.</p>
            <p><strong>Conclusion</strong> DVH evaluations collected during real iterative planning encode reproducible, protocol-specific signals that predict how expert planners improve plans and determine when planning should stop. This work demonstrates the feasibility of a plan quality prediction and decision-support framework that can assist planners directly and provide clinically grounded context for automated and AI-driven planning systems.</p>
          </div>
        </div>
      </section>
    """

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PlanEval Trajectory Dashboard</title>
    <style>
      :root {{
        --deep-blue: #152456;
        --bright-blue: #59b7df;
        --slate: #4e5259;
        --steel: #8e9089;
        --silver: #dfe1df;
        --volt-green: #ece819;
        --academic-red: #e53e30;
        --paper: #f7f7f4;
        --ink: #1b2330;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "GT Haptik", "Museo Sans", "Source Sans 3", sans-serif;
        color: var(--ink);
        background: linear-gradient(135deg, #ffffff 0%, var(--silver) 45%, #ffffff 100%);
        min-height: 100vh;
      }}

      body::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: linear-gradient(120deg, rgba(21, 36, 86, 0.08) 0%, transparent 40%),
          repeating-linear-gradient(160deg, rgba(21, 36, 86, 0.06) 0, rgba(21, 36, 86, 0.06) 1px, transparent 1px, transparent 16px);
        pointer-events: none;
        z-index: -1;
      }}

      .page {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 36px 24px 64px;
      }}

      .hero {{
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 24px;
        align-items: center;
        background: var(--paper);
        border-radius: 20px;
        padding: 28px 32px;
        box-shadow: 0 20px 40px rgba(21, 36, 86, 0.12);
        border: 1px solid rgba(21, 36, 86, 0.08);
        animation: fadeUp 0.6s ease-out;
      }}

      .hero img {{
        width: 120px;
        height: auto;
      }}

      .hero h1 {{
        margin: 0;
        font-family: "GT Sectra Fine", "Cormorant Garamond", serif;
        font-size: 32px;
        color: var(--deep-blue);
      }}

      .hero p {{
        margin: 6px 0 0;
        color: var(--slate);
        font-size: 15px;
        letter-spacing: 0.2px;
      }}

      .hero .tagline {{
        margin-top: 4px;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--bright-blue);
      }}

      .refresh {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 999px;
        background: var(--deep-blue);
        color: #ffffff;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }}

      .refresh:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(21, 36, 86, 0.25);
      }}

      .tabs {{
        margin-top: 18px;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }}

      .tabs a {{
        text-decoration: none;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid rgba(21, 36, 86, 0.15);
        color: var(--deep-blue);
        font-weight: 600;
        background: #ffffff;
      }}

      .grid {{
        margin-top: 24px;
        display: grid;
        gap: 20px;
      }}

      .abstract-figures {{
        grid-template-columns: 1fr;
      }}

      .abstract-figures .progress-grid {{
        grid-template-columns: repeat(5, minmax(180px, 1fr));
      }}

      .row-block {{
        margin-top: 12px;
      }}

      .row-block h4 {{
        margin: 0 0 8px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--steel);
      }}

      .kpis {{
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}

      .phase2-kpis {{
        margin-top: 16px;
      }}

      .phase3-kpis {{
        margin-top: 16px;
      }}

      .confusion-matrix {{
        overflow: auto;
        max-height: 360px;
      }}

      .card {{
        background: #ffffff;
        padding: 20px 22px;
        border-radius: 18px;
        border: 1px solid rgba(21, 36, 86, 0.08);
        box-shadow: 0 14px 30px rgba(21, 36, 86, 0.08);
        animation: fadeUp 0.6s ease-out;
      }}

      .card h3 {{
        margin: 0 0 8px;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--steel);
      }}

      .card .value {{
        font-size: 28px;
        font-weight: 700;
        color: var(--deep-blue);
      }}

      .card .meta {{
        margin-top: 8px;
        font-size: 13px;
        color: var(--slate);
      }}

      .panel {{
        background: #ffffff;
        border-radius: 20px;
        padding: 22px 24px;
        border: 1px solid rgba(21, 36, 86, 0.08);
        box-shadow: 0 18px 36px rgba(21, 36, 86, 0.1);
      }}

      .panel h2 {{
        margin: 0 0 8px;
        font-family: "GT Sectra Fine", "Cormorant Garamond", serif;
        color: var(--deep-blue);
        font-size: 22px;
      }}

      .panel p {{
        margin: 0 0 16px;
        color: var(--slate);
        font-size: 14px;
      }}

      .abstract-panel {{
        background: linear-gradient(135deg, #ffffff 0%, #eef2f6 100%);
        border: 1px solid rgba(21, 36, 86, 0.12);
      }}

      .abstract-text p {{
        margin: 0 0 12px;
        line-height: 1.5;
      }}

      .abstract-text strong {{
        color: var(--deep-blue);
      }}

      .hint {{
        margin: -6px 0 16px;
        color: var(--steel);
        font-size: 12px;
      }}

      .bar-row {{
        display: grid;
        grid-template-columns: 140px 1fr 70px;
        gap: 12px;
        align-items: center;
        margin-bottom: 10px;
      }}

      .bar-label {{
        font-size: 13px;
        color: var(--slate);
      }}

      .bar-track {{
        background: var(--silver);
        border-radius: 999px;
        overflow: hidden;
        height: 10px;
      }}

      .bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, var(--bright-blue), var(--deep-blue));
      }}

      .bar-value {{
        font-size: 12px;
        color: var(--steel);
        text-align: right;
      }}

      .bar-axis {{
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: var(--steel);
        margin: -4px 0 8px;
        padding-left: 4px;
        padding-right: 4px;
      }}

      .protocol-table.compact th,
      .protocol-table.compact td {{
        padding: 8px 10px;
        font-size: 12px;
      }}

      .split-grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      }}

      .mini-block {{
        background: #f7f7f4;
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 14px;
        padding: 12px;
      }}

      .mini-block h4 {{
        margin: 0 0 8px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--steel);
      }}

      .bar-value {{
        text-align: right;
        font-size: 13px;
        color: var(--deep-blue);
        font-weight: 600;
      }}

      .histogram {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(32px, 1fr));
        gap: 10px;
        align-items: end;
        height: 160px;
        padding: 8px 0 16px;
      }}

      .hist-bar {{
        display: grid;
        grid-template-rows: 1fr auto auto;
        align-items: end;
        height: 100%;
      }}

      .hist-fill {{
        width: 100%;
        background: linear-gradient(180deg, var(--bright-blue), var(--deep-blue));
        border-radius: 8px 8px 4px 4px;
      }}

      .hist-label {{
        text-align: center;
        font-size: 11px;
        color: var(--slate);
        margin-top: 6px;
      }}

      .hist-count {{
        text-align: center;
        font-size: 10px;
        color: var(--steel);
      }}

      .hist-note {{
        margin-top: 6px;
        font-size: 12px;
        color: var(--steel);
      }}

      .score-plot {{
        width: 100%;
        height: 200px;
      }}

      .pipeline-grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}

      .phase {{
        padding: 16px;
        border-radius: 16px;
        background: var(--paper);
        border: 1px solid rgba(21, 36, 86, 0.1);
      }}

      .phase h4 {{
        margin: 0 0 6px;
        color: var(--deep-blue);
        font-size: 16px;
      }}

      .phase p {{
        margin: 0;
        font-size: 13px;
        color: var(--slate);
      }}

      .progress {{
        margin-top: 12px;
        background: var(--silver);
        border-radius: 999px;
        overflow: hidden;
        height: 8px;
      }}

      .progress span {{
        display: block;
        height: 100%;
        background: var(--volt-green);
      }}

      .protocol-form {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
        margin: 8px 0 16px;
      }}

      .protocol-form input {{
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid rgba(21, 36, 86, 0.2);
        width: 120px;
      }}

      .protocol-form button {{
        padding: 8px 14px;
        border-radius: 999px;
        border: none;
        background: var(--deep-blue);
        color: #ffffff;
        font-weight: 600;
        cursor: pointer;
      }}

      .population-grid {{
        display: grid;
        grid-template-columns: minmax(220px, 1fr) minmax(320px, 2fr);
        gap: 16px;
        align-items: start;
      }}

      .protocol-list {{
        max-height: 520px;
        overflow: auto;
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 12px;
        background: #ffffff;
      }}

      .protocol-detail-panel {{
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 16px;
        padding: 16px;
        background: #ffffff;
        position: sticky;
        top: 18px;
      }}

      .protocol-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}

      .protocol-table th,
      .protocol-table td {{
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid var(--silver);
      }}

      .protocol-table th {{
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 11px;
        color: var(--steel);
        background: #ffffff;
        position: sticky;
        top: 0;
        z-index: 1;
      }}

      .protocol-link {{
        color: var(--deep-blue);
        text-decoration: none;
        font-weight: 600;
      }}

      .protocol-link:hover {{
        text-decoration: underline;
      }}

      .protocol-table tr.selected {{
        background: rgba(89, 183, 223, 0.15);
      }}

      .protocol-detail {{
        margin-top: 20px;
        display: grid;
        gap: 16px;
      }}

      .scorecard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 16px;
        margin: 12px 0 8px;
      }}

      .score-distribution {{
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 14px;
        padding: 10px 12px;
        background: #ffffff;
      }}

      .score-distribution-meta {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 11px;
        color: var(--steel);
        margin-bottom: 6px;
      }}

      .score-distribution a {{
        cursor: pointer;
      }}

      .scorecard {{
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 14px;
        padding: 10px 12px;
        background: #ffffff;
      }}

      .scorecard-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 6px;
      }}

      .scorecard-header h4 {{
        margin: 0;
        font-size: 14px;
        color: var(--deep-blue);
      }}

      .scorecard-score {{
        font-size: 12px;
        color: var(--steel);
      }}

      .daisy-plot {{
        width: 100%;
        height: auto;
      }}

      .scorecard-legend {{
        display: flex;
        gap: 12px;
        align-items: center;
        font-size: 11px;
        color: var(--steel);
        margin-top: 6px;
      }}

      .legend-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #111111;
        margin-right: 6px;
      }}

      .legend-dot.requested {{
        background: #888888;
      }}

      .legend-swatch {{
        display: inline-block;
        width: 14px;
        height: 9px;
        border-radius: 4px;
        background: #bbf7d0;
        border: 1px solid #d1d5db;
        margin-right: 6px;
      }}

      .legend-swatch.p2 {{
        opacity: 0.7;
      }}

      .legend-swatch.p3 {{
        opacity: 0.4;
      }}

      .score-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        max-height: 520px;
        overflow: auto;
        padding-right: 6px;
      }}

      .progress-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }}

      .score-card {{
        border: 1px solid rgba(21, 36, 86, 0.08);
        border-radius: 14px;
        padding: 12px;
        background: #ffffff;
      }}

      .score-card h4 {{
        margin: 0 0 6px;
        font-size: 14px;
        color: var(--deep-blue);
      }}

      .score-meta {{
        font-size: 11px;
        color: var(--steel);
        margin-bottom: 8px;
      }}


      .score-sparkline {{
        width: 100%;
        height: auto;
      }}

      .protocol-detail-header h3 {{
        margin: 0;
        font-size: 20px;
        color: var(--deep-blue);
      }}

      .protocol-detail-header p {{
        margin: 6px 0 0;
        color: var(--slate);
        font-size: 13px;
      }}

      .related-templates {{
        margin-top: 10px;
        font-size: 12px;
        color: var(--slate);
      }}

      .related-templates summary {{
        cursor: pointer;
        color: var(--deep-blue);
        font-weight: 600;
      }}

      .related-templates ul {{
        margin: 8px 0 0 18px;
        padding: 0;
        max-height: 160px;
        overflow: auto;
      }}

      .structure-group {{
        border: 1px solid rgba(21, 36, 86, 0.12);
        border-radius: 14px;
        padding: 10px 12px;
        background: var(--paper);
      }}

      .structure-group summary {{
        cursor: pointer;
        font-weight: 600;
        color: var(--deep-blue);
      }}

      .structure-group[open] summary {{
        margin-bottom: 12px;
      }}

      .constraint-grid {{
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        margin-top: 12px;
      }}

      .constraint-card {{
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        border: 1px solid rgba(21, 36, 86, 0.08);
      }}

      .constraint-title {{
        font-weight: 600;
        font-size: 13px;
        color: var(--deep-blue);
      }}

      .constraint-meta {{
        font-size: 11px;
        color: var(--steel);
        margin: 6px 0 8px;
      }}

      .violin-plot {{
        width: 100%;
        height: 130px;
      }}

      .violin-empty {{
        font-size: 11px;
        color: var(--steel);
        padding: 12px 0;
        text-align: center;
      }}

      .footer {{
        margin-top: 24px;
        font-size: 12px;
        color: var(--steel);
      }}

      .empty {{
        font-size: 13px;
        color: var(--steel);
      }}

      @keyframes fadeUp {{
        from {{
          opacity: 0;
          transform: translateY(10px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}

      @media (max-width: 860px) {{
        .hero {{
          grid-template-columns: 1fr;
          text-align: center;
        }}

        .hero img {{
          margin: 0 auto;
        }}

        .bar-row {{
          grid-template-columns: 1fr;
          gap: 6px;
        }}

        .bar-value {{
          text-align: left;
        }}

        .score-plot {{
          height: 240px;
        }}

        .population-grid {{
          grid-template-columns: 1fr;
        }}

        .abstract-figures .progress-grid {{
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }}

        .protocol-list {{
          max-height: 360px;
        }}

        .protocol-detail-panel {{
          position: static;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <header class="hero">
        <img src="{logo_uri}" alt="Jefferson logo" />
        <div>
          <h1>PlanEval Trajectory Dashboard</h1>
          <p class="tagline">Home of Sidney Kimmel Medical College</p>
          <p>Qualified plans only (>=2 attempts and last attempt approved).</p>
          <p>Last refresh: {metrics.get("generated_at", "n/a")}</p>
        </div>
        <a class="refresh" href="/?refresh=1">Refresh Data</a>
      </header>

      <nav class="tabs">
        <a href="#abstract">Abstract</a>
        <a href="#abstract-figures">Abstract Figures</a>
        <a href="#overview">Overview</a>
        <a href="#population">Population Score DB</a>
        <a href="#phase2">Phase 2 Dataset</a>
        <a href="#modeling">Modeling</a>
      </nav>

      {abstract_html}

      <section id="abstract-figures" class="grid abstract-figures">
        <div class="panel">
          <h2>Q1: Will the next iteration be better?</h2>
          <p>Top‑5 vs bottom‑5 protocols by improvement-direction accuracy (proxy label uses chronological order).</p>
          <p class="hint">Each card shows model accuracy and mean ± SD plan score trajectory across normalized iteration (0% → 100%).</p>
          <div class="row-block">
            <h4>Best 5 (mean {pref_best_mean_display})</h4>
            {pref_best_progress}
          </div>
          <div class="row-block">
            <h4>Worst 5 (mean {pref_worst_mean_display})</h4>
            {pref_worst_progress}
          </div>
        </div>
        <div class="panel">
          <h2>Q2: Which structure family improves next?</h2>
          <p>Top‑5 vs bottom‑5 protocols by next‑focus top‑3 accuracy.</p>
          <p class="hint">Each card shows top‑3 accuracy and the protocol‑specific structure family distribution (top 5 families by next‑focus labels).</p>
          <div class="row-block">
            <h4>Best 5 (mean {focus_best_mean_display})</h4>
            {focus_best_progress}
          </div>
          <div class="row-block">
            <h4>Worst 5 (mean {focus_worst_mean_display})</h4>
            {focus_worst_progress}
          </div>
        </div>
        <div class="panel">
          <h2>Q3: Should planning stop?</h2>
          <p>Top‑5 vs bottom‑5 protocols by stop/continue balanced accuracy.</p>
          <p class="hint">Each card shows balanced accuracy and mean ± SD plan score trajectory across normalized iteration (0% → 100%).</p>
          <div class="row-block">
            <h4>Best 5 (mean {stop_best_mean_display})</h4>
            {stop_best_progress}
          </div>
          <div class="row-block">
            <h4>Worst 5 (mean {stop_worst_mean_display})</h4>
            {stop_worst_progress}
          </div>
        </div>
      </section>

      <section id="overview" class="grid kpis">
        <div class="card">
          <h3>Patients</h3>
          <div class="value">{_format_int(int(totals.get("patients", 0)))}</div>
          <div class="meta">Qualified patient IDs</div>
        </div>
        <div class="card">
          <h3>Plans</h3>
          <div class="value">{_format_int(int(totals.get("plans", 0)))}</div>
          <div class="meta">Qualified plans</div>
        </div>
        <div class="card">
          <h3>Evaluations</h3>
          <div class="value">{_format_int(int(totals.get("evaluations", 0)))}</div>
          <div class="meta">Attempts in qualified plans</div>
        </div>
        <div class="card">
          <h3>Avg Attempts</h3>
          <div class="value">{totals.get("avg_attempts", 0.0)}</div>
          <div class="meta">Per qualified plan</div>
        </div>
        <div class="card">
          <h3>Total Constraints</h3>
          <div class="value">{_format_int(int(totals.get("total_constraints", 0)))}</div>
          <div class="meta">Avg per evaluation: {totals.get("avg_constraints", 0.0)}</div>
        </div>
        <div class="card">
          <h3>Unique Structures</h3>
          <div class="value">{_format_int(int(totals.get("unique_structures", 0)))}</div>
          <div class="meta">Across qualified evaluations</div>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Attempt Count Histogram</h2>
          <p>Plan counts by attempt number (2 to 10 shown).</p>
          {histogram_html}
        </div>
        <div class="panel">
          <h2>Plan Score Progression</h2>
          <p>Interquartile range with median by attempt number (1 to 10 shown). Scores use protocol-specific CPDs from final approved plans.</p>
          {score_plot}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Protocol Score Improvement</h2>
          <p>Median plan score trend per protocol using that protocol's CPD. X-axis is normalized iteration percent (0% = first attempt, 100% = last). Y-axis is plan score (protocol min-max).</p>
          <p class="hint">Plan score is a weighted mean of constraint percentiles (Priority 1 weight 2, Priority 2 weight 1). Direction comes from goal operators (<= lower is better, >= higher is better).</p>
          {protocol_score_grid}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Approval Status (Qualified Plans)</h2>
          <p>All attempts from qualified plans only.</p>
          {approval_rows}
        </div>
        <div class="panel">
          <h2>Pipeline Status</h2>
          <p>Phase 1 active. {phase2_status_line} {phase3_status_line}</p>
          <div class="pipeline-grid">
            <div class="phase">
              <h4>Phase 1: Data Extraction</h4>
              <p>Trajectory filtering and summary metrics.</p>
              <div class="progress"><span style="width: 100%"></span></div>
            </div>
            <div class="phase">
              <h4>Phase 2: Feature Extraction</h4>
              <p>{phase2_status_line}</p>
              <div class="progress"><span style="width: {phase2_progress}"></span></div>
            </div>
            <div class="phase">
              <h4>Phase 3: Modeling</h4>
              <p>{phase3_status_line}</p>
              <div class="progress"><span style="width: {phase3_progress}"></span></div>
            </div>
          </div>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Data Window</h2>
          <p>Created date range for qualified evaluations.</p>
          <div class="card" style="margin-top: 12px;">
            <h3>Range</h3>
            <div class="value">{date_range.get("min", "n/a")} to {date_range.get("max", "n/a")}</div>
            <div class="meta">Based on evaluation created_at timestamps</div>
          </div>
          <div class="footer">No PHI displayed. All values are aggregated.</div>
        </div>
      </section>

      <section id="population" class="grid">
        <div class="panel">
          <h2>Population Score Database Explorer</h2>
          <p>Protocols with enough approved plans to build percentile-based population scores.</p>
          <p class="hint">Click a protocol to view canonical constraint distributions from final approved plans (related templates included).</p>
          <form class="protocol-form" method="get" action="#population">
            <label for="min_protocol_plans">Minimum approved plans</label>
            <input id="min_protocol_plans" name="min_protocol_plans" type="number" min="1" value="{min_protocol_plans}" />
            <button type="submit">Apply</button>
          </form>
          <div class="population-grid">
            <div class="protocol-list">
              {protocol_table}
            </div>
            <div class="protocol-detail-panel">
              {protocol_detail_html or '<div class="empty">Select a protocol to view constraint distributions.</div>'}
            </div>
          </div>
          <div class="footer">Scoring model not built yet. This table only identifies candidate protocols.</div>
        </div>
      </section>

      <section id="phase2" class="grid">
        <div class="panel">
          <h2>Phase 2 Dataset</h2>
          <p>Derived plan-attempt features and labels written to <code>data/derived/</code>.</p>
          <p class="hint">Generated: {phase2_generated_at} · Coverage threshold: {phase2_coverage_label} · Plateau delta: {phase2_plateau_label}</p>
          {phase2_error_html}
          <div class="grid kpis phase2-kpis">
            <div class="card">
              <h3>Attempts</h3>
              <div class="value">{_format_int(phase2_attempts)}</div>
              <div class="meta">Rows in plan_attempt_features.jsonl</div>
            </div>
            <div class="card">
              <h3>Plans</h3>
              <div class="value">{_format_int(phase2_plans)}</div>
              <div class="meta">Unique patient-plan pairs</div>
            </div>
            <div class="card">
              <h3>Protocols</h3>
              <div class="value">{_format_int(phase2_protocols)}</div>
              <div class="meta">Protocols in Phase 2 data</div>
            </div>
            <div class="card">
              <h3>Stop Rate</h3>
              <div class="value">{_format_percent(phase2_stop_rate, scale=100, decimals=1)}</div>
              <div class="meta">Label = stop under plateau rule</div>
            </div>
            <div class="card">
              <h3>Score Range</h3>
              <div class="value">{_format_percent(score_min, scale=1.0)} – {_format_percent(score_max, scale=1.0)}</div>
              <div class="meta">Plan score percentiles</div>
            </div>
            <div class="card">
              <h3>Coverage Range</h3>
              <div class="value">{_format_percent(coverage_min, scale=100)} – {_format_percent(coverage_max, scale=100)}</div>
              <div class="meta">Matched constraint coverage</div>
            </div>
          </div>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Label Distribution</h2>
          <p>Stop vs continue labels across all attempts.</p>
          {phase2_label_rows}
        </div>
        <div class="panel">
          <h2>Plan Score Distribution</h2>
          <p>Weighted percentile plan scores (0-100).</p>
          <div class="hint">{score_summary}</div>
          {phase2_score_rows}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Coverage Distribution</h2>
          <p>Matched constraint coverage per attempt.</p>
          <div class="hint">{coverage_summary}</div>
          {phase2_coverage_rows}
        </div>
        <div class="panel">
          <h2>Phase 2 Protocol Summary</h2>
          <p>Top protocols by plan count with label rate and medians.</p>
          {phase2_protocol_table}
        </div>
      </section>

      <section id="modeling" class="grid">
        <div class="panel">
          <h2>Modeling (Phase 3)</h2>
          <p>Per-protocol models trained on Phase 2 attempt features.</p>
          <p class="hint">Generated: {phase3_generated_at} · Min plans/protocol: {phase3_min_plans_display} · Split: {phase3_split_label}</p>
          <p class="hint">Family features enabled: {phase3_family_label}</p>
          <p class="hint">Features: {html.escape(phase3_features_label)}</p>
          <p class="hint">Excluded: {html.escape(phase3_excluded_label)}</p>
          {phase3_error_html}
          <div class="grid kpis phase3-kpis">
            <div class="card">
              <h3>Preference Acc</h3>
              <div class="value">{_format_percent(phase3_task1_acc, scale=100, decimals=1)}</div>
              <div class="meta">Plan preference ranking accuracy</div>
            </div>
            <div class="card">
              <h3>Preference AUC</h3>
              <div class="value">{_format_percent(phase3_task1_auc, scale=100, decimals=1)}</div>
              <div class="meta">Plan preference ROC AUC</div>
            </div>
            <div class="card">
              <h3>Remaining Steps MAE</h3>
              <div class="value">{_format_number(phase3_task2_mae)}</div>
              <div class="meta">Remaining-iterations error</div>
            </div>
            <div class="card">
              <h3>Next-Focus Acc</h3>
              <div class="value">{_format_percent(phase3_task3_acc, scale=100, decimals=1)}</div>
              <div class="meta">Next improvement accuracy</div>
            </div>
            <div class="card">
              <h3>Next-Focus Bal Acc</h3>
              <div class="value">{_format_percent(phase3_task3_bal, scale=100, decimals=1)}</div>
              <div class="meta">Class-balanced accuracy</div>
            </div>
            <div class="card">
              <h3>Next-Focus Top-3</h3>
              <div class="value">{_format_percent(phase3_task3_top3, scale=100, decimals=1)}</div>
              <div class="meta">Top-3 accuracy</div>
            </div>
            <div class="card">
              <h3>Next-Focus Top-5</h3>
              <div class="value">{_format_percent(phase3_task3_top5, scale=100, decimals=1)}</div>
              <div class="meta">Top-5 accuracy</div>
            </div>
          </div>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Abstract Focus</h2>
          <p>
            Three clinical questions guide the modeling: (1) Will the next iteration be better than the current one (proxy label uses chronological order)? (2) Which structure family is most likely to improve next? (3) Should planning stop?
            The clearest, most consistent signal appears in (1) and (3), with protocol-dependent variability.
          </p>
          <p class="hint">
            Across eligible protocols, improvement-direction accuracy {abstract_pref_acc} vs {abstract_pref_base} baseline; stop/continue balanced accuracy {abstract_stop_bal} vs {abstract_stop_base}.
            Next-focus top‑3 {abstract_focus_top3} vs baseline {abstract_focus_top3_base} with balanced accuracy {abstract_focus_bal} due to label imbalance.
          </p>
          <p class="hint">
            Representative high-signal protocols (examples): {html.escape(useful_protocols_display)}.
          </p>
          <p class="hint">
            See the Abstract Figures tab for best vs worst protocol comparisons.
          </p>
        </div>
        <div class="panel">
          <h2>Alternative Approaches (Phase 3B)</h2>
          <p class="hint">Generated: {phase3_alt_generated_at} · Min plans/protocol: {phase3_alt_min_plans_display} · Family features: {phase3_alt_family_label}</p>
          {phase3_alt_error_html}
          {alt_summary_table}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Global Pooled Models</h2>
          <p>Pooled across protocols to test generalization beyond per‑protocol training.</p>
          {alt_global_table}
        </div>
        <div class="panel">
          <h2>Modeling Overview</h2>
          <p>
            These charts compare each model against a simple baseline (always choose the most common outcome).
            We see the clearest gains in iteration-to-iteration preference and stop/continue decisions, while remaining-steps and next-focus are more sensitive to data imbalance.
          </p>
          <p class="hint">
            Clinical use when it works: (1) rank competing iterations for approval readiness, (2) decide whether additional iterations are likely to help, and (3) shortlist likely next improvement targets.
          </p>
          <p class="hint">Protocols where the model is most helpful (preference + next-focus gains): {html.escape(useful_protocols_display)}</p>
          {phase3_analysis_error_html}
          <div class="grid">
            <div class="panel">
              <h3>Preference Δ Accuracy</h3>
              <p class="hint">Model minus baseline preference accuracy.</p>
              {task1_histogram}
            </div>
            <div class="panel">
              <h3>Remaining Steps Δ MAE</h3>
              <p class="hint">Baseline minus model MAE (positive is better).</p>
              {task2_histogram}
            </div>
          </div>
        </div>
      </section>


      <section class="grid">
        <div class="panel">
          <h2>Next-Focus vs Class Imbalance</h2>
          <p>Balanced accuracy gains vs. dominant family share per protocol.</p>
          {task3_scatter}
        </div>
        <div class="panel">
          <h2>Next-Focus Δ Balanced Accuracy</h2>
          <p class="hint">Model minus baseline balanced accuracy.</p>
          {task3_bal_histogram}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Protocol Imbalance (Most Dominant)</h2>
          <p>Protocols where one family dominates the next-improvement labels.</p>
          {imbalance_high_table}
        </div>
        <div class="panel">
          <h2>Protocol Imbalance (Most Diverse)</h2>
          <p>Protocols with more balanced label distributions.</p>
          {imbalance_low_table}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Baseline Comparison</h2>
          <p>Simple baselines for context (no model training).</p>
          {phase3_baseline_error_html}
          <table class="protocol-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Preference Accuracy</td>
                <td>{_format_percent(phase3_baseline_task1.get("accuracy"), scale=100, decimals=1)}</td>
                <td>{_format_percent(phase3_task1_acc, scale=100, decimals=1)}</td>
              </tr>
              <tr>
                <td>Preference AUC</td>
                <td>n/a</td>
                <td>{_format_percent(phase3_task1_auc, scale=100, decimals=1)}</td>
              </tr>
              <tr>
                <td>Remaining Steps MAE</td>
                <td>{_format_number(phase3_baseline_task2.get("mae"))}</td>
                <td>{_format_number(phase3_task2_mae)}</td>
              </tr>
              <tr>
                <td>Next-Focus Accuracy</td>
                <td>{_format_percent(phase3_baseline_task3.get("accuracy"), scale=100, decimals=1)}</td>
                <td>{_format_percent(phase3_task3_acc, scale=100, decimals=1)}</td>
              </tr>
              <tr>
                <td>Next-Focus Balanced Acc</td>
                <td>{_format_percent(phase3_baseline_task3.get("balanced_accuracy"), scale=100, decimals=1)}</td>
                <td>{_format_percent(phase3_task3_bal, scale=100, decimals=1)}</td>
              </tr>
              <tr>
                <td>Next-Focus Top-3</td>
                <td>{_format_percent(phase3_baseline_task3.get("top3_accuracy"), scale=100, decimals=1)}</td>
                <td>{_format_percent(phase3_task3_top3, scale=100, decimals=1)}</td>
              </tr>
              <tr>
                <td>Next-Focus Top-5</td>
                <td>{_format_percent(phase3_baseline_task3.get("top5_accuracy"), scale=100, decimals=1)}</td>
                <td>{_format_percent(phase3_task3_top5, scale=100, decimals=1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Min-Plans Sweep</h2>
          <p>Macro metrics across thresholds. Generated: {phase3_sweep_generated_at}</p>
          {phase3_sweep_error_html}
          {phase3_sweep_table}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Plan Preference Ranking</h2>
          <p>Does this iteration look better than the prior one?</p>
          {phase3_task1_rows}
        </div>
        <div class="panel">
          <h2>Remaining Iterations Estimation</h2>
          <p>Estimate how much work remains before approval.</p>
          {phase3_task2_rows}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Next-Focus Top-K Curves</h2>
          <p>Top-k accuracy (k=1..5) for the largest protocols.</p>
          {topk_grid}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Plan Preference Improvements</h2>
          <p>Largest gains over baseline (mean-percentile rule).</p>
          {task1_gain_table}
        </div>
        <div class="panel">
          <h2>Plan Preference Declines</h2>
          <p>Protocols where the baseline is stronger than the model.</p>
          {task1_loss_table}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Next-Focus Balanced Accuracy Gains</h2>
          <p>Balanced accuracy highlights improvements beyond class imbalance.</p>
          {task3_bal_gain_table}
        </div>
        <div class="panel">
          <h2>Next-Focus Top-3 Gains</h2>
          <p>Top-3 means the true family appears in the model's three most likely predictions.</p>
          {task3_top3_gain_table}
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Next-Focus Prediction</h2>
          <p>Which structure family is most likely to improve next?</p>
          {phase3_task3_rows}
        </div>
        <div class="panel">
          <h2>Next-Improvement Confusion Matrix</h2>
          {phase3_confusion}
        </div>
      </section>
    </div>
  </body>
</html>
"""


def _select_logo_uri() -> str:
    for filename, _mime in ASSET_CANDIDATES:
        if (ROOT_DIR / filename).exists():
            return f"/static/{filename}"
    return ""


def _read_static(path: Path) -> bytes:
    return path.read_bytes()


def _get_static_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _get_metrics(force_refresh: bool) -> Dict[str, Any]:
    global _METRICS_CACHE, _METRICS_ERROR
    if _METRICS_CACHE and not force_refresh:
        return _METRICS_CACHE

    try:
        _METRICS_CACHE = compute_metrics(DEFAULT_MONGODB_URI, "planeval", "evaluations")
        _METRICS_ERROR = ""
    except Exception as exc:  # noqa: BLE001 - display error in UI
        _METRICS_ERROR = str(exc)
        _METRICS_CACHE = {
            "generated_at": _utc_now_iso(),
            "totals": {},
            "attempt_histogram": OrderedDict(),
            "attempt_overflow": 0,
            "approval_distribution": OrderedDict(),
            "score_stats": [],
            "score_range": {"min": 0.0, "max": 1.0},
            "protocol_score_trends": [],
            "protocols": [],
            "date_range": {},
        }
    return _METRICS_CACHE


def _get_phase2_metrics(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE2_CACHE, _PHASE2_ERROR
    if _PHASE2_CACHE and not force_refresh:
        return _PHASE2_CACHE

    summary: Dict[str, Any] = {}
    if PHASE2_SUMMARY_PATH.exists():
        try:
            summary = json.loads(PHASE2_SUMMARY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    if not PHASE2_ATTEMPTS_PATH.exists():
        _PHASE2_ERROR = f"Missing {PHASE2_ATTEMPTS_PATH}"
        _PHASE2_CACHE = {
            "summary": summary,
            "attempts": 0,
            "plans": 0,
            "protocols": 0,
            "label_counts": OrderedDict(),
            "score_bins": OrderedDict(),
            "score_overflow": 0,
            "coverage_bins": OrderedDict(),
            "coverage_overflow": 0,
            "score_stats": {},
            "coverage_stats": {},
            "protocol_rows": [],
            "protocol_progress": {},
            "family_progress": {},
            "constraint_count": 0,
            "unique_structures": 0,
            "generated_at": summary.get("generated_at"),
            "error": _PHASE2_ERROR,
        }
        return _PHASE2_CACHE

    attempt_count = 0
    plan_keys: set = set()
    protocol_names: set = set()
    label_counts = Counter()
    score_values: List[float] = []
    coverage_values: List[float] = []
    progress_bins = 5
    progress_stats: Dict[str, List[Dict[str, float]]] = defaultdict(
        lambda: [
            {"count": 0.0, "sum": 0.0, "sumsq": 0.0} for _ in range(progress_bins)
        ]
    )
    attempt_progress_map: Dict[Tuple[str, Any, Any, Any], float] = {}
    protocol_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "plan_keys": set(),
            "attempt_count": 0,
            "stop_count": 0,
            "scores": [],
            "coverages": [],
        }
    )

    with PHASE2_ATTEMPTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            attempt_count += 1
            patient_id = record.get("patient_id")
            plan_id = record.get("plan_id")
            if patient_id is not None or plan_id is not None:
                plan_keys.add((patient_id, plan_id))

            protocol_name = record.get("protocol_name") or "Unknown"
            protocol_names.add(protocol_name)
            stats = protocol_stats[protocol_name]
            stats["attempt_count"] += 1
            if patient_id is not None or plan_id is not None:
                stats["plan_keys"].add((patient_id, plan_id))

            label_stop = record.get("label_stop")
            if label_stop:
                label_counts["stop"] += 1
                stats["stop_count"] += 1
            else:
                label_counts["continue"] += 1

            attempt_number = record.get("attempt_number")
            attempt_progress = _normalize_numeric(record.get("attempt_progress"))
            if attempt_progress is None:
                idx = _normalize_numeric(record.get("attempt_index"))
                total = _normalize_numeric(record.get("attempt_count"))
                if idx is not None and total is not None and total > 1:
                    attempt_progress = (idx - 1) / (total - 1)
            if (
                attempt_number is not None
                and attempt_progress is not None
                and patient_id is not None
                and plan_id is not None
            ):
                attempt_progress_map[
                    (protocol_name, patient_id, plan_id, attempt_number)
                ] = float(attempt_progress)

            score = _normalize_numeric(record.get("plan_score"))
            if score is not None:
                score_values.append(score)
                stats["scores"].append(score)

                progress = attempt_progress
                if progress is not None:
                    bin_idx = int(progress * progress_bins)
                    if bin_idx >= progress_bins:
                        bin_idx = progress_bins - 1
                    bin_stats = progress_stats[protocol_name][bin_idx]
                    bin_stats["count"] += 1.0
                    bin_stats["sum"] += score
                    bin_stats["sumsq"] += score * score

            coverage = _normalize_numeric(record.get("coverage_pct"))
            if coverage is not None:
                coverage_values.append(coverage)
                stats["coverages"].append(coverage)

    constraint_count = 0
    structure_names: set = set()
    family_progress_stats: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(
        lambda: [
            {"count": 0.0, "sum": 0.0, "sumsq": 0.0} for _ in range(progress_bins)
        ]
    )
    if PHASE2_CONSTRAINTS_PATH.exists():
        with PHASE2_CONSTRAINTS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                constraint_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                protocol = record.get("protocol_name") or "Unknown"
                patient_id = record.get("patient_id")
                plan_id = record.get("plan_id")
                attempt_number = record.get("attempt_number")
                if patient_id is None or plan_id is None or attempt_number is None:
                    continue
                progress = attempt_progress_map.get(
                    (protocol, patient_id, plan_id, attempt_number)
                )
                if progress is None:
                    continue
                percentile = _normalize_numeric(record.get("percentile"))
                if percentile is None:
                    continue
                structure = record.get("structure_tg263") or record.get("structure")
                if structure:
                    structure_names.add(str(structure))
                family = _structure_family(structure)
                bin_idx = int(progress * progress_bins)
                if bin_idx >= progress_bins:
                    bin_idx = progress_bins - 1
                stats = family_progress_stats[(protocol, family)][bin_idx]
                stats["count"] += 1.0
                stats["sum"] += percentile
                stats["sumsq"] += percentile * percentile

    score_bins = [(i, i + 10) for i in range(0, 100, 10)]
    score_labels = [f"{i}-{i + 10}%" for i in range(0, 100, 10)]
    score_distribution, score_overflow = _build_bin_counts(
        score_values, score_bins, score_labels
    )

    coverage_bins = [(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    coverage_labels = ["<70%", "70-80%", "80-90%", "90-100%"]
    coverage_distribution, coverage_overflow = _build_bin_counts(
        coverage_values, coverage_bins, coverage_labels
    )

    score_stats: Dict[str, Any] = {}
    if score_values:
        score_stats = {
            "count": len(score_values),
            "min": min(score_values),
            "max": max(score_values),
            "median": _percentile(score_values, 50.0),
        }

    coverage_stats: Dict[str, Any] = {}
    if coverage_values:
        coverage_stats = {
            "count": len(coverage_values),
            "min": min(coverage_values),
            "max": max(coverage_values),
            "median": _percentile(coverage_values, 50.0),
        }

    protocol_progress: Dict[str, List[Dict[str, Any]]] = {}
    for protocol_name, bins in progress_stats.items():
        points = []
        for idx, stats in enumerate(bins):
            count = int(stats["count"])
            mean = None
            std = None
            if count > 0:
                mean = stats["sum"] / stats["count"]
                variance = (stats["sumsq"] / stats["count"]) - (mean * mean)
                std = math.sqrt(max(variance, 0.0))
            points.append(
                {
                    "bin": idx,
                    "center": (idx + 0.5) / progress_bins,
                    "mean": mean,
                    "std": std,
                    "count": count,
                }
            )
        protocol_progress[protocol_name] = points

    family_progress: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for (protocol, family), bins in family_progress_stats.items():
        points = []
        for idx, stats in enumerate(bins):
            count = int(stats["count"])
            mean = None
            std = None
            if count > 0:
                mean = stats["sum"] / stats["count"]
                variance = (stats["sumsq"] / stats["count"]) - (mean * mean)
                std = math.sqrt(max(variance, 0.0))
            points.append(
                {
                    "bin": idx,
                    "center": (idx + 0.5) / progress_bins,
                    "mean": mean,
                    "std": std,
                    "count": count,
                }
            )
        family_progress[protocol][family] = points

    protocol_rows: List[Dict[str, Any]] = []
    for protocol_name, stats in protocol_stats.items():
        attempt_total = stats["attempt_count"]
        stop_rate = (stats["stop_count"] / attempt_total) if attempt_total else None
        protocol_rows.append(
            {
                "protocol_name": protocol_name,
                "plan_count": len(stats["plan_keys"]),
                "attempt_count": attempt_total,
                "stop_rate": stop_rate,
                "median_score": _percentile(stats["scores"], 50.0)
                if stats["scores"]
                else None,
                "median_coverage": _percentile(stats["coverages"], 50.0)
                if stats["coverages"]
                else None,
            }
        )
    protocol_rows.sort(
        key=lambda item: (item.get("plan_count", 0), item.get("attempt_count", 0)),
        reverse=True,
    )

    _PHASE2_ERROR = ""
    _PHASE2_CACHE = {
        "summary": summary,
        "attempts": attempt_count,
        "plans": len(plan_keys),
        "protocols": len(protocol_names),
        "label_counts": OrderedDict(sorted(label_counts.items())),
        "score_bins": score_distribution,
        "score_overflow": score_overflow,
        "coverage_bins": coverage_distribution,
        "coverage_overflow": coverage_overflow,
        "score_stats": score_stats,
        "coverage_stats": coverage_stats,
        "protocol_rows": protocol_rows,
        "protocol_progress": protocol_progress,
        "family_progress": family_progress,
        "constraint_count": constraint_count,
        "unique_structures": len(structure_names),
        "generated_at": summary.get("generated_at"),
        "error": _PHASE2_ERROR,
    }
    return _PHASE2_CACHE


def _get_phase3_metrics(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE3_CACHE, _PHASE3_ERROR
    if _PHASE3_CACHE and not force_refresh:
        return _PHASE3_CACHE

    if not PHASE3_METRICS_PATH.exists():
        _PHASE3_ERROR = f"Missing {PHASE3_METRICS_PATH}"
        _PHASE3_CACHE = {
            "summary": {},
            "task1": {"macro": {}, "protocols": []},
            "task2": {"macro": {}, "protocols": []},
            "task3": {"macro": {}, "protocols": [], "labels": [], "confusion_matrix": []},
            "error": _PHASE3_ERROR,
        }
        return _PHASE3_CACHE

    try:
        payload = json.loads(PHASE3_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}

    _PHASE3_ERROR = ""
    _PHASE3_CACHE = {
        "summary": {
            "generated_at": payload.get("generated_at"),
            "min_plans_per_protocol": payload.get("min_plans_per_protocol"),
            "splits": payload.get("splits", {}),
            "features": payload.get("features", []),
            "excluded_features": payload.get("excluded_features", []),
            "include_family_features": payload.get("include_family_features"),
        },
        "task1": payload.get("task1", {"macro": {}, "protocols": []}),
        "task2": payload.get("task2", {"macro": {}, "protocols": []}),
        "task3": payload.get("task3", {"macro": {}, "protocols": [], "labels": [], "confusion_matrix": []}),
        "error": _PHASE3_ERROR,
    }
    return _PHASE3_CACHE


def _get_phase3_sweep(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE3_SWEEP_CACHE, _PHASE3_SWEEP_ERROR
    if _PHASE3_SWEEP_CACHE and not force_refresh:
        return _PHASE3_SWEEP_CACHE

    if not PHASE3_SWEEP_PATH.exists():
        _PHASE3_SWEEP_ERROR = f"Missing {PHASE3_SWEEP_PATH}"
        _PHASE3_SWEEP_CACHE = {
            "thresholds": [],
            "generated_at": None,
            "error": _PHASE3_SWEEP_ERROR,
        }
        return _PHASE3_SWEEP_CACHE

    try:
        payload = json.loads(PHASE3_SWEEP_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}

    _PHASE3_SWEEP_ERROR = ""
    _PHASE3_SWEEP_CACHE = {
        "thresholds": payload.get("thresholds", []),
        "generated_at": payload.get("generated_at"),
        "error": _PHASE3_SWEEP_ERROR,
    }
    return _PHASE3_SWEEP_CACHE


def _get_phase3_baselines(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE3_BASELINE_CACHE, _PHASE3_BASELINE_ERROR
    if _PHASE3_BASELINE_CACHE and not force_refresh:
        return _PHASE3_BASELINE_CACHE

    if not PHASE3_BASELINE_PATH.exists():
        _PHASE3_BASELINE_ERROR = f"Missing {PHASE3_BASELINE_PATH}"
        _PHASE3_BASELINE_CACHE = {
            "task1": {"macro": {}},
            "task2": {"macro": {}},
            "task3": {"macro": {}},
            "error": _PHASE3_BASELINE_ERROR,
        }
        return _PHASE3_BASELINE_CACHE

    try:
        payload = json.loads(PHASE3_BASELINE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}

    _PHASE3_BASELINE_ERROR = ""
    _PHASE3_BASELINE_CACHE = {
        "task1": payload.get("task1", {"macro": {}}),
        "task2": payload.get("task2", {"macro": {}}),
        "task3": payload.get("task3", {"macro": {}}),
        "error": _PHASE3_BASELINE_ERROR,
    }
    return _PHASE3_BASELINE_CACHE


def _get_phase3_analysis(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE3_ANALYSIS_CACHE, _PHASE3_ANALYSIS_ERROR
    if _PHASE3_ANALYSIS_CACHE and not force_refresh:
        return _PHASE3_ANALYSIS_CACHE

    if not PHASE3_ANALYSIS_PATH.exists():
        _PHASE3_ANALYSIS_ERROR = f"Missing {PHASE3_ANALYSIS_PATH}"
        _PHASE3_ANALYSIS_CACHE = {
            "protocols": [],
            "generated_at": None,
            "error": _PHASE3_ANALYSIS_ERROR,
        }
        return _PHASE3_ANALYSIS_CACHE

    try:
        payload = json.loads(PHASE3_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}

    _PHASE3_ANALYSIS_ERROR = ""
    _PHASE3_ANALYSIS_CACHE = {
        "protocols": payload.get("protocols", []),
        "generated_at": payload.get("generated_at"),
        "error": _PHASE3_ANALYSIS_ERROR,
    }
    return _PHASE3_ANALYSIS_CACHE


def _get_phase3_alternatives(force_refresh: bool) -> Dict[str, Any]:
    global _PHASE3_ALTERNATIVES_CACHE, _PHASE3_ALTERNATIVES_ERROR
    if _PHASE3_ALTERNATIVES_CACHE and not force_refresh:
        return _PHASE3_ALTERNATIVES_CACHE

    if not PHASE3_ALTERNATIVES_PATH.exists():
        _PHASE3_ALTERNATIVES_ERROR = f"Missing {PHASE3_ALTERNATIVES_PATH}"
        _PHASE3_ALTERNATIVES_CACHE = {
            "generated_at": None,
            "settings": {},
            "stop_continue": {"macro": {}, "baseline": {}, "global": {}, "protocols": []},
            "ordinal_progress": {"macro": {}, "baseline": {}, "global": {}, "protocols": []},
            "ranking_correlation": {"macro": {}, "baseline": {}, "protocols": []},
            "next_focus_recommender": {"macro": {}, "baseline": {}, "protocols": []},
            "error": _PHASE3_ALTERNATIVES_ERROR,
        }
        return _PHASE3_ALTERNATIVES_CACHE

    try:
        payload = json.loads(PHASE3_ALTERNATIVES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}

    _PHASE3_ALTERNATIVES_ERROR = ""
    _PHASE3_ALTERNATIVES_CACHE = {
        "generated_at": payload.get("generated_at"),
        "settings": payload.get("settings", {}),
        "stop_continue": payload.get("stop_continue", {"macro": {}, "baseline": {}, "global": {}, "protocols": []}),
        "ordinal_progress": payload.get("ordinal_progress", {"macro": {}, "baseline": {}, "global": {}, "protocols": []}),
        "ranking_correlation": payload.get("ranking_correlation", {"macro": {}, "baseline": {}, "protocols": []}),
        "next_focus_recommender": payload.get("next_focus_recommender", {"macro": {}, "baseline": {}, "protocols": []}),
        "error": _PHASE3_ALTERNATIVES_ERROR,
    }
    return _PHASE3_ALTERNATIVES_CACHE


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path == "/":
            qs = parse_qs(parsed.query)
            force_refresh = "refresh" in qs
            min_protocol_plans = _parse_int(
                qs.get("min_protocol_plans", [str(DEFAULT_MIN_PROTOCOL_PLANS)])[0],
                DEFAULT_MIN_PROTOCOL_PLANS,
            )
            selected_protocol = qs.get("protocol", [None])[0]
            if selected_protocol is not None and not isinstance(selected_protocol, str):
                selected_protocol = _stringify(selected_protocol)
            if isinstance(selected_protocol, str):
                selected_protocol = selected_protocol.strip()
            selected_plan_idx = None
            raw_plan_idx = qs.get("plan_idx", [None])[0]
            if raw_plan_idx is not None:
                try:
                    selected_plan_idx = int(raw_plan_idx)
                except (TypeError, ValueError):
                    selected_plan_idx = None
            protocol_detail_html = ""
            if selected_protocol:
                try:
                    protocol_detail = compute_protocol_detail(
                        DEFAULT_MONGODB_URI,
                        "planeval",
                        "evaluations",
                        selected_protocol,
                        selected_plan_idx,
                    )
                    protocol_detail_html = _render_protocol_detail(
                        protocol_detail, min_protocol_plans
                    )
                except Exception as exc:  # noqa: BLE001 - display error in UI
                    protocol_detail_html = (
                        "<div class=\"protocol-detail\">"
                        f"<div class=\"empty\">Protocol load error: {html.escape(str(exc))}</div>"
                        "</div>"
                    )
            metrics = _get_metrics(force_refresh)
            phase2_metrics = _get_phase2_metrics(force_refresh)
            phase3_metrics = _get_phase3_metrics(force_refresh)
            phase3_sweep = _get_phase3_sweep(force_refresh)
            phase3_baselines = _get_phase3_baselines(force_refresh)
            phase3_analysis = _get_phase3_analysis(force_refresh)
            phase3_alternatives = _get_phase3_alternatives(force_refresh)
            logo_uri = _select_logo_uri()
            html_payload = _render_dashboard(
                metrics,
                phase2_metrics,
                phase3_metrics,
                phase3_sweep,
                phase3_baselines,
                phase3_analysis,
                phase3_alternatives,
                min_protocol_plans,
                logo_uri,
                selected_protocol,
                protocol_detail_html,
            )

            if _METRICS_ERROR:
                html_payload = html_payload.replace(
                    "</header>",
                    f"<p style=\"color: var(--academic-red); margin-top: 8px;\">"
                    f"Data load error: {_METRICS_ERROR}</p></header>",
                )

            self._send_response(200, "text/html", html_payload.encode("utf-8"))
            return

        if parsed.path == "/metrics.json":
            metrics = _get_metrics(False)
            payload = json.dumps(metrics, indent=2, default=str)
            self._send_response(200, "application/json", payload.encode("utf-8"))
            return

        if parsed.path.startswith("/static/"):
            static_name = parsed.path.replace("/static/", "", 1)
            static_path = ROOT_DIR / static_name
            if static_path.exists() and static_path.is_file():
                content = _read_static(static_path)
                self._send_response(200, _get_static_mime(static_path), content)
                return
            self._send_response(404, "text/plain", b"Not found")
            return

        self._send_response(404, "text/plain", b"Not found")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - required
        return

    def _send_response(self, code: int, mime: str, content: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            return


def run_server(port: int) -> None:
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        port = _parse_int(sys.argv[1], DEFAULT_PORT)
    run_server(port)
