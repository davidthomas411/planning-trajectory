import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymongo import MongoClient

DEFAULT_MONGODB_URI = os.getenv(
    "PLANEVAL_MONGODB_URI",
    "mongodb://analyst:.hLeUaWqd2xu3ath3K.B~Uej@10.187.138.252:27017/planeval?authSource=admin",
)


def _sort_attempts(attempts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    attempts = list(attempts)
    if not attempts:
        return []

    has_all_attempt_numbers = all(
        "attempt_number" in attempt and attempt["attempt_number"] is not None
        for attempt in attempts
    )
    if has_all_attempt_numbers:
        return sorted(attempts, key=lambda attempt: attempt["attempt_number"])

    has_all_created_at = all(
        "created_at" in attempt and attempt["created_at"] is not None
        for attempt in attempts
    )
    if has_all_created_at:
        return sorted(attempts, key=lambda attempt: attempt["created_at"])

    def fallback_key(attempt: Dict[str, Any]) -> Tuple[int, Any]:
        attempt_number = attempt.get("attempt_number")
        if attempt_number is not None:
            return (0, attempt_number)
        created_at = attempt.get("created_at")
        if created_at is not None:
            return (1, created_at)
        updated_at = attempt.get("updated_at")
        if updated_at is not None:
            return (2, updated_at)
        return (3, 0)

    return sorted(attempts, key=fallback_key)


def _print_summary(
    trajectories: Dict[Tuple[str, str], List[Dict[str, Any]]]
) -> None:
    plan_count = len(trajectories)
    iteration_counts = Counter(len(attempts) for attempts in trajectories.values())
    iteration_distribution = dict(sorted(iteration_counts.items()))

    print("Plan count:", plan_count)
    print("Iteration count distribution:", iteration_distribution)

    if not trajectories:
        print("Example key format: <none>")
        print("Example attempt fields: <none>")
        return

    example_key = next(iter(trajectories.keys()))
    example_attempt = trajectories[example_key][0]
    example_fields = sorted(example_attempt.keys())

    print("Example key format:", ("<patient_id>", "<plan_id>"))
    print("Example attempt fields:", example_fields)


def load_plan_trajectories(
    mongo_uri: Optional[str] = None,
    db_name: str = "planeval",
    collection_name: str = "evaluations",
    min_attempts: int = 2,
    print_summary: bool = True,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Load plan trajectories from MongoDB and return a mapping of
    (patient_id, plan_id) -> [attempt_1, attempt_2, ...].
    """
    uri = mongo_uri or DEFAULT_MONGODB_URI
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    collection = client[db_name][collection_name]

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
                "last_attempt_number": {"$last": "$attempt_number"},
                "last_is_approved": {"$last": "$approval.is_approved"},
                "attempts": {"$push": "$$ROOT"},
            }
        },
        {
            "$match": {
                "attempt_count": {"$gte": min_attempts},
                "last_is_approved": True,
            }
        },
    ]

    trajectories: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    cursor = collection.aggregate(pipeline, allowDiskUse=True)
    for doc in cursor:
        patient_id = doc["_id"]["patient_id"]
        plan_id = doc["_id"]["plan_id"]
        attempts = _sort_attempts(doc["attempts"])
        trajectories[(patient_id, plan_id)] = attempts

    client.close()

    if print_summary:
        _print_summary(trajectories)

    return trajectories
