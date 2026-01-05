#!/usr/bin/env python3
"""Export derived JSONL files to CSV for student handoff."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ATTEMPT_PREFERRED_ORDER = [
    "protocol_name",
    "patient_id",
    "plan_id",
    "attempt_number",
    "attempt_index",
    "attempt_count",
    "attempt_progress",
    "created_at",
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
    "plan_score",
    "label_stop",
    "future_best_delta",
]

CONSTRAINT_PREFERRED_ORDER = [
    "protocol_name",
    "patient_id",
    "plan_id",
    "attempt_number",
    "structure",
    "structure_tg263",
    "metric_display",
    "metric_type",
    "metric_subtype",
    "priority",
    "goal_operator",
    "goal_value",
    "achieved_value",
    "pass",
    "margin",
    "normalized_margin",
    "near_limit",
    "percentile",
]


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _load_jsonl(path: Path, limit: int | None = None) -> Iterable[Dict]:
    with path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if limit is not None and index >= limit:
                break
            yield json.loads(line)


def _field_order(keys: List[str], preferred: List[str]) -> List[str]:
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key not in ordered))
    return ordered


def _write_csv(output_path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})
            count += 1
    return count


def _export_one(name: str, input_path: Path, output_path: Path, preferred: List[str], limit: int | None) -> int:
    rows = list(_load_jsonl(input_path, limit=limit))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    keys = sorted({key for row in rows for key in row.keys()})
    fieldnames = _field_order(keys, preferred)
    row_count = _write_csv(output_path, rows, fieldnames)
    print(f"Exported {row_count} rows to {output_path}")
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phase-2 JSONL files to CSV.")
    parser.add_argument(
        "--attempts-jsonl",
        default="data/derived/plan_attempt_features.jsonl",
        help="Path to plan attempt JSONL file.",
    )
    parser.add_argument(
        "--constraints-jsonl",
        default="data/derived/constraint_features.jsonl",
        help="Path to constraint JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/derived/exports",
        help="Output directory for CSV exports.",
    )
    parser.add_argument(
        "--only",
        choices=["attempts", "constraints", "all"],
        default="all",
        help="Which exports to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick testing.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.only in {"attempts", "all"}:
        _export_one(
            "attempts",
            Path(args.attempts_jsonl),
            out_dir / "plan_attempts.csv",
            ATTEMPT_PREFERRED_ORDER,
            args.limit,
        )
    if args.only in {"constraints", "all"}:
        _export_one(
            "constraints",
            Path(args.constraints_jsonl),
            out_dir / "constraint_evaluations.csv",
            CONSTRAINT_PREFERRED_ORDER,
            args.limit,
        )
    print(f"Export completed at {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
