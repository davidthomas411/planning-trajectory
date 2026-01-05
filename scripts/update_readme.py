#!/usr/bin/env python3
"""Update README with dashboard snapshot and abstract text.

This reads aggregated metrics from data/derived and the draft abstract text,
then injects them into README.md between marked sections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
ABSTRACT_PATH = ROOT / "draft_abstract.md"
PHASE2_SUMMARY_PATH = ROOT / "data/derived/phase2_summary.json"
PHASE3_METRICS_PATH = ROOT / "data/derived/phase3_metrics.json"
PHASE3_BASELINES_PATH = ROOT / "data/derived/phase3_baselines.json"
PHASE3_ALTERNATIVES_PATH = ROOT / "data/derived/phase3_alternatives.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _format_int(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _top_protocols_by_count(protocols: List[Dict[str, Any]], n: int = 5) -> List[Tuple[str, int]]:
    rows = []
    for item in protocols:
        name = item.get("protocol_name")
        count = item.get("plan_count")
        if not name or count is None:
            continue
        rows.append((name, int(count)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]


def _build_snapshot() -> str:
    phase2 = _load_json(PHASE2_SUMMARY_PATH)
    phase3 = _load_json(PHASE3_METRICS_PATH)
    baselines = _load_json(PHASE3_BASELINES_PATH)
    alternatives = _load_json(PHASE3_ALTERNATIVES_PATH)

    lines: List[str] = []
    refreshed = phase3.get("generated_at") or phase2.get("generated_at") or "n/a"
    lines.append(f"Last refreshed: {refreshed}")

    if phase2:
        lines.append("")
        lines.append("Dataset (Phase 2)")
        lines.append(f"- Qualified plans: {_format_int(phase2.get('qualified_plans'))}")
        lines.append(f"- Evaluation attempts: {_format_int(phase2.get('attempts_written'))}")
        lines.append(f"- Protocols observed: {_format_int(phase2.get('protocols'))}")
        lines.append(
            f"- Minimum coverage: {_format_pct(phase2.get('min_coverage_pct'))}"
        )
        lines.append(f"- Minimum attempts: {_format_int(phase2.get('min_attempts'))}")
        lines.append(f"- Plateau delta: {_format_float(phase2.get('plateau_delta'), 1)}")

    if phase3:
        lines.append("")
        lines.append("Modeling (Phase 3, macro averages)")
        task1 = phase3.get("task1", {}).get("macro", {})
        task2 = phase3.get("task2", {}).get("macro", {})
        task3 = phase3.get("task3", {}).get("macro", {})
        base1 = baselines.get("task1", {}).get("macro", {})
        base2 = baselines.get("task2", {}).get("macro", {})
        base3 = baselines.get("task3", {}).get("macro", {})

        lines.append(
            "- Q1: Next iteration better? "
            f"accuracy {_format_pct(task1.get('accuracy'))} (baseline {_format_pct(base1.get('accuracy'))}), "
            f"AUC {_format_float(task1.get('auc'))}"
        )
        lines.append(
            "- Q2: Which structure family improves next? "
            f"top-1 {_format_pct(task3.get('accuracy'))}, "
            f"top-3 {_format_pct(task3.get('top3_accuracy'))}, "
            f"top-5 {_format_pct(task3.get('top5_accuracy'))}"
        )
        lines.append(
            "- Q3: Remaining iterations (MAE) "
            f"{_format_float(task2.get('mae'))} (baseline {_format_float(base2.get('mae'))})"
        )

        stop = alternatives.get("stop_continue", {}).get("macro", {})
        if stop:
            lines.append(
                "- Stop/continue classifier (alternate): "
                f"accuracy {_format_pct(stop.get('accuracy'))}, "
                f"balanced {_format_pct(stop.get('balanced_accuracy'))}, "
                f"AUC {_format_float(stop.get('auc'))}"
            )

        protocol_count = phase3.get("protocol_count")
        min_plans = phase3.get("min_plans_per_protocol")
        split = phase3.get("splits", {})
        lines.append("")
        lines.append("Model configuration")
        lines.append(f"- Protocols modeled: {_format_int(protocol_count)}")
        lines.append(f"- Minimum plans per protocol: {_format_int(min_plans)}")
        lines.append(
            "- Train/val/test split: "
            f"{split.get('train', 'n/a')}/"
            f"{split.get('val', 'n/a')}/"
            f"{split.get('test', 'n/a')} (seed {split.get('seed', 'n/a')})"
        )

        top_protocols = _top_protocols_by_count(phase3.get("task1", {}).get("protocols", []))
        if top_protocols:
            lines.append("")
            lines.append("Top protocols by plan count (modeled)")
            for name, count in top_protocols:
                lines.append(f"- {name}: {count} plans")

    lines.append("")
    lines.append("Note: values are aggregated; no patient identifiers are included.")
    return "\n".join(lines)


def _replace_block(text: str, marker: str, new_block: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start not in text or end not in text:
        raise ValueError(f"Missing markers for {marker}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{new_block.strip()}\n{end}{after}"


def main() -> None:
    readme = README_PATH.read_text()
    snapshot = _build_snapshot()
    abstract_text = ABSTRACT_PATH.read_text().strip() if ABSTRACT_PATH.exists() else ""
    abstract_block = abstract_text or "(Draft abstract missing.)"

    readme = _replace_block(readme, "AUTO-SNAPSHOT", snapshot)
    readme = _replace_block(readme, "AUTO-ABSTRACT", abstract_block)

    README_PATH.write_text(readme)
    print("README updated.")


if __name__ == "__main__":
    main()
