#!/usr/bin/env python3
"""Render summary SVG figures from derived metrics.

Outputs are safe to commit (aggregated only) and used in README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = ROOT / "data/derived"
FIG_DIR = ROOT / "docs/figures"

PHASE2_SUMMARY_PATH = DERIVED_DIR / "phase2_summary.json"
PHASE3_METRICS_PATH = DERIVED_DIR / "phase3_metrics.json"
PHASE3_BASELINES_PATH = DERIVED_DIR / "phase3_baselines.json"
CONSTRAINTS_PATH = DERIVED_DIR / "constraint_features.jsonl"

COLOR_BG = "#ffffff"
COLOR_GRID = "#e6e9ef"
COLOR_TEXT = "#1d2433"
COLOR_MUTED = "#5b6575"
COLOR_MODEL = "#2f6fed"
COLOR_BASELINE = "#93a4bf"
COLOR_CARD = "#f7f8fb"
COLOR_CARD_BORDER = "#d7dde8"


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


def _format_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{COLOR_BG}" />',
    ]


def _svg_footer() -> str:
    return "</svg>"


def _text(x: float, y: float, text: str, size: int = 14, weight: str = "400", color: str = COLOR_TEXT, anchor: str = "start") -> str:
    safe_text = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" font-family="IBM Plex Sans, Segoe UI, sans-serif">'
        f"{safe_text}</text>"
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str = COLOR_GRID, width: float = 1.0) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" />'


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: Optional[str] = None, radius: int = 0) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    radius_attr = f' rx="{radius}" ry="{radius}"' if radius else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{stroke_attr}{radius_attr} />'


def _count_constraints(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _write_svg(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def render_dataset_overview(phase2: Dict[str, Any]) -> None:
    width, height = 980, 240
    padding = 24
    gap = 16
    card_w = (width - padding * 2 - gap * 3) / 4
    card_h = 140
    start_y = 64

    plans = phase2.get("qualified_plans")
    attempts = phase2.get("attempts_written")
    protocols = phase2.get("protocols")
    constraints = _count_constraints(CONSTRAINTS_PATH)

    cards = [
        ("Qualified plans", _format_int(plans)),
        ("Evaluation attempts", _format_int(attempts)),
        ("Protocols", _format_int(protocols)),
        ("Constraint evaluations", _format_int(constraints)),
    ]

    lines = _svg_header(width, height)
    lines.append(_text(padding, 36, "Dataset Overview", size=20, weight="600"))

    for idx, (label, value) in enumerate(cards):
        x = padding + idx * (card_w + gap)
        y = start_y
        lines.append(_rect(x, y, card_w, card_h, COLOR_CARD, stroke=COLOR_CARD_BORDER, radius=14))
        lines.append(_text(x + 16, y + 44, value, size=28, weight="700"))
        lines.append(_text(x + 16, y + 78, label, size=13, weight="500", color=COLOR_MUTED))

    lines.append(_svg_footer())
    _write_svg(FIG_DIR / "dataset_overview.svg", lines)


def render_grouped_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    model_values: Sequence[Optional[float]],
    baseline_values: Sequence[Optional[float]],
    value_format,
    y_max: Optional[float] = None,
    y_label: str = "",
    note: Optional[str] = None,
) -> None:
    width, height = 980, 360
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 70, 70

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_value = max([v for v in list(model_values) + list(baseline_values) if v is not None] or [1.0])
    if y_max is None:
        y_max = max_value * 1.2
    y_max = max(y_max, 1e-6)

    group_count = len(labels)
    group_width = plot_w / max(group_count, 1)
    bar_width = min(48, group_width / 3)
    bar_gap = bar_width * 0.4

    lines = _svg_header(width, height)
    lines.append(_text(margin_left, 36, title, size=20, weight="600"))
    if note:
        lines.append(_text(margin_left, 56, note, size=12, weight="400", color=COLOR_MUTED))

    # Grid + axis labels
    ticks = 5
    for i in range(ticks + 1):
        y_value = (y_max / ticks) * i
        y = margin_top + plot_h - (y_value / y_max) * plot_h
        lines.append(_line(margin_left, y, margin_left + plot_w, y))
        lines.append(_text(margin_left - 8, y + 4, value_format(y_value), size=11, color=COLOR_MUTED, anchor="end"))

    if y_label:
        lines.append(_text(margin_left, margin_top - 12, y_label, size=12, color=COLOR_MUTED))

    for idx, label in enumerate(labels):
        group_x = margin_left + idx * group_width + group_width / 2
        model = model_values[idx]
        baseline = baseline_values[idx]

        bars = [(model, COLOR_MODEL, -bar_width / 2 - bar_gap / 2), (baseline, COLOR_BASELINE, bar_width / 2 + bar_gap / 2)]
        for value, color, offset in bars:
            if value is None:
                continue
            bar_h = (value / y_max) * plot_h
            x = group_x + offset - bar_width / 2
            y = margin_top + plot_h - bar_h
            lines.append(_rect(x, y, bar_width, bar_h, color, radius=6))
            lines.append(_text(x + bar_width / 2, y - 6, value_format(value), size=11, anchor="middle"))

        lines.append(_text(group_x, margin_top + plot_h + 28, label, size=12, anchor="middle"))

    # Legend
    legend_y = height - 24
    lines.append(_rect(margin_left, legend_y - 12, 12, 12, COLOR_MODEL, radius=3))
    lines.append(_text(margin_left + 18, legend_y - 2, "Model", size=12, color=COLOR_MUTED))
    lines.append(_rect(margin_left + 80, legend_y - 12, 12, 12, COLOR_BASELINE, radius=3))
    lines.append(_text(margin_left + 98, legend_y - 2, "Baseline", size=12, color=COLOR_MUTED))

    lines.append(_svg_footer())
    _write_svg(path, lines)


def render_top_bottom_chart(protocols: List[Dict[str, Any]]) -> None:
    items = []
    for item in protocols:
        name = item.get("protocol_name")
        acc = (item.get("metrics") or {}).get("accuracy")
        if name and acc is not None:
            items.append((name, float(acc)))

    if not items:
        return

    items.sort(key=lambda x: x[1])
    bottom = items[:5]
    top = items[-5:][::-1]
    rows = top + bottom

    width, height = 980, 420
    margin_left, margin_right = 220, 40
    margin_top, margin_bottom = 70, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_value = max([v for _, v in rows] or [1.0])
    y_step = plot_h / max(len(rows), 1)

    lines = _svg_header(width, height)
    lines.append(_text(margin_left, 36, "Top vs Bottom Protocols (Q1 Accuracy)", size=20, weight="600"))

    for idx, (name, value) in enumerate(rows):
        y = margin_top + idx * y_step + y_step * 0.2
        bar_h = y_step * 0.6
        bar_w = (value / max_value) * plot_w
        lines.append(_rect(margin_left, y, bar_w, bar_h, COLOR_MODEL, radius=6))
        lines.append(_text(margin_left - 12, y + bar_h * 0.7, name, size=11, anchor="end"))
        lines.append(_text(margin_left + bar_w + 6, y + bar_h * 0.7, _format_pct(value), size=11, color=COLOR_MUTED))

    # Axis note
    lines.append(_text(margin_left, height - 12, "Higher is better", size=11, color=COLOR_MUTED))

    lines.append(_svg_footer())
    _write_svg(FIG_DIR / "top_bottom_q1_accuracy.svg", lines)


def main() -> None:
    phase2 = _load_json(PHASE2_SUMMARY_PATH)
    phase3 = _load_json(PHASE3_METRICS_PATH)
    baselines = _load_json(PHASE3_BASELINES_PATH)

    render_dataset_overview(phase2)

    task1 = phase3.get("task1", {}).get("macro", {})
    task2 = phase3.get("task2", {}).get("macro", {})
    task3 = phase3.get("task3", {}).get("macro", {})

    base1 = baselines.get("task1", {}).get("macro", {})
    base2 = baselines.get("task2", {}).get("macro", {})
    base3 = baselines.get("task3", {}).get("macro", {})

    render_grouped_bar_chart(
        FIG_DIR / "q1_accuracy.svg",
        "Q1: Next Iteration Better?",
        ["Accuracy"],
        [task1.get("accuracy")],
        [base1.get("accuracy")],
        lambda v: _format_pct(v),
        y_max=1.0,
        y_label="Accuracy",
    )

    render_grouped_bar_chart(
        FIG_DIR / "q2_top3.svg",
        "Q2: Next Structure Family (Top-3)",
        ["Top-3"],
        [task3.get("top3_accuracy")],
        [base3.get("top3_accuracy")],
        lambda v: _format_pct(v),
        y_max=1.0,
        y_label="Top-3 accuracy",
    )

    render_grouped_bar_chart(
        FIG_DIR / "q3_mae.svg",
        "Q3: Remaining Iterations (MAE)",
        ["MAE"],
        [task2.get("mae")],
        [base2.get("mae")],
        lambda v: _format_float(v, 2),
        y_label="Mean absolute error",
        note="Lower is better",
    )

    render_top_bottom_chart(phase3.get("task1", {}).get("protocols", []))

    print("Figures rendered in docs/figures")


if __name__ == "__main__":
    main()
