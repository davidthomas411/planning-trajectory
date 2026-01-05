#!/usr/bin/env python3
"""Build a static project page in docs/index.html for GitHub Pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = ROOT / "data/derived"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

PHASE2_SUMMARY_PATH = DERIVED_DIR / "phase2_summary.json"
PHASE3_METRICS_PATH = DERIVED_DIR / "phase3_metrics.json"
PHASE3_BASELINES_PATH = DERIVED_DIR / "phase3_baselines.json"
ABSTRACT_PATH = ROOT / "draft_abstract.md"
CONSTRAINTS_PATH = DERIVED_DIR / "constraint_features.jsonl"
LOGO_SOURCE = ROOT / "jefferson-university-2.svg"
LOGO_TARGET = DOCS_DIR / "assets" / "tju-logo-j.svg"


INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


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


def _count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _markdown_to_html(text: str) -> str:
    lines = text.strip().splitlines()
    html_lines: List[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{stripped[2:].strip()}</h2>")
            continue
        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[3:].strip()}</h3>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item = stripped[2:].strip()
            item = INLINE_BOLD_RE.sub(lambda match: f"<strong>{match.group(1)}</strong>", item)
            html_lines.append(f"<li>{item}</li>")
            continue
        content = INLINE_BOLD_RE.sub(lambda match: f"<strong>{match.group(1)}</strong>", stripped)
        html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _figure_block(filename: str, title: str, caption: str) -> str:
    path = FIG_DIR / filename
    if not path.exists():
        return ""
    return (
        "<figure class=\"figure-card\">"
        f"<button class=\"figure-button\" data-src=\"figures/{filename}\" data-title=\"{title}\" aria-label=\"Open figure\">"
        f"<img src=\"figures/{filename}\" alt=\"{title}\" loading=\"lazy\">"
        "</button>"
        f"<figcaption><strong>{title}</strong><span>{caption}</span></figcaption>"
        "</figure>"
    )


def main() -> None:
    phase2 = _load_json(PHASE2_SUMMARY_PATH)
    phase3 = _load_json(PHASE3_METRICS_PATH)
    baselines = _load_json(PHASE3_BASELINES_PATH)

    abstract = ABSTRACT_PATH.read_text() if ABSTRACT_PATH.exists() else ""
    abstract_html = _markdown_to_html(abstract) if abstract else "<p>Draft abstract not available.</p>"

    constraints = _count_lines(CONSTRAINTS_PATH)

    task1 = phase3.get("task1", {}).get("macro", {})
    task2 = phase3.get("task2", {}).get("macro", {})
    task3 = phase3.get("task3", {}).get("macro", {})
    base1 = baselines.get("task1", {}).get("macro", {})
    base2 = baselines.get("task2", {}).get("macro", {})

    updated_at = phase3.get("generated_at") or phase2.get("generated_at") or "n/a"

    glance = [
        ("Approved plans", _format_int(phase2.get("qualified_plans"))),
        ("Evaluation attempts", _format_int(phase2.get("attempts_written"))),
        ("Constraint evaluations", _format_int(constraints)),
        ("Protocols", _format_int(phase2.get("protocols"))),
    ]

    metric_cards = [
        (
            "Q1: Next iteration better",
            f"Accuracy {_format_pct(task1.get('accuracy'))}",
            f"Baseline {_format_pct(base1.get('accuracy'))}",
        ),
        (
            "Q2: Next structure family",
            f"Top-3 {_format_pct(task3.get('top3_accuracy'))}",
            f"Top-5 {_format_pct(task3.get('top5_accuracy'))}",
        ),
        (
            "Q3: Remaining iterations",
            f"MAE {_format_float(task2.get('mae'))}",
            f"Baseline {_format_float(base2.get('mae'))}",
        ),
    ]

    figures = "\n".join(
        block
        for block in [
            _figure_block(
                "dataset_overview.svg",
                "Dataset overview",
                "Aggregated counts from qualified plans and evaluations.",
            ),
            _figure_block(
                "q1_accuracy.svg",
                "Q1 accuracy",
                "Model vs baseline for next-iteration improvement.",
            ),
            _figure_block(
                "q2_top3.svg",
                "Q2 top-3 accuracy",
                "Structure family prediction success rate.",
            ),
            _figure_block(
                "q3_mae.svg",
                "Q3 remaining iterations",
                "Mean absolute error vs baseline (lower is better).",
            ),
            _figure_block(
                "top_bottom_q1_accuracy.svg",
                "Top vs bottom protocols",
                "Protocol variability for Q1 performance.",
            ),
        ]
        if block
    )

    script = """<script>
    const lightbox = document.getElementById("lightbox");
    const lightboxImage = document.querySelector(".lightbox-image");
    const lightboxTitle = document.querySelector(".lightbox-title");
    const buttons = document.querySelectorAll(".figure-button");

    const closeLightbox = () => {
      lightbox.classList.remove("open");
      lightbox.setAttribute("aria-hidden", "true");
      lightboxImage.src = "";
      lightboxImage.alt = "";
      lightboxTitle.textContent = "";
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const src = button.dataset.src;
        const title = button.dataset.title || "Figure";
        lightboxImage.src = src;
        lightboxImage.alt = title;
        lightboxTitle.textContent = title;
        lightbox.classList.add("open");
        lightbox.setAttribute("aria-hidden", "false");
      });
    });

    lightbox.addEventListener("click", (event) => {
      const target = event.target;
      if (target && target.dataset && target.dataset.close) {
        closeLightbox();
      }
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeLightbox();
      }
    });
  </script>"""

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta name=\"description\" content=\"Planning trajectory learning: DVH evaluation analysis, dashboard, and abstract figures.\">
  <title>Planning Trajectory Learning</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap\" rel=\"stylesheet\">
  <link rel=\"stylesheet\" href=\"site.css\">
</head>
<body>
  <div class=\"page\">
    <header class=\"hero\">
      <div>
        <div class=\"brand\">
          <img class=\"logo\" src=\"assets/tju-logo-j.svg\" alt=\"Thomas Jefferson University\">
          <p class=\"tag\">Planning Trajectory Learning</p>
        </div>
        <h1>From DVH evaluations to protocol-specific decision support</h1>
        <p class=\"subtitle\">This project turns iterative plan evaluations into interpretable models that highlight when plans improve, what structure family to focus on next, and when further iterations are unlikely to help.</p>
        <div class=\"hero-meta\">
          <span>Last refreshed: {updated_at}</span>
          <span>GitHub: <a href=\"https://github.com/davidthomas411/planning-trajectory\">planning-trajectory</a></span>
        </div>
      </div>
      <div class=\"hero-card\">
        <h3>Project at a glance</h3>
        <div class=\"glance-grid\">
          {''.join([f'<div class="glance-item"><span>{label}</span><strong>{value}</strong></div>' for label, value in glance])}
        </div>
      </div>
    </header>

    <section class=\"metrics\">
      {''.join([f'<div class="metric-card"><h4>{title}</h4><p>{line1}</p><p class="muted">{line2}</p></div>' for title, line1, line2 in metric_cards])}
    </section>

    <section class=\"figures\">
      <div class=\"section-header\">
        <h2>Figures</h2>
        <p>Aggregated summaries for the abstract and dashboard.</p>
      </div>
      <div class=\"figure-grid\">
        {figures}
      </div>
    </section>

    <section class=\"abstract\">
      <div class=\"section-header\">
        <h2>Draft Abstract</h2>
        <p>Updated from draft_abstract.md.</p>
      </div>
      <div class=\"abstract-body\">
        {abstract_html}
      </div>
    </section>

    <section class=\"next-steps\">
      <div class=\"section-header\">
        <h2>How to refresh this page</h2>
        <p>Run locally after you regenerate the data.</p>
      </div>
      <div class=\"code-block\">
        <code>
python3 scripts/render_figures.py
python3 scripts/update_readme.py
python3 scripts/build_site.py
        </code>
      </div>
    </section>

    <div class=\"lightbox\" id=\"lightbox\" aria-hidden=\"true\" role=\"dialog\">
      <div class=\"lightbox-backdrop\" data-close=\"true\"></div>
      <div class=\"lightbox-content\" role=\"document\">
        <button class=\"lightbox-close\" data-close=\"true\" aria-label=\"Close\">x</button>
        <img class=\"lightbox-image\" src=\"\" alt=\"\">
        <p class=\"lightbox-title\"></p>
      </div>
    </div>

    <footer>
      <p>All figures are aggregated and contain no patient identifiers. Data access is read-only.</p>
    </footer>
  </div>
  {script}
</body>
</html>
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    LOGO_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if LOGO_SOURCE.exists():
        copy2(LOGO_SOURCE, LOGO_TARGET)
    (DOCS_DIR / "index.html").write_text(html)
    print("docs/index.html written")


if __name__ == "__main__":
    main()
