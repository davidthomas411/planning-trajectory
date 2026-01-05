import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.phase2_dataset import build_phase2_dataset
from src.data.phase3_modeling import run_phase3_modeling

DEFAULT_THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
DEFAULT_MIN_PROTOCOLS = 20
DEFAULT_MIN_PLANS_PER_PROTOCOL = 20


@dataclass(frozen=True)
class SweepResult:
    coverage_threshold: float
    protocol_count: int
    task1_auc: Optional[float]
    task1_accuracy: Optional[float]
    task2_mae: Optional[float]
    task3_accuracy: Optional[float]
    elapsed_seconds: float


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def run_coverage_sweep(
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    min_protocols: int = DEFAULT_MIN_PROTOCOLS,
    min_plans_per_protocol: int = DEFAULT_MIN_PLANS_PER_PROTOCOL,
    include_family_features: bool = True,
    output_dir: str = "data/derived",
    output_path: Optional[Path] = None,
    mongo_uri: Optional[str] = None,
) -> Dict[str, Any]:
    thresholds = list(thresholds)
    output_path = output_path or (ROOT_DIR / output_dir / "coverage_sweep.json")

    results: List[SweepResult] = []
    best: Optional[SweepResult] = None

    print(
        f"[Sweep] Starting coverage sweep: thresholds={thresholds} "
        f"min_protocols>={min_protocols}, min_plans_per_protocol={min_plans_per_protocol}"
    )

    for threshold in thresholds:
        start = time.time()
        print(f"[Sweep] Building Phase 2 (coverage={threshold:.2f})")
        build_phase2_dataset(
            mongo_uri=mongo_uri,
            min_coverage_pct=threshold,
            output_dir=output_dir,
            print_summary=False,
        )
        print(f"[Sweep] Running Phase 3 (coverage={threshold:.2f})")
        metrics = run_phase3_modeling(
            min_plans_per_protocol=min_plans_per_protocol,
            include_family_features=include_family_features,
            write_output=False,
            verbose=False,
        )
        protocol_count = len(metrics.get("task1", {}).get("protocols", []))
        task1_auc = metrics.get("task1", {}).get("macro", {}).get("auc")
        task1_acc = metrics.get("task1", {}).get("macro", {}).get("accuracy")
        task2_mae = metrics.get("task2", {}).get("macro", {}).get("mae")
        task3_acc = metrics.get("task3", {}).get("macro", {}).get("accuracy")
        elapsed = time.time() - start

        result = SweepResult(
            coverage_threshold=threshold,
            protocol_count=protocol_count,
            task1_auc=task1_auc,
            task1_accuracy=task1_acc,
            task2_mae=task2_mae,
            task3_accuracy=task3_acc,
            elapsed_seconds=elapsed,
        )
        results.append(result)

        print(
            f"[Sweep] coverage={threshold:.2f} "
            f"protocols={protocol_count} "
            f"task1_auc={_format_float(task1_auc)} "
            f"task1_acc={_format_float(task1_acc)} "
            f"task2_mae={_format_float(task2_mae)} "
            f"task3_acc={_format_float(task3_acc)} "
            f"elapsed={elapsed:.1f}s"
        )

        if protocol_count >= min_protocols and task1_auc is not None:
            if best is None or task1_auc > (best.task1_auc or -1):
                best = result

    if best is None:
        print("[Sweep] No threshold met the minimum protocol count.")
    else:
        print(
            f"[Sweep] Best threshold: {best.coverage_threshold:.2f} "
            f"(task1_auc={_format_float(best.task1_auc)}, protocols={best.protocol_count})"
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "min_protocols_required": min_protocols,
        "min_plans_per_protocol": min_plans_per_protocol,
        "include_family_features": include_family_features,
        "thresholds": [
            {
                "coverage_threshold": result.coverage_threshold,
                "protocols": result.protocol_count,
                "task1_auc": result.task1_auc,
                "task1_accuracy": result.task1_accuracy,
                "task2_mae": result.task2_mae,
                "task3_accuracy": result.task3_accuracy,
                "elapsed_seconds": result.elapsed_seconds,
            }
            for result in results
        ],
        "best_threshold": best.coverage_threshold if best else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Sweep] Results written to {output_path}")

    if best is not None:
        print(f"[Sweep] Rebuilding Phase 2 for best coverage={best.coverage_threshold:.2f}")
        build_phase2_dataset(
            mongo_uri=mongo_uri,
            min_coverage_pct=best.coverage_threshold,
            output_dir=output_dir,
            print_summary=False,
        )
        print("[Sweep] Writing Phase 3 metrics for best threshold")
        run_phase3_modeling(
            min_plans_per_protocol=min_plans_per_protocol,
            include_family_features=include_family_features,
            write_output=True,
            verbose=True,
        )

    return payload


if __name__ == "__main__":
    run_coverage_sweep()
