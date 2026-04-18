from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.safety_guardrails import RuleBasedSafetyGuard  # noqa: E402


@dataclass
class SafetyMetrics:
    case_count: int = 0
    input_case_count: int = 0
    output_case_count: int = 0
    category_cases: int = 0
    category_hits: int = 0
    action_cases: int = 0
    action_hits: int = 0
    severity_cases: int = 0
    severity_hits: int = 0
    expected_hit_codes: int = 0
    hit_code_hits: int = 0
    forbidden_hit_codes: int = 0
    forbidden_hit_violations: int = 0
    expected_direct_response: int = 0
    direct_response_hits: int = 0
    expected_continue: int = 0
    continue_hits: int = 0
    unsafe_output_cases: int = 0
    unsafe_output_hits: int = 0
    safe_output_cases: int = 0
    safe_output_hits: int = 0
    rewrite_cases: int = 0
    rewrite_hits: int = 0
    rewrite_residual_unsafe: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "input_case_count": self.input_case_count,
            "output_case_count": self.output_case_count,
            "input_category_accuracy": ratio(self.category_hits, self.category_cases),
            "action_accuracy": ratio(self.action_hits, self.action_cases),
            "severity_accuracy": ratio(self.severity_hits, self.severity_cases),
            "expected_hit_code_recall": ratio(self.hit_code_hits, self.expected_hit_codes),
            "forbidden_hit_violation_rate": ratio(
                self.forbidden_hit_violations,
                self.forbidden_hit_codes,
            ),
            "direct_response_recall": ratio(
                self.direct_response_hits,
                self.expected_direct_response,
            ),
            "continue_pass_rate": ratio(self.continue_hits, self.expected_continue),
            "unsafe_output_recall": ratio(
                self.unsafe_output_hits,
                self.unsafe_output_cases,
            ),
            "safe_output_pass_rate": ratio(
                self.safe_output_hits,
                self.safe_output_cases,
            ),
            "rewrite_success_rate": ratio(self.rewrite_hits, self.rewrite_cases),
            "rewrite_residual_unsafe_count": self.rewrite_residual_unsafe,
            "failure_count": len(self.failures),
            "failures": self.failures[:100],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedGraphQA rule-based safety eval.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BACKEND_ROOT / "evals" / "datasets" / "safety_guardrails.jsonl",
        help="JSONL dataset path for safety guardrail eval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_ROOT / "evals" / "runs",
        help="Directory for JSON eval reports.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit cases for debugging. 0 means all cases.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Append one JSONL summary row to this file after each successful eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    started = time.perf_counter()

    cases = load_jsonl(args.dataset, args.max_cases)
    guard = RuleBasedSafetyGuard()
    metrics = evaluate_cases(guard, cases, show_progress=not args.no_progress)

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    report = {
        "run_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_ms": duration_ms,
        "config": {
            "dataset": str(args.dataset),
            "max_cases": args.max_cases or None,
            "guard": "RuleBasedSafetyGuard",
        },
        "metrics": metrics.to_report(),
        "group_metrics": group_metrics(cases, metrics.failures),
    }

    report_path = args.output_dir / f"safety_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report_path, report)
    history_path = args.history_file or (args.output_dir / "safety_eval_history.jsonl")
    append_history(history_path, report, report_path)
    print_summary(report, report_path, history_path)


def evaluate_cases(
    guard: RuleBasedSafetyGuard,
    cases: list[dict[str, Any]],
    show_progress: bool,
) -> SafetyMetrics:
    metrics = SafetyMetrics()
    for case in iter_progress(cases, "safety", show_progress):
        metrics.case_count += 1
        if case.get("type") == "input":
            metrics.input_case_count += 1
            evaluate_input_case(guard, case, metrics)
        elif case.get("type") == "output":
            metrics.output_case_count += 1
            evaluate_output_case(guard, case, metrics)
        else:
            append_failure(
                metrics,
                case,
                reason="unknown_case_type",
                actual={"type": case.get("type")},
            )
    return metrics


def evaluate_input_case(
    guard: RuleBasedSafetyGuard,
    case: dict[str, Any],
    metrics: SafetyMetrics,
) -> None:
    expected = case.get("expected") or {}
    actual = guard.assess_input(str(case.get("query") or "")).to_dict()
    actual_hit_codes = hit_codes(actual)

    check_field(metrics, case, expected, actual, "category", "category")
    check_field(metrics, case, expected, actual, "action", "action")
    check_field(metrics, case, expected, actual, "severity", "severity")
    check_hit_codes(metrics, case, expected, actual, actual_hit_codes)

    expected_action = expected.get("action")
    if expected_action == "direct_response":
        metrics.expected_direct_response += 1
        if actual.get("action") == "direct_response":
            metrics.direct_response_hits += 1
    elif expected_action == "continue":
        metrics.expected_continue += 1
        if actual.get("action") == "continue":
            metrics.continue_hits += 1


def evaluate_output_case(
    guard: RuleBasedSafetyGuard,
    case: dict[str, Any],
    metrics: SafetyMetrics,
) -> None:
    expected = case.get("expected") or {}
    result = guard.guard_output(
        str(case.get("answer") or ""),
        query=str(case.get("query") or ""),
        evidence=[str(item) for item in case.get("evidence") or []],
    )
    actual = result.to_dict()
    actual_hit_codes = hit_codes(actual)

    check_bool_field(metrics, case, expected, actual, "safe")
    check_field(metrics, case, expected, actual, "action", "action")
    check_hit_codes(metrics, case, expected, actual, actual_hit_codes)

    if expected.get("safe") is False:
        metrics.unsafe_output_cases += 1
        if actual.get("safe") is False:
            metrics.unsafe_output_hits += 1
    elif expected.get("safe") is True:
        metrics.safe_output_cases += 1
        if actual.get("safe") is True:
            metrics.safe_output_hits += 1

    if expected.get("action") == "rewrite":
        metrics.rewrite_cases += 1
        residual = guard.guard_output(
            result.answer,
            query=str(case.get("query") or ""),
            evidence=[str(item) for item in case.get("evidence") or []],
        )
        if residual.safe:
            metrics.rewrite_hits += 1
        else:
            metrics.rewrite_residual_unsafe += 1
            append_failure(
                metrics,
                case,
                reason="rewrite_residual_unsafe",
                actual={
                    "first_pass": actual,
                    "rewritten_answer": result.answer,
                    "second_pass": residual.to_dict(),
                },
            )


def check_field(
    metrics: SafetyMetrics,
    case: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    expected_key: str,
    metric_name: str,
) -> None:
    if expected_key not in expected:
        return
    expected_value = expected.get(expected_key)
    actual_value = actual.get(expected_key)
    if metric_name == "category":
        metrics.category_cases += 1
    elif metric_name == "action":
        metrics.action_cases += 1
    elif metric_name == "severity":
        metrics.severity_cases += 1

    if actual_value == expected_value:
        if metric_name == "category":
            metrics.category_hits += 1
        elif metric_name == "action":
            metrics.action_hits += 1
        elif metric_name == "severity":
            metrics.severity_hits += 1
        return

    append_failure(
        metrics,
        case,
        reason=f"{expected_key}_mismatch",
        actual=actual,
        expected={expected_key: expected_value},
    )


def check_bool_field(
    metrics: SafetyMetrics,
    case: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    key: str,
) -> None:
    if key not in expected:
        return
    if actual.get(key) == expected.get(key):
        return
    append_failure(
        metrics,
        case,
        reason=f"{key}_mismatch",
        actual=actual,
        expected={key: expected.get(key)},
    )


def check_hit_codes(
    metrics: SafetyMetrics,
    case: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    actual_hit_codes: set[str],
) -> None:
    for code in expected.get("hit_codes", []):
        metrics.expected_hit_codes += 1
        if code in actual_hit_codes:
            metrics.hit_code_hits += 1
        else:
            append_failure(
                metrics,
                case,
                reason="missing_hit_code",
                actual=actual,
                expected={"hit_code": code},
            )
    for code in expected.get("must_not_hit_codes", []):
        metrics.forbidden_hit_codes += 1
        if code in actual_hit_codes:
            metrics.forbidden_hit_violations += 1
            append_failure(
                metrics,
                case,
                reason="forbidden_hit_code",
                actual=actual,
                expected={"must_not_hit_code": code},
            )


def append_failure(
    metrics: SafetyMetrics,
    case: dict[str, Any],
    reason: str,
    actual: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> None:
    if len(metrics.failures) >= 100:
        return
    metrics.failures.append(
        {
            "id": case.get("id"),
            "type": case.get("type"),
            "group": case.get("group"),
            "reason": reason,
            "query": case.get("query"),
            "answer": case.get("answer"),
            "expected": expected or case.get("expected"),
            "actual": actual,
        }
    )


def hit_codes(result: dict[str, Any]) -> set[str]:
    return {
        str(item.get("code"))
        for item in result.get("hits", [])
        if isinstance(item, dict) and item.get("code")
    }


def load_jsonl(path: Path, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if max_cases and len(rows) >= max_cases:
                break
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_history(path: Path, report: dict[str, Any], report_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report.get("metrics", {})
    row = {
        "run_at": report.get("run_at"),
        "finished_at": report.get("finished_at"),
        "duration_ms": report.get("duration_ms"),
        "duration_seconds": round(float(report.get("duration_ms") or 0) / 1000, 3),
        "report_path": str(report_path),
        "config": report.get("config", {}),
        "metrics": {
            "case_count": metrics.get("case_count"),
            "input_case_count": metrics.get("input_case_count"),
            "output_case_count": metrics.get("output_case_count"),
            "input_category_accuracy": metrics.get("input_category_accuracy"),
            "action_accuracy": metrics.get("action_accuracy"),
            "expected_hit_code_recall": metrics.get("expected_hit_code_recall"),
            "forbidden_hit_violation_rate": metrics.get("forbidden_hit_violation_rate"),
            "direct_response_recall": metrics.get("direct_response_recall"),
            "continue_pass_rate": metrics.get("continue_pass_rate"),
            "unsafe_output_recall": metrics.get("unsafe_output_recall"),
            "safe_output_pass_rate": metrics.get("safe_output_pass_rate"),
            "rewrite_success_rate": metrics.get("rewrite_success_rate"),
            "failure_count": metrics.get("failure_count"),
        },
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def group_metrics(cases: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, int]] = {}
    for case in cases:
        group = str(case.get("group") or "unknown")
        by_group.setdefault(group, {"case_count": 0, "failure_count": 0})
        by_group[group]["case_count"] += 1
    for failure in failures:
        group = str(failure.get("group") or "unknown")
        by_group.setdefault(group, {"case_count": 0, "failure_count": 0})
        by_group[group]["failure_count"] += 1
    for item in by_group.values():
        item["pass_rate"] = ratio(
            item["case_count"] - item["failure_count"],
            item["case_count"],
        )
    return by_group


def iter_progress(
    cases: list[dict[str, Any]],
    label: str,
    enabled: bool,
):
    if not enabled:
        yield from cases
        return
    try:
        from tqdm import tqdm

        yield from tqdm(cases, desc=label, unit="case", dynamic_ncols=True, file=sys.stdout)
        return
    except Exception:
        pass

    total = len(cases)
    for idx, case in enumerate(cases, 1):
        print(f"\r{label}: {idx}/{total}", end="", flush=True)
        yield case
    print()


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def print_summary(report: dict[str, Any], report_path: Path, history_path: Path) -> None:
    metrics = report["metrics"]
    print("MedGraphQA Safety Eval")
    print(f"duration_seconds: {round(float(report.get('duration_ms') or 0) / 1000, 3)}")
    print(f"cases: {metrics['case_count']}")
    print(f"input cases: {metrics['input_case_count']}")
    print(f"output cases: {metrics['output_case_count']}")
    print(f"input category_accuracy: {metrics['input_category_accuracy']}")
    print(f"action_accuracy: {metrics['action_accuracy']}")
    print(f"expected_hit_code_recall: {metrics['expected_hit_code_recall']}")
    print(f"forbidden_hit_violation_rate: {metrics['forbidden_hit_violation_rate']}")
    print(f"direct_response_recall: {metrics['direct_response_recall']}")
    print(f"continue_pass_rate: {metrics['continue_pass_rate']}")
    print(f"unsafe_output_recall: {metrics['unsafe_output_recall']}")
    print(f"safe_output_pass_rate: {metrics['safe_output_pass_rate']}")
    print(f"rewrite_success_rate: {metrics['rewrite_success_rate']}")
    print(f"failure_count: {metrics['failure_count']}")
    print(f"report: {report_path}")
    print(f"history: {history_path}")


if __name__ == "__main__":
    main()
