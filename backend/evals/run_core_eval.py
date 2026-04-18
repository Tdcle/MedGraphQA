from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.container import build_container  # noqa: E402
from app.services import chat_memory  # noqa: E402
from app.services.clinical_context import ClinicalContext  # noqa: E402
from app.services.entity_search import EntityCandidate  # noqa: E402
from app.services.medical_qa_graph import MedicalQAGraph  # noqa: E402


DEFAULT_INTENTS = ["disease_cure_way"]


@dataclass
class CasePrediction:
    positive_symptoms: list[str]
    negated_symptoms: list[str]
    diseases: list[str]
    decision: str | None
    follow_up_turns: int = 0


@dataclass
class MetricBucket:
    case_count: int = 0
    expected_positive: int = 0
    hit_positive: int = 0
    expected_negated: int = 0
    hit_negated: int = 0
    negated_false_positive: int = 0
    disease_cases: int = 0
    disease_hits: int = 0
    follow_up_over_limit: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "positive_symptom_recall": ratio(self.hit_positive, self.expected_positive),
            "clinical_negation_recall": ratio(self.hit_negated, self.expected_negated),
            "negated_symptom_false_positive_rate": ratio(
                self.negated_false_positive,
                self.expected_negated,
            ),
            "disease_top5_recall": ratio(self.disease_hits, self.disease_cases),
            "follow_up_over_limit_count": self.follow_up_over_limit,
            "failures": self.failures[:50],
        }


class CoreEvaluator:
    def __init__(
        self,
        skip_clinical_llm: bool = False,
        clinical_timeout_seconds: int = 15,
    ) -> None:
        self.container = build_container(settings)
        graph = self.container.chat_service.graph
        self.entity_normalizer = graph.entity_normalizer
        self.clinical_context_service = graph.clinical_context_service
        self._override_clinical_timeout(clinical_timeout_seconds)
        self.knowledge = graph.knowledge
        self.skip_clinical_llm = skip_clinical_llm
        self.clinical_timeout_seconds = clinical_timeout_seconds
        self.max_follow_up_turns = graph.disease_max_follow_up_turns
        self.possible_confidence_threshold = graph.disease_possible_confidence_threshold
        self.possible_candidate_limit = graph.disease_possible_candidate_limit

    def reset_clinical_stats(self) -> None:
        reset = getattr(self.clinical_context_service, "reset_stats", None)
        if callable(reset):
            reset()

    def clinical_stats(self) -> dict[str, Any]:
        snapshot = getattr(self.clinical_context_service, "stats_snapshot", None)
        stats = snapshot() if callable(snapshot) else {}
        total = int(stats.get("extract_total") or 0)
        llm_calls = int(stats.get("llm_call_count") or 0)
        rules_only = int(stats.get("rules_only_count") or 0)
        stats["llm_call_rate"] = ratio(llm_calls, total)
        stats["rules_only_rate"] = ratio(rules_only, total)
        return stats

    def _override_clinical_timeout(self, timeout_seconds: int) -> None:
        if timeout_seconds <= 0:
            return
        llm_service = getattr(self.clinical_context_service, "llm_service", None)
        if llm_service is not None and hasattr(llm_service, "timeout_seconds"):
            llm_service.timeout_seconds = timeout_seconds

    def predict_single_turn(self, query: str) -> CasePrediction:
        state = self._process_turn(
            query=query,
            previous_context={},
            previous_entities=[],
            follow_up_turns=0,
        )
        return state["prediction"]

    def predict_multi_turn(self, turns: list[str]) -> CasePrediction:
        context: dict[str, Any] = {}
        entities: list[EntityCandidate] = []
        follow_up_turns = 0
        prediction = CasePrediction([], [], [], None)
        for query in turns:
            state = self._process_turn(
                query=query,
                previous_context=context,
                previous_entities=entities,
                follow_up_turns=follow_up_turns,
            )
            context = state["context"]
            entities = state["entities"]
            prediction = state["prediction"]
            if prediction.decision == "ask_follow_up":
                follow_up_turns += 1
            else:
                follow_up_turns = 0
        prediction.follow_up_turns = follow_up_turns
        return prediction

    def close(self) -> None:
        for service in [
            getattr(self.container, "auth_repository", None),
            getattr(self.container, "memory_repository", None),
            getattr(self.container, "entity_repository", None),
        ]:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    def _process_turn(
        self,
        query: str,
        previous_context: dict[str, Any],
        previous_entities: list[EntityCandidate],
        follow_up_turns: int,
    ) -> dict[str, Any]:
        expected_types = self.entity_normalizer.expected_types(DEFAULT_INTENTS)
        ner_terms = self.entity_normalizer.extract_mention_terms(
            query=query,
            expected_types=expected_types,
        )
        if self.skip_clinical_llm:
            clinical_context = ClinicalContext()
            context = previous_context or {}
        else:
            clinical_context = self.clinical_context_service.extract(
                query=query,
                previous_context=previous_context,
                entities=[],
                entity_hints=ner_terms,
            )
            context = clinical_context.to_dict()

        terms = MedicalQAGraph._filter_negated_terms(
            list(ner_terms),
            clinical_context.negated_symptoms,
        )
        for term in self.clinical_context_service.symptom_terms(clinical_context):
            if (
                term
                and term not in terms
                and not MedicalQAGraph._is_negated_text(term, clinical_context.negated_symptoms)
            ):
                terms.append(term)

        if terms:
            current_entities = self.entity_normalizer.resolve_terms(
                terms=terms,
                query=query,
                intents=DEFAULT_INTENTS,
                allow_vector=True,
            )
        else:
            current_entities = self.entity_normalizer.resolve(
                query=query,
                intents=DEFAULT_INTENTS,
            )
        current_entities = MedicalQAGraph._filter_negated_entities(
            current_entities,
            clinical_context.negated_symptoms,
        )
        previous_entities = MedicalQAGraph._filter_negated_entities(
            previous_entities,
            clinical_context.negated_symptoms,
        )
        entities = MedicalQAGraph._filter_negated_entities(
            chat_memory.merge_entities(previous_entities, current_entities),
            clinical_context.negated_symptoms,
        )

        result = self.knowledge.gather(
            effective_query=query,
            intents=DEFAULT_INTENTS,
            entities=entities,
            follow_up_turns=follow_up_turns,
            max_follow_up_turns=self.max_follow_up_turns,
            possible_confidence_threshold=self.possible_confidence_threshold,
            possible_candidate_limit=self.possible_candidate_limit,
        )
        prediction = CasePrediction(
            positive_symptoms=[
                item.canonical_name
                for item in result.entities
                if item.entity_type == "疾病症状"
            ],
            negated_symptoms=list(clinical_context.negated_symptoms),
            diseases=self._disease_names(result),
            decision=result.disease_resolution.decision
            if result.disease_resolution
            else None,
        )
        return {
            "context": context,
            "entities": result.entities,
            "prediction": prediction,
        }

    @staticmethod
    def _disease_names(result) -> list[str]:
        names: list[str] = []
        resolution = result.disease_resolution
        if resolution:
            if resolution.disease_name:
                names.append(resolution.disease_name)
            for item in resolution.candidates[:5]:
                if item.disease not in names:
                    names.append(item.disease)
        for item in result.entities:
            if item.entity_type == "疾病" and item.canonical_name not in names:
                names.append(item.canonical_name)
        return names[:5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MedGraphQA core eval.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=BACKEND_ROOT / "evals" / "datasets",
        help="Directory containing core_single_turn.jsonl and core_multi_turn.jsonl.",
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
        help="Limit cases per dataset for debugging. 0 means all cases.",
    )
    parser.add_argument(
        "--skip-clinical-llm",
        action="store_true",
        help="Skip clinical-context LLM extraction for a fast smoke test.",
    )
    parser.add_argument(
        "--clinical-timeout-seconds",
        type=int,
        default=15,
        help="Timeout for each clinical-context LLM call during eval.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show service error stack traces while running eval.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Append one JSONL summary row to this file after each successful eval.",
    )
    return parser.parse_args()


def configure_eval_logging(show_errors: bool) -> None:
    level = logging.ERROR if show_errors else logging.CRITICAL
    for name in [
        "medgraphqa.llm",
        "medgraphqa.clinical_context",
        "medgraphqa.entity_ner",
        "medgraphqa.embedding",
        "medgraphqa.entity_search",
        "medgraphqa.kg",
        "medgraphqa.knowledge",
        "medgraphqa.chat",
        "elastic_transport.transport",
    ]:
        logging.getLogger(name).setLevel(level)


def main() -> None:
    args = parse_args()
    configure_eval_logging(args.show_errors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    started = time.perf_counter()
    evaluator = CoreEvaluator(
        skip_clinical_llm=args.skip_clinical_llm,
        clinical_timeout_seconds=args.clinical_timeout_seconds,
    )
    try:
        single_cases = load_jsonl(args.dataset_dir / "core_single_turn.jsonl", args.max_cases)
        multi_cases = load_jsonl(args.dataset_dir / "core_multi_turn.jsonl", args.max_cases)

        evaluator.reset_clinical_stats()
        single_metrics = eval_single_turn(
            evaluator,
            single_cases,
            show_progress=not args.no_progress,
        )
        single_clinical_stats = evaluator.clinical_stats()
        evaluator.reset_clinical_stats()
        multi_metrics = eval_multi_turn(
            evaluator,
            multi_cases,
            show_progress=not args.no_progress,
        )
        multi_clinical_stats = evaluator.clinical_stats()

        report = {
            "run_at": started_at.isoformat(timespec="seconds"),
            "config": {
                "dataset_dir": str(args.dataset_dir),
                "max_cases": args.max_cases or None,
                "skip_clinical_llm": args.skip_clinical_llm,
                "clinical_timeout_seconds": evaluator.clinical_timeout_seconds,
                "clinical_context_provider": settings.clinical_context_provider,
                "clinical_context_model": settings.clinical_context_model,
                "disease_max_follow_up_turns": evaluator.max_follow_up_turns,
                "disease_possible_confidence_threshold": evaluator.possible_confidence_threshold,
                "disease_possible_candidate_limit": evaluator.possible_candidate_limit,
            },
            "single_turn": single_metrics.to_report(),
            "multi_turn": multi_metrics.to_report(),
            "clinical_context": {
                "single_turn": single_clinical_stats,
                "multi_turn": multi_clinical_stats,
                "total": combine_clinical_stats(single_clinical_stats, multi_clinical_stats),
            },
        }
        finished_at = datetime.now()
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        report["finished_at"] = finished_at.isoformat(timespec="seconds")
        report["duration_ms"] = duration_ms
        report_path = args.output_dir / f"core_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        write_json(report_path, report)
        history_path = args.history_file or (args.output_dir / "core_eval_history.jsonl")
        append_history(history_path, report, report_path)
        print_summary(report, report_path)
        print(f"history: {history_path}")
    finally:
        evaluator.close()


def eval_single_turn(
    evaluator: CoreEvaluator,
    cases: list[dict[str, Any]],
    show_progress: bool = True,
) -> MetricBucket:
    metrics = MetricBucket()
    for case in iter_progress(cases, "single_turn", show_progress):
        metrics.case_count += 1
        target = case.get("target") or {}
        prediction = evaluator.predict_single_turn(str(case.get("query") or ""))
        update_common_metrics(metrics, case, target, prediction)
    return metrics


def eval_multi_turn(
    evaluator: CoreEvaluator,
    cases: list[dict[str, Any]],
    show_progress: bool = True,
) -> MetricBucket:
    metrics = MetricBucket()
    for case in iter_progress(cases, "multi_turn", show_progress):
        metrics.case_count += 1
        target = case.get("target") or {}
        prediction = evaluator.predict_multi_turn([str(item) for item in case.get("turns") or []])
        update_common_metrics(metrics, case, target, prediction)
        max_turns = int(target.get("max_follow_up_turns") or evaluator.max_follow_up_turns)
        if prediction.follow_up_turns > max_turns:
            metrics.follow_up_over_limit += 1
            append_failure(
                metrics,
                case,
                prediction,
                reason="follow_up_over_limit",
            )
    return metrics


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


def update_common_metrics(
    metrics: MetricBucket,
    case: dict[str, Any],
    target: dict[str, Any],
    prediction: CasePrediction,
) -> None:
    expected_positive = target.get("expected_positive_symptoms") or target.get("expected_symptoms") or []
    expected_negated = target.get("expected_negated_symptoms") or []
    expected_diseases = target.get("expected_possible_diseases") or target.get("expected_disease_in_top_k") or []

    positive_hits = count_hits(expected_positive, prediction.positive_symptoms)
    negated_hits = count_hits(expected_negated, prediction.negated_symptoms)
    negated_false_positive = count_hits(expected_negated, prediction.positive_symptoms)
    disease_hit = any_term_hit(expected_diseases, prediction.diseases)

    metrics.expected_positive += len(expected_positive)
    metrics.hit_positive += positive_hits
    metrics.expected_negated += len(expected_negated)
    metrics.hit_negated += negated_hits
    metrics.negated_false_positive += negated_false_positive
    if expected_diseases:
        metrics.disease_cases += 1
        metrics.disease_hits += 1 if disease_hit else 0

    if positive_hits < len(expected_positive):
        append_failure(metrics, case, prediction, reason="positive_symptom_miss")
    if negated_false_positive:
        append_failure(metrics, case, prediction, reason="negated_symptom_as_positive")
    if expected_diseases and not disease_hit:
        append_failure(metrics, case, prediction, reason="disease_top5_miss")


def count_hits(expected: list[str], predicted: list[str]) -> int:
    return sum(1 for item in expected if any_term_hit([item], predicted))


def any_term_hit(expected: list[str], predicted: list[str]) -> bool:
    return any(
        same_term(left, right)
        for left in expected
        for right in predicted
    )


def same_term(left: str, right: str) -> bool:
    return MedicalQAGraph._same_clinical_term(left, right)


def append_failure(
    metrics: MetricBucket,
    case: dict[str, Any],
    prediction: CasePrediction,
    reason: str,
) -> None:
    if len(metrics.failures) >= 50:
        return
    metrics.failures.append(
        {
            "id": case.get("id"),
            "reason": reason,
            "query": case.get("query"),
            "turns": case.get("turns"),
            "target": case.get("target"),
            "prediction": {
                "positive_symptoms": prediction.positive_symptoms,
                "negated_symptoms": prediction.negated_symptoms,
                "diseases": prediction.diseases,
                "decision": prediction.decision,
                "follow_up_turns": prediction.follow_up_turns,
            },
        }
    )


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def load_jsonl(path: Path, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_cases and len(rows) >= max_cases:
                break
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_history(path: Path, report: dict[str, Any], report_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_at": report.get("run_at"),
        "finished_at": report.get("finished_at"),
        "duration_ms": report.get("duration_ms"),
        "duration_seconds": round(float(report.get("duration_ms") or 0) / 1000, 3),
        "report_path": str(report_path),
        "config": report.get("config", {}),
        "single_turn": history_metrics(report.get("single_turn", {})),
        "multi_turn": history_metrics(report.get("multi_turn", {})),
        "clinical_context": report.get("clinical_context", {}),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def history_metrics(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_count": section.get("case_count"),
        "positive_symptom_recall": section.get("positive_symptom_recall"),
        "clinical_negation_recall": section.get("clinical_negation_recall"),
        "negated_symptom_false_positive_rate": section.get("negated_symptom_false_positive_rate"),
        "disease_top5_recall": section.get("disease_top5_recall"),
        "follow_up_over_limit_count": section.get("follow_up_over_limit_count"),
        "failure_count": len(section.get("failures") or []),
    }


def combine_clinical_stats(*sections: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "extract_total",
        "rules_only_count",
        "llm_call_count",
        "llm_success_count",
        "llm_fallback_count",
        "json_parse_error_count",
        "extract_error_count",
        "disabled_count",
    ]
    total = {key: sum(int(section.get(key) or 0) for section in sections) for key in keys}
    total["llm_call_rate"] = ratio(total["llm_call_count"], total["extract_total"])
    total["rules_only_rate"] = ratio(total["rules_only_count"], total["extract_total"])
    return total


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    single = report["single_turn"]
    multi = report["multi_turn"]
    print("MedGraphQA Core Eval")
    print(f"duration_seconds: {round(float(report.get('duration_ms') or 0) / 1000, 3)}")
    print(f"single_turn cases: {single['case_count']}")
    print(f"multi_turn cases:  {multi['case_count']}")
    print(f"single disease_top5_recall: {single['disease_top5_recall']}")
    print(f"multi disease_top5_recall:  {multi['disease_top5_recall']}")
    print(f"single positive_symptom_recall: {single['positive_symptom_recall']}")
    print(f"multi positive_symptom_recall:  {multi['positive_symptom_recall']}")
    print(f"single negated_false_positive_rate: {single['negated_symptom_false_positive_rate']}")
    print(f"multi negated_false_positive_rate:  {multi['negated_symptom_false_positive_rate']}")
    print(f"multi follow_up_over_limit_count: {multi['follow_up_over_limit_count']}")
    clinical = report.get("clinical_context", {}).get("total", {})
    if clinical:
        print(
            "clinical_context: "
            f"extract_total={clinical.get('extract_total')} "
            f"llm_call_count={clinical.get('llm_call_count')} "
            f"rules_only_count={clinical.get('rules_only_count')} "
            f"llm_call_rate={clinical.get('llm_call_rate')}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
