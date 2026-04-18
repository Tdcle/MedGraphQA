from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from evals.generators.case_builder import EvalCaseBuilder  # noqa: E402
from evals.generators.kg_client import EvalDataSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MedGraphQA eval datasets from Neo4j and Postgres."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_ROOT / "evals" / "datasets",
        help="Output directory for JSONL datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of cases to generate for each core dataset.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also generate fine-grained auxiliary datasets.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep old generated JSONL files in the output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for template choices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for old_file in args.output.glob("*.jsonl"):
            old_file.unlink()

    data_source = EvalDataSource(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        neo4j_database=settings.neo4j_database,
        postgres_dsn=settings.postgres_dsn,
    )
    profiles = data_source.fetch_disease_profiles(limit=max(args.limit * 4, 200))
    names = [item.name for item in profiles]
    for profile in profiles:
        names.extend(profile.symptoms)
    aliases = data_source.fetch_aliases(names)
    alias_samples = data_source.fetch_alias_samples(limit=max(args.limit, 80))

    builder = EvalCaseBuilder(
        profiles=profiles,
        aliases=aliases,
        alias_samples=alias_samples,
        seed=args.seed,
    )
    datasets = builder.build_all(limit=args.limit) if args.full else builder.build_core(limit=args.limit)
    for dataset_name, cases in datasets.items():
        write_jsonl(args.output / f"{dataset_name}.jsonl", cases)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_database": settings.neo4j_database,
            "postgres": "configured",
        },
        "profile_count": len(profiles),
        "alias_sample_count": len(alias_samples),
        "datasets": {name: len(cases) for name, cases in datasets.items()},
        "notes": [
            "All labels are derived from the current KG and entity alias tables.",
            "Core datasets focus on the highest-value checks: symptom extraction, negation filtering, disease top-k recall, and follow-up turn limit.",
            "Disease labels should be evaluated as top-k recall, not guaranteed single-label diagnosis.",
            "Use --full only when you need fine-grained auxiliary datasets.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
