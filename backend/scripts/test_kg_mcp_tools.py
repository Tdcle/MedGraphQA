from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp_servers.kg.kg_tools import KGMcpTools  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test MedGraphQA KG MCP tools.")
    parser.add_argument("--disease", default="感冒", help="Disease name to inspect.")
    parser.add_argument("--symptom", default="腹痛", help="Symptom name to inspect.")
    parser.add_argument(
        "--symptoms",
        nargs="*",
        default=["咽痛", "鼻塞", "流鼻涕"],
        help="Symptoms for KG candidate lookup.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    tools = KGMcpTools.from_settings()
    payload = {
        "ping": tools.ping(),
        "schema": tools.schema(),
        "disease_profile": tools.get_disease_profile(args.disease),
        "diseases_by_symptom": tools.get_diseases_by_symptom(args.symptom, limit=10),
        "candidates": tools.get_disease_candidates_by_symptoms(args.symptoms, limit=5),
        "inspect_entity": tools.inspect_entity(args.symptom, sample_limit=5),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
