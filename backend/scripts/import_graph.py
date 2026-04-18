import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import py2neo
from tqdm import tqdm


def parse_line(line: str) -> dict | None:
    raw = line.strip()
    if not raw:
        return None
    if raw.endswith(","):
        raw = raw[:-1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


def upsert_entity(graph: py2neo.Graph, label: str, name: str) -> None:
    cypher = f"MERGE (n:{label} {{名称:$name}})"
    graph.run(cypher, name=name)


def upsert_disease(graph: py2neo.Graph, payload: dict) -> None:
    cypher = """
    MERGE (d:疾病 {名称:$name})
    SET d.疾病简介=$desc,
        d.疾病病因=$cause,
        d.预防措施=$prevent,
        d.治疗周期=$cycle,
        d.治愈概率=$prob,
        d.疾病易感人群=$easy_get
    """
    graph.run(
        cypher,
        name=payload.get("name", ""),
        desc=payload.get("desc", ""),
        cause=payload.get("cause", ""),
        prevent=payload.get("prevent", ""),
        cycle=payload.get("cure_lasttime", ""),
        prob=payload.get("cured_prob", ""),
        easy_get=payload.get("easy_get", ""),
    )


def create_relation(
    graph: py2neo.Graph, source_label: str, source_name: str, relation: str, target_label: str, target_name: str
) -> None:
    cypher = (
        f"MATCH (a:{source_label} {{名称:$source_name}}), (b:{target_label} {{名称:$target_name}}) "
        f"MERGE (a)-[:`{relation}`]->(b)"
    )
    graph.run(cypher, source_name=source_name, target_name=target_name)


def ensure_entities(graph: py2neo.Graph, label: str, values: Iterable[str]) -> None:
    for value in values:
        if isinstance(value, str) and value.strip():
            upsert_entity(graph, label, value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Import medical_new_2.json into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4j")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--input", default=str(Path(__file__).resolve().parents[2] / "data" / "medical_new_2.json"))
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    graph = py2neo.Graph(args.uri, user=args.user, password=args.password, name=args.database)
    if args.clear:
        graph.run("MATCH (n) DETACH DELETE n")

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    for line in tqdm(lines, desc="Importing"):
        item = parse_line(line)
        if not item:
            continue

        disease_name = item.get("name", "").strip()
        if not disease_name:
            continue
        upsert_disease(graph, item)

        drugs = item.get("common_drug", []) + item.get("recommand_drug", [])
        foods_do = item.get("do_eat", []) + item.get("recommand_eat", [])
        foods_not = item.get("not_eat", [])
        checks = item.get("check", [])
        departments = item.get("cure_department", [])
        symptoms = item.get("symptom", [])
        cure_ways = item.get("cure_way", [])
        accompany = item.get("acompany", [])

        ensure_entities(graph, "药品", drugs)
        ensure_entities(graph, "食物", foods_do + foods_not)
        ensure_entities(graph, "检查项目", checks)
        ensure_entities(graph, "科目", departments)
        ensure_entities(graph, "疾病症状", symptoms)
        ensure_entities(graph, "治疗方法", cure_ways)
        ensure_entities(graph, "疾病", accompany)

        for drug in drugs:
            create_relation(graph, "疾病", disease_name, "疾病使用药品", "药品", drug)
        for food in foods_do:
            create_relation(graph, "疾病", disease_name, "疾病宜吃食物", "食物", food)
        for food in foods_not:
            create_relation(graph, "疾病", disease_name, "疾病忌吃食物", "食物", food)
        for check in checks:
            create_relation(graph, "疾病", disease_name, "疾病所需检查", "检查项目", check)
        if departments:
            create_relation(graph, "疾病", disease_name, "疾病所属科目", "科目", departments[-1])
        for symptom in symptoms:
            create_relation(graph, "疾病", disease_name, "疾病的症状", "疾病症状", symptom)
        for way in cure_ways:
            if isinstance(way, list):
                way = way[0] if way else ""
            if isinstance(way, str) and way.strip():
                create_relation(graph, "疾病", disease_name, "治疗的方法", "治疗方法", way)
        for dis in accompany:
            create_relation(graph, "疾病", disease_name, "疾病并发疾病", "疾病", dis)

        for item_detail in item.get("drug_detail", []):
            parts = item_detail.split(",")
            if len(parts) < 2:
                continue
            product = parts[0].strip()
            producer = parts[-1].strip()
            if not product or not producer:
                continue
            upsert_entity(graph, "药品", product)
            upsert_entity(graph, "药品商", producer)
            create_relation(graph, "药品商", producer, "生产", "药品", product)

    print("Done.")


if __name__ == "__main__":
    main()

