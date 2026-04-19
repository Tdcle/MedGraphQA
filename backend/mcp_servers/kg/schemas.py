from __future__ import annotations

from typing import Sequence


ATTR_FIELDS = (
    "疾病简介",
    "疾病病因",
    "预防措施",
    "治疗周期",
    "治愈概率",
    "疾病易感人群",
)

RELATION_TARGETS = {
    "疾病使用药品": "药品",
    "疾病宜吃食物": "食物",
    "疾病忌吃食物": "食物",
    "疾病所需检查": "检查项目",
    "疾病所属科目": "科目",
    "疾病的症状": "疾病症状",
    "治疗的方法": "治疗方法",
    "疾病并发疾病": "疾病",
}

ENTITY_LABELS = (
    "疾病",
    "疾病症状",
    "药品",
    "药品商",
    "食物",
    "检查项目",
    "治疗方法",
    "科目",
)

MAX_NAME_LENGTH = 80
MAX_SYMPTOMS = 20
MAX_LIMIT = 50


def ok(data: dict | list | str | int | bool | None = None, **extra) -> dict:
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return payload


def error(code: str, message: str, **extra) -> dict:
    payload = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    payload.update(extra)
    return payload


def clean_name(value: str, field_name: str = "name") -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空")
    if len(text) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} 过长，最多 {MAX_NAME_LENGTH} 个字符")
    return text


def clean_names(values: Sequence[str], field_name: str = "names") -> list[str]:
    if not values:
        raise ValueError(f"{field_name} 不能为空")
    result: list[str] = []
    for item in values:
        text = clean_name(str(item), field_name=field_name)
        if text not in result:
            result.append(text)
    if len(result) > MAX_SYMPTOMS:
        raise ValueError(f"{field_name} 最多允许 {MAX_SYMPTOMS} 项")
    return result


def clean_limit(value: int | None, default: int = 10) -> int:
    limit = int(value or default)
    if limit < 1:
        raise ValueError("limit 必须大于 0")
    return min(limit, MAX_LIMIT)


def clean_attribute(field_name: str) -> str:
    field = clean_name(field_name, "field_name")
    if field not in ATTR_FIELDS:
        raise ValueError(f"不支持的疾病属性：{field}")
    return field


def clean_relation(relation: str, target_label: str | None = None) -> tuple[str, str]:
    rel = clean_name(relation, "relation")
    expected_target = RELATION_TARGETS.get(rel)
    if not expected_target:
        raise ValueError(f"不支持的疾病关系：{rel}")
    target = clean_name(target_label or expected_target, "target_label")
    if target != expected_target:
        raise ValueError(f"关系 {rel} 的目标标签必须是 {expected_target}")
    return rel, target
