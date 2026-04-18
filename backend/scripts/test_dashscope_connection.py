import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config() -> dict:
    load_env_file(BACKEND_ROOT / ".env")
    load_env_file(PROJECT_ROOT / ".env")
    config_path = Path(os.getenv("APP_CONFIG_FILE", BACKEND_ROOT / "config.json"))
    if not config_path.is_absolute():
        config_path = BACKEND_ROOT / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "***" + value[-4:]


def build_payload(
    model: str,
    prompt: str,
    json_mode: bool,
    enable_thinking: bool | None,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    return payload


def parse_bool_option(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"auto", "none", "default"}:
        return None
    return normalized in {"1", "true", "yes", "on"}


def clinical_context_prompt(query: str) -> str:
    return (
        "你是医疗对话结构化抽取器。只抽取用户明确说出的事实，不诊断，不补全。\n"
        "只输出一个 JSON 对象，不要 Markdown，不要解释。\n"
        "字段：symptoms, negated_symptoms, red_flags, known_diseases, medications, allergies, pregnancy, user_goal, missing_info。\n"
        "symptoms 元素字段：name, body_part, severity, duration, progression, quality, frequency。\n"
        "示例：上腹部轻微疼痛一天 -> "
        '{"symptoms":[{"name":"疼痛","body_part":"上腹部","severity":"轻微","duration":"一天","progression":null,"quality":"疼痛","frequency":null}],'
        '"negated_symptoms":[],"red_flags":[],"known_diseases":[],"medications":[],"allergies":[],"pregnancy":null,"user_goal":null,"missing_info":[]}\n'
        "上一轮已知上下文：{}\n"
        "实体提示：无\n"
        f"用户本轮输入：{query}"
    )


def main() -> int:
    config = load_config()
    clinical_config = config.get("clinical_context", {})
    llm_config = config.get("llm", {})

    parser = argparse.ArgumentParser(description="Test DashScope compatible chat API.")
    parser.add_argument("--api-base", default=clinical_config.get("api_base") or llm_config.get("api_base") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default=clinical_config.get("model") or "qwen3.5-flash")
    parser.add_argument("--api-key-env", default="CLINICAL_CONTEXT_API_KEY")
    parser.add_argument("--timeout", type=int, default=int(clinical_config.get("timeout_seconds", 30)))
    parser.add_argument("--json-mode", action="store_true", help="Test response_format={type: json_object}.")
    parser.add_argument(
        "--enable-thinking",
        default=str(clinical_config.get("enable_thinking", "auto")).lower(),
        help="true/false/auto. For clinical extraction, use false.",
    )
    parser.add_argument("--prompt", default='请只回答一句话：“连接成功”。')
    parser.add_argument("--clinical-context", action="store_true", help="Use a prompt close to backend clinical context extraction.")
    parser.add_argument("--query", default="今天早上起来肚子疼")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env) or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("ERROR: missing API key. Set CLINICAL_CONTEXT_API_KEY or DASHSCOPE_API_KEY in backend/.env", file=sys.stderr)
        return 2

    api_base = args.api_base.rstrip("/")
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    prompt = clinical_context_prompt(args.query) if args.clinical_context else args.prompt
    enable_thinking = parse_bool_option(args.enable_thinking)
    payload = build_payload(
        args.model,
        prompt,
        args.json_mode or args.clinical_context,
        enable_thinking,
    )

    print("DashScope connection test")
    print(f"api_base: {api_base}")
    print(f"model: {args.model}")
    print(f"api_key_env: {args.api_key_env}")
    print(f"api_key: {mask_key(api_key)}")
    print(f"timeout: {args.timeout}s")
    print(f"json_mode: {args.json_mode or args.clinical_context}")
    print(f"enable_thinking: {enable_thinking}")
    print(f"prompt_len: {len(prompt)}")

    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"status_code: {response.status_code}")
        print(f"elapsed_ms: {elapsed_ms:.2f}")
        if not response.ok:
            print("response_text:")
            print(response.text[:4000])
            return 1

        data = response.json()
        choices = data.get("choices") or []
        content = ""
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        print("response_content:")
        print(content)
        return 0
    except requests.exceptions.Timeout as exc:
        print(f"ERROR: request timeout: {exc}", file=sys.stderr)
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: request failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: response is not json: {exc}", file=sys.stderr)
        print(response.text[:4000])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
