import logging
import json

import requests

from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.llm")


class DashScopeService:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        enable_thinking: bool | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking

    def _payload(self, prompt: str, *, json_mode: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        return payload

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for dashscope provider")
        with log_operation(
            logger,
            "llm.generate",
            provider="dashscope",
            model=self.model,
            enable_thinking=self.enable_thinking,
            prompt_len=len(prompt),
        ) as result:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = self._payload(prompt)
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            result["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                result["answer_len"] = 0
                return ""
            message = choices[0].get("message", {})
            answer = (message.get("content") or "").strip()
            result["answer_len"] = len(answer)
            return answer

    def generate_json(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for dashscope provider")
        with log_operation(
            logger,
            "llm.generate_json",
            provider="dashscope",
            model=self.model,
            enable_thinking=self.enable_thinking,
            prompt_len=len(prompt),
        ) as result:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = self._payload(prompt, json_mode=True)
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            result["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                result["answer_len"] = 0
                return ""
            message = choices[0].get("message", {})
            answer = (message.get("content") or "").strip()
            result["answer_len"] = len(answer)
            return answer


class OllamaService:
    def __init__(
        self,
        api_base: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        enable_thinking: bool | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking

    def _payload(self, prompt: str, *, json_mode: bool = False, stream: bool = False) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if self.enable_thinking is not None:
            payload["think"] = self.enable_thinking
        return payload

    def generate(self, prompt: str) -> str:
        with log_operation(
            logger,
            "llm.generate",
            provider="ollama",
            model=self.model,
            enable_thinking=self.enable_thinking,
            prompt_len=len(prompt),
        ) as result:
            payload = self._payload(prompt)
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            result["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            answer = (data.get("response") or "").strip()
            result["answer_len"] = len(answer)
            return answer

    def generate_json(self, prompt: str) -> str:
        with log_operation(
            logger,
            "llm.generate_json",
            provider="ollama",
            model=self.model,
            enable_thinking=self.enable_thinking,
            prompt_len=len(prompt),
        ) as result:
            payload = self._payload(prompt, json_mode=True)
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            result["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            answer = (data.get("response") or "").strip()
            result["answer_len"] = len(answer)
            return answer

    def generate_stream(self, prompt: str):
        with log_operation(
            logger,
            "llm.generate_stream",
            provider="ollama",
            model=self.model,
            enable_thinking=self.enable_thinking,
            prompt_len=len(prompt),
        ) as result:
            payload = self._payload(prompt, stream=True)
            token_count = 0
            char_count = 0
            with requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout_seconds,
            ) as response:
                result["http_status"] = response.status_code
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    data = json.loads(line)
                    token = data.get("response") or ""
                    if token:
                        token_count += 1
                        char_count += len(token)
                        result["token_count"] = token_count
                        result["answer_len"] = char_count
                        yield token
                    if data.get("done"):
                        break
