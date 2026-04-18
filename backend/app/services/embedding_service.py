import logging
from typing import Sequence

import requests

from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.embedding")


class EntityEmbeddingService:
    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str,
        api_key: str,
        timeout_seconds: int,
        enabled: bool,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

    def embed_one(self, text: str) -> list[float] | None:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else None

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not self.enabled or not clean_texts:
            return []
        with log_operation(
            logger,
            "embedding.embed_batch",
            provider=self.provider,
            model=self.model,
            batch_size=len(clean_texts),
        ) as result:
            if self.provider == "dashscope":
                vectors = self._embed_dashscope(clean_texts)
            elif self.provider == "ollama":
                vectors = self._embed_ollama(clean_texts)
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
            result["vector_count"] = len(vectors)
            result["dimension"] = len(vectors[0]) if vectors else 0
            return vectors

    def _embed_dashscope(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for DashScope embeddings")
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(map(float, item["embedding"])) for item in rows]

    def _embed_ollama(self, texts: Sequence[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.api_base}/api/embed",
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings") or []
        return [list(map(float, item)) for item in embeddings]
