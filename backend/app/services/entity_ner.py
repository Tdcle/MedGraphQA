import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.entity_ner")


DEFAULT_LABELS = [
    "<PAD>",
    "B-疾病",
    "I-疾病",
    "O",
    "B-疾病症状",
    "I-疾病症状",
    "B-检查项目",
    "I-检查项目",
    "B-治疗方法",
    "I-治疗方法",
    "B-药品商",
    "I-药品商",
    "B-药品",
    "I-药品",
    "B-食物",
    "I-食物",
    "B-科目",
    "I-科目",
]


@dataclass
class EntityMention:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float


class NullEntityMentionExtractor:
    enabled = False

    def extract(self, text: str, expected_types: Sequence[str] | None = None) -> list[EntityMention]:
        return []

    def extract_terms(self, text: str, expected_types: Sequence[str] | None = None) -> list[str]:
        return []


class RobertaRnnEntityMentionExtractor:
    def __init__(
        self,
        enabled: bool,
        model_path: Path,
        pretrained_model: str,
        labels: Sequence[str] | None = None,
        max_length: int = 128,
        device: str = "cpu",
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        confidence_threshold: float = 0.4,
    ) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path)
        self.pretrained_model = pretrained_model
        self.labels = list(labels or DEFAULT_LABELS)
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels)}
        self.max_length = max_length
        self.device_name = device
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.confidence_threshold = confidence_threshold
        self._ready = False
        self._torch = None
        self._tokenizer = None
        self._model = None

    def extract_terms(self, text: str, expected_types: Sequence[str] | None = None) -> list[str]:
        terms: list[str] = []
        for mention in self.extract(text, expected_types):
            if mention.text not in terms:
                terms.append(mention.text)
        return terms

    def extract(
        self,
        text: str,
        expected_types: Sequence[str] | None = None,
    ) -> list[EntityMention]:
        if not self.enabled or not text.strip():
            return []
        with log_operation(
            logger,
            "entity_ner.extract",
            model="bert_rnn",
            text_len=len(text),
            expected_types=",".join(expected_types or []),
        ) as result:
            if not self._ensure_ready():
                result["mention_count"] = 0
                result["fallback"] = "model_unavailable"
                return []
            expected = set(expected_types or [])
            try:
                mentions = self._predict(text, expected)
                result["mention_count"] = len(mentions)
                result["mentions"] = "|".join(item.text for item in mentions[:8])
                return mentions
            except Exception:
                logger.exception("entity ner inference failed")
                result["mention_count"] = 0
                result["fallback"] = "inference_error"
                return []

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        try:
            import torch
            from transformers import BertTokenizer

            self._torch = torch
            self._tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
            self._model = _BertRnnTokenClassifier(
                pretrained_model=self.pretrained_model,
                num_labels=len(self.labels),
                rnn_hidden_size=self.rnn_hidden_size,
                rnn_num_layers=self.rnn_num_layers,
            )
            state = torch.load(self.model_path, map_location="cpu")
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            self._model.load_state_dict(state_dict, strict=True)
            device = torch.device(self.device_name if self.device_name != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            self._model.eval()
            self._device = device
            self._ready = True
            logger.info(
                "operation=entity_ner.load_model status=ok model=bert_rnn path=%s device=%s",
                self.model_path,
                device,
            )
            return True
        except Exception:
            logger.exception(
                "operation=entity_ner.load_model status=error model=bert_rnn path=%s",
                self.model_path,
            )
            self.enabled = False
            return False

    def _predict(self, text: str, expected_types: set[str]) -> list[EntityMention]:
        torch = self._torch
        text = text.strip()
        chars = list(text[: max(self.max_length - 2, 1)])
        if not chars:
            return []

        token_ids = self._tokenizer.encode(
            "".join(chars),
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(token_ids)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
            label_ids = probs.argmax(dim=-1).tolist()

        body_label_ids = label_ids[1 : len(chars) + 1]
        body_probs = probs[1 : len(chars) + 1]
        char_labels = [self.id_to_label.get(label_id, "O") for label_id in body_label_ids]
        char_scores = [
            float(body_probs[idx, label_id].item())
            for idx, label_id in enumerate(body_label_ids)
        ]

        mentions = self._decode_mentions(chars, char_labels, char_scores)
        if expected_types:
            mentions = [item for item in mentions if item.entity_type in expected_types]
        return [
            item
            for item in mentions
            if item.confidence >= self.confidence_threshold and item.text.strip()
        ]

    @staticmethod
    def _decode_mentions(
        chars: Sequence[str],
        labels: Sequence[str],
        scores: Sequence[float],
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        current_type = ""
        start = -1
        current_scores: list[float] = []

        def flush(end: int) -> None:
            nonlocal current_type, start, current_scores
            if start >= 0 and current_type:
                text = "".join(chars[start:end]).strip()
                if text:
                    confidence = sum(current_scores) / max(len(current_scores), 1)
                    mentions.append(EntityMention(text, current_type, start, end, confidence))
            current_type = ""
            start = -1
            current_scores = []

        for idx, label in enumerate(labels):
            if label in {"O", "<PAD>"} or "-" not in label:
                flush(idx)
                continue
            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or entity_type != current_type:
                flush(idx)
                current_type = entity_type
                start = idx
                current_scores = [scores[idx]]
            elif prefix == "I" and start >= 0:
                current_scores.append(scores[idx])
            else:
                flush(idx)
        flush(len(chars))
        return mentions


class _BertRnnTokenClassifier:
    def __new__(
        cls,
        pretrained_model: str,
        num_labels: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
    ):
        import torch
        from transformers import BertModel

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bert = BertModel.from_pretrained(pretrained_model)
                self.gru = torch.nn.RNN(
                    input_size=768,
                    hidden_size=rnn_hidden_size,
                    num_layers=rnn_num_layers,
                    batch_first=True,
                    bidirectional=True,
                )
                self.classifier = torch.nn.Linear(rnn_hidden_size * 2, num_labels)

            def forward(self, x):
                sequence_output, _ = self.bert(
                    x,
                    attention_mask=(x > 0),
                    return_dict=False,
                )
                gru_output, _ = self.gru(sequence_output)
                return self.classifier(gru_output)

        return Model()
