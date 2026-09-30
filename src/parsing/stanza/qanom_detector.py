"""Contextual nominalization detection with the QANom classifier.

Stanza supplies noun candidates; the lexicon only canonicalizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class QANomResult:
    word_index: int
    confidence: float
    verb_form: str = ""


class QANomDetector:
    """Score every Stanza noun with the QANom contextual classifier."""

    MODEL_NAME = "kleinay/nominalization-candidate-classifier"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        use_gpu: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.threshold = float(threshold)
        self.use_gpu = bool(use_gpu)
        self.local_files_only = bool(local_files_only)
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QANom nominalization detection requires torch and transformers."
            ) from exc

        self._device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.MODEL_NAME,
            local_files_only=self.local_files_only,
        ).to(self._device)
        self._model.eval()

    @staticmethod
    @lru_cache(maxsize=4096)
    def _canonicalize_after_detection(noun: str) -> str:
        """Derive a verb form after QANom has classified the noun as eventive."""
        low = noun.lower()
        try:
            import lemminflect
        except ImportError:
            forms = ()
        else:
            forms = tuple(lemminflect.getLemma(low, upos="VERB") or ())
        candidate = forms[0].lower() if forms else ""
        if candidate and candidate != low:
            return candidate

        # Canonicalization only; QANom already decided eventiveness.
        from src.models.candidate._nominal_canon import mapped_verbal_form

        mapped = mapped_verbal_form(low)
        return "" if mapped == low else mapped

    def detect(self, words: list[str], upos: list[str]) -> dict[int, QANomResult]:
        """Return positive QANom classifications keyed by Stanza word index."""
        return self.detect_many([(words, upos)])[0]

    def detect_many(
        self,
        items: list[tuple[list[str], list[str]]],
    ) -> list[dict[int, QANomResult]]:
        """Batch-score Stanza noun candidates with one QANom forward pass."""
        if not items:
            return []
        for words, upos in items:
            if len(words) != len(upos):
                raise ValueError("QANom words and UPOS sequences must have equal length.")

        noun_indices_by_item = [
            [i for i, pos in enumerate(upos) if pos == "NOUN"]
            for _words, upos in items
        ]
        if not any(noun_indices_by_item):
            return [{} for _ in items]

        self._load()
        import torch

        word_batches = [words for words, _upos in items]
        encoded = self._tokenizer(
            word_batches,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        word_ids_by_item = [
            encoded.word_ids(batch_index=i) for i in range(len(items))
        ]
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits

        positive = logits[:, :, 0].sigmoid()
        negative = logits[:, :, 1].sigmoid()
        probabilities = (positive + (1.0 - negative)) / 2.0

        output: list[dict[int, QANomResult]] = []
        for batch_i, ((words, _upos), noun_indices, word_ids) in enumerate(
            zip(items, noun_indices_by_item, word_ids_by_item)
        ):
            first_token_for_word: dict[int, int] = {}
            for token_i, word_i in enumerate(word_ids):
                if word_i is not None:
                    first_token_for_word.setdefault(word_i, token_i)

            results: dict[int, QANomResult] = {}
            for word_i in noun_indices:
                token_i = first_token_for_word.get(word_i)
                if token_i is None:
                    continue
                confidence = float(probabilities[batch_i, token_i].item())
                if confidence < self.threshold:
                    continue
                results[word_i] = QANomResult(
                    word_index=word_i,
                    confidence=confidence,
                    verb_form=self._canonicalize_after_detection(words[word_i]),
                )
            output.append(results)
        return output
