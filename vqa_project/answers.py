from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .vqa_normalization import normalize_vqa_answer, normalize_vqa_answer_group

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = _PUNCT_RE.sub(" ", answer)
    answer = _SPACE_RE.sub(" ", answer)
    return answer.strip()


@dataclass(frozen=True)
class AnswerVocab:
    idx_to_answer: list[str]
    normalization: str = "legacy"

    @property
    def answer_to_idx(self) -> dict[str, int]:
        return {answer: idx for idx, answer in enumerate(self.idx_to_answer)}

    def __len__(self) -> int:
        return len(self.idx_to_answer)

    def encode(self, answer: str) -> int | None:
        answer_to_idx = self.answer_to_idx
        if self.normalization == "vqa_v2" and answer in answer_to_idx:
            return answer_to_idx[answer]
        normalizer = normalize_vqa_answer if self.normalization == "vqa_v2" else normalize_answer
        normalized = normalizer(answer)
        return answer_to_idx.get(normalized)

    def decode(self, idx: int) -> str:
        return self.idx_to_answer[idx]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"idx_to_answer": self.idx_to_answer, "normalization": self.normalization},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "AnswerVocab":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            idx_to_answer=list(payload["idx_to_answer"]),
            normalization=str(payload.get("normalization", "legacy")),
        )


def _iter_annotation_answers(annotation: dict) -> list[str]:
    if annotation.get("answers"):
        return [item["answer"] for item in annotation["answers"] if item.get("answer")]
    if annotation.get("multiple_choice_answer"):
        return [annotation["multiple_choice_answer"]]
    return []


def build_answer_vocab(
    annotation_path: str | Path,
    top_k: int,
    normalization: str = "legacy",
) -> AnswerVocab:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    counter: Counter[str] = Counter()
    for annotation in payload["annotations"]:
        answers = _iter_annotation_answers(annotation)
        if normalization == "vqa_v2":
            normalized_answers = normalize_vqa_answer_group(answers)
        elif normalization == "legacy":
            normalized_answers = [normalize_answer(answer) for answer in answers]
        else:
            raise ValueError(f"Unsupported answer normalization: {normalization}")
        for normalized in normalized_answers:
            if normalized:
                counter[normalized] += 1

    most_common = [answer for answer, _ in counter.most_common(top_k)]
    if not most_common:
        raise ValueError(f"No answers found in {annotation_path}")
    return AnswerVocab(most_common, normalization=normalization)
