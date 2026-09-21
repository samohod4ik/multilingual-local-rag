"""Optional answer beside the evidence packet."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from multilingual_local_rag.generation.citations import valid_citations
from multilingual_local_rag.generation.prompt import build_prompt
from multilingual_local_rag.retrieval.hybrid import EvidenceItem, SearchResult

Generator = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class AnswerResult:
    evidence: tuple[EvidenceItem, ...]
    answer: str | None
    citations: tuple[str, ...]
    abstained: bool
    degraded: bool
    degradation: str | None


def answer_question(
    result: SearchResult, question: str, generate: Generator | None
) -> AnswerResult:
    if not result.evidence or generate is None:
        return AnswerResult(
            evidence=result.evidence,
            answer=None,
            citations=(),
            abstained=True,
            degraded=result.degraded,
            degradation=result.degradation,
        )
    text = generate(build_prompt(question, result.evidence))
    citations = valid_citations(text, len(result.evidence))
    if not citations:
        return AnswerResult(
            evidence=result.evidence,
            answer=None,
            citations=(),
            abstained=True,
            degraded=result.degraded,
            degradation=result.degradation,
        )
    return AnswerResult(
        evidence=result.evidence,
        answer=text,
        citations=citations,
        abstained=False,
        degraded=result.degraded,
        degradation=result.degradation,
    )
