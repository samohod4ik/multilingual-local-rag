"""Unicode BM25. Parameters match the usual k1=1.2, b=0.75 defaults."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(text) if len(token) > 1)


class BM25Index:
    def __init__(
        self, documents: Sequence[tuple[str, str]], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.ids = [doc_id for doc_id, _text in documents]
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, (_doc_id, text) in enumerate(documents):
            counts = Counter(tokenize(text))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term].append((index, frequency))
        total = sum(self._lengths)
        self._avg = total / max(1, len(self.ids))
        self._df = {term: len(rows) for term, rows in self._postings.items()}

    def search(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        scores = [0.0] * len(self.ids)
        for term in set(tokenize(query)):
            rows = self._postings.get(term)
            if not rows:
                continue
            idf = math.log(1 + (len(self.ids) - self._df[term] + 0.5) / (self._df[term] + 0.5))
            for index, frequency in rows:
                length = self._lengths[index] or 1
                denom = frequency + self.k1 * (1 - self.b + self.b * length / self._avg)
                scores[index] += idf * (frequency * (self.k1 + 1)) / denom
        ranked = [(self.ids[index], score) for index, score in enumerate(scores) if score > 0]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if limit is None:
            return ranked
        return ranked[:limit]
