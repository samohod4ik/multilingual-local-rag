"""Benchmark annotations. These types are not the adapter product record."""

from __future__ import annotations

import re
from dataclasses import dataclass

from multilingual_local_rag.contracts import SourceDocument, _require

PUBLIC_V1_LANGUAGES = frozenset({"ru", "en"})
CATEGORIES = frozenset(
    {
        "same_language_lexical",
        "paraphrase",
        "cross_language",
        "identifier_rich",
        "hub_distractor",
        "conflicting_superseded",
        "exact_evidence",
    }
)
ROLES = frozenset({"primary", "hub", "distractor", "superseded"})
GRADES = frozenset({0, 1, 3})
_GROUP_RE = re.compile(r"^GRP-[0-9]{3}$")
_QUERY_RE = re.compile(r"^QRY-[0-9]{3}$")


@dataclass(frozen=True, slots=True)
class SourceGroup:
    group_id: str
    fold: str
    topic: str
    category: str

    def __post_init__(self) -> None:
        _require(bool(_GROUP_RE.match(self.group_id)), f"bad group_id {self.group_id!r}")
        _require(bool(self.fold.strip()), "fold must be non-empty")
        _require(bool(self.topic.strip()), "topic must be non-empty")
        _require(self.category in CATEGORIES, f"bad category {self.category!r}")


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    """Corpus row. Group, pair, role, and supersession stay on this type."""

    source_id: str
    group_id: str
    language: str
    title: str
    text: str
    source_uri: str
    content_hash: str
    revision: str
    media_type: str
    pair_id: str | None = None
    role: str = "primary"
    supersedes: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_GROUP_RE.match(self.group_id)), f"bad group_id {self.group_id!r}")
        _require(self.language in PUBLIC_V1_LANGUAGES, f"public-v1 language {self.language!r}")
        _require(bool(self.title.strip()), "title must be non-empty")
        _require(self.role in ROLES, f"bad role {self.role!r}")
        _require(self.supersedes != self.source_id, "document cannot supersede itself")
        if self.role == "superseded":
            _require(self.supersedes is None, "superseded row must not itself supersede")


@dataclass(frozen=True, slots=True)
class Qrel:
    source_id: str
    grade: int

    def __post_init__(self) -> None:
        _require(self.grade in GRADES, f"grade must be 0, 1, or 3, got {self.grade}")


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    source_id: str
    snippet: str

    def __post_init__(self) -> None:
        _require(bool(self.snippet), "evidence snippet must be non-empty")


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    group_id: str
    language: str
    category: str
    text: str
    qrels: tuple[Qrel, ...]
    evidence: tuple[EvidenceSpan, ...]

    def __post_init__(self) -> None:
        _require(bool(_QUERY_RE.match(self.query_id)), f"bad query_id {self.query_id!r}")
        _require(bool(_GROUP_RE.match(self.group_id)), f"bad group_id {self.group_id!r}")
        _require(self.language in PUBLIC_V1_LANGUAGES, f"query language {self.language!r}")
        _require(self.category in CATEGORIES, f"bad category {self.category!r}")
        _require(bool(self.text.strip()), "query text must be non-empty")
        ids = [qrel.source_id for qrel in self.qrels]
        _require(len(ids) == len(set(ids)), "duplicate qrel source_id")


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    groups: tuple[SourceGroup, ...]
    documents: tuple[BenchmarkDocument, ...]
    queries: tuple[QueryCase, ...]


def to_source_document(document: BenchmarkDocument) -> SourceDocument:
    """Project a benchmark row onto the product record. Annotations are dropped."""
    return SourceDocument(
        source_id=document.source_id,
        source_uri=document.source_uri,
        text=document.text,
        content_hash=document.content_hash,
        revision=document.revision,
        media_type=document.media_type,
        metadata={"title": document.title},
        language=document.language,
    )


def validate_dataset(dataset: BenchmarkDataset) -> None:
    """Check public-v1 gold, grades, pairs, and source-group split integrity."""
    groups = {group.group_id: group for group in dataset.groups}
    _require(len(groups) == len(dataset.groups), "duplicate group_id")
    _require(len(groups) == 40, f"expected 40 source groups, got {len(groups)}")
    docs = {doc.source_id: doc for doc in dataset.documents}
    _require(len(docs) == len(dataset.documents), "duplicate source_id")
    _require(len(dataset.queries) == 80, f"expected 80 queries, got {len(dataset.queries)}")
    seen_categories = {group.category for group in dataset.groups}
    _require(seen_categories == CATEGORIES, f"missing categories {CATEGORIES - seen_categories}")

    by_group_docs: dict[str, list[BenchmarkDocument]] = {gid: [] for gid in groups}
    for doc in dataset.documents:
        _require(doc.group_id in groups, f"{doc.source_id} has unknown group")
        _require(bool(doc.content_hash), f"{doc.source_id} missing hash")
        by_group_docs[doc.group_id].append(doc)
        if doc.supersedes is not None:
            _require(doc.supersedes in docs, f"{doc.source_id} supersedes unknown id")
            _require(docs[doc.supersedes].group_id == doc.group_id, "supersedes crosses groups")

    by_group_queries: dict[str, list[QueryCase]] = {gid: [] for gid in groups}
    query_ids: set[str] = set()
    for query in dataset.queries:
        _require(query.query_id not in query_ids, f"duplicate query {query.query_id}")
        query_ids.add(query.query_id)
        _require(query.group_id in groups, f"{query.query_id} has unknown group")
        _require(
            query.category == groups[query.group_id].category, "query category drifts from group"
        )
        by_group_queries[query.group_id].append(query)
        _check_query(query, docs)

    for group_id, queries in by_group_queries.items():
        langs = sorted(query.language for query in queries)
        _require(langs == ["en", "ru"], f"{group_id} must have exactly one en and one ru query")
        pair_ids = {doc.pair_id for doc in by_group_docs[group_id] if doc.role == "primary"}
        _require(
            len(pair_ids) == 1 and None not in pair_ids, f"{group_id} primary pair is not unique"
        )


def _check_query(query: QueryCase, docs: dict[str, BenchmarkDocument]) -> None:
    grades = {qrel.source_id: qrel.grade for qrel in query.qrels}
    _require(3 in grades.values(), f"{query.query_id} has no grade-3 document")
    _require(all(source_id in docs for source_id in grades), f"{query.query_id} qrel id missing")
    grade3 = [docs[source_id] for source_id, grade in grades.items() if grade == 3]
    _require(len(grade3) == 1, f"{query.query_id} must have exactly one grade-3 document")
    gold = grade3[0]
    if query.category == "cross_language":
        _require(
            gold.language != query.language,
            f"{query.query_id} cross-language gold matches query language",
        )
    else:
        _require(gold.language == query.language, f"{query.query_id} same-language gold differs")
    for source_id, grade in grades.items():
        doc = docs[source_id]
        if grade == 1:
            _require(
                doc.language != gold.language, f"{query.query_id} grade 1 is not the other language"
            )
            _require(
                doc.pair_id == gold.pair_id, f"{query.query_id} grade 1 is not the translation pair"
            )
        if grade == 3:
            _require(doc.role == "primary", f"{query.query_id} grade 3 is not primary")
    _require(bool(query.evidence), f"{query.query_id} missing evidence")
    for span in query.evidence:
        _require(
            grades.get(span.source_id) == 3,
            f"{query.query_id} evidence is not on the grade-3 document",
        )
        _require(
            span.snippet in docs[span.source_id].text,
            f"{query.query_id} evidence is not an exact substring",
        )
