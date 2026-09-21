from dataclasses import fields

from multilingual_local_rag.benchmark import (
    BenchmarkDataset,
    BenchmarkDocument,
    to_source_document,
    validate_document_integrity,
)
from multilingual_local_rag.contracts import SourceDocument, sha256_text


def _product(**overrides: object) -> SourceDocument:
    text = "A plain note."
    payload: dict[str, object] = {
        "source_id": "DOC-1",
        "source_uri": "notes/DOC-1.txt",
        "text": text,
        "content_hash": sha256_text(text),
        "revision": "1",
        "media_type": "text/plain",
        "metadata": {"title": "Note"},
        "language": "fr",
    }
    payload.update(overrides)
    return SourceDocument(**payload)  # type: ignore[arg-type]


def test_product_fields_exclude_benchmark_annotations() -> None:
    names = {item.name for item in fields(SourceDocument)}
    assert names == {
        "source_id",
        "source_uri",
        "text",
        "content_hash",
        "revision",
        "media_type",
        "metadata",
        "language",
    }
    document = _product()
    assert document.language == "fr"
    assert not hasattr(document, "group_id")
    assert not hasattr(document, "pair_id")
    assert not hasattr(document, "role")
    assert not hasattr(document, "supersedes")


def test_open_language_tag_accepts_more_than_ru_en() -> None:
    assert _product(language="zh-Hans").language == "zh-Hans"
    assert _product(language=None).language is None


def test_source_uri_rejects_escape() -> None:
    for uri in ("../outside.txt", "/abs.txt", "C:/abs.txt", "notes/../x.txt"):
        try:
            _product(source_uri=uri)
        except ValueError:
            continue
        raise AssertionError(uri)


def test_mutated_content_hash_fails_document_integrity() -> None:
    text = "Bench row."
    digest = sha256_text(text)
    mutated = ("0" if digest[0] != "0" else "1") + digest[1:]
    row = BenchmarkDocument(
        source_id="DOC-E-001",
        group_id="GRP-001",
        language="en",
        title="Amberlathe",
        text=text,
        source_uri="groups/GRP-001/DOC-E-001.txt",
        content_hash=mutated,
        revision="1",
        media_type="text/plain",
        pair_id="PAIR-001",
        role="primary",
    )
    dataset = BenchmarkDataset(groups=(), documents=(row,), queries=())
    try:
        validate_document_integrity(dataset.documents)
    except ValueError:
        return
    raise AssertionError("mutated hash")


def test_projection_drops_annotations() -> None:
    text = "Bench row."
    row = BenchmarkDocument(
        source_id="DOC-E-001",
        group_id="GRP-001",
        language="en",
        title="Amberlathe",
        text=text,
        source_uri="groups/GRP-001/DOC-E-001.txt",
        content_hash=sha256_text(text),
        revision="1",
        media_type="text/plain",
        pair_id="PAIR-001",
        role="primary",
        supersedes="DOC-S-001",
    )
    projected = to_source_document(row)
    assert projected.metadata["title"] == "Amberlathe"
    assert not hasattr(projected, "group_id")
    assert not hasattr(projected, "supersedes")
