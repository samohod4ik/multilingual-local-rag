from collections.abc import Iterator

from multilingual_local_rag.adapters.base import AdapterInfo, SourceAdapter
from multilingual_local_rag.contracts import SourceDocument, sha256_text


class _MemoryAdapter:
    def __init__(self, documents: tuple[SourceDocument, ...]) -> None:
        self._documents = documents

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo("memory", "0", "memory")

    def iter_documents(self) -> Iterator[SourceDocument]:
        yield from self._documents

    def content_hash(self) -> str:
        return sha256_text("".join(doc.content_hash for doc in self._documents))


def test_adapter_protocol_yields_product_records() -> None:
    text = "hello"
    document = SourceDocument(
        source_id="DOC-1",
        source_uri="notes/DOC-1.txt",
        text=text,
        content_hash=sha256_text(text),
        revision="1",
        media_type="text/plain",
        metadata={},
        language=None,
    )
    adapter: SourceAdapter = _MemoryAdapter((document,))
    assert list(adapter.iter_documents()) == [document]
