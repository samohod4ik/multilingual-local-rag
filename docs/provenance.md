# Provenance

Phase 1 code in this repository was written for this project. It was not copied from another repository.

## Inventory of local-vault-agent-rag `d450f987`

Commit `d450f9874301b64b3002e188d766266f2464c02f` was listed read-only before this phase. No blob from that tree was copied.

Excluded, not extracted:

- `vault-skeleton/`, `skills/`, the chat-gateway integration, Telegram tests, and vault policy or allowlist files. Those are product-specific.
- `reference/*.ps1`, `reference/*.bat`, and service examples. Those bind a particular host layout.
- `reference/rag_*.py`. The hybrid module is a flat script that imports vault ranking helpers and NumPy. Copying it would pull those couplings into this package and would widen the product record.

What was reused as an idea only: Unicode tokenization, BM25 with `k1=1.2` and `b=0.75`, an immutable snapshot pointer, and a content-addressed vector cache. Those behaviors were reimplemented here against `SourceDocument`.
