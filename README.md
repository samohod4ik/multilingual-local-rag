# multilingual-local-rag

Local-first retrieval experiments over pluggable text sources.

## What exists in this version

This version publishes typed contracts, a synthetic Russian/English benchmark, and an evaluator. It does **not** implement retrieval, reranking, indexing, or answer generation.

- Product records are `SourceDocument` values: `source_id`, `source_uri`, `text`, `content_hash`, `revision`, `media_type`, `metadata`, and an optional BCP 47 `language` tag.
- Benchmark groups, translation pairs, roles, supersession, and grades live in separate annotation types. See [docs/contracts.md](docs/contracts.md).
- `benchmarks/public-v1/` is an original synthetic fixture: 40 source groups and 80 paired queries.
- The `map` metric is mean average precision over the full ranked list. It is not MAP@8. nDCG is reported at 8.

## Checks

Use the repository virtualenv on Python 3.12. From the repository root:

```text
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m multilingual_local_rag.cli validate-benchmark benchmarks/public-v1
```

On Unix, call `.venv/bin/python` instead of the Windows path above.

## License

MIT for original application code. Model and dataset licenses, when any are added later, stay separate.
