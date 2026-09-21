# Architecture

A text store implements `SourceAdapter` and yields `SourceDocument` records. Those records are chunked, stored in an immutable snapshot, and searched with BM25. A quality profile also embeds chunks, unions the lexical and dense candidate sets, and reranks that union.

Benchmark groups, grades, and translation pairs are not fields on `SourceDocument`. See [contracts.md](contracts.md).

If the pinned embedding or reranker runtime is missing, search stays on the lexical ranking and sets `degraded` to true. It does not invent a dense score.

Indexes, caches, and model weights stay outside git.
