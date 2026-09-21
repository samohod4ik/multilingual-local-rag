# Operations

`mlrag index` writes a snapshot directory and then replaces `current.json`. A failed embedding does not move that pointer.

`mlrag search --profile lexical` uses BM25 only. `--profile quality` uses the union and reranker when that runtime is installed. Otherwise the result is marked degraded.

Keep the data root off the git worktree. Snapshots contain copies of source text. Retention keeps the current snapshot and two older ones, ordered by directory modification time.

Lexical ranking is a real fallback, not the same quality as the pinned GPU profile. No hardware number in this repository is a service level.
