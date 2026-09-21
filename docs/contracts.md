# Contracts

Adapters yield `SourceDocument` records. Those records describe text a store can hand to an indexer. They do not describe how a benchmark grades that text.

## Product record

| Field | Meaning |
| --- | --- |
| `source_id` | Stable identifier inside one source. |
| `source_uri` | Relative URI. Absolute paths and `..` are rejected. |
| `text` | UTF-8 text. |
| `content_hash` | SHA-256 of `text`. |
| `revision` | Caller-defined revision token. |
| `media_type` | Such as `text/plain` or `text/markdown`. |
| `metadata` | String map. Titles belong here. |
| `language` | Optional BCP 47 tag. The product type does not close the set at Russian and English. |

`group_id`, `pair_id`, `role`, and `supersedes` are not product fields.

## Benchmark annotations

`public-v1` restricts its own fixture to `ru` and `en`. That restriction is checked when the fixture loads. It is not a property of `SourceDocument`.

Each source group has one Russian query and one English query. The pair stays in one fold.

Grades are 3, 1, and 0. Grade 3 is the designated supporting document and must contain the evidence snippet as an exact substring. Grade 1 is the other-language translation of the same fact. For `cross_language`, grade 3 is the opposite-language document. Binary Hit, Recall, MRR, and MAP count only grade 3. nDCG@8 uses the raw grades.

`map` averages average precision over the whole ranked list.

## Providers

Embedder and reranker protocols are declared so later code can implement them. Importing the package does not import a model runtime.
