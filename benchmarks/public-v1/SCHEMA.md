# public-v1 schema

This fixture is original synthetic text. It is a regression set, not a hidden acceptance test.

`groups.jsonl` holds one row per source group: `group_id`, `fold`, `topic`, `category`. A Russian/English translation pair shares that group and therefore one fold.

`documents.jsonl` holds benchmark documents. `group_id`, `pair_id`, `role`, and `supersedes` are annotations. They are not fields of the product `SourceDocument`.

`queries.jsonl` holds one Russian query and one English query per group. `language` is the language of the query text.

Grades used here are 3, 1, and 0. Grade 2 is not used.

- Grade 3 is the designated supporting document for that query class, and the evidence snippet is an exact substring of its text.
- Grade 1 is the same-fact translation in the other language. It is related, not success.
- Grade 0 is not relevant.

For `cross_language`, grade 3 is the opposite-language document. For every other category, grade 3 is the same-language document.

Binary Hit, Recall, MRR, and MAP treat only grade 3 as relevant. `map` is mean average precision over the full ranked list. nDCG@8 uses the raw grades.
