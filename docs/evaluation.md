# Evaluation

`benchmarks/public-v1` has 40 source groups and 80 queries. Each group has one Russian query and one English query, and that pair shares one fold.

Grades are 3, 1, and 0. Hit, Recall, MRR, and MAP count only grade 3. `map` is mean average precision over the full ranked list. nDCG uses the raw grades at cutoff 8. Evidence recall needs the gold snippet as an exact substring.

`evaluate` and `compare` reject a run that is not the full query-id set. Hypothesis cards in `research/hypotheses` stay `open` until a recorded run says otherwise. H004 does not claim a result.
