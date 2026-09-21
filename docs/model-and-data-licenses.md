# Model and data licenses

The MIT license covers original application code only.

`benchmarks/public-v1` is original synthetic text written for this repository. It is not a public corpus under another license.

`models/manifest.json` names the pinned embedder and reranker slots. Their weights are not in git. Whoever downloads those weights must follow the upstream license. This repository does not relicense them.

Development dependencies are pinned with hashes in `requirements-dev.lock`. Their licenses stay with the upstream projects. See [THIRD_PARTY.md](../THIRD_PARTY.md).
