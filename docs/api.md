# HTTP API

`mlrag serve` listens on `127.0.0.1` only.

- `GET /v1/capabilities`
- `GET /v1/health`
- `POST /v1/search` with `{"query": "..."}`
- `POST /v1/answer` with `{"query": "..."}`

Search and answer responses include `evidence`. An answer that cites an unknown source id is discarded and the call abstains. The evidence list remains.

The body limit is 16 KiB. The `Host` header must be loopback. Query text is not written to a log.
