# Source adapters

`FilesystemAdapter` reads `.txt`, `.md`, and `.markdown` files as UTF-8, including a leading BOM. It does not follow symlinks or other reparse points. On Windows, a resolved path must stay inside the root even when letter case differs.

Secret-like names such as `.env`, `credentials.md`, and `id_rsa.txt` are skipped. Returned `source_uri` values are relative. Empty files are skipped.

Another store implements the same `SourceAdapter` protocol. v1 does not ship extra connectors.
