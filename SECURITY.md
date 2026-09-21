# Security

Report vulnerabilities privately to the repository owner. Do not open a public issue that includes a proof of concept against a host you do not control.

The optional HTTP service binds only to loopback and does not record query text. Do not point it at a public interface.

Do not commit indexes. They would contain copies of source text.

Manifest and source URIs reject absolute paths and `..` segments.
