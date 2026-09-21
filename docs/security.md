# Security notes

See also [SECURITY.md](../SECURITY.md).

The filesystem adapter and manifest resolver reject `..` and absolute paths. The HTTP service binds to loopback, checks the `Host` header, and does not follow client redirects.

Do not commit indexes. They copy source text. Report vulnerabilities privately to the repository owner.
