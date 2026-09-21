# public-v1

Original synthetic Russian and English notes. Forty source groups, eighty queries.

Regenerate with the repository virtualenv:

```text
.venv\Scripts\python.exe benchmarks/public-v1/generate.py
```

The generator imports the package, so `PYTHONPATH` must include `src` until the package is installed. The committed JSONL files are the fixture the tests load.
