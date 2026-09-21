# Installation

Python 3.12 is required. From a clone:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

On Unix, use `.venv/bin/python` instead of the Windows path. Do not call an unqualified `python` if another environment is earlier on `PATH`.

CPU checks:

```text
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest -q
```

Pinned GPU weights are not installed by these commands.
