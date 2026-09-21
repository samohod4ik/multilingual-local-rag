from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "docs/architecture.md",
    "docs/source-adapters.md",
    "docs/installation.md",
    "docs/api.md",
    "docs/operations.md",
    "docs/evaluation.md",
    "docs/security.md",
    "docs/model-and-data-licenses.md",
    "docs/provenance.md",
    "docs/roadmap.md",
    "docs/contracts.md",
)


def test_required_docs_exist() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert missing == []
