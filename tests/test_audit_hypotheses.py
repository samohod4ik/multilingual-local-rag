from pathlib import Path

from multilingual_local_rag.audit import audit_tree

ROOT = Path(__file__).resolve().parents[1]


def test_repository_audit_is_clean() -> None:
    assert audit_tree(ROOT) == []


def test_audit_detects_slash_user_prefix(tmp_path: Path) -> None:
    payload = "C:/" + "Users/" + "example\n"
    (tmp_path / "note.txt").write_text(payload, encoding="utf-8")
    problems = audit_tree(tmp_path)
    assert any(item.endswith(": user-prefix") for item in problems)


def test_hypotheses_cover_h001_through_h007() -> None:
    folder = ROOT / "research" / "hypotheses"
    for number in range(1, 8):
        text = (folder / f"H{number:03d}.md").read_text(encoding="utf-8")
        assert "status: open" in text
    h004 = (folder / "H004.md").read_text(encoding="utf-8")
    assert "status: open" in h004
    assert "not the default" in h004
    assert "No result is claimed" in h004
