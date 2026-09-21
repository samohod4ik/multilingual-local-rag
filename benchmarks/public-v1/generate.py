"""Write the original synthetic public-v1 fixture. Do not import another corpus."""

from __future__ import annotations

import json
from pathlib import Path

from multilingual_local_rag.contracts import sha256_text

NAMES = (
    "Amberlathe",
    "Brineclock",
    "Cedarwinch",
    "Dunewell",
    "Echoforge",
    "Farrowgate",
    "Glimmerock",
    "Hearthloom",
    "Irisvault",
    "Juniperlock",
    "Kestrelspool",
    "Larkmeter",
    "Mossbridge",
    "Nettlebarge",
    "Ospreykiln",
    "Ploverdock",
    "Quillpress",
    "Rowanstill",
    "Sablepump",
    "Thornferry",
    "Umberloom",
    "Vellumrack",
    "Willowcrane",
    "Yarrowpress",
    "Zephyrvat",
    "Ashencoil",
    "Bramblefan",
    "Cindervane",
    "Driftanchor",
    "Emberrail",
    "Frosthinge",
    "Gorselever",
    "Hazelvein",
    "Inkbarrel",
    "Jasperloom",
    "Kindleweir",
    "Lichenpress",
    "Marrowgate",
    "Nightlatch",
    "Oakensieve",
)
UNITS = (
    ("revolutions", "оборотов"),
    ("minutes", "минут"),
    ("litres", "литров"),
    ("degrees", "градусов"),
    ("percent", "процентов"),
)


def category_for(index: int) -> str:
    if index <= 8:
        return "same_language_lexical"
    if index <= 16:
        return "paraphrase"
    if index <= 24:
        return "cross_language"
    if index <= 28:
        return "identifier_rich"
    if index <= 32:
        return "hub_distractor"
    if index <= 36:
        return "conflicting_superseded"
    return "exact_evidence"


def _en_span(name: str, number: int, unit: str, serial: str, category: str) -> str:
    base = f"The rated limit of the {name} unit is {number} {unit}."
    if category == "identifier_rich":
        return f"{base} Serial {serial}."
    return base


def _ru_span(name: str, number: int, unit: str, serial: str, category: str) -> str:
    base = f"Номинальный предел установки {name} равен {number} {unit}."
    if category == "identifier_rich":
        return f"{base} Серийный номер {serial}."
    return base


def _body(span: str, category: str) -> str:
    if category == "exact_evidence":
        return f"Workshop note. {span} Keep the log beside the unit."
    return span


def _query_text(name: str, language: str, category: str, serial: str) -> str:
    if category == "paraphrase":
        if language == "en":
            return f"How large is the cap recorded for {name}?"
        return f"Какой потолок записан для установки {name}?"
    if category == "identifier_rich":
        if language == "en":
            return f"What rated limit belongs to serial {serial}?"
        return f"Какой номинальный предел у серийного номера {serial}?"
    if language == "en":
        return f"What is the rated limit of the {name} unit?"
    return f"Каков номинальный предел установки {name}?"


def build_rows() -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    groups: list[dict[str, object]] = []
    documents: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    query_number = 1
    for index, name in enumerate(NAMES, start=1):
        category = category_for(index)
        group_id = f"GRP-{index:03d}"
        pair_id = f"PAIR-{index:03d}"
        unit_en, unit_ru = UNITS[(index - 1) % len(UNITS)]
        number = 20 + index * 3
        serial = f"SN-{index:04d}"
        groups.append(
            {
                "group_id": group_id,
                "fold": f"fold-{(index % 5) + 1}",
                "topic": name,
                "category": category,
            }
        )
        en_span = _en_span(name, number, unit_en, serial, category)
        ru_span = _ru_span(name, number, unit_ru, serial, category)
        en_text = _body(en_span, category)
        ru_text = _body(ru_span, category)
        en_id = f"DOC-E-{index:03d}"
        ru_id = f"DOC-R-{index:03d}"
        en_doc = _doc(en_id, group_id, "en", name, en_text, pair_id, "primary", None, "1")
        ru_doc = _doc(ru_id, group_id, "ru", name, ru_text, pair_id, "primary", None, "1")
        if category == "conflicting_superseded":
            old_id = f"DOC-S-{index:03d}"
            old_text = _en_span(name, number - 5, unit_en, serial, "same_language_lexical")
            documents.append(
                _doc(old_id, group_id, "en", name, old_text, None, "superseded", None, "0")
            )
            en_doc["supersedes"] = old_id
        if category == "hub_distractor":
            hub = f"Index card mentions {name} and does not state a numeric limit."
            distractor = f"The rated limit of the {name} unit is 9999 {unit_en}."
            documents.append(
                _doc(f"DOC-H-{index:03d}", group_id, "en", name, hub, None, "hub", None, "1")
            )
            documents.append(
                _doc(
                    f"DOC-D-{index:03d}",
                    group_id,
                    "en",
                    name,
                    distractor,
                    None,
                    "distractor",
                    None,
                    "1",
                )
            )
        documents.append(en_doc)
        documents.append(ru_doc)
        for language, gold_id, other_id, span in (
            ("en", en_id, ru_id, en_span),
            ("ru", ru_id, en_id, ru_span),
        ):
            if category == "cross_language":
                grade3, grade1, snippet = (
                    other_id,
                    gold_id,
                    ru_span if language == "en" else en_span,
                )
            else:
                grade3, grade1, snippet = gold_id, other_id, span
            qrels = [
                {"source_id": grade3, "grade": 3},
                {"source_id": grade1, "grade": 1},
            ]
            queries.append(
                {
                    "query_id": f"QRY-{query_number:03d}",
                    "group_id": group_id,
                    "language": language,
                    "category": category,
                    "text": _query_text(name, language, category, serial),
                    "qrels": qrels,
                    "evidence": [{"source_id": grade3, "snippet": snippet}],
                }
            )
            query_number += 1
    return groups, documents, queries


def _doc(
    source_id: str,
    group_id: str,
    language: str,
    title: str,
    text: str,
    pair_id: str | None,
    role: str,
    supersedes: str | None,
    revision: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "group_id": group_id,
        "language": language,
        "title": title,
        "text": text,
        "source_uri": f"groups/{group_id}/{source_id}.txt",
        "content_hash": sha256_text(text),
        "revision": revision,
        "media_type": "text/plain",
        "pair_id": pair_id,
        "role": role,
        "supersedes": supersedes,
    }


def write_fixture(directory: str | Path) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    groups, documents, queries = build_rows()
    _dump(root / "groups.jsonl", groups)
    _dump(root / "documents.jsonl", documents)
    _dump(root / "queries.jsonl", queries)


def _dump(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    write_fixture(Path(__file__).resolve().parent)
