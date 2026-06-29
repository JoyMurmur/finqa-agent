"""
Data models and loader for ConvFinQA dataset.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from tabulate import tabulate

DATA_PATH = Path(__file__).parent.parent / "data" / "convfinqa_dataset.json"


class Document(BaseModel):
    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class Dialogue(BaseModel):
    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | str]
    qa_split: list[bool]


class Features(BaseModel):
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features


@lru_cache(maxsize=1)
def _load() -> dict:
    """Load and cache the full dataset (20MB JSON read once per process)."""
    with open(DATA_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_index() -> dict[str, dict]:
    """Build a flat id→record dict for O(1) lookup across all splits."""
    data = _load()
    return {r["id"]: r for split in ("train", "dev") for r in data.get(split, [])}


def get_record(record_id: str) -> ConvFinQARecord | None:
    """Get a record by ID."""
    raw = _build_index().get(record_id)
    return ConvFinQARecord.model_validate(raw) if raw is not None else None


def get_records(split: str = "dev") -> list[ConvFinQARecord]:
    """Get all records from a split."""
    return [ConvFinQARecord.model_validate(r) for r in _load().get(split, [])]


def doc_to_markdown(record: ConvFinQARecord) -> str:
    """Render record document into a compact markdown context."""
    df = pd.DataFrame(record.doc.table)
    table_md = tabulate(df, headers="keys", tablefmt="pipe", showindex=True)
    return (
        f"### Context (Pre-table text)\n{record.doc.pre_text}\n\n"
        f"---\n\n### Table\n{table_md}\n\n"
        f"---\n\n### Notes (Post-table text)\n{record.doc.post_text}\n"
    )


if __name__ == "__main__":
    _RECORD_ID = "Single_RSG/2008/page_114.pdf-2"
    records = get_records(split="train")
    record = get_record(record_id=_RECORD_ID)
    assert record is not None
    md = doc_to_markdown(record)
    print(md)  # noqa: T201
