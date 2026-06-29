"""Tests for data loading and document parsing."""

from src.data import doc_to_markdown, get_record

## Better to create a dummy record that always present in the data for testing
TEST_RECORD_ID = "Single_MRO/2007/page_134.pdf-1"


def test_get_record_by_id():
    record = get_record(TEST_RECORD_ID)
    assert record is not None
    assert record.id == TEST_RECORD_ID


def test_get_record_missing_id():
    assert get_record("nonexistent-id") is None


def test_doc_to_markdown_structure():
    """Markdown output must contain all three sections — the LLM prompt depends on this layout."""
    record = get_record(TEST_RECORD_ID)
    assert record is not None
    md = doc_to_markdown(record)
    assert "## Context" in md
    assert "## Table" in md
    assert "## Notes" in md
