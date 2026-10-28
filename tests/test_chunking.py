import pytest
from app.chunking import split_text
from app.agent import evidence_is_sufficient

def test_split_text_keeps_source_and_ids():
    chunks = split_text("alpha beta gamma delta epsilon zeta eta theta", "guide.md", chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(chunk.source == "guide.md" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("hello", "a.txt", chunk_size=10, overlap=10)

def test_evidence_gate():
    assert evidence_is_sufficient([{"text": "x" * 90}], 80)
    assert not evidence_is_sufficient([{"text": "too short"}], 80)
