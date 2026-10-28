from dataclasses import dataclass
from pathlib import Path
import re

SUPPORTED_SUFFIXES = {".txt", ".md"}

@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    chunk_id: str
    start_char: int

def split_text(text: str, source: str, chunk_size: int = 900, overlap: int = 120) -> list[TextChunk]:
    """Split text into overlapping chunks while retaining source and offset metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    cleaned = re.sub(r"\s+", " ", text).strip()
    chunks: list[TextChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(TextChunk(piece, source, f"{Path(source).stem}:{len(chunks):04d}", start))
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks

def load_text_documents(directory: str) -> list[TextChunk]:
    """Load .txt and .md files and chunk them deterministically."""
    root = Path(directory)
    if not root.exists():
        return []
    chunks: list[TextChunk] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES):
        chunks.extend(split_text(path.read_text(encoding="utf-8"), path.name))
    return chunks
