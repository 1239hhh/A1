from dataclasses import asdict
from pathlib import Path
import json
from typing import Any
import numpy as np
from app.chunking import TextChunk

class EvidenceRetriever:
    """BGE-M3 dense retrieval with optional BGE cross-encoder reranking."""
    def __init__(self, embedding_model: str = "BAAI/bge-m3", reranker_model: str = "BAAI/bge-reranker-v2-m3", index_dir: str = "data/index"):
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.index_dir = Path(index_dir)
        self.chunks: list[TextChunk] = []
        self._embedder: Any = None
        self._reranker: Any = None
        self._index: Any = None

    def _load_embedder(self):
        if self._embedder is None:
            from FlagEmbedding import BGEM3FlagModel
            self._embedder = BGEM3FlagModel(self.embedding_model_name, use_fp16=False)
        return self._embedder

    def _load_reranker(self):
        if self._reranker is None:
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(self.reranker_model_name, use_fp16=False)
        return self._reranker

    def build(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            raise ValueError("No text chunks to index")
        import faiss
        vectors = self._load_embedder().encode([c.text for c in chunks], batch_size=8, max_length=1024)["dense_vecs"]
        vectors = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self.chunks = chunks
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "vectors.faiss"))
        (self.index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        return len(chunks)

    def load(self) -> bool:
        import faiss
        index_file, metadata_file = self.index_dir / "vectors.faiss", self.index_dir / "chunks.json"
        if not index_file.exists() or not metadata_file.exists():
            return False
        self._index = faiss.read_index(str(index_file))
        self.chunks = [TextChunk(**item) for item in json.loads(metadata_file.read_text(encoding="utf-8"))]
        return True

    def search(self, query: str, top_k: int = 8, rerank_top_k: int = 4) -> list[dict[str, Any]]:
        if self._index is None and not self.load():
            raise RuntimeError("Index not found. Ingest documents first.")
        vector = np.asarray(self._load_embedder().encode([query])["dense_vecs"], dtype="float32")
        import faiss
        faiss.normalize_L2(vector)
        scores, ids = self._index.search(vector, min(top_k, len(self.chunks)))
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                chunk = self.chunks[int(idx)]
                candidates.append({"text": chunk.text, "source": chunk.source, "chunk_id": chunk.chunk_id, "score": float(score)})
        if not candidates:
            return []
        rerank_scores = self._load_reranker().compute_score([[query, item["text"]] for item in candidates], normalize=True)
        if isinstance(rerank_scores, (float, int)):
            rerank_scores = [rerank_scores]
        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(score)
        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:rerank_top_k]
