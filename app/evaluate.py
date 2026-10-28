"""Small, transparent retrieval evaluation; no benchmark claims are hard-coded."""
import json
from pathlib import Path
from app.config import settings
from app.retrieval import EvidenceRetriever

def evaluate(dataset_path: str = "evals/golden_qa.json") -> dict:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    retriever = EvidenceRetriever(settings.embedding_model, settings.reranker_model, settings.index_dir)
    if not retriever.load():
        raise RuntimeError("No index found. Run document ingestion first.")
    hits, reciprocal_ranks = 0, []
    for item in dataset:
        results = retriever.search(item["question"])
        expected = set(item["relevant_sources"])
        rank = next((i for i, result in enumerate(results, 1) if result["source"] in expected), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    total = len(dataset)
    return {"questions": total, "recall_at_k": hits / total if total else 0.0, "mrr_at_k": sum(reciprocal_ranks) / total if total else 0.0, "k": settings.rerank_top_k}

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
