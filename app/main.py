from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.config import settings
from app.chunking import load_text_documents
from app.retrieval import EvidenceRetriever
from app.agent import KnowledgeAgent

app = FastAPI(title="EvidenceFlow Knowledge Agent", version="0.1.0", description="Source-grounded enterprise knowledge assistant with bounded agentic retrieval.")
retriever = EvidenceRetriever(settings.embedding_model, settings.reranker_model, settings.index_dir)
agent: KnowledgeAgent | None = None

class IngestResponse(BaseModel):
    indexed_chunks: int
    data_dir: str

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

@app.get("/health")
def health():
    return {"status": "ok", "index_exists": (Path(settings.index_dir) / "vectors.faiss").exists()}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    chunks = load_text_documents(settings.data_dir)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No .txt or .md documents found in {settings.data_dir}")
    try:
        count = retriever.build(chunks)
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing retrieval dependency: {exc}") from exc
    return IngestResponse(indexed_chunks=count, data_dir=settings.data_dir)

@app.post("/query")
def query(request: QueryRequest):
    global agent
    try:
        if agent is None:
            agent = KnowledgeAgent(retriever)
        return agent.ask(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed. Check model configuration and service logs. ({type(exc).__name__})") from exc
