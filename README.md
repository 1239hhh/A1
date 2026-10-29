# EvidenceFlow — Enterprise Knowledge Agent

A source-grounded enterprise Q&A portfolio project built around BGE-M3, FAISS, BGE cross-encoder reranking, LangGraph and FastAPI.

> **Status:** initial implementation scaffold. Run the checks and complete the evaluation before reporting measured performance in a CV. The bundled handbook is synthetic.

## Architecture
TXT/Markdown → source-aware chunking → BGE-M3 embeddings → FAISS retrieval → BGE reranking → LangGraph evidence check / bounded retry → cited answer or safe abstention → FastAPI.

## Features
- Source/chunk metadata preserved through retrieval.
- BGE-M3 dense embeddings, FAISS similarity search, BGE reranking.
- LangGraph workflow with evidence checks, bounded retries, and safe abstention.
- FastAPI endpoints: `/health`, `/ingest`, `/query`.
- Small Recall@K / MRR@K evaluation harness.
- `.env.example` and `.gitignore` keep credentials out of source control.

## Windows quick start
Run commands from the repository root in PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
pytest -q
uvicorn app.main:app --reload
```

If PowerShell blocks virtual-environment activation, use `.venv\\Scripts\\python.exe -m pip install -r requirements.txt` and `.venv\\Scripts\\python.exe -m pytest -q` instead. Installing requirements installs pytest; do not run tests before that step.

Set `LLM_API_KEY` in `.env` before querying. Never commit `.env`. Open http://127.0.0.1:8000/docs, call `POST /ingest`, then `POST /query`. The FastAPI entry point is `app.main:app` (the `app/` package contains the application). Avoid `uvicorn app:app`, because the root `app.py` conflicts with the `app/` package name.

Run `python -m app.evaluate` after indexing. The bundled four-question set is a smoke test only; expand it with manually verified questions before claiming performance improvements.

## Honest evaluation notes
The evidence gate is currently a transparent length heuristic, not a semantic judge. The retry uses deterministic query reformulation. Evaluate both against a labelled baseline rather than assuming they improve quality. PDF ingestion, authentication, CI, and deployment hardening are future work.

## Reference
The high-level design was informed by studying [pratham-jaiswal/agentic-rag](https://github.com/pratham-jaiswal/agentic-rag), an MIT-licensed project. No source files are copied into this implementation.