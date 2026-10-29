# EvidenceFlow | Enterprise Knowledge Agent

EvidenceFlow is a source-grounded question-answering system for internal knowledge bases. It combines dense retrieval, cross-encoder reranking, and a bounded LangGraph workflow to produce answers with source references or abstain when retrieved evidence is insufficient.

## Highlights

- **Retrieval pipeline:** BGE-M3 embeddings and FAISS similarity search, followed by BGE reranking.
- **Agent orchestration:** LangGraph nodes for retrieval, evidence sufficiency checks, bounded retry, and answer generation.
- **Grounded responses:** source filenames and chunk identifiers are carried into the generation context for traceability.
- **FastAPI service:** HTTP endpoints for health checks, document ingestion, and question answering.
- **Retrieval evaluation:** a small evaluation harness reports Recall@K and MRR@K against labelled relevant sources.
- **Configuration hygiene:** environment-based model settings and Git exclusions for local secrets.

## Architecture

```text
TXT / Markdown files
        |
        v
Chunking + source metadata
        |
        v
BGE-M3 embeddings -> FAISS retrieval
        |
        v
BGE cross-encoder reranking
        |
        v
LangGraph evidence check
   | sufficient       | insufficient
   v                  v
Answer with sources   Bounded retry -> answer or abstain
        |
        v
      FastAPI
```

## Technology

Python · FastAPI · LangGraph · LangChain · BGE-M3 · FAISS · BGE Reranker · pytest

## Run locally (Windows / PowerShell)

Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `LLM_API_KEY` in `.env` to a valid compatible chat-completions API key. Keep `.env` private and never commit it.

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service and index status |
| POST | `/ingest` | Chunk supported TXT/Markdown files and build the vector index |
| POST | `/query` | Retrieve evidence and generate a source-grounded answer |

To ingest the included example handbook, call `POST /ingest` before sending a question to `POST /query`.

Example request:

```json
{
  "question": "What should an employee do if they still cannot sign in?"
}
```

## Tests and evaluation

Run the unit tests:

```powershell
python -m pytest -q
```

After ingesting documents, run the retrieval evaluation:

```powershell
python -m app.evaluate
```

The evaluation script reports Recall@K and MRR@K on the questions in `evals/golden_qa.json`. For a meaningful comparison, expand the dataset with manually verified questions and compare retrieval configurations using the same labelled set. No benchmark result is claimed in this repository.

## Current scope

The ingestion pipeline currently supports TXT and Markdown. The evidence sufficiency gate is a configurable length-based heuristic, and the retry applies a deterministic query reformulation. PDF ingestion, a semantic evidence judge, authentication, and deployment automation are potential next improvements.

## Acknowledgement

The design was informed by studying [pratham-jaiswal/agentic-rag](https://github.com/pratham-jaiswal/agentic-rag), an MIT-licensed reference project. This repository's implementation and evaluation should be reviewed and extended as part of the portfolio project.
