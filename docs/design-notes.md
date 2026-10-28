# Design notes

## Reliability choices
1. Keep retrieval separate from orchestration so it can be tested and replaced independently.
2. Preserve filename and chunk ID metadata from ingestion through the answer prompt and response.
3. Bound retries to prevent runaway agent loops.
4. Abstain when retrieved evidence does not meet the configured minimum rather than fabricate a confident answer.
5. Treat document contents as untrusted input in the generation prompt.

## Evaluation plan
1. Write 30–50 questions and manually mark the relevant source documents/chunks.
2. Compare dense retrieval with and without reranking at a fixed K.
3. Measure Recall@K and MRR@K on the same held-out questions.
4. Separately label citation correctness, faithfulness, and abstention quality for generated answers.
5. Record model versions, settings, latency, and failed cases. Avoid conclusions from the tiny bundled smoke-test set.
