# Implementation: Budget-Constrained Self-Corrective RAG

## Project Objective

Make **Budget-Constrained Self-Corrective RAG with Failure-Aware Routing** the
project's only question-answering application.

Reuse the existing document ingestion, vector database, and LLM integration.
Do not retain a separate legacy ReAct or normal-RAG answering path.

First inspect the existing repository, architecture, backend, frontend, retrieval pipeline, LLM integration, vector database, and existing evaluation utilities.

Preserve the reusable document ingestion, vector store, configuration, and
evaluation functionality.

## Research Contribution

Implement and evaluate a rate-limited corrective retrieval policy.

Research question:

Can a maximum of two corrective retrieval attempts reduce retrieval overhead and response latency while maintaining answer quality and evidence support?

The correction budget is:

- Initial retrieval: not counted as a correction.
- Correction attempt 1: rewritten query + local retrieval.
- Correction attempt 2: external web search.
- Maximum corrective attempts per question: 2.
- Never execute additional corrective retrieval after the budget is exhausted.

## Required Implementation

### 1. Retrieval Evidence Grader

Create or extend an evidence evaluation module.

For each retrieval result, evaluate:

- Relevance to the question.
- Whether the evidence contains information necessary to answer.
- Whether evidence is insufficient or contradictory.
- Whether the answer should proceed, correct, or abstain.

Use existing models and infrastructure wherever possible.

Do not treat vector similarity as factual correctness.

### 2. Rate-Limited CRAG Pipeline

Implement:

Initial retrieval:
- Retrieve documents from the existing vector database.
- Evaluate retrieval evidence.
- If sufficient, generate the answer.

Correction attempt 1:
- Rewrite the original query.
- Perform another local retrieval.
- Re-evaluate evidence.
- If sufficient, generate the answer.

Correction attempt 2:
- If local evidence is still insufficient, execute web search.
- Merge relevant local and external evidence.
- Validate evidence before generation.
- If no sufficient evidence exists, abstain.

Ensure the correction counter is maintained independently for each question.

Expose the complete correction trace in the backend response.

### 3. Comparative Evaluation

Implement three selectable evaluation policies:

A. BASELINE_RAG
- Initial retrieval.
- Direct generation.

B. STANDARD_CRAG
- Initial retrieval.
- Evidence evaluation.
- One corrective action if retrieval is insufficient.

C. RATE_LIMITED_CRAG
- Initial retrieval.
- Query rewrite and local re-retrieval.
- Web fallback if needed.
- Maximum two corrective attempts.

Use identical questions, corpus, model configuration, and evaluation settings for fair comparison.

### 4. Evaluation Metrics

Record:

- Total questions evaluated.
- Answer correctness, where ground truth is available.
- Evidence support / grounding.
- Retrieval attempts per question.
- Web search calls.
- Correction budget utilization.
- End-to-end latency.
- Retrieval latency.
- Generation latency.
- Abstention rate.
- Correction success rate.

Do not fabricate evaluation results or claim improvements without measurements.

### 5. Research Evaluation Dataset

Create a small reproducible evaluation dataset containing:

- Questions answerable from local documents.
- Questions requiring query rewriting.
- Questions answerable only through web search.
- Questions with irrelevant local retrieval.
- Questions with conflicting evidence.
- Questions with insufficient evidence requiring abstention.

Provide expected answers or reference evidence where feasible.

Support repeated evaluation using the same dataset.

### 6. Frontend Research Dashboard

Extend the existing Streamlit frontend.

Add a CRAG research dashboard containing:

- Question input.
- Policy selection.
- Final generated answer.
- Retrieved local evidence.
- External web evidence when used.
- Evidence grading result.
- Correction attempt counter.
- Query rewriting history.
- Routing decisions.
- Sources and citations.
- Total latency.
- Retrieval and web search counts.
- Abstention indicator.

Add an evaluation page displaying comparative results for:

- Baseline RAG.
- Standard CRAG.
- Rate-Limited CRAG.

Use tables and charts for latency, answer quality, search calls, and correction success.

### 7. Explainability

For each query, expose a trace similar to:

{
  "initial_retrieval": "INSUFFICIENT",
  "correction_attempt_1": "QUERY_REWRITE",
  "local_retrieval_after_rewrite": "INSUFFICIENT",
  "correction_attempt_2": "WEB_SEARCH",
  "final_evidence_status": "SUFFICIENT",
  "final_action": "GENERATE",
  "correction_count": 2
}

Adapt the actual schema to the existing application.

### 8. Reliability

- Handle web search failures.
- Handle empty retrieval.
- Handle model failures.
- Handle unsupported questions.
- Enforce correction budget strictly.
- Prevent infinite loops.
- Preserve source citations.
- Never silently invent evidence.
- Never fabricate evaluation metrics.

### 9. Implementation Constraints

- Reuse existing dependencies.
- Do not unnecessarily change database schemas.
- Do not replace the existing vector database.
- Do not rebuild the frontend.
- Keep the implementation compatible with the existing local environment.
- Avoid introducing unnecessary paid APIs.
- Use existing LLM integration wherever possible.
- If web search requires an API key, implement a configurable provider and clearly document setup.
- Route all question answering through the CRAG pipeline; keep the three
  research policies selectable within that pipeline.

## Execution Instructions

Work incrementally:

1. Inspect existing repository.
2. Identify reusable components.
3. Provide a concise implementation plan.
4. Implement backend correction pipeline.
5. Implement correction budget enforcement.
6. Implement policy comparison.
7. Implement evaluation dataset and metrics.
8. Implement frontend research dashboard.
9. Add tests.
10. Update README and research methodology documentation.
11. Verify the CRAG answering and evaluation workflows remain intact.
12. Run the application and report actual results.

Do not stop after providing an architecture explanation.

Actually implement the changes in the existing repository.

At the end, report:
- Implemented components.
- Modified files.
- Commands to run.
- Tests executed and their results.
- Remaining limitations.
- How to demonstrate the research contribution during the review.