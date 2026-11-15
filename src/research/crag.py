"""Budget-constrained self-corrective retrieval pipeline."""

import json
import re
import time
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import requests
from langchain_core.documents import Document

from src.research.web_search import search_wikipedia


class CRAGPipeline:
    """Run the strict two-correction retrieval path against a retriever and LLM."""

    _MAX_CORRECTIONS = 2

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        web_search: Callable[[str], List[Document]] = search_wikipedia,
    ):
        self.retriever = retriever
        self.llm = llm
        self.web_search = web_search

    def run(self, question: str) -> Dict[str, Any]:
        """Answer a question and return evidence, metrics, and an explainability trace."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        started = time.perf_counter()
        metrics = {
            "retrieval_attempts": 0,
            "web_search_calls": 0,
            "correction_count": 0,
            "correction_budget": self._MAX_CORRECTIONS,
            "retrieval_latency_seconds": 0.0,
            "web_search_latency_seconds": 0.0,
            "generation_latency_seconds": 0.0,
        }
        trace: List[Dict[str, Any]] = []
        query_history = [question]
        initial_docs = self._retrieve(question, metrics)
        local_docs = list(initial_docs)
        web_docs: List[Document] = []
        grade = self.grade_evidence(question, initial_docs)
        trace.append(self._grade_trace("initial_retrieval", grade, len(initial_docs)))
        sufficient = grade["sufficient"]

        if not sufficient:
            self._rewrite_and_retrieve(question, query_history, local_docs, metrics, trace)
            grade = self.grade_evidence(question, local_docs)
            trace.append(self._grade_trace("local_retrieval_after_rewrite", grade, len(local_docs)))
            sufficient = grade["sufficient"]

            if not sufficient and metrics["correction_count"] < self._MAX_CORRECTIONS:
                metrics["correction_count"] += 1
                metrics["web_search_calls"] += 1
                web_started = time.perf_counter()
                try:
                    web_docs = self.web_search(query_history[-1])
                except requests.RequestException as exc:
                    trace.append(
                        {
                            "step": "correction_attempt_2",
                            "action": "WEB_SEARCH",
                            "evidence_status": "UNAVAILABLE",
                            "document_count": 0,
                            "error": str(exc),
                        }
                    )
                else:
                    trace.append(
                        {
                            "step": "correction_attempt_2",
                            "action": "WEB_SEARCH",
                            "evidence_status": "RETRIEVED" if web_docs else "EMPTY",
                            "document_count": len(web_docs),
                        }
                    )
                finally:
                    metrics["web_search_latency_seconds"] += time.perf_counter() - web_started

                combined_docs = self._deduplicate_documents(local_docs + web_docs)
                grade = self.grade_evidence(question, combined_docs)
                trace.append(
                    self._grade_trace(
                        "final_evidence_evaluation", grade, len(combined_docs)
                    )
                )
                sufficient = grade["sufficient"]

        evidence_docs = self._deduplicate_documents(local_docs + web_docs)
        answer = "I can't answer this from the available evidence."
        action = "ABSTAIN"
        if sufficient:
            generation_started = time.perf_counter()
            answer = self._generate_answer(question, evidence_docs)
            metrics["generation_latency_seconds"] = time.perf_counter() - generation_started
            action = "GENERATE"

        trace.append(
            {
                "step": "final",
                "evidence_status": grade["status"],
                "action": action,
                "correction_count": metrics["correction_count"],
            }
        )
        metrics["correction_budget_utilization"] = (
            metrics["correction_count"] / metrics["correction_budget"]
            if metrics["correction_budget"]
            else 0.0
        )
        metrics["total_latency_seconds"] = time.perf_counter() - started

        return {
            "question": question,
            "pipeline": "BUDGET_CONSTRAINED_CRAG",
            "answer": answer,
            "abstained": action == "ABSTAIN",
            "retrieved_docs": evidence_docs,
            "local_evidence": local_docs,
            "web_evidence": web_docs,
            "evidence_grade": grade,
            "query_history": query_history,
            "trace": trace,
            "metrics": metrics,
        }

    def grade_evidence(
        self, question: str, documents: Sequence[Document], answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use the configured LLM to judge evidence, not vector similarity."""
        if not documents:
            return {
                "status": "INSUFFICIENT",
                "relevant": False,
                "sufficient": False,
                "contradictory": False,
                "relevance_score": 0.0,
                "answer_supported": False if answer is not None else None,
                "rationale": "No evidence was retrieved.",
            }
        evidence = self._format_documents(documents)
        answer_section = f"\nCandidate answer:\n{answer}" if answer else ""
        prompt = (
            "TASK: GRADE\n"
            "You are an evidence evaluator. Judge only the supplied evidence; do not use "
            "outside knowledge. Relevance means the evidence addresses the question. "
            "Sufficiency means it contains the facts needed for a reliable answer. "
            "Contradiction means supplied passages materially disagree. "
            "If evidence is empty, it is insufficient.\n"
            "Return only a JSON object with keys: relevant (boolean), sufficient "
            "(boolean), contradictory (boolean), relevance_score (number 0..1), "
            "answer_supported (boolean or null), rationale (short string).\n"
            "Set answer_supported only when a candidate answer is supplied; it must be "
            "false if any material claim is unsupported.\n"
            f"Question:\n{question}\n{answer_section}\nEvidence:\n{evidence}"
        )
        raw = self._invoke_text(prompt)
        try:
            payload = self._extract_json(raw)
            relevant = self._require_bool(payload, "relevant")
            sufficient = self._require_bool(payload, "sufficient")
            contradictory = self._require_bool(payload, "contradictory")
            answer_supported = payload.get("answer_supported")
            if answer_supported is not None and not isinstance(answer_supported, bool):
                raise ValueError("answer_supported must be a boolean or null.")
            if answer is not None and not isinstance(answer_supported, bool):
                raise ValueError("The grader must return answer_supported for a candidate answer.")
            score = float(payload["relevance_score"])
            if not 0 <= score <= 1:
                raise ValueError("relevance_score must be between 0 and 1.")
            rationale = payload["rationale"]
            if not isinstance(rationale, str):
                raise ValueError("rationale must be a string.")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Evidence grader returned invalid JSON: {raw}") from exc

        status = "CONTRADICTORY" if contradictory else (
            "SUFFICIENT" if sufficient and relevant else "INSUFFICIENT"
        )
        return {
            "status": status,
            "relevant": relevant,
            "sufficient": sufficient and relevant and not contradictory,
            "contradictory": contradictory,
            "relevance_score": score,
            "answer_supported": answer_supported,
            "rationale": rationale,
        }

    def grade_answer_support(self, question: str, answer: str, documents: Sequence[Document]) -> bool:
        """Return the grader's explicit answer-support judgment for evaluation."""
        result = self.grade_evidence(question, documents, answer)
        return bool(result["answer_supported"])

    def _rewrite_and_retrieve(
        self,
        question: str,
        query_history: List[str],
        local_docs: List[Document],
        metrics: Dict[str, Any],
        trace: List[Dict[str, Any]],
    ) -> None:
        if metrics["correction_count"] >= metrics["correction_budget"]:
            return
        metrics["correction_count"] += 1
        rewrite_prompt = (
            "TASK: REWRITE\n"
            "Rewrite the user's question as one concise search query that preserves its "
            "intent and adds useful terminology. Return only the query.\n"
            f"Question: {question}"
        )
        rewritten = self._invoke_text(rewrite_prompt).strip().strip('"')
        if not rewritten:
            raise ValueError("The query rewriter returned an empty query.")
        query_history.append(rewritten)
        docs = self._retrieve(rewritten, metrics)
        local_docs.extend(docs)
        trace.append(
            {
                "step": "correction_attempt_1",
                "action": "QUERY_REWRITE",
                "query": rewritten,
                "document_count": len(docs),
            }
        )

    def _retrieve(self, query: str, metrics: Dict[str, Any]) -> List[Document]:
        started = time.perf_counter()
        try:
            docs = self.retriever.invoke(query)
        finally:
            metrics["retrieval_latency_seconds"] += time.perf_counter() - started
            metrics["retrieval_attempts"] += 1
        return list(docs or [])

    def _generate_answer(self, question: str, documents: Sequence[Document]) -> str:
        prompt = (
            "TASK: ANSWER\n"
            "Answer the question using only the evidence below. Cite evidence inline using "
            "[n] markers corresponding to the evidence list. If the evidence does not "
            "support a claim, do not include it. Be concise.\n"
            f"Question: {question}\nEvidence:\n{self._format_documents(documents)}"
        )
        answer = self._invoke_text(prompt).strip()
        if not answer:
            raise ValueError("The answer generator returned an empty answer.")
        return answer

    def _invoke_text(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise json.JSONDecodeError("No JSON object found", raw, 0)
        payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict):
            raise TypeError("Evidence grader response must be a JSON object.")
        return payload

    @staticmethod
    def _require_bool(payload: Dict[str, Any], key: str) -> bool:
        value = payload[key]
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a boolean.")
        return value

    @staticmethod
    def _format_documents(documents: Sequence[Document]) -> str:
        if not documents:
            return "(no evidence retrieved)"
        return "\n\n".join(
            f"[{index}] Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Title: {doc.metadata.get('title', 'untitled')}\n"
            f"{doc.page_content[:1800]}"
            for index, doc in enumerate(documents, start=1)
        )

    @staticmethod
    def _deduplicate_documents(documents: Iterable[Document]) -> List[Document]:
        unique: List[Document] = []
        seen = set()
        for doc in documents:
            key = (
                repr(doc.metadata.get("source")),
                repr(doc.metadata.get("page")),
                doc.page_content,
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    @staticmethod
    def _grade_trace(step: str, grade: Dict[str, Any], document_count: int) -> Dict[str, Any]:
        return {
            "step": step,
            "evidence_status": grade["status"],
            "document_count": document_count,
            "relevance_score": grade["relevance_score"],
            "rationale": grade["rationale"],
            "action": "GENERATE" if grade["sufficient"] else "CORRECT",
        }


def load_evaluation_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the versioned JSON evaluation dataset."""
    with open(path, encoding="utf-8") as dataset_file:
        payload = json.load(dataset_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError("Evaluation dataset must contain a questions list.")
    return payload["questions"]


def token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-words token F1 for a reproducible reference-answer score."""
    normalize = lambda text: re.findall(r"\w+", text.lower())
    predicted_tokens = normalize(prediction)
    reference_tokens = normalize(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    remaining = list(reference_tokens)
    overlap = 0
    for token in predicted_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def evaluate_dataset(
    pipeline: CRAGPipeline, cases: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate the strict budget-constrained pipeline on the versioned dataset."""
    rows = []
    for case in cases:
        result = pipeline.run(case["question"])
        if result["abstained"]:
            answer_supported = None
            judge_latency = 0.0
        else:
            judge_started = time.perf_counter()
            answer_supported = pipeline.grade_answer_support(
                case["question"], result["answer"], result["retrieved_docs"]
            )
            judge_latency = time.perf_counter() - judge_started
        reference = case.get("expected_answer")
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": result["answer"],
                "answer_supported": answer_supported,
                "evidence_sufficiency": result["evidence_grade"]["sufficient"],
                "token_f1": token_f1(result["answer"], reference) if reference else None,
                "abstained": result["abstained"],
                "metrics": result["metrics"],
                "evaluation_judge_latency_seconds": judge_latency,
                "trace": result["trace"],
            }
        )
    return {"summary": _summarize(rows), "details": rows}


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    token_scores = [row["token_f1"] for row in rows if row["token_f1"] is not None]
    judged_answers = [row for row in rows if row["answer_supported"] is not None]
    corrected = [row for row in rows if row["metrics"]["correction_count"] > 0]
    budget = CRAGPipeline._MAX_CORRECTIONS * count
    return {
        "pipeline": "BUDGET_CONSTRAINED_CRAG",
        "total_questions": count,
        "answer_correctness_token_f1": mean(token_scores) if token_scores else None,
        "evidence_support_rate": (
            mean(float(row["answer_supported"]) for row in judged_answers)
            if judged_answers
            else None
        ),
        "retrieval_attempts_per_question": (
            mean(row["metrics"]["retrieval_attempts"] for row in rows) if rows else 0.0
        ),
        "web_search_calls": sum(row["metrics"]["web_search_calls"] for row in rows),
        "correction_budget_utilization": (
            sum(row["metrics"]["correction_count"] for row in rows) / budget if budget else 0.0
        ),
        "end_to_end_latency_seconds": (
            mean(row["metrics"]["total_latency_seconds"] for row in rows) if rows else 0.0
        ),
        "retrieval_latency_seconds": (
            mean(row["metrics"]["retrieval_latency_seconds"] for row in rows) if rows else 0.0
        ),
        "web_search_latency_seconds": (
            mean(row["metrics"]["web_search_latency_seconds"] for row in rows) if rows else 0.0
        ),
        "generation_latency_seconds": (
            mean(row["metrics"]["generation_latency_seconds"] for row in rows) if rows else 0.0
        ),
        "abstention_rate": (
            mean(float(row["abstained"]) for row in rows) if rows else 0.0
        ),
        "correction_success_rate": (
            mean(float(row["evidence_sufficiency"] is True) for row in corrected)
            if corrected
            else None
        ),
        "evaluation_judge_latency_seconds": (
            mean(row["evaluation_judge_latency_seconds"] for row in rows) if rows else 0.0
        ),
    }
