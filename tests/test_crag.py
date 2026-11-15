import json
import unittest

from langchain_core.documents import Document

from src.research.crag import CRAGPipeline, evaluate_dataset, token_f1


def grade(sufficient, *, contradictory=False, relevant=True, answer_supported=True):
    status = "SUFFICIENT" if sufficient else "INSUFFICIENT"
    return json.dumps(
        {
            "relevant": relevant,
            "sufficient": sufficient,
            "contradictory": contradictory,
            "relevance_score": 0.9 if relevant else 0.1,
            "answer_supported": answer_supported,
            "rationale": status,
        }
    )


class FakeLLM:
    def __init__(self, grade_results=()):
        self.grade_results = iter(grade_results)

    def invoke(self, prompt):
        if "TASK: GRADE" in prompt:
            result = next(self.grade_results)
            return result if isinstance(result, str) else grade(result)
        if "TASK: REWRITE" in prompt:
            return "rewritten search query"
        if "TASK: ANSWER" in prompt:
            return "Supported answer [1]."
        raise AssertionError(f"Unexpected prompt: {prompt}")


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = (
            documents if documents is not None else [Document(page_content="evidence")]
        )
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class CRAGPipelineTests(unittest.TestCase):
    def test_pipeline_stops_after_two_corrections(self):
        retriever = FakeRetriever()
        web_calls = []
        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, False, True]),
            web_search=lambda _query: (
                web_calls.append(_query) or [Document(page_content="web evidence")]
            ),
        )

        result = pipeline.run("question")

        self.assertEqual(result["metrics"]["correction_count"], 2)
        self.assertEqual(result["metrics"]["retrieval_attempts"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 1)
        self.assertEqual(len(web_calls), 1)
        self.assertEqual(result["metrics"]["correction_budget_utilization"], 1)
        self.assertFalse(result["abstained"])

    def test_pipeline_abstains_when_web_search_fails(self):
        retriever = FakeRetriever()

        def fail_search(_query):
            import requests

            raise requests.Timeout(f"search timed out for {_query}")

        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, False]),
            web_search=fail_search,
        )

        result = pipeline.run("question")

        self.assertTrue(result["abstained"])
        self.assertEqual(result["metrics"]["correction_count"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 1)
        self.assertEqual(result["trace"][-2]["evidence_status"], "UNAVAILABLE")

    def test_empty_evidence_runs_both_attempts_then_abstains(self):
        pipeline = CRAGPipeline(
            FakeRetriever([]),
            FakeLLM(),
            web_search=lambda query: (
                [] if query == "rewritten search query" else self.fail(query)
            ),
        )

        result = pipeline.run("question")

        self.assertTrue(result["abstained"])
        self.assertEqual(result["evidence_grade"]["status"], "INSUFFICIENT")
        self.assertEqual(result["metrics"]["correction_count"], 2)
        self.assertEqual(result["metrics"]["retrieval_attempts"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 1)
        self.assertEqual(result["metrics"]["generation_latency_seconds"], 0)

    def test_conflicting_evidence_is_never_sufficient(self):
        pipeline = CRAGPipeline(
            FakeRetriever(),
            FakeLLM([grade(False, contradictory=True)]),
        )

        result = pipeline.grade_evidence("question", [Document(page_content="evidence")])

        self.assertEqual(result["status"], "CONTRADICTORY")
        self.assertFalse(result["sufficient"])

    def test_each_question_gets_an_independent_correction_budget(self):
        retriever = FakeRetriever()
        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, True, False, True]),
            web_search=lambda query: self.fail(f"No web search expected: {query}"),
        )

        first = pipeline.run("first")
        second = pipeline.run("second")

        self.assertEqual(first["metrics"]["correction_count"], 1)
        self.assertEqual(second["metrics"]["correction_count"], 1)

    def test_token_f1_rewards_partial_reference_match(self):
        self.assertGreater(token_f1("Transformer uses attention", "The Transformer uses attention only"), 0)
        self.assertEqual(token_f1("", "reference"), 0)

    def test_evaluation_records_strict_pipeline_metrics(self):
        pipeline = CRAGPipeline(
            FakeRetriever(),
            FakeLLM([True, True]),
        )
        cases = [
            {
                "id": "sample",
                "category": "local_answerable",
                "question": "question",
                "expected_answer": "Supported answer",
            }
        ]

        results = evaluate_dataset(pipeline, cases)

        self.assertEqual(results["summary"]["pipeline"], "BUDGET_CONSTRAINED_CRAG")
        self.assertEqual(results["summary"]["total_questions"], 1)
        self.assertEqual(results["summary"]["evidence_support_rate"], 1)
        self.assertEqual(len(results["details"]), 1)


if __name__ == "__main__":
    unittest.main()
