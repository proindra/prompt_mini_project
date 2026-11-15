import json
import unittest

from langchain_core.documents import Document

from src.research.crag import CRAGPipeline, Policy, evaluate_policies, token_f1


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
    def test_rate_limited_policy_stops_after_two_corrections(self):
        retriever = FakeRetriever()
        web_calls = []
        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, False, True]),
            web_search=lambda _query: (
                web_calls.append(_query) or [Document(page_content="web evidence")]
            ),
        )

        result = pipeline.run("question", Policy.RATE_LIMITED_CRAG)

        self.assertEqual(result["metrics"]["correction_count"], 2)
        self.assertEqual(result["metrics"]["retrieval_attempts"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 1)
        self.assertEqual(len(web_calls), 1)
        self.assertEqual(result["metrics"]["correction_budget_utilization"], 1)
        self.assertFalse(result["abstained"])

    def test_standard_policy_uses_one_correction_and_never_searches_web(self):
        retriever = FakeRetriever()
        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, False]),
            web_search=lambda query: self.fail(
                f"STANDARD_CRAG must not search the web: {query}"
            ),
        )

        result = pipeline.run("question", Policy.STANDARD_CRAG)

        self.assertEqual(result["metrics"]["correction_count"], 1)
        self.assertEqual(result["metrics"]["retrieval_attempts"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 0)
        self.assertTrue(result["abstained"])

    def test_rate_limited_policy_abstains_when_web_search_fails(self):
        retriever = FakeRetriever()

        def fail_search(_query):
            import requests

            raise requests.Timeout(f"search timed out for {_query}")

        pipeline = CRAGPipeline(
            retriever,
            FakeLLM([False, False]),
            web_search=fail_search,
        )

        result = pipeline.run("question", Policy.RATE_LIMITED_CRAG)

        self.assertTrue(result["abstained"])
        self.assertEqual(result["metrics"]["correction_count"], 2)
        self.assertEqual(result["metrics"]["web_search_calls"], 1)
        self.assertEqual(result["trace"][-2]["evidence_status"], "UNAVAILABLE")

    def test_empty_baseline_retrieval_abstains_without_generation(self):
        retriever = FakeRetriever([])
        pipeline = CRAGPipeline(retriever, FakeLLM())

        result = pipeline.run("question", Policy.BASELINE_RAG)

        self.assertTrue(result["abstained"])
        self.assertEqual(result["metrics"]["retrieval_attempts"], 1)
        self.assertEqual(result["metrics"]["generation_latency_seconds"], 0)

    def test_empty_evidence_cannot_be_graded_as_sufficient(self):
        pipeline = CRAGPipeline(
            FakeRetriever([]),
            FakeLLM(),
            web_search=lambda query: (
                [] if query == "rewritten search query" else self.fail(query)
            ),
        )

        result = pipeline.run("question", Policy.RATE_LIMITED_CRAG)

        self.assertTrue(result["abstained"])
        self.assertEqual(result["evidence_grade"]["status"], "INSUFFICIENT")
        self.assertEqual(result["metrics"]["correction_count"], 2)

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

        first = pipeline.run("first", Policy.RATE_LIMITED_CRAG)
        second = pipeline.run("second", Policy.RATE_LIMITED_CRAG)

        self.assertEqual(first["metrics"]["correction_count"], 1)
        self.assertEqual(second["metrics"]["correction_count"], 1)

    def test_token_f1_rewards_partial_reference_match(self):
        self.assertGreater(token_f1("Transformer uses attention", "The Transformer uses attention only"), 0)
        self.assertEqual(token_f1("", "reference"), 0)

    def test_evaluation_records_all_policies_and_support_metrics(self):
        pipeline = CRAGPipeline(
            FakeRetriever(),
            FakeLLM([True, True, True, True, True]),
        )
        cases = [
            {
                "id": "sample",
                "category": "local_answerable",
                "question": "question",
                "expected_answer": "Supported answer",
            }
        ]

        results = evaluate_policies(pipeline, cases)

        self.assertEqual(set(results), {policy.value for policy in Policy})
        for policy_result in results.values():
            self.assertEqual(policy_result["summary"]["total_questions"], 1)
            self.assertEqual(policy_result["summary"]["evidence_support_rate"], 1)
            self.assertEqual(len(policy_result["details"]), 1)


if __name__ == "__main__":
    unittest.main()
