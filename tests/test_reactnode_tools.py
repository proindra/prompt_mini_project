import unittest

from langchain_core.documents import Document

from src.node.reactnode import RAGNodes


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return [Document(page_content="retrieved evidence")]


class ReActToolSchemaTests(unittest.TestCase):
    def test_tools_expose_query_argument_expected_by_groq(self):
        retriever = FakeRetriever()
        tools = RAGNodes(retriever, llm=None)._build_tools()

        self.assertEqual([tool.name for tool in tools], ["retriever", "wikipedia"])
        for tool in tools:
            self.assertEqual(set(tool.args), {"query"})

    def test_retriever_tool_receives_structured_query(self):
        retriever = FakeRetriever()
        tools = RAGNodes(retriever, llm=None)._build_tools()

        result = tools[0].invoke({"query": "transformer"})

        self.assertEqual(retriever.queries, ["transformer"])
        self.assertIn("retrieved evidence", result)


if __name__ == "__main__":
    unittest.main()
