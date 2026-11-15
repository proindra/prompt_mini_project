"""Wikipedia search adapter shared by the RAG agent and research pipeline."""

import os
from typing import List

import requests
from langchain_core.documents import Document


def search_wikipedia(query: str) -> List[Document]:
    """Return Wikipedia extracts as cited documents."""
    endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": os.getenv("USER_AGENT", "BudgetCRAG/0.1")}
    search_response = requests.get(
        endpoint,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query[:300],
            "srlimit": 3,
            "format": "json",
        },
        headers=headers,
        timeout=15,
    )
    search_response.raise_for_status()
    search_results = search_response.json()["query"]["search"]
    if not search_results:
        return []

    page_response = requests.get(
        endpoint,
        params={
            "action": "query",
            "pageids": "|".join(str(result["pageid"]) for result in search_results),
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 1200,
            "format": "json",
        },
        headers=headers,
        timeout=15,
    )
    page_response.raise_for_status()
    pages = page_response.json()["query"]["pages"]

    documents = []
    for result in search_results:
        page = pages.get(str(result["pageid"]), {})
        extract = page.get("extract", "")
        if extract:
            page_id = result["pageid"]
            documents.append(
                Document(
                    page_content=extract,
                    metadata={
                        "title": result["title"],
                        "source": f"https://en.wikipedia.org/?curid={page_id}",
                        "page_id": page_id,
                        "source_type": "web",
                    },
                )
            )
    return documents
