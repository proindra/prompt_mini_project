"""Streamlit interface for Budget-Constrained Self-Corrective RAG."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.research.crag import (
    CRAGPipeline,
    evaluate_dataset,
    load_evaluation_dataset,
)
from src.vectorstore.vectorstore import VectorStore


st.set_page_config(
    page_title="Budget-Constrained Self-Corrective RAG",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    if "crag_pipeline" not in st.session_state:
        st.session_state.crag_pipeline = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "latest_research_result" not in st.session_state:
        st.session_state.latest_research_result = None
    if "research_history" not in st.session_state:
        st.session_state.research_history = []


@st.cache_resource
def initialize_crag():
    """Initialize the CRAG pipeline and shared vector index once per process."""
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        vector_store = VectorStore()
        documents = doc_processor.process_urls(Config.DEFAULT_URLS)
        vector_store.create_vectorstore(documents)

        retriever = vector_store.get_retriever()
        return CRAGPipeline(retriever, llm), len(documents)
    except Exception as exc:
        st.error(f"Failed to initialize: {exc}")
        return None, 0


def show_documents(documents, citation_documents=None) -> None:
    if not documents:
        st.info("No evidence was retrieved.")
        return
    citation_documents = citation_documents or documents
    citation_numbers = {
        (
            repr(document.metadata.get("source")),
            repr(document.metadata.get("page")),
            document.page_content,
        ): index
        for index, document in enumerate(citation_documents, start=1)
    }
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        title = document.metadata.get("title", f"Evidence {index}")
        key = (
            repr(document.metadata.get("source")),
            repr(document.metadata.get("page")),
            document.page_content,
        )
        citation = citation_numbers.get(key, index)
        with st.expander(f"[{citation}] {title} — {source}"):
            st.write(document.page_content)


def render_crag_dashboard() -> None:
    st.subheader("Budget-Constrained CRAG")
    st.caption(
        "Every question follows the same strict path: initial local retrieval and grading, "
        "one rewritten local retry, then one web-search fallback if needed."
    )
    with st.form("research_form"):
        question = st.text_input(
            "User question",
            placeholder="Ask a question about the indexed documents or a general topic",
        )
        submitted = st.form_submit_button("Submit question")

    if submitted and question.strip():
        with st.spinner("Running the two-attempt CRAG pipeline..."):
            result = st.session_state.crag_pipeline.run(question)
        st.session_state.latest_research_result = result
        st.session_state.research_history.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "abstained": result["abstained"],
                "evidence_grade": result["evidence_grade"],
                "trace": result["trace"],
                "metrics": result["metrics"].copy(),
            }
        )

    result = st.session_state.get("latest_research_result")
    if not result:
        return
    pipeline_name = result.get("pipeline")
    if pipeline_name is None:
        st.warning(
            "This saved result is from an older pipeline version. Restart the Streamlit "
            "server and submit the question again."
        )
        return

    st.markdown("### Answer")
    if result["abstained"]:
        st.warning(result["answer"])
    else:
        st.success(result["answer"])
    st.caption(
        f"Pipeline: {pipeline_name} · "
        f"Abstained: {'Yes' if result['abstained'] else 'No'}"
    )

    grade = result["evidence_grade"]
    st.markdown("**Final evidence grading:** " + grade["status"])
    st.caption(grade["rationale"])

    st.markdown("### Evidence")
    evidence_columns = st.columns(2)
    with evidence_columns[0]:
        st.markdown("**Local evidence**")
        show_documents(result["local_evidence"], result["retrieved_docs"])
    with evidence_columns[1]:
        st.markdown("**External web evidence**")
        show_documents(result["web_evidence"], result["retrieved_docs"])

    st.markdown("### Explainability trace")
    st.json(
        {
            "query_rewrites": result["query_history"],
            "steps": result["trace"],
            "metrics": result["metrics"],
        }
    )

    render_evaluation_dashboard(st.session_state.research_history)


def render_evaluation_dashboard(history) -> None:
    st.markdown("### Evaluation Dashboard")
    total = len(history)
    if not total:
        st.info("Submit a question to populate the dashboard with actual pipeline results.")
        return

    answered = [run for run in history if not run["abstained"]]
    corrected = [run for run in history if run["metrics"]["correction_count"] > 0]
    correction_successes = [
        run for run in corrected if run["evidence_grade"]["sufficient"]
    ]
    columns = st.columns(5)
    columns[0].metric("Questions", total)
    columns[1].metric(
        "Evidence-sufficient answers",
        f"{len(answered)} ({len(answered) / total:.0%})",
    )
    columns[2].metric(
        "Mean latency",
        f"{sum(run['metrics']['total_latency_seconds'] for run in history) / total:.2f}s",
    )
    columns[3].metric(
        "Local retrievals",
        sum(run["metrics"]["retrieval_attempts"] for run in history),
    )
    columns[4].metric(
        "Web searches",
        sum(run["metrics"]["web_search_calls"] for run in history),
    )

    correction_rate = (
        len(correction_successes) / len(corrected) if corrected else None
    )
    st.metric(
        "Correction success rate",
        f"{correction_rate:.0%}" if correction_rate is not None else "N/A",
        help="Among questions that used at least one correction, the share ending with sufficient evidence.",
    )
    st.dataframe(
        [
            {
                "question": run["question"],
                "answer": run["answer"],
                "trace": " | ".join(
                    f"{step['step']}: "
                    f"{step.get('evidence_status', step.get('action', 'unknown'))}"
                    for step in run["trace"]
                ),
                "final_evidence": run["evidence_grade"]["status"],
                "action": "ABSTAIN" if run["abstained"] else "GENERATE",
                "corrections": (
                    f"{run['metrics']['correction_count']} / "
                    f"{run['metrics']['correction_budget']}"
                ),
                "local_retrievals": run["metrics"]["retrieval_attempts"],
                "web_searches": run["metrics"]["web_search_calls"],
                "latency_seconds": run["metrics"]["total_latency_seconds"],
            }
            for run in history
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "These session metrics summarize actual questions processed by the strict pipeline. "
        "Answer correctness is measured separately in the reference-dataset evaluation."
    )


def render_dataset_evaluation() -> None:
    dataset_path = Path(__file__).parent / "data" / "evaluation_dataset.json"
    cases = load_evaluation_dataset(str(dataset_path))
    st.caption(
        f"Dataset: {len(cases)} questions. Runs the strict two-correction pipeline and "
        "an answer-support judge; this makes live LLM calls."
    )
    st.dataframe(
        [{"id": case["id"], "category": case["category"], "question": case["question"]} for case in cases],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Evaluate strict CRAG pipeline", key="run_evaluation"):
        with st.spinner("Evaluating the dataset; this may take several minutes..."):
            st.session_state.evaluation_results = evaluate_dataset(
                st.session_state.crag_pipeline, cases
            )

    results = st.session_state.evaluation_results
    if not results:
        return

    summary = results["summary"]
    st.markdown("### Measured pipeline results")
    st.dataframe([summary], use_container_width=True, hide_index=True)

    import pandas as pd

    summary_frame = pd.DataFrame([summary]).set_index("pipeline")
    st.markdown("### Latency and answer quality")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.bar_chart(
            summary_frame[
                [
                    "end_to_end_latency_seconds",
                    "retrieval_latency_seconds",
                    "web_search_latency_seconds",
                    "generation_latency_seconds",
                ]
            ]
        )
    with chart_columns[1]:
        st.bar_chart(
            summary_frame[
                ["answer_correctness_token_f1", "evidence_support_rate", "correction_success_rate"]
            ]
        )
    st.markdown("### Search and correction use")
    st.bar_chart(
        summary_frame[
            ["web_search_calls", "correction_budget_utilization", "abstention_rate"]
        ]
    )

    st.caption(
        "Answer correctness is token F1 against dataset references. Evidence support is the "
        "LLM judge's support decision among generated answers. Evaluation-judge latency is "
        "reported separately from pipeline latency. Results are generated live and are not "
        "pre-populated."
    )
    export = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        "Download evaluation results (JSON)",
        data=export,
        file_name="budget-constrained-self-corrective-rag-results.json",
        mime="application/json",
    )

    with st.expander("Per-question results and traces"):
        st.dataframe(results["details"], use_container_width=True, hide_index=True)


def main() -> None:
    init_session_state()
    st.title("Budget-Constrained Self-Corrective RAG")
    st.markdown(
        "Document-grounded answers with budget-constrained self-corrective retrieval."
    )

    if not st.session_state.initialized:
        with st.spinner("Loading documents and models..."):
            crag_pipeline, chunk_count = initialize_crag()
        if crag_pipeline:
            st.session_state.crag_pipeline = crag_pipeline
            st.session_state.initialized = True
            st.success(f"System ready — {chunk_count} document chunks indexed.")
        else:
            return

    render_crag_dashboard()
    with st.expander("Run versioned dataset evaluation"):
        render_dataset_evaluation()


if __name__ == "__main__":
    main()
