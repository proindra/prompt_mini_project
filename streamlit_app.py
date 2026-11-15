"""Streamlit interface for the existing RAG app and CRAG research prototype."""

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder
from src.research.crag import (
    CRAGPipeline,
    Policy,
    evaluate_policies,
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
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "research_pipeline" not in st.session_state:
        st.session_state.research_pipeline = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None


@st.cache_resource
def initialize_rag():
    """Initialize the existing RAG components once per Streamlit process."""
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
        graph_builder = GraphBuilder(retriever=retriever, llm=llm)
        graph_builder.build()
        return graph_builder, len(documents), CRAGPipeline(retriever, llm)
    except Exception as exc:
        st.error(f"Failed to initialize: {exc}")
        return None, 0, None


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


def render_chat() -> None:
    st.subheader("Document Search")
    st.caption("Ask the existing RAG agent about the indexed documents.")
    with st.form("search_form"):
        question = st.text_input("Enter your question:", placeholder="What would you like to know?")
        submitted = st.form_submit_button("Search")

    if submitted and question.strip():
        with st.spinner("Searching..."):
            started = time.perf_counter()
            result = st.session_state.rag_system.run(question)
            elapsed = time.perf_counter() - started
        st.session_state.history.append(
            {"question": question, "answer": result["answer"], "time": elapsed}
        )
        st.markdown("### Answer")
        st.success(result["answer"])
        with st.expander("Source documents"):
            show_documents(result["retrieved_docs"])
        st.caption(f"Response time: {elapsed:.2f} seconds")

    if st.session_state.history:
        st.markdown("### Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}")
            st.caption(f"Time: {item['time']:.2f}s")


def render_research_dashboard() -> None:
    st.subheader("Budget-Constrained CRAG")
    st.caption(
        "Each submitted question has an independent correction counter. "
        "The rate-limited policy allows one local query rewrite and one web search."
    )
    policy = st.selectbox("Evaluation policy", [item.value for item in Policy], key="research_policy")
    with st.form("research_form"):
        question = st.text_input(
            "Research question",
            placeholder="Ask a question about the indexed documents or a general topic",
        )
        submitted = st.form_submit_button("Run policy")

    if submitted and question.strip():
        with st.spinner(f"Running {policy}..."):
            result = st.session_state.research_pipeline.run(question, Policy(policy))
        st.session_state.latest_research_result = result

    result = st.session_state.get("latest_research_result")
    if not result:
        return

    st.markdown("### Answer")
    if result["abstained"]:
        st.warning(result["answer"])
    else:
        st.success(result["answer"])
    st.caption(f"Policy: {result['policy']} · Abstained: {'Yes' if result['abstained'] else 'No'}")

    metrics = result["metrics"]
    columns = st.columns(5)
    columns[0].metric("Corrections", f"{metrics['correction_count']} / {metrics['correction_budget']}")
    columns[1].metric("Local retrievals", metrics["retrieval_attempts"])
    columns[2].metric("Web searches", metrics["web_search_calls"])
    columns[3].metric("Latency", f"{metrics['total_latency_seconds']:.2f}s")
    columns[4].metric("Budget used", f"{metrics['correction_budget_utilization']:.0%}")

    grade = result["evidence_grade"]
    if grade:
        st.markdown("**Final evidence grading:** " + grade["status"])
        st.caption(grade["rationale"])
    else:
        st.markdown("**Final evidence grading:** Not evaluated by BASELINE_RAG")

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
            "metrics": metrics,
        }
    )


def render_evaluation() -> None:
    st.subheader("Comparative Evaluation")
    dataset_path = Path(__file__).parent / "data" / "evaluation_dataset.json"
    cases = load_evaluation_dataset(str(dataset_path))
    st.caption(
        f"Dataset: {len(cases)} questions. Runs all three policies and an additional "
        "answer-support judge for each result; this makes live LLM calls."
    )
    st.dataframe(
        [{"id": case["id"], "category": case["category"], "question": case["question"]} for case in cases],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Run comparative evaluation", key="run_evaluation"):
        with st.spinner("Evaluating all policies; this may take several minutes..."):
            st.session_state.evaluation_results = evaluate_policies(
                st.session_state.research_pipeline, cases
            )

    results = st.session_state.evaluation_results
    if not results:
        return

    summaries = [value["summary"] for value in results.values()]
    st.markdown("### Measured policy results")
    st.dataframe(summaries, use_container_width=True, hide_index=True)

    import pandas as pd

    summary_frame = pd.DataFrame(summaries).set_index("policy")
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
        for policy_name, value in results.items():
            st.markdown(f"**{policy_name}**")
            st.dataframe(value["details"], use_container_width=True, hide_index=True)


def main() -> None:
    init_session_state()
    st.title("Budget-Constrained Self-Corrective RAG")
    st.markdown(
        "Document-grounded answers with failure-aware, rate-limited corrective retrieval."
    )

    if not st.session_state.initialized:
        with st.spinner("Loading documents and models..."):
            rag_system, chunk_count, research_pipeline = initialize_rag()
        if rag_system and research_pipeline:
            st.session_state.rag_system = rag_system
            st.session_state.research_pipeline = research_pipeline
            st.session_state.initialized = True
            st.success(f"System ready — {chunk_count} document chunks indexed.")
        else:
            return

    chat_tab, research_tab, evaluation_tab = st.tabs(
        ["Document Search", "CRAG Research", "Evaluation"]
    )
    with chat_tab:
        render_chat()
    with research_tab:
        render_research_dashboard()
    with evaluation_tab:
        render_evaluation()


if __name__ == "__main__":
    main()
