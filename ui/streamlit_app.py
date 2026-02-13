"""Streamlit UI for Graph-Memory-R1.

4 tabs: Chat, Graph Explorer, Training, Benchmark.
Port: 8505
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from ui.i18n import get_translator

st.set_page_config(page_title="Graph-Memory-R1", page_icon="🧠", layout="wide")


# ── Sidebar ──────────────────────────────────────────────────

lang = st.sidebar.selectbox("Language / Язык", ["en", "ru"], index=0)
st.session_state["lang"] = lang
t = get_translator(lang)

st.sidebar.title(t("app_title"))
st.sidebar.caption(t("app_subtitle"))

st.title(t("app_title"))
st.caption(t("app_subtitle"))


# ── Cached Resources ─────────────────────────────────────────

@st.cache_resource
def get_graph_memory():
    from memory.graph_memory import GraphMemory
    from openai import OpenAI
    from config import get_settings

    settings = get_settings()
    client = OpenAI(api_key=settings.openai.api_key) if settings.openai.api_key else None

    def embedder(text: str) -> list[float]:
        if client is None:
            return []
        resp = client.embeddings.create(
            model=settings.openai.embedding_model,
            input=text,
            dimensions=settings.openai.embedding_dimensions,
        )
        return resp.data[0].embedding

    embed_fn = embedder if client else None
    graph = GraphMemory(embedder=embed_fn)
    graph.init_schema()
    return graph


@st.cache_resource
def get_memory_manager():
    from agents.memory_manager import MemoryManager
    graph = get_graph_memory()
    return MemoryManager(graph=graph)


@st.cache_resource
def get_answer_agent():
    from agents.answer_agent import AnswerAgent
    graph = get_graph_memory()
    return AnswerAgent(graph=graph)


# ── Tabs ─────────────────────────────────────────────────────

tab_chat, tab_graph, tab_training, tab_benchmark = st.tabs([
    t("tab_chat"), t("tab_graph"), t("tab_training"), t("tab_benchmark"),
])


# ── Tab 1: Chat ──────────────────────────────────────────────

with tab_chat:
    st.header(t("chat_header"))

    # Process conversation
    conv_text = st.text_area(
        t("chat_input"),
        placeholder=t("chat_placeholder"),
        height=200,
    )

    if st.button(t("chat_process"), disabled=not conv_text.strip()):
        with st.spinner(t("chat_processing")):
            mm = get_memory_manager()
            result = mm.process_chunk(conv_text)

        st.subheader(t("chat_operations"))
        st.info(t("chat_op_count", total=result.total, ok=result.successful, fail=result.failed))

        for op in result.operations:
            icon = "✅" if op.success else "❌"
            st.write(f"{icon} **{op.action}** ({op.memory_type}): {op.result}")

    st.divider()

    # Ask question
    question = st.text_input(
        t("chat_question"),
        placeholder=t("chat_question_placeholder"),
    )

    if st.button(t("chat_ask"), disabled=not question.strip()):
        with st.spinner(t("chat_asking")):
            agent = get_answer_agent()
            result = agent.answer(question)

        st.subheader(t("chat_answer"))
        st.write(result["answer"])

        with st.expander(t("chat_sources", count=result["num_sources"])):
            for src in result.get("sources", []):
                st.write(f"- [{src.get('id', '?')}] {src.get('content', '')}")


# ── Tab 2: Graph Explorer ────────────────────────────────────

with tab_graph:
    st.header(t("graph_header"))

    graph = get_graph_memory()
    stats = graph.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(t("graph_nodes", count=""), stats["total_nodes"])
    with col2:
        st.metric(t("graph_edges", count=""), stats["total_edges"])

    # Node counts by type
    st.subheader(t("graph_stats"))
    for label, count in stats.get("nodes", {}).items():
        st.write(f"- **{label}**: {count}")

    # Edge types
    for edge_type, count in stats.get("edges", {}).items():
        st.write(f"- **{edge_type}**: {count}")

    # Core memory
    core = graph.get_node("core", "core")
    if core and core.get("content"):
        st.subheader(t("graph_core"))
        st.write(core["content"])

    # Rendered memory
    st.subheader("Memory Snapshot")
    rendered = graph.render_memory()
    if rendered.strip():
        st.text(rendered)
    else:
        st.info(t("no_data"))

    # Clear
    st.divider()
    confirm = st.text_input(t("graph_clear_confirm"), key="clear_confirm")
    if st.button(t("graph_clear"), disabled=confirm != "DELETE"):
        graph.clear()
        st.success(t("graph_cleared"))
        st.cache_resource.clear()
        st.rerun()


# ── Tab 3: Training ──────────────────────────────────────────

with tab_training:
    st.header(t("train_header"))
    st.info(t("train_not_started"))

    st.subheader(t("train_header"))

    dataset_path = st.text_input(
        t("train_dataset"),
        value="data/locomo/locomo10.json",
    )
    epochs = st.number_input(t("train_epochs"), min_value=1, max_value=50, value=3)
    lr = st.number_input(t("train_lr"), min_value=1e-7, max_value=1e-3, value=1e-5, format="%.1e")

    if st.button(t("train_start")):
        st.warning(t("train_running"))
        st.info("Training requires GPU and LoCoMo dataset. Run `python training/train.py` from terminal.")

    # Load LoRA
    st.divider()
    st.subheader(t("train_load_lora"))
    lora_path = st.text_input(t("train_lora_path"), value="checkpoints/final_lora")
    if st.button(t("train_load_lora"), key="load_lora_btn"):
        if os.path.exists(lora_path):
            mm = get_memory_manager()
            mm.load_lora(lora_path)
            st.success(t("train_lora_loaded"))
        else:
            st.error(t("error", msg=f"Path not found: {lora_path}"))


# ── Tab 4: Benchmark ─────────────────────────────────────────

with tab_benchmark:
    st.header(t("bench_header"))

    bench_dataset = st.text_input(
        t("bench_dataset"),
        value="data/locomo/locomo10.json",
        key="bench_ds",
    )
    max_items = st.number_input(t("bench_max_items"), min_value=1, max_value=1000, value=20)
    use_judge = st.checkbox(t("bench_use_judge"), value=False)

    if st.button(t("bench_run")):
        if not os.path.exists(bench_dataset):
            st.error(t("error", msg=f"Dataset not found: {bench_dataset}"))
        else:
            from training.dataset import load_locomo
            from evaluation.benchmark import run_benchmark
            from evaluation.llm_judge import judge_answer

            data = load_locomo(bench_dataset)
            qa_pairs = data["qa_pairs"][:max_items]

            agent = get_answer_agent()
            answer_fn = lambda q: agent.answer(q)["answer"]
            judge_fn = judge_answer if use_judge else None

            progress = st.progress(0)
            results_list = []

            for i, qa in enumerate(qa_pairs):
                predicted = answer_fn(qa.question)
                from evaluation.metrics import compute_all_metrics
                metrics = compute_all_metrics(predicted, qa.answer)

                result = {
                    "Q#": i + 1,
                    "Category": qa.category,
                    "F1": f"{metrics['f1']:.3f}",
                    "Question": qa.question[:80],
                }
                results_list.append(result)
                progress.progress((i + 1) / len(qa_pairs))

            st.subheader(t("bench_results"))

            avg_f1 = sum(float(r["F1"]) for r in results_list) / len(results_list) if results_list else 0
            st.metric("Average F1", f"{avg_f1:.3f}")

            import pandas as pd
            st.dataframe(pd.DataFrame(results_list), use_container_width=True)
