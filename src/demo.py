"""
Streamlit demo for the ConvFinQA agent.
Run: streamlit run demo.py
"""

import logging

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from src.agent import Agent
from src.data import ConvFinQARecord, doc_to_markdown, get_record, get_records

load_dotenv()


@st.cache_data
def _load_records():
    return get_records("dev")


class StreamlitLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        st.write(f"`{record.name.split('.')[-1]}` — {record.getMessage()}")


def _extract_trace(state: dict, msg_start: int = 0) -> dict:
    turn_messages = list(state.get("messages", []))[msg_start:]
    tool_calls: list[dict] = []
    for msg in turn_messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []) or []:
                if expr := (tc.get("args") or {}).get("expression", ""):
                    tool_calls.append({"expr": expr, "result": None})
        elif (
            isinstance(msg, ToolMessage)
            and tool_calls
            and tool_calls[-1]["result"] is None
        ):
            tool_calls[-1]["result"] = msg.content
    reflection = state.get("reflection")
    return {
        "tool_calls": tool_calls,
        "reflection": dict(reflection) if reflection else None,
        "retry_count": state.get("retry_count", 0),
        "tool_call_count": state.get("tool_call_count", 0),
    }


def _render_trace(trace: dict) -> None:
    col_solver, col_reflect = st.columns(2)
    with col_solver:
        st.subheader("Solver")
        st.metric("Calculator calls", trace.get("tool_call_count", 0))
        for tc in trace.get("tool_calls") or []:
            st.code(
                f"Expression: {tc['expr']}\nResult:     {tc['result']}", language=None
            )
    with col_reflect:
        st.subheader("Reflector")
        st.metric("Retries", max(0, trace.get("retry_count", 0) - 1))
        if reflection := trace.get("reflection"):
            is_correct = reflection.get("is_correct")
            st.markdown("✓ Correct" if is_correct else "✗ Incorrect")
            st.markdown("**Critique**")
            st.markdown(reflection.get("critique"))


st.set_page_config(page_title="ConvFinQA Demo", layout="wide")
st.title("Conversational Financial Q&A Agent Demo")
st.subheader("Suncorp Case Interview - Joy Zhao")
st.divider()

dev_records = _load_records()
record_ids = [r.id for r in dev_records]

if "agent" not in st.session_state:
    from src.agent.settings import load_configs

    st.session_state.agent = Agent(*load_configs())
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "record" not in st.session_state:
    st.session_state.record = None
if "doc_markdown" not in st.session_state:
    st.session_state.doc_markdown = None

sel_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
with sel_col:
    current_idx = (
        record_ids.index(st.session_state.record.id) if st.session_state.record else 0
    )
    chosen_id = st.selectbox(
        "Record", record_ids, index=current_idx, label_visibility="collapsed"
    )
with btn_col:
    load_clicked = st.button("Load", type="primary", use_container_width=True)

if load_clicked or st.session_state.record is None:
    record = get_record(chosen_id)
    if record is None:
        st.error(f"Record not found: {chosen_id}")
        st.stop()
    doc_md = doc_to_markdown(record)
    st.session_state.record = record
    st.session_state.doc_markdown = doc_md
    st.session_state.agent_state = st.session_state.agent.initialize_chat(doc_md)
    st.session_state.chat_history = []
    if load_clicked:
        st.rerun()

record: ConvFinQARecord = st.session_state.record
col_doc, col_chat, col_qa = st.columns([3, 4, 2], gap="medium")

with col_doc:
    st.subheader("Document")
    with st.container(height=600, border=True):
        st.markdown(st.session_state.doc_markdown)

with col_chat:
    st.subheader("Chat")

    messages_container = st.container(height=600, border=True)

    with messages_container:
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if entry["role"] == "assistant" and entry.get("trace"):
                    with st.expander("Agent trace", expanded=False):
                        _render_trace(entry["trace"])

    if prompt := st.chat_input("Ask a question about this document…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with messages_container:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                nodes_logger = logging.getLogger("src.agent.nodes")
                handler = StreamlitLogHandler()
                handler.setLevel(logging.INFO)
                nodes_logger.addHandler(handler)

                msg_start = len(st.session_state.agent_state["messages"])

                with st.status("Agent thinking…", expanded=False) as status:
                    try:
                        new_state, answer = st.session_state.agent.chat_turn(
                            st.session_state.agent_state, prompt
                        )
                    finally:
                        nodes_logger.removeHandler(handler)
                    status.update(label="Done!", state="complete")

                trace = _extract_trace(new_state, msg_start)

        st.session_state.agent_state = new_state
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"**{answer}**", "trace": trace}
        )
        st.rerun()

with col_qa:
    st.subheader("Ground Truth Reference")
    with st.container(height=600, border=True):
        questions = record.dialogue.conv_questions
        answers = record.dialogue.executed_answers
        for i, (q, a) in enumerate(zip(questions, answers, strict=False), start=1):
            st.markdown(f"**Turn {i}**")
            st.code(q, language=None)
            st.markdown(f"Answer: `{a}`")
            if i < len(questions):
                st.divider()
