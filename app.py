# app.py
# Verqi — Your AI-powered study assistant
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from document_processor import create_vector_store
from rag_engine import prepare_answer, friendly_error
from study_tools import (
    generate_summary, suggest_questions, generate_quiz,
    generate_flashcards, extract_key_concepts,
)
from math_solver import solve_text, solve_image

# --- Setup ---
load_dotenv()


def get_api_key():
    """Read the key from a local .env, or from Streamlit Cloud secrets."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return None


API_KEY = get_api_key()

ICON_PATH = str(Path(__file__).parent / "assets" / "verqi_icon.png")

st.set_page_config(
    page_title="Verqi — Your AI study assistant",
    page_icon=ICON_PATH,
    layout="wide",
)


def show_error(context: str, error: Exception) -> None:
    """Log the full error for the developer, show a friendly message to the user."""
    print(f"[Verqi] {context}: {error}")
    st.error(friendly_error(error))


# --- Session state ---
for key, default in [
    ("vector_store", None), ("messages", []), ("processed_files", []),
    ("doc_text", ""), ("summary", None), ("questions", None),
    ("quiz", None), ("flashcards", None), ("concepts", None),
    ("math_solution", None), ("last_math_mode", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Brand header ---
BRAND = """
<div style="text-align:center; padding:0.5rem 0 0.25rem;">
  <div style="display:inline-flex; align-items:center; gap:14px;">
    <svg width="50" height="44" viewBox="0 0 50 44" xmlns="http://www.w3.org/2000/svg">
      <polyline points="14,14 28,36 42,14" fill="none" stroke="#4F45A0"
                stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="8,8 22,30 36,8" fill="none" stroke="#8F85E8"
                stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span style="font-size:46px; font-weight:600; letter-spacing:-0.03em;
                 color:#E9E7F2; line-height:1;">Verqi</span>
  </div>
  <p style="color:#9A93B8; margin:0.6rem 0 0; font-size:15px;">
    Your AI-powered study assistant.
  </p>
</div>
<style>
[data-testid="stTabs"] [role="tablist"] { justify-content: center; }
</style>
"""
st.markdown(BRAND, unsafe_allow_html=True)

# --- API key guard ---
if not API_KEY:
    st.error("No Gemini API key found. Add GOOGLE_API_KEY to your .env file (local) "
             "or to the app's Secrets (Streamlit Cloud).")
    st.stop()

# --- Sidebar: document upload ---
with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader(
        "Upload PDF, Word, or text files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Reading and embedding your documents..."):
                try:
                    store, docs, raw_text = create_vector_store(uploaded, API_KEY)
                    st.session_state.vector_store = store
                    st.session_state.doc_text = raw_text
                    st.session_state.processed_files = [f.name for f in uploaded]
                    st.session_state.messages = []
                    for k in ("summary", "questions", "quiz", "flashcards", "concepts"):
                        st.session_state[k] = None
                    st.success(f"Processed {len(uploaded)} file(s) into {len(docs)} sections.")
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    show_error("processing documents", e)

    if st.session_state.processed_files:
        st.markdown("**Ready to chat with:**")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "Developed by **Azmain Tahmid Abir**  \n"
        "[LinkedIn](https://www.linkedin.com/in/azmain-abir) · "
        "[GitHub](https://github.com/azmainabir)"
    )


def render_sources(sources):
    with st.expander("Sources"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**{i}. {src['source']}**")
            st.caption(src["preview"])


# --- Tabs ---
tab_chat, tab_study, tab_math = st.tabs(["💬 Chat", "📖 Study Tools", "🧮 Math Solver"])

# ===== CHAT TAB =====
with tab_chat:
    if st.session_state.vector_store is None:
        st.info("Upload documents in the sidebar and click **Process documents** to start chatting.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # one retrieval powers both the answer and its citations
                    source_docs, stream = prepare_answer(
                        st.session_state.vector_store, prompt, API_KEY,
                        st.session_state.messages[:-1],
                    )
                    full_answer = st.write_stream(stream)

                    sources = [
                        {"source": d.metadata.get("source", "unknown"),
                         "preview": d.page_content[:200] + "..."}
                        for d in source_docs
                    ]
                    render_sources(sources)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_answer, "sources": sources}
                    )
                except Exception as e:
                    show_error("answering question", e)

# ===== STUDY TOOLS TAB =====
with tab_study:
    st.subheader("Study Tools")
    if st.session_state.vector_store is None:
        st.info("Process a document first to generate study aids.")
    else:
        st.caption("Generate study aids from your uploaded documents.")
        text = st.session_state.doc_text
        c1, c2, c3, c4, c5 = st.columns(5)

        tools = [
            (c1, "Summary", "summary", generate_summary, "Summarizing..."),
            (c2, "Questions", "questions", suggest_questions, "Generating questions..."),
            (c3, "Quiz", "quiz", generate_quiz, "Building a quiz..."),
            (c4, "Flashcards", "flashcards", generate_flashcards, "Making flashcards..."),
            (c5, "Key Points", "concepts", extract_key_concepts, "Extracting key concepts..."),
        ]
        for col, label, state_key, fn, spinner_text in tools:
            if col.button(label, use_container_width=True):
                with st.spinner(spinner_text):
                    try:
                        st.session_state[state_key] = fn(text, API_KEY)
                    except Exception as e:
                        show_error(f"generating {label.lower()}", e)

        if st.session_state.summary:
            st.markdown("### Summary")
            st.write(st.session_state.summary)

        if st.session_state.concepts:
            st.markdown("### Key Concepts")
            for c in st.session_state.concepts:
                st.markdown(f"- {c}")

        if st.session_state.questions:
            st.markdown("### Suggested Questions")
            for q in st.session_state.questions:
                st.markdown(f"- {q}")

        if st.session_state.quiz:
            st.markdown("### Quiz")
            for i, item in enumerate(st.session_state.quiz, 1):
                st.markdown(f"**Q{i}. {item['question']}**")
                for opt in item["options"]:
                    st.markdown(f"- {opt}")
                with st.expander("Show answer"):
                    st.markdown(f"**Answer:** {item['answer']}")
                    if item["explanation"]:
                        st.caption(item["explanation"])

        if st.session_state.flashcards:
            st.markdown("### Flashcards")
            st.caption("Click a term to reveal its definition.")
            for card in st.session_state.flashcards:
                with st.expander(card["term"]):
                    st.write(card["definition"])

# ===== MATH SOLVER TAB =====
with tab_math:
    st.subheader("Math Solver")
    st.caption("Type a math problem or upload a photo of one, and get a step-by-step solution.")

    mode = st.radio("Input type", ["Type it", "Upload image"], horizontal=True)

    # clear a stale solution when switching input modes
    if st.session_state.last_math_mode != mode:
        st.session_state.math_solution = None
        st.session_state.last_math_mode = mode

    if mode == "Type it":
        problem = st.text_area("Enter the math problem", height=120,
                               placeholder="e.g. Solve for x:  3x + 7 = 22")
        if st.button("Solve", type="primary", key="solve_text_btn"):
            if problem.strip():
                with st.spinner("Solving..."):
                    try:
                        st.session_state.math_solution = solve_text(problem, API_KEY)
                    except Exception as e:
                        show_error("solving math (text)", e)
            else:
                st.warning("Please enter a math problem first.")
    else:
        img = st.file_uploader("Upload an image of the problem",
                               type=["png", "jpg", "jpeg", "webp"], key="math_img")
        note = st.text_input("Optional note (e.g. 'solve for x' or 'find the area')")
        if img:
            st.image(img, width=320)
        if st.button("Solve", type="primary", key="solve_img_btn"):
            if img:
                with st.spinner("Reading and solving..."):
                    try:
                        st.session_state.math_solution = solve_image(
                            img.getvalue(), img.type, API_KEY, note
                        )
                    except Exception as e:
                        show_error("solving math (image)", e)
            else:
                st.warning("Please upload an image first.")

    if st.session_state.math_solution:
        st.markdown("### Solution")
        st.markdown(st.session_state.math_solution)
