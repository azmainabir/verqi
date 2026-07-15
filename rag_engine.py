# rag_engine.py
# Verqi — Chat with your documents (RAG)
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import time
from typing import List, Dict, Iterator, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# --- Configuration ---
CHAT_MODEL = "gemini-flash-lite-latest"
RETRIEVE_K = 4            # how many chunks to pull for each question
MAX_RETRIES = 4          # retries for transient API errors
BASE_DELAY = 1.0         # seconds; grows exponentially (1s, 2s, 4s)

SYSTEM_RULES = (
    "You are Verqi, a study assistant that answers strictly from the provided "
    "document context. Follow these rules:\n"
    "1. Use ONLY the information in the context to answer.\n"
    "2. If the answer is not in the context, reply exactly: "
    "\"I couldn't find that in your documents.\" Do not guess or use outside knowledge.\n"
    "3. Be clear and concise. Use bullet points or steps when it helps.\n"
    "4. Answer in the same language the user asked the question in."
)

# Errors that are worth retrying (temporary, not our fault)
_TRANSIENT = ("503", "unavailable", "high demand", "429",
              "resource_exhausted", "getaddrinfo", "deadline", "timeout")


def _is_transient(error: Exception) -> bool:
    msg = str(error).lower()
    return any(k in msg for k in _TRANSIENT)


def _extract_text(content: Union[str, list]) -> str:
    """
    Gemini 3.x returns content as a string OR a list of blocks like
    [{'type': 'text', 'text': '...'}]. Pull out only the visible text,
    dropping signature/thinking blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def get_llm(api_key: str, streaming: bool = True) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model, tuned for fast grounded answers."""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=1.0,          # Gemini 3+ performs best at 1.0; lower can degrade/loop
        thinking_level="low",     # fast responses; retrieval already supplies the context
        disable_streaming=not streaming,
    )


def _format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context block with source labels."""
    blocks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[Source {i}: {source}]\n{d.page_content}")
    return "\n\n".join(blocks)


def _build_messages(question: str, context: str, history: List[Dict]) -> List:
    """Assemble the message list: system rules, recent history, and the grounded question."""
    messages = [("system", SYSTEM_RULES)]
    # include the last few turns for conversational follow-ups
    for turn in history[-6:]:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append((role, turn["content"]))
    user_prompt = (
        f"Answer the question using only the context below.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )
    messages.append(("user", user_prompt))
    return messages


def retrieve(store, question: str, k: int = RETRIEVE_K) -> List[Document]:
    """Find the most relevant chunks for a question."""
    return store.similarity_search(question, k=k)


def answer_stream(
    store, question: str, api_key: str, history: List[Dict] = None
) -> Iterator[str]:
    """
    Stream a grounded answer token-by-token, with automatic retry on
    transient errors (only before the first token, to avoid duplication).
    """
    history = history or []
    docs = retrieve(store, question)
    context = _format_context(docs)
    llm = get_llm(api_key, streaming=True)
    messages = _build_messages(question, context, history)

    for attempt in range(MAX_RETRIES):
        yielded = False
        try:
            for chunk in llm.stream(messages):
                text = _extract_text(chunk.content)
                if text:
                    yielded = True
                    yield text
            return
        except Exception as e:
            if (not yielded) and _is_transient(e) and attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
                continue
            raise


def answer(store, question: str, api_key: str, history: List[Dict] = None) -> str:
    """Non-streaming grounded answer, with automatic retry on transient errors."""
    history = history or []
    docs = retrieve(store, question)
    context = _format_context(docs)
    llm = get_llm(api_key, streaming=False)
    messages = _build_messages(question, context, history)

    for attempt in range(MAX_RETRIES):
        try:
            return _extract_text(llm.invoke(messages).content)
        except Exception as e:
            if _is_transient(e) and attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
                continue
            raise


def get_sources(store, question: str, k: int = RETRIEVE_K) -> List[Document]:
    """Return the chunks used to answer, for the citation display."""
    return retrieve(store, question, k=k)