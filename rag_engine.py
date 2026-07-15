# rag_engine.py
# Verqi — Your AI-powered study assistant
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import time
from typing import List, Dict, Iterator, Tuple, Union
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
    "4. Answer in the same language the user asked the question in.\n"
    "5. Never mention the underlying model or provider — you are simply Verqi."
)

# Errors worth retrying (temporary, not our fault)
_TRANSIENT = ("503", "unavailable", "high demand", "429",
              "resource_exhausted", "getaddrinfo", "deadline", "timeout")


def _is_transient(error: Exception) -> bool:
    return any(k in str(error).lower() for k in _TRANSIENT)


def friendly_error(error: Exception) -> str:
    """Turn a raw API exception into a message that's safe to show a user."""
    msg = str(error).lower()
    if any(k in msg for k in ("429", "resource_exhausted", "quota")):
        return "Verqi has reached its usage limit for today. Please try again tomorrow."
    if any(k in msg for k in ("503", "unavailable", "high demand", "overloaded")):
        return "Verqi is busy right now. Please try again in a moment."
    if any(k in msg for k in ("getaddrinfo", "connection", "timeout", "deadline")):
        return "Network problem. Check your connection and try again."
    if any(k in msg for k in ("api key", "permission", "401", "403")):
        return "Verqi isn't configured correctly. Please contact the developer."
    if any(k in msg for k in ("not found", "404")):
        return "Verqi is temporarily unavailable. Please try again later."
    return "Something went wrong. Please try again."


def _extract_text(content: Union[str, list]) -> str:
    """Gemini returns a string OR a list of blocks; keep only the visible text."""
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


def get_llm(api_key: str, streaming: bool = True,
            json_mode: bool = False) -> ChatGoogleGenerativeAI:
    """Create the chat model. json_mode forces valid JSON output."""
    kwargs = dict(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=1.0,
        thinking_level="low",     # fast; retrieval already supplies the context
        disable_streaming=not streaming,
    )
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[Source {i}: {source}]\n{d.page_content}")
    return "\n\n".join(blocks)


def _build_messages(question: str, context: str, history: List[Dict]) -> List:
    messages = [("system", SYSTEM_RULES)]
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


def prepare_answer(store, question: str, api_key: str,
                   history: List[Dict] = None) -> Tuple[List[Document], Iterator[str]]:
    """
    Retrieve ONCE, then return (source_docs, answer_generator).
    Retrieving a single time halves the embedding calls and the latency
    compared with fetching the answer and its sources separately.
    """
    history = history or []
    docs = retrieve(store, question)
    context = _format_context(docs)
    llm = get_llm(api_key, streaming=True)
    messages = _build_messages(question, context, history)

    def generate():
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
                # only retry before the first token, so nothing is duplicated
                if (not yielded) and _is_transient(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                raise

    return docs, generate()