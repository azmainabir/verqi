# study_tools.py
# Verqi — Chat with your documents (RAG)
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import json
import re
import time
from typing import List, Dict

from rag_engine import get_llm, _extract_text, _is_transient, MAX_RETRIES, BASE_DELAY

MAX_CHARS = 24000  # cap the document text sent to the model (controls latency/tokens)


def _truncate(text: str) -> str:
    return text[:MAX_CHARS]


def _invoke_with_retry(llm, messages) -> str:
    """Call the model once, retrying on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return _extract_text(llm.invoke(messages).content)
        except Exception as e:
            if _is_transient(e) and attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
                continue
            raise


def _parse_json(raw: str):
    """Robustly extract a JSON array/object from a possibly-messy LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    cleaned_fixed = re.sub(r",(\s*[\]}])", r"\1", cleaned)

    for candidate in (cleaned, cleaned_fixed):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            snippet = re.sub(r",(\s*[\]}])", r"\1", cleaned[start:end + 1])
            try:
                return json.loads(snippet)
            except Exception:
                continue

    raise ValueError("Could not parse a valid JSON response from the model.")


# --- Public study-tool functions ---
def generate_summary(text: str, api_key: str) -> str:
    llm = get_llm(api_key, streaming=False)
    messages = [
        ("system", "You are a study assistant that writes clear, concise summaries."),
        ("user", "Write a concise summary (4-6 sentences) of the following document:\n\n"
                 + _truncate(text)),
    ]
    return _invoke_with_retry(llm, messages)


def suggest_questions(text: str, api_key: str, n: int = 4) -> List[str]:
    llm = get_llm(api_key, streaming=False)
    messages = [
        ("system", "You generate study questions. Respond ONLY with a JSON array of strings."),
        ("user", f"Based on this document, generate {n} insightful questions a student might ask "
                 f"to study it. Respond as a JSON array of {n} question strings.\n\n"
                 + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    return [str(q) for q in data][:n]


def generate_quiz(text: str, api_key: str, n: int = 5) -> List[Dict]:
    llm = get_llm(api_key, streaming=False)
    schema = ('[{"question": "...", "options": ["A", "B", "C", "D"], '
              '"answer": "the exact correct option text", "explanation": "..."}]')
    messages = [
        ("system", "You create multiple-choice quizzes. Respond ONLY with valid JSON."),
        ("user", f"Create a {n}-question multiple-choice quiz from this document. Each question "
                 f"has exactly 4 options, one correct answer (matching one option exactly), and a "
                 f"short explanation. Respond as a JSON array shaped like: {schema}\n\n"
                 + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    quiz = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q, opts, ans = item.get("question"), item.get("options"), item.get("answer")
        if q and isinstance(opts, list) and len(opts) >= 2 and ans is not None:
            quiz.append({
                "question": str(q),
                "options": [str(o) for o in opts],
                "answer": str(ans),
                "explanation": str(item.get("explanation", "")),
            })
    if not quiz:
        raise ValueError("The model did not return a usable quiz. Please try again.")
    return quiz[:n]


def generate_flashcards(text: str, api_key: str, n: int = 8) -> List[Dict]:
    llm = get_llm(api_key, streaming=False)
    schema = '[{"term": "...", "definition": "..."}]'
    messages = [
        ("system", "You create study flashcards. Respond ONLY with valid JSON."),
        ("user", f"Create {n} flashcards (a key term and its definition) from this document. "
                 f"Respond as a JSON array shaped like: {schema}\n\n" + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    cards = []
    for item in data:
        if isinstance(item, dict) and item.get("term") and item.get("definition"):
            cards.append({"term": str(item["term"]), "definition": str(item["definition"])})
    if not cards:
        raise ValueError("The model did not return usable flashcards. Please try again.")
    return cards[:n]


def extract_key_concepts(text: str, api_key: str, n: int = 6) -> List[str]:
    llm = get_llm(api_key, streaming=False)
    messages = [
        ("system", "You extract key concepts. Respond ONLY with a JSON array of short strings."),
        ("user", f"List the {n} most important key concepts or points from this document. "
                 f"Respond as a JSON array of {n} short strings.\n\n" + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    return [str(c) for c in data][:n]