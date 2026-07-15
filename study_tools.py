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

MAX_CHARS = 100000  # how much document text the study tools read


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
    # remove markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    # remove trailing commas before closing brackets (common LLM slip)
    cleaned_fixed = re.sub(r",(\s*[\]}])", r"\1", cleaned)

    for candidate in (cleaned, cleaned_fixed):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # fall back: grab the outermost [...] or {...}
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start:end + 1]
            snippet = re.sub(r",(\s*[\]}])", r"\1", snippet)
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


QUESTION_STYLES = {
    "Short (1 mark)": (
        "short factual questions, each answerable in one or two sentences",
        1,
    ),
    "Medium (5 marks)": (
        "questions requiring a structured paragraph answer covering 3-5 key points",
        5,
    ),
    "Long (8 marks)": (
        "broad essay-style questions requiring a detailed multi-paragraph answer "
        "with explanation and examples",
        8,
    ),
    "Mixed": (
        "a mix of short (1 mark), medium (5 marks) and long (8 marks) questions",
        0,
    ),
}


def generate_questions(text: str, api_key: str, n: int = 5,
                       style: str = "Medium (5 marks)") -> List[Dict]:
    """Generate exam-style questions with model answers and mark values."""
    guidance, marks = QUESTION_STYLES.get(style, QUESTION_STYLES["Medium (5 marks)"])
    llm = get_llm(api_key, streaming=False, json_mode=True)
    schema = '[{"question": "...", "answer": "...", "marks": 5}]'
    if marks:
        mark_rule = f'Set "marks" to {marks} for every question.'
    else:
        mark_rule = 'Set "marks" to 1, 5 or 8 to match each question\'s depth.'
    messages = [
        ("system", "You are Verqi, an exam question setter. Respond ONLY with valid JSON."),
        ("user",
         f"From the document below, write {n} exam questions with model answers.\n"
         f"Style: {guidance}.\n"
         f"{mark_rule}\n"
         f"Answers must be based only on the document, and their length must match "
         f"the marks (1 mark = brief, 5 marks = a solid paragraph, 8 marks = detailed).\n"
         f"Respond as a JSON array shaped like: {schema}\n\n" + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            try:
                m = int(item.get("marks", marks or 5))
            except (TypeError, ValueError):
                m = marks or 5
            out.append({"question": str(item["question"]),
                        "answer": str(item["answer"]),
                        "marks": m})
    if not out:
        raise ValueError("The model did not return usable questions. Please try again.")
    return out[:n]


def generate_quiz(text: str, api_key: str, n: int = 5) -> List[Dict]:
    llm = get_llm(api_key, streaming=False, json_mode=True)
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
    llm = get_llm(api_key, streaming=False, json_mode=True)
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
    llm = get_llm(api_key, streaming=False, json_mode=True)
    messages = [
        ("system", "You extract key concepts. Respond ONLY with a JSON array of short strings."),
        ("user", f"List the {n} most important key concepts or points from this document. "
                 f"Respond as a JSON array of {n} short strings.\n\n" + _truncate(text)),
    ]
    data = _parse_json(_invoke_with_retry(llm, messages))
    return [str(c) for c in data][:n]
