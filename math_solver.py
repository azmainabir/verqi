# math_solver.py
# Verqi — Chat with your documents (RAG)
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import base64
import time
from langchain_core.messages import HumanMessage, SystemMessage
from rag_engine import get_llm, _extract_text, _is_transient, MAX_RETRIES, BASE_DELAY

MATH_SYSTEM = (
    "You are Verqi, a friendly study assistant created by Azmain Tahmid Abir. "
    "Your job here is to solve math problems. Solve the problem step by step, "
    "showing your reasoning clearly in simple language a student can follow. "
    "Format any math clearly. End with the final result on its own line, "
    "labeled 'Answer:'. "
    "If the input is not a math problem, briefly say you're Verqi's math solver "
    "and invite them to enter a math problem or upload a photo of one. "
    "Never say you are a large language model or mention Google, OpenAI, or Gemini — "
    "you are simply Verqi."
)


def _invoke_with_retry(llm, messages):
    for attempt in range(MAX_RETRIES):
        try:
            return _extract_text(llm.invoke(messages).content)
        except Exception as e:
            if _is_transient(e) and attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
                continue
            raise


def solve_text(problem: str, api_key: str) -> str:
    llm = get_llm(api_key, streaming=False)
    messages = [
        SystemMessage(content=MATH_SYSTEM),
        HumanMessage(content=f"Solve this math problem step by step:\n\n{problem}"),
    ]
    return _invoke_with_retry(llm, messages)


def solve_image(image_bytes: bytes, mime_type: str, api_key: str, note: str = "") -> str:
    llm = get_llm(api_key, streaming=False)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = "Solve the math problem in this image step by step."
    if note.strip():
        prompt += f" Student's note: {note.strip()}"
    messages = [
        SystemMessage(content=MATH_SYSTEM),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
        ]),
    ]
    return _invoke_with_retry(llm, messages)