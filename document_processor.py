# document_processor.py
# Verqi — Your AI-powered study assistant
# Developed by Azmain Tahmid Abir
# LinkedIn: https://www.linkedin.com/in/azmain-abir
# GitHub:   https://github.com/azmainabir

import io
import os
import tempfile
from typing import List, Tuple

import docx2txt
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Configuration ---
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CHUNKS = 400          # guards the shared API quota against oversized uploads
SUPPORTED_TYPES = (".pdf", ".docx", ".txt")


# --- Text extraction per file type ---
def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    # docx2txt needs a real file path, so write to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
    finally:
        os.remove(tmp_path)


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def extract_text(filename: str, data: bytes) -> str:
    """Route a file to the correct extractor based on its extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(data)
    if lower.endswith(".docx"):
        return _extract_docx(data)
    if lower.endswith(".txt"):
        return _extract_txt(data)
    raise ValueError(f"Unsupported file type: {filename}")


# --- Chunking ---
def build_documents(files) -> Tuple[List[Document], str]:
    """
    Turn uploaded files into (chunked Documents, clean raw text).
    The raw text is kept separately so study tools don't receive the
    duplicated overlap regions that chunking deliberately creates.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: List[Document] = []
    raw_parts: List[str] = []
    for f in files:
        raw = extract_text(f.name, f.getvalue())
        if not raw.strip():
            continue
        raw_parts.append(raw)
        for i, chunk in enumerate(splitter.split_text(raw)):
            docs.append(
                Document(page_content=chunk, metadata={"source": f.name, "chunk": i})
            )
    return docs, "\n\n".join(raw_parts)


# --- Vector store ---
def create_vector_store(files, api_key: str):
    """Extract, chunk, and embed uploads. Returns (store, chunks, raw_text)."""
    docs, raw_text = build_documents(files)
    if not docs:
        raise ValueError("No readable text found in the uploaded file(s). "
                         "If this is a scanned PDF, try a text-based file instead.")
    if len(docs) > MAX_CHUNKS:
        raise ValueError(
            f"These documents are too large ({len(docs)} sections). "
            f"Please upload something smaller — up to about {MAX_CHUNKS} sections."
        )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    store = FAISS.from_documents(docs, embeddings)
    return store, docs, raw_text