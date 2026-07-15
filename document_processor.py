# document_processor.py
# Verqi — Chat with your documents (RAG)
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
SUPPORTED_TYPES = (".pdf", ".docx", ".txt")


# --- Text extraction per file type ---
def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts)


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
def build_documents(files) -> List[Document]:
    """
    Turn uploaded files into a list of chunked LangChain Documents.
    `files` is a list of Streamlit UploadedFile objects (each has .name and .getvalue()).
    Each chunk carries its source filename so answers can be cited.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: List[Document] = []
    for f in files:
        raw = extract_text(f.name, f.getvalue())
        if not raw.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(raw)):
            docs.append(
                Document(page_content=chunk, metadata={"source": f.name, "chunk": i})
            )
    return docs


# --- Vector store ---
def create_vector_store(files, api_key: str) -> Tuple[FAISS, List[Document]]:
    """
    Extract, chunk, embed the uploaded files and return a FAISS store plus the raw chunks.
    Raises ValueError if no readable text is found.
    """
    docs = build_documents(files)
    if not docs:
        raise ValueError("No readable text found in the uploaded file(s).")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    store = FAISS.from_documents(docs, embeddings)
    return store, docs