import os
import shutil
from typing import List
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client()
    return _chroma_client


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    filename = Path(pdf_path).name
    docs = []

    try:
        import fitz
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source_file": filename, "page": page_num, "source": pdf_path},
                ))
        pdf_doc.close()
        if docs:
            return docs
    except Exception as e:
        print(f"pymupdf failed for {filename}: {e}")

    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={"source_file": filename, "page": page_num, "source": pdf_path},
                    ))
        if docs:
            return docs
    except Exception as e:
        print(f"PyPDF2 failed for {filename}: {e}")

    return docs


def process_pdfs(pdf_paths: List[str]):
    all_docs = []
    errors = []

    for pdf_path in pdf_paths:
        try:
            docs = extract_text_from_pdf(pdf_path)
            if docs:
                all_docs.extend(docs)
            else:
                errors.append(f"{Path(pdf_path).name}: no text extracted (may be image-only/scanned PDF)")
        except Exception as e:
            errors.append(f"{Path(pdf_path).name}: {str(e)}")

    if not all_docs:
        error_detail = "\n".join(errors) if errors else "Unknown error"
        raise ValueError(
            f"No text could be extracted from the uploaded PDFs.\n\n"
            f"Details:\n{error_detail}\n\n"
            f"Note: Image-only or scanned PDFs require OCR and are not supported."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        raise ValueError("No text chunks could be created from the documents.")

    embeddings = get_embeddings()
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_vector_store():
    client = get_chroma_client()
    embeddings = get_embeddings()
    try:
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        return vector_store
    except Exception:
        return None
