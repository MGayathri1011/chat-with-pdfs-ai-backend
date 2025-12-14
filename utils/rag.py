from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import tempfile
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None


def ingest_pdf(pdf_bytes):
    global vectorstore

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    os.remove(pdf_path)


def chat_with_pdf(question: str):
    if vectorstore is None:
        return "Please upload a PDF first."

    docs = vectorstore.similarity_search(question, k=3)
    return docs[0].page_content
