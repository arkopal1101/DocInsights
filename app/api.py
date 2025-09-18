import asyncio
import os
import shutil
import tempfile
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel

from app.rag_pipeline import build_rag_pipeline, answer_question
from app.helpers import text_split, load_embedder, build_vectorstore, load_reranker, load_memory

SESSION_DIR_TTL = 3600
SESSION_STORAGE_DIR = os.path.join(tempfile.gettempdir(), "docinsights_sessions")
os.makedirs(SESSION_STORAGE_DIR, exist_ok=True)

session_retrievers = {}
session_memory = {}


def cleanup_stale_sessions():
    """Remove session directories and in-memory data that have gone stale."""
    if not os.path.isdir(SESSION_STORAGE_DIR):
        return

    now = time.time()
    for entry in os.listdir(SESSION_STORAGE_DIR):
        session_path = os.path.join(SESSION_STORAGE_DIR, entry)
        if not os.path.isdir(session_path):
            continue

        try:
            last_modified = os.path.getmtime(session_path)
        except OSError:
            continue

        if now - last_modified > SESSION_DIR_TTL:
            shutil.rmtree(session_path, ignore_errors=True)
            session_retrievers.pop(entry, None)
            session_memory.pop(entry, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stale_sessions()

    async def periodic_cleanup():
        while True:
            cleanup_stale_sessions()
            await asyncio.sleep(SESSION_DIR_TTL)
            cleanup_stale_sessions()

    cleanup_task = asyncio.create_task(periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class UploadRequest(BaseModel):
    session_id: str


llm = build_rag_pipeline()
app = FastAPI(lifespan=lifespan)


@app.post("/upload_pdfs")
async def upload_pdfs(session_id: str, files: list[UploadFile] = File(...)):
    cleanup_stale_sessions()

    session_dir = os.path.join(SESSION_STORAGE_DIR, session_id)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
    os.makedirs(session_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

    loader = PyPDFDirectoryLoader(path=session_dir)
    docs = loader.load()

    text_splitter = text_split()
    docs_split = text_splitter.split_documents(docs)

    bm25_retriever = BM25Retriever.from_documents(docs_split)
    bm25_retriever.k = 3

    embeddings = load_embedder()
    vector_store = build_vectorstore(docs_split, embeddings)
    retriever_vectordb = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_vectordb],
                                           weights=[0.4, 0.6])
    retriever = load_reranker(ensemble_retriever)

    session_retrievers[session_id] = retriever

    os.utime(session_dir, None)

    if session_id not in session_memory:
        session_memory[session_id] = load_memory(llm)

    return {"status": "ok", "message": "PDFs uploaded and retriever ready"}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    retriever = session_retrievers.get(req.session_id)
    memory = session_memory.get(req.session_id)
    if retriever is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet for this session_id")
    session_dir = os.path.join(SESSION_STORAGE_DIR, req.session_id)
    if os.path.isdir(session_dir):
        os.utime(session_dir, None)
    return answer_question(req.question, retriever, llm, memory)
