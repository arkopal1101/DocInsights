from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel

from app.rag_pipeline import build_rag_pipeline, answer_question
from app.helpers import text_split, load_embedder, build_vectorstore, load_reranker, load_memory

app = FastAPI()

session_retrievers = {}
session_memory = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class UploadRequest(BaseModel):
    session_id: str


llm = build_rag_pipeline()


@app.post("/upload_pdfs")
async def upload_pdfs(session_id: str, files: list[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

    loader = PyPDFDirectoryLoader(path=temp_dir)
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

    if session_id not in session_memory:
        session_memory[session_id] = load_memory(llm)

    return {"status": "ok", "message": "PDFs uploaded and retriever ready"}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    retriever = session_retrievers.get(req.session_id)
    memory = session_memory.get(req.session_id)
    if retriever is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet for this session_id")
    return answer_question(req.question, retriever, llm, memory)
