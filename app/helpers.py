import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


def load_model():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return llm


def load_memory(llm):
    summary_memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )

    return summary_memory


def format_sources(docs):
    return [
        {
            "page": doc.metadata.get("page", "N/A"),
            "source": os.path.basename(doc.metadata.get("source", "N/A")),
            "snippet": doc.page_content
        }
        for doc in docs
    ]


def text_split():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    return text_splitter


def load_embedder():
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    return embedder


def build_vectorstore(docs, embedder):
    vector_db = Chroma.from_documents(
        docs,
        embedding=embedder
    )
    return vector_db


def load_reranker(ensemble_retriever):
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder_model)
    rerank_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
    )

    return rerank_retriever
