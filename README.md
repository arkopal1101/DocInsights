# DocInsights

DocInsights is a lightweight document question answering system. A FastAPI backend ingests PDFs and uses a retrieval‑augmented generation (RAG) pipeline, while a Streamlit frontend offers an easy chat interface.

## Features
- Upload PDF documents to build a searchable knowledge base
- Ask natural language questions and receive answers with source snippets
- Combines BM25 and vector search with cross‑encoder reranking

## Requirements
- Python 3.11+
- An OpenAI API key available in a `.env` file (used by LangChain components)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App
1. **Start the backend**
   ```bash
   uvicorn app.api:app --reload
   ```
2. **Launch the frontend** in a separate shell:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

The Streamlit UI lets you upload PDFs and ask questions about their content. The backend exposes two endpoints:

- `POST /upload_pdfs` – send one or more PDF files to index
- `POST /ask` – send `{ "question": "..." }` to query the indexed documents

## Project Structure
- `app/` – FastAPI app and RAG helpers
- `frontend/` – Streamlit interface
- `requirements.txt` – Python dependencies

## License
This project is provided without a specific license.
