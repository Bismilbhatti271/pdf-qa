# PDF Q&A Chatbot

Chat with your PDFs using RAG (Retrieval-Augmented Generation). Upload a PDF, ask questions in plain English, and get answers with citations to the exact source chunks — no OpenAI key required.

## How it works

Two phases:

1. **Ingestion** — each PDF is read with `pypdf`, split into small text chunks, and converted into vector embeddings. Embeddings are computed **locally on your machine** (free, no API calls) with [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`) and stored in **ChromaDB** (`chroma_db/`).

2. **Question answering** — when you ask a question:
   - the question is embedded with the same local model
   - ChromaDB retrieves the 4 most similar text chunks from your PDFs
   - those chunks + your question are sent to **Groq** (LLaMA 3.3 70B, via the OpenAI-compatible API) which writes the answer
   - sources are shown under each answer with relevance scores

The AI never sees whole documents — only the relevant excerpts. That keeps answers fast, cheap, and grounded in your actual files.

## Requirements

- Python 3.12+
- A free [Groq](https://console.groq.com) API key

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your `.env` file (the app reads `GROQ_API_KEY`):

```powershell
Copy-Item .env.example .env
# edit .env and put your real key:
# GROQ_API_KEY=gsk_...
```

## Usage

### Option A — everything from the browser (recommended)

```powershell
streamlit run app.py
```

Open **http://localhost:8501**. Upload PDFs in the sidebar — they are indexed automatically and you can ask about them immediately.

### Option B — index PDFs from the command line

Drop PDFs into `data/`, then:

```powershell
python ingest.py
```

Re-run `ingest.py` whenever you add PDFs manually. The chat app at `streamlit run app.py` serves both options.

## Project structure

- `data/` — your PDFs (uploaded via the UI or placed here manually)
- `ingest.py` — reads PDFs, chunks, embeds locally, stores in ChromaDB
- `app.py` — Streamlit chat UI with PDF upload, chat memory, and source citations
- `chroma_db/` — vector database (auto-generated)
- `.env` — your Groq API key (never commit this)

## Tech stack

| Component | Tool |
|---|---|
| Chat UI | Streamlit |
| RAG framework | LlamaIndex |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` (local, free) |
| Vector store | ChromaDB |
| PDF parsing | pypdf |
