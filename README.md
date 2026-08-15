# PDF Q&A Chatbot

Chat with your PDFs using RAG (Retrieval-Augmented Generation).

## Setup

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add your OpenAI API key:

```powershell
Copy-Item .env.example .env
# then edit .env and put your real key
```

3. Drop your PDFs into the `data/` folder.

4. Build the index (run once, re-run whenever you add PDFs):

```powershell
python ingest.py
```

5. Launch the chatbot:

```powershell
streamlit run app.py
```

## Project structure

- `data/` — put your PDFs here
- `ingest.py` — reads PDFs, chunks them, embeds and stores them in ChromaDB
- `app.py` — Streamlit chat UI with memory and source citations
