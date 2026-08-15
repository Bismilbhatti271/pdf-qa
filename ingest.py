import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

sys.path.insert(0, str(BASE_DIR))

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.readers.file import PDFReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb


def main() -> None:
    if "GROQ_API_KEY" not in os.environ:
        sys.exit("GROQ_API_KEY not set. Add it to .env.")

    Settings.llm = OpenAILike(
        model="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.1,
        is_chat_model=True,
    )
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {DATA_DIR}. Drop your PDFs there first.")
    print(f"Indexing {len(pdfs)} PDF(s): {', '.join(p.name for p in pdfs)}")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
        file_extractor={".pdf": PDFReader()},
    ).load_data()

    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(name="pdf_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=str(BASE_DIR / "storage"))

    print(f"Done. Indexed {len(documents)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()