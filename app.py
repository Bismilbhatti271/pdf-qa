import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF Q&A Chatbot")
st.caption("Ask questions about your PDFs. Upload new ones in the sidebar.")


def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    return chroma_client.get_or_create_collection(name="pdf_docs")


def ingest_uploaded_pdf(content: bytes, filename: str) -> None:
    dest = DATA_DIR / filename
    dest.write_bytes(content)
    with st.spinner(f"Indexing {filename}..."):
        Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        docs = SimpleDirectoryReader(
            input_files=[str(dest)],
            file_extractor={".pdf": PDFReader()},
        ).load_data()
        vector_store = ChromaVectorStore(chroma_collection=get_collection())
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    load_engine.clear()


@st.cache_resource
def load_engine():
    if "GROQ_API_KEY" not in os.environ:
        st.stop("GROQ_API_KEY not set. Add it to .env.")

    Settings.llm = OpenAILike(
        model="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.1,
        is_chat_model=True,
    )
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(name="pdf_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=index.as_query_engine(similarity_top_k=4),
        memory=memory,
        verbose=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    engine = load_engine()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = engine.chat(prompt)
        st.markdown(str(response))
        with st.expander("Sources"):
            for node in response.source_nodes:
                st.write(f"- **{node.node.metadata.get('file_name', 'unknown')}** (score {node.score:.2f})")
    st.session_state.messages.append({"role": "assistant", "content": str(response)})


with st.sidebar:
    st.header("📤 Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded is not None:
        if not (DATA_DIR / uploaded.name).exists():
            ingest_uploaded_pdf(uploaded.getvalue(), uploaded.name)
            st.success(f"Indexed `{uploaded.name}` — you can ask questions about it now.")
        else:
            st.info(f"`{uploaded.name}` is already indexed.")

    st.divider()
    st.header("📚 Indexed PDFs")
    pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    if pdfs:
        for name in pdfs:
            st.write(f"- {name}")
    else:
        st.write("No PDFs yet.")