import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

VECTOR_DB_PATH = "faiss_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_memory_store = {}


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0
    )


def load_documents(file_paths: List[str]):
    documents = []

    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def create_or_update_vectorstore(chunks):
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_DB_PATH)


def get_memory(session_id: str):
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return _memory_store[session_id]


def get_rag_chain(session_id: str):
    if not os.path.exists(VECTOR_DB_PATH):
        raise RuntimeError("No vector store found. Upload documents first.")

    embeddings = get_embeddings()

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()
    memory = get_memory(session_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain