## importing required libraries
import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

## loading variables from .env file 
load_dotenv()

VECTOR_DB_PATH = "faiss_index"
store_memory = {}


## creating HuggingFace Embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## initializing LLM from Open Source - ChatGroq
def get_llm():
    return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama3-8b-8192")

## loading documents 
def load_documents(file_paths : List[str]):
    documents = [] ## all the documents will be stored here 
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path) 
        elif path.endswith(".txt"):
            loader = TextLoader(path , encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    return documents

## splitting data into chunks 
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter( chunk_size = 500 , chunk_overlap=100)
    return splitter.split_documents(documents)

## creating vector DB 
def create_or_update_vectorstore(chunks):
    embeddings = get_embeddings()
    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(VECTOR_DB_PATH,embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH) ## saving it locally

## create session - id for maintaining the conversation history 
def get_memory(session_id : str):
    if session_id not in store_memory:
        store_memory[session_id] = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True
        )
    return store_memory[session_id]

## Finally creating RAG
## Here we are creating the entire RAG Chain
def get_rag_chain(session_id : str):
    if not os.path.exists(VECTOR_DB_PATH):
        raise RuntimeError("NO vector DB found.")
    embeddings = get_embeddings()
    db = FAISS.load_local(
        VECTOR_DB_PATH,embeddings)
    ## creating retriever for querying Vector DB
    retriever = db.as_retriever(search_kwargs={"k": 4})
    ## initializing llm
    llm = get_llm()
    memory = get_memory(session_id)
    ## chain
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True
    )
    return chain ## returning the entire chain