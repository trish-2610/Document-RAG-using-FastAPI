# Document RAG using FastAPI

This project is a simple Document Question Answering system built using FastAPI and LangChain.
Here the users can upload PDF or TXT files and ask questions that are answered strictly based on the uploaded documents using a Retrieval-Augmented Generation (RAG) pipeline.

## Tech Stack
- FastAPI ( web )
- LangChain ( RAG )
- Groq (LLM) 
- Hugging Face Embeddings
- FAISS (Vector Store)

## Setup and Run

### 1. Create and activate virtual environment
###### python -m venv venv
###### venv\Scripts\activate   ## Windows

### 2. Install dependencies
###### pip install -r requirements.txt

### 3. Create a .env file in the root directory
###### GROQ_API_KEY=your_api_key_here

### 4. Run the server
###### uvicorn main:app --reload
###### Server will start at : http://127.0.0.1:8000

### 5. API Endpoints
###### Upload Documents : POST /upload
###### Ask Question : POST /ask

###### Sample Request Code
{
  "session_id": "session_1",
  "question": "What is this document about?"
}

## Project Structure 
Document-QA-RAG/
│
├── main.py               ## FastAPI application ( API endpoints )
├── rag_pipeline.py       ## RAG logic : loaders , chunking , embeddings retriever and memory
├── requirements.txt      ## Project dependencies
├── README.md             ## Project documentation
├── .env                  ## Environment variables ( Groq API key )
├── .gitignore            ## Ignored files and folders
│
├── uploads/              # Uploaded PDF/TXT documents (created at runtime)
└── faiss_index/          # FAISS vector store (created after upload)
