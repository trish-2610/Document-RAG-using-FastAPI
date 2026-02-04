## importing libraries
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_pipeline import (load_documents,chunk_documents,create_or_update_vectorstore,get_rag_chain)

UPLOAD_DIR = "uploads"

app = FastAPI(title = "Document RAG")

## creating class for Query
class Question(BaseModel):
    session_id: str
    question: str

## /upload endpoint (route)
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)) :
    if not files :
        raise HTTPException(status_code = 400, detail=" No files uploaded !!!! ")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths = []
    
    ## reading and processing files
    for file in files:
        file_path = os.path.join(UPLOAD_DIR , file.filename)
        with open(file_path , "wb") as f:
            f.write(await file.read())
        saved_paths.append(file_path)
    try:
        documents = load_documents(saved_paths) ## load documents from user
        chunks = chunk_documents(documents) ## splitting documents 
        create_or_update_vectorstore(chunks) ## create vector_store 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "message" : "Successfully uploaded the documents",
        "files_uploaded" : len(files),
        "chunks_created" : len(chunks)
    }

## /ask route
@app.post("/ask")
def ask_question(payload : Question):
    try:
        chain = get_rag_chain(payload.session_id)
        result = chain({"question" : payload.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return { ## output 
        "answer": result["answer"],
        "sources": [
            {
                "content": doc.page_content , "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }