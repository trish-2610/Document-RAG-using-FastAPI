import os
from typing import List

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from rag_pipeline import (
    load_documents,
    chunk_documents,
    create_or_update_vectorstore,
    get_rag_chain
)

UPLOAD_DIR = "uploads"

app = FastAPI(title="Document QA RAG API")


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        saved_paths.append(file_path)

    documents = load_documents(saved_paths)
    chunks = chunk_documents(documents)
    create_or_update_vectorstore(chunks)

    return {
        "status": "success",
        "files_uploaded": len(files),
        "chunks_created": len(chunks)
    }


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    chain = get_rag_chain(payload.session_id)

    result = chain({
        "question": payload.question
    })

    return {
        "answer": result["answer"],
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }
