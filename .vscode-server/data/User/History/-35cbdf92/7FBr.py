import os
import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
import shutil

app = FastAPI()

# Global variables for PDF processing and conversations
global_docsearch = None
global_llm = None
conversations = {}

# Default dataset path
DEFAULT_DATASET_PATH = "dataset.pdf"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = Field(None)

def process_pdf(file_path):
    """
    Process PDF file and create vector store
    """
    # Read the PDF file
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Create LLM 
    llm_local = ChatOllama(model="mistral:instruct")
    
    return docsearch, llm_local

@app.on_event("startup")
async def startup_event():
    """
    Load default dataset on application startup
    """
    global global_docsearch, global_llm
    try:
        global_docsearch, global_llm = process_pdf(DEFAULT_DATASET_PATH)
        print(f"Loaded default dataset from {DEFAULT_DATASET_PATH}")
    except Exception as e:
        print(f"Error loading default dataset: {e}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    API endpoint to upload PDF and process it
    """
    global global_docsearch, global_llm
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the uploaded PDF
        global_docsearch, global_llm = process_pdf(temp_path)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return JSONResponse(content={"message": "PDF processed successfully"})
    
    except Exception as e:
        # Remove temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
@app.get("/chat")
async def chat_with_pdf(request: Request):
    """
    API endpoint for chatting with the processed PDF
    """
    global global_docsearch, global_llm
    
    # Handle both GET and POST requests
    if request.method == 'GET':
        params = dict(request.query_params)
        message = params.get('message')
        conversation_id = params.get('conversation_id', 'default')
    else:  # POST
        try:
            body = await request.json()
            message = body.get('message')
            conversation_id = body.get('conversation_id', 'default')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid request body")
    
    # Validate inputs
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # Ensure conversation exists
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Simulate a simple response (replace with actual LLM processing)
    response = f"You said: {message}"
    
    # Store conversation
    conversations[conversation_id].append({
        "message": message,
        "response": response
    })
    
    return {
        "message": message,
        "response": response,
        "conversation_id": conversation_id
    }

@app.get("/conversations")
async def list_conversations():
    """
    API endpoint to list active conversations
    """
    return {
        "conversations": list(conversations.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)