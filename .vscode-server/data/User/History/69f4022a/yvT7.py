import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, List
import PyPDF2
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain

import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chat API",
    description="API for chatting with PDF documents using Groq LLM",
    version="1.0.0"
)

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    
class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: Optional[List[str]] = None

# Global variables to store conversations and document retrieval chain
conversations: Dict[str, ConversationChain] = {}
document_chains: Dict[str, ConversationalRetrievalChain] = {}

def create_groq_chat(temperature: float = 0.7):
    """Initialize a new Groq chat client"""
    try:
        chat = ChatGroq(
            groq_api_key=os.environ.get('GROQ_API_KEY'),
            model_name="mixtral-8x7b-32768",
            temperature=temperature
        )
        return ConversationChain(
            llm=chat,
            memory=ConversationBufferMemory(),
            verbose=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize chat: {str(e)}")

def process_pdf(file_path: str):
    """Process PDF and create vector store"""
    # Read the PDF file
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create conversational retrieval chain
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API_KEY'),
        model_name="mixtral-8x7b-32768"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

def get_conversation(conversation_id: str = "default") -> ConversationChain:
    """Get or create a conversation chain for the given ID"""
    if conversation_id not in conversations:
        conversations[conversation_id] = create_groq_chat()
    return conversations[conversation_id]

# Default PDF file path
DEFAULT_PDF_PATH = "651778122-The-Ultimate-Guide-to-Generative-AI-Studio-on-Google-Cloud-s-Vertex-AI.pdf"

# Load default PDF on startup if exists
default_chain = process_pdf(DEFAULT_PDF_PATH) if os.path.exists(DEFAULT_PDF_PATH) else None
document_chains["default"] = default_chain if default_chain else None

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    try:
        # Save uploaded file
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the uploaded PDF
        chain = process_pdf(file.filename)
        document_chains[file.filename] = chain
        
        return {"message": "PDF processed successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/chat")
async def chat_get(
    message: str,
    conversation_id: Optional[str] = "default",
    pdf_context: Optional[str] = None,
    conversation: ConversationChain = Depends(get_conversation)
):
    """
    GET endpoint for chat messages with optional PDF context
    """
    try:
        # If PDF context is specified and exists, use PDF-based chain
        if pdf_context and pdf_context in document_chains and document_chains[pdf_context]:
            res = document_chains[pdf_context].invoke({"question": message})
            sources = [doc.metadata['source'] for doc in res.get('source_documents', [])] if res.get('source_documents') else []
            
            return ChatResponse(
                response=res['answer'],
                conversation_id=conversation_id,
                sources=sources
            )
        
        # Otherwise, use standard conversation chain
        response = conversation.predict(input=message)
        return ChatResponse(
            response=response,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/chat")
async def chat_post(
    request: ChatRequest,
    pdf_context: Optional[str] = None
):
    """
    POST endpoint for chat messages with optional PDF context
    """
    try:
        # If PDF context is specified and exists, use PDF-based chain
        if pdf_context and pdf_context in document_chains and document_chains[pdf_context]:
            res = document_chains[pdf_context].invoke({"question": request.message})
            sources = [doc.metadata['source'] for doc in res.get('source_documents', [])] if res.get('source_documents') else []
            
            return ChatResponse(
                response=res['answer'],
                conversation_id=request.conversation_id,
                sources=sources
            )
        
        # Get or create conversation chain
        conversation = get_conversation(request.conversation_id)
        response = conversation.predict(input=request.message)
        
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the PDF Chat API",
        "version": "1.0.0",
        "default_pdf": DEFAULT_PDF_PATH if os.path.exists(DEFAULT_PDF_PATH) else "No default PDF loaded",
        "endpoints": {
            "POST /upload-pdf": "Upload and process a new PDF",
            "GET /chat": "Send a message with optional PDF context",
            "POST /chat": "Send a message with optional PDF context"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)