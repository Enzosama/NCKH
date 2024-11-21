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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
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
    pdf_text = ""
    try:
        # Read the PDF file
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text
                else:
                    print(f"Warning: Page {page} has no extractable text.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    if not texts:
        raise HTTPException(status_code=400, detail="No valid text extracted from PDF.")

    # Create metadata for each chunk
    metadatas = [{"source": f"chunk-{i}"} for i in range(len(texts))]

    try:
        llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name="mixtral-8x7b-32768"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Groq LLM: {str(e)}")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
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
try:
    default_chain = process_pdf(DEFAULT_PDF_PATH) if os.path.exists(DEFAULT_PDF_PATH) else None
    document_chains["default"] = default_chain if default_chain else None
except Exception as e:
    print(f"Error processing default PDF: {e}")
    default_chain = None
    document_chains["default"] = None

@app.post("/chat")
async def chat_post(request: ChatRequest, pdf_context: Optional[str] = None):
    try:
        if pdf_context:
            if pdf_context not in document_chains or document_chains[pdf_context] is None:
                raise HTTPException(status_code=404, detail=f"No chain found for context '{pdf_context}'")
            
            res = document_chains[pdf_context].invoke({"question": request.message})
            sources = [
                doc.metadata.get('source', "Unknown source")
                for doc in res.get('source_documents', [])
            ]

            return ChatResponse(
                response=res.get('answer', "No answer generated."),
                conversation_id=request.conversation_id,
                sources=sources
            )

        # Default to conversational chat if no PDF context
        conversation = get_conversation(request.conversation_id)
        response = conversation.predict(input=request.message)

        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    except HTTPException as e:
        raise e
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