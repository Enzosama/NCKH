from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from datasets import load_datasets
import uvicorn


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Groq Chatbot API",
    description="A simple API for interacting with Groq's language models",
    version="1.0.0"
)

data = load_datasets(
    
)


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    
class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# Store conversations in memory
conversations: Dict[str, ConversationChain] = {}

def create_groq_chat():
    """Initialize a new Groq chat client"""
    api_key = "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"
    
    try:
        chat = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.7
        )
        return ConversationChain(
            llm=chat,
            memory=ConversationBufferMemory(),
            verbose=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize chat: {str(e)}")

def get_conversation(conversation_id: str = "default") -> ConversationChain:
    """Get or create a conversation chain for the given ID"""
    if conversation_id not in conversations:
        conversations[conversation_id] = create_groq_chat()
    return conversations[conversation_id]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Groq Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Check API health",
            "GET /chat": "Send a message using query parameters",
            "POST /chat": "Send a message using JSON body"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/chat")
async def chat_get(
    message: str,
    conversation_id: Optional[str] = "default",
    conversation: ConversationChain = Depends(get_conversation)
):
    """
    GET endpoint for chat messages
    
    Args:
        message: The message to send to the chatbot
        conversation_id: Optional ID to maintain separate conversations
    """
    try:
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
    conversation: ConversationChain = Depends(get_conversation)
):
    """
    POST endpoint for chat messages
    
    Args:
        request: ChatRequest object containing message and optional conversation_id
    """
    try:
        response = conversation.predict(input=request.message)
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/conversations")
async def list_conversations():
    """List all active conversation IDs"""
    return {
        "conversations": list(conversations.keys()),
        "count": len(conversations)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)