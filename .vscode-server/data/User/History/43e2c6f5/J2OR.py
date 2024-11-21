import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Conversation storage
conversations = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@app.get("/chat")
async def chat_get(message: str, conversation_id: Optional[str] = None):
    """
    Handle GET request for chat
    """
    # Ensure conversation_id exists
    if not conversation_id:
        conversation_id = 'default'
    
    # Initialize conversation if not exists
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Store and respond to message
    response_text = f"Response to: {message}"
    conversations[conversation_id].append({
        "message": message,
        "response": response_text
    })
    
    return {
        "message": message,
        "response": response_text,
        "conversation_id": conversation_id
    }

@app.post("/chat")
async def chat_post(request: Request):
    """
    Handle POST request for chat
    """
    try:
        # Parse JSON body
        body = await request.json()
        message = body.get('message')
        conversation_id = body.get('conversation_id', 'default')
        
        # Validate message
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Initialize conversation if not exists
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Store and respond to message
        response_text = f"Response to: {message}"
        conversations[conversation_id].append({
            "message": message,
            "response": response_text
        })
        
        return {
            "message": message,
            "response": response_text,
            "conversation_id": conversation_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/conversations")
async def list_conversations():
    """
    List active conversations
    """
    return {
        "conversations": list(conversations.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)