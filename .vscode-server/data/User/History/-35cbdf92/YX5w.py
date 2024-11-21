import os
import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import shutil

app = FastAPI()

# Global variables for PDF processing
global_docsearch = None
global_chain = None
global_message_history = ChatMessageHistory()

# Default dataset path
DEFAULT_DATASET_PATH = "651778122-The-Ultimate-Guide-to-Generative-AI-Studio-on-Google-Cloud-s-Vertex-AI.pdf"

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
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=global_message_history,
        return_messages=True,
    )

    # Create LLM chains
    llm_local = ChatOllama(model="mistral:instruct")
    
    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    return docsearch, chain

@app.on_event("startup")
async def startup_event():
    """
    Load default dataset on application startup
    """
    global global_docsearch, global_chain
    try:
        global_docsearch, global_chain = process_pdf(DEFAULT_DATASET_PATH)
        print(f"Loaded default dataset from {DEFAULT_DATASET_PATH}")
    except Exception as e:
        print(f"Error loading default dataset: {e}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    API endpoint to upload PDF and process it
    """
    global global_docsearch, global_chain, global_message_history
    
    # Reset message history
    global_message_history.clear()
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the uploaded PDF
        global_docsearch, global_chain = process_pdf(temp_path)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return JSONResponse(content={"message": "PDF processed successfully"})
    
    except Exception as e:
        # Remove temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat_with_pdf(query: str = Query(..., min_length=1)):
    """
    API endpoint for chatting with the processed PDF
    """
    global global_chain, global_message_history
    
    if global_chain is None:
        raise HTTPException(status_code=400, detail="No PDF processed yet")
    
    try:
        # Call the chain with the user's query
        res = global_chain({"question": query})
        
        return {
            "answer": res["answer"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in res.get("source_documents", [])
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reset-chat/")
async def reset_chat():
    """
    API endpoint to reset chat history
    """
    global global_message_history
    global_message_history.clear()
    return {"message": "Chat history reset successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)