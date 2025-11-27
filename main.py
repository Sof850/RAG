from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Global variables to hold the model and vectorstore
rag_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load resources
    print("🚀 Starting up... Loading RAG resources...")
    
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Database not found at {CHROMA_PATH}")
        # In a real app, we might want to raise an error or handle this gracefully
        rag_resources["error"] = "Database not found. Please run data_cutter.py first."
    else:
        print(f"📂 Loading ChromaDB from {CHROMA_PATH}...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        rag_resources["vectorstore"] = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        print(f"🤖 Loading AI Model ({MODEL_ID})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
            
            rag_resources["tokenizer"] = tokenizer
            rag_resources["pipe"] = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                device=-1,  # Run on CPU
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            print("✅ AI Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            rag_resources["error"] = f"Error loading model: {str(e)}"

    yield
    
    # Shutdown: Clean up resources if needed
    print("👋 Shutting down...")
    rag_resources.clear()

app = FastAPI(lifespan=lifespan, title="RAG Chat API")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if "error" in rag_resources:
        raise HTTPException(status_code=500, detail=rag_resources["error"])
    
    if "vectorstore" not in rag_resources or "pipe" not in rag_resources:
        raise HTTPException(status_code=503, detail="RAG resources not initialized yet.")

    query = request.query
    vectorstore = rag_resources["vectorstore"]
    pipe = rag_resources["pipe"]
    tokenizer = rag_resources["tokenizer"]

    # Search documents
    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Prepare prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate response
    outputs = pipe(prompt)
    generated_text = outputs[0]['generated_text']

    # Extract response
    if "<|im_start|>assistant" in generated_text:
        response_text = generated_text.split("<|im_start|>assistant")[-1].strip()
    elif prompt in generated_text:
        response_text = generated_text.replace(prompt, "").strip()
    else:
        response_text = generated_text

    return ChatResponse(
        response=response_text,
        sources=[doc.page_content[:100] + "..." for doc in results]
    )

@app.get("/")
async def root():
    return {"message": "RAG Chat API is running. Send POST requests to /chat"}
