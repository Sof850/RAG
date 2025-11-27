import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from data_cutter import create_db

# Constants
CHROMA_PATH = "chroma_db"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("🚀 Starting app...")

# 1. Initialize/Load Database
print("🔄 Initializing database from data folder...")
# We rebuild the DB on startup to ensure it matches the current data
try:
    vectorstore = create_db()
    print("✅ Database created successfully!")
except Exception as e:
    print(f"❌ Error creating database: {e}")
    # Fallback: try to load if exists, though create_db should have handled it
    if os.path.exists(CHROMA_PATH):
        print("⚠️ Attempting to load existing database...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        raise e

# 2. Load AI Model
print(f"🤖 Loading AI Model ({MODEL_ID})...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    pipe = pipeline(
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
    raise e

def chat_function(message, history):
    print(f"📨 Received query: {message}")
    
    # Search documents
    results = vectorstore.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Prepare prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}
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
        
    return response_text

# Create Gradio Interface
demo = gr.ChatInterface(
    fn=chat_function,
    title="RAG Chat with Your Data",
    description=f"Ask questions about your documents. Powered by {MODEL_ID}.",
    examples=["What is the main topic?", "Summarize the content."],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
