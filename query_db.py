from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
import os
import torch

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"

def main():
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Database not found at {CHROMA_PATH}")
        print("Please run data_cutter.py first to create the database.")
        return
    
    # Load the existing vector store
    print(f"📂 Loading ChromaDB from {CHROMA_PATH}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Load the LLM (Writer)
    # Using Qwen2.5-0.5B-Instruct: A modern, high-performance small model
    print("🤖 Loading AI Model (Qwen/Qwen2.5-0.5B-Instruct)...")
    print("   (This is a smarter model with a larger context window!)")
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: You might need to install torch with CPU support if you haven't already.")
        return

    # Create a pipeline for text generation
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
    
    print(f"✅ AI Model loaded successfully!")
    
    # Interactive query loop
    print("\n" + "="*60)
    print("🤖 RAG Chatbot (Powered by Qwen2.5)")
    print("="*60)
    print("Ask a question about your documents (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print(f"🔍 Searching documents...")
        results = vectorstore.similarity_search(query, k=3)
        
        # Prepare context from results
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Create prompt using ChatML format for Qwen
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"🤔 Thinking...")
        
        # Generate response
        outputs = pipe(prompt)
        response = outputs[0]['generated_text']
        
        # Extract just the assistant's response (remove the prompt)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()
            
        print(f"\n🤖 AI: {response}")
        print("-" * 60)
        
       # Optional: Show sources
        #print(f"\nSources used:")
        #for i, doc in enumerate(results, 1):
        #    print(f"[{i}] {doc.page_content[:100]}...")
        #print("-" * 60)

if __name__ == "__main__":
    main()
