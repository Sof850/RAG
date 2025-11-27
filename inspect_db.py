from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"

def inspect_database():
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
    
    # Get collection
    collection = vectorstore._collection
    
    print(f"\n{'='*70}")
    print(f"📊 CHROMADB DATABASE INSPECTION")
    print(f"{'='*70}\n")
    
    # Database info
    print(f"📁 Database Path: {os.path.abspath(CHROMA_PATH)}")
    print(f"📦 Collection Name: {collection.name}")
    print(f"📝 Total Documents: {collection.count()}")
    
    # Get all data (limit to first 10 for display)
    print(f"\n{'='*70}")
    print(f"📄 SAMPLE DOCUMENTS (showing first 10)")
    print(f"{'='*70}\n")
    
    # Fetch first 10 documents
    results = collection.get(limit=10, include=['documents', 'metadatas'])
    
    for i, (doc_id, document, metadata) in enumerate(zip(
        results['ids'], 
        results['documents'], 
        results['metadatas']
    ), 1):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc_id}")
        print(f"Content Preview: {document[:200]}...")
        print(f"Metadata: {metadata}")
        print(f"Content Length: {len(document)} characters")
    
    print(f"\n{'='*70}")
    print(f"💡 TIP: Use query_db.py to search the database interactively!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    inspect_database()
