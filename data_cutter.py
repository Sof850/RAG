from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

def create_db():
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function= len,
        add_start_index= True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} document(s)")
    print(f"Split into {len(chunks)} chunks")

    # Clear existing database if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"\nClearing existing database at {CHROMA_PATH}...")
        import shutil
        shutil.rmtree(CHROMA_PATH)

    # Create embeddings and vector store
    print(f"\nCreating ChromaDB vector store with HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store from documents
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Successfully created ChromaDB with {len(chunks)} chunks!")
    print(f"📁 Database saved to: {CHROMA_PATH}")
    return vectorstore

if __name__ == "__main__":
    vectorstore = create_db()
    
    # Test the vector store with a simple query
    print(f"\n🔍 Testing vector store with a sample query...")
    test_results = vectorstore.similarity_search("Alice", k=3)
    print(f"Found {len(test_results)} relevant chunks for query 'Alice'")
    print(f"\nFirst result preview:")
    print(f"{test_results[0].page_content[:200]}..." if test_results else "No results")