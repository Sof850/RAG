# RAG Document Processor with ChromaDB

This project loads documents, splits them into chunks, and creates a ChromaDB vector store using **local HuggingFace embeddings** (free, no API key required) for Retrieval-Augmented Generation (RAG).

## Setup

1. **Install dependencies**:
   ```bash
   source .venv/bin/activate
   pip install langchain-community langchain-text-splitters langchain-huggingface langchain-chroma chromadb python-dotenv sentence-transformers
   ```

2. **No API Key Required!**
   - We are using the `all-MiniLM-L6-v2` model which runs locally on your machine.
   - It's fast, free, and effective for English text.

## Usage

Run the script to process documents and create the vector database:

```bash
.venv/bin/python data_cutter.py
```

## What it does

1. **Loads documents** from the `data/` directory (currently processes `*.txt` files)
2. **Splits documents** into chunks:
   - Chunk size: 1000 characters
   - Chunk overlap: 500 characters (for context preservation)
3. **Creates embeddings** using the local `all-MiniLM-L6-v2` model
4. **Stores in ChromaDB** at `chroma_db/` directory
5. **Tests the database** with a sample similarity search

## Accessing the Database

Once you've created the database, you have two ways to explore it:

### 1. **Interactive Query Interface** (Recommended)
```bash
.venv/bin/python query_db.py
```
This opens an interactive interface where you can:
- Search the database with natural language queries
- See the top 5 most relevant chunks
- View similarity scores
- Get database statistics

### 2. **Database Inspection**
```bash
.venv/bin/python inspect_db.py
```
This shows you:
- Total number of documents
- Sample documents with metadata
- Database location and structure

## Files

- `data_cutter.py` - Main script to create the database
- `query_db.py` - Interactive query interface
- `inspect_db.py` - Database inspection tool
- `data/` - Directory containing your text documents
- `chroma_db/` - ChromaDB vector store (created after running data_cutter.py)

## Database Location

The ChromaDB database is stored locally at:
```
/home/user/Documents/RAG/chroma_db/
```

It's a directory containing:
- SQLite database files
- Vector embeddings
- Metadata

You can access it programmatically using the scripts above, or by loading it in your own Python code:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Now you can query it
results = vectorstore.similarity_search("your query here", k=5)
```
