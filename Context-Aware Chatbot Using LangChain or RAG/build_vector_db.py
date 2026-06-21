"""
build_vector_db.py
==================
One-time script to build the FAISS vector store from text documents.

Usage:
    python build_vector_db.py

This script:
1. Loads all .txt files from the data/ directory
2. Splits them into chunks using RecursiveCharacterTextSplitter
3. Creates embeddings using HuggingFace's all-MiniLM-L6-v2 model (runs locally)
4. Saves the FAISS index to faiss_index/ directory
"""

import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(data_dir: str = "data"):
    """Load all .txt files from the data directory."""
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in '{data_dir}/' directory. "
            "Please add text documents to the data/ folder."
        )

    print(f"Found {len(txt_files)} document(s) in '{data_dir}/':")

    for file_path in txt_files:
        print(f"  [Doc] Loading: {os.path.basename(file_path)}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"\n[Split] Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vector_store(chunks, save_dir="faiss_index"):
    """Create FAISS vector store from document chunks."""
    print("\n[Embed] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    print("   (This may take a moment on first run as the model downloads ~80MB)")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[DB] Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embedding)

    # Save to disk
    db.save_local(save_dir)
    print(f"\n[Success] Vector database saved to '{save_dir}/'")
    print(f"   Files created: {save_dir}/index.faiss, {save_dir}/index.pkl")

    return db


def main():
    print("=" * 60)
    print("  FAISS Vector Database Builder")
    print("  Context-Aware RAG Chatbot - Task 4")
    print("=" * 60)
    print()

    # Step 1: Load documents
    documents = load_documents("data")

    # Step 2: Split into chunks
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    # Step 3: Create and save vector store
    db = create_vector_store(chunks, "faiss_index")

    # Step 4: Quick test - verify the vector store works
    print("\n[Test] Quick Test - Searching for 'What is machine learning?'")
    results = db.similarity_search("What is machine learning?", k=2)
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"   Result {i+1} [{source}]: {preview}...")

    print("\n" + "=" * 60)
    print("  [Success] Vector database is ready!")
    print("  Run 'streamlit run app.py' to start the chatbot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
