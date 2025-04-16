from retriever import load_documents, chunk_text, build_faiss_index

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("✂️ Chunking text...")
    chunks = chunk_text(docs)

    print(f"Total chunks: {len(chunks)}")
    print("Building FAISS index...")
    build_faiss_index(chunks)

    print("✅ Index built and saved!")
