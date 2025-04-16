from sentence_transformers import SentenceTransformer
import faiss
import os
import fitz  # PyMuPDF
import pickle

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "embeddings/index.faiss"
CHUNKS_PATH = "embeddings/chunks.pkl"
DOCS_PATH = "data/documents/"

model = SentenceTransformer(EMBEDDING_MODEL)

def load_documents():
    """
    Load PDF documents from the DOCS_PATH.
    Returns a list of dicts with keys:
    - 'source': a string identifier (filename and page number)
    - 'text': the text extracted from the page
    """
    docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.lower().endswith(".pdf"):
            doc_path = os.path.join(DOCS_PATH, filename)
            doc = fitz.open(doc_path)
            for page_no, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    docs.append({"source": f"{filename} - Page {page_no}", "text": text})
    return docs

def chunk_text(docs):
    """
    Chunk each document and preserve the source info.
    Returns a list of chunks, each is a dict with keys:
    - 'text': the text chunk
    - 'source': the source info from the parent document
    """
    chunks = []
    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i: i + CHUNK_SIZE]
            chunks.append({"text": chunk, "source": source})
    return chunks

def build_faiss_index(chunks):
    """
    Build a FAISS index from the text of the chunks.
    Save the vector index and the mapping to chunks (with sources).
    """
    # Encode using only the text
    embeddings = model.encode([chunk["text"] for chunk in chunks])
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_PATH)
    print("✅ FAISS index created and saved.")

def load_faiss_index():
    """
    Load the FAISS index and corresponding chunks mapping.
    """
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_relevant_chunks(question, top_k=3):
    """
    Return the top_k most relevant chunks (with their source info) based on the question.
    """
    index, chunks = load_faiss_index()
    q_embed = model.encode([question])
    distances, indices = index.search(q_embed, top_k)
    # Return list of dictionaries containing text and source for each retrieved chunk.
    return [chunks[i] for i in indices[0]]
