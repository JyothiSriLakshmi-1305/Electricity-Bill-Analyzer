
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_chunker import (
    load_bill_data,
    create_document_chunks
)


# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIR = Path("vector_db")
INDEX_FILE = VECTOR_DB_DIR / "electricity_bills.index"
METADATA_FILE = VECTOR_DB_DIR / "chunks_metadata.json"


def generate_embeddings(chunks):
    """Generate normalized embeddings for document chunks."""

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [chunk["content"] for chunk in chunks]

    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Invalid embedding dimensions.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    return embeddings


def create_faiss_index(embeddings):
    """Create a FAISS index using cosine similarity."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_vector_database(index, chunks, embeddings):
    """Save the FAISS index and corresponding chunk metadata."""

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "total_chunks": len(chunks),
        "similarity_metric": "cosine_similarity",
        "chunks": chunks
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nVector database saved successfully.")
    print("FAISS index:", INDEX_FILE)
    print("Chunk metadata:", METADATA_FILE)


def main():
    print("\nELECTRICITY BILL EMBEDDING & FAISS INDEXING")
    print("=" * 60)

    # Step 1: Load historical bills
    bills = load_bill_data()
    print("Historical bills loaded:", len(bills))

    # Step 2: Generate document chunks
    chunks = create_document_chunks(bills)
    print("Document chunks generated:", len(chunks))

    if not chunks:
        raise ValueError("No document chunks available.")

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    print("\nEMBEDDING VALIDATION")
    print("=" * 60)
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])
    print("Number of vectors:", embeddings.shape[0])
    print("All values finite:", np.isfinite(embeddings).all())

    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)

    print("\nFAISS INDEX VALIDATION")
    print("=" * 60)
    print("Index type:", type(index).__name__)
    print("Vectors stored:", index.ntotal)
    print("Vector dimension:", index.d)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index count does not match chunks.")

    # Step 5: Save index and metadata
    save_vector_database(index, chunks, embeddings)

    print("\nRESULT: EMBEDDING AND INDEXING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
