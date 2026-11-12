
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = Path("vector_db/electricity_bills.index")
METADATA_FILE = Path("vector_db/chunks_metadata.json")

TOP_K = 3


def load_vector_database():
    """Load the FAISS index and its corresponding chunk metadata."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {METADATA_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    chunks = metadata["chunks"]

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata counts do not match.")

    if metadata["model_name"] != MODEL_NAME:
        raise ValueError("Embedding model does not match the saved index.")

    return index, chunks


def search_documents(question, index, chunks, model, top_k=TOP_K):
    """Retrieve the most relevant chunks for a user question."""

    if not question or not question.strip():
        raise ValueError("Please enter a valid question.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    number_of_results = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        number_of_results
    )

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position == -1:
            continue

        results.append({
            "score": float(score),
            "content": chunks[position]["content"],
            "metadata": chunks[position]["metadata"]
        })

    return results


def main():
    print("\nELECTRICITY BILL SEMANTIC SEARCH")
    print("=" * 60)

    index, chunks = load_vector_database()

    print("FAISS vectors loaded:", index.ntotal)
    print("Metadata chunks loaded:", len(chunks))

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    question = "How many units of electricity were consumed in May?"

    print("\nUser question:", question)

    results = search_documents(
        question,
        index,
        chunks,
        model
    )

    print("\nTOP MATCHING DOCUMENT CHUNKS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print("Similarity score:", round(result["score"], 4))
        print("Month:", result["metadata"]["month"])
        print("Source:", result["metadata"]["source_file"])
        print("Chunk index:", result["metadata"]["chunk_index"])
        print("Content:", result["content"])
        print("-" * 60)

    print("\nSemantic search completed.")


if __name__ == "__main__":
    main()
