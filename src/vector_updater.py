"""Incrementally add one bill's chunks to the existing FAISS database.

This module preserves the current index and appends only new vectors.
It uses the same embedding model, normalization, chunk format and metadata
schema as src/embed_documents.py and src/retriever.py.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_chunker import create_document_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_FILE = VECTOR_DB_DIR / "electricity_bills.index"
METADATA_FILE = VECTOR_DB_DIR / "chunks_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class VectorUpdateError(RuntimeError):
    """Raised when an incremental FAISS update cannot be completed."""


def _read_metadata():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {METADATA_FILE}")

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorUpdateError("Could not read FAISS metadata.") from exc

    if metadata.get("model_name") != MODEL_NAME:
        raise VectorUpdateError(
            "Saved FAISS database uses a different embedding model."
        )

    if metadata.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise VectorUpdateError(
            "Saved FAISS metadata has an unexpected embedding dimension."
        )

    if not isinstance(metadata.get("chunks"), list):
        raise VectorUpdateError("FAISS metadata has an invalid chunks list.")

    return metadata


def _atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(payload, temp_file, indent=4, ensure_ascii=False)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        return Path(temp_file.name)


def add_bill_to_faiss(bill_record):
    """Create chunks and append their normalized embeddings to existing FAISS.

    The bill record must already be present in the structured JSON dataset.
    This function updates only the FAISS index and its chunk metadata.
    """
    if not isinstance(bill_record, dict):
        raise VectorUpdateError("A bill record dictionary is required.")

    source_file = str(bill_record.get("source_file", "")).strip()
    month = str(bill_record.get("month", "")).strip()

    if not source_file or not month:
        raise VectorUpdateError("Bill record must contain source_file and month.")

    metadata = _read_metadata()
    existing_chunks = metadata["chunks"]

    if any(
        (chunk.get("metadata") or {}).get("source_file", "").casefold()
        == source_file.casefold()
        for chunk in existing_chunks
    ):
        raise VectorUpdateError(
            f"FAISS already contains chunks for {source_file}."
        )

    index = faiss.read_index(str(INDEX_FILE))

    if index.ntotal != len(existing_chunks):
        raise VectorUpdateError(
            "Existing FAISS vector count does not match metadata count."
        )

    if index.d != EMBEDDING_DIMENSION:
        raise VectorUpdateError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional vectors; "
            f"found {index.d}."
        )

    if not isinstance(index, faiss.IndexFlatIP):
        raise VectorUpdateError(
            "Incremental update currently supports IndexFlatIP only."
        )

    new_chunks = create_document_chunks([bill_record])
    if not new_chunks:
        raise VectorUpdateError(
            "No chunks were generated. The bill's raw_text may be empty."
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = [chunk["content"] for chunk in new_chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(new_chunks), EMBEDDING_DIMENSION):
        raise VectorUpdateError(
            f"Unexpected embedding shape: {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise VectorUpdateError("Generated embeddings contain invalid values.")

    # Work on an in-memory copy so the original on-disk index is untouched
    # until all new chunks and embeddings have passed validation.
    updated_index = faiss.clone_index(index)
    updated_index.add(embeddings)

    updated_chunks = existing_chunks + new_chunks
    if updated_index.ntotal != len(updated_chunks):
        raise VectorUpdateError(
            "Updated FAISS count does not match updated chunk metadata."
        )

    updated_metadata = dict(metadata)
    updated_metadata.update({
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_chunks": len(updated_chunks),
        "similarity_metric": "cosine_similarity",
        "chunks": updated_chunks,
    })

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    temp_index = None
    temp_metadata = None
    index_backup = INDEX_FILE.with_suffix(".index.bak")
    metadata_backup = METADATA_FILE.with_suffix(".json.bak")

    try:
        # Prepare and verify both replacement files before touching originals.
        with tempfile.NamedTemporaryFile(
            dir=VECTOR_DB_DIR,
            prefix=".electricity_bills.",
            suffix=".index.tmp",
            delete=False,
        ) as temp_file:
            temp_index = Path(temp_file.name)

        faiss.write_index(updated_index, str(temp_index))
        check_index = faiss.read_index(str(temp_index))
        if check_index.ntotal != len(updated_chunks):
            raise VectorUpdateError("Temporary FAISS index verification failed.")

        temp_metadata = _atomic_write_json(METADATA_FILE, updated_metadata)
        with temp_metadata.open("r", encoding="utf-8") as file:
            check_metadata = json.load(file)

        if len(check_metadata["chunks"]) != check_index.ntotal:
            raise VectorUpdateError("Temporary metadata verification failed.")

        # Back up both originals. Restore them if either replacement fails.
        shutil.copy2(INDEX_FILE, index_backup)
        shutil.copy2(METADATA_FILE, metadata_backup)

        os.replace(temp_index, INDEX_FILE)
        temp_index = None

        try:
            os.replace(temp_metadata, METADATA_FILE)
            temp_metadata = None
        except Exception:
            shutil.copy2(index_backup, INDEX_FILE)
            raise

    except Exception as exc:
        if isinstance(exc, VectorUpdateError):
            raise
        raise VectorUpdateError(
            "Could not safely save the updated FAISS database. "
            "The previous database has been preserved where possible."
        ) from exc
    finally:
        for temp_path in (temp_index, temp_metadata):
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        # Keep .bak files as recovery copies after a successful update.
        # Remove incomplete backup files only if an original backup is absent.
        if not index_backup.exists() or not metadata_backup.exists():
            pass

    return {
        "source_file": source_file,
        "month": month,
        "new_chunks": len(new_chunks),
        "previous_vectors": index.ntotal,
        "total_vectors": updated_index.ntotal,
        "index_path": str(INDEX_FILE),
        "metadata_path": str(METADATA_FILE),
        "backup_index": str(index_backup),
        "backup_metadata": str(metadata_backup),
    }
