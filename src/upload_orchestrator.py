"""Coordinate the complete electricity-bill upload workflow."""

import os
import tempfile
from pathlib import Path

from config import EXTRACTED_DATA_FILE
from src.upload_pipeline import process_uploaded_pdf
from src.dataset_manager import append_bill_record
from src.vector_updater import (
    add_bill_to_faiss,
    INDEX_FILE,
    METADATA_FILE,
)
from src.rag_pipeline import refresh_rag_resources


class UploadOrchestrationError(RuntimeError):
    """Raised when the complete upload workflow cannot finish safely."""


def _restore_file(path, original_bytes):
    """Restore a file atomically from its original bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".rollback",
        delete=False,
    ) as temp_file:
        temp_file.write(original_bytes)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)

    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def upload_and_index_bill(uploaded_file):
    """Process, validate, save, and index one uploaded electricity bill."""

    processed = None
    dataset_updated = False

    dataset_path = Path(EXTRACTED_DATA_FILE)

    # Capture the original state before modifying any persistent data.
    try:
        original_dataset = dataset_path.read_bytes()
        original_index = Path(INDEX_FILE).read_bytes()
        original_metadata = Path(METADATA_FILE).read_bytes()
    except OSError as exc:
        raise UploadOrchestrationError(
            "Could not access the existing dataset or vector database. "
            "No upload was processed."
        ) from exc

    try:
        # Stage 1: Extract and validate the uploaded PDF.
        processed = process_uploaded_pdf(uploaded_file)
        bill_record = processed["record"]

        # Stage 2: Add the validated bill to the structured dataset.
        dataset_result = append_bill_record(bill_record)
        dataset_updated = True

        # Stage 3: Add only the new bill's chunks to FAISS.
        vector_result = add_bill_to_faiss(bill_record)

        # Stage 4: Clear cached data so future questions use the new bill.
        refresh_rag_resources()

        return {
            "success": True,
            "record": bill_record,
            "dataset": dataset_result,
            "vector": vector_result,
            "stored_path": processed["stored_path"],
            "original_filename": processed["original_filename"],
        }

    except Exception as exc:
        rollback_errors = []

        # Restore the JSON dataset if it was already updated.
        if dataset_updated:
            try:
                _restore_file(dataset_path, original_dataset)
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"Dataset rollback failed: {rollback_exc}"
                )

        # Restore both FAISS files to their original state.
        for path, original_bytes in (
            (Path(INDEX_FILE), original_index),
            (Path(METADATA_FILE), original_metadata),
        ):
            try:
                _restore_file(path, original_bytes)
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"Vector database rollback failed for "
                    f"{path.name}: {rollback_exc}"
                )

        # Remove the uploaded PDF if the workflow did not complete.
        if processed:
            try:
                Path(processed["stored_path"]).unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"Uploaded PDF cleanup failed: {rollback_exc}"
                )

        # Ensure future requests do not use stale cached resources.
        try:
            refresh_rag_resources()
        except Exception as rollback_exc:
            rollback_errors.append(
                f"RAG cache refresh failed: {rollback_exc}"
            )

        if rollback_errors:
            raise UploadOrchestrationError(
                f"Upload failed: {type(exc).__name__}: {exc}. "
                + " | ".join(rollback_errors)
            ) from exc

        raise UploadOrchestrationError(
            f"Upload was not completed. "
            f"{type(exc).__name__}: {exc}. "
            "The previous dataset and vector database were restored."
        ) from exc