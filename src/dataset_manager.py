"""Safe updates to the extracted electricity-bill dataset."""

import json
import os
import tempfile
from pathlib import Path

from config import EXTRACTED_DATA_FILE


IDENTITY_FIELDS = (
    "bill_date",
    "billed_units",
    "present_reading_kwh",
    "previous_reading_kwh",
    "net_bill_amount",
)


class DatasetUpdateError(ValueError):
    """Raised when a bill cannot safely be added to the dataset."""


def _load_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Bill dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise DatasetUpdateError("The bill dataset must be a JSON list.")

    return records


def _is_same_bill(existing, incoming):
    return all(
        existing.get(field) is not None
        and incoming.get(field) is not None
        and str(existing[field]) == str(incoming[field])
        for field in IDENTITY_FIELDS
    )


def append_bill_record(record, dataset_path=None):
    """Append one validated bill record using an atomic JSON replacement.

    The caller must pass the record returned by process_uploaded_pdf().
    This function updates only the structured JSON dataset; it does not
    update FAISS or modify historical_bills.json.
    """
    if not isinstance(record, dict):
        raise DatasetUpdateError("A valid bill record is required.")

    required = (
        "source_file",
        "month",
        "bill_date",
        "due_date",
        "billed_units",
        "present_reading_kwh",
        "previous_reading_kwh",
        "energy_charges",
        "fixed_charges",
        "customer_charges",
        "net_bill_amount",
        "raw_text",
    )
    missing = [key for key in required if record.get(key) is None]
    if missing:
        raise DatasetUpdateError(
            "Bill record is missing required fields: " + ", ".join(missing)
        )

    path = Path(dataset_path) if dataset_path else Path(EXTRACTED_DATA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(path)

    incoming_source = str(record["source_file"]).strip().casefold()
    for existing in records:
        existing_source = str(existing.get("source_file", "")).strip().casefold()

        if existing_source and existing_source == incoming_source:
            raise DatasetUpdateError(
                f"A record with source filename {record['source_file']!r} "
                "already exists."
            )

        if _is_same_bill(existing, record):
            raise DatasetUpdateError(
                "A bill with the same date, readings, units and net amount "
                "already exists in the dataset."
            )

    updated_records = records + [record]

    # Write to a temporary file in the same directory, then atomically replace.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(
                updated_records,
                temp_file,
                indent=4,
                ensure_ascii=False,
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        # Verify the temporary JSON before replacing the live dataset.
        with temp_path.open("r", encoding="utf-8") as check_file:
            verified = json.load(check_file)

        if len(verified) != len(records) + 1:
            raise DatasetUpdateError("Dataset verification failed.")

        os.replace(temp_path, path)
        temp_path = None

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return {
        "total_records": len(updated_records),
        "added_record": record["source_file"],
        "dataset_path": str(path),
    }
