"""Validation and structured extraction for newly uploaded electricity bills.

This module is the first stage of the dynamic upload workflow.
It does NOT modify the historical dataset or FAISS database.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from config import (
    UPLOAD_DIR,
    EXTRACTED_DATA_FILE,
    MAX_UPLOAD_SIZE_MB,
)
from src.bill_extractor import extract_bill_data


REQUIRED_FIELDS = (
    "bill_date",
    "due_date",
    "billed_units",
    "present_reading_kwh",
    "previous_reading_kwh",
    "energy_charges",
    "fixed_charges",
    "customer_charges",
    "net_bill_amount",
)

DATE_FORMATS = ("%d-%b-%Y", "%d/%b/%Y", "%d-%m-%Y", "%d/%m/%Y")


class UploadValidationError(ValueError):
    """Raised when an uploaded PDF cannot safely be accepted."""


def _parse_bill_date(value):
    if not value:
        raise UploadValidationError("Bill date could not be extracted.")

    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            continue

    raise UploadValidationError(
        f"Unsupported bill date format: {value!r}."
    )


def _read_existing_records():
    data_path = Path(EXTRACTED_DATA_FILE)

    if not data_path.exists():
        return []

    try:
        with data_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise UploadValidationError(
            "The existing extracted-bills dataset could not be read. "
            "No upload was saved."
        ) from exc

    if not isinstance(records, list):
        raise UploadValidationError(
            "The existing extracted-bills dataset has an invalid format."
        )

    return records


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _find_duplicate_hash(file_hash):
    """Check existing historical and uploaded PDFs for identical content."""
    project_root = Path(__file__).resolve().parent.parent
    candidate_dirs = (
        project_root / "data" / "historical",
        Path(UPLOAD_DIR),
    )

    for directory in candidate_dirs:
        if not directory.exists():
            continue

        for path in directory.glob("*.pdf"):
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue

            if digest == file_hash:
                return path.name

    return None


def _find_duplicate_bill(new_record, existing_records):
    """Detect likely re-scans using stable bill fields, not filename alone."""
    identity_fields = (
        "bill_date",
        "billed_units",
        "present_reading_kwh",
        "previous_reading_kwh",
        "net_bill_amount",
    )

    for record in existing_records:
        if all(
            record.get(field) is not None
            and new_record.get(field) is not None
            and str(record.get(field)) == str(new_record.get(field))
            for field in identity_fields
        ):
            return record.get("source_file", "an existing bill")

    return None


def _validate_extracted_record(record):
    missing = [
        field for field in REQUIRED_FIELDS
        if record.get(field) is None
    ]

    if missing:
        raise UploadValidationError(
            "The bill could not be extracted completely. "
            "Missing fields: " + ", ".join(missing)
        )

    bill_date = _parse_bill_date(record["bill_date"])
    due_date = _parse_bill_date(record["due_date"])

    if due_date < bill_date:
        raise UploadValidationError(
            "The extracted due date is earlier than the bill date."
        )

    numeric_fields = REQUIRED_FIELDS[2:]
    for field in numeric_fields:
        value = record.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise UploadValidationError(
                f"The extracted value for {field} is invalid."
            )

    units = record["billed_units"]
    reading_difference = (
        record["present_reading_kwh"]
        - record["previous_reading_kwh"]
    )

    if record["present_reading_kwh"] < record["previous_reading_kwh"]:
        raise UploadValidationError(
            "Present meter reading is lower than previous meter reading."
        )

    # Meter readings may be fractional, so use a small numeric tolerance.
    if abs(reading_difference - units) > 0.01:
        raise UploadValidationError(
            "Meter-reading validation failed: present reading minus "
            "previous reading does not match billed units."
        )

    record["month"] = bill_date.strftime("%B")
    return record


def process_uploaded_pdf(uploaded_file):
    """Validate and extract a Streamlit UploadedFile-like PDF.

    Returns a dictionary containing the extracted record, SHA-256 hash,
    and saved PDF path. It does not append to JSON or update FAISS.
    """
    if uploaded_file is None:
        raise UploadValidationError("Please select a PDF file.")

    original_name = Path(getattr(uploaded_file, "name", "upload.pdf")).name

    if Path(original_name).suffix.lower() != ".pdf":
        raise UploadValidationError("Only PDF files are supported.")

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise UploadValidationError("The uploaded file is empty.")

    max_bytes = int(MAX_UPLOAD_SIZE_MB) * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise UploadValidationError(
            f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit."
        )

    if not file_bytes.startswith(b"%PDF-"):
        raise UploadValidationError(
            "The uploaded file does not appear to be a valid PDF."
        )

    file_hash = _sha256(file_bytes)

    duplicate_pdf = _find_duplicate_hash(file_hash)
    if duplicate_pdf:
        raise UploadValidationError(
            f"This PDF has already been uploaded or exists in the "
            f"historical dataset as {duplicate_pdf}."
        )

    existing_records = _read_existing_records()

    # Use a temporary file for the existing path-based PDF extractor.
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(original_name).stem[:60] or "electricity_bill"
    stored_name = f"{safe_stem}_{file_hash[:12]}.pdf"
    final_path = upload_dir / stored_name
    temporary_path = upload_dir / f".{stored_name}.tmp.pdf"

    try:
        with temporary_path.open("wb") as file:
            file.write(file_bytes)
            file.flush()
            os.fsync(file.fileno())

        extracted = extract_bill_data(str(temporary_path))
        extracted["source_file"] = original_name
        extracted = _validate_extracted_record(extracted)

        duplicate_bill = _find_duplicate_bill(extracted, existing_records)
        if duplicate_bill:
            raise UploadValidationError(
                f"This appears to be the same bill already present as "
                f"{duplicate_bill}."
            )

        # Keep the validated PDF for the next processing stage.
        os.replace(temporary_path, final_path)

    except UploadValidationError:
        temporary_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise UploadValidationError(
            "PDF extraction failed. Check the PDF quality and try again."
        ) from exc

    return {
        "record": extracted,
        "file_hash": file_hash,
        "stored_path": str(final_path),
        "original_filename": original_name,
    }
