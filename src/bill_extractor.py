
import re
from pathlib import Path

from src.pdf_processor import process_pdf


def extract_number(text, label_pattern):
    """Extract a numeric value appearing after a field label."""

    pattern = (
        label_pattern
        + r"\s*[:\-]?\s*([0-9,]+(?:\.[0-9]{1,2})?)"
    )

    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return None

    return float(match.group(1).replace(",", ""))


def extract_bill_data(pdf_path):
    """Extract structured electricity bill fields from a PDF."""

    result = process_pdf(pdf_path)

    # Preserve all extracted page text.
    full_text = "\n".join(
        page["text"] for page in result["pages"]
    )

    # Extract bill date and due date from the bill summary section.
    bill_section = full_text.split("Meter Readings")[0]

    dates = re.findall(
        r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{4}\b",
        bill_section,
        re.IGNORECASE
    )

    bill_date = dates[-2] if len(dates) >= 2 else None
    due_date = dates[-1] if len(dates) >= 2 else None

    bill_data = {
        "source_file": Path(pdf_path).name,
        "extraction_method": result["extraction_method"],
        "ocr_confidence": next(
            (
                page["ocr_confidence"]
                for page in result["pages"]
                if page["ocr_confidence"] is not None
            ),
            None
        ),
        "bill_date": bill_date,
        "due_date": due_date,
        "billed_units": extract_number(
            full_text, r"Import\s*Units"
        ),
        "present_reading_kwh": extract_number(
            full_text, r"Present\s*Reading\s*\(kWh\)"
        ),
        "previous_reading_kwh": extract_number(
            full_text, r"Previous\s*Reading\s*\(kWh\)"
        ),
        "energy_charges": extract_number(
            full_text, r"Energy\s*Charges"
        ),
        "fixed_charges": extract_number(
            full_text, r"Fixed\s*Charges"
        ),
        "customer_charges": extract_number(
            full_text, r"Customer\s*Charges"
        ),
        "net_bill_amount": extract_number(
            full_text, r"Net\s*Bill\s*Amount"
        ),


        "raw_text": full_text
    }

    return bill_data


if __name__ == "__main__":

    pdf_path = Path("data/historical/AUG.pdf")

    bill = extract_bill_data(pdf_path)

    print("\nSTRUCTURED BILL EXTRACTION")
    print("=" * 45)

    for key, value in bill.items():
        if key != "raw_text":
            print(f"{key}: {value}")

    print("\nExtraction completed.")
