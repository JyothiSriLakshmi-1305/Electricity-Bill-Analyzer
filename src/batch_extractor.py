
import json
from pathlib import Path

from src.bill_extractor import extract_bill_data


# Project directories
INPUT_DIR = Path("data/historical")
OUTPUT_FILE = Path("data/processed/extracted_bills_draft.json")


# Normalize month names based on PDF filenames
MONTH_NAMES = {
    "APRIL": "April",
    "MAY": "May",
    "JUNE": "June",
    "JULY": "July",
    "AUG": "August"
}


def batch_extract():

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:", INPUT_DIR)
        return

    extracted_bills = []
    failed_files = []

    print("\nBATCH PDF EXTRACTION")
    print("=" * 50)

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:
            bill_data = extract_bill_data(pdf_path)

            # Add normalized month name based on the PDF filename.
            bill_data["month"] = MONTH_NAMES.get(
                pdf_path.stem.upper(),
                pdf_path.stem.title()
            )

            extracted_bills.append(bill_data)

            print("Status: SUCCESS")
            print("Month:", bill_data["month"])
            print("Bill date:", bill_data["bill_date"])
            print("Due date:", bill_data["due_date"])
            print("Units:", bill_data["billed_units"])
            print("Net bill amount:", bill_data["net_bill_amount"])
            print("OCR confidence:", bill_data["ocr_confidence"])

        except Exception as error:
            failed_files.append(pdf_path.name)
            print("Status: FAILED")
            print("Error:", error)

    # Save results separately; do not overwrite verified data.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            extracted_bills,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("BATCH EXTRACTION SUMMARY")
    print("Total PDFs:", len(pdf_files))
    print("Successful:", len(extracted_bills))
    print("Failed:", len(failed_files))
    print("Output file:", OUTPUT_FILE)

    if failed_files:
        print("Failed files:", failed_files)

    print("\nBatch extraction completed.")


if __name__ == "__main__":
    batch_extract()
