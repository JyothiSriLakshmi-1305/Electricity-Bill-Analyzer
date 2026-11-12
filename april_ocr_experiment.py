"""
April APEPDCL Bill - OCR Experiment
---------------------------------
Purpose:
1. Render each PDF page at 300 DPI.
2. Preprocess the page for OCR.
3. Extract text using Tesseract.
4. Identify a few important bill fields for manual verification.

Install:
    pip install pymupdf pillow pytesseract

Also install the Tesseract OCR application on Windows and ensure
`tesseract.exe` is available on PATH.
"""

import re
import sys
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_page(page, dpi=300):
    """Render a PDF page and prepare a clean image for OCR."""
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(1.6)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda pixel: 255 if pixel > 165 else 0)

    return image


def extract_candidate_fields(text):
    """Extract candidate values; all values must be checked against the bill."""
    patterns = {
        "net_bill_amount": r"Net Bill Amount\s*([0-9,]+\.\d{2})",
        "present_reading_kwh": r"Present Reading \(kWh\)\s*([0-9,]+)",
        "previous_reading_kwh": r"Previous Reading \(kWh\)\s*([0-9,]+)",
        "energy_charges": r"Energy Charges\s*([0-9,]+\.\d{2})",
        "fixed_charges": r"Fixed Charges\s*([0-9,]+\.\d{2})",
        "customer_charges": r"Customer Charges\s*([0-9,]+\.\d{2})",
    }

    results = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        results[field] = match.group(1) if match else None

    # Bill headers are arranged in columns, so OCR often puts all four
    # labels on one line and their values on the next line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if all(label in line.lower() for label in
               ("bill date", "billed units", "due date")):
            value_line = lines[index + 1]
            date_matches = list(re.finditer(r"[0-9]{2}-[A-Za-z]{3}-[0-9]{4}", value_line))
            results["bill_date"] = date_matches[0].group(0) if len(date_matches) >= 2 else None
            results["due_date"] = date_matches[-1].group(0) if len(date_matches) >= 2 else None
            if len(date_matches) >= 2:
                between_dates = value_line[date_matches[0].end():date_matches[1].start()]
                unit_match = re.search(r"\b[0-9]{1,5}\b", between_dates)
                results["billed_units"] = unit_match.group(0) if unit_match else None
            else:
                results["billed_units"] = None
            break
    return results


def run_ocr(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    all_text = []

    print(f"File: {pdf_path.name}")
    print(f"Pages: {len(document)}")
    print("OCR settings: English | 300 DPI | PSM 6")
    print("-" * 55)

    for page_number, page in enumerate(document, start=1):
        image = preprocess_page(page, dpi=300)
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )
        all_text.append(text)
        print(f"\n--- PAGE {page_number} OCR TEXT ---\n")
        print(text)

    combined_text = "\n".join(all_text)
    fields = extract_candidate_fields(combined_text)

    print("\n--- CANDIDATE FIELD EXTRACTION ---")
    for field, value in fields.items():
        print(f"{field}: {value if value is not None else 'NOT FOUND'}")

    print("\nIMPORTANT: OCR output is not trusted bill data until manually validated.")
    document.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python april_ocr_experiment.py APRIL.pdf")
        raise SystemExit(1)
    run_ocr(sys.argv[1])
