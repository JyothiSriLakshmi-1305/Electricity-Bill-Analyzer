
import pymupdf
import pytesseract

from PIL import Image, ImageEnhance
from pytesseract import Output
from pathlib import Path


def process_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    doc = pymupdf.open(pdf_path)

    results = {
        "source_file": pdf_path.name,
        "total_pages": len(doc),
        "pages": [],
        "extraction_method": None
    }

    methods = set()

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        confidence = None
        method = "digital_text"

        if not text:
            method = "ocr"

            pix = page.get_pixmap(dpi=300, alpha=False)

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            image = image.convert("L")
            image = ImageEnhance.Contrast(image).enhance(1.5)

            data = pytesseract.image_to_data(
                image,
                lang="eng",
                config="--oem 3 --psm 6",
                output_type=Output.DICT
            )

            words = []
            confidence_values = []

            for word, conf in zip(data["text"], data["conf"]):
                word = word.strip()

                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    continue

                if word and conf >= 0:
                    words.append(word)
                    confidence_values.append(conf)

            text = " ".join(words)

            if confidence_values:
                confidence = round(
                    sum(confidence_values) / len(confidence_values), 2
                )

        methods.add(method)

        results["pages"].append({
            "page_number": page_number,
            "text": text,
            "extraction_method": method,
            "ocr_confidence": confidence
        })

    doc.close()

    if len(methods) == 1:
        results["extraction_method"] = methods.pop()
    else:
        results["extraction_method"] = "mixed"

    return results


if __name__ == "__main__":
    pdf = Path("data/historical/MAY.pdf")

    result = process_pdf(pdf)

    print("\nPDF PROCESSING REPORT")
    print("=" * 40)
    print("File:", result["source_file"])
    print("Pages:", result["total_pages"])
    print("Method:", result["extraction_method"])

    for page in result["pages"]:
        print("\nPage:", page["page_number"])
        print("Method:", page["extraction_method"])
        print("OCR confidence:", page["ocr_confidence"])
        print("Extracted text preview:")
        print(page["text"][:1000])

    print("\nPDF processing completed.")
