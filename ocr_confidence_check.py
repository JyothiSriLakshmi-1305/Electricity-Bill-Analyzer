
import pymupdf
import pytesseract
from PIL import Image, ImageEnhance
from pytesseract import Output
from pathlib import Path

PDF_FOLDER = Path("data/historical")

print("\nOCR CONFIDENCE REPORT")
print("=" * 45)

for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
    if pdf_path.stem.upper() not in {
        "APRIL", "MAY", "JUNE", "JULY", "AUG"
    }:
        continue

    doc = pymupdf.open(pdf_path)
    all_confidences = []
    total_words = 0

    for page in doc:
        pix = page.get_pixmap(dpi=300, alpha=False)
        image = Image.frombytes(
            "RGB", [pix.width, pix.height], pix.samples
        )

        image = image.convert("L")
        image = ImageEnhance.Contrast(image).enhance(1.5)

        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--oem 3 --psm 6",
            output_type=Output.DICT
        )

        for word, confidence in zip(
            data["text"], data["conf"]
        ):
            word = word.strip()

            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                continue

            if word and confidence >= 0:
                all_confidences.append(confidence)
                total_words += 1

    if all_confidences:
        average = sum(all_confidences) / len(all_confidences)
        print(f"\n{pdf_path.name}")
        print(f"Recognized words: {total_words}")
        print(f"Average OCR confidence: {average:.2f}%")
    else:
        print(f"\n{pdf_path.name}: No words recognized")

    doc.close()
