
import json
from pathlib import Path
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("data/processed/extracted_bills_draft.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PREVIEW_LENGTH = 180


# --------------------------------------------------
# LOAD EXTRACTED BILL DATA
# --------------------------------------------------

def load_bill_data():
    """Load extracted electricity bill records."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        bills = json.load(file)

    if not isinstance(bills, list):
        raise ValueError("Expected a list of bill records.")

    if not bills:
        raise ValueError("No bill records found.")

    return bills


# --------------------------------------------------
# INITIALIZE TEXT SPLITTER
# --------------------------------------------------

def create_text_splitter():
    """Configure the recursive character text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        strip_whitespace=True
    )


# --------------------------------------------------
# CREATE DOCUMENT CHUNKS
# --------------------------------------------------

def create_document_chunks(bills):
    """Split bill text and attach source metadata."""

    splitter = create_text_splitter()

    all_chunks = []

    for bill in bills:

        raw_text = bill.get("raw_text", "")

        source_file = bill.get("source_file")
        month = bill.get("month")

        if not raw_text or not raw_text.strip():
            print(f"WARNING: Empty text in {source_file}")
            continue

        if not source_file or not month:
            print("WARNING: Missing source metadata.")
            continue

        text_chunks = splitter.split_text(raw_text)

        for index, text_content in enumerate(text_chunks):

            if not text_content.strip():
                continue

            chunk = {
                "content": text_content,
                "metadata": {
                    "month": month,
                    "source_file": source_file,
                    "chunk_index": index
                }
            }

            all_chunks.append(chunk)

    return all_chunks


# --------------------------------------------------
# INSPECT ALL GENERATED CHUNKS
# --------------------------------------------------

def inspect_chunks(chunks):
    """Display chunk previews and metadata."""

    print("\nALL CHUNKS INSPECTION")
    print("=" * 65)

    for chunk in chunks:

        metadata = chunk["metadata"]
        content = chunk["content"]

        preview = content[:PREVIEW_LENGTH]

        if len(content) > PREVIEW_LENGTH:
            preview += "..."

        print(f"\nMonth: {metadata['month']}")
        print(f"Source: {metadata['source_file']}")
        print(f"Chunk Index: {metadata['chunk_index']}")
        print(f"Character Count: {len(content)}")
        print(f"Content Preview: {preview}")

        print("-" * 65)


# --------------------------------------------------
# GENERATE CHUNKING SUMMARY
# --------------------------------------------------

def generate_summary(chunks, bills):
    """Display chunk statistics and validation results."""

    month_counts = Counter(
        chunk["metadata"]["month"]
        for chunk in chunks
    )

    invalid_chunks = [
        chunk for chunk in chunks
        if (
            not chunk["content"].strip()
            or not chunk["metadata"].get("source_file")
            or not chunk["metadata"].get("month")
        )
    ]

    print("\nCHUNKING SUMMARY")
    print("=" * 65)

    print("Total bills loaded:", len(bills))
    print("Total chunks created:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Empty or invalid chunks:", len(invalid_chunks))

    print("\nChunks per month:")

    for month, count in sorted(month_counts.items()):
        print(f"{month}: {count}")

    print("\nAverage chunk length:")

    if chunks:
        average_length = sum(
            len(chunk["content"]) for chunk in chunks
        ) / len(chunks)

        print(f"{average_length:.2f} characters")
    else:
        print("No chunks generated.")

    print("\nValidation:")

    if chunks and not invalid_chunks:
        print("PASS: All generated chunks have content and metadata.")
    else:
        print("WARNING: Check empty or invalid chunks.")

    print("\nDocument chunking completed.")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nELECTRICITY BILL DOCUMENT CHUNKING")
    print("=" * 65)

    bills = load_bill_data()

    print("Historical bills loaded:", len(bills))

    chunks = create_document_chunks(bills)

    if not chunks:
        print("No chunks generated. Check the source dataset.")
        return

    inspect_chunks(chunks)

    generate_summary(chunks, bills)


if __name__ == "__main__":
    main()
