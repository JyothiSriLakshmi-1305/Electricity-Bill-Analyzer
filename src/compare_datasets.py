
import json
from pathlib import Path


# Dataset file paths
DRAFT_FILE = Path("data/processed/extracted_bills_draft.json")
REFERENCE_FILE = Path("data/processed/historical_bills.json")


# Fields to compare
FIELDS_TO_COMPARE = [
    "bill_date",
    "due_date",
    "billed_units",
    "present_reading_kwh",
    "previous_reading_kwh",
    "energy_charges",
    "fixed_charges",
    "customer_charges",
    "net_bill_amount"
]


# Numeric fields
NUMERIC_FIELDS = {
    "billed_units",
    "present_reading_kwh",
    "previous_reading_kwh",
    "energy_charges",
    "fixed_charges",
    "customer_charges",
    "net_bill_amount"
}

TOLERANCE = 0.01


def load_json(file_path):
    """Load and return JSON data."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_values(field, extracted, reference):
    """Compare two values with numeric tolerance."""

    if extracted is None or reference is None:
        return extracted == reference

    if field in NUMERIC_FIELDS:
        try:
            return abs(
                float(extracted) - float(reference)
            ) < TOLERANCE
        except (ValueError, TypeError):
            return False

    return extracted == reference


def create_record_map(records, dataset_name):
    """Create a mapping using source filenames."""

    record_map = {}

    for record in records:
        source_file = record.get("source_file")

        if not source_file:
            print(
                f"WARNING: Missing source_file in {dataset_name}"
            )
            continue

        if source_file in record_map:
            print(
                f"WARNING: Duplicate record: {source_file} "
                f"in {dataset_name}"
            )
            continue

        record_map[source_file] = record

    return record_map


def compare_datasets():

    print("\nELECTRICITY BILL DATASET VALIDATION")
    print("=" * 55)

    extracted_records = load_json(DRAFT_FILE)
    reference_records = load_json(REFERENCE_FILE)

    if not isinstance(extracted_records, list):
        raise ValueError("Draft dataset must contain a JSON list.")

    if not isinstance(reference_records, list):
        raise ValueError("Reference dataset must contain a JSON list.")

    extracted_map = create_record_map(
        extracted_records, "draft dataset"
    )

    reference_map = create_record_map(
        reference_records, "reference dataset"
    )

    extracted_files = set(extracted_map)
    reference_files = set(reference_map)

    print("\nDATASET SUMMARY")
    print("-" * 55)
    print("Extracted records:", len(extracted_records))
    print("Reference records:", len(reference_records))

    missing_files = reference_files - extracted_files
    extra_files = extracted_files - reference_files

    if missing_files:
        print("Missing files:", sorted(missing_files))

    if extra_files:
        print("Unexpected files:", sorted(extra_files))

    total_matches = 0
    total_mismatches = 0

    print("\nFIELD-BY-FIELD COMPARISON")
    print("=" * 55)

    for source_file in sorted(extracted_files & reference_files):

        extracted = extracted_map[source_file]
        reference = reference_map[source_file]

        print(f"\nBill: {source_file}")
        print("-" * 55)

        for field in FIELDS_TO_COMPARE:

            extracted_value = extracted.get(field)
            reference_value = reference.get(field)

            if compare_values(
                field, extracted_value, reference_value
            ):
                total_matches += 1
                print(f"[PASS] {field}")
            else:
                total_mismatches += 1
                print(f"[FAIL] {field}")
                print("       Extracted:", extracted_value)
                print("       Reference:", reference_value)

    print("\n" + "=" * 55)
    print("FINAL VALIDATION REPORT")
    print("=" * 55)

    print("Matching fields:", total_matches)
    print("Mismatched fields:", total_mismatches)
    print("Missing files:", len(missing_files))
    print("Unexpected files:", len(extra_files))

    if (
        total_mismatches == 0
        and not missing_files
        and not extra_files
        and len(extracted_records) == len(reference_records)
        and len(extracted_map) == len(extracted_records)
        and len(reference_map) == len(reference_records)
    ):
        print("\nRESULT: ALL DATASET CHECKS PASSED")

    else:
        print("\nRESULT: VALIDATION NEEDS ATTENTION")


if __name__ == "__main__":
    try:
        compare_datasets()

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print("\nVALIDATION ERROR:", error)
