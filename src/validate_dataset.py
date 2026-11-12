
import json
from pathlib import Path

file_path = Path("data/processed/historical_bills.json")

with file_path.open("r", encoding="utf-8") as file:
    bills = json.load(file)

required_fields = [
    "month",
    "bill_date",
    "due_date",
    "billed_units",
    "present_reading_kwh",
    "previous_reading_kwh",
    "net_bill_amount",
    "source_file"
]

print("HISTORICAL DATA VALIDATION")
print("-" * 35)

assert len(bills) == 5, "Expected exactly 5 monthly records"

for bill in bills:
    missing = [field for field in required_fields if field not in bill]

    if missing:
        print(f"{bill['month']}: Missing fields {missing}")
        continue

    difference = (
        bill["present_reading_kwh"]
        - bill["previous_reading_kwh"]
    )

    status = "PASS" if difference == bill["billed_units"] else "CHECK"

    print(
        f"{bill['month']}: {status} | "
        f"Units: {bill['billed_units']} | "
        f"Reading difference: {difference}"
    )

print("-" * 35)
print("Total monthly records:", len(bills))
