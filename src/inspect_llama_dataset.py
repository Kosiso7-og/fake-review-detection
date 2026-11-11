
from pathlib import Path
from collections import Counter
import zipfile
import csv
import hashlib

DATA_DIR = Path("data/raw/op_spam_ai")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

records = []

for sentiment in ["negative", "positive"]:
    zip_path = DATA_DIR / f"generated_LLama_{sentiment}.zip"

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt"):
                continue

            if name.startswith("__MACOSX/") or "/._" in name:
                continue

            text = archive.read(name).decode(
                "utf-8-sig", errors="replace"
            ).strip()

            records.append({
                "filename": Path(name).name,
                "sentiment": sentiment,
                "generator": "Llama",
                "review_text": text,
                "character_count": len(text),
                "word_count": len(text.split()),
                "is_empty": not bool(text),
                "text_hash": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest() if text else ""
            })

print("LLAMA DATASET AUDIT")
print("=" * 45)
print("Total files:", len(records))

nonempty = [r for r in records if not r["is_empty"]]
print("Nonempty reviews:", len(nonempty))
print("Empty reviews:", len(records) - len(nonempty))

counts = Counter(r["review_text"] for r in nonempty)
duplicate_groups = {
    text: count
    for text, count in counts.items()
    if count > 1
}

print("Unique nonempty texts:", len(counts))
print("Duplicate nonempty groups:", len(duplicate_groups))
print("Extra duplicate copies:", sum(
    count - 1 for count in duplicate_groups.values()
))

for sentiment in ["negative", "positive"]:
    subset = [r for r in records if r["sentiment"] == sentiment]
    usable = [r for r in subset if not r["is_empty"]]

    print(f"\n{sentiment.upper()}")
    print("Total:", len(subset))
    print("Nonempty:", len(usable))
    print("Empty:", len(subset) - len(usable))

output_path = RESULTS_DIR / "llama_dataset_audit.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=records[0].keys()
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nAudit saved to: {output_path}")
