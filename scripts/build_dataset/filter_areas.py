import json

import os

KEEP_AREAS = [
    "Biology",
    "Artificial Intelligence",
    "Medicine",
    "Neuroscience",
    "Physics",
    "Materials Science",
    "Environmental Science",
    "Chemistry",
]

input_path = os.path.join(os.path.dirname(__file__), "all_data", "FutureScience.jsonl")

output_dir = os.path.join(os.path.dirname(__file__), "filtered")

output_path = os.path.join(output_dir, "filtered_alreas_futurescience.jsonl")

os.makedirs(output_dir, exist_ok=True)

kept = 0

total = 0

with open(input_path, "r") as infile, open(output_path, "w") as outfile:

    for line in infile:

        line = line.strip()

        if not line:

            continue

        total += 1

        entry = json.loads(line)

        if entry.get("main_area") in KEEP_AREAS:

            outfile.write(json.dumps(entry) + "\n")

            kept += 1

print(f"Done. Kept {kept}/{total} entries -> {output_path}")
