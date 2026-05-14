"""
Merge MCQ data from a source JSONL into a target JSONL matched by paper_link.
Usage: python merge_benchmark.py <target.jsonl> <source.jsonl> <output.jsonl>
"""

import json

import sys

from pathlib import Path


def merge_mcq_data(target_file, source_file, output_file):

    source_data = {}

    with open(source_file, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            entry = json.loads(line)

            link = entry.get("paper_link")

            if link:

                source_data[link] = {
                    "mcq_question": entry.get("mcq_question"),
                    "mcq_choices": entry.get("mcq_choices"),
                    "mcq_answer_key": entry.get("mcq_answer_key"),
                }

    updated_count = 0

    with open(target_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:

            if not line.strip():
                continue

            entry = json.loads(line)

            link = entry.get("paper_link")

            if link in source_data:

                entry.update(source_data[link])

                updated_count += 1

            f_out.write(json.dumps(entry) + "\n")

    print(f"Done! Processed all entries. Replaced MCQ data for {updated_count} matching links.")


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print("Usage: python merge_benchmark.py <target.jsonl> <source.jsonl> <output.jsonl>")

        sys.exit(1)

    merge_mcq_data(sys.argv[1], sys.argv[2], sys.argv[3])
