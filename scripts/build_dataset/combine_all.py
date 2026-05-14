"""
combine_all.py — Merge all categorized JSONL benchmark files into one.

Combines categorized JSONL files from:
  cell, hugging_face_top_ai_papers, leaderboard_questions,
  nature, science, top_ai_papers

into a single FutureScience.jsonl file. Adds a 'source' field to each record
to identify which dataset it came from.

Usage
-----
  python combine_all.py
  python combine_all.py -o /custom/output/path/FutureScience.jsonl
"""

import json

import argparse

from pathlib import Path

SOURCES = [
    ("cell", "cell/cell_filtered_domain_with_dates_benchmark_all_categorized.jsonl"),
    (
        "hugging_face_top_ai_papers",
        "hugging_face_top_ai_papers/hf_top_ai_papers_final_categorized.jsonl",
    ),
    ("leaderboard_questions", "leaderboard_questions/leaderboard_fixed_date_categorized.jsonl"),
    ("nature", "nature/nature_filtered_domain_with_dates_benchmark_all_categorized.jsonl"),
    ("science", "science/science_filter_domain_with_dates_benchmark_all_categorized.jsonl"),
    ("top_ai_papers", "top_ai_papers/march_2026_top_10_ai_papers_final_categorized.jsonl"),
]


BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark_data"

DEFAULT_OUTPUT = BENCHMARK_DIR / "all_data" / "FutureScience.jsonl"


def main():

    parser = argparse.ArgumentParser(
        description="Combine all categorized JSONL benchmark files into one."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"Output path. Default: {DEFAULT_OUTPUT}",
    )

    args = parser.parse_args()

    outpath = Path(args.output) if args.output else DEFAULT_OUTPUT

    outpath.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    source_counts = {}

    print(f"\n{'='*60}")

    print(f"  Combining categorized JSONL files")

    print(f"{'='*60}\n")

    with open(outpath, "w", encoding="utf-8") as out_f:

        for source_name, rel_path in SOURCES:

            fpath = BENCHMARK_DIR / rel_path

            if not fpath.exists():

                print(f"  ⚠ Skipping (not found): {fpath}")

                continue

            count = 0

            with open(fpath, "r", encoding="utf-8") as in_f:

                for line in in_f:

                    if not line.strip():

                        continue

                    record = json.loads(line)

                    record["source"] = source_name

                    out_f.write(json.dumps(record) + "\n")

                    count += 1

            source_counts[source_name] = count

            total += count

            print(f"  ✓ {source_name:<30} {count:>5} records")

    print(f"\n{'='*60}")

    print(f"  ✅ Output written to: {outpath}")

    print(f"  📊 {total} total records from {len(source_counts)} sources")

    print(f"{'='*60}\n")


if __name__ == "__main__":

    main()
