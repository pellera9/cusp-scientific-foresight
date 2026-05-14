"""
view_data.py — Print statistics about the FutureScience benchmark dataset.

Reads FutureScience.jsonl and displays:
  • Overall dataset size
  • Source breakdown
  • Main area & sub-category distributions
  • Question length statistics (binary, MCQ, FRQ, date prediction)
  • Publication date distribution
  • Ground truth date distribution
  • MCQ choice counts
  • Field completeness

Usage
-----
  python view_data.py
  python view_data.py /path/to/FutureScience.jsonl
"""

import json

import sys

import argparse

import statistics

from pathlib import Path

from collections import Counter

DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "benchmark_data" / "all_data" / "FutureScience.jsonl"
)


def load_records(path: Path) -> list[dict]:

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                records.append(json.loads(line))

    return records


def len_stats(values: list[int], label: str):
    """Print min/max/mean/median for a list of lengths."""

    if not values:

        print(f"    {label}: no data")

        return

    print(f"    {label}:")

    print(f"      Count:  {len(values)}")

    print(f"      Min:    {min(values):,} chars")

    print(f"      Max:    {max(values):,} chars")

    print(f"      Mean:   {statistics.mean(values):,.0f} chars")

    print(f"      Median: {statistics.median(values):,.0f} chars")


def section(title: str):

    print(f"\n  {'─'*56}")

    print(f"  {title}")

    print(f"  {'─'*56}")


def main():

    parser = argparse.ArgumentParser(description="View stats about FutureScience.jsonl")

    parser.add_argument(
        "jsonl_path", nargs="?", default=str(DEFAULT_PATH), help="Path to FutureScience.jsonl"
    )

    args = parser.parse_args()

    path = Path(args.jsonl_path)

    if not path.exists():

        print(f"Error: File not found: {path}", file=sys.stderr)

        sys.exit(2)

    records = load_records(path)

    total = len(records)

    print(f"\n{'='*60}")

    print(f"  📊 FutureScience Dataset Statistics")

    print(f"  File: {path.name}")

    print(f"  Total records: {total:,}")

    print(f"{'='*60}")

    section("📁 Source Breakdown")

    source_counts = Counter(r.get("source", "unknown") for r in records)

    for src, cnt in source_counts.most_common():

        pct = cnt / total * 100

        bar = "█" * int(pct / 2)

        print(f"    {src:<35} {cnt:>5}  ({pct:5.1f}%)  {bar}")

    section("🏷️  Main Areas")

    area_counts = Counter(r.get("main_area", "Unknown") for r in records)

    for area, cnt in area_counts.most_common():

        pct = cnt / total * 100

        bar = "█" * int(pct / 2)

        print(f"    {area:<35} {cnt:>5}  ({pct:5.1f}%)  {bar}")

    section("🔖 Sub-Categories (top 25)")

    all_subs = []

    records_with_subs = 0

    sub_counts_per_record = []

    for r in records:

        subs = r.get("sub_categories", [])

        if isinstance(subs, str):

            subs = [s.strip() for s in subs.split(",") if s.strip()]

        if subs:

            records_with_subs += 1

        sub_counts_per_record.append(len(subs))

        all_subs.extend(subs)

    sub_counts = Counter(all_subs)

    print(
        f"    Records with sub-categories: {records_with_subs:,}/{total:,} ({records_with_subs/total*100:.1f}%)"
    )

    print(f"    Unique sub-categories:       {len(sub_counts):,}")

    print(f"    Avg sub-categories/record:   {statistics.mean(sub_counts_per_record):.1f}")

    print()

    for sub, cnt in sub_counts.most_common(25):

        print(f"    {sub:<45} {cnt:>5}")

    section("🔀 Main Area × Source")

    sources = sorted(source_counts.keys())

    hdr = f"    {'Area':<30}"

    for s in sources:

        hdr += f" {s[:10]:>10}"

    hdr += f" {'Total':>8}"

    print(hdr)

    print(f"    {'-'*30}" + f" {'-'*10}" * len(sources) + f" {'-'*8}")

    for area, _ in area_counts.most_common():

        row = f"    {area:<30}"

        row_total = 0

        for s in sources:

            cnt = sum(1 for r in records if r.get("main_area") == area and r.get("source") == s)

            row += f" {cnt:>10}"

            row_total += cnt

        row += f" {row_total:>8}"

        print(row)

    section("📏 Question / Prompt Lengths")

    text_fields = [
        ("binary_question", "Binary Question"),
        ("binary_question_perturbed", "Binary Question (perturbed)"),
        ("mcq_question", "MCQ Question"),
        ("frq_prompt", "FRQ Prompt"),
        ("date_prediction_prompt", "Date Prediction Prompt"),
        ("source_abstract", "Source Abstract"),
        ("problem_statement", "Problem Statement"),
        ("technical_approach", "Technical Approach"),
        ("results_and_metrics", "Results & Metrics"),
    ]

    for field, label in text_fields:

        lengths = [len(str(r[field])) for r in records if r.get(field)]

        len_stats(lengths, label)

    section("🔢 MCQ Choices")

    choice_counts = [len(r["mcq_choices"]) for r in records if r.get("mcq_choices")]

    if choice_counts:

        choice_counter = Counter(choice_counts)

        for n, cnt in sorted(choice_counter.items()):

            print(f"    {n} choices: {cnt:,} records")

        answer_keys = Counter(r.get("mcq_answer_key") for r in records if r.get("mcq_choices"))

        print(f"\n    Answer key distribution:")

        for key, cnt in sorted(answer_keys.items()):

            print(f"      Key {key}: {cnt:,} ({cnt/sum(answer_keys.values())*100:.1f}%)")

    section("📅 Publication Dates")

    pub_dates = Counter(r.get("publication_date", "unknown") for r in records)

    for d, cnt in sorted(pub_dates.items()):

        bar = "█" * max(1, int(cnt / total * 100))

        print(f"    {d:<12} {cnt:>5}  {bar}")

    section("🎯 Ground Truth Dates")

    gt_dates = Counter(r.get("ground_truth_date", "unknown") for r in records)

    for d, cnt in sorted(gt_dates.items()):

        bar = "█" * max(1, int(cnt / total * 100))

        print(f"    {d:<12} {cnt:>5}  {bar}")

    section("✅ Field Completeness")

    all_fields = set()

    for r in records:

        all_fields.update(r.keys())

    for field in sorted(all_fields):

        present = sum(
            1 for r in records if r.get(field) is not None and str(r.get(field, "")).strip()
        )

        pct = present / total * 100

        status = "✓" if pct == 100 else "△" if pct > 50 else "✗"

        print(f"    {status} {field:<35} {present:>5}/{total}  ({pct:5.1f}%)")

    print(f"\n{'='*60}")

    print(f"  Done. {total:,} records analyzed.")

    print(f"{'='*60}\n")


if __name__ == "__main__":

    main()
