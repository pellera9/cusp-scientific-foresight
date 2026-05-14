"""
categorize.py — Classify scientific abstracts by research domain using LLM.

Reads a JSONL file with a 'source_abstract' field, sends each abstract to the
LLM, and adds domain labels:
  • main_area          – broad research area (e.g. "Chemistry", "Neuroscience")
  • sub_categories     – list of specific sub-categories

Labels are NOT hardcoded — the LLM decides the best-fit domain name and any
secondary disciplines that the work spans.

Usage
-----
  python categorize.py <input.jsonl>
  python categorize.py <input.jsonl> -o labeled_output.jsonl

Output
------
  Writes a new JSONL with added 'main_area' and 'sub_categories' fields.
"""

import json

import os

import re

import sys

import time

import argparse

from pathlib import Path

from dotenv import load_dotenv

from tqdm import tqdm

load_dotenv()

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")

API_VERSION = os.environ.get("AZURE_OPENAI_API_VER", "2024-12-01-preview")

DEPLOYMENT = "gpt-4o-mini"

_llm_client = None


def _get_client():

    global _llm_client

    if _llm_client is None:

        from openai import AzureOpenAI

        _llm_client = AzureOpenAI(
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
        )

    return _llm_client


def llm_call(system_prompt: str, user_prompt: str) -> str:
    """Single LLM call, returns raw text."""

    client = _get_client()

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    return resp.choices[0].message.content.strip()


SYSTEM_PROMPT = """\
You are a scientific paper classifier. Given an abstract, you must assign:

1. **main_area**: The single broad research area this paper belongs to.
   Use a large, high-level canonical label (e.g., "Artificial Intelligence",
   "Biology", "Physics", "Chemistry", "Neuroscience", "Medicine",
   "Environmental Science", "Materials Science").
   Focus on the macro-level discipline.

2. **sub_categories**: A list of specific sub-categories and specialized areas
   this work falls into or contributes to.
   Include the specific technical domains, methodologies, or niche fields (e.g., "Computer Vision",
   "Organic Chemistry", "Quantum Computing", "Deep Learning", "Cell Biology").
   Leave as an empty list [] if it's too general.

Return EXACTLY one JSON object:
{"main_area": "...", "sub_categories": ["...", "..."]}

Rules:
- main_area MUST be a single, broad discipline (like "Physics" rather than "Condensed Matter Physics").
- sub_categories can contain multiple, more specific sub-fields.
- Return valid JSON only, no extra text."""


def classify_abstract(abstract: str) -> tuple[str, list[str]]:
    """
    Classify a single abstract via the LLM.

    Returns
    -------
    (main_area, sub_categories)
    """

    if not abstract or len(str(abstract).strip()) < 30:

        return "Unclassified", []

    try:

        raw = llm_call(SYSTEM_PROMPT, f"Abstract:\n{abstract}")

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)

        if json_match:

            data = json.loads(json_match.group())

        else:

            data = json.loads(raw)

        primary = data.get("main_area", "Unclassified")

        secondary = data.get("sub_categories", [])

        if isinstance(secondary, str):

            secondary = [s.strip() for s in secondary.split(",") if s.strip()]

        elif not isinstance(secondary, list):

            secondary = []

        else:

            secondary = [str(s).strip() for s in secondary if s]

        return primary, secondary

    except Exception as e:

        print(f"  ⚠ Classification failed: {e}", file=sys.stderr)

        return "Unclassified", []


def main():

    parser = argparse.ArgumentParser(
        description="Add domain labels to a JSONL of scientific abstracts using LLM classification."
    )

    parser.add_argument("jsonl_in", help="Input JSONL file with a 'source_abstract' field")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL path. Default: <input>_categorized.jsonl",
    )

    args = parser.parse_args()

    inpath = Path(args.jsonl_in)

    if not inpath.exists():

        print(f"Error: File not found: {inpath}", file=sys.stderr)

        sys.exit(2)

    records = []

    with open(inpath, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                records.append(json.loads(line))

    total = len(records)

    main_areas = []

    sub_categories_list = []

    print(f"\n{'='*60}")

    print(f"  Categorizing {total} abstracts from {inpath.name}")

    print(f"  Using LLM: {DEPLOYMENT}")

    print(f"{'='*60}\n")

    outpath = Path(args.output) if args.output else inpath.with_stem(inpath.stem + "_categorized")

    with open(outpath, "w", encoding="utf-8") as out_f:

        for i in tqdm(range(total), desc="Classifying"):

            record = records[i]

            abstract = str(record.get("source_abstract", ""))

            primary, secondary = classify_abstract(abstract)

            main_areas.append(primary)

            sub_categories_list.append(secondary)

            record["main_area"] = primary

            record["sub_categories"] = secondary

            out_f.write(json.dumps(record) + "\n")

            time.sleep(0.1)

    from collections import Counter

    primary_counts = Counter(main_areas)

    all_secondary = []

    for s_list in sub_categories_list:

        for s in s_list:

            if s:

                all_secondary.append(s.strip())

    secondary_counts = Counter(all_secondary)

    n_categorized = sum(1 for s in sub_categories_list if s)

    print(f"\n  MAIN AREAS")

    print(f"  {'Area':<40} {'Count':>6}  {'%':>6}")

    print(f"  {'-'*40} {'-'*6}  {'-'*6}")

    for domain, count in primary_counts.most_common():

        pct = count / total * 100

        print(f"  {domain:<40} {count:>6}  {pct:>5.1f}%")

    if secondary_counts:

        print(f"\n  SUB-CATEGORIES  ({n_categorized}/{total} papers have specific sub-categories)")

        print(f"  {'Sub-category':<40} {'Mentions':>8}")

        print(f"  {'-'*40} {'-'*8}")

        for domain, count in secondary_counts.most_common(15):

            print(f"  {domain:<40} {count:>8}")

    print(f"\n{'='*60}")

    print(f"  ✅ Output written to: {outpath}")

    print(f"  📊 {total} abstracts → {len(primary_counts)} unique main areas")

    print(f"  🔀 {n_categorized} categorized papers detected")

    print(f"{'='*60}\n")


if __name__ == "__main__":

    main()
