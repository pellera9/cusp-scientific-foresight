"""
categorize_ai.py — Classify AI scientific abstracts into sub-categories using LLM.

Reads a JSONL file containing paper metadata (including a 'source_abstract' field),
sends each abstract to the LLM, and adds:
  • main_area          – always "Artificial Intelligence"
  • sub_categories     – list of specific AI sub-categories

Labels are NOT hardcoded — the LLM decides the best-fit sub-categories.

Usage
-----
  python categorize_ai.py <input.jsonl>
  python categorize_ai.py <input.jsonl> -o labeled_output.jsonl

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
You are a scientific paper classifier sub-specializing in Artificial Intelligence.
Given an abstract of an AI paper, you must assign:

1. **sub_categories**: A list of specific sub-categories and specialized areas
   this work falls into or contributes to within Artificial Intelligence.
   Include the specific technical domains, methodologies, or niche fields
   (e.g., "Large Language Models", "Computer Vision", "Reinforcement Learning",
   "Diffusion Models", "Robotics", "Natural Language Processing", "Graph Neural Networks").
   Do NOT just say "Artificial Intelligence" or "Machine Learning". Be specific.

Return EXACTLY one JSON object:
{"sub_categories": ["...", "..."]}

Rules:
- sub_categories can contain multiple, more specific sub-fields.
- Return valid JSON only, no extra text."""


def classify_abstract(abstract: str) -> list[str]:
    """
    Classify a single abstract via the LLM.

    Returns
    -------
    sub_categories
    """

    if not abstract or len(str(abstract).strip()) < 30:

        return []

    try:

        raw = llm_call(SYSTEM_PROMPT, f"Abstract:\n{abstract}")

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)

        if json_match:

            data = json.loads(json_match.group())

        else:

            data = json.loads(raw)

        secondary = data.get("sub_categories", [])

        if isinstance(secondary, str):

            secondary = [s.strip() for s in secondary.split(",") if s.strip()]

        elif not isinstance(secondary, list):

            secondary = []

        else:

            secondary = [str(s).strip() for s in secondary if s]

        return secondary

    except Exception as e:

        print(f"  ⚠ Classification failed: {e}", file=sys.stderr)

        return []


def main():

    parser = argparse.ArgumentParser(
        description="Add AI sub-categories to a JSONL of scientific abstracts."
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

    sub_categories_list = []

    print(f"\n{'='*60}")

    print(f"  Categorizing {total} abstracts from {inpath.name}")

    print(f"  Using LLM: {DEPLOYMENT}")

    print(f"  Main area: Artificial Intelligence (fixed)")

    print(f"{'='*60}\n")

    outpath = Path(args.output) if args.output else inpath.with_stem(inpath.stem + "_categorized")

    with open(outpath, "w", encoding="utf-8") as out_f:

        for i in tqdm(range(total), desc="Classifying"):

            record = records[i]

            abstract = str(record.get("source_abstract", ""))

            secondary = classify_abstract(abstract)

            sub_categories_list.append(secondary)

            record["main_area"] = "Artificial Intelligence"

            record["sub_categories"] = secondary

            out_f.write(json.dumps(record) + "\n")

            time.sleep(0.1)

    from collections import Counter

    all_secondary = []

    for s_list in sub_categories_list:

        for s in s_list:

            if s:

                all_secondary.append(s.strip())

    secondary_counts = Counter(all_secondary)

    n_categorized = sum(1 for s in sub_categories_list if s)

    if secondary_counts:

        print(f"\n  AI SUB-CATEGORIES  ({n_categorized}/{total} papers categorized)")

        print(f"  {'Sub-category':<40} {'Mentions':>8}")

        print(f"  {'-'*40} {'-'*8}")

        for domain, count in secondary_counts.most_common(20):

            print(f"  {domain:<40} {count:>8}")

    print(f"\n{'='*60}")

    print(f"  ✅ Output written to: {outpath}")

    print(f"  📊 {total} abstracts -> {len(secondary_counts)} unique AI sub-categories")

    print(f"{'='*60}\n")


if __name__ == "__main__":

    main()
