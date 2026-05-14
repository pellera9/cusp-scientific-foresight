"""
categorize_leaderboard.py — Classify AI leaderboard questions into sub-categories.

Reads a JSONL file containing leaderboard benchmark questions, and adds:
  • main_area          – always "Artificial Intelligence"
  • sub_categories     – always includes "AI Benchmarks", plus LLM-determined
                         sub-fields based on the binary_question content

Usage
-----
  python categorize_leaderboard.py <input.jsonl>
  python categorize_leaderboard.py <input.jsonl> -o labeled_output.jsonl

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
You are a classifier for AI benchmark questions. Given a question about an AI
benchmark or leaderboard, you must identify the specific AI sub-fields that
the benchmark measures or relates to.

Return EXACTLY one JSON object:
{"extra_sub_categories": ["...", "..."]}

Rules:
- Identify the specific AI domains the benchmark targets (e.g., "Natural Language Processing",
  "Computer Vision", "Reinforcement Learning", "Large Language Models", "Speech Recognition",
  "Machine Translation", "Reading Comprehension", "Code Generation", "Reasoning",
  "Mathematical Reasoning", "Multimodal Learning", "Text Generation").
- Be specific — do NOT just say "Artificial Intelligence" or "Machine Learning".
- Include 1-3 relevant sub-fields.
- Return valid JSON only, no extra text."""


def classify_question(question: str) -> list[str]:
    """
    Classify a single benchmark question via the LLM.

    Returns
    -------
    extra_sub_categories (without "AI Benchmarks", that is added later)
    """

    if not question or len(str(question).strip()) < 20:

        return []

    try:

        raw = llm_call(SYSTEM_PROMPT, f"Benchmark question:\n{question}")

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)

        if json_match:

            data = json.loads(json_match.group())

        else:

            data = json.loads(raw)

        extras = data.get("extra_sub_categories", [])

        if isinstance(extras, str):

            extras = [s.strip() for s in extras.split(",") if s.strip()]

        elif not isinstance(extras, list):

            extras = []

        else:

            extras = [str(s).strip() for s in extras if s]

        return extras

    except Exception as e:

        print(f"  ⚠ Classification failed: {e}", file=sys.stderr)

        return []


def main():

    parser = argparse.ArgumentParser(
        description="Add AI sub-categories to a JSONL of leaderboard benchmark questions."
    )

    parser.add_argument("jsonl_in", help="Input JSONL file with a 'binary_question' field")

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

    print(f"  Categorizing {total} leaderboard questions from {inpath.name}")

    print(f"  Using LLM: {DEPLOYMENT}")

    print(f"  Main area: Artificial Intelligence (fixed)")

    print(f"  Base sub-category: AI Benchmarks (always included)")

    print(f"{'='*60}\n")

    outpath = Path(args.output) if args.output else inpath.with_stem(inpath.stem + "_categorized")

    with open(outpath, "w", encoding="utf-8") as out_f:

        for i in tqdm(range(total), desc="Classifying"):

            record = records[i]

            question = str(record.get("binary_question", ""))

            extras = classify_question(question)

            sub_cats = ["AI Benchmarks"] + [s for s in extras if s != "AI Benchmarks"]

            sub_categories_list.append(sub_cats)

            record["main_area"] = "Artificial Intelligence"

            record["sub_categories"] = sub_cats

            out_f.write(json.dumps(record) + "\n")

            time.sleep(0.1)

    from collections import Counter

    all_sub = []

    for s_list in sub_categories_list:

        for s in s_list:

            if s:

                all_sub.append(s.strip())

    sub_counts = Counter(all_sub)

    n_categorized = sum(1 for s in sub_categories_list if len(s) > 1)

    if sub_counts:

        print(
            f"\n  AI SUB-CATEGORIES  ({n_categorized}/{total} have extra sub-categories beyond 'AI Benchmarks')"
        )

        print(f"  {'Sub-category':<40} {'Mentions':>8}")

        print(f"  {'-'*40} {'-'*8}")

        for domain, count in sub_counts.most_common(20):

            print(f"  {domain:<40} {count:>8}")

    print(f"\n{'='*60}")

    print(f"  ✅ Output written to: {outpath}")

    print(f"  📊 {total} questions -> {len(sub_counts)} unique sub-categories")

    print(f"{'='*60}\n")


if __name__ == "__main__":

    main()
