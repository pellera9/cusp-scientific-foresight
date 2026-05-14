"""
verifier.py — Pre-processing filter for FutureScience benchmark pipeline.

Reads a raw CSV of paper abstracts and filters out entries that do NOT contain
verifiable, measurable findings. This saves compute by skipping vague abstracts
before running create_bench.py.

Usage:
  python verifier.py <input.csv> --domain ai
  python verifier.py <input.csv> --domain auto    # auto-detect domain per abstract
  python verifier.py <input.csv> --domain chem

Output:
  <input>_verified.csv   — rows that passed (feed to create_bench.py)
  <input>_rejected.csv   — rows that failed (with rejection reason)
"""

import json

import os

import re

import sys

import time

import argparse

from pathlib import Path

import pandas as pd

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
            api_version=API_VERSION, azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY
        )

    return _llm_client


def llm_judge(system_prompt: str, user_prompt: str) -> str:
    """Quick LLM call for verification."""

    client = _get_client()

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()


DOMAIN_CRITERIA = {
    "ai": {
        "name": "Artificial Intelligence",
        "criteria": [
            "Describes a concrete technical breakthrough with a clear method or approach, "
            "validated on a recognized benchmark, competition, or evaluation (e.g., CASP, "
            "MATH, ImageNet, MMLU), even without exact numeric scores.",
            "Reports specific performance metrics (accuracy, F1, BLEU, perplexity, etc.) "
            "or demonstrates measurable improvement over prior methods.",
            "Achieves a clearly defined capability milestone (e.g., 'first method to do X', "
            "'matches or exceeds human performance on Y') with a describable method.",
        ],
    },
    "chem": {
        "name": "Chemistry",
        "criteria": [
            "Describes a concrete synthesis, reaction, or material discovery with a clear "
            "method that produces a verifiable outcome (new compound, new reaction pathway, "
            "new material property).",
            "Reports measurable quantities (yields, selectivity, binding affinity, conductivity, "
            "rates) or demonstrates improvement over prior methods.",
            "Achieves a capability milestone (e.g., 'first synthesis of X', 'enables Y at "
            "room temperature') with a describable approach.",
        ],
    },
    "bio": {
        "name": "Biology / Life Sciences",
        "criteria": [
            "Describes a concrete biological discovery with a clear experimental method "
            "and verifiable outcome (new mechanism, pathway, gene function, therapeutic effect).",
            "Reports measurable biological quantities (fold changes, survival rates, expression "
            "levels, p-values) or demonstrates improvement over prior methods.",
            "Achieves a capability milestone (e.g., 'first demonstration of X', 'identifies "
            "the mechanism behind Y') with a describable experimental approach.",
        ],
    },
    "physics": {
        "name": "Physics",
        "criteria": [
            "Describes a concrete experimental or theoretical breakthrough with a clear "
            "method and verifiable outcome (new measurement, new phenomenon, new prediction).",
            "Reports measurable physical quantities (precision, resolution, energy scales, "
            "cross-sections) or demonstrates improvement over prior methods.",
            "Achieves a capability milestone (e.g., 'first observation of X', 'achieves "
            "coherence time of Y') with a describable approach.",
        ],
    },
    "general": {
        "name": "General Science",
        "criteria": [
            "Describes a concrete scientific breakthrough or discovery with a clear method "
            "and a verifiable outcome that could be independently reproduced or validated.",
            "Reports measurable results or demonstrates clear improvement over prior work, "
            "even if described qualitatively (e.g., 'greatly outperforming other methods').",
            "Achieves a defined capability milestone with a describable approach, validated "
            "against a recognized standard, baseline, or prior state of the art.",
        ],
    },
}

ALL_DOMAIN_KEYS = [k for k in DOMAIN_CRITERIA if k != "general"]


def build_criteria_prompt(domain: str) -> str:
    """Build the LLM system prompt with domain-specific criteria."""

    info = DOMAIN_CRITERIA[domain]

    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(info["criteria"]))

    return (
        f"You are a scientific paper screener for the {info['name']} domain.\n\n"
        f"Given a paper abstract, determine if it contains AT LEAST ONE concrete, "
        f"verifiable result or breakthrough matching any of these criteria:\n"
        f"{criteria_text}\n\n"
        f"An abstract passes if it "
        f"describes a clear breakthrough with a concrete method and a verifiable outcome "
        f"(e.g., 'first method to achieve X', 'outperforms all prior methods on Y benchmark').\n"
        f"FAIL abstracts that are purely descriptive, speculative, or review-like "
        f"with no concrete result or method.\n\n"
        f"Reply with exactly one line:\n"
        f"PASS: <one-sentence summary of the concrete result found>\n"
        f"or\n"
        f"FAIL: <one-sentence reason why no concrete result was found>"
    )


def detect_domain(abstract: str) -> str:
    """Auto-detect the scientific domain of an abstract."""

    domain_list = ", ".join(f"{k} ({DOMAIN_CRITERIA[k]['name']})" for k in ALL_DOMAIN_KEYS)

    try:

        result = llm_judge(
            "You classify scientific paper abstracts into research domains. "
            f"Available domains: {domain_list}. "
            "Reply with ONLY the domain key (e.g., ai, bio, chem, physics). "
            "If the paper doesn't clearly fit any domain, reply: general",
            f"Abstract:\n{abstract}",
        )

        domain = result.strip().lower().split()[0].rstrip(".,:")

        if domain in DOMAIN_CRITERIA:

            return domain

        return "general"

    except Exception:

        return "general"


def screen_abstract(abstract: str, domain: str) -> tuple[bool, str, str]:
    """
    Screen a single abstract for verifiable results.
    Returns (passes, reason_string, detected_domain).
    """

    if not abstract or len(abstract.strip()) < 50:

        return False, "Abstract too short or empty", domain

    actual_domain = domain

    if domain == "auto":

        actual_domain = detect_domain(abstract)

    try:

        system_prompt = build_criteria_prompt(actual_domain)

        verdict = llm_judge(system_prompt, f"Abstract:\n{abstract}")

        print(verdict)

        if verdict.upper().startswith("PASS"):

            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else "Has measurable result"

            return True, reason, actual_domain

        else:

            reason = (
                verdict.split(":", 1)[1].strip() if ":" in verdict else "No measurable result found"
            )

            return False, reason, actual_domain

    except Exception as e:

        print(e)

        return True, f"LLM check failed ({e}), defaulting to pass", actual_domain


def main():

    parser = argparse.ArgumentParser(
        description="Pre-filter CSV abstracts for verifiable results before benchmark generation."
    )

    parser.add_argument("csv_in", help="Input CSV file with 'abstract' column")

    parser.add_argument(
        "--domain",
        required=True,
        choices=list(DOMAIN_CRITERIA.keys()) + ["auto"],
        help="Domain for criteria. Use 'auto' to detect per-abstract.",
    )

    parser.add_argument(
        "--abstract-col",
        default="abstract",
        help="Name of the abstract column (default: 'abstract')",
    )

    args = parser.parse_args()

    inpath = Path(args.csv_in)

    if not inpath.exists():

        print(f"File not found: {inpath}", file=sys.stderr)

        sys.exit(2)

    df = pd.read_csv(inpath)

    if args.abstract_col not in df.columns:

        print(
            f"Column '{args.abstract_col}' not found. Available: {list(df.columns)}",
            file=sys.stderr,
        )

        sys.exit(2)

    total = len(df)

    print(f"\n{'='*70}")

    print(f"  Screening {total} abstracts from {inpath.name}")

    if args.domain != "auto":

        domain_info = DOMAIN_CRITERIA[args.domain]

        print(f"  Domain: {domain_info['name']}")

        print(f"  Criteria (abstract must meet ≥1):")

        for i, c in enumerate(domain_info["criteria"]):

            print(f"    {i+1}. {c[:90]}...")

    else:

        print(f"  Domain: AUTO-DETECT per abstract")

        domain_names = ", ".join(f'{k} ({v["name"]})' for k, v in DOMAIN_CRITERIA.items())

        print(f"  Available: {domain_names}")

    print(f"{'='*70}\n")

    pass_mask = []

    reasons = []

    detected_domains = []

    for i in tqdm(range(total), desc="Screening"):

        abstract = str(df.iloc[i][args.abstract_col])

        passed, reason, det_domain = screen_abstract(abstract, args.domain)

        pass_mask.append(passed)

        reasons.append(reason)

        detected_domains.append(det_domain)

        time.sleep(0.1)

    df["_verifier_pass"] = pass_mask

    df["_verifier_reason"] = reasons

    df["_detected_domain"] = detected_domains

    passed_df = df[df["_verifier_pass"]].copy()

    rejected_df = df[~df["_verifier_pass"]].copy()

    passed_out = passed_df.drop(columns=["_verifier_pass", "_verifier_reason"])

    if "_detected_domain" in passed_out.columns:

        passed_out = passed_out.rename(columns={"_detected_domain": "domain"})

    n_pass = len(passed_df)

    n_fail = len(rejected_df)

    out_verified = inpath.with_stem(inpath.stem + "_verified")

    out_rejected = inpath.with_stem(inpath.stem + "_rejected")

    passed_out.to_csv(out_verified, index=False)

    rejected_df.to_csv(out_rejected, index=False)

    if n_fail > 0:

        print(f"\n⚠  REJECTED: {n_fail}/{total} abstracts\n")

        for _, row in rejected_df.iterrows():

            title = str(row.get("Title", row.get("title", "?")))[:60]

            reason = row["_verifier_reason"]

            det = row.get("_detected_domain", "?")

            domain_tag = f" [{det}]" if args.domain == "auto" else ""

            print(f'  ✗{domain_tag} "{title}..."')

            print(f"    → {reason}\n")

    print(f"{'='*70}")

    print(f"  SUMMARY: {n_pass}/{total} passed, {n_fail}/{total} rejected")

    print(f"  ✅ Verified CSV: {out_verified}")

    print(f"  ❌ Rejected CSV: {out_rejected}")

    print(f"{'='*70}\n")


if __name__ == "__main__":

    main()
