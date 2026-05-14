"""

Key behavior:
- Uses the per-row Date column as-is and stores it as publication_date.
- Converts YYYY-MM to Mon YYYY for human-readable questions.
- Deterministically extracts the best score only from model columns.
- Uses separate LLM calls for:
    * background_short (1-2 sentences WITHOUT numeric scores)
    * mcq_json (JSON) to generate MCQ background/question/choices/answer key
    * date_json (JSON) to generate date background + month-year prediction prompt
- Writes one JSON object per line.
- Uses Azure SDK if available, otherwise a REST fallback adapter.
"""

import os

import sys

import json

import time

import argparse

import uuid

import re

import math

from datetime import datetime

import pandas as pd

import requests

from dotenv import load_dotenv

from tqdm import tqdm

load_dotenv()

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")

API_VERSION = os.environ.get("AZURE_OPENAI_API_VER", "2024-12-01-preview")

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

if not AZURE_ENDPOINT or not AZURE_KEY or not DEPLOYMENT:

    raise RuntimeError(
        "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and AZURE_OPENAI_DEPLOYMENT "
        "in your environment or .env file."
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate benchmark forecasting dataset from a CSV leaderboard."
    )

    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")

    parser.add_argument(
        "--max-rows", type=int, default=None, help="Maximum number of rows to process."
    )

    return parser.parse_args()


args = parse_args()

CSV_PATH = args.csv

MAX_ROWS = args.max_rows

csv_base = os.path.splitext(os.path.basename(CSV_PATH))[0]

OUTPUT_JSONL = f"{csv_base}.jsonl"


client = None

try:

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
    )

except Exception:

    class AzureRESTAdapter:

        def __init__(self, endpoint, api_key, api_version):

            self.endpoint = endpoint.rstrip("/")

            self.api_key = api_key

            self.api_version = api_version

        def chat_completions_create(
            self, deployment_id, messages, temperature=0.0, max_tokens=1024, top_p=1.0
        ):

            url = f"{self.endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version={self.api_version}"

            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }

            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            r = requests.post(url, headers=headers, json=payload, timeout=120)

            r.raise_for_status()

            return r.json()

        @property
        def chat(self):

            class C:

                def __init__(self, outer):

                    self._o = outer

                @property
                def completions(self):

                    class CC:

                        def __init__(self, outer):

                            self._o = outer

                        def create(self, **kwargs):

                            deployment = (
                                kwargs.get("deployment")
                                or kwargs.get("deployment_id")
                                or kwargs.get("model")
                            )

                            messages = kwargs.get("messages")

                            if not messages:

                                prompt = kwargs.get("input") or kwargs.get("prompt") or ""

                                messages = [{"role": "user", "content": prompt}]

                            temperature = kwargs.get("temperature", 0.0)

                            max_tokens = kwargs.get("max_tokens", 1024)

                            top_p = kwargs.get("top_p", 1.0)

                            return self._o._o.chat_completions_create(
                                deployment,
                                messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                            )

                    return CC(self)

            return C(self)

    client = AzureRESTAdapter(AZURE_ENDPOINT, AZURE_KEY, API_VERSION)


def call_gpt(prompt, temperature=0.6, max_tokens=1200, deployment_id=None):

    deployment_id = deployment_id or DEPLOYMENT

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise, neutral benchmark-background and prompt generator "
                "for a scientific forecasting dataset."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        model=deployment_id,
    )

    if isinstance(response, dict):

        try:

            return response["choices"][0]["message"]["content"].strip()

        except Exception:

            try:

                return response["choices"][0]["text"].strip()

            except Exception:

                raise RuntimeError("Unexpected REST response format.")

    else:

        try:

            return response.choices[0].message.content.strip()

        except Exception:

            return str(response).strip()


def normalize_col_name(c):

    return re.sub(r"\s+", " ", str(c).strip().lower())


def is_unnamed_column(c):

    return normalize_col_name(c).startswith("unnamed")


def find_first_matching_column(columns, candidates):

    normalized_candidates = {normalize_col_name(x) for x in candidates}

    for c in columns:

        if normalize_col_name(c) in normalized_candidates:

            return c

    return None


def safe_json_loads(s):

    try:

        return json.loads(s)

    except Exception:

        m = re.search(r"\{.*\}", s, flags=re.S)

        if m:

            try:

                return json.loads(m.group(0))

            except Exception:

                return None

    return None


def format_month_year_from_yyyy_mm(yyyy_mm):

    try:

        year, month = yyyy_mm.split("-")

        dt = datetime(int(year), int(month), 1)

        return dt.strftime("%b %Y")

    except Exception:

        return yyyy_mm


def parse_score_cell(val):
    """
    Parse score from a leaderboard cell.

    Handles:
      92.2
      74.70%
      59.2% (Deep Res.)
      77.9% (Pro)
      88.3
      -
      blank
    """

    if pd.isna(val):

        return None

    s = str(val).strip()

    if s in {"", "-", "—", "–", "nan", "None"}:

        return None

    s = s.replace(",", "")

    m = re.search(r"(\d+(?:\.\d+)?)", s)

    if not m:

        return None

    try:

        return float(m.group(1))

    except Exception:

        return None


def display_score_from_cell(raw_value, parsed_score):
    """
    Preserve annotated text if the original cell had a percent sign.
    Otherwise format the parsed numeric score as a percent.
    """

    if raw_value is not None and not pd.isna(raw_value):

        s = re.sub(r"\s+", " ", str(raw_value).strip())

        if "%" in s:

            return s

    if parsed_score is None:

        return ""

    s = f"{parsed_score:.2f}".rstrip("0").rstrip(".")

    return f"{s}%"


def extract_best_score(row, model_cols):
    """
    Deterministically extract the best score from model columns only.
    Returns:
      best_col, best_score_float, best_raw_cell, best_display_score
    """

    best_score = None

    best_col = None

    best_raw = None

    for col in model_cols:

        raw = row.get(col, None)

        score = parse_score_cell(raw)

        if score is None:

            continue

        if best_score is None or score > best_score:

            best_score = score

            best_col = col

            best_raw = raw

    if best_score is None:

        return None, None, None, None

    best_display = display_score_from_cell(best_raw, best_score)

    return best_col, round(best_score, 2), best_raw, best_display


def clean_round_up(x):

    return int(math.ceil(x / 5.0) * 5)


def compute_realistic_threshold(best_score):

    return clean_round_up(best_score + 3)


def compute_inflated_threshold(best_score):

    inflated = min(best_score + 5, 100)

    return clean_round_up(inflated)


def pct_str(n):

    s = f"{n:.2f}".rstrip("0").rstrip(".")

    return f"{s}%"


def make_mcq_fallback(best_score, best_display_score, display_month_year, benchmark_label):

    a = max(0, round(best_score - 10, 1))

    b = best_display_score if best_display_score else pct_str(best_score)

    c = min(100, round(best_score + 10, 1))

    d = min(100, round(best_score + 20, 1))

    choices = [pct_str(a), b, pct_str(c), pct_str(d)]

    answer_key = 1

    question = (
        f"By {display_month_year}, what do you predict the highest score achieved by a publicly "
        f"reported AI system on {benchmark_label} will be?"
    )

    background = f"{benchmark_label} is a benchmark used to evaluate AI systems on this task."

    return {
        "mcq_background": background,
        "mcq_question": question,
        "mcq_choices": choices,
        "mcq_answer_key": answer_key,
    }


df_raw = pd.read_csv(CSV_PATH)

df = df_raw.copy()

benchmark_col = find_first_matching_column(
    df.columns,
    ["Benchmark (Metric)", "Metric / Benchmark", "Benchmark", "Metric"],
)

if benchmark_col is None:

    benchmark_col = df.columns[0]

date_col = find_first_matching_column(df.columns, ["Date", "date"])

desc_col = find_first_matching_column(
    df.columns,
    ["description", "Benchmark Description", "Description", "benchmark description", "desc"],
)

meta_cols = {benchmark_col}

if date_col is not None:

    meta_cols.add(date_col)

if desc_col is not None:

    meta_cols.add(desc_col)

model_cols = [c for c in df.columns if c not in meta_cols and not is_unnamed_column(c)]

if MAX_ROWS is not None:

    df = df.head(MAX_ROWS)

print(f"Rows to process: {len(df)}")

print(f"Benchmark column: {benchmark_col}")

print(f"Date column: {date_col}")

print(f"Description column: {desc_col}")

print(f"Model columns: {model_cols}")


with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:

            benchmark_label = str(row.get(benchmark_col, "")).strip()

            description = str(row.get(desc_col, "")).strip() if desc_col else ""

            if date_col and row.get(date_col, None) not in (None, ""):

                publication_date_raw = str(row.get(date_col)).strip()

                publication_date = publication_date_raw

            else:

                publication_date_raw = ""

                publication_date = ""

            display_month_year = (
                format_month_year_from_yyyy_mm(publication_date_raw)
                if publication_date_raw
                else "Feb 2026"
            )

            best_col, best_score, best_raw_cell, best_display_score = extract_best_score(
                row, model_cols
            )

            if best_score is None:

                continue

            realistic_threshold = compute_realistic_threshold(best_score)

            inflated_threshold = compute_inflated_threshold(best_score)

            realistic_threshold_display = pct_str(realistic_threshold)

            inflated_threshold_display = pct_str(inflated_threshold)

            background_prompt_short = f"""
Produce a neutral 1-2 sentence background describing what the benchmark named below measures.
DO NOT include any numeric performance scores in this background.
Return only the 1-2 sentence background (no JSON wrapper).

Benchmark Name: {benchmark_label}
Description: {description}
"""

            background_short = call_gpt(
                background_prompt_short, temperature=0.2, max_tokens=180
            ).strip()

            background_short = re.sub(r"\s+", " ", background_short)

            if not background_short.endswith("."):

                background_short = background_short.rstrip(".") + "."

            binary_question = (
                f"{background_short} By {display_month_year}, do you think there will be a publicly "
                f"reported AI system that achieves {best_display_score} or higher on {benchmark_label}?"
            )

            binary_question_perturbed = (
                f"{background_short} By {display_month_year}, do you think there will be a publicly "
                f"reported AI system that achieves {inflated_threshold_display} or higher on {benchmark_label}?"
            )

            mcq_prompt = f"""
Return ONLY a JSON object with the exact keys:
  "mcq_background": string,
  "mcq_question": string,
  "mcq_choices": [4 strings],
  "mcq_answer_key": integer (0-3)

Constraints:
- mcq_background: 1-2 sentence factual background. DO NOT include the best score ({best_display_score}) in the background.
- mcq_question: Ask what the highest score achieved by a publicly reported AI system on {benchmark_label} will be by {display_month_year}.
- mcq_choices: Exactly 4 choices (strings) representing different percentages.
- One choice MUST be exactly "{best_display_score}".
- mcq_answer_key: 0-based index selecting the choice that is "{best_display_score}".
- Return only the JSON object (no extra commentary).

Benchmark Name: {benchmark_label}
Description: {description}
For your knowledge only: the current best score in the CSV is {best_display_score}.
Best score came from model column: {best_col}
Best raw cell text: {best_raw_cell}
Do NOT reveal that fact in mcq_background or mcq_question.
"""

            generated_mcq = call_gpt(mcq_prompt, temperature=0.6, max_tokens=900)

            parsed_mcq = safe_json_loads(generated_mcq)

            if not parsed_mcq:

                generated_mcq = call_gpt(
                    "Return ONLY the JSON object (no commentary).\n\n" + mcq_prompt,
                    temperature=0.2,
                    max_tokens=900,
                )

                parsed_mcq = safe_json_loads(generated_mcq)

            if not parsed_mcq:

                mcq = make_mcq_fallback(
                    best_score, best_display_score, display_month_year, benchmark_label
                )

                mcq_background = mcq["mcq_background"]

                mcq_question = mcq["mcq_question"]

                mcq_choices = mcq["mcq_choices"]

                mcq_answer_key = mcq["mcq_answer_key"]

            else:

                mcq_background = str(parsed_mcq.get("mcq_background", background_short)).strip()

                mcq_question = str(
                    parsed_mcq.get(
                        "mcq_question",
                        f"By {display_month_year}, what do you predict the highest score achieved by a publicly reported AI system on {benchmark_label} will be?",
                    )
                ).strip()

                mcq_choices = parsed_mcq.get("mcq_choices", [])

                mcq_answer_key = parsed_mcq.get("mcq_answer_key", None)

                if not isinstance(mcq_choices, list) or len(mcq_choices) != 4:

                    mcq_choices = make_mcq_fallback(
                        best_score, best_display_score, display_month_year, benchmark_label
                    )["mcq_choices"]

                choice_texts = [str(x).strip() for x in mcq_choices]

                if best_display_score not in choice_texts:

                    mcq_choices = make_mcq_fallback(
                        best_score, best_display_score, display_month_year, benchmark_label
                    )["mcq_choices"]

                    choice_texts = [str(x).strip() for x in mcq_choices]

                if not isinstance(mcq_answer_key, int) or not (0 <= mcq_answer_key < 4):

                    try:

                        mcq_answer_key = choice_texts.index(best_display_score)

                    except Exception:

                        mcq_answer_key = 1

            if not mcq_background.endswith("."):

                mcq_background += "."

            date_prompt = f"""
Return ONLY a JSON object with the exact keys:
  "date_background": string,
  "date_prediction_prompt": string

Constraints:
- date_background: 1-2 sentence factual background. DO NOT include the best score ({best_display_score}) in the background.
- date_prediction_prompt: Ask "by which date in the future do you think an AI system will reach {best_display_score} on {benchmark_label}?".
- Explicitly include the instruction: "Return in YYYY-MM format." at the end.
- Return only the JSON object.

Benchmark Name: {benchmark_label}
Description: {description}
Current Best Score (as recorded in the CSV): {best_display_score}
Best score came from model column: {best_col}
Best raw cell text: {best_raw_cell}
"""

            generated_date = call_gpt(date_prompt, temperature=0.6, max_tokens=700)

            parsed_date = safe_json_loads(generated_date)

            if not parsed_date:

                generated_date = call_gpt(
                    "Return ONLY the JSON object (no commentary).\n\n" + date_prompt,
                    temperature=0.2,
                    max_tokens=700,
                )

                parsed_date = safe_json_loads(generated_date)

            if not parsed_date:

                date_background = background_short

                date_prediction_prompt = (
                    f"By which date in the future do you think an AI system will reach {best_display_score} "
                    f"on {benchmark_label}? Return in YYYY-MM format."
                )

            else:

                date_background = str(parsed_date.get("date_background", background_short)).strip()

                date_prediction_prompt = str(
                    parsed_date.get(
                        "date_prediction_prompt",
                        f"By which date in the future do you think an AI system will reach {best_display_score} on {benchmark_label}? Return in YYYY-MM format.",
                    )
                ).strip()

            if not date_background.endswith("."):

                date_background += "."

            entry = {
                "id": str(uuid.uuid4()),
                "row_index": int(idx),
                "benchmark_label": benchmark_label,
                "publication_date": publication_date,
                "best_score_model": best_col,
                "best_score_cell_raw": (
                    None if best_raw_cell is None or pd.isna(best_raw_cell) else str(best_raw_cell)
                ),
                "best_score_cell_display": best_display_score,
                "best_score_value": best_score,
                "model_columns_considered": model_cols,
                "binary_question": binary_question,
                "binary_question_perturbed": binary_question_perturbed,
                "binary_perturbation_detail": (
                    f"Realistic threshold = {realistic_threshold_display}, "
                    f"Inflated threshold = {inflated_threshold_display}"
                ),
                "mcq_question": f"{mcq_background} {mcq_question}",
                "mcq_choices": mcq_choices,
                "mcq_answer_key": mcq_answer_key,
                "date_prediction_prompt": f"{date_background} {date_prediction_prompt}",
                "ground_truth_date": publication_date,
            }

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            out_f.flush()

            time.sleep(0.15)

        except Exception as e:

            print(f"Error processing row {idx}: {e}", file=sys.stderr)

            continue

print(f"Done. JSONL written to {OUTPUT_JSONL}")
