import os

import sys

import json

import time

import argparse

import uuid

import re

import random

from datetime import datetime

from functools import wraps

from dotenv import load_dotenv

import pandas as pd

import requests

from tqdm import tqdm

load_dotenv()

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")

API_VERSION = os.environ.get("AZURE_OPENAI_API_VER", "2024-12-01-preview")

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

if not AZURE_ENDPOINT or not AZURE_KEY:

    raise RuntimeError(
        "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in your environment or .env file."
    )


client = None

try:

    from openai import AzureOpenAI

    client = AzureOpenAI(api_version=API_VERSION, azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY)

except Exception:

    class AzureRESTAdapter:

        def __init__(self, endpoint, api_key, api_version):

            self.endpoint = endpoint.rstrip("/")

            self.api_key = api_key

            self.api_version = api_version

        def chat_completions_create(
            self, deployment_id, messages, temperature=0.0, max_tokens=1024
        ):

            url = f"{self.endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version={self.api_version}"

            headers = {"api-key": self.api_key, "Content-Type": "application/json"}

            payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}

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

                            return self._o._o.chat_completions_create(
                                deployment, messages, temperature=temperature, max_tokens=max_tokens
                            )

                    return CC(self)

            return C(self)

    client = AzureRESTAdapter(AZURE_ENDPOINT, AZURE_KEY, API_VERSION)


def retry_on_exception(max_retries=3, base_delay=1.0, allowed_exceptions=(Exception,)):

    def _decorator(fn):

        @wraps(fn)
        def _wrapped(*args, **kwargs):

            last_exc = None

            for attempt in range(1, max_retries + 1):

                try:

                    return fn(*args, **kwargs)

                except allowed_exceptions as e:

                    last_exc = e

                    if attempt == max_retries:

                        raise

                    time.sleep(base_delay * (2 ** (attempt - 1)))

            raise last_exc

        return _wrapped

    return _decorator


def extract_text_from_response(resp):

    if resp is None:

        return ""

    if isinstance(resp, str):

        return resp

    if isinstance(resp, dict):

        if "choices" in resp and isinstance(resp["choices"], list) and len(resp["choices"]) > 0:

            first = resp["choices"][0]

            if isinstance(first, dict):

                if (
                    "message" in first
                    and isinstance(first["message"], dict)
                    and "content" in first["message"]
                ):

                    return first["message"]["content"]

                if "text" in first:

                    return first["text"]

        if "output" in resp and isinstance(resp["output"], list):

            parts = []

            for o in resp["output"]:

                if isinstance(o, dict) and "content" in o and isinstance(o["content"], list):

                    for c in o["content"]:

                        if isinstance(c, dict) and "text" in c:

                            parts.append(c["text"])

                        elif isinstance(c, str):

                            parts.append(c)

                elif isinstance(o, str):

                    parts.append(o)

            if parts:

                return "\n".join(parts)

        for key in ("text", "response", "output_text"):

            if key in resp and isinstance(resp[key], str):

                return resp[key]

        return json.dumps(resp)

    if hasattr(resp, "choices") and isinstance(resp.choices, list) and len(resp.choices) > 0:

        first = resp.choices[0]

        if hasattr(first, "message") and hasattr(first.message, "content"):

            return first.message.content

    return str(resp)


@retry_on_exception(max_retries=3, base_delay=1.0)
def call_chat_completion(messages, deployment=DEPLOYMENT, temperature=0.0, max_tokens=1024):

    last_exc = None

    try:

        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and callable(getattr(client.chat.completions, "create"))
        ):

            resp = client.chat.completions.create(
                model=deployment, messages=messages, temperature=temperature, max_tokens=max_tokens
            )

            return extract_text_from_response(resp)

    except Exception as e:

        last_exc = e

    try:

        if hasattr(client, "chat_completions_create"):

            resp = client.chat_completions_create(
                deployment, messages, temperature=temperature, max_tokens=max_tokens
            )

            return extract_text_from_response(resp)

    except Exception as e:

        last_exc = e

    raise RuntimeError(f"LLM call failed. Last error: {last_exc}")


def extract_json_from_text(text):

    if not text or not isinstance(text, str):

        return None

    start = text.find("{")

    if start == -1:

        return None

    depth = 0

    for i in range(start, len(text)):

        if text[i] == "{":

            depth += 1

        elif text[i] == "}":

            depth -= 1

            if depth == 0:

                candidate = text[start : i + 1]

                try:

                    return json.loads(candidate)

                except Exception:

                    cleaned = re.sub(r",\s*}", "}", candidate)

                    cleaned = re.sub(r",\s*]", "]", cleaned)

                    try:

                        return json.loads(cleaned)

                    except Exception:

                        return None

    return None


def parse_date_safe(s):
    """Return a date object for YYYY-MM-DD, or the raw 'YYYY-MM' string for month-only dates."""

    if s is None:

        return None

    s_str = str(s).strip()

    if re.fullmatch(r"\d{4}-\d{2}", s_str):

        return s_str

    try:

        return datetime.strptime(s_str, "%Y-%m-%d").date()

    except Exception:

        try:

            return pd.to_datetime(s).date()

        except Exception:

            return None


def fmt_iso_to_readable(iso_date_str):
    """Convert 'YYYY-MM-DD' → 'Month Day, Year' or 'YYYY-MM' → 'Month Year'."""

    try:

        d = datetime.strptime(iso_date_str, "%Y-%m-%d").date()

        return f"{d.strftime('%B')} {d.day}, {d.year}"

    except Exception:

        pass

    m = re.fullmatch(r"(\d{4})-(\d{2})", iso_date_str)

    if m:

        try:

            d = datetime.strptime(f"{m.group(1)}-{m.group(2)}-01", "%Y-%m-%d").date()

            return f"{d.strftime('%B')} {d.year}"

        except Exception:

            pass

    return iso_date_str


CALL_A_SYSTEM = (
    "You are an expert research scientist. Read the user's abstract and RETURN EXACTLY one JSON object. "
    "The JSON must have three keys: 'results_and_metrics', 'technical_approach', and 'problem_statement'.\n\n"
    "- 'results_and_metrics': A single sentence capturing ONLY the measurable outcomes, performance numbers, "
    "benchmark results, or demonstrated capabilities across ANY scientific domain (e.g., AI accuracy, biological activity, physical limits). "
    "You MUST include specific quantitative details, exact percentage improvements, precise experimental conditions, "
    "and actual public benchmark or entity names if present in the abstract. Do NOT write a vague or generalized "
    "summary (e.g., do NOT say 'improves accuracy' or 'increases efficiency', instead say 'achieves 94.2% accuracy on X benchmark', "
    "'increases protein binding affinity by 2-fold', or 'synthesizes a material with a superconducting transition at 135 K'). "
    "DO NOT mention the method, architecture, or technique used — only the verifiable outcome. Replace specific proposed model or system names with 'a system' or 'a method' where appropriate. "
    "CRITICAL: Do NOT use any novel terms, metric names, or concepts that are introduced for the first time in this paper (e.g., 'deep-thinking ratio', 'Grokked-Score'). "
    "A model from before the knowledge cutoff will not know these terms. Instead, describe them functionally (e.g., 'the proportion of tokens undergoing significant internal revisions').\n"
    "- 'technical_approach': A detailed, technical, method-oriented specification of HOW the result was achieved. "
    "Include the specific mechanism, experimental design, architectural shift, or algorithmic innovation "
    "(e.g., 'uses sparse autoencoders to map internal activations', 'employs a high-pressure diamond anvil cell', "
    "or 'targets the XYZ pathway via a small-molecule inhibitor'). DO NOT include specific specific proposed model names or brands; "
    "replace them with 'a system' or 'a method'. CRITICAL: Do NOT include any named techniques, "
    "named algorithms, novel terms introduced in the paper, or acronyms (e.g., do NOT say 'GRPO', 'CRISPR', 'NMR', 'LoRA'). "
    "Instead, describe what the technique DOES mechanistically "
    "(e.g., instead of 'GRPO', say 'a group-level relative policy optimization that compares ' "
    "'multiple outputs', or instead of 'CRISPR-Cas9', say 'an RNA-guided endonuclease system that induces targeted double-strand breaks'). "
    "This field is for internal answer-key use only.\n"
    "- 'problem_statement': A detailed technical description (3–4 sentences) of the research problem "
    "and the limitations of previous methods. Describe what was broken, missing, or inadequate BEFORE "
    "this paper existed. CRITICAL: Do NOT mention anything about the solution, method, or results — "
    "describe only the 'lock', never the 'key'. Do NOT include any narrative about the paper or authors. "
    "Do NOT use novel terms introduced in the paper.\n\n"
    "Return JSON only. Focus on technical precision over brevity."
)

CALL_A_USER = "Abstract:\n{abstract}\n\nProduce the JSON described above. Return JSON only."

CALL_B_SYSTEM = (
    "You are a forecasting question writer for scientific benchmarks. "
    "You produce precise, resolution-ready binary questions with verifiable outcomes. "
    'Return EXACTLY one JSON object: {"question":"..."} and nothing else.'
)

CALL_B_USER = (
    'Problem context: "{problem_statement}"\n'
    'Result achieved: "{results_and_metrics}"\n'
    "Target date: {human_date}\n\n"
    "Write ONE binary (Yes/No) forecasting question.\n\n"
    "Rules:\n"
    "- The question must begin with: 'Will a method' or 'By {human_date}, will a method'.\n"
    "- EXACT GRAMMATICAL FLUENCY IS CRITICAL. You must adapt the verb tense from 'Result achieved' to fit the sentence.\n"
    "  Bad Example: Will a method achieve Achieves a robust positive correlation...\n"
    "  Good Example: Will a method achieve a robust positive correlation...\n"
    "  Bad Example: By 2026-03, will a method Demonstrates a 10% increase...\n"
    "  Good Example: By 2026-03, will a method demonstrate a 10% increase...\n"
    "- Do not blindly copy-paste if it breaks grammar. ALWAYS fix the verb!\n"
    "- End the question with 'by {human_date}?' if not already at the start.\n"
    "- Refer ONLY to measurable results (metrics, benchmarks, numeric improvements, or concrete capabilities).\n"
    "- Do NOT mention any methods, techniques, architectures, or how the result is achieved.\n"
    "- Do NOT use words like 'demonstrate', 'enable', or 'by using'.\n"
    "- Keep it to a single clear sentence.\n\n"
    "Return JSON with key 'binary_question'."
)


CALL_B_NEG_SYSTEM = (
    "You are a research evaluation expert who designs counterfactual scenarios "
    "for academic forecasting benchmarks. Your role is to create plausible but "
    "INCORRECT alternative result claims. Return EXACTLY one JSON object."
)

CALL_B_NEG_USER = (
    'Original result claim: "{results_and_metrics}"\n'
    'Problem context: "{problem_statement}"\n\n'
    "Create a COUNTERFACTUAL ALTERNATIVE version of this result claim that is plausible-sounding "
    "but was NOT actually achieved.\n\n"
    "RULES:\n"
    "1. Keep ALL benchmark names, dataset names, and task names EXACTLY the same. "
    "Do NOT change which benchmark or dataset is referenced.\n"
    "2. ONLY modify an EXISTING numeric score/threshold, or add a credible unmet constraint. "
    "3. IF modifying an existing numeric score, RAISE it enough so the original result definitively "
    "does NOT satisfy the perturbed claim (e.g., if original is 94.2%, change to 95.8%; if 51.7%, change to 54.5%). "
    "Make the increase a clear shift so there is no ambiguity, but still physically plausible.\n"
    "4. IF the original claim has no specific numbers, you MUST add a highly specific, definitive unmet constraint "
    "Make this constraint significant enough that it's noticeably harder to satisfy than the original.\n"
    "5. The perturbed claim must be plausible and not absurd.\n"
    "6. Keep the same length, style, and level of specificity.\n\n"
    "Return JSON with:\n"
    "- 'perturbed_result': The counterfactual alternative result claim\n"
    "- 'changed_detail': Which aspect of the result was modified"
)

CALL_C_SYSTEM = (
    "You are a technical forecasting analyst who designs extraordinarily difficult, graduate-level evaluations. "
    "Your task is to create a multiple-choice question that tests whether an expert "
    "can predict the specific technical path taken to solve a research challenge. "
    "The distractors must be GENUINELY PLAUSIBLE AND HIGHLY DECEPTIVE — a PhD-level expert "
    "should struggle to identify the correct answer unless they know the exact paper. "
    "Return EXACTLY one JSON object."
)

CALL_C_USER = (
    'Problem Statement: "{problem_statement}"\n'
    'Result Achieved: "{results_and_metrics}"\n'
    'Correct Approach (for choice generation ONLY — DO NOT leak into the question stem): "{technical_approach}"\n'
    "Target Date: {human_date}\n\n"
    "Generate a very difficult expert-level MCQ. Return JSON only.\n\n"
    "STEM REQUIREMENTS:\n"
    "- The stem must explicitly but briefly summarize the core challenge from the Problem Statement (in 1-2 clauses max),\n"
    "  followed by asking which proposed solution will achieve the Result Achieved by {human_date}.\n"
    "- Example structure: 'Given the challenge of [Problem Statement summary], which of the following approaches is most likely to achieve [Result Achieved] by {human_date}?'\n"
    "- Make it read naturally as a forward-looking forecasting question.\n"
    "- Do NOT use retrospective, past-tense wording (e.g., avoid 'was introduced' or 'achieved'). Treat the target date as a future milestone.\n"
    "- Embed the measurable outcome from Result Achieved.\n"
    "- Do NOT mention any terminology from the Correct Approach in the stem.\n\n"
    "CHOICE REQUIREMENTS:\n"
    "- Provide exactly 4 choices.\n"
    "- The distractors MUST be extremely difficult. They should represent real, highly competitive alternative approaches that experts would genuinely consider for the same problem.\n"
    "- CRITICAL: All choices MUST BE EXTREMELY SHORT AND CONCISE (maximum 15-20 words). Do NOT write long, multi-clause paragraphs. State only the core mechanism.\n"
    "- The incorrect answers must solve the exact same problem statement and theoretically achieve the exact same result, differing ONLY in the core mechanism.\n"
    "- FORBIDDEN DISTRACTORS: No antonyms, no obvious negatives, no generic scaling answers, no trivial ablations. Do not make distractors sound worse or less effective than the correct answer.\n"
    "- Do NOT use named algorithms, novel terms introduced in the paper, or acronyms; describe mechanisms functionally instead.\n"
    "- Ensure all choices have identical length, structure, and academic tone.\n\n"
    "Return JSON with keys:\n"
    "- 'question'\n"
    "- 'choices' (array of 4 strings; first is correct)\n"
    "- 'answer_key' (0)\n"
)

CALL_D_SYSTEM = (
    "You are a scientific task-setter who designs research challenge prompts. "
    "Your goal is to write a prompt that gives a researcher a problem and asks them for a proposed solution. "
    "Return EXACTLY one JSON object."
)

CALL_D_USER = (
    'Problem Statement: "{problem_statement}"\n'
    "Deadline: {human_date}\n\n"
    "Write a concise free-response prompt (max 60 words) with this structure:\n"
    "  'Given [problem description], propose a concrete method that could solve this problem by [date]. Provide: (A) a high-level method description, (B) a technical implementation plan.'\n\n"
    "RULES:\n"
    "1. The problem description must come ONLY from the Problem Statement.\n"
    "2. DO NOT mention any specific method, architecture, technique, or approach.\n"
    "3. DO NOT include any narrative about a paper or discovery.\n\n"
    "Return JSON with key 'prompt'. Return JSON only."
)


def normalize_result_with_llm(text):
    """
    Use the LLM to rewrite a result claim into a predicate phrase suitable for:
      "By {human_date}, will a method {predicate}?"
    Returns the predicate string (no surrounding quotes, no trailing period).
    Falls back to the original cleaned text on error.
    """

    if not text or not isinstance(text, str):

        return text

    raw = text.strip().strip('"').strip("'").strip()

    raw = raw.rstrip(".").strip()

    system_msg = (
        "You will rewrite the user's RESULT CLAIM into a single short predicate phrase "
        "suitable for insertion after the fragment 'will a method '.\n\n"
        "Constraints:\n"
        "1) Do NOT change facts, numbers, or benchmark/dataset names.\n"
        "2) Return ONLY the predicate phrase (no question mark, no surrounding quotes, no commentary).\n"
        "3) Start with a lower-case verb in infinitive-like form (e.g., 'achieve', 'reach', 'obtain', 'attain').\n"
        "4) Keep it short and factual (under 15 words).\n"
        "5) If the claim already is a predicate, just fix capitalization/punctuation."
    )

    user_msg = raw

    try:

        polished = call_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            deployment=DEPLOYMENT,
            temperature=0.0,
            max_tokens=80,
        )

        if isinstance(polished, str):

            polished = polished.strip().strip('"').strip("'").rstrip(".").strip()

            if polished and re.match(r"^[a-z0-9]", polished):

                return polished

    except Exception:

        pass

    t = raw

    if t and t[0].isalpha():

        t = t[0].lower() + t[1:]

    return t


def sanitize_discovery(discovery: str, human_date: str) -> str:

    if not discovery:

        return discovery

    d = discovery.strip()

    d = d.replace("<DATE>", human_date).replace("<date>", human_date)

    m = re.match(r"^\s*By\s+[A-Za-z0-9\-, ]+,\s*will\s+(.*)$", d, flags=re.I)

    if m:

        d = m.group(1).strip()

        if d and d[0].isupper():

            d = d[0].lower() + d[1:]

        if not d.startswith("there"):

            d = "there be " + d

    d = d.strip().strip('"').strip("'").rstrip(".").strip()

    return d


ALL_QUESTION_TYPES = {"binary", "binary_perturbed", "mcq", "frq", "date_prediction"}


def process_row_separate_calls(row, deployment=DEPLOYMENT, question_types=None):

    if question_types is None:

        question_types = ALL_QUESTION_TYPES

    abstract = row.get("abstract", "") if isinstance(row, dict) else row.abstract

    if isinstance(row, dict):

        date_raw = (
            row.get("date")
            or row.get("date_published")
            or row.get("pubdate")
            or row.get("post_date_iso")
            or row.get("first_date_found")
            or row.get("first_date_found")
        )

        paper_link = (
            row.get("link")
            or row.get("url")
            or row.get("paper_link")
            or row.get("crossref_url")
            or row.get("oa_url")
            or row.get("doi_url")
            or (f"https://doi.org/{row.get('doi')}" if row.get("doi") else None)
        )

    else:

        date_raw = (
            getattr(row, "date", None)
            or getattr(row, "date_published", None)
            or getattr(row, "pubdate", None)
            or getattr(row, "post_date_iso", None)
            or getattr(row, "first_date_found", None)
            or getattr(row, "first_date_found", None)
        )

        paper_link = (
            getattr(row, "link", None)
            or getattr(row, "url", None)
            or getattr(row, "paper_link", None)
            or getattr(row, "crossref_url", None)
            or getattr(row, "oa_url", None)
            or getattr(row, "doi_url", None)
            or (f"https://doi.org/{getattr(row, 'doi')}" if getattr(row, "doi", None) else None)
        )

    parsed = parse_date_safe(date_raw)

    if parsed is None:

        parsed = datetime.utcnow().date()

    pub_date_iso = parsed if isinstance(parsed, str) else parsed.isoformat()

    human_date = fmt_iso_to_readable(pub_date_iso)

    messages_a = [
        {"role": "system", "content": CALL_A_SYSTEM},
        {"role": "user", "content": CALL_A_USER.format(abstract=abstract)},
    ]

    raw_a = call_chat_completion(messages_a, deployment=deployment, temperature=0.0, max_tokens=500)

    json_a = extract_json_from_text(raw_a) or {}

    results_and_metrics = (json_a.get("results_and_metrics") or "").strip()

    technical_approach = (json_a.get("technical_approach") or "").strip()

    problem_statement = (json_a.get("problem_statement") or "").strip()

    if not results_and_metrics:

        results_and_metrics = abstract.split(".")[0].strip()[:240]

    if not technical_approach:

        simple = re.sub(r'["""\u2018\u2019].+?["""\u2018\u2019]', "a system", results_and_metrics)

        simple = re.sub(
            r"\b[A-Za-z]*[A-Z][a-zA-Z0-9]*\b",
            lambda m: "a system" if any(ch.isupper() for ch in m.group(0)[1:]) else m.group(0),
            simple,
        )

        simple = re.sub(r"\s+", " ", simple).strip().rstrip(".")

        technical_approach = (
            f"a method that {simple[0].lower() + simple[1:]}"
            if simple
            else "a novel method addressing this problem"
        )

    if not problem_statement:

        sents = [s.strip() for s in re.split(r"\.\s+", abstract) if s.strip()]

        bg_parts = []

        for s in sents[:4]:

            if results_and_metrics and results_and_metrics.lower() in s.lower():

                continue

            bg_parts.append(s)

            if len(bg_parts) >= 2:

                break

        problem_statement = " ".join(bg_parts)[:300].rstrip(".").strip()

        if not problem_statement:

            problem_statement = "The field faces unresolved methodological challenges and limitations in prior approaches."

    technical_approach = sanitize_discovery(technical_approach, human_date)

    problem_statement = problem_statement.rstrip(".").strip()

    binary_question = None

    binary_question_perturbed = None

    perturbed_result = ""

    perturbation_detail = ""

    if "binary" in question_types or "binary_perturbed" in question_types:

        messages_b = [
            {"role": "system", "content": CALL_B_SYSTEM},
            {
                "role": "user",
                "content": CALL_B_USER.format(
                    problem_statement=problem_statement,
                    results_and_metrics=results_and_metrics,
                    human_date=human_date,
                ),
            },
        ]

        raw_b = call_chat_completion(
            messages_b, deployment=deployment, temperature=0.0, max_tokens=200
        )

        json_b = extract_json_from_text(raw_b) or {}

        binary_question = (
            json_b.get("question")
            or f"By {human_date}, will a method achieve {results_and_metrics}?"
        )

    if "binary_perturbed" in question_types:

        messages_b_neg = [
            {"role": "system", "content": CALL_B_NEG_SYSTEM},
            {
                "role": "user",
                "content": CALL_B_NEG_USER.format(
                    results_and_metrics=results_and_metrics, problem_statement=problem_statement
                ),
            },
        ]

        raw_b_neg = call_chat_completion(
            messages_b_neg, deployment=deployment, temperature=0.4, max_tokens=300
        )

        json_b_neg = extract_json_from_text(raw_b_neg) or {}

        perturbed_result = (json_b_neg.get("perturbed_result") or "").strip()

        perturbation_detail = (json_b_neg.get("changed_detail") or "").strip()

        if perturbed_result:

            perturbed_result = normalize_result_with_llm(perturbed_result)

            messages_b2 = [
                {"role": "system", "content": CALL_B_SYSTEM},
                {
                    "role": "user",
                    "content": CALL_B_USER.format(
                        problem_statement=problem_statement,
                        results_and_metrics=perturbed_result,
                        human_date=human_date,
                    ),
                },
            ]

            raw_b2 = call_chat_completion(
                messages_b2, deployment=deployment, temperature=0.0, max_tokens=200
            )

            json_b2 = extract_json_from_text(raw_b2) or {}

            binary_question_perturbed = (
                json_b2.get("question")
                or f"By {human_date}, will a method achieve {perturbed_result}?"
            )

    mcq_question = None

    mcq_choices = []

    mcq_answer_key = None

    if "mcq" in question_types:

        messages_c = [
            {"role": "system", "content": CALL_C_SYSTEM},
            {
                "role": "user",
                "content": CALL_C_USER.format(
                    technical_approach=technical_approach,
                    problem_statement=problem_statement,
                    results_and_metrics=results_and_metrics,
                    human_date=human_date,
                ),
            },
        ]

        raw_c = call_chat_completion(
            messages_c, deployment=deployment, temperature=0.3, max_tokens=600
        )

        json_c = extract_json_from_text(raw_c) or {}

        mcq_question = (
            json_c.get("question")
            or f"Given this problem, which of the following approaches is most likely to achieve this result by {human_date}?"
        )

        mcq_choices = json_c.get("choices") or [
            technical_approach,
            "A subtle variant using the exact same framework but modifying a single specific mechanism.",
            "A highly competitive alternative that tackles the exact same bottleneck using a credible substitute.",
            "A common pitfall approach that uses similar terminology but represents a slightly different architectural choice.",
        ]

        mcq_choices = [str(c).strip() for c in mcq_choices][:4]

        while len(mcq_choices) < 4:

            mcq_choices.append(
                "A plausible alternative mechanism for addressing the same bottleneck."
            )

        if "answer_key" in json_c:

            try:

                ak = int(json_c["answer_key"])

                if 0 <= ak <= 3:

                    mcq_answer_key = ak

            except Exception:

                mcq_answer_key = None

        if mcq_answer_key is None:

            found = None

            approach_low = technical_approach.lower()

            for idx, ch in enumerate(mcq_choices):

                ch_low = ch.lower()

                if approach_low in ch_low or ch_low in approach_low:

                    found = idx

                    break

            mcq_answer_key = int(found) if found is not None else 0

    frq_prompt = None

    if "frq" in question_types:

        messages_d = [
            {"role": "system", "content": CALL_D_SYSTEM},
            {
                "role": "user",
                "content": CALL_D_USER.format(
                    problem_statement=problem_statement, human_date=human_date
                ),
            },
        ]

        raw_d = call_chat_completion(
            messages_d, deployment=deployment, temperature=0.3, max_tokens=360
        )

        json_d = extract_json_from_text(raw_d) or {}

        frq_prompt = json_d.get("prompt")

        if not frq_prompt:

            frq_prompt = (
                f"Given the problem: '{problem_statement}', propose a concrete method that could achieve "
                f"'{results_and_metrics}' by {human_date}. "
                "Provide: (A) a high-level method description, (B) a technical implementation plan."
            )

    date_prediction_prompt = None

    if "date_prediction" in question_types:

        raw_date_prompt = (
            f"By what month and year (in YYYY-MM format) do you predict a method will "
            f"{results_and_metrics}? Respond with exactly one date in YYYY-MM format."
        )

        grammar_messages = [
            {
                "role": "system",
                "content": (
                    "Fix ONLY the grammar and fluency of the following question. "
                    "Do NOT add, remove, or rephrase any information. "
                    "Do NOT add explanations. Return ONLY the corrected question text."
                ),
            },
            {"role": "user", "content": raw_date_prompt},
        ]

        try:

            polished = call_chat_completion(
                grammar_messages, deployment=deployment, temperature=0.0, max_tokens=150
            )

            if polished and len(polished) < len(raw_date_prompt) * 2:

                date_prediction_prompt = polished.strip().strip('"').strip("'")

            else:

                date_prediction_prompt = raw_date_prompt

        except Exception:

            date_prediction_prompt = raw_date_prompt

    out = {
        "id": str(uuid.uuid4()),
        "row_index": row.get("_row_index", None) if isinstance(row, dict) else None,
        "publication_date": pub_date_iso,
        "results_and_metrics": results_and_metrics,
        "technical_approach": technical_approach,
        "problem_statement": problem_statement,
        "binary_question": binary_question,
        "binary_question_perturbed": binary_question_perturbed,
        "binary_perturbed_result": perturbed_result,
        "binary_perturbation_detail": perturbation_detail,
        "mcq_question": mcq_question,
        "mcq_choices": mcq_choices,
        "mcq_answer_key": mcq_answer_key,
        "frq_prompt": frq_prompt,
        "date_prediction_prompt": date_prediction_prompt,
        "ground_truth_date": pub_date_iso,
        "paper_link": paper_link,
        "source_abstract": abstract,
    }

    return out


def main():

    parser = argparse.ArgumentParser(
        description="Create HF-ready forecasting benchmark JSONL (discovery + background)."
    )

    parser.add_argument(
        "csv_in", help="Input CSV path (requires 'abstract' and 'date' or 'date_published' columns)"
    )

    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional: max rows to process (for testing)"
    )

    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["all"],
        choices=["all", "binary", "binary_perturbed", "mcq", "frq", "date_prediction"],
        help="Which question types to generate (default: all). " "E.g. --question-types binary mcq",
    )

    args = parser.parse_args()

    if "all" in args.question_types:

        selected_types = ALL_QUESTION_TYPES

    else:

        selected_types = set(args.question_types)

    print(f"Question types to generate: {', '.join(sorted(selected_types))}")

    inpath = args.csv_in

    if not os.path.exists(inpath):

        print("Input CSV not found:", inpath, file=sys.stderr)

        sys.exit(2)

    df = pd.read_csv(inpath)

    if "abstract" not in df.columns or not (
        ("date" in df.columns)
        or ("date_published" in df.columns)
        or ("pubdate" in df.columns)
        or ("post_date_iso" in df.columns)
        or ("first_date_found" in df.columns)
        or ("first_date_found" in df.columns)
    ):

        print(
            "Input CSV must contain 'abstract' and either 'date', 'date_published', 'pubdate', 'post_date_iso', or 'first_date_found' columns.",
            file=sys.stderr,
        )

        sys.exit(2)

    if selected_types == ALL_QUESTION_TYPES:

        type_suffix = "_benchmark_all"

    else:

        type_suffix = "_benchmark_" + "_".join(sorted(selected_types))

    outpath = os.path.splitext(inpath)[0] + type_suffix + ".jsonl"

    total = len(df)

    limit = args.max_rows if args.max_rows is not None else total

    print(
        f"Processing up to {limit} rows from {total} total -> {outpath} (deployment={DEPLOYMENT})"
    )

    written = 0

    with open(outpath, "w", encoding="utf-8") as fout:

        for i in tqdm(range(min(limit, total)), desc="Generating Benchmark"):

            row_series = df.iloc[i]

            row = row_series.to_dict()

            row["_row_index"] = int(i)

            try:

                result = process_row_separate_calls(
                    row, deployment=DEPLOYMENT, question_types=selected_types
                )

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

                fout.flush()

                written += 1

                time.sleep(0.25)

            except Exception as e:

                print(f"Error processing row {i}: {e}", file=sys.stderr)

                fallback = {
                    "id": str(uuid.uuid4()),
                    "row_index": i,
                    "publication_date": row.get("date")
                    or row.get("date_published")
                    or row.get("pubdate")
                    or row.get("post_date_iso")
                    or row.get("first_date_found")
                    or row.get("first_date_found"),
                    "results_and_metrics": None,
                    "technical_approach": None,
                    "problem_statement": None,
                    "binary_question": None,
                    "mcq_question": None,
                    "mcq_choices": [],
                    "mcq_answer_key": None,
                    "frq_prompt": None,
                    "paper_link": (
                        row.get("link")
                        or row.get("url")
                        or row.get("paper_link")
                        or row.get("crossref_url")
                        or row.get("oa_url")
                        or row.get("doi_url")
                        or (f"https://doi.org/{row.get('doi')}" if row.get("doi") else None)
                    ),
                    "source_abstract": row.get("abstract", ""),
                }

                fout.write(json.dumps(fallback, ensure_ascii=False) + "\n")

                fout.flush()

    print(f"Done — wrote {written} items to {outpath}")


if __name__ == "__main__":

    main()
