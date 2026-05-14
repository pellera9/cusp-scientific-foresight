import os

import json

import logging

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from openai import AzureOpenAI

try:

    from dotenv import load_dotenv

    load_dotenv()

except ImportError:

    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")

AZURE_OPENAI_API_VER = os.environ.get("AZURE_OPENAI_API_VER", "2024-12-01-preview")

DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

INPUT_FILE = "../benchmark_data/CUSP/merged_validated_cusp.jsonl"

OUTPUT_FILE = "../benchmark_data/CUSP/merged_validated_cusp_fixed.jsonl"

LOG_FILE = "../benchmark_data/CUSP/grammar_changes.log"

SYSTEM_PROMPT = """You are a scientific proofreader. You will be given a question created for a forecasting benchmark.
Sometimes the question is created by naively stringing together a prefix like "By <date>, will a method achieve " with a full result sentence.
This results in grammatically awkward sentences with incorrect verb tenses or duplicated subject-verb combinations. For example:
Bad: "By March 2026, will a method achieve The infrastructure demonstrated consistent propagation..."
Good: "By March 2026, will a method achieve infrastructure demonstrating consistent propagation..." 
OR Good: "By March 2026, will a method involve an infrastructure that demonstrates consistent propagation..."

Bad: "By what month and year (in YYYY-MM format) do you predict a method will the infrastructure demonstrate consistent propagation..."
Good: "By what month and year (in YYYY-MM format) do you predict a method will be introduced where the infrastructure demonstrates consistent propagation..."

Your job is to fix the grammar so it reads smoothly as a proper English question, without changing any of the meaning, technical terminology, dates, or numbers.
If the grammar is already perfect, do not change it.

You must respond with a JSON object with two fields:
{
  "needs_fixing": <boolean>,
  "fixed_text": "<the fixed question, or the exact original text if no fixing is needed>"
}
"""


def get_client() -> AzureOpenAI:

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VER,
    )


def fix_text(client: AzureOpenAI, text: str) -> str:

    if not text or not isinstance(text, str):

        return text

    try:

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=400,
        )

        result_json = response.choices[0].message.content

        result = json.loads(result_json)

        if result.get("needs_fixing") and result.get("fixed_text"):

            return result["fixed_text"]

        else:

            return text

    except Exception as e:

        logging.error(f"Error calling LLM or parsing: {e} | Text: {text[:50]}...")

        return text


def process_item(item: dict, max_retries=3) -> tuple:

    client = get_client()

    changes = []

    fields_to_check = ["binary_question", "binary_question_perturbed", "date_prediction_prompt"]

    for field in fields_to_check:

        if field in item and item[field]:

            original_text = item[field]

            for attempt in range(max_retries):

                try:

                    fixed_text = fix_text(client, original_text)

                    if fixed_text and fixed_text != original_text:

                        item[field] = fixed_text

                        changes.append(
                            {
                                "id": item.get("id", "unknown_id"),
                                "field": field,
                                "original": original_text,
                                "fixed": fixed_text,
                            }
                        )

                    break

                except Exception as e:

                    if attempt == max_retries - 1:

                        logging.error(
                            f"Failed to process field {field} after {max_retries} attempts: {e}"
                        )

                    else:

                        pass

    return item, changes


def main():

    input_path = os.path.abspath(INPUT_FILE)

    output_path = os.path.abspath(OUTPUT_FILE)

    if not os.path.exists(input_path):

        logging.error(f"Input file not found: {input_path}")

        return

    logging.info(f"Reading from {input_path}")

    items = []

    with open(input_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                items.append(json.loads(line))

    logging.info(f"Loaded {len(items)} items. Processing with Azure GPT-4o-mini...")

    all_changes = []

    max_workers = 20

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(process_item, item): i for i, item in enumerate(items)}

        with tqdm(total=len(items), desc="Fixing grammar") as pbar:

            for future in as_completed(futures):

                try:

                    processed_item, item_changes = future.result()

                    idx = futures[future]

                    items[idx] = processed_item

                    all_changes.extend(item_changes)

                except Exception as e:

                    logging.error(f"Exception during processing: {e}")

                pbar.update(1)

    logging.info(f"Writing to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:

        for item in items:

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log_path = os.path.abspath(LOG_FILE)

    logging.info(f"Writing changes log to {log_path}")

    with open(log_path, "w", encoding="utf-8") as lf:

        lf.write(f"Total changes made: {len(all_changes)}\n")

        lf.write("=" * 80 + "\n\n")

        for c in all_changes:

            lf.write(f"ID: {c['id']}\n")

            lf.write(f"Field: {c['field']}\n")

            lf.write(f"Original: {c['original']}\n")

            lf.write(f"Fixed:    {c['fixed']}\n")

            lf.write("-" * 80 + "\n")

    logging.info("Done!")


if __name__ == "__main__":

    main()
