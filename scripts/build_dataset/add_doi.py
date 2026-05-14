import pandas as pd

import re

import sys


def add_doi_column(input_csv):
    """
    Reads a CSV file, extracts DOI from 'paper_link' column,
    adds it as a new column called 'DOI',
    and overwrites the original file.
    """

    df = pd.read_csv(input_csv)

    if "paper_link" not in df.columns:

        raise ValueError("The file does not contain a 'paper_link' column.")

    df["DOI"] = df["paper_link"].str.extract(r"(10\.\S+)")

    df.to_csv(input_csv, index=False)

    print("DOI column added successfully.")

    print(f"File overwritten: {input_csv}")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python script.py <input_csv>")

        sys.exit(1)

    input_file = sys.argv[1]

    add_doi_column(input_file)
