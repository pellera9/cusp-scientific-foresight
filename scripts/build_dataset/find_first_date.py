"""
find_first_date.py — Find the earliest publication / preprint date for papers in a CSV.

Usage:
    python find_first_date.py <csv_path>

Searches Crossref, Semantic Scholar, OpenAlex, Europe PMC, arXiv, and
bioRxiv/medRxiv for each DOI. Adds two new columns:
    first_date_found  — YYYY-MM of the earliest date discovered
    first_date_source — which API returned that date

Results are cached in doi_date_cache.json next to the CSV so re-runs are instant.
"""

import argparse

import asyncio

import json

import os

import re

import sys

import time

import xml.etree.ElementTree as ET

from pathlib import Path

import aiohttp

import pandas as pd

from tqdm.asyncio import tqdm as atqdm

MAX_CONCURRENT = 15

TIMEOUT_SEC = 30

MAX_RETRIES = 3

CACHE_SAVE_INTERVAL = 50

HEADERS = {
    "User-Agent": "FutureScience-DateFinder/1.0 (mailto:research@example.com)",
    "Accept": "application/json",
}


def _ym(date_str: str | None) -> str | None:
    """Normalise any date-ish string to 'YYYY-MM' or None."""

    if not date_str:

        return None

    date_str = str(date_str).strip()

    m = re.match(r"(\d{4})[-/](\d{1,2})", date_str)

    if m:

        return f"{m.group(1)}-{int(m.group(2)):02d}"

    m = re.match(r"^(\d{4})$", date_str)

    if m:

        return f"{m.group(1)}-01"

    return None


def _earliest(*dates: str | None) -> str | None:
    """Return the lexicographically smallest YYYY-MM (i.e. earliest)."""

    valid = [d for d in dates if d]

    return min(valid) if valid else None


def _cache_path(csv_path: str) -> str:

    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), "doi_date_cache.json")


def load_cache(csv_path: str) -> dict:

    p = _cache_path(csv_path)

    if os.path.exists(p):

        with open(p) as f:

            return json.load(f)

    return {}


def save_cache(csv_path: str, cache: dict):

    p = _cache_path(csv_path)

    with open(p, "w") as f:

        json.dump(cache, f, indent=1)


async def _get_json(
    session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, params: dict | None = None
) -> dict | None:
    """GET JSON with retry + backoff."""

    for attempt in range(MAX_RETRIES):

        try:

            async with sem:

                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)
                ) as r:

                    if r.status == 200:

                        return await r.json(content_type=None)

                    if r.status == 429:

                        wait = int(r.headers.get("Retry-After", 2**attempt))

                        await asyncio.sleep(min(wait, 30))

                        continue

                    if r.status in (404, 406, 400):

                        return None

                    await asyncio.sleep(2**attempt)

        except (aiohttp.ClientError, asyncio.TimeoutError):

            await asyncio.sleep(2**attempt)

    return None


async def _get_text(
    session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, params: dict | None = None
) -> str | None:

    for attempt in range(MAX_RETRIES):

        try:

            async with sem:

                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)
                ) as r:

                    if r.status == 200:

                        return await r.text()

                    if r.status == 429:

                        wait = int(r.headers.get("Retry-After", 2**attempt))

                        await asyncio.sleep(min(wait, 30))

                        continue

                    if r.status in (404, 406, 400):

                        return None

                    await asyncio.sleep(2**attempt)

        except (aiohttp.ClientError, asyncio.TimeoutError):

            await asyncio.sleep(2**attempt)

    return None


async def fetch_crossref(session, doi, sem):
    """Crossref: created, posted, published-online, published-print dates."""

    results = []

    data = await _get_json(session, f"https://api.crossref.org/works/{doi}", sem)

    if not data:

        return results

    msg = data.get("message", {})

    for field in ("created", "posted", "published-online", "published-print", "issued"):

        parts = msg.get(field, {}).get("date-parts", [[]])

        if parts and parts[0]:

            p = parts[0]

            yr = p[0] if len(p) > 0 else None

            mo = p[1] if len(p) > 1 else 1

            if yr:

                results.append((_ym(f"{yr}-{mo}"), f"Crossref ({field})"))

    return results


async def fetch_semantic_scholar(session, doi, sem):
    """Semantic Scholar: publicationDate + external arXiv ID."""

    results = []

    arxiv_id = None

    data = await _get_json(
        session,
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
        sem,
        params={"fields": "publicationDate,externalIds"},
    )

    if not data:

        return results, arxiv_id

    pub = _ym(data.get("publicationDate"))

    if pub:

        results.append((pub, "Semantic Scholar"))

    ext = data.get("externalIds") or {}

    arxiv_id = ext.get("ArXiv")

    return results, arxiv_id


async def fetch_openalex(session, doi, sem):
    """OpenAlex: publication_date, best_oa_location dates."""

    results = []

    data = await _get_json(
        session,
        f"https://api.openalex.org/works/doi:{doi}",
        sem,
        params={"mailto": "research@example.com"},
    )

    if not data:

        return results

    pub = _ym(data.get("publication_date"))

    if pub:

        results.append((pub, "OpenAlex"))

    for loc_key in ("primary_location", "best_oa_location"):

        loc = data.get(loc_key, {}) or {}

        src = loc.get("source", {}) or {}

        src_type = src.get("type", "")

        if src_type == "repository":

            pub2 = _ym(data.get("publication_date"))

            if pub2:

                results.append((pub2, f"OpenAlex (repository)"))

    for loc in data.get("locations") or []:

        if loc and loc.get("source", {}) and loc["source"].get("type") == "repository":

            pass

    return results


async def fetch_europepmc(session, doi, sem):
    """Europe PMC: firstPublicationDate, firstIndexDate."""

    results = []

    data = await _get_json(
        session,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        sem,
        params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"},
    )

    if not data:

        return results

    result_list = data.get("resultList", {}).get("result", [])

    for item in result_list[:3]:

        fp = _ym(item.get("firstPublicationDate"))

        if fp:

            results.append((fp, "Europe PMC"))

        fi = _ym(item.get("firstIndexDate"))

        if fi:

            results.append((fi, "Europe PMC (index)"))

        pub_type = item.get("pubType", "")

        if "preprint" in str(pub_type).lower():

            fp2 = _ym(item.get("firstPublicationDate"))

            if fp2:

                results.append((fp2, "Europe PMC (preprint)"))

    return results


def _parse_arxiv_entries(text, source_label="arXiv"):
    """Parse arXiv Atom XML and extract (YYYY-MM, source) tuples."""

    results = []

    if not text:

        return results

    try:

        root = ET.fromstring(text)

        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):

            id_text = entry.findtext("atom:id", namespaces=ns) or ""

            if "api/errors" in id_text:

                continue

            pub = entry.findtext("atom:published", namespaces=ns)

            ym = _ym(pub)

            if ym:

                results.append((ym, source_label))

            upd = entry.findtext("atom:updated", namespaces=ns)

            ym2 = _ym(upd)

            if ym2:

                results.append((ym2, f"{source_label} (updated)"))

    except ET.ParseError:

        pass

    return results


async def fetch_arxiv_by_id(session, arxiv_id, sem):
    """arXiv API: lookup by arXiv ID."""

    if not arxiv_id:

        return []

    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    text = await _get_text(
        session,
        "https://export.arxiv.org/api/query",
        sem,
        params={"id_list": clean_id, "max_results": "1"},
    )

    return _parse_arxiv_entries(text, "arXiv")


async def fetch_arxiv_by_title(session, title, sem):
    """arXiv API: search by paper title to find preprints."""

    if not title:

        return []

    clean = re.sub(r"[^\w\s]", " ", title).strip()

    words = clean.split()[:15]

    query = 'ti:"' + " ".join(words) + '"'

    text = await _get_text(
        session,
        "https://export.arxiv.org/api/query",
        sem,
        params={"search_query": query, "max_results": "3"},
    )

    return _parse_arxiv_entries(text, "arXiv (title search)")


async def fetch_biorxiv(session, doi, sem):
    """bioRxiv / medRxiv: preprint posting dates."""

    results = []

    for server in ("biorxiv", "medrxiv"):

        data = await _get_json(
            session,
            f"https://api.biorxiv.org/details/{server}/10.1101/{doi.split('/')[-1]}",
            sem,
        )

        if data and data.get("collection"):

            for item in data["collection"][:5]:

                d = _ym(item.get("date"))

                if d:

                    results.append((d, f"{server}"))

            if results:

                break

    if not results:

        for server in ("biorxiv", "medrxiv"):

            data = await _get_json(
                session,
                f"https://api.biorxiv.org/pubs/{server}/{doi}",
                sem,
            )

            if data and data.get("collection"):

                for item in data["collection"][:5]:

                    preprint_doi = item.get("preprint_doi")

                    preprint_date = _ym(item.get("preprint_date"))

                    if preprint_date:

                        results.append((preprint_date, f"{server} (preprint)"))

                if results:

                    break

    return results


async def find_earliest_for_doi(session, doi, sem, title=None):
    """Query all sources for one DOI, return (earliest_date, source)."""

    all_dates = []

    arxiv_id = None

    cr_t = asyncio.create_task(fetch_crossref(session, doi, sem))

    ss_t = asyncio.create_task(fetch_semantic_scholar(session, doi, sem))

    oa_t = asyncio.create_task(fetch_openalex(session, doi, sem))

    ep_t = asyncio.create_task(fetch_europepmc(session, doi, sem))

    br_t = asyncio.create_task(fetch_biorxiv(session, doi, sem))

    ax_title_t = asyncio.create_task(fetch_arxiv_by_title(session, title, sem))

    cr_res = await cr_t

    all_dates.extend(cr_res)

    ss_res = await ss_t

    if isinstance(ss_res, tuple):

        ss_dates, arxiv_id = ss_res

        all_dates.extend(ss_dates)

    else:

        all_dates.extend(ss_res)

    all_dates.extend(await oa_t)

    all_dates.extend(await ep_t)

    all_dates.extend(await br_t)

    all_dates.extend(await ax_title_t)

    if arxiv_id:

        ax_res = await fetch_arxiv_by_id(session, arxiv_id, sem)

        all_dates.extend(ax_res)

    if not all_dates:

        return None, None

    all_dates = [(d, s) for d, s in all_dates if d]

    if not all_dates:

        return None, None

    earliest = min(all_dates, key=lambda x: x[0])

    return earliest


async def process_csv(csv_path: str):
    """Main async pipeline."""

    df = pd.read_csv(csv_path)

    if "doi" not in df.columns:

        print("ERROR: CSV must contain a 'doi' column.", file=sys.stderr)

        sys.exit(1)

    cache = load_cache(csv_path)

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    doi_to_title = {}

    if "title" in df.columns:

        for _, row in df[["doi", "title"]].dropna(subset=["doi"]).iterrows():

            doi_to_title[row["doi"]] = row.get("title")

    dois = df["doi"].dropna().unique().tolist()

    to_fetch = [d for d in dois if d not in cache]

    cached_count = len(dois) - len(to_fetch)

    print(f"📋 {len(dois)} unique DOIs | {cached_count} cached | {len(to_fetch)} to fetch")

    if to_fetch:

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=5)

        async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:

            processed = 0

            pbar = atqdm(total=len(to_fetch), desc="Fetching dates", unit="doi")

            for i in range(0, len(to_fetch), CACHE_SAVE_INTERVAL):

                batch = to_fetch[i : i + CACHE_SAVE_INTERVAL]

                tasks = []

                for doi in batch:

                    title = doi_to_title.get(doi)

                    tasks.append(find_earliest_for_doi(session, doi, sem, title=title))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for doi, result in zip(batch, results):

                    if isinstance(result, Exception):

                        cache[doi] = {"date": None, "source": None, "error": str(result)}

                    else:

                        date, source = result

                        cache[doi] = {"date": date, "source": source}

                    processed += 1

                    pbar.update(1)

                save_cache(csv_path, cache)

            pbar.close()

    df["first_date_found"] = df["doi"].map(
        lambda d: cache.get(d, {}).get("date") if pd.notna(d) else None
    )

    df["first_date_source"] = df["doi"].map(
        lambda d: cache.get(d, {}).get("source") if pd.notna(d) else None
    )

    stem = Path(csv_path).stem

    out_path = Path(csv_path).parent / f"{stem}_with_dates.csv"

    df.to_csv(out_path, index=False)

    print(f"\n✅ Output written to {out_path}")

    found = df["first_date_found"].notna().sum()

    total = len(df)

    print(f"📊 Dates found: {found}/{total} ({found/total*100:.1f}%)")

    src_counts = df["first_date_source"].value_counts()

    if not src_counts.empty:

        print("\n📈 Source breakdown:")

        for src, count in src_counts.items():

            print(f"   {src}: {count}")


def main():

    parser = argparse.ArgumentParser(
        description="Find earliest publication/preprint dates for papers in a CSV."
    )

    parser.add_argument("csv_path", help="Path to input CSV file (must have 'doi' column)")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):

        print(f"ERROR: File not found: {args.csv_path}", file=sys.stderr)

        sys.exit(1)

    asyncio.run(process_csv(args.csv_path))


if __name__ == "__main__":

    main()
