"""Download SEC EDGAR Financial Statement Data Sets (quarterly zips), Q1 + Q2 of 2012 - 2025.

    python -P bench/scripts/get_company_financials.py

Index: https://www.sec.gov/dera/data/financial-statement-data-sets
Each zip holds sub.txt / num.txt / tag.txt / pre.txt for all XBRL filings accepted in that quarter.
Only Q1 and Q2 are fetched: they contain the 10-K filings of firms with fiscal years ending
roughly September - March (December year-ends dominate), ~28 zips, ~2.7 GB in total.

Files go to bench/data/raw/company_financials/ (gitignored); zips already present are skipped.
Each download is appended to download_log.txt (URL, UTC date, size).
SEC fair-access policy: declared User-Agent, sequential requests with a pause (well below 10/s).
"""
import datetime as dt
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.domain import RAW  # noqa: E402

OUT = RAW / "company_financials"
BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
UA = "Laksh Agarwal lakshagarwal65@gmail.com"
FIRST_YEAR, LAST_YEAR = 2012, 2025
QUARTERS = (1, 2)
PAUSE = 2.0  # seconds between requests


def quarter_names():
    return [f"{y}q{q}" for y in range(FIRST_YEAR, LAST_YEAR + 1) for q in QUARTERS]


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    tmp.replace(dest)
    return size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in quarter_names():
        url = f"{BASE}/{name}.zip"
        dest = OUT / f"{name}.zip"
        if dest.exists():
            print(f"skip {dest.name} (present, {dest.stat().st_size} bytes)")
            total += dest.stat().st_size
            continue
        for attempt in range(4):
            try:
                size = fetch(url, dest)
                break
            except Exception as e:  # network hiccup: back off and retry
                if attempt == 3:
                    raise
                print(f"retry {name}: {e}")
                time.sleep(10 * (attempt + 1))
        total += size
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(OUT / "download_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{url}\t{size} bytes\n")
        print(f"got {dest.name}: {size} bytes", flush=True)
        time.sleep(PAUSE)
    print(f"total {total} bytes in {len(quarter_names())} zips")


if __name__ == "__main__":
    main()
