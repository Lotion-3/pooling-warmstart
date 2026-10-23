"""Download FDIC call report balance sheets (BankFind Suite financials API), 2015Q1 - 2024Q4.

    python -P bench/scripts/get_bank_balance_sheets.py

One JSON file per quarter goes to bench/data/raw/bank_balance_sheets/ (gitignored); quarters already
present are skipped.  Each download is appended to download_log.txt (URL, UTC date, size).
Requests are sequential with a short pause; no API key is needed.
"""
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.domain import RAW  # noqa: E402

OUT = RAW / "bank_balance_sheets"
ENDPOINT = "https://api.fdic.gov/banks/financials"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-benchmark-data-download"
PAGE = 10_000          # API maximum per request
PAUSE = 1.5            # seconds between requests

# Balance-sheet items (RISVIEW / call report schedule RC, $ thousands)
FIELDS = [
    "CERT", "REPDTE",
    # assets
    "ASSET", "CHBAL", "FREPO", "SC", "TRADE", "LNLSNET", "BKPREM", "ORE", "INTAN", "AOA",
    # liabilities
    "LIAB", "DEP", "FREPP", "TRADEL", "OTHBRF", "SUBND", "ALLOTHL",
    # equity and the balance-sheet total
    "EQTOT", "EQ", "EQCONSUB", "LIABEQ",
]


def quarters(first_year=2015, last_year=2024):
    for y in range(first_year, last_year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            yield f"{y}{md}"


def url_for(repdte, offset):
    q = {"filters": f"REPDTE:{repdte}", "fields": ",".join(FIELDS), "limit": PAGE, "offset": offset,
         "sort_by": "CERT", "sort_order": "ASC", "format": "json"}
    return ENDPOINT + "?" + urllib.parse.urlencode(q, safe=":,")


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
            return json.loads(raw), len(raw)
        except Exception as e:  # network hiccup: back off and retry
            if attempt == 3:
                raise
            print(f"  retry after error: {e}")
            time.sleep(10 * (attempt + 1))


def fetch_quarter(repdte):
    rows, urls, size, offset = [], [], 0, 0
    while True:
        url = url_for(repdte, offset)
        js, n = get_json(url)
        urls.append(url)
        size += n
        page = [r["data"] for r in js["data"]]
        rows += page
        total = js["meta"]["total"]
        offset += len(page)
        if not page or offset >= total:
            break
        time.sleep(PAUSE)
    assert len(rows) == total, f"{repdte}: got {len(rows)} of {total} rows"
    return rows, urls, size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for repdte in quarters():
        dest = OUT / f"financials_{repdte}.json"
        if dest.exists():
            print(f"skip {dest.name} (present, {dest.stat().st_size} bytes)")
            continue
        rows, urls, size = fetch_quarter(repdte)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = json.dumps({"repdte": repdte, "downloaded": stamp, "urls": urls, "fields": FIELDS, "rows": rows})
        tmp = dest.with_suffix(".part")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(dest)
        with open(OUT / "download_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{' '.join(urls)}\t{size} bytes received, {len(rows)} rows, "
                    f"saved {len(body)} bytes\n")
        print(f"got {dest.name}: {len(rows)} banks, {size} bytes")
        time.sleep(PAUSE)


if __name__ == "__main__":
    main()
