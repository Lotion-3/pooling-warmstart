"""Download the FAOSTAT Food Balance Sheets (2010-) bulk file, normalized layout.

    python -P bench/scripts/get_food_balance.py

The URL is the FileLocation listed for dataset code FBS in the FAOSTAT bulk index
(https://bulks-faostat.fao.org/production/datasets_E.json). No account is needed.
Files go to bench/data/raw/food_balance/ (gitignored); files already present are skipped.
Each download is appended to download_log.txt (URL, UTC date, size).
"""
import datetime as dt
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.domain import RAW  # noqa: E402

OUT = RAW / "food_balance"
URLS = [
    "https://bulks-faostat.fao.org/production/FoodBalanceSheets_E_All_Data_(Normalized).zip",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-benchmark-data-download"


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            size += len(chunk)
    tmp.replace(dest)
    return size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        dest = OUT / urllib.parse.unquote(url.rsplit("/", 1)[1])
        if dest.exists():
            print(f"skip {dest.name} (present, {dest.stat().st_size} bytes)")
            continue
        size = fetch(url, dest)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(OUT / "download_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{url}\t{size} bytes\n")
        print(f"got {dest.name}: {size} bytes")


if __name__ == "__main__":
    main()
