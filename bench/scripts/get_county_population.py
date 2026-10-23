"""Download U.S. Census Bureau county population estimates (totals + components of change).

    python -P bench/scripts/get_county_population.py

Files go to bench/data/raw/county_population/ (gitignored); files already present are skipped.
Each download is appended to download_log.txt (URL, UTC date, size).
"""
import datetime as dt
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.domain import RAW  # noqa: E402

OUT = RAW / "county_population"
BASE = "https://www2.census.gov/programs-surveys/popest/datasets"
URLS = [
    # Vintage 2020: April 2010 base .. July 2020 (the last vintage on the 2010 Census base)
    f"{BASE}/2010-2020/counties/totals/co-est2020-alldata.csv",
    # Vintage 2025: April 2020 base .. July 2025 (latest vintage on the 2020 Census base)
    f"{BASE}/2020-2025/counties/totals/co-est2025-alldata.csv",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-benchmark-data-download"


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        dest = OUT / url.rsplit("/", 1)[1]
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
