"""Download NREL ResStock 2025 release 1 (AMY2018) baseline metadata + annual results for a few states.

    python -P bench/scripts/get_building_end_use.py

Source: OEDI data lake (bucket oedi-data-lake, anonymous HTTPS), dataset "End-Use Load Profiles for
the U.S. Building Stock".  One baseline (upgrade 0) parquet file per state, 3-55 MB each, 363 MB for
the 20 states.
Files go to bench/data/raw/building_end_use/ (gitignored); files already present are skipped.
Each download is appended to download_log.txt (URL, UTC date, size).
"""
import datetime as dt
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.domain import RAW  # noqa: E402

OUT = RAW / "building_end_use"
RELEASE = "2025/resstock_amy2018_release_1"
BASE = ("https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
        f"end-use-load-profiles-for-us-building-stock/{RELEASE}")
# twenty states spanning hot/mixed/cold climates and all four metered fuels (fuel oil is material only
# in the Northeast: CT, MA, ME, NH, NJ, NY, PA)
STATES = ["AZ", "CA", "CO", "CT", "FL", "GA", "IL", "MA", "ME", "MI", "MN", "NC", "NH", "NJ", "NY", "PA",
          "TX", "VA", "WA", "WI"]
URLS = [f"{BASE}/data_dictionary.tsv"] + [
    f"{BASE}/metadata_and_annual_results/by_state/full/parquet/state={s}/{s}_upgrade0.parquet" for s in STATES]
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
