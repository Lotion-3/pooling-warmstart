"""Download helper for the data access audit.

Every request is appended to data/raw/access_log.jsonl. Raw files are saved
exactly as downloaded and never overwritten: if the target name exists, a
UTC timestamp suffix is added.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LOG = RAW / "access_log.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.stem}__{stamp}{path.suffix}")


def fetch(url: str, subdir: str, name: str | None = None, save: bool = True, timeout: int = 60):
    """GET url; save body under data/raw/<subdir>/<name>. Returns (response|None, path|None, error|None)."""
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "url": url}
    resp, path, err = None, None, None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        entry.update(
            status=resp.status_code,
            final_url=resp.url,
            content_type=resp.headers.get("Content-Type"),
            bytes=len(resp.content),
        )
        if save and resp.ok:
            name = name or Path(requests.utils.urlparse(resp.url).path).name or "index.html"
            d = RAW / subdir
            d.mkdir(parents=True, exist_ok=True)
            path = _unique_path(d / name)
            path.write_bytes(resp.content)
            entry["saved_as"] = str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception as e:  # record the exact error
        err = f"{type(e).__name__}: {e}"
        entry["error"] = err
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))
    return resp, path, err


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
