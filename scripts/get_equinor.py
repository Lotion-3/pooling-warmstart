"""Download the Equinor crude assay page and every per-grade assay XLSX into data/raw/equinor/.

The assay files are not redistributed in this repository (Equinor: "for guidance only"); run this
script to fetch them. Files that already exist are left untouched. Equinor updates assays over time,
so a fresh download can differ from the versions used for the published results (see data_report.md).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch import RAW, fetch  # noqa: E402

PAGE_URL = "https://www.equinor.com/energy/crude-oil-assays"
page_path = RAW / "equinor" / "equinor_crude_oil_assays.html"
if not page_path.exists():
    fetch(PAGE_URL, "equinor", page_path.name)
page = page_path.read_text(encoding="utf-8")
# The Next.js flight payload is JSON with escaped quotes; unescape once for matching.
flat = page.replace('\\"', '"')

pat = re.compile(
    r'"text":"(?P<text>[^"]+)"\}\],"markDefs":\[\{"_key":"[0-9a-f]+","_type":"attachment",'
    r'"attachment":\{"extension":"xlsx","id":"[0-9a-f]+","originalFilename":"(?P<fn>[^"]+)",'
    r'"type":"attachment","url":"(?P<url>[^"]+)"'
)
links = {}
for m in pat.finditer(flat):
    links.setdefault(m["url"], {"grade": m["text"], "original_filename": m["fn"], "url": m["url"]})
links = list(links.values())
print(f"found {len(links)} assay links")
links_path = RAW / "equinor" / "assay_links.json"
if not links_path.exists():
    links_path.write_text(json.dumps(links, indent=1, ensure_ascii=False), encoding="utf-8")

for L in links:
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", L["original_filename"])
    if not (RAW / "equinor" / "assays" / safe).exists():
        fetch(L["url"], "equinor/assays", safe)
