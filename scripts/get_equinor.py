"""Extract grade -> assay XLSX links from the saved Equinor page and download each file."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch import RAW, fetch  # noqa: E402

page = (RAW / "equinor" / "equinor_crude_oil_assays.html").read_text(encoding="utf-8")
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
(RAW / "equinor" / "assay_links.json").write_text(json.dumps(links, indent=1, ensure_ascii=False), encoding="utf-8")

for L in links:
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", L["original_filename"])
    fetch(L["url"], "equinor/assays", safe)
