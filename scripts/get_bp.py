"""Download the BP crude assay page and every per-grade assay file into data/raw/bp/.

The assay files are not redistributed in this repository (BP: "for informational purposes only");
run this script to fetch them. Files that already exist are left untouched. Some BP files are real
.xls and some are .xlsx saved with an .xls name; scripts/clean.py handles both.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch import RAW, fetch  # noqa: E402

# The old ".../bp-global-energy-trading/.../crude-assays.html" URL now redirects to the bp homepage.
PAGE_URL = ("https://www.bp.com/en/global/bp-supply-trading-and-shipping/documents-and-downloads/"
            "technical-downloads/crudes-assays.html")
page_path = RAW / "bp" / "bp_crudes_assays.html"
if not page_path.exists():
    fetch(PAGE_URL, "bp", page_path.name)
page = page_path.read_text(encoding="utf-8")

links = sorted(set(re.findall(r"(/content/dam/[^\"\s]+?\.xls)", page, re.I)))
print(f"found {len(links)} assay links")
for link in links:
    name = link.rsplit("/", 1)[-1]
    if not (RAW / "bp" / "assays" / name).exists():
        fetch("https://www.bp.com" + link, "bp/assays", name)
