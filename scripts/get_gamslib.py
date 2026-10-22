"""Download the GAMS Model Library `pool` model (SEQ=254) into data/raw/gamslib/.

Used only to validate the pooling formulation (Phase 1): it holds the literature data for the
Haverly, Ben-Tal, Foulds, Adhya and RT2 instances. Not redistributed here (GAMS Development Corp.
copyright); files that already exist are left untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch import RAW, fetch  # noqa: E402

BASE = "https://www.gams.com/latest/gamslib_ml/"
for name in ["pool.254", "poolmod.inc", "libhtml/gamslib_pool.html"]:
    target = RAW / "gamslib" / name.rsplit("/", 1)[-1]
    if not target.exists():
        fetch(BASE + name, "gamslib", target.name)
