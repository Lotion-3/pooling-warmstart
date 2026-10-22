"""Merge Phase 4 batch files into single parquet files: data/solved/solved.parquet, solved_fixed.parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_parquet_dir, write_parquet  # noqa: E402

for parts, name in [("parts", "solved.parquet"), ("parts_fixed", "solved_fixed.parquet")]:
    d = read_parquet_dir(ROOT / "data" / "solved" / parts)
    if len(d):
        d = d.drop_duplicates("inst_id", keep="last")
        write_parquet(d, ROOT / "data" / "solved" / name)
        print(name, len(d), "rows")
