"""File writes that tolerate OneDrive / antivirus locks: write to a temp name, then retry the rename."""
import os
import time
from pathlib import Path

import pandas as pd


def _retry(fn, tries=20, wait=0.5):
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait * (1 + k))


def write_parquet(df: pd.DataFrame, path: Path):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    _retry(lambda: df.to_parquet(tmp, index=False))
    _retry(lambda: os.replace(tmp, path))


def append_parquet_part(df: pd.DataFrame, directory: Path, tag: str):
    """Batched append: each batch is one part file; read the directory back as one table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_parquet(df, directory / f"part-{tag}.parquet")


def read_parquet_dir(directory: Path) -> pd.DataFrame:
    files = sorted(Path(directory).glob("part-*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([_retry(lambda f=f: pd.read_parquet(f)) for f in files], ignore_index=True)
