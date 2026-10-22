"""Summarise a smoke-test round: status, times, IPOPT hit / trap / infeasible rates."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_parquet_dir  # noqa: E402

d = read_parquet_dir(Path(sys.argv[1]))
print("n", len(d), "| SCIP status", Counter(d.glob_status), "| certified", int(d.certified.sum()))
print("SCIP time s: mean %.2f  p95 %.2f  max %.2f" % (d.glob_time.mean(), d.glob_time.quantile(.95), d.glob_time.max()))
c = d[d.certified]
ipt = np.concatenate(c.ip_time.values)
print("IPOPT time/start s: mean %.3f  p95 %.3f" % (ipt.mean(), np.quantile(ipt, .95)))
print("over certified: hit %.3f | TRAP %.3f | infeasible-end %.3f | beats-global %d"
      % (c.ip_hit_rate.mean(), c.ip_trap_rate.mean(), 1 - c.ip_feasible_rate.mean(), int(c.ip_beats_global.sum())))
print("instances with trap>0: %d / %d" % ((c.ip_trap_rate > 0).sum(), len(c)))
print("IPOPT terms", Counter(np.concatenate(c.ip_term.values)).most_common(5))
print("binding specs per inst (mean)", np.mean([np.sum(b) for b in c.bind]),
      "| active direct arcs", np.mean([np.sum(z) for z in c.act_z]), "| active tank->CDU", np.mean([np.sum(y) for y in c.act_y]))
print("glob obj: ", c.glob_obj.describe()[["mean", "min", "max"]].round(1).to_dict())
