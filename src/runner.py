"""Solve instances: SCIP global + IPOPT multistart + optimal structure. Parallel, batched, resumable.

  python -P src/runner.py --out data/solved/parts --workers 19 [--ids ids.txt] [--limit N]
                          [--time-limit 60] [--starts 20] [--instances data/instances/instances.parquet]
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]

HIT_TOL, FEAS_TOL, ACTIVE = 1e-4, 1e-5, 1e-6


def solve_record(row: dict, time_limit=60.0, n_starts=20, seed=0) -> dict:
    logging.getLogger("pyomo").setLevel(logging.ERROR)
    from pooling import binding_specs, build_pyomo, evaluate, random_start, rel_gap, solve_ipopt, solve_scip
    from realdata import from_row
    inst = from_row(row)
    rec = {"inst_id": row["inst_id"]}
    g = solve_scip(inst, time_limit=time_limit)
    rec.update(glob_status=g["status"], glob_obj=float(g["obj"]), glob_dual=float(g["dual"]),
               glob_gap=float(g["gap"]), glob_time=g["time"], glob_nodes=int(g["nodes"]))
    has = np.isfinite(g["obj"])
    if has:
        q, y, z = g["q"], g["y"], g["z"]
        e = evaluate(inst, q, y, z)
        x = q * y.sum(1)[None, :]  # crude -> tank flows
        rec.update(glob_max_viol=e["max_viol"], glob_q=q.ravel().tolist(), glob_y=y.ravel().tolist(),
                   glob_z=z.ravel().tolist(),
                   act_x=(x > ACTIVE).ravel().tolist(), act_y=(y > ACTIVE).ravel().tolist(),
                   act_z=(z > ACTIVE).ravel().tolist(),
                   bind=binding_specs(inst, q, y, z).tolist())
    certified = g["status"] == "optimal" or (has and g["gap"] <= HIT_TOL)
    rec["certified"] = bool(certified)
    ref = g["obj"] if has else np.nan
    m = build_pyomo(inst, "pq", rlt=False)
    rng = np.random.default_rng([seed, zlib.crc32(row["inst_id"].encode())])
    objs, terms, times, viols, hits = [], [], [], [], []
    for _ in range(n_starts):
        r = solve_ipopt(inst, m, random_start(inst, rng))
        objs.append(r["obj"]); terms.append(r["term"]); times.append(r["time"]); viols.append(r["max_viol"])
        hits.append(bool(has and r["max_viol"] <= FEAS_TOL and rel_gap(r["obj"], ref) <= HIT_TOL))
    rec.update(ip_obj=objs, ip_term=terms, ip_time=times, ip_viol=viols, ip_hit=hits)
    feas = np.array(viols) <= FEAS_TOL
    rec["ip_feasible_rate"] = float(feas.mean())
    rec["ip_hit_rate"] = float(np.mean(hits))
    # trap: IPOPT converged to a FEASIBLE point that is worse than the global optimum
    rec["ip_trap_rate"] = float(np.mean([f and not h for f, h in zip(feas, hits)])) if has else np.nan
    if has:
        better = [o for o, v in zip(objs, viols) if v <= FEAS_TOL and rel_gap(o, ref) < -HIT_TOL]
        rec["ip_beats_global"] = len(better)  # should be 0 for certified instances (sanity check)
    return rec


def _work(args):
    row, tl, ns = args
    try:
        return solve_record(row, tl, ns)
    except Exception as e:  # keep the run going; record the failure
        return {"inst_id": row["inst_id"], "error": f"{type(e).__name__}: {e}"}


def main():
    from io_utils import append_parquet_part, read_parquet_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=str(ROOT / "data" / "instances" / "instances.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--log", default=None)
    ap.add_argument("--deadline", default=None, help="HH:MM local; stop submitting new work after this")
    a = ap.parse_args()
    df = pd.read_parquet(a.instances)
    if a.ids:
        ids = [s.strip() for s in Path(a.ids).read_text().split() if s.strip()]
        df = df.set_index("inst_id").loc[ids].reset_index()
    done = read_parquet_dir(Path(a.out))
    done_ids = set(done["inst_id"]) if len(done) else set()
    todo = df[~df.inst_id.isin(done_ids)]
    if a.limit:
        todo = todo.head(a.limit)
    cols = [c for c in df.columns if c in ("inst_id", "name") or c.endswith("_shape") or c in
            ("c", "d", "src_lo", "src_up", "pool_cap", "prod_lo", "prod_up", "lam", "q_up", "y_up", "z_up",
             "spec_prod", "spec_alpha", "spec_beta", "h")]
    rows = todo[cols].to_dict("records")
    log = open(a.log, "a", encoding="utf-8") if a.log else None

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if log:
            log.write(line + "\n"); log.flush()

    deadline = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
        if deadline < time.time():
            deadline += 86400
    say(f"start: {len(rows)} to do ({len(done_ids)} already done), workers={a.workers}, tl={a.time_limit}s, starts={a.starts}")
    t0, last_log, buf, n = time.time(), time.time(), [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=200) as pool:
        it = pool.imap_unordered(_work, ((r, a.time_limit, a.starts) for r in rows), chunksize=1)
        for rec in it:
            buf.append(rec); n += 1
            if len(buf) >= a.batch:
                append_parquet_part(pd.DataFrame(buf), Path(a.out), f"{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}-{n}")
                buf = []
            if time.time() - last_log > 300 or n == len(rows):
                rate = n / (time.time() - t0)
                say(f"{n}/{len(rows)} done, {rate*60:.1f}/min, ETA {(len(rows)-n)/max(rate,1e-9)/60:.0f} min")
                last_log = time.time()
            if deadline and time.time() > deadline:
                say("deadline reached: stopping"); pool.terminate(); break
    if buf:
        append_parquet_part(pd.DataFrame(buf), Path(a.out), f"{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}-{n}")
    say(f"finished {n} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
