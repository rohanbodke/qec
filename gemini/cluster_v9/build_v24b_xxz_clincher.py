#!/usr/bin/env python3
"""
build_v24b_xxz_clincher.py
Builds and verifies the V24b surgical follow-up burst on SubMIT using `kmc_engine_v19`
(`prep_logicals`, `verify`, and 2-trial `self_test` gate) and emits a 32-shard
8-thread/shard SLURM manifest `manifest_v24b.txt` + `v24b_code_specs.json`.

Targets (4 new pure-L=3 3-step `XXZ` codes + 2 beta=7.0 extensions from `data_v24`):
  1. `v24b_3s_X2_X2_Z2`      (tor1, N=110,400, K=4, dim_F=2.892, w_max=(16,18))
  2. `v24b_3s_X2_X2_Zjog1`   (tor1, N=92,640,  K=4, dim_F=2.839, w_max=(14,18))
  3. `v24b_plan_3s_X2_X2_Z1` (planar css_dual, N=89,472,  K=2, dim_F=2.828)
  4. `v24b_plan_3s_X2_X2_Z2` (planar css_dual, N=110,400, K=2, dim_F=2.892)
  5. `v24b_ref_2s_X5r4_Z3r1` (`data_v24/v24_2s_X5r4_Z3r1`, N=11,840, K=4, beta=7.0)
  6. `v24b_ref_3s_X2_X2_Z1` (`data_v24/v24_3s_X2_X2_Z1`, N=89,472, K=4, beta=7.0)
"""

import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v24b")
DATA_V24_DIR = os.path.join(WORK_DIR, "data_v24")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_v23_extensionA import build_complex_from_steps, build_flag_color_css_bdry


def build_code_job(job):
    tag, seed_kind, steps, k_exp = job
    comp, L_blk = build_complex_from_steps(seed_kind, steps)
    bdry_mode = "css_dual" if seed_kind == "planar" else "color"
    N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="welded", bdry_mode=bdry_mode)

    for sec, checks, stabs in [("X_err", HZ, HX), ("Z_err", HX, HZ)]:
        raw_p = os.path.join(DATA_DIR, f"{tag}_{sec}.raw")
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(raw_p, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, cpath], check=True, stdout=subprocess.DEVNULL)
        os.remove(raw_p)
        vout = subprocess.check_output([ENGINE, "verify", cpath], text=True).strip()
        assert f"K={k_exp}" in vout and "bad_comm=0" in vout, f"{tag}_{sec}: {vout}"
        subprocess.run(
            [ENGINE, "self_test", cpath, "4.5", "2", "1", "1.0", "2.0", "2", "50000", "2"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    F = len(comp.faces)
    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    if seed_kind == "tor1":
        assert min(dx) == 3 and min(dz) == 3, f"{tag}: min_deg=({min(dx)},{min(dz)}) != 3"
    d4_edges = sum(1 for inc in comp.edge_faces if len(inc) >= 4)

    return tag, {
        "tag": tag,
        "seed_kind": seed_kind,
        "N": N,
        "F": F,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "K": k_exp,
        "L_blk": L_blk,
        "dim_F": (math.log(F) / math.log(L_blk)) if L_blk > 1 else 0.0,
        "d4_edge_count": d4_edges,
        "max_edge_deg": max(len(inc) for inc in comp.edge_faces),
        "max_wX": max(len(r) for r in HX),
        "max_wZ": max(len(r) for r in HZ),
        "mean_wX": sum(len(r) for r in HX) / len(HX),
        "mean_wZ": sum(len(r) for r in HZ) / len(HZ),
        "min_dX": min(dx),
        "max_dX": max(dx),
        "min_dZ": min(dz),
        "max_dZ": max(dz),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "results_v24b"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "logs_v24b"), exist_ok=True)

    jobs = [
        ("v24b_3s_X2_X2_Z2",      "tor1",   [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, [1, 2], "Z")], 4),
        ("v24b_3s_X2_X2_Zjog1",   "tor1",   [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, "jog1", "Z")], 4),
        ("v24b_plan_3s_X2_X2_Z1", "planar", [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, [1], "Z")],    2),
        ("v24b_plan_3s_X2_X2_Z2", "planar", [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, [1, 2], "Z")], 2),
    ]

    specs = {}
    for job in jobs:
        tag, info = build_code_job(job)
        specs[tag] = info
        print(
            f"  [BUILD] {tag:24s}: N={info['N']:7,d} K={info['K']} L={info['L_blk']:2d} "
            f"dimF={info['dim_F']:.3f} d_min=({info['min_dX']},{info['min_dZ']}) "
            f"w_max=({info['max_wX']},{info['max_wZ']})",
            flush=True,
        )

    # Metadata entries for existing data_v24 codes extended to beta=7.0
    specs["v24b_ref_2s_X5r4_Z3r1"] = {
        "tag": "v24b_ref_2s_X5r4_Z3r1",
        "seed_kind": "tor1",
        "N": 11840,
        "K": 4,
        "L_blk": 15,
        "dim_F": 2.695679169,
    }
    specs["v24b_ref_3s_X2_X2_Z1"] = {
        "tag": "v24b_ref_3s_X2_X2_Z1",
        "seed_kind": "tor1",
        "N": 89472,
        "K": 4,
        "L_blk": 27,
        "dim_F": 2.828070694,
    }

    with open(os.path.join(WORK_DIR, "v24b_code_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    tasks = []

    def add_th(tag, code_dir, code_stem, beta, n_tot, per_task, lg0, lg1, ng, max_fl, w_mult=1.0):
        N = specs[tag]["N"]
        base_w = (max(1.0, N / 1000.0) ** 1.05) * (max_fl / 4e7) * (per_task / 16.0) * w_mult
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(code_dir, f"{code_stem}_{sec}.code")
            assert os.path.exists(cpath), f"Missing {cpath}"
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"{tag}_{sec}__b{beta:.2f}__o{off}"
                tasks.append(
                    (base_w, f"{tid}|paired_readout|{cpath}|{beta:.5f}|{nt}|{off}|{lg0:.4f}|{lg1:.4f}|{ng}|{int(max_fl)}")
                )

    def add_iid(tag, code_dir, code_stem, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23):
        N = specs[tag]["N"]
        base_w = max(5.0, N / 400.0)
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(code_dir, f"{code_stem}_{sec}.code")
            assert os.path.exists(cpath), f"Missing {cpath}"
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"v24b_iid__{tag}_{sec}__o{off}"
                tasks.append((base_w, f"{tid}|iid_readout|{cpath}|{nt}|{off}|{p0:.5f}|{p1:.5f}|{ng}"))

    # 1. v24b_3s_X2_X2_Z2 (beta = 5.0..7.0 + iid)
    for b, ntot, pt, lg0, lg1, ng, mfl in [
        (5.0, 64, 16, 2.0,  8.0, 25,  80_000_000),
        (5.5, 64, 16, 3.0,  9.0, 25, 100_000_000),
        (6.0, 64, 16, 4.5, 10.0, 23, 120_000_000),
        (6.5, 64, 16, 5.5, 11.0, 23, 150_000_000),
        (7.0, 32, 16, 6.5, 12.0, 23, 150_000_000),
    ]:
        add_th("v24b_3s_X2_X2_Z2", DATA_DIR, "v24b_3s_X2_X2_Z2", b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.5)
    add_iid("v24b_3s_X2_X2_Z2", DATA_DIR, "v24b_3s_X2_X2_Z2")

    # 2. v24b_3s_X2_X2_Zjog1 (beta = 5.5, 6.0, 6.5)
    for b, ntot, pt, lg0, lg1, ng, mfl in [
        (5.5, 64, 16, 3.0,  8.5, 23,  80_000_000),
        (6.0, 64, 16, 4.0,  9.5, 23, 100_000_000),
        (6.5, 64, 16, 5.0, 10.5, 23, 120_000_000),
    ]:
        add_th("v24b_3s_X2_X2_Zjog1", DATA_DIR, "v24b_3s_X2_X2_Zjog1", b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.3)

    # 3. v24b_plan_3s_X2_X2_Z1 (beta = 5.5, 6.0, 6.5)
    for b, ntot, pt, lg0, lg1, ng, mfl in [
        (5.5, 64, 16, 3.0,  8.5, 23,  80_000_000),
        (6.0, 64, 16, 4.0,  9.5, 23, 100_000_000),
        (6.5, 64, 16, 5.0, 10.5, 23, 120_000_000),
    ]:
        add_th("v24b_plan_3s_X2_X2_Z1", DATA_DIR, "v24b_plan_3s_X2_X2_Z1", b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.2)

    # 4. v24b_plan_3s_X2_X2_Z2 (beta = 6.0, 6.5)
    for b, ntot, pt, lg0, lg1, ng, mfl in [
        (6.0, 64, 16, 4.0, 10.0, 25, 120_000_000),
        (6.5, 64, 16, 5.0, 11.0, 25, 150_000_000),
    ]:
        add_th("v24b_plan_3s_X2_X2_Z2", DATA_DIR, "v24b_plan_3s_X2_X2_Z2", b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.5)

    # 5. beta = 7.0 extensions on existing data_v24 codes
    add_th("v24b_ref_2s_X5r4_Z3r1", DATA_V24_DIR, "v24_2s_X5r4_Z3r1", 7.0, 64, 16, 6.0, 11.5, 23, 150_000_000, w_mult=1.5)
    add_th("v24b_ref_3s_X2_X2_Z1", DATA_V24_DIR, "v24_3s_X2_X2_Z1", 7.0, 32, 16, 6.0, 11.5, 23, 150_000_000, w_mult=2.5)

    n_shards = 32
    bins = [(0.0, i) for i in range(n_shards)]
    heapq.heapify(bins)
    assigned = [[] for _ in range(n_shards)]
    for w, line in sorted(tasks, key=lambda x: -x[0]):
        load, sh = heapq.heappop(bins)
        assigned[sh].append(line)
        heapq.heappush(bins, (load + w, sh))

    mpath = os.path.join(WORK_DIR, "manifest_v24b.txt")
    with open(mpath, "w") as out:
        for sh in range(n_shards):
            for line in assigned[sh]:
                out.write(f"{sh}|{line}\n")
    print(f"Wrote {len(tasks)} tasks across {n_shards} shards to {mpath}")


if __name__ == "__main__":
    main()
