#!/usr/bin/env python3
"""
build_v24_degree3_balanced.py
Builds and verifies the V24 strictly 3D-local degree-3 X/Z-balanced welded 4.8.8
flag color-code campaign using `kmc_engine_v19` (`prep_logicals`, `verify`, and
2-trial `self_test` gate) and emits an 80-shard SLURM manifest `manifest_v24.txt`
+ `v24_code_specs.json`.

Two degree-3 balancing mechanisms (12 codes total):
  1. High-rib step-1 bottleneck reinforcement:
     - 1-step controls: `v24_1s_L4r3` (448), `v24_1s_L5r4` (720)
     - 2-step torus: `v24_2s_X4r3_Z3r1` (7,488), `v24_2s_X4r3_Z3jog1` (7,744),
       `v24_2s_X4r3_Z3r2` (9,216), `v24_2s_X5r4_Z3r1` (11,840), `v24_2s_X5r4_Z3r2` (14,560)
     - 2-step open-boundary planar `css_dual` (K=2):
       `v24_plan_2s_X4r3_Z3r1` (7,488), `v24_plan_2s_X5r4_Z3r1` (11,840)
  2. Non-alternating pure-L=3 3-step `XXZ` welding order:
     - `v24_3s_X1_X2_Z1` (81,888), `v24_3s_X2_X1_Z1` (82,752), `v24_3s_X2_X2_Z1` (89,472)
"""

import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v24")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_v18_rework import gf2_rank_bigint
from build_v23_extensionA import build_complex_from_steps, build_flag_color_css_bdry


def build_code_job(job):
    tag, seed_kind, steps, k_exp = job
    comp, L_blk = build_complex_from_steps(seed_kind, steps)
    bdry_mode = "css_dual" if seed_kind == "planar" else "color"
    N, HX, HZ, _ = build_flag_color_css_bdry(comp, seam_mode="welded", bdry_mode=bdry_mode)

    if N <= 8000:
        rX = gf2_rank_bigint(HX)
        rZ = gf2_rank_bigint(HZ)
        k_tot = N - rX - rZ
        assert k_tot == k_exp, f"{tag}: got K={k_tot} != {k_exp}"
    else:
        rX, rZ = -1, -1

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
        "rank_HX": rX,
        "rank_HZ": rZ,
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
        "gamma_X_b50": sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dz if d > 0),
        "gamma_Z_b50": sum(1.0 / (1.0 + math.exp(5.0 * d)) for d in dx if d > 0),
        "gamma_X_b65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz if d > 0),
        "gamma_Z_b65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx if d > 0),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "results_v24"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "logs_v24"), exist_ok=True)

    jobs = [
        ("v24_1s_L4r3",           "tor1",   [(4, [1, 2, 3], "X")],                                   4),
        ("v24_1s_L5r4",           "tor1",   [(5, [1, 2, 3, 4], "X")],                                4),
        ("v24_2s_X4r3_Z3r1",      "tor1",   [(4, [1, 2, 3], "X"), (3, [1], "Z")],                    4),
        ("v24_2s_X4r3_Z3jog1",    "tor1",   [(4, [1, 2, 3], "X"), (3, "jog1", "Z")],                 4),
        ("v24_2s_X4r3_Z3r2",      "tor1",   [(4, [1, 2, 3], "X"), (3, [1, 2], "Z")],                 4),
        ("v24_2s_X5r4_Z3r1",      "tor1",   [(5, [1, 2, 3, 4], "X"), (3, [1], "Z")],                 4),
        ("v24_2s_X5r4_Z3r2",      "tor1",   [(5, [1, 2, 3, 4], "X"), (3, [1, 2], "Z")],              4),
        ("v24_3s_X1_X2_Z1",       "tor1",   [(3, [1], "X"), (3, [1, 2], "X"), (3, [1], "Z")],        4),
        ("v24_3s_X2_X1_Z1",       "tor1",   [(3, [1, 2], "X"), (3, [1], "X"), (3, [1], "Z")],        4),
        ("v24_3s_X2_X2_Z1",       "tor1",   [(3, [1, 2], "X"), (3, [1, 2], "X"), (3, [1], "Z")],     4),
        ("v24_plan_2s_X4r3_Z3r1", "planar", [(4, [1, 2, 3], "X"), (3, [1], "Z")],                    2),
        ("v24_plan_2s_X5r4_Z3r1", "planar", [(5, [1, 2, 3, 4], "X"), (3, [1], "Z")],                 2),
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

    with open(os.path.join(WORK_DIR, "v24_code_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    tasks = []

    def add_th(tag, beta, n_tot, per_task, lg0, lg1, ng, max_fl, w_mult=1.0):
        N = specs[tag]["N"]
        base_w = (max(1.0, N / 1000.0) ** 1.05) * (max_fl / 4e7) * (per_task / 16.0) * w_mult
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"{tag}_{sec}__b{beta:.2f}__o{off}"
                tasks.append(
                    (base_w, f"{tid}|paired_readout|{cpath}|{beta:.5f}|{nt}|{off}|{lg0:.4f}|{lg1:.4f}|{ng}|{int(max_fl)}")
                )

    def add_iid(tag, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23):
        N = specs[tag]["N"]
        base_w = max(5.0, N / 400.0)
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"v24_iid__{tag}_{sec}__o{off}"
                tasks.append((base_w, f"{tid}|iid_readout|{cpath}|{nt}|{off}|{p0:.5f}|{p1:.5f}|{ng}"))

    # 1. 1-step controls
    grid_1s = [
        (5.0, 96, 32, 2.0,  9.0, 29,  60_000_000),
        (5.5, 96, 32, 3.0, 10.0, 29,  80_000_000),
        (6.0, 96, 32, 3.5, 10.5, 29, 100_000_000),
        (6.5, 96, 32, 4.0, 11.0, 29, 120_000_000),
    ]
    for t in ("v24_1s_L4r3", "v24_1s_L5r4"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_1s:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=0.5)

    # 2. 2-step torus flagship curves
    grid_2s_flag = [
        (4.5, 64, 32, 2.0,  7.0, 21,  60_000_000),
        (5.0, 64, 32, 2.5,  8.0, 23,  60_000_000),
        (5.5, 64, 32, 3.5,  9.0, 23,  80_000_000),
        (6.0, 64, 16, 4.5,  9.5, 21, 100_000_000),
        (6.5, 96, 16, 5.5, 10.5, 21, 150_000_000),
    ]
    for t in ("v24_2s_X4r3_Z3r1", "v24_2s_X5r4_Z3r1"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s_flag:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=1.5)

    # 3. 2-step torus companion curves
    grid_2s_comp = [
        (5.0, 64, 32, 2.5,  8.0, 23,  60_000_000),
        (5.5, 64, 32, 3.5,  9.0, 23,  80_000_000),
        (6.0, 64, 16, 4.5,  9.5, 21, 100_000_000),
        (6.5, 96, 16, 5.5, 10.5, 21, 150_000_000),
    ]
    for t in ("v24_2s_X4r3_Z3jog1", "v24_2s_X4r3_Z3r2", "v24_2s_X5r4_Z3r2"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s_comp:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=1.8)

    # 4. 3-step pure-L=3 XXZ curves
    grid_3s_x1x2z1 = [
        (5.0, 64, 16, 2.0,  7.5, 23,  60_000_000),
        (5.5, 64, 16, 3.0,  8.5, 23,  80_000_000),
        (6.0, 64, 16, 4.0,  9.5, 23, 100_000_000),
        (6.5, 64, 16, 5.0, 10.5, 23, 120_000_000),
    ]
    for b, ntot, pt, lg0, lg1, ng, mfl in grid_3s_x1x2z1:
        add_th("v24_3s_X1_X2_Z1", b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.5)

    for t in ("v24_3s_X2_X1_Z1", "v24_3s_X2_X2_Z1"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_3s_x1x2z1[1:]:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.5)

    # 5. Open-boundary planar css_dual (K=2)
    grid_plan = [
        (5.0, 64, 32, 2.0,  7.5, 23,  60_000_000),
        (5.5, 64, 32, 3.0,  8.5, 23,  80_000_000),
        (6.0, 64, 16, 3.5,  9.5, 25, 100_000_000),
        (6.5, 96, 16, 4.5, 10.0, 23, 120_000_000),
    ]
    for t in ("v24_plan_2s_X4r3_Z3r1", "v24_plan_2s_X5r4_Z3r1"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_plan:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=1.5)

    # 6. Paired 4-decoder i.i.d. threshold scans
    for t in (
        "v24_2s_X4r3_Z3r1",
        "v24_2s_X5r4_Z3r1",
        "v24_2s_X5r4_Z3r2",
        "v24_3s_X1_X2_Z1",
        "v24_plan_2s_X4r3_Z3r1",
    ):
        add_iid(t, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23)

    n_shards = 80
    bins = [(0.0, i) for i in range(n_shards)]
    heapq.heapify(bins)
    assigned = [[] for _ in range(n_shards)]
    for w, line in sorted(tasks, key=lambda x: -x[0]):
        load, sh = heapq.heappop(bins)
        assigned[sh].append(line)
        heapq.heappush(bins, (load + w, sh))

    mpath = os.path.join(WORK_DIR, "manifest_v24.txt")
    with open(mpath, "w") as out:
        for sh in range(n_shards):
            for line in assigned[sh]:
                out.write(f"{sh}|{line}\n")
    print(f"Wrote {len(tasks)} tasks across {n_shards} shards to {mpath}")


if __name__ == "__main__":
    main()
