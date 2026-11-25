#!/usr/bin/env python3
"""
build_v20_benchmarks.py
Builds and mathematically verifies all V20 cross-family finite-size benchmark `.code` files
using `kmc_engine_v19 prep_logicals` and `verify` (exact CSS commutation, full GF(2) ranks,
and anticommuting logical basis), records exact code parameters and per-qubit check degrees
`d_tot = (sum |S_X| + sum |S_Z|) / N` in `v20_code_specs.json`, and emits a load-balanced
64-shard SLURM manifest `manifest_v20.txt`.

Code families compared under identical BKL Glauber dynamics and paired 4-decoder readout
(`UF`, `Cool-UF`, `Hier-UF`, `BP-OSD`):
  1. 2D Toric Code (`v20_tor2d_L8, L16, L32, L64`): N = 128..8,192, K = 2, w_max = 4, d_tot = 4.00.
  2. Single-Scale 3D Welded Surface Codes (`v20_mich1c_L2, L3, L4, L6, L8`): 1-cycle (X_L, Z_L)
     with a single centered bridge cross, N = 254..12,556, K = 1, w_max <= 6, d_tot ~ 4.1..4.2.
  3. Haah's 3D Cubic Code 1 (`v20_haah_L5, L7, L9, L11, L13, L15, L17`): N = 250..9,826,
     K = 2 (and K = 50 at L = 15 = 2^4 - 1), w_max = 8, d_tot = 8.00.
  4. 4D Toric Code (`v20_tor4d_L3, L4, L5`): N = 486..3,750, K = 6, w_max = 6, d_tot = 8.00.
  5. Multiscale Selective & Palindromic 3D Welded Hierarchies (`v20_weld_*`) evaluated at exact
     matched per-qubit energy budgets `beta_q = beta * (d_tot / 4.0) in {5.50, 6.00, 6.50, 7.00, 7.55}`.
"""

from concurrent.futures import ProcessPoolExecutor
import heapq
import itertools
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v20")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import GeneralCubical2D, seed_planar_11, sheaf_to_css_with_seams
from build_v18_rework import apply_step_general, gf2_rank_bigint
from build_v9_quarter_million import push_X_logical, push_Z_logical


def seed_torus_1face():
    return GeneralCubical2D(1, [(0, 0), (0, 0)], ["none", "none"], [(0, 0, 0, 0, 0, 0, 1, 1)])


def write_and_prep_general_css(tag, family, N, HX, HZ, expected_k=None):
    rX = gf2_rank_bigint(HX)
    rZ = gf2_rank_bigint(HZ)
    k_tot = N - rX - rZ
    if expected_k is not None:
        assert k_tot == expected_k, f"{tag}: got K={k_tot}, expected {expected_k}"

    for sec, checks, stabs in [("X_err", HZ, HX), ("Z_err", HX, HZ)]:
        raw_p = os.path.join(DATA_DIR, f"{tag}_{sec}.raw")
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(raw_p, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, sorted(r))) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, sorted(r))) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, cpath], check=True, stdout=subprocess.DEVNULL)
        os.remove(raw_p)
        vout = subprocess.check_output([ENGINE, "verify", cpath], text=True).strip()
        assert f"K={k_tot}" in vout and "bad_comm=0" in vout, f"{tag}_{sec}: {vout}"

    wX_sum = sum(len(r) for r in HX)
    wZ_sum = sum(len(r) for r in HZ)
    max_wX = max((len(r) for r in HX), default=0)
    max_wZ = max((len(r) for r in HZ), default=0)
    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1

    return {
        "tag": tag,
        "family": family,
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": k_tot,
        "max_wX": max_wX,
        "max_wZ": max_wZ,
        "max_w": max(max_wX, max_wZ),
        "mean_dX": wX_sum / N,
        "mean_dZ": wZ_sum / N,
        "d_tot": (wX_sum + wZ_sum) / N,
        "max_dX": max(dx),
        "max_dZ": max(dz),
    }


def build_welded_spec(tag, family, seed_kind, steps, expected_k):
    if seed_kind == "planar":
        comp = seed_planar_11()
        lx_list, lz_list = [{0, 1}], [{0}]
    elif seed_kind == "tor1":
        comp = seed_torus_1face()
        lx_list, lz_list = [{0}, {1}], [{0}, {1}]
    else:
        raise ValueError(seed_kind)

    cyl = set()
    for L, rb, it, gm in steps:
        pcomp = comp
        comp, cyl, meta = apply_step_general(pcomp, L, rb, it, prev_cyl=cyl, gadget_mode=gm)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx_list = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx_list]
        lz_list = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz_list]

    rX = gf2_rank_bigint(HX)
    rZ = gf2_rank_bigint(HZ)
    k_tot = N - rX - rZ
    assert k_tot == expected_k, f"{tag}: K={k_tot} != {expected_k}"

    for sec, checks, stabs, dlogs in [
        ("X_err", HZ, HX, lz_list),
        ("Z_err", HX, HZ, lx_list),
    ]:
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(cpath, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)} {k_tot}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for lg in dlogs:
                slg = sorted(lg)
                out.write(str(len(slg)) + " " + " ".join(map(str, slg)) + "\n")
        subprocess.run([ENGINE, "verify", cpath], check=True, stdout=subprocess.DEVNULL)

    wX_sum = sum(len(r) for r in HX)
    wZ_sum = sum(len(r) for r in HZ)
    max_wX = max(len(r) for r in HX)
    max_wZ = max(len(r) for r in HZ)
    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1

    return {
        "tag": tag,
        "family": family,
        "seed": seed_kind,
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": k_tot,
        "max_wX": max_wX,
        "max_wZ": max_wZ,
        "max_w": max(max_wX, max_wZ),
        "mean_dX": wX_sum / N,
        "mean_dZ": wZ_sum / N,
        "d_tot": (wX_sum + wZ_sum) / N,
        "max_dX": max(dx),
        "max_dZ": max(dz),
    }


def build_code_job(job):
    kind = job[0]
    if kind == "tor2d":
        _, tag, L = job
        def eid(x, y, d):
            return 2 * ((x % L) * L + (y % L)) + d
        hx = [[eid(x, y, 0), eid(x - 1, y, 0), eid(x, y, 1), eid(x, y - 1, 1)]
              for x in range(L) for y in range(L)]
        hz = [[eid(x, y, 0), eid(x, y + 1, 0), eid(x, y, 1), eid(x + 1, y, 1)]
              for x in range(L) for y in range(L)]
        return write_and_prep_general_css(tag, "toric_2d", 2 * L * L, hx, hz, expected_k=2)

    if kind == "haah3d":
        _, tag, L = job
        def qid(x, y, z, s):
            return 2 * (((x % L) * L + (y % L)) * L + (z % L)) + s
        hx, hz = [], []
        for x, y, z in itertools.product(range(L), repeat=3):
            hx.append([
                qid(x, y, z, 0), qid(x + 1, y + 1, z, 0), qid(x + 1, y, z + 1, 0), qid(x, y + 1, z + 1, 0),
                qid(x, y, z, 1), qid(x + 1, y, z, 1), qid(x, y + 1, z, 1), qid(x, y, z + 1, 1),
            ])
            hz.append([
                qid(x + 1, y + 1, z + 1, 0), qid(x + 1, y + 1, z, 0), qid(x + 1, y, z + 1, 0), qid(x, y + 1, z + 1, 0),
                qid(x + 1, y + 1, z + 1, 1), qid(x, y, z + 1, 1), qid(x, y + 1, z, 1), qid(x + 1, y, z, 1),
            ])
        exp_k = 50 if L == 15 else 2
        return write_and_prep_general_css(tag, "haah_3d", 2 * (L ** 3), hx, hz, expected_k=exp_k)

    if kind == "tor4d":
        _, tag, L = job
        planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pl_idx = {p: i for i, p in enumerate(planes)}
        def q4(c, p):
            x, y, z, w = [v % L for v in c]
            return 6 * (((x * L + y) * L + z) * L + w) + pl_idx[p]
        hx, hz = [], []
        for c in itertools.product(range(L), repeat=4):
            for a in range(4):
                row = []
                for b in range(4):
                    if b == a:
                        continue
                    p = (min(a, b), max(a, b))
                    row.append(q4(c, p))
                    c2 = list(c)
                    c2[b] -= 1
                    row.append(q4(c2, p))
                hx.append(row)
            for omit in range(4):
                dirs = [d for d in range(4) if d != omit]
                row = []
                for i in range(3):
                    d_other = dirs[i]
                    p = tuple(sorted([d for d in dirs if d != d_other]))
                    row.append(q4(c, p))
                    c2 = list(c)
                    c2[d_other] += 1
                    row.append(q4(c2, p))
                hz.append(row)
        return write_and_prep_general_css(tag, "toric_4d", 6 * (L ** 4), hx, hz, expected_k=6)

    if kind == "welded":
        _, tag, family, seed_kind, steps, exp_k = job
        return build_welded_spec(tag, family, seed_kind, steps, exp_k)

    raise ValueError(kind)


def pack_shards(weighted_tasks, num_shards, out_path):
    weighted_tasks.sort(key=lambda x: -x[0])
    heap = [(0.0, s) for s in range(num_shards)]
    heapq.heapify(heap)
    assigned = [[] for _ in range(num_shards)]
    for cost, tstr in weighted_tasks:
        c_tot, sh = heapq.heappop(heap)
        assigned[sh].append(tstr)
        heapq.heappush(heap, (c_tot + cost, sh))
    with open(out_path, "w") as f:
        for sh in range(num_shards):
            for ln in assigned[sh]:
                f.write(f"{sh}|{ln}\n")
    loads = [c for c, _ in heap]
    print(f"Packed {len(weighted_tasks)} tasks across {num_shards} shards (load min={min(loads):.1f}, max={max(loads):.1f})")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    jobs = [
        # 1. 2D Toric Codes
        ("tor2d", "v20_tor2d_L8",  8),
        ("tor2d", "v20_tor2d_L16", 16),
        ("tor2d", "v20_tor2d_L32", 32),
        ("tor2d", "v20_tor2d_L64", 64),
        # 2. Single-Scale 3D Welded Surface Codes (1-cycle Michnicki-type centered single bridge)
        ("welded", "v20_mich1c_L2", "welded_1cycle", "planar", [(2, [1], "X", "full"), (2, [1], "Z", "full")], 1),
        ("welded", "v20_mich1c_L3", "welded_1cycle", "planar", [(3, [1], "X", "full"), (3, [1], "Z", "full")], 1),
        ("welded", "v20_mich1c_L4", "welded_1cycle", "planar", [(4, [2], "X", "full"), (4, [2], "Z", "full")], 1),
        ("welded", "v20_mich1c_L6", "welded_1cycle", "planar", [(6, [3], "X", "full"), (6, [3], "Z", "full")], 1),
        ("welded", "v20_mich1c_L8", "welded_1cycle", "planar", [(8, [4], "X", "full"), (8, [4], "Z", "full")], 1),
        # 3. Haah's 3D Cubic Code 1 across odd L
        ("haah3d", "v20_haah_L5",  5),
        ("haah3d", "v20_haah_L7",  7),
        ("haah3d", "v20_haah_L9",  9),
        ("haah3d", "v20_haah_L11", 11),
        ("haah3d", "v20_haah_L13", 13),
        ("haah3d", "v20_haah_L15", 15),
        ("haah3d", "v20_haah_L17", 17),
        # 4. 4D Toric Codes
        ("tor4d", "v20_tor4d_L3", 3),
        ("tor4d", "v20_tor4d_L4", 4),
        ("tor4d", "v20_tor4d_L5", 5),
        # 5. Multiscale Selective & Palindromic Welded Codes (for exact matched beta_q runs)
        ("welded", "v20_weld_3s_X2R3ZX2", "welded_hier", "planar", [(2, [1], "X", "full"), (3, [1, 2], "Z", "full"), (2, [1], "X", "full")], 1),
        ("welded", "v20_weld_3s_R3X2",    "welded_hier", "planar", [(3, [1, 2], "X", "full"), (3, [1, 2], "Z", "full"), (2, [1], "X", "full")], 1),
        ("welded", "v20_weld_4s_C2C2",    "welded_hier", "planar", [(2, [1], "X", "full"), (2, [1], "Z", "full"), (2, [1], "X", "full"), (2, [1], "Z", "full")], 1),
        ("welded", "v20_weld_4s_sqXZZX",  "welded_hier", "planar", [(2, [1], "X", "full"), (2, [1], "Z", "full"), (2, [1], "Z", "full"), (2, [1], "X", "full")], 1),
        ("welded", "v20_weld_4s_torXZZX", "welded_hier", "tor1",   [(2, [1], "X", "full"), (2, [1], "Z", "full"), (2, [1], "Z", "full"), (2, [1], "X", "full")], 2),
        ("welded", "v20_weld_4s_C3C2ZX",  "welded_hier", "planar", [(3, [1], "X", "full"), (3, [1], "Z", "full"), (2, [1], "Z", "full"), (2, [1], "X", "full")], 1),
    ]

    specs = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for meta in ex.map(build_code_job, jobs):
            specs[meta["tag"]] = meta
            print(f"Built {meta['tag']:22s}: N={meta['N']:7,d} K={meta['K']:3d} w_max={meta['max_w']} d_tot={meta['d_tot']:.4f}")

    with open(os.path.join(WORK_DIR, "v20_code_specs.json"), "w") as f:
        json.dump(specs, f, indent=2)

    # Assemble paired_readout tasks:
    # Mode label encodes normalization clearly: `bq6.50` (exact per-qubit beta_q = beta * d_tot / 4) vs `br6.50` (raw check beta = 6.50)
    arms = []

    # A. Temperature sweeps across beta_q in {5.50, 6.00, 6.50, 7.00, 7.55}
    for bq in (5.50, 6.00, 6.50, 7.00, 7.55):
        arms.append(("v20_tor2d_L16",       f"bq{bq:.2f}", bq * 4.0 / specs["v20_tor2d_L16"]["d_tot"],       1.0, 4.8, 39, 128, 30000000))
        arms.append(("v20_mich1c_L4",       f"bq{bq:.2f}", bq * 4.0 / specs["v20_mich1c_L4"]["d_tot"],       1.5, 5.3, 39, 128, 40000000))
        arms.append(("v20_haah_L7",         f"bq{bq:.2f}", bq * 4.0 / specs["v20_haah_L7"]["d_tot"],         2.0, 6.8, 33, 128, 40000000))
        arms.append(("v20_haah_L11",        f"bq{bq:.2f}", bq * 4.0 / specs["v20_haah_L11"]["d_tot"],        2.0, 6.8, 33, 128, 50000000))
        arms.append(("v20_weld_4s_sqXZZX",  f"bq{bq:.2f}", bq * 4.0 / specs["v20_weld_4s_sqXZZX"]["d_tot"],  2.8, 7.8, 31, 128, 120000000))
        arms.append(("v20_weld_4s_torXZZX", f"bq{bq:.2f}", bq * 4.0 / specs["v20_weld_4s_torXZZX"]["d_tot"], 3.0, 8.0, 31, 128, 120000000))
        if bq <= 7.00:
            arms.append(("v20_tor4d_L4",    f"bq{bq:.2f}", bq * 4.0 / specs["v20_tor4d_L4"]["d_tot"],        3.0, 7.6, 29, 96,  60000000))

    # B. System-size sweeps at fixed per-qubit energy budget beta_q = 6.50 (and 7.55 on Haah & Toric2D)
    for tag in ("v20_tor2d_L8", "v20_tor2d_L32", "v20_tor2d_L64"):
        arms.append((tag, "bq6.50", 6.50 * 4.0 / specs[tag]["d_tot"], 1.0, 4.6, 37, 128, 40000000))
        arms.append((tag, "bq7.55", 7.55 * 4.0 / specs[tag]["d_tot"], 1.5, 5.1, 37, 128, 40000000))

    for tag in ("v20_mich1c_L2", "v20_mich1c_L3", "v20_mich1c_L6", "v20_mich1c_L8"):
        arms.append((tag, "bq6.50", 6.50 * 4.0 / specs[tag]["d_tot"], 1.5, 5.1, 37, 128, 50000000))
        arms.append((tag, "br6.50", 6.50,                             1.5, 5.1, 37, 128, 50000000))
    arms.append(("v20_mich1c_L4", "br6.50", 6.50, 1.5, 5.1, 37, 128, 50000000))

    for tag in ("v20_haah_L5", "v20_haah_L9", "v20_haah_L13", "v20_haah_L15", "v20_haah_L17"):
        arms.append((tag, "bq6.50", 6.50 * 4.0 / specs[tag]["d_tot"], 2.0, 6.2, 29, 128, 60000000))
        arms.append((tag, "bq7.55", 7.55 * 4.0 / specs[tag]["d_tot"], 2.5, 6.9, 30, 96,  80000000))

    for tag in ("v20_tor4d_L3", "v20_tor4d_L5"):
        arms.append((tag, "bq6.50", 6.50 * 4.0 / specs[tag]["d_tot"], 3.0, 7.6, 29, 96, 80000000))

    for tag, lg1, maxf in [
        ("v20_weld_3s_X2R3ZX2", 6.0, 60000000),
        ("v20_weld_3s_R3X2",    6.2, 80000000),
        ("v20_weld_4s_C2C2",    6.2, 80000000),
        ("v20_weld_4s_C3C2ZX",  7.0, 120000000),
    ]:
        arms.append((tag, "bq6.50", 6.50 * 4.0 / specs[tag]["d_tot"], 2.2, lg1, 29, 96, maxf))

    # C. Unnormalized raw J=1 (`br6.50`, plus `bg6.50` equal bulk single-flip gap beta=3.25 == bq6.50 on Haah/4D, and equal bulk gap on welded = br6.50)
    arms.append(("v20_haah_L7",  "br6.50", 6.50, 4.0, 9.0, 26, 64, 80000000))
    arms.append(("v20_haah_L11", "br6.50", 6.50, 4.0, 8.6, 24, 64, 80000000))
    arms.append(("v20_tor4d_L4", "br6.50", 6.50, 4.0, 8.8, 25, 64, 80000000))

    # Split into chunks of 32 trials for fine-grained parallel load balancing across 64 submit workers
    weighted = []
    chunk = 32
    for tag, mode_lbl, beta_val, lg0, lg1, ng, tot_trials, maxf in arms:
        n_val = specs[tag]["N"]
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, tot_trials, chunk):
                ntr = min(chunk, tot_trials - off)
                tid = f"v20__{tag}_{sec}__{mode_lbl}__b{beta_val:.5f}__o{off}"
                cost = max(1.0, (n_val / 2000.0) * (ntr / 32.0) * max(1.0, lg1 - 3.5))
                cmd_s = f"{tid}|paired_readout|{cpath}|{beta_val:.6f}|{ntr}|{off + 1}|{lg0:.4f}|{lg1:.4f}|{ng}|{maxf}"
                weighted.append((cost, cmd_s))

    pack_shards(weighted, 64, os.path.join(WORK_DIR, "manifest_v20.txt"))


if __name__ == "__main__":
    main()
