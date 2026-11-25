#!/usr/bin/env python3
"""
build_v19_rebalance_and_decoders.py
Builds and mathematically verifies all V19 `.code` files (`H_X * H_Z^T == 0`, exact GF(2) ranks,
`L_X . L_Z^T == I_K`, and C++ `kmc_engine_v19 verify`), records exact code specs in
`v19_code_specs.json`, and writes a load-balanced 96-shard SLURM manifest `manifest_v19.txt`
covering BOTH new research directions:

  Direction 1 — Sector-Rebalanced & Palindromic Hierarchies:
    - Suite R1a (Complete 4-Step Welding-Order Quartet at L=2: XZXZ, ZXZX, XZZX, ZXXZ):
        Planar (K=1) and symmetric 1-face Torus (K=2) across beta in {5.50, 6.00, 6.50, 7.00}.
    - Suite R1b (3D-Continuable Coarse-L=3 Palindromic Hierarchies: C3_C2ZX, RX3_C2ZX, RZ3_C2ZX, R3_C2ZX
        vs XZXZ references C3_C2, RX3_C2, RZ3_C2, R3_C2):
        Planar (K=1) at beta = 6.50 and beta = 7.00.
    - Suite R2 (Ultra-Compact 3-Step Balanced (X, Z, X) Family, N = 1,942..10,580):
        `X2_Z2_X2`, `X2_Z3r1_X2`, `X2_Z3r2_X2` (2 bridges in bottleneck middle Z step),
        `X3r2_Z2_X2` (matched-size control: 2 bridges in non-limiting outer X step),
        `X3r2_Z3r2_X2`, and `Z2_X2_Z2` at beta in {6.00, 6.50, 7.00}.

  Direction 2 — Separating Thermal Basin Crossing from Classical Decoder Suboptimality:
    - Every V19 task runs `paired_readout` evaluating `UF`, `Cool-UF`, `Hier-UF`, and `BP-OSD`
      simultaneously on IDENTICAL BKL trajectories while tagging exact zero-syndrome ('G') and
      T=0 quench-settled ('Q') wrong-basin states.
    - Suite D_cal: 1-step `C2` (N=19, beta=7.00, n=2,048) calibrated directly against exact
      analytical Markov maximum-likelihood recovery P^opt(t), plus 1-cycle modules (C2, C3, RX3, R3)
      and fine-L=3 hierarchies (`C2_C3`, `C2_RX3`, `C2_R3`) at beta = 6.50.
    - Suite D_iid: Unrelaxed i.i.d. bit-flip error-rate sweeps (`iid_readout`) on `C2_C2`, `C2_XZZX`,
      `C3_C2`, `C2_R3`, and `RX3_C2` demonstrating substantial `BP-OSD` / `Cool-UF` / `Hier-UF`
      gains under dense unrelaxed syndromes vs bath-dominated saddle crossing under cold Glauber noise.
"""

from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v19")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import GeneralCubical2D, seed_planar_11, sheaf_to_css_with_seams
from build_v18_rework import apply_step_general, gf2_rank_bigint
from build_v9_quarter_million import push_X_logical, push_Z_logical


def seed_torus_1face():
    return GeneralCubical2D(1, [(0, 0), (0, 0)], ["none", "none"], [(0, 0, 0, 0, 0, 0, 1, 1)])


def build_one_v19_code(spec):
    tag, seed_kind, steps, expected_k = spec
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

    K = len(lx_list)
    assert K == expected_k
    for a in range(K):
        assert sum(1 for r in HZ if (sum(1 for q in r if q in lx_list[a]) & 1)) == 0
        assert sum(1 for r in HX if (sum(1 for q in r if q in lz_list[a]) & 1)) == 0
        for b in range(K):
            assert (len(lx_list[a] & lz_list[b]) & 1) == (1 if a == b else 0)

    rX = gf2_rank_bigint(HX)
    rZ = gf2_rank_bigint(HZ)
    k_tot = N - rX - rZ
    assert k_tot == expected_k, f"{tag}: got K={k_tot}, expected {expected_k}"

    for sec, checks, stabs, dlogs in [
        ("X_err", HZ, HX, lz_list),
        ("Z_err", HX, HZ, lx_list),
    ]:
        if len(checks) == 0:
            continue
        cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        with open(cpath, "w") as out:
            out.write(f"{N} {len(checks)} {len(stabs)} {K}\n")
            for r in checks:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                out.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for lg in dlogs:
                slg = sorted(lg)
                out.write(str(len(slg)) + " " + " ".join(map(str, slg)) + "\n")
        subprocess.run([ENGINE, "verify", cpath], check=True, stdout=subprocess.DEVNULL)

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
        "seed": seed_kind,
        "steps": [(L, list(rb), it, gm) for (L, rb, it, gm) in steps],
        "N": N,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": k_tot,
        "clean_hazard_X_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dz),
        "clean_hazard_Z_65": sum(1.0 / (1.0 + math.exp(6.5 * d)) for d in dx),
        "clean_hazard_X_70": sum(1.0 / (1.0 + math.exp(7.0 * d)) for d in dz),
        "clean_hazard_Z_70": sum(1.0 / (1.0 + math.exp(7.0 * d)) for d in dx),
    }


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
            for tstr in assigned[sh]:
                f.write(f"{sh}|{tstr}\n")
    loads = [c for (c, _) in heap]
    print(
        f"Wrote {out_path}: tasks={len(weighted_tasks)}, shards={num_shards}, "
        f"load=[{min(loads):.1f}, {max(loads):.1f}]",
        flush=True,
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    X2 = (2, [1], "X", "full")
    Z2 = (2, [1], "Z", "full")
    X3_1 = (3, [1], "X", "full")
    Z3_1 = (3, [1], "Z", "full")
    X3_2 = (3, [1, 2], "X", "full")
    Z3_2 = (3, [1, 2], "Z", "full")

    specs = [
        # D_cal: 1-step C2 (exact Markov MLD calibration) & 1-cycle modules
        ("v19_sq_C2_1step",    "planar", [X2],                   1),
        ("v19_sq_C2_1cyc",     "planar", [X2, Z2],               1),
        ("v19_sq_C3_1cyc",     "planar", [X3_1, Z3_1],           1),
        ("v19_sq_RX3_1cyc",    "planar", [X3_2, Z3_1],           1),
        ("v19_sq_R3_1cyc",     "planar", [X3_2, Z3_2],           1),
        # R2: Ultra-compact 3-step balanced (X, Z, X) family + (Z, X, Z) dual
        ("v19_sq_3s_X2Z2X2",     "planar", [X2, Z2, X2],         1),
        ("v19_sq_3s_X2Z3r1X2",   "planar", [X2, Z3_1, X2],       1),
        ("v19_sq_3s_X2Z3r2X2",   "planar", [X2, Z3_2, X2],       1),
        ("v19_sq_3s_X3r2Z2X2",   "planar", [X3_2, Z2, X2],       1),
        ("v19_sq_3s_X3r2Z3r2X2", "planar", [X3_2, Z3_2, X2],     1),
        ("v19_sq_3s_Z2X2Z2",     "planar", [Z2, X2, Z2],         1),
        # R1a: Complete L=2 4-step Welding-Order Quartet (Planar K=1 & Torus K=2)
        ("v19_sq_C2_XZXZ",     "planar", [X2, Z2, X2, Z2],       1),
        ("v19_sq_C2_ZXZX",     "planar", [Z2, X2, Z2, X2],       1),
        ("v19_sq_C2_XZZX",     "planar", [X2, Z2, Z2, X2],       1),
        ("v19_sq_C2_ZXXZ",     "planar", [Z2, X2, X2, Z2],       1),
        ("v19_tor_C2_XZXZ",    "tor1",   [X2, Z2, X2, Z2],       2),
        ("v19_tor_C2_ZXZX",    "tor1",   [Z2, X2, Z2, X2],       2),
        ("v19_tor_C2_XZZX",    "tor1",   [X2, Z2, Z2, X2],       2),
        ("v19_tor_C2_ZXXZ",    "tor1",   [Z2, X2, X2, Z2],       2),
        # R1b & D_hier: Coarse-L=3 Continuable Palindromic (XZZX) vs Standard (XZXZ) + Fine-L=3 Decoder Stress
        ("v19_sq_C3_C2ZX",     "planar", [X3_1, Z3_1, Z2, X2],   1),
        ("v19_sq_RX3_C2ZX",    "planar", [X3_2, Z3_1, Z2, X2],   1),
        ("v19_sq_RZ3_C2ZX",    "planar", [X3_1, Z3_2, Z2, X2],   1),
        ("v19_sq_R3_C2ZX",     "planar", [X3_2, Z3_2, Z2, X2],   1),
        ("v19_sq_C3_C2",       "planar", [X3_1, Z3_1, X2, Z2],   1),
        ("v19_sq_RX3_C2",      "planar", [X3_2, Z3_1, X2, Z2],   1),
        ("v19_sq_R3_C2",       "planar", [X3_2, Z3_2, X2, Z2],   1),
        ("v19_sq_C2_C3",       "planar", [X2, Z2, X3_1, Z3_1],   1),
        ("v19_sq_C2_RX3",      "planar", [X2, Z2, X3_2, Z3_1],   1),
        ("v19_sq_C2_R3",       "planar", [X2, Z2, X3_2, Z3_2],   1),
    ]

    meta_out = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for info in ex.map(build_one_v19_code, specs):
            meta_out[info["tag"]] = info
            print(
                f"Verified {info['tag']:24s}: N={info['N']:7,d} K={info['K']} "
                f"(rX={info['rank_HX']:6,d}, rZ={info['rank_HZ']:6,d})",
                flush=True,
            )

    with open(os.path.join(WORK_DIR, "v19_code_specs.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    tasks = []

    def add_paired(prefix, tag, sec, b, trials, off, lg0, lg1, ng, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        tid = f"{prefix}__{tag}_{sec}__b{b:.2f}__o{off}"
        tasks.append((cost, f"{tid}|paired_readout|{cp}|{b:.2f}|{trials}|{off}|{lg0:.2f}|{lg1:.2f}|{ng}|{cap}"))

    def add_iid(prefix, tag, sec, trials, off, p0, p1, ng, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        tid = f"{prefix}__{tag}_{sec}__iid__o{off}"
        tasks.append((cost, f"{tid}|iid_readout|{cp}|{trials}|{off}|{p0:.5f}|{p1:.5f}|{ng}"))

    # 1. Exact MLD Calibration on 1-step C2 (N=19, beta=7.0, n_X=2048, n_Z=2048) & 1-cycle modules (beta=6.5, 256/256):
    for sh in range(4):
        add_paired("dcal", "v19_sq_C2_1step", "X_err", 7.00, 512, 512 * sh, 0.50, 6.00, 56, 200_000_000, 5.0)
        add_paired("dcal", "v19_sq_C2_1step", "Z_err", 7.00, 512, 512 * sh, 0.50, 6.00, 56, 200_000_000, 4.0)
    for tag in ["v19_sq_C2_1cyc", "v19_sq_C3_1cyc", "v19_sq_RX3_1cyc", "v19_sq_R3_1cyc"]:
        for sec in ("X_err", "Z_err"):
            for sh in range(4):
                add_paired("dmod", tag, sec, 6.50, 64, 64 * sh, 1.50, 5.60, 42, 400_000_000, 8.0)

    # 2. Suite R2: 3-Step Balanced (X, Z, X) Family across beta in {6.00, 6.50, 7.00} (n_X=128, n_Z=128):
    r2_tags = [
        "v19_sq_3s_X2Z2X2", "v19_sq_3s_X2Z3r1X2", "v19_sq_3s_X2Z3r2X2",
        "v19_sq_3s_X3r2Z2X2", "v19_sq_3s_X3r2Z3r2X2", "v19_sq_3s_Z2X2Z2",
    ]
    for tag in r2_tags:
        for b, lg1, cap in [(6.00, 6.20, 600_000_000), (6.50, 7.00, 900_000_000), (7.00, 7.80, 1_500_000_000)]:
            for sec in ("X_err", "Z_err"):
                for sh in range(8):
                    add_paired("r2", tag, sec, b, 16, 16 * sh, 1.80, lg1, 45, cap, 14.0)

    # 3. Suite R1a: Complete L=2 Welding-Order Quartet (Planar & Torus) across beta in {5.50, 6.00, 6.50, 7.00}:
    quartet_tags = [
        "v19_sq_C2_XZXZ", "v19_sq_C2_ZXZX", "v19_sq_C2_XZZX", "v19_sq_C2_ZXXZ",
        "v19_tor_C2_XZXZ", "v19_tor_C2_ZXZX", "v19_tor_C2_XZZX", "v19_tor_C2_ZXXZ",
    ]
    for tag in quartet_tags:
        # Warm/mid betas (5.50, 6.00): 96 X, 96 Z (uncensored Arrhenius & dense-defect decoder test!)
        for b, lg0, lg1, cap, cst in [
            (5.50, 1.50, 5.50, 500_000_000, 14.0),
            (6.00, 2.00, 6.60, 900_000_000, 18.0),
        ]:
            for sec in ("X_err", "Z_err"):
                for sh in range(12):
                    add_paired("r1a", tag, sec, b, 8, 8 * sh, lg0, lg1, 41, cap, cst)
        # Main operating point beta = 6.50: 128 X, 128 Z (step 0.10):
        lg1_65 = 6.50 if ("XZXZ" in tag or "ZXZX" in tag) else 8.00
        ng_65 = int(round((lg1_65 - 2.40) / 0.10)) + 1
        cap_65 = 500_000_000 if lg1_65 < 7.0 else 2_200_000_000
        for sec in ("X_err", "Z_err"):
            for sh in range(16):
                add_paired("r1a", tag, sec, 6.50, 8, 8 * sh, 2.40, lg1_65, ng_65, cap_65, 22.0)
        # Cold point beta = 7.00: 96 X, 96 Z (step 0.10):
        lg1_70 = 7.30 if ("XZXZ" in tag or "ZXZX" in tag) else 7.80
        ng_70 = int(round((lg1_70 - 3.00) / 0.10)) + 1
        cap_70 = 800_000_000 if lg1_70 < 7.5 else 1_400_000_000
        for sec in ("X_err", "Z_err"):
            for sh in range(12):
                add_paired("r1a", tag, sec, 7.00, 8, 8 * sh, 3.00, lg1_70, ng_70, cap_70, 28.0)

    # 4. Suite R1b & D_hier: Coarse-L=3 Palindromic (XZZX) vs Standard (XZXZ) + Fine-L=3 Decoder Comparison:
    for tag in ("v19_sq_C3_C2ZX", "v19_sq_RX3_C2ZX", "v19_sq_RZ3_C2ZX", "v19_sq_R3_C2ZX"):
        for sec in ("X_err", "Z_err"):
            for sh in range(8):
                add_paired("r1b60", tag, sec, 6.00, 8, 8 * sh, 2.00, 6.80, 49, 600_000_000, 16.0)

    b65_big = [
        "v19_sq_C3_C2ZX", "v19_sq_RX3_C2ZX", "v19_sq_RZ3_C2ZX", "v19_sq_R3_C2ZX",
        "v19_sq_C3_C2", "v19_sq_RX3_C2", "v19_sq_R3_C2",
        "v19_sq_C2_C3", "v19_sq_C2_RX3", "v19_sq_C2_R3",
    ]
    for tag in b65_big:
        if tag in ("v19_sq_C2_C3", "v19_sq_C2_RX3", "v19_sq_C2_R3"):
            lg1, ng, cap = 6.00, 45, 1_200_000_000
        elif tag in ("v19_sq_C3_C2", "v19_sq_RX3_C2", "v19_sq_R3_C2"):
            lg1, ng, cap = 6.25, 34, 400_000_000
        else:
            lg1, ng, cap = 7.60, 45, 2_400_000_000
        for sec in ("X_err", "Z_err"):
            for sh in range(12):
                add_paired("r1b", tag, sec, 6.50, 8, 8 * sh, 2.20, lg1, ng, cap, 28.0)

    # At beta = 7.00 on Continuable Frontier (64 X, 64 Z):
    for tag in ["v19_sq_C3_C2ZX", "v19_sq_RX3_C2ZX", "v19_sq_RZ3_C2ZX", "v19_sq_R3_C2ZX", "v19_sq_RX3_C2", "v19_sq_R3_C2", "v19_sq_C3_C2"]:
        lg1 = 7.30 if tag in ("v19_sq_C3_C2", "v19_sq_R3_C2") else 7.80
        ng = int(round((lg1 - 3.00) / 0.10)) + 1
        cap = 900_000_000 if lg1 < 7.5 else 1_600_000_000
        for sec in ("X_err", "Z_err"):
            for sh in range(8):
                add_paired("r1c", tag, sec, 7.00, 8, 8 * sh, 3.00, lg1, ng, cap, 32.0)

    # 5. Suite D_iid: Unrelaxed i.i.d. Physical Bit-Flip Sweeps (`iid_readout`, 256 trials, p in [0.004, 0.085], 30 pts):
    for tag in ["v19_sq_C2_XZXZ", "v19_sq_C2_XZZX", "v19_sq_C3_C2", "v19_sq_C2_R3", "v19_sq_RX3_C2"]:
        for sec in ("X_err", "Z_err"):
            for sh in range(8):
                add_iid("diid", tag, sec, 32, 32 * sh, 0.00400, 0.08500, 30, 12.0)

    pack_shards(tasks, 120, os.path.join(WORK_DIR, "manifest_v19.txt"))


if __name__ == "__main__":
    main()
