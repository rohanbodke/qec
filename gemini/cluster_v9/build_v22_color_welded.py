#!/usr/bin/env python3
"""
build_v22_color_welded.py
Builds and verifies the V22 3D-Local Welded 4.8.8 Flag Color-Code campaign using `kmc_engine_v19`
(`prep_logicals`, `verify`, and 2-trial `self_test` gate) and emits a balanced 96-shard SLURM manifest
`manifest_v22.txt` + `v22_code_specs.json`.

Code families (all periodic torus seeds with K = 4, deg >= 3 per sector -> dE_1 >= 3 and ZERO dE = 0 bulk flips):
  1. Unwelded 2D 4.8.8 Color Code Tori: `v22_cc2d_L3` (72), `L6` (288), `L9` (648), `L12` (1,152)
  2. 1-Step 3D-Local Welded Color Codes: `v22_cc1s_L3r1` (192), `L3r2` (240), `L3r2_sd` (240 self-dual ablation)
  3. 2-Step (1-Cycle) Pure-L=3 3D-Local Welded Color Codes (L_blk = 9, dim_F < 3):
     `v22_cc2s_C3` (3,792), `RX3` (4,128), `RZ3` (4,704), `R3` (5,088), `R3_sd` (5,088 self-dual), `mix32` (2,208)
  4. 3-Step Pure-L=3 3D-Local Welded Color Codes (L_blk = 27, dim_F < 3):
     `v22_cc3s_X3r1Z3r1X3r1` (75,504), `X3r1Z3r2X3r1` (81,888), `R3X3r2` (110,400)
"""

from collections import Counter, defaultdict
import heapq
import json
import math
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data_v22")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import GeneralCubical2D
from build_v18_rework import gf2_rank_bigint
from build_v21_3d_local import seed_torus_1face, apply_step_L3_general


def build_flag_color_code(comp, seam_mode="welded"):
    flags = []
    fid_map = {}
    vert_flags = [[] for _ in range(comp.num_vertices)]
    for f_idx, (v00, v10, v11, v01, eb, et, el, er) in enumerate(comp.faces):
        for v, e in [(v00, eb), (v00, el), (v10, eb), (v10, er), (v11, et), (v11, er), (v01, et), (v01, el)]:
            qid = len(flags)
            fid_map[(v, e, f_idx)] = qid
            flags.append((v, e, f_idx))
            vert_flags[v].append(qid)
    N = len(flags)
    HX, HZ = [], []

    for f_idx in range(len(comp.faces)):
        row = list(range(8 * f_idx, 8 * (f_idx + 1)))
        HX.append(row)
        HZ.append(list(row))

    for e_idx, inc in enumerate(comp.edge_faces):
        d = len(inc)
        u, w = comp.edges[e_idx]
        col = comp.edge_colors[e_idx]
        if d == 1:
            if col == "blue":
                row = sorted([fid_map[(u, e_idx, inc[0])], fid_map[(w, e_idx, inc[0])]])
                HX.append(row)
                HZ.append(list(row))
        elif d == 2 or seam_mode == "selfdual":
            for j in range(d - 1):
                f1, f2 = inc[j], inc[j + 1]
                row = sorted([
                    fid_map[(u, e_idx, f1)], fid_map[(w, e_idx, f1)],
                    fid_map[(u, e_idx, f2)], fid_map[(w, e_idx, f2)],
                ])
                HX.append(row)
                HZ.append(list(row))
        else:
            pair_rows = [
                sorted([
                    fid_map[(u, e_idx, inc[j])], fid_map[(w, e_idx, inc[j])],
                    fid_map[(u, e_idx, inc[j + 1])], fid_map[(w, e_idx, inc[j + 1])],
                ])
                for j in range(d - 1)
            ]
            sum_row = sorted(
                [fid_map[(u, e_idx, fj)] for fj in inc] + [fid_map[(w, e_idx, fj)] for fj in inc]
            )
            if col == "blue":
                for r in pair_rows:
                    HX.append(r)
                HZ.append(sum_row)
            else:
                for r in pair_rows:
                    HZ.append(r)
                HX.append(sum_row)

    for v in range(comp.num_vertices):
        f_at_v = sorted(set(comp.vert_faces[v]))
        nf = len(f_at_v)
        if nf == 0:
            continue
        f_to_i = {f: i for i, f in enumerate(f_at_v)}
        v_qids = vert_flags[v]
        for same_col, target_H in [("red", HX), ("blue", HZ)]:
            cons = []
            for e_idx in comp.vert_edges[v]:
                inc = [f for f in comp.edge_faces[e_idx] if f in f_to_i]
                d = len(comp.edge_faces[e_idx])
                col = comp.edge_colors[e_idx]
                if d == 1:
                    if col == "blue":
                        cons.append(1 << f_to_i[inc[0]])
                elif d == 2 or seam_mode == "selfdual" or col == same_col:
                    for j in range(len(inc) - 1):
                        cons.append((1 << f_to_i[inc[j]]) ^ (1 << f_to_i[inc[j + 1]]))
                else:
                    m = 0
                    for f in inc:
                        m ^= (1 << f_to_i[f])
                    cons.append(m)

            rref = []
            for c in cons:
                x = c
                for r in rref:
                    lb = r.bit_length() - 1
                    if (x >> lb) & 1:
                        x ^= r
                if x:
                    lb = x.bit_length() - 1
                    rref = [r ^ x if ((r >> lb) & 1) else r for r in rref]
                    rref.append(x)
                    rref.sort(reverse=True)
            piv_cols = {r.bit_length() - 1 for r in rref}
            null_basis = []
            for fb in range(nf):
                if fb in piv_cols:
                    continue
                vec = 1 << fb
                for r in rref:
                    lb = r.bit_length() - 1
                    if (r >> fb) & 1:
                        vec |= (1 << lb)
                null_basis.append(vec)

            dim = len(null_basis)
            if 1 < dim <= 7:
                cands = []
                for mask in range(1, 1 << dim):
                    vf = 0
                    for i in range(dim):
                        if (mask >> i) & 1:
                            vf ^= null_basis[i]
                    cands.append((vf.bit_count(), mask, vf))
                cands.sort()
                cr, chosen = [], []
                for _, _, vf in cands:
                    x = vf
                    for r in cr:
                        lb = r.bit_length() - 1
                        if (x >> lb) & 1:
                            x ^= r
                    if x:
                        lb = x.bit_length() - 1
                        cr = [r ^ x if ((r >> lb) & 1) else r for r in cr]
                        cr.append(x)
                        chosen.append(vf)
                        if len(chosen) == dim:
                            break
                null_basis = chosen

            for vf in null_basis:
                fset = {f_at_v[i] for i in range(nf) if (vf >> i) & 1}
                row = sorted([qi for qi in v_qids if flags[qi][2] in fset])
                target_H.append(row)

    return N, HX, HZ


def build_2d_488_torus(L):
    def vid(x, y):
        return (x % L) * L + (y % L)

    edges = []
    he = [[None] * L for _ in range(L)]
    ve = [[None] * L for _ in range(L)]
    for x in range(L):
        for y in range(L):
            he[x][y] = len(edges)
            edges.append((vid(x, y), vid(x + 1, y)))
            ve[x][y] = len(edges)
            edges.append((vid(x, y), vid(x, y + 1)))
    faces = []
    for x in range(L):
        for y in range(L):
            faces.append((
                vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1),
                he[x][y], he[x][(y + 1) % L], ve[x][y], ve[(x + 1) % L][y],
            ))
    comp = GeneralCubical2D(L * L, edges, ["none"] * len(edges), faces)
    N, HX, HZ = build_flag_color_code(comp, seam_mode="selfdual")
    return N, HX, HZ, len(faces)


def build_code_job(job):
    tag, kind, arg, seam_mode = job
    if kind == "cc2d":
        L_blk = arg
        N, HX, HZ, F = build_2d_488_torus(L_blk)
    else:
        comp = seed_torus_1face()
        cyl = set()
        L_blk = 1
        for (L, rb, it) in arg:
            comp, cyl, _ = apply_step_L3_general(comp, L, rb, it, prev_cyl=cyl)
            L_blk *= L
        F = len(comp.faces)
        N, HX, HZ = build_flag_color_code(comp, seam_mode=seam_mode)

    k_exp = 4
    if N <= 6000:
        rX = gf2_rank_bigint(HX)
        rZ = gf2_rank_bigint(HZ)
        k_tot = N - rX - rZ
        assert k_tot == k_exp, f"{tag}: got K={k_tot} != {k_exp}"
    else:
        rX, rZ, k_tot = -1, -1, k_exp

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
        # 2-trial self_test gate (verifies every decoded correction strictly matches syndrome!)
        subprocess.run([ENGINE, "self_test", cpath, "4.5", "2", "1", "1.0", "2.0", "2", "50000", "2"],
                       check=True, stdout=subprocess.DEVNULL)

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1
    dx = [0] * N
    for r in HX:
        for q in r:
            dx[q] += 1
    return tag, {
        "tag": tag,
        "kind": kind,
        "seam_mode": seam_mode,
        "N": N,
        "F": F,
        "M_X": len(HX),
        "M_Z": len(HZ),
        "rank_HX": rX,
        "rank_HZ": rZ,
        "K": k_exp,
        "L_blk": L_blk,
        "dim_F": (math.log(F) / math.log(L_blk)) if L_blk > 1 else 0.0,
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
    os.makedirs(os.path.join(WORK_DIR, "results_v22"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "logs_v22"), exist_ok=True)

    jobs = [
        ("v22_cc2d_L3",           "cc2d", 3,                                  "selfdual"),
        ("v22_cc2d_L6",           "cc2d", 6,                                  "selfdual"),
        ("v22_cc2d_L9",           "cc2d", 9,                                  "selfdual"),
        ("v22_cc2d_L12",          "cc2d", 12,                                 "selfdual"),
        ("v22_cc1s_L3r1",         "weld", [(3, [1], "X")],                     "welded"),
        ("v22_cc1s_L3r2",         "weld", [(3, [1, 2], "X")],                  "welded"),
        ("v22_cc1s_L3r2_sd",      "weld", [(3, [1, 2], "X")],                  "selfdual"),
        ("v22_cc2s_mix32",        "weld", [(3, [1, 2], "X"), (2, [1], "Z")],   "welded"),
        ("v22_cc2s_C3",           "weld", [(3, [1], "X"),    (3, [1], "Z")],   "welded"),
        ("v22_cc2s_RX3",          "weld", [(3, [1, 2], "X"), (3, [1], "Z")],   "welded"),
        ("v22_cc2s_RZ3",          "weld", [(3, [1], "X"),    (3, [1, 2], "Z")],"welded"),
        ("v22_cc2s_R3",           "weld", [(3, [1, 2], "X"), (3, [1, 2], "Z")],"welded"),
        ("v22_cc2s_R3_sd",        "weld", [(3, [1, 2], "X"), (3, [1, 2], "Z")],"selfdual"),
        ("v22_cc3s_X3r1Z3r1X3r1", "weld", [(3, [1], "X"), (3, [1], "Z"), (3, [1], "X")],       "welded"),
        ("v22_cc3s_X3r1Z3r2X3r1", "weld", [(3, [1], "X"), (3, [1, 2], "Z"), (3, [1], "X")],    "welded"),
        ("v22_cc3s_R3X3r2",       "weld", [(3, [1, 2], "X"), (3, [1, 2], "Z"), (3, [1, 2], "X")],"welded"),
    ]

    specs = {}
    for job in jobs:
        tag, info = build_code_job(job)
        specs[tag] = info
        print(
            f"  [BUILD] {tag:24s}: N={info['N']:7,d} M=({info['M_X']},{info['M_Z']}) "
            f"d=({info['min_dX']},{info['min_dZ']}) w_mean=({info['mean_wX']:.2f},{info['mean_wZ']:.2f}) "
            f"dimF={info['dim_F']:.3f}",
            flush=True,
        )

    with open(os.path.join(WORK_DIR, "v22_code_specs.json"), "w") as f:
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
                tasks.append((base_w, f"{tid}|paired_readout|{cpath}|{beta:.5f}|{nt}|{off}|{lg0:.4f}|{lg1:.4f}|{ng}|{int(max_fl)}"))

    def add_iid(tag, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23):
        N = specs[tag]["N"]
        base_w = max(5.0, N / 400.0)
        for sec in ("X_err", "Z_err"):
            cpath = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
            for off in range(0, n_tot, per_task):
                nt = min(per_task, n_tot - off)
                tid = f"v22_iid__{tag}_{sec}__o{off}"
                tasks.append((base_w, f"{tid}|iid_readout|{cpath}|{nt}|{off}|{p0:.5f}|{p1:.5f}|{ng}"))

    # 1. Unwelded 2D 4.8.8 tori & 1-step 3D welded (beta in 4.0..6.5)
    grid_1s = [
        (4.0, 96, 32, 1.0, 6.5, 23, 50_000_000),
        (4.5, 96, 32, 1.5, 7.5, 25, 50_000_000),
        (5.0, 96, 32, 2.0, 8.0, 25, 50_000_000),
        (5.5, 96, 32, 3.0, 8.5, 23, 50_000_000),
        (6.0, 96, 32, 3.5, 9.0, 23, 50_000_000),
        (6.5, 96, 32, 4.0, 9.5, 23, 50_000_000),
    ]
    for t in ("v22_cc2d_L3", "v22_cc2d_L6", "v22_cc2d_L9", "v22_cc2d_L12",
              "v22_cc1s_L3r1", "v22_cc1s_L3r2", "v22_cc1s_L3r2_sd"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_1s:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=0.5)

    # 2. 2-step 1-cycle 3D-local welded color codes
    grid_2s = [
        (4.0, 64, 32, 1.5, 6.0, 19, 60_000_000),
        (4.5, 64, 32, 2.0, 7.0, 21, 60_000_000),
        (5.0, 96, 32, 2.5, 8.0, 23, 60_000_000),
        (5.5, 64, 32, 3.5, 8.5, 21, 60_000_000),
        (6.0, 64, 32, 4.5, 9.0, 19, 60_000_000),
        (6.5, 96, 32, 5.0, 9.5, 19, 80_000_000),
    ]
    for t in ("v22_cc2s_mix32", "v22_cc2s_C3", "v22_cc2s_RX3", "v22_cc2s_RZ3", "v22_cc2s_R3", "v22_cc2s_R3_sd"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_2s:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=1.5)

    # 3. 3-step pure-L=3 3D-local welded color codes (N = 75.5k .. 110.4k)
    grid_3s = [
        (4.5, 64, 16, 1.5, 5.5, 17, 60_000_000),
        (5.0, 64, 16, 2.0, 6.8, 20, 60_000_000),
        (5.5, 64, 16, 3.0, 7.8, 20, 60_000_000),
        (6.0, 64, 16, 4.0, 8.8, 20, 60_000_000),
        (6.5, 64, 16, 5.0, 9.5, 19, 60_000_000),
    ]
    for t in ("v22_cc3s_X3r1Z3r1X3r1", "v22_cc3s_X3r1Z3r2X3r1", "v22_cc3s_R3X3r2"):
        for b, ntot, pt, lg0, lg1, ng, mfl in grid_3s:
            add_th(t, b, ntot, pt, lg0, lg1, ng, mfl, w_mult=2.5)

    # 4. Paired 4-decoder i.i.d. threshold scans
    for t in ("v22_cc2d_L9", "v22_cc2s_R3", "v22_cc3s_X3r1Z3r2X3r1"):
        add_iid(t, n_tot=64, per_task=16, p0=0.005, p1=0.115, ng=23)

    n_shards = 96
    bins = [(0.0, i) for i in range(n_shards)]
    heapq.heapify(bins)
    assigned = [[] for _ in range(n_shards)]
    for w, line in sorted(tasks, key=lambda x: -x[0]):
        load, sh = heapq.heappop(bins)
        assigned[sh].append(line)
        heapq.heappush(bins, (load + w, sh))

    mpath = os.path.join(WORK_DIR, "manifest_v22.txt")
    with open(mpath, "w") as out:
        for sh in range(n_shards):
            for line in assigned[sh]:
                out.write(f"{sh}|{line}\n")
    print(f"Wrote {len(tasks)} tasks across {n_shards} shards to {mpath}")


if __name__ == "__main__":
    main()
