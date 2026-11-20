#!/usr/bin/env python3
"""
build_v12_euclidean_and_highrate.py
Builds and verifies all codes, wire-graded coupling schedules, and Slurm manifest for V12
addressing Open Problems #1 and #2:

  1A. Minimal 3D-Euclidean-Compatible `3x3` Single-Cross Hierarchy (`dim = log_3(24) = 2.893 < 3`,
      exact cycle contraction 10/63 ≈ 0.159):
      - `sq_rx_L3r1` s=1,2,3 (41 -> 935 -> 22,097q, X_err)
      - `sq_L3r1`    s=1,2,3 (41 -> 1,271 -> 24,449q, X_err & Z_err)
  1B. Defeating Wire Entropy on Collision-Free Z^3 Cubic Grids:
      - `sq_bdl_8_8_r2` (56,312q), `sq_z3_4_6_r2` (10,280q), `sq_z3_3_5_r1` (3,067q)
      - Both uniform (J = 1) and bounded wire-graded hills (`wire_hill.j`, J in [1.0, 2.2])
  2A. Beating Crystal & Layer Codes Below 2,500 Qubits Without Finite-Size Turnover:
      - `sq_hyb21` s=2 (586q, K=1) -> s=4 (129,148q, K=1)
      - `sq_hyb21_iso` s=2 (790q, K=1)
      - `rottor_hyb21` s=2 (1,172q, K=2, boundaryless!) & `rottor_hyb21_iso` s=2 (1,580q, K=2)
  2B. Constant-Rate Multi-Qubit Hyperbolic Square-Tiling Welded Memories:
      - `hyp46_unif` ({4,6} 12-square seed, K=6): uniform 2x2 (3,876q = 646q/log)
      - `hyp46_hyb21` ({4,6} 12-square seed, K=6): balanced hyb21 (7,032q = 1,172q/log)
      - `hyp45_unif` ({4,5} 20-square seed, K=6): uniform 2x2 (6,460q)
      - `hyp45_10q_unif` (Coxeter {4,5} 40-square seed, K=10): uniform 2x2 (12,920q)
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys
import tempfile
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v10")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    GeneralCubical2D,
    seed_planar_11,
    seed_rot_torus_422,
    sheaf_to_css_with_seams,
)
from build_v9_quarter_million import (
    apply_sheaf_step_with_maps,
    push_X_logical,
    push_Z_logical,
    verify_and_write_code,
)


def build_qd_hyperbolic_seed(D, gens):
    sub = {0}
    for g in gens:
        sub |= {s ^ g for s in list(sub)}

    def rep(x):
        return min(x ^ s for s in sub)

    vmap, verts = {}, []
    for x in range(1 << D):
        r = rep(x)
        if r not in vmap:
            vmap[r] = len(verts)
            verts.append(r)
    emap, edges = {}, []
    for x in range(1 << D):
        for ax in range(D):
            orb_rep = min(tuple(sorted((y, y ^ (1 << ax)))) for y in [x ^ s for s in sub])
            if orb_rep not in emap:
                emap[orb_rep] = len(edges)
                edges.append((vmap[rep(x)], vmap[rep(x ^ (1 << ax))]))
    fmap, faces = {}, []
    for i in range(D):
        j = (i + 1) % D
        for x in range(1 << D):
            if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                pts = [x, x ^ (1 << i), x ^ (1 << i) ^ (1 << j), x ^ (1 << j)]
                f_orb = min(tuple(sorted(p ^ s for p in pts)) for s in sub)
                if f_orb in fmap:
                    continue
                fmap[f_orb] = 1
                v00, v10, v11, v01 = [vmap[rep(p)] for p in pts]

                def ge(a, b):
                    return emap[min(tuple(sorted((a ^ s, b ^ s))) for s in sub)]

                faces.append((v00, v10, v11, v01, ge(pts[0], pts[1]), ge(pts[3], pts[2]), ge(pts[0], pts[3]), ge(pts[1], pts[2])))
    return GeneralCubical2D(len(verts), edges, ["none"] * len(edges), faces)


def extract_paired_seed_logicals(comp):
    nq, hx, hz, *_ = sheaf_to_css_with_seams(comp)
    with tempfile.TemporaryDirectory() as td:
        rx, cx = f"{td}/x.raw", f"{td}/x.code"
        rz, cz = f"{td}/z.raw", f"{td}/z.code"
        with open(rx, "w") as f:
            f.write(f"{nq} {len(hz)} {len(hx)}\n")
            for r in hz:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in hx:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        with open(rz, "w") as f:
            f.write(f"{nq} {len(hx)} {len(hz)}\n")
            for r in hx:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in hz:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", rx, cx], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([ENGINE, "prep_logicals", rz, cz], check=True, stdout=subprocess.DEVNULL)

        def read_dlogs(path):
            with open(path) as f:
                _, mc, ms, k = map(int, f.readline().split())
                for _ in range(mc + ms):
                    f.readline()
                return [set(map(int, f.readline().split()[1:])) for _ in range(k)]

        lz_raw = read_dlogs(cx)
        lx_raw = read_dlogs(cz)
    K = len(lx_raw)
    mat = []
    for a in range(K):
        row = 0
        for b in range(K):
            if len(lx_raw[a] & lz_raw[b]) & 1:
                row |= (1 << b)
        mat.append([row, 1 << a])
    for col in range(K):
        piv = next(r for r in range(col, K) if (mat[r][0] >> col) & 1)
        mat[col], mat[piv] = mat[piv], mat[col]
        for r in range(K):
            if r != col and ((mat[r][0] >> col) & 1):
                mat[r][0] ^= mat[col][0]
                mat[r][1] ^= mat[col][1]
    lx_paired = []
    for a in range(K):
        comb = mat[a][1]
        s = set()
        for b in range(K):
            if (comb >> b) & 1:
                s ^= lx_raw[b]
        lx_paired.append(s)
    return lx_paired, lz_raw


def build_general_chain(spec):
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE
    name, seed_spec, sched, export_scales, aniso, sectors = spec
    t0 = time.time()
    if seed_spec == "planar":
        comp = seed_planar_11()
        lx, lz = [{0, 1}], [{0}]
    elif seed_spec == "rottor":
        comp = seed_rot_torus_422()
        lx, lz = [{0, 1}, {0, 2}], [{1, 3}, {2, 3}]
    else:
        D, gens = seed_spec
        comp = build_qd_hyperbolic_seed(D, gens)
        lx, lz = extract_paired_seed_logicals(comp)

    cyl = set()
    for s, (L, rb, it) in enumerate(sched, 1):
        pcomp = comp
        comp, cyl, meta = apply_sheaf_step_with_maps(pcomp, L, rb, it, prev_cyl=(cyl if aniso else None))
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]
        if s in export_scales:
            verify_and_write_code(N, HX, HZ, lx, lz, f"{name}_s{s}", sectors)
    return f"{name} (N_top={N}, {time.time()-t0:.1f}s)"


def export_wire_hill_for_checks(N, checks, out_path):
    # Junction checks (touching >=3-way seam qubits or boundary weight <=3) get J = 1.0 (10 tenths);
    # wire checks step up by +0.4 per graph step into passive wires, capped at J_max = 2.2 (22 tenths).
    q_deg = [0] * N
    q2c = [[] for _ in range(N)]
    for ci, r in enumerate(checks):
        for q in r:
            q_deg[q] += 1
            q2c[q].append(ci)
    dist = [-1] * len(checks)
    dq = deque()
    for ci, r in enumerate(checks):
        if any(q_deg[q] >= 3 for q in r) or len(r) <= 3:
            dist[ci] = 0
            dq.append(ci)
    while dq:
        u = dq.popleft()
        for q in checks[u]:
            for v in q2c[q]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
    with open(out_path, "w") as f:
        for d in dist:
            f.write(f"{min(22, 10 + 4 * max(0, d))}\n")
    return max(dist)


def build_z3_and_wire_hill(spec):
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE
    name, sched = spec
    t0 = time.time()
    comp = seed_planar_11()
    lx, lz = [{0, 1}], [{0}]
    for (L, rb, it) in sched:
        pcomp = comp
        comp, _, meta = apply_sheaf_step_with_maps(pcomp, L, rb, it, prev_cyl=None)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]
    verify_and_write_code(N, HX, HZ, lx, lz, name, ["X_err", "Z_err"])
    md_x = export_wire_hill_for_checks(N, HZ, os.path.join(DATA_DIR, f"{name}_X_err_wire_hill.j"))
    md_z = export_wire_hill_for_checks(N, HX, os.path.join(DATA_DIR, f"{name}_Z_err_wire_hill.j"))
    return f"{name} (N={N}, max_wire_dist_X={md_x}, max_wire_dist_Z={md_z}, {time.time()-t0:.1f}s)"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    chain_specs = [
        # 1A. Minimal 3D-Volume-Compatible 3x3 Single-Cross (`dim = 2.893 < 3`)
        ("sq_rx_L3r1",      "planar", [(3, [1], "X")] * 3,                                         [1, 2, 3], False, ["X_err"]),
        ("sq_L3r1",         "planar", [(3, [1], "X"), (3, [1], "Z"), (3, [1], "X")],               [1, 2, 3], False, ["X_err", "Z_err"]),
        # 2A. Sector-Balanced Compact Cycle 1 (586q / 1,172q) -> Cycle 2 (129,148q)
        ("sq_hyb21",        "planar", [(3, [1, 2], "X"), (2, [1], "Z"), (3, [1, 2], "X"), (2, [1], "Z")], [2, 4], True, ["X_err", "Z_err"]),
        ("sq_hyb21_iso",    "planar", [(3, [1, 2], "X"), (2, [1], "Z")],                           [2],       False, ["X_err", "Z_err"]),
        ("rottor_hyb21",    "rottor", [(3, [1, 2], "X"), (2, [1], "Z")],                           [2],       True,  ["X_err", "Z_err"]),
        ("rottor_hyb21_iso","rottor", [(3, [1, 2], "X"), (2, [1], "Z")],                           [2],       False, ["X_err", "Z_err"]),
        # 2B. High-Rate Hyperbolic Square-Tiling Welded Memories (K = 6 and K = 10)
        ("hyp46_unif",      (6, [7, 13, 19]), [(2, [1], "X"), (2, [1], "Z")],                      [2],       False, ["X_err", "Z_err"]),
        ("hyp46_hyb21",     (6, [7, 13, 19]), [(3, [1, 2], "X"), (2, [1], "Z")],                   [2],       True,  ["X_err", "Z_err"]),
        ("hyp45_unif",      (5, [31]),        [(2, [1], "X"), (2, [1], "Z")],                      [2],       False, ["X_err", "Z_err"]),
        ("hyp45_10q_unif",  (5, []),          [(2, [1], "X"), (2, [1], "Z")],                      [2],       False, ["X_err", "Z_err"]),
    ]
    z3_specs = [
        ("sq_z3_3_5_r1", [(3, [2], "X"),    (5, [4], "Z")]),
        ("sq_z3_4_6_r2", [(4, [2, 3], "X"), (6, [4, 5], "Z")]),
        ("sq_bdl_8_8_r2",[(8, [4, 5], "X"), (8, [4, 5], "Z")]),
    ]

    with ProcessPoolExecutor(max_workers=4) as ex:
        for msg in ex.map(build_general_chain, chain_specs):
            print("Built chain:", msg, flush=True)
        for msg in ex.map(build_z3_and_wire_hill, z3_specs):
            print("Built Z^3:", msg, flush=True)

    fast = []
    heavy = []

    # 1A. 3D-Compatible (L=3, r=1) Single-Sector & Full Quantum Scaling
    for s, tr in [(1, 200), (2, 100), (3, 40)]:
        for b in ("1.40", "1.60", "1.75", "1.85", "1.95", "2.05", "2.15", "2.35", "2.55", "2.75", "2.95", "3.15", "3.35", "3.55"):
            tgt = heavy if s == 3 and float(b) >= 1.95 else fast
            tgt.append(f"kmc__sq_rx_L3r1_s{s}_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_rx_L3r1_s{s}_X_err.code|uniform|uf|{b}|{tr}|500000000")
    for s, tr in [(1, 160), (2, 80), (3, 32)]:
        for b in ("4.50", "5.00", "5.50", "6.00", "6.25", "6.50", "7.00", "7.50"):
            if b in ("7.00", "7.50") and s == 2:
                continue
            tgt = heavy if s == 3 else fast
            for sec in ("X_err", "Z_err"):
                if s in (1, 3) and sec == "X_err" and float(b) >= 6.00:
                    continue  # At odd s (after X step), Z_err is the limiting sector!
                if s == 2 and sec == "Z_err" and float(b) >= 6.00:
                    continue  # At s=2 (after Z step), X_err is the limiting sector!
                tgt.append(f"kmc__sq_L3r1_s{s}_{sec}__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_L3r1_s{s}_{sec}.code|uniform|uf|{b}|{tr}|500000000")

    # 1B. Collision-Free Z^3 Cubic Grids: Uniform vs Wire-Energy Hills (Defeating Wire Entropy!)
    for zn, tr in [("sq_z3_3_5_r1", 80), ("sq_z3_4_6_r2", 48), ("sq_bdl_8_8_r2", 24)]:
        for b in ("5.00", "5.50", "6.00", "6.50", "7.00"):
            tgt = heavy if zn == "sq_bdl_8_8_r2" and float(b) >= 6.00 else fast
            jp = f"{DATA_DIR}/{zn}_X_err_wire_hill.j"
            fast.append(f"kmc__{zn}_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/{zn}_X_err.code|uniform|uf|{b}|{tr}|400000000")
            tgt.append(f"kmc__{zn}_X_err__wire_hill__uf__b{b}|kmc|{DATA_DIR}/{zn}_X_err.code|{jp}|uf|{b}|{tr}|450000000")
        # Also verify Z_err >> X_err at b=5.50, 6.00:
        for b in ("5.50", "6.00"):
            jpz = f"{DATA_DIR}/{zn}_Z_err_wire_hill.j"
            tgt = heavy if zn == "sq_bdl_8_8_r2" else fast
            tgt.append(f"kmc__{zn}_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/{zn}_Z_err.code|uniform|uf|{b}|24|400000000")
            tgt.append(f"kmc__{zn}_Z_err__wire_hill__uf__b{b}|kmc|{DATA_DIR}/{zn}_Z_err.code|{jpz}|uf|{b}|24|400000000")

    # 2A. Sub-2,500-Qubit Sector-Balanced Welded Codes (Cycle 1: 586q, 790q, 1,172q, 1,580q -> Cycle 2: 129,148q!)
    for tag, tr in [("sq_hyb21_s2", 80), ("sq_hyb21_iso_s2", 80), ("rottor_hyb21_s2", 64), ("rottor_hyb21_iso_s2", 64)]:
        for sec in ("X_err", "Z_err"):
            for b in ("5.00", "5.50", "6.00", "6.50", "7.00"):
                tgt = heavy if "rottor" in tag and sec == "Z_err" and float(b) >= 6.50 else fast
                tgt.append(f"kmc__{tag}_{sec}__uniform__uf__b{b}|kmc|{DATA_DIR}/{tag}_{sec}.code|uniform|uf|{b}|{tr}|350000000")
    for sec in ("X_err", "Z_err"):
        for b in ("4.50", "5.00", "5.25", "5.50", "5.75", "6.00", "6.25", "6.40"):
            if sec == "Z_err" and b in ("6.25", "6.40"):
                continue
            heavy.append(f"kmc__sq_hyb21_s4_{sec}__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_hyb21_s4_{sec}.code|uniform|uf|{b}|16|850000000")

    # 2B. High-Rate Hyperbolic Square-Tiling Welded Memories (K = 6 and K = 10 Logical Qubits!)
    for htag, tr in [("hyp46_unif_s2", 48), ("hyp46_hyb21_s2", 36), ("hyp45_unif_s2", 36), ("hyp45_10q_unif_s2", 28)]:
        for sec in ("X_err", "Z_err"):
            for b in ("5.00", "5.50", "6.00", "6.50"):
                if sec == "Z_err" and b == "6.50":
                    continue  # Z_err is already > 10^6 at b=5.50..6.00!
                is_hvy = (sec == "Z_err" and b == "6.00") or (htag == "hyp46_hyb21_s2" and float(b) >= 6.00) or htag == "hyp45_10q_unif_s2"
                tgt = heavy if is_hvy else fast
                tgt.append(f"kmc__{htag}_{sec}__uniform__uf__b{b}|kmc|{DATA_DIR}/{htag}_{sec}.code|uniform|uf|{b}|{tr}|450000000")

    n_fast_sh = 14
    n_heavy_sh = len(heavy)
    man_p = os.path.join(WORK_DIR, "manifest_v12.txt")
    with open(man_p, "w") as f:
        for i, ln in enumerate(fast):
            f.write(f"{i % n_fast_sh}|{ln}\n")
        for i, ln in enumerate(heavy):
            f.write(f"{n_fast_sh + i}|{ln}\n")
    print(f"Wrote {len(fast)} fast (0..{n_fast_sh-1}) + {len(heavy)} heavy ({n_fast_sh}..{n_fast_sh+n_heavy_sh-1}) -> total_shards={n_fast_sh+n_heavy_sh}", flush=True)


if __name__ == "__main__":
    main()
