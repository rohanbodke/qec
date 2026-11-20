#!/usr/bin/env python3
"""
build_v11_rg_and_cooling.py
Builds and verifies all codes, RGV2 quantum zoom-out hierarchies (.rg), and O(N) hierarchical RG
readout tables (.rgdec) for the V11 campaign:
  1. 100% Unified Uniform 2x2 3-Level Quantum Crossover (`sq_iso_L2r1` s=1,3,5: 19 -> 2,911 -> 423,559q)
  2. 5-Scale Uniform 2x2 Single-Sector Crossover (`sq_rx_iso` s=1..5: 19 -> 321,383q) + RG vs UF Head-to-Head
  3. Alternating (X, Z) Quantum Syndrome Zoom-Out Staircase (`rottor_iso_L2r1_s4.rg` 94,774q & `sq_iso_L2r1_s5.rg` 423,559q)
  4. Multi-Scale Cooling-Time Scaling Law t_cool(N) across 4 scales on 3 families (up to 94,774 qubits)
"""

from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v10")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import seed_planar_11, seed_rot_torus_422, sheaf_to_css_with_seams
from build_v9_quarter_million import (
    apply_sheaf_step_with_maps,
    edge_qstarts,
    push_X_logical,
    push_Z_logical,
    verify_and_write_code,
)


def compute_sheet_lut(HZ, f_start, e2q, he, ve):
    cand_eids = [he[0][1], he[1][1], ve[1][0], ve[1][1]]
    cand_qs, q_mask = [], []
    q2f = {}
    for k in range(4):
        for q in HZ[f_start + k]:
            q2f.setdefault(q, []).append(k)
    for eid in cand_eids:
        for q in e2q[eid]:
            m = 0
            for k in q2f.get(q, []):
                m |= (1 << k)
            if m:
                cand_qs.append(q)
                q_mask.append(m)
                break
    lut = {}
    for sub in range(1 << len(cand_qs)):
        m, qs = 0, []
        for i in range(len(cand_qs)):
            if (sub >> i) & 1:
                m ^= q_mask[i]
                qs.append(cand_qs[i])
        if (bin(m).count("1") % 2 == 0) and (m not in lut or len(qs) < len(lut[m])):
            lut[m] = qs
    assert len(lut) == 8, f"Expected 8 even masks, got {len(lut)}"
    return lut


def compute_z_cell_basis(HZ, q2f_global, pf, n_xf):
    cell_f = (
        [4 * pf + k for k in range(4)]
        + [n_xf + 4 * pf + k for k in range(4)]
        + [2 * n_xf + 4 * pf + k for k in range(4)]
    )
    fset = set(cell_f)
    fmap = {f: i for i, f in enumerate(cell_f)}
    int_qs = sorted({q for f in cell_f for q in HZ[f] if all(ff in fset for ff in q2f_global[q])})
    piv_v, piv_q = {}, {}
    for q in int_qs:
        v = 0
        for ff in q2f_global[q]:
            v |= (1 << fmap[ff])
        cur_qs = {q}
        while v:
            b = (v & -v).bit_length() - 1
            if b in piv_v:
                v ^= piv_v[b]
                cur_qs ^= piv_q[b]
            else:
                piv_v[b] = v
                piv_q[b] = cur_qs
                break
    assert len(piv_v) >= 11
    basis = []
    for tb in range(11):
        v = (1 << tb) | (1 << 11)
        sol = set()
        while v:
            b = (v & -v).bit_length() - 1
            v ^= piv_v[b]
            sol ^= piv_q[b]
        basis.append(sorted(sol))
    return basis


def write_rgv2_file(rg_p, s_top, rg_lvs):
    with open(rg_p, "w") as f:
        f.write(f"RGV2 {s_top}\n")
        for s_idx in range(s_top + 1):
            nq, hz, nxf, pfs, bfs, stp = rg_lvs[s_idx]
            f.write(f"{nq} {len(hz)} {nxf} {len(pfs)} {stp}\n")
            for r in hz:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            if s_idx >= 1:
                for fs, cnt in pfs:
                    f.write(f"{fs} {cnt}\n")
                if stp == "Z":
                    for bs, bcnt in bfs:
                        f.write(f"{bs} {bcnt}\n")


def write_rgdec_file(dp, s_top, dec_lvs):
    with open(dp, "w") as f:
        f.write(f"{s_top}\n")
        for s_idx in range(s_top + 1):
            nq, hz, nxf, npf, stp, L0, L1, cdata = dec_lvs[s_idx]
            f.write(f"{nq} {len(hz)} {nxf} {npf} {stp}\n")
            for r in hz:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            if s_idx >= 1:
                for q0, q1 in zip(L0, L1):
                    f.write(f"{len(q0)} " + " ".join(map(str, q0)) + f" {len(q1)} " + " ".join(map(str, q1)) + "\n")
                if stp == "X":
                    for f0, f1, lut0, lut1 in cdata:
                        f.write(f"{f0} {f1}\n")
                        for m in sorted(lut0.keys()):
                            qs = lut0[m]
                            f.write(f"{m} {len(qs)} " + " ".join(map(str, qs)) + "\n")
                        for m in sorted(lut1.keys()):
                            qs = lut1[m]
                            f.write(f"{m} {len(qs)} " + " ".join(map(str, qs)) + "\n")
                else:
                    for f0, f1, bf, zb in cdata:
                        f.write(f"{f0} {f1} {bf}\n")
                        for qs in zb:
                            f.write(f"{len(qs)} " + " ".join(map(str, qs)) + "\n")


def build_family(spec):
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE
    name, kind, steps, code_scales, sectors, rgv2_scales, rgdec_scales = spec
    t0 = time.time()
    comp = seed_planar_11() if kind == "planar" else seed_rot_torus_422()
    lx, lz = ([{0, 1}], [{0}]) if kind == "planar" else ([{0, 1}, {0, 2}], [{1, 3}, {2, 3}])
    N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)

    rg_lvs = [(N, HZ, 0, [], [], "S")]
    dec_lvs = [(N, HZ, 0, 0, "S", None, None, None)]

    for s, it in enumerate(steps, 1):
        pcomp = comp
        p_qs, pN = edge_qstarts(pcomp)
        comp, _, meta = apply_sheaf_step_with_maps(pcomp, 2, [1], it, None)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]

        if s in code_scales:
            verify_and_write_code(N, HX, HZ, lx, lz, f"{name}_s{s}", sectors)

        n_xe, n_xf = meta["n_xe"], meta["n_xf"]
        n_pf = len(pcomp.faces)
        pf_spans = [(4 * pf, 4) for pf in range(n_pf)]
        bf_spans = [(2 * n_xf + 4 * pf, 4) for pf in range(n_pf)]
        rg_lvs.append((N, HZ, n_xf, pf_spans, bf_spans, it))

        if rgdec_scales:
            c_qs, _ = edge_qstarts(comp)
            e2q = []
            for eid, inc in enumerate(comp.edge_faces):
                d = len(inc)
                col = comp.edge_colors[eid]
                nq = 0 if d == 0 else (1 if (d == 2 or col == "blue") else d - 1)
                e2q.append([c_qs[eid] + j for j in range(nq)])
            L0 = [[] for _ in range(pN)]
            L1 = [[] for _ in range(pN)]
            for peid, inc in enumerate(pcomp.edge_faces):
                d = len(inc)
                col = pcomp.edge_colors[peid]
                nq = 0 if d == 0 else (1 if (d == 2 or col == "blue") else d - 1)
                s0 = meta["edge_sub_eids"][peid][0]
                for j in range(nq):
                    pq = p_qs[peid] + j
                    L0[pq] = [e2q[s0][j]]
                    L1[pq] = [e2q[s0 + n_xe][j]]
            if it == "X":
                cell_data = []
                for pf, (Lx, Ly, he, ve, fs) in enumerate(meta["face_grids"]):
                    he1 = [[he[x][y] + n_xe for y in range(Ly + 1)] for x in range(Lx)]
                    ve1 = [[ve[x][y] + n_xe for y in range(Ly)] for x in range(Lx + 1)]
                    lut0 = compute_sheet_lut(HZ, fs, e2q, he, ve)
                    lut1 = compute_sheet_lut(HZ, n_xf + fs, e2q, he1, ve1)
                    cell_data.append((fs, n_xf + fs, lut0, lut1))
                dec_lvs.append((N, HZ, n_xf, n_pf, "X", L0, L1, cell_data))
            else:
                q2f = [[] for _ in range(N)]
                for f, row in enumerate(HZ):
                    for q in row:
                        q2f[q].append(f)
                cell_data = []
                for pf in range(n_pf):
                    zb = compute_z_cell_basis(HZ, q2f, pf, n_xf)
                    cell_data.append((4 * pf, n_xf + 4 * pf, 2 * n_xf + 4 * pf, zb))
                dec_lvs.append((N, HZ, n_xf, n_pf, "Z", L0, L1, cell_data))

        if s in rgv2_scales:
            rg_p = os.path.join(DATA_DIR, f"{name}_s{s}_v2.rg")
            if not os.path.exists(rg_p):
                write_rgv2_file(rg_p, s, rg_lvs)
                print(f"  Wrote RGV2 {rg_p}", flush=True)

        if s in rgdec_scales:
            dp = os.path.join(DATA_DIR, f"{name}_s{s}.rgdec")
            if not os.path.exists(dp):
                write_rgdec_file(dp, s, dec_lvs)
                print(f"  Wrote RGDEC {dp}", flush=True)

    return f"{name} ({time.time()-t0:.1f}s)"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Ensure build_v9_quarter_million uses kmc_engine_v10:
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE

    specs = [
        ("sq_rx_iso",       "planar", ["X", "X", "X", "X", "X"], [1, 2, 3, 4, 5], ["X_err"],          [],     [1, 2, 3, 4, 5]),
        ("rottor_iso_L2r1", "torus",  ["X", "Z", "X", "Z"],      [1, 2, 3, 4],    ["X_err", "Z_err"], [4],    [2, 4]),
        ("sq_iso_L2r1",     "planar", ["X", "Z", "X", "Z", "X"], [1, 2, 3, 4, 5], ["X_err", "Z_err"], [5],    [1, 3, 5]),
    ]
    with ProcessPoolExecutor(max_workers=3) as ex:
        for msg in ex.map(build_family, specs):
            print("Finished family:", msg, flush=True)

    fast = []
    heavy = []

    # =========================================================================
    # 1A. Unified Uniform 2x2 3-Level Quantum Crossover (`sq_iso_L2r1` s=1,3,5)
    # =========================================================================
    for s, tr in [(1, 160), (3, 64)]:
        for b in ("5.85", "6.10", "6.35"):
            fast.append(f"kmc__sq_iso_L2r1_s{s}_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s{s}_Z_err.code|uniform|uf|{b}|{tr}|200000000")
    for b in ("5.50", "5.75", "5.85", "6.00", "6.10", "6.25", "6.35", "6.50"):
        heavy.append(f"kmc__sq_iso_L2r1_s5_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s5_Z_err.code|uniform|uf|{b}|20|350000000")

    # Strong X_err sector on 3-level quantum hierarchy (UF + RG_DEC up to 423,559q!):
    for s, tr in [(1, 160), (3, 64)]:
        for b in ("3.50", "4.00", "4.50", "4.75"):
            fast.append(f"kmc__sq_iso_L2r1_s{s}_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s{s}_X_err.code|uniform|uf|{b}|{tr}|200000000")
            fast.append(f"kmc__sq_iso_L2r1_s{s}_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/sq_iso_L2r1_s{s}_X_err.code|{DATA_DIR}/sq_iso_L2r1_s{s}.rgdec|{b}|{tr}|200000000")
    for b in ("3.50", "4.00", "4.50", "4.75"):
        heavy.append(f"kmc__sq_iso_L2r1_s5_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s5_X_err.code|uniform|uf|{b}|16|300000000")
        heavy.append(f"kmc__sq_iso_L2r1_s5_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/sq_iso_L2r1_s5_X_err.code|{DATA_DIR}/sq_iso_L2r1_s5.rgdec|{b}|16|300000000")

    # =========================================================================
    # 1B. 5-Scale Uniform 2x2 Crossover (`sq_rx_iso` s=1..5) & RG vs UF Head-to-Head
    # =========================================================================
    for b in ("1.75", "1.90", "2.00", "2.05", "2.10", "2.15", "2.20", "2.25"):
        heavy.append(f"kmc__sq_rx_iso_s5_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_rx_iso_s5_X_err.code|uniform|uf|{b}|24|350000000")
        heavy.append(f"kmc__sq_rx_iso_s5_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/sq_rx_iso_s5_X_err.code|{DATA_DIR}/sq_rx_iso_s5.rgdec|{b}|24|350000000")

    for s, tr in [(1, 200), (2, 120), (3, 80), (4, 36)]:
        for b in ("1.75", "1.90", "2.00", "2.05", "2.10", "2.15", "2.20", "2.25"):
            tgt = fast if s <= 3 else heavy
            tgt.append(f"kmc__sq_rx_iso_s{s}_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/sq_rx_iso_s{s}_X_err.code|{DATA_DIR}/sq_rx_iso_s{s}.rgdec|{b}|{tr}|250000000")
            if b in ("2.05", "2.20"):
                tgt.append(f"kmc__sq_rx_iso_s{s}_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_rx_iso_s{s}_X_err.code|uniform|uf|{b}|{tr}|250000000")

    # RG_DEC vs UF on Boundaryless Quantum Torus (`rottor_iso_L2r1` s=2: 646q, s=4: 94,774q):
    for b in ("5.50", "6.00", "6.25", "6.50", "6.65"):
        fast.append(f"kmc__rottor_iso_L2r1_s2_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/rottor_iso_L2r1_s2_X_err.code|{DATA_DIR}/rottor_iso_L2r1_s2.rgdec|{b}|120|80000000")
        heavy.append(f"kmc__rottor_iso_L2r1_s4_X_err__uniform__rg_dec__b{b}|rg_kmc|{DATA_DIR}/rottor_iso_L2r1_s4_X_err.code|{DATA_DIR}/rottor_iso_L2r1_s4.rgdec|{b}|24|300000000")

    # =========================================================================
    # 1C. Live Alternating (X, Z) Quantum Syndrome Zoom-Out Staircase (`RGV2`)
    # =========================================================================
    for b in ("3.50", "4.50", "5.50", "6.00", "6.25", "6.50", "7.00", "7.50"):
        fast.append(f"rgflow__rottor_iso_L2r1_s4__b{b}|rg_flow|{DATA_DIR}/rottor_iso_L2r1_s4_v2.rg|{b}|20.0|48")
        heavy.append(f"rgflow__sq_iso_L2r1_s5__b{b}|rg_flow|{DATA_DIR}/sq_iso_L2r1_s5_v2.rg|{b}|20.0|32")

    # =========================================================================
    # 2. Multi-Scale Cooling-Time Scaling Law t_cool(N) Across 4 Scales
    # =========================================================================
    t_grid = ("25", "100", "400", "1600", "6400", "25600", "102400")
    for proto in ("anneal", "quench"):
        # (a) Boundaryless Quantum Torus: s=1 (38q), s=2 (646q), s=3 (8,174q), s=4 (94,774q)
        for s, tr in [(1, 120), (2, 80), (3, 48), (4, 24)]:
            for tt in t_grid:
                if s == 1 and int(tt) > 6400:
                    continue
                tgt = heavy if (s == 4 or (s == 3 and int(tt) >= 25600)) else fast
                tgt.append(f"cool__rottor_iso_L2r1_s{s}_X_err__{proto}__hot__t{tt}|cool|{DATA_DIR}/rottor_iso_L2r1_s{s}_X_err.code|uniform|uf|{proto}|hot|1.5|7.5|{tt}|{tr}")

        # (b) Planar Full Quantum Hierarchy: s=1 (19q), s=2 (323q), s=3 (2,911q), s=4 (47,387q)
        for s, tr in [(1, 120), (2, 80), (3, 48), (4, 24)]:
            for tt in t_grid:
                if s == 1 and int(tt) > 6400:
                    continue
                tgt = heavy if (s == 4 and int(tt) >= 1600) else fast
                tgt.append(f"cool__sq_iso_L2r1_s{s}_X_err__{proto}__hot__t{tt}|cool|{DATA_DIR}/sq_iso_L2r1_s{s}_X_err.code|uniform|uf|{proto}|hot|1.5|8.0|{tt}|{tr}")

        # (c) Planar Pure X-Welded Hierarchy: s=1 (19q), s=2 (203q), s=3 (2,311q), s=4 (27,107q)
        for s, tr in [(1, 120), (2, 80), (3, 48), (4, 24)]:
            for tt in t_grid:
                if s == 1 and int(tt) > 6400:
                    continue
                tgt = heavy if (s == 4 and int(tt) >= 1600) else fast
                tgt.append(f"cool__sq_rx_iso_s{s}_X_err__{proto}__hot__t{tt}|cool|{DATA_DIR}/sq_rx_iso_s{s}_X_err.code|uniform|uf|{proto}|hot|1.0|7.5|{tt}|{tr}")

    # Pack fast tasks into 10 shards (0..9), and pack heavy tasks ~2 per shard into 46 shards (10..55):
    n_fast_sh = 10
    n_heavy_sh = 46
    man_p = os.path.join(WORK_DIR, "manifest_v11.txt")
    with open(man_p, "w") as f:
        for i, ln in enumerate(fast):
            f.write(f"{i % n_fast_sh}|{ln}\n")
        for i, ln in enumerate(heavy):
            f.write(f"{n_fast_sh + (i % n_heavy_sh)}|{ln}\n")
    print(
        f"Wrote {len(fast)} fast (0..{n_fast_sh-1}) + {len(heavy)} heavy ({n_fast_sh}..{n_fast_sh+n_heavy_sh-1}) "
        f"-> total_shards={n_fast_sh + n_heavy_sh}",
        flush=True,
    )


if __name__ == "__main__":
    main()
