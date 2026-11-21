#!/usr/bin/env python3
"""
build_v14_intervention.py
Builds and verifies all check-scale intervention weight files (.j), seed codes,
and the SLURM manifest (`manifest_v14.txt`) for the V14 campaign:

  Suite 1: Selected-Scale Coupling Intervention H(lambda; U) on Cycle 2
           (`sq_iso_L2r1_s4`, 47,387q K=1; `rottor_iso_L2r1_s4`, 94,774q K=2)
           and Cycle 1 (`sq_iso_L2r1_s2`, 323q; `rottor_iso_L2r1_s2`, 646q):
           - `macro_weld`: coarse Cycle-1 weld bridges/seams (s in {1,2}),
           - `sub_weld`: equal-count, exact-weight-histogram-matched fine Cycle-2
             sub-cell welds (s in {3,4}, spatially stratified across the complex),
           - `all_c2_weld`: full Cycle-2 sub-cell weld set (s in {3,4}).
           Evaluated at lambda in {0.2, 0.4, 0.6, 0.8} across beta in {5.50, 6.00, 6.25, 6.50}
           in BOTH X_err and Z_err sectors!
  Suite 2: Coarse-Scale Excursion & Escape Diagnostic (`excursion_kmc`) across
           uniform (lambda=1.0), `macro_weld`, and `sub_weld`.
  Suite 3: Paired Readout Cadence Sensitivity Audit (`paired_cadence`) comparing
           eta in {0.0 (every flip) / 0.02, 0.05, 0.20, 0.50} on identical trajectories.
  Suite 4: Coarse-Syndrome Flow Burn-In Sweep & Matched-Weight i.i.d. Reference (`rg_flow_v14`).
  Suite 5: 3x3 Single-Cross beta=7.50 Two-Sector Completion + Exact Matched-beta_eff Wire Test.
"""

from collections import Counter, defaultdict
import json
import math
import os
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v14")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    seed_planar_11,
    seed_rot_torus_422,
    sheaf_to_css_with_seams,
)
from build_v9_quarter_million import (
    apply_sheaf_step_with_maps,
    verify_and_write_code,
)
import build_v9_quarter_million as b9

b9.ENGINE = ENGINE


def read_code_checks(path):
    with open(path) as f:
        header = f.readline().split()
        n, mc, ms, k = map(int, header)
        checks = []
        for _ in range(mc):
            parts = list(map(int, f.readline().split()))
            checks.append(sorted(parts[1:]))
    return n, mc, ms, k, checks


def build_css_with_vertex_tags(comp):
    """Exact replica of sheaf_to_css_with_seams returning (N, HX_with_v, HZ, q_to_edge)."""
    edge_qstart, qubits, q_to_edge = [], [], []
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        edge_qstart.append(len(qubits))
        if d == 0:
            continue
        elif d == 2 or col == "blue":
            qubits.append((e_idx, {f: 1 for f in inc_faces}))
            q_to_edge.append(e_idx)
        elif col == "red":
            for j in range(d - 1):
                qubits.append((e_idx, {inc_faces[j]: 1, inc_faces[j + 1]: 1}))
                q_to_edge.append(e_idx)

    hz = [[] for _ in range(len(comp.faces))]
    for qi, (_, vec) in enumerate(qubits):
        for fi in vec:
            hz[fi].append(qi)

    hx_v = []
    for v in range(comp.num_vertices):
        v_faces = sorted(set(comp.vert_faces[v]))
        if not v_faces:
            continue
        m = len(v_faces)
        f_to_loc = {f: i for i, f in enumerate(v_faces)}
        constraints = []
        for e_idx in comp.vert_edges[v]:
            inc = comp.edge_faces[e_idx]
            d = len(inc)
            col = comp.edge_colors[e_idx]
            if d == 0:
                continue
            elif d == 2 or col == "blue":
                for j in range(1, d):
                    constraints.append((1 << f_to_loc[inc[0]]) ^ (1 << f_to_loc[inc[j]]))
            elif col == "red":
                row = 0
                for f in inc:
                    row ^= (1 << f_to_loc[f])
                constraints.append(row)
        pivots = {}
        for r in constraints:
            x = r
            while x:
                lsb = (x & -x).bit_length() - 1
                if lsb in pivots:
                    x ^= pivots[lsb]
                else:
                    pivots[lsb] = x
                    break
        free_vars = [i for i in range(m) if i not in pivots]
        if not free_vars:
            continue
        for p in sorted(pivots.keys(), reverse=True):
            for p2 in pivots:
                if p2 != p and ((pivots[p2] >> p) & 1):
                    pivots[p2] ^= pivots[p]
        raw_basis = []
        for fv in free_vars:
            alpha = 1 << fv
            for p, prow in pivots.items():
                if (prow >> fv) & 1:
                    alpha |= 1 << p
            raw_basis.append(alpha)

        def alpha_to_qsupp(alpha):
            xs = []
            for e_idx in comp.vert_edges[v]:
                inc = comp.edge_faces[e_idx]
                d = len(inc)
                col = comp.edge_colors[e_idx]
                q0 = edge_qstart[e_idx]
                if d == 0:
                    continue
                elif d == 2 or col == "blue":
                    if (alpha >> f_to_loc[inc[0]]) & 1:
                        xs.append(q0)
                elif col == "red":
                    pref = 0
                    for j in range(d - 1):
                        pref ^= (alpha >> f_to_loc[inc[j]]) & 1
                        if pref:
                            xs.append(q0 + j)
            return sorted(set(xs))

        dim_fv = len(raw_basis)
        if 1 < dim_fv <= 7:
            cands = []
            for mask in range(1, 1 << dim_fv):
                alpha = 0
                for b in range(dim_fv):
                    if (mask >> b) & 1:
                        alpha ^= raw_basis[b]
                supp = alpha_to_qsupp(alpha)
                if supp:
                    cands.append((len(supp), mask, supp))
            cands.sort(key=lambda t: t[0])
            basis_masks = {}
            chosen = []
            for _, mask, supp in cands:
                x = mask
                while x:
                    lsb = (x & -x).bit_length() - 1
                    if lsb in basis_masks:
                        x ^= basis_masks[lsb]
                    else:
                        basis_masks[lsb] = x
                        chosen.append(supp)
                        break
                if len(chosen) == dim_fv:
                    break
            for s0 in chosen:
                hx_v.append((v, s0))
        else:
            for alpha in raw_basis:
                supp = alpha_to_qsupp(alpha)
                if supp:
                    hx_v.append((v, supp))

    return len(qubits), hx_v, [sorted(r) for r in hz], q_to_edge


def select_matched_strided(pool_indices, checks, quota_by_wt):
    """Deterministically selects spatially strided check indices matching quota_by_wt."""
    by_wt = defaultdict(list)
    for idx in sorted(pool_indices):
        by_wt[len(checks[idx])].append(idx)
    selected = set()
    for wt, q in sorted(quota_by_wt.items()):
        if q == 0:
            continue
        arr = by_wt[wt]
        assert len(arr) >= q, f"Insufficient pool for wt={wt}: {len(arr)} < {q}"
        if len(arr) == q:
            selected.update(arr)
        else:
            # Deterministic Bresenham-style spatial stride across [0, len(arr)):
            picked = [arr[(i * len(arr)) // q] for i in range(q)]
            assert len(set(picked)) == q
            selected.update(picked)
    return selected


def write_j_subset(m_total, subset_indices, j_tenths_val, out_path):
    with open(out_path, "w") as f:
        for i in range(m_total):
            f.write(f"{j_tenths_val if i in subset_indices else 10}\n")


def build_cycle2_interventions(seed_fn, name):
    comp = seed_fn()
    f_orig = [0] * len(comp.faces)
    v_orig = [0] * comp.num_vertices
    e_weld = [0] * len(comp.edges)

    for s, it in [(1, "X"), (2, "Z"), (3, "X"), (4, "Z")]:
        pcomp = comp
        comp, _, meta = apply_sheaf_step_with_maps(pcomp, 2, [1], it, None)
        n_xe, n_xf = meta["n_xe"], meta["n_xf"]
        nf = []
        for c in (0, 1):
            for pf in range(len(pcomp.faces)):
                nf.extend([f_orig[pf]] * 4)
        while len(nf) < len(comp.faces):
            nf.append(s)
        f_orig = nf

        xp_v = comp.num_vertices // 2
        xv = [v_orig[pv] for pv in range(pcomp.num_vertices)] + [s] * (xp_v - pcomp.num_vertices)
        v_orig = xv + xv

        xp_ew = [0] * n_xe
        for pe, subs in enumerate(meta["edge_sub_eids"]):
            w0 = e_weld[pe] if len(pcomp.edge_faces[pe]) >= 3 else 0
            for eid in subs:
                xp_ew[eid] = w0
        new_ew = xp_ew + xp_ew + [0] * (len(comp.edges) - 2 * n_xe)
        for bf in range(2 * n_xf, len(comp.faces)):
            _, _, _, _, eb, et, el, er = comp.faces[bf]
            for ee in (eb, et, el, er):
                if new_ew[ee] == 0:
                    new_ew[ee] = s
        for eid in range(len(comp.edges)):
            if len(comp.edge_faces[eid]) < 3:
                new_ew[eid] = 0
        e_weld = new_ew

    n_q, hx_v, hz, q_to_edge = build_css_with_vertex_tags(comp)
    q_eweld = [e_weld[eid] for eid in q_to_edge]

    # Hard row-by-row verification against the actual .code files on SubMIT!
    nx_f, mcx_f, _, _, hz_disk = read_code_checks(os.path.join(DATA_DIR, f"{name}_X_err.code"))
    nz_f, mcz_f, _, _, hx_disk = read_code_checks(os.path.join(DATA_DIR, f"{name}_Z_err.code"))
    assert nx_f == n_q and nz_f == n_q
    assert hz == hz_disk, f"HZ row mismatch on {name}!"
    assert [supp for _, supp in hx_v] == hx_disk, f"HX row mismatch on {name}!"

    # 1. Z-checks (X_err sector):
    #    - pool_macro_Z: f_orig in {1, 2} (Cycle-1 bridge faces)
    #    - pool_sub_Z:   f_orig in {3, 4} (Cycle-2 bridge faces)
    pool_macro_z = [i for i, o in enumerate(f_orig) if o in (1, 2)]
    pool_sub_z = [i for i, o in enumerate(f_orig) if o in (3, 4)]
    c_macro_z = Counter(len(hz[i]) for i in pool_macro_z)
    c_sub_z = Counter(len(hz[i]) for i in pool_sub_z)
    quota_z = {w: min(c_macro_z[w], c_sub_z[w]) for w in sorted(set(c_macro_z) | set(c_sub_z))}
    u_macro_z = select_matched_strided(pool_macro_z, hz, quota_z)
    u_sub_z = select_matched_strided(pool_sub_z, hz, quota_z)
    u_all_c2_z = set(pool_sub_z)
    assert u_macro_z.isdisjoint(u_sub_z)

    # 2. X-checks (Z_err sector):
    #    - pool_macro_X: oldest weld seam touched in {1, 2} (Cycle-1 weld seam checks)
    #    - pool_sub_X:   oldest weld seam touched in {3, 4} AND vertex touches zero Cycle-1 bridge faces
    hx_checks = [supp for _, supp in hx_v]
    pool_macro_x, pool_sub_x, pool_all_c2_x = [], [], []
    for i, (v, supp) in enumerate(hx_v):
        ws = [q_eweld[q] for q in supp if q_eweld[q] > 0]
        oldest = min(ws) if ws else 0
        touches_c1_bf = any(f_orig[f] in (1, 2) for f in comp.vert_faces[v])
        if oldest in (1, 2):
            pool_macro_x.append(i)
        elif oldest in (3, 4):
            pool_all_c2_x.append(i)
            if not touches_c1_bf:
                pool_sub_x.append(i)

    c_macro_x = Counter(len(hx_checks[i]) for i in pool_macro_x)
    c_sub_x = Counter(len(hx_checks[i]) for i in pool_sub_x)
    quota_x = {w: min(c_macro_x[w], c_sub_x[w]) for w in sorted(set(c_macro_x) | set(c_sub_x))}
    u_macro_x = select_matched_strided(pool_macro_x, hx_checks, quota_x)
    u_sub_x = select_matched_strided(pool_sub_x, hx_checks, quota_x)
    u_all_c2_x = set(pool_all_c2_x)
    assert u_macro_x.isdisjoint(u_sub_x)

    for lam_tenths in (2, 4, 6, 8):
        for tag_u, uz, ux in [
            ("macro_weld", u_macro_z, u_macro_x),
            ("sub_weld", u_sub_z, u_sub_x),
            ("all_c2_weld", u_all_c2_z, u_all_c2_x),
        ]:
            write_j_subset(
                len(hz),
                uz,
                lam_tenths,
                os.path.join(DATA_DIR, f"{name}_X_err_{tag_u}_l{lam_tenths:02d}.j"),
            )
            write_j_subset(
                len(hx_checks),
                ux,
                lam_tenths,
                os.path.join(DATA_DIR, f"{name}_Z_err_{tag_u}_l{lam_tenths:02d}.j"),
            )

    summary = {
        "name": name,
        "N": n_q,
        "M_Z": len(hz),
        "M_X": len(hx_checks),
        "matched_Z_count": len(u_macro_z),
        "matched_Z_quota_by_wt": quota_z,
        "pool_macro_Z_total": len(pool_macro_z),
        "pool_sub_Z_total": len(pool_sub_z),
        "matched_X_count": len(u_macro_x),
        "matched_X_quota_by_wt": quota_x,
        "pool_macro_X_total": len(pool_macro_x),
        "pool_sub_X_total": len(pool_sub_x),
        "all_c2_X_total": len(u_all_c2_x),
    }
    print("Built intervention sets:", json.dumps(summary), flush=True)
    return summary


def ensure_seed_codes():
    p0 = seed_planar_11()
    n0, hx0, hz0, *_ = sheaf_to_css_with_seams(p0)
    verify_and_write_code(n0, hx0, hz0, [{0, 1}], [{0}], "sq_iso_L2r1_s0", ["X_err", "Z_err"])

    t0 = seed_rot_torus_422()
    nt0, hxt0, hzt0, *_ = sheaf_to_css_with_seams(t0)
    verify_and_write_code(nt0, hxt0, hzt0, [{0, 1}, {0, 2}], [{1, 3}, {2, 3}], "rottor_iso_L2r1_s0", ["X_err", "Z_err"])


def exact_wire_beta(beta0, w):
    t = math.tanh(0.5 * beta0) ** (1.0 / w)
    return 2.0 * math.atanh(t)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_seed_codes()

    meta_sq = build_cycle2_interventions(seed_planar_11, "sq_iso_L2r1_s4")
    meta_tor = build_cycle2_interventions(seed_rot_torus_422, "rottor_iso_L2r1_s4")
    with open(os.path.join(WORK_DIR, "intervention_metadata_v14.json"), "w") as f:
        json.dump({"sq_iso_L2r1_s4": meta_sq, "rottor_iso_L2r1_s4": meta_tor}, f, indent=2)

    heavy_tasks = []
    fast_tasks = []

    # -------------------------------------------------------------------------
    # SUITE 1: Selected-Scale Coupling Intervention H(lambda; U) on Cycle 2
    # -------------------------------------------------------------------------
    for cname in ("sq_iso_L2r1_s4", "rottor_iso_L2r1_s4"):
        for tag_u in ("macro_weld", "sub_weld", "all_c2_weld"):
            for lam in (2, 4, 6, 8):
                for b in ("5.50", "6.00", "6.25", "6.50"):
                    for sec in ("X_err", "Z_err"):
                        cp = f"{DATA_DIR}/{cname}_{sec}.code"
                        jp = f"{DATA_DIR}/{cname}_{sec}_{tag_u}_l{lam:02d}.j"
                        tid = f"interv__{cname}_{sec}__{tag_u}__l{lam:02d}__b{b}"
                        # For Z_err (ultra-long sector on Cycle 2 at high lambda/beta), 28 trials w/ 130M cap suffices
                        is_long_z = (sec == "Z_err" and lam >= 6 and float(b) >= 6.00)
                        tr = 28 if is_long_z else 36
                        mfl = 130000000 if is_long_z else 220000000
                        tgt = heavy_tasks if (lam >= 6 and float(b) >= 6.00) else fast_tasks
                        tgt.append(f"{tid}|kmc|{cp}|{jp}|uf|{b}|{tr}|{mfl}")

    # -------------------------------------------------------------------------
    # SUITE 2: Coarse-Scale Excursion & Escape Diagnostic (`excursion_kmc`)
    # -------------------------------------------------------------------------
    for cname in ("sq_iso_L2r1_s4", "rottor_iso_L2r1_s4"):
        cp = f"{DATA_DIR}/{cname}_X_err.code"
        dp = f"{DATA_DIR}/{cname}.rgdec"
        j_specs = [("uniform", "uniform")]
        for tag_u in ("macro_weld", "sub_weld"):
            for lam in (4, 6):
                j_specs.append((f"{tag_u}_l{lam:02d}", f"{DATA_DIR}/{cname}_X_err_{tag_u}_l{lam:02d}.j"))
        for jtag, jp in j_specs:
            for b in ("5.50", "6.00", "6.25", "6.50"):
                tid = f"excur__{cname}_X_err__{jtag}__b{b}"
                tgt = heavy_tasks if (jtag == "uniform" or float(b) >= 6.25) else fast_tasks
                tgt.append(f"{tid}|excursion_kmc|{cp}|{jp}|{dp}|{b}|36|220000000")

    # -------------------------------------------------------------------------
    # SUITE 3: Paired Readout Cadence Sensitivity Audit (`paired_cadence`)
    # -------------------------------------------------------------------------
    for cname in ("sq_iso_L2r1_s0", "rottor_iso_L2r1_s0"):
        for sec in ("X_err", "Z_err"):
            if cname == "sq_iso_L2r1_s0" and sec == "X_err":
                pass  # still valid 1-check or 2-check seed!
            cp = f"{DATA_DIR}/{cname}_{sec}.code"
            if os.path.exists(cp):
                for b in ("4.50", "5.50", "6.50"):
                    fast_tasks.append(f"cadence__{cname}_{sec}__b{b}|paired_cadence|{cp}|{b}|2000|10000000|0.00")

    for cname in ("sq_iso_L2r1_s2", "rottor_iso_L2r1_s2"):
        for sec in ("X_err", "Z_err"):
            cp = f"{DATA_DIR}/{cname}_{sec}.code"
            for b in ("5.00", "5.50", "6.00", "6.50"):
                fast_tasks.append(f"cadence__{cname}_{sec}__b{b}|paired_cadence|{cp}|{b}|120|60000000|0.00")

    for cname in ("sq_iso_L2r1_s4", "rottor_iso_L2r1_s4"):
        for sec in ("X_err", "Z_err"):
            cp = f"{DATA_DIR}/{cname}_{sec}.code"
            heavy_tasks.append(f"cadence__{cname}_{sec}__b5.50_eta0|paired_cadence|{cp}|5.50|48|80000000|0.00")
            for b in ("6.00", "6.25", "6.50"):
                tr = 24 if (sec == "Z_err" and float(b) >= 6.25) else 36
                mfl = 120000000 if (sec == "Z_err" and float(b) >= 6.25) else 200000000
                heavy_tasks.append(f"cadence__{cname}_{sec}__b{b}_eta02|paired_cadence|{cp}|{b}|{tr}|{mfl}|0.02")

    # -------------------------------------------------------------------------
    # SUITE 4: Coarse-Syndrome Flow Burn-In & Matched-Weight i.i.d. Reference
    # -------------------------------------------------------------------------
    for b in ("3.50", "4.50", "5.50", "5.75", "6.00", "6.25", "6.50", "7.00"):
        fast_tasks.append(
            f"rgflow14__rottor_s4__t20__b{b}|rg_flow_v14|{DATA_DIR}/rottor_iso_L2r1_s4_X_err.code|{DATA_DIR}/rottor_iso_L2r1_s4.rgdec|{DATA_DIR}/rottor_iso_L2r1_s4_v2.rg|{b}|20.0|96"
        )
    for teq in ("5.0", "80.0", "320.0"):
        for b in ("5.50", "6.00", "6.25", "6.50"):
            tgt = heavy_tasks if float(teq) >= 80.0 else fast_tasks
            tgt.append(
                f"rgflow14__rottor_s4__t{teq}__b{b}|rg_flow_v14|{DATA_DIR}/rottor_iso_L2r1_s4_X_err.code|{DATA_DIR}/rottor_iso_L2r1_s4.rgdec|{DATA_DIR}/rottor_iso_L2r1_s4_v2.rg|{b}|{teq}|96"
            )

    # -------------------------------------------------------------------------
    # SUITE 5: 3x3 beta=7.50 Two-Sector Completion & Exact Matched-beta_eff Wires
    # -------------------------------------------------------------------------
    for s, tr in [(1, 120), (2, 80), (3, 32)]:
        for sec in ("X_err", "Z_err"):
            tr_use = 16 if (s == 3 and sec == "X_err") else tr
            mfl = 150000000 if (s == 3 and sec == "X_err") else 250000000
            heavy_tasks.append(
                f"tlist__sq_L3r1_s{s}_{sec}__b7.50|kmc|{DATA_DIR}/sq_L3r1_s{s}_{sec}.code|uniform|uf|7.50|{tr_use}|{mfl}"
            )

    for b0 in (5.00, 5.50, 6.00):
        for w in (1, 2, 4, 8, 16):
            bw = exact_wire_beta(b0, w)
            for fam, cname, tr in [("sq", "wire_sq_L2r1_s2", 120), ("tor", "wire_tor_L2r1_s2", 96)]:
                cp = f"{DATA_DIR}/{cname}_w{w}_X_err.code"
                tid = f"wire_beff__{fam}_w{w}__beff{b0:.2f}"
                tgt = heavy_tasks if (w >= 8 and b0 >= 5.50) else fast_tasks
                tgt.append(f"{tid}|kmc|{cp}|uniform|uf|{bw:.6f}|{tr}|200000000")

    all_tasks = heavy_tasks + fast_tasks
    n_shards = 120
    manifest_p = os.path.join(WORK_DIR, "manifest_v14.txt")
    with open(manifest_p, "w") as f:
        for idx, t in enumerate(all_tasks):
            f.write(f"{idx % n_shards}|{t}\n")
    print(
        f"Wrote {len(all_tasks)} tasks ({len(heavy_tasks)} heavy + {len(fast_tasks)} fast) across {n_shards} shards to {manifest_p}",
        flush=True,
    )


if __name__ == "__main__":
    main()
