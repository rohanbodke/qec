#!/usr/bin/env python3
"""
build_v13_referee_upgrade.py
Builds all codes, controlled wire-length sweeps, and task manifests for V13:
  1. Controlled wire-length sweep w in {1, 2, 4, 8, 16} on fixed uniform 2x2 welded cores
     (`sq_iso_L2r1_s2_X_err` N0=323 K=1 and `rottor_iso_L2r1_s2_X_err` N0=646 K=2).
  2. Paired-trajectory & paired i.i.d. single-shot benchmarks (`uf` vs `rg_dec` vs strict O(N) `rg_strict`).
  3. Upgraded coarse-syndrome flow (`rg_flow_v13`) recording exact check counts, denominators, SEMs, & logical class.
  4. Complete two-sector survival time lists (`t_list`) for exact tau_Q = int_0^inf S_X(t) S_Z(t) dt:
     - Uniform 2x2 planar (`sq_iso_L2r1` s=1..5) and torus (`rottor_iso_L2r1` s=1..4)
     - Uniform 3x3 single-cross quantum (`sq_L3r1` s=1..3: 41, 1,271, 24,449q)
     - Matched 2-rib direct graph (`sq_aniso_L3r2_s2` 1,480q) vs cubic-grid routed (`3,067`, `10,280`, `56,312`q).
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v13")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import seed_planar_11, seed_rot_torus_422, sheaf_to_css_with_seams
from build_v9_quarter_million import (
    apply_sheaf_step_with_maps,
    edge_qstarts,
    push_X_logical,
    push_Z_logical,
    verify_and_write_code,
)
from build_v11_rg_and_cooling import compute_sheet_lut, compute_z_cell_basis, write_rgv2_file, write_rgdec_file


def load_code_file(path):
    with open(path) as f:
        toks = list(map(int, f.read().split()))
    idx = 0
    N, M_check, M_stab, K = toks[idx:idx + 4]
    idx += 4
    checks, stabs, logs = [], [], []
    for _ in range(M_check):
        d = toks[idx]; idx += 1
        checks.append(toks[idx:idx + d]); idx += d
    for _ in range(M_stab):
        d = toks[idx]; idx += 1
        stabs.append(toks[idx:idx + d]); idx += d
    for _ in range(K):
        d = toks[idx]; idx += 1
        logs.append(toks[idx:idx + d]); idx += d
    return N, checks, stabs, logs


def build_wire_subdivided_code(base_code_path, out_prefix, w_list):
    N0, checks0, stabs0, logs0 = load_code_file(base_code_path)
    M0 = len(checks0)
    K = len(logs0)
    built = []
    for w in w_list:
        out_path = os.path.join(DATA_DIR, f"{out_prefix}_w{w}_X_err.code")
        if w == 1:
            Nw, ch_w, st_w, lg_w = N0, checks0, stabs0, logs0
        else:
            Nw = N0 + (w - 1) * M0
            ch_w = []
            q_legs = []
            for c_idx, row in enumerate(checks0):
                qc = row[0]
                q_legs.append(qc)
                rest = row[1:]
                u_base = N0 + c_idx * (w - 1)
                ch_w.append(rest + [u_base])
                for k in range(w - 2):
                    ch_w.append([u_base + k, u_base + k + 1])
                ch_w.append([u_base + (w - 2), qc])

            def lift_vec(supp):
                sset = set(supp)
                out = list(supp)
                for c_idx, qc in enumerate(q_legs):
                    if qc in sset:
                        u_base = N0 + c_idx * (w - 1)
                        out.extend(range(u_base, u_base + (w - 1)))
                return sorted(out)

            st_w = [lift_vec(s) for s in stabs0]
            lg_w = logs0

        with open(out_path, "w") as f:
            f.write(f"{Nw} {len(ch_w)} {len(st_w)} {K}\n")
            for r in ch_w:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in st_w:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
            for r in lg_w:
                f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "verify", out_path], check=True, stdout=subprocess.DEVNULL)
        built.append((w, Nw, len(ch_w)))
    return built


def ensure_core_and_2rib_codes():
    import build_v9_quarter_million as b9
    b9.ENGINE = ENGINE

    # 1. Ensure `sq_aniso_L3r2_s2` (1,480q 2-rib direct graph Cycle 1) exists and is verified:
    comp = seed_planar_11()
    lx, lz = [{0, 1}], [{0}]
    prev_cyl = None
    for it in ("X", "Z"):
        pcomp = comp
        comp, prev_cyl, meta = apply_sheaf_step_with_maps(pcomp, 3, [1, 2], it, prev_cyl=prev_cyl)
        N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]
    verify_and_write_code(N, HX, HZ, lx, lz, "sq_aniso_L3r2_s2", ["X_err", "Z_err"])
    print(f"Verified sq_aniso_L3r2_s2: N={N}", flush=True)

    # 2. Ensure `sq_rx_iso_s5_v2.rg` exists so `rg_flow_v13` can run on 5-scale single-sector too!
    rg_rx5 = os.path.join(DATA_DIR, "sq_rx_iso_s5_v2.rg")
    if not os.path.exists(rg_rx5):
        c_rx = seed_planar_11()
        N0, _, HZ0, *_ = sheaf_to_css_with_seams(c_rx)
        rg_lvs = [(N0, HZ0, 0, [], [], "S")]
        for s in range(1, 6):
            pc = c_rx
            c_rx, _, meta = apply_sheaf_step_with_maps(pc, 2, [1], "X", None)
            Ns, _, HZs, *_ = sheaf_to_css_with_seams(c_rx)
            n_xf = meta["n_xf"]
            n_pf = len(pc.faces)
            pf_spans = [(4 * pf, 4) for pf in range(n_pf)]
            bf_spans = [(2 * n_xf + 4 * pf, 4) for pf in range(n_pf)]
            rg_lvs.append((Ns, HZs, n_xf, pf_spans, bf_spans, "X"))
        write_rgv2_file(rg_rx5, 5, rg_lvs)
        print(f"Wrote {rg_rx5}", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_core_and_2rib_codes()

    w_sq = build_wire_subdivided_code(
        os.path.join(DATA_DIR, "sq_iso_L2r1_s2_X_err.code"),
        "wire_sq_L2r1_s2",
        [1, 2, 4, 8, 16],
    )
    print("Built wire_sq_L2r1_s2:", w_sq, flush=True)

    w_tor = build_wire_subdivided_code(
        os.path.join(DATA_DIR, "rottor_iso_L2r1_s2_X_err.code"),
        "wire_tor_L2r1_s2",
        [1, 2, 4, 8, 16],
    )
    print("Built wire_tor_L2r1_s2:", w_tor, flush=True)

    fast_tasks = []
    heavy_tasks = []

    # -------------------------------------------------------------------------
    # SUITE 1: Controlled Wire-Length Sweep (w = 1, 2, 4, 8, 16)
    # -------------------------------------------------------------------------
    for w in (1, 2, 4, 8, 16):
        for b in ("4.50", "5.00", "5.50", "6.00", "6.50", "7.00", "7.50", "8.00", "8.50"):
            # Skip ultra-cold w=1,2 at 8.0,8.5 so no capping wasted time; w>=4 needs up to 8.50!
            if w == 1 and float(b) > 7.00:
                continue
            if w == 2 and float(b) > 7.50:
                continue
            if w == 4 and float(b) > 8.00:
                continue
            cp = f"{DATA_DIR}/wire_sq_L2r1_s2_w{w}_X_err.code"
            tid = f"wire__sq_w{w}__b{b}"
            tgt = heavy_tasks if (float(b) >= 7.50) else fast_tasks
            tgt.append(f"{tid}|kmc|{cp}|uniform|uf|{b}|120|150000000")

        for b in ("5.00", "5.50", "6.00", "6.50", "7.00", "7.50", "8.00", "8.50"):
            if w == 1 and float(b) > 6.50:
                continue
            if w == 2 and float(b) > 7.00:
                continue
            if w == 4 and float(b) > 7.50:
                continue
            if w == 8 and float(b) > 8.00:
                continue
            cp = f"{DATA_DIR}/wire_tor_L2r1_s2_w{w}_X_err.code"
            tid = f"wire__tor_w{w}__b{b}"
            tgt = heavy_tasks if (float(b) >= 7.00) else fast_tasks
            tgt.append(f"{tid}|kmc|{cp}|uniform|uf|{b}|80|150000000")

    # -------------------------------------------------------------------------
    # SUITE 2: Paired Trajectory & Paired i.i.d. Readout Benchmarks
    # -------------------------------------------------------------------------
    for s, tr in [(2, 96), (4, 40)]:
        cp = f"{DATA_DIR}/rottor_iso_L2r1_s{s}_X_err.code"
        dp = f"{DATA_DIR}/rottor_iso_L2r1_s{s}.rgdec"
        for b in ("5.00", "5.50", "6.00", "6.25", "6.50", "6.65"):
            tgt = heavy_tasks if s == 4 else fast_tasks
            tgt.append(f"paired_kmc__rottor_s{s}__b{b}|paired_rg_kmc|{cp}|{dp}|{b}|{tr}|250000000")
        for p in ("0.0050", "0.0100", "0.0150", "0.0200", "0.0300", "0.0400", "0.0500", "0.0600", "0.0800", "0.1000"):
            shots = 2000 if s == 2 else 600
            tgt = heavy_tasks if s == 4 and float(p) >= 0.05 else fast_tasks
            tgt.append(f"iid_paired__rottor_s{s}__p{p}|iid_paired|{cp}|{dp}|{p}|{shots}")

    for s, tr in [(1, 160), (2, 120), (3, 96), (4, 48), (5, 28)]:
        cp = f"{DATA_DIR}/sq_rx_iso_s{s}_X_err.code"
        dp = f"{DATA_DIR}/sq_rx_iso_s{s}.rgdec"
        for b in ("1.75", "1.90", "2.00", "2.10", "2.15", "2.20", "2.25"):
            if s == 5 and b == "2.25":
                heavy_tasks.append(f"paired_kmc__sq_rx_s{s}__b{b}|paired_rg_kmc|{cp}|{dp}|{b}|20|220000000")
            else:
                tgt = heavy_tasks if s >= 4 else fast_tasks
                tgt.append(f"paired_kmc__sq_rx_s{s}__b{b}|paired_rg_kmc|{cp}|{dp}|{b}|{tr}|200000000")
        # Timing & i.i.d. accuracy across all 5 scales at p=0.02, 0.04, 0.06:
        for p in ("0.0200", "0.0400", "0.0600"):
            shots = 1000 if s <= 3 else (400 if s == 4 else 150)
            fast_tasks.append(f"iid_paired__sq_rx_s{s}__p{p}|iid_paired|{cp}|{dp}|{p}|{shots}")

    # -------------------------------------------------------------------------
    # SUITE 3: Coarse-Syndrome Flow v13 (counts, denominators, SEMs, & logical fail)
    # -------------------------------------------------------------------------
    for b in ("3.50", "4.50", "5.50", "5.75", "6.00", "6.25", "6.50", "7.00"):
        fast_tasks.append(
            f"rgflow13__rottor_s4__b{b}|rg_flow_v13|{DATA_DIR}/rottor_iso_L2r1_s4_X_err.code|{DATA_DIR}/rottor_iso_L2r1_s4.rgdec|{DATA_DIR}/rottor_iso_L2r1_s4_v2.rg|{b}|20.0|96"
        )
        heavy_tasks.append(
            f"rgflow13__sq_iso_s5__b{b}|rg_flow_v13|{DATA_DIR}/sq_iso_L2r1_s5_X_err.code|{DATA_DIR}/sq_iso_L2r1_s5.rgdec|{DATA_DIR}/sq_iso_L2r1_s5_v2.rg|{b}|20.0|64"
        )
    for b in ("1.35", "1.60", "1.75", "1.86", "2.00", "2.15", "2.35", "2.65"):
        heavy_tasks.append(
            f"rgflow13__sq_rx_s5__b{b}|rg_flow_v13|{DATA_DIR}/sq_rx_iso_s5_X_err.code|{DATA_DIR}/sq_rx_iso_s5.rgdec|{DATA_DIR}/sq_rx_iso_s5_v2.rg|{b}|20.0|96"
        )

    # -------------------------------------------------------------------------
    # SUITE 4: Complete Two-Sector Survival Curves (`t_list`) for Exact tau_Q
    # -------------------------------------------------------------------------
    # 4A. Uniform 3x3 single-cross quantum (`sq_L3r1` s=1, 2, 3: 41, 1,271, 24,449q)
    for s, tr in [(1, 120), (2, 80), (3, 32)]:
        for b in ("4.50", "5.00", "5.50", "6.00", "6.25", "6.50", "7.00"):
            for sec in ("X_err", "Z_err"):
                # At s=3 X_err for b>=6.25, lifetime >> 10^7 while Z_err is <= 3e4; run 16 trials with 150M cap so survival integral up to max(T_Z) is 100% exact!
                tr_use = 16 if (s == 3 and sec == "X_err" and float(b) >= 6.00) else tr
                mfl = 150000000 if (s == 3 and sec == "X_err" and float(b) >= 6.00) else 250000000
                tgt = heavy_tasks if s == 3 else fast_tasks
                tgt.append(f"tlist__sq_L3r1_s{s}_{sec}__b{b}|kmc|{DATA_DIR}/sq_L3r1_s{s}_{sec}.code|uniform|uf|{b}|{tr_use}|{mfl}")

    # 4B. Matched 2-rib direct graph (`sq_aniso_L3r2_s2` 1,480q) vs 1-rib (`sq_iso_L2r1_s2` 323q) vs routed grids (`3,067`, `10,280`, `56,312`q)
    for cname, tr in [
        ("sq_iso_L2r1_s2", 96),
        ("sq_aniso_L3r2_s2", 80),
        ("sq_z3_3_5_r1", 80),
        ("sq_z3_4_6_r2", 48),
        ("sq_bdl_8_8_r2", 28),
    ]:
        for b in ("5.00", "5.50", "6.00", "6.50", "7.00"):
            if cname in ("sq_iso_L2r1_s2", "sq_aniso_L3r2_s2") and b == "7.00":
                continue
            for sec in ("X_err", "Z_err"):
                tgt = heavy_tasks if cname in ("sq_z3_4_6_r2", "sq_bdl_8_8_r2") else fast_tasks
                mfl = 150000000 if (sec == "Z_err" and float(b) >= 6.50) else 250000000
                tgt.append(f"tlist__{cname}_{sec}__b{b}|kmc|{DATA_DIR}/{cname}_{sec}.code|uniform|uf|{b}|{tr}|{mfl}")

    # 4C. Unified Uniform 2x2 Planar (`sq_iso_L2r1` s=1,2,3,4,5) & Torus (`rottor_iso_L2r1` s=1,2,3,4) two-sector t_list
    for s, tr in [(1, 120), (2, 96), (3, 64), (4, 36)]:
        for b in ("5.00", "5.50", "6.00", "6.25", "6.50"):
            for sec in ("X_err", "Z_err"):
                if s == 2 and b in ("5.00", "5.50", "6.00", "6.50"):
                    pass  # already in 4B for X_err/Z_err, skip duplicate for sq_iso_L2r1_s2!
                else:
                    tgt = heavy_tasks if s == 4 else fast_tasks
                    mfl = 150000000 if ((s % 2 == 1 and sec == "X_err") or (s % 2 == 0 and sec == "Z_err")) else 250000000
                    tgt.append(f"tlist__sq_iso_L2r1_s{s}_{sec}__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s{s}_{sec}.code|uniform|uf|{b}|{tr}|{mfl}")
            # Torus s=1..4:
            for sec in ("X_err", "Z_err"):
                tgt = heavy_tasks if s == 4 else fast_tasks
                mfl = 120000000 if ((s % 2 == 1 and sec == "X_err") or (s % 2 == 0 and sec == "Z_err")) else 250000000
                tr_t = 24 if (s >= 3 and mfl == 120000000 and float(b) >= 6.00) else tr
                tgt.append(f"tlist__rottor_iso_L2r1_s{s}_{sec}__b{b}|kmc|{DATA_DIR}/rottor_iso_L2r1_s{s}_{sec}.code|uniform|uf|{b}|{tr_t}|{mfl}")

    # Planar s=5 (423,559q) two-sector at b=5.50, 6.00, 6.25, 6.50:
    for b in ("5.50", "6.00", "6.25", "6.50"):
        heavy_tasks.append(f"tlist__sq_iso_L2r1_s5_Z_err__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s5_Z_err.code|uniform|uf|{b}|20|350000000")
        heavy_tasks.append(f"tlist__sq_iso_L2r1_s5_X_err__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s5_X_err.code|uniform|uf|{b}|12|80000000")

    # Shard evenly across 120 array shards so every shard finishes in ~5-10 minutes!
    all_tasks = heavy_tasks + fast_tasks
    n_shards = 120
    manifest_p = os.path.join(WORK_DIR, "manifest_v13.txt")
    with open(manifest_p, "w") as f:
        for idx, t in enumerate(all_tasks):
            f.write(f"{idx % n_shards}|{t}\n")
    print(f"Wrote {len(all_tasks)} tasks ({len(heavy_tasks)} heavy + {len(fast_tasks)} fast) across {n_shards} shards to {manifest_p}", flush=True)


if __name__ == "__main__":
    main()
