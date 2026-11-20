#!/usr/bin/env python3
"""
build_v6b_followup.py
Reuses all pre-built `.code` files in `/work/submit/harshila/6s980_qmem_v6/data/`:
  1. Builds Independent-Set Pinning-Trap Couplings (`pin_norm.j` with exact <J> == 1.0000 & `pin_bnd.j` in [1.0, 2.0])
     on Z^3 cubic-grid codes up to BDL's 56,312-qubit `sq_bdl_8_8_r2`.
  2. Emits `manifest_v6b.txt` for fast strided FFS (`kmc_engine_v6b`) + uncapped moderate-beta KMC on s=3,4,5 & Z^3 pinning traps.
"""

import os

WORK_DIR = "/work/submit/harshila/6s980_qmem_v6"
DATA_DIR = os.path.join(WORK_DIR, "data")
MOV_DIR = os.path.join(WORK_DIR, "movies")


def make_pinning_traps(tag):
    cpath = os.path.join(DATA_DIR, f"{tag}.code")
    with open(cpath) as f:
        hdr = list(map(int, f.readline().split()))
        N, M = hdr[0], hdr[1]
        checks = []
        q_to_c = [[] for _ in range(N)]
        for ci in range(M):
            parts = list(map(int, f.readline().split()))
            row = parts[1:]
            checks.append(row)
            for q in row:
                q_to_c[q].append(ci)

    # Greedily select an independent set of non-junction checks (no shared qubits, no deg>=3 seam qubits!)
    blocked_c = [False] * M
    is_trap = [False] * M
    for ci, row in enumerate(checks):
        # Keep 3-way junctions & high-weight checks as strong walls!
        if len(row) >= 5 or any(len(q_to_c[q]) >= 3 for q in row):
            blocked_c[ci] = True
    for ci, row in enumerate(checks):
        if blocked_c[ci]:
            continue
        is_trap[ci] = True
        for q in row:
            for nb_c in q_to_c[q]:
                blocked_c[nb_c] = True

    # 1) pin_bnd: traps = 10 (J=1.0), walls = 20 (J=2.0)
    jbnd = [10 if is_trap[ci] else 20 for ci in range(M)]
    # 2) pin_norm: traps = raw 3, walls = raw 13 -> rescaled to exact sum == 10 * M (<J> == 1.0000)!
    jraw = [3 if is_trap[ci] else 13 for ci in range(M)]
    mr = sum(jraw) / M
    jnorm = [max(3, int(round(10.0 * v / mr))) for v in jraw]
    diff = 10 * M - sum(jnorm)
    # Adjust walls first so traps stay deep
    wall_ids = [ci for ci in range(M) if not is_trap[ci]] or list(range(M))
    idx = 0
    while diff != 0 and idx < 6 * len(wall_ids):
        ci = wall_ids[idx % len(wall_ids)]
        if diff > 0:
            jnorm[ci] += 1
            diff -= 1
        elif diff < 0 and jnorm[ci] > 6:
            jnorm[ci] -= 1
            diff += 1
        idx += 1

    for suffix, arr in [("pin_norm", jnorm), ("pin_bnd", jbnd)]:
        with open(os.path.join(DATA_DIR, f"{tag}_{suffix}.j"), "w") as out:
            out.write("\n".join(str(x) for x in arr) + "\n")
    n_tr = sum(is_trap)
    print(f"  {tag}: traps={n_tr}/{M} ({100.0*n_tr/M:.1f}%), <J_norm>={sum(jnorm)/(10.0*M):.5f}")


def main():
    z_codes = ["sq_z3_3_5_r1", "sq_z3_4_6_r2", "tor2_z3_4_6_r2", "sq_bdl_8_8_r2"]
    for zc in z_codes:
        for sec in ["X_err", "Z_err"]:
            make_pinning_traps(f"{zc}_{sec}")

    tasks = []

    def add_kmc(tag, j_mode, dec, beta, trials, max_flips):
        j_arg = "uniform" if j_mode == "uniform" else os.path.join(DATA_DIR, f"{tag}_{j_mode}.j")
        tid = f"kmc__{tag}__{j_mode}__{dec}__b{beta:.2f}"
        tasks.append(f"{tid}|kmc|{os.path.join(DATA_DIR, tag + '.code')}|{j_arg}|{dec}|{beta:.2f}|{trials}|{max_flips}")

    def add_ffs(tag, j_mode, dec, beta, shots, reps, save_movie=False):
        j_arg = "uniform" if j_mode == "uniform" else os.path.join(DATA_DIR, f"{tag}_{j_mode}.j")
        tid = f"ffs__{tag}__{j_mode}__{dec}__b{beta:.2f}"
        mov = os.path.join(MOV_DIR, f"{tid}.txt") if save_movie else ""
        tasks.append(f"{tid}|ffs|{os.path.join(DATA_DIR, tag + '.code')}|{j_arg}|{dec}|{beta:.2f}|{shots}|{reps}|{mov}")

    # 1. Pinning-Trap Z^3 Rescue (Up to BDL N=56,312!) + missing uniform sq_bdl_8_8_r2 points
    for zc in z_codes:
        for sec in ["X_err", "Z_err"]:
            tag = f"{zc}_{sec}"
            tr = 20 if ("bdl_8_8" in zc or "tor2" in zc) else 32
            for jm in ["pin_norm", "pin_bnd"]:
                for b in [5.50, 6.00, 6.50, 7.00]:
                    add_kmc(tag, jm, "uf", b, tr, 90_000_000)
            if zc == "sq_bdl_8_8_r2":
                for b in [5.50, 6.00, 7.00]:
                    add_kmc(tag, "uniform", "uf", b, 20, 90_000_000)

    # 2. Fast Strided FFS on Boundaryless [[4,2,2]] `rottor_L2r1` s=2, 3, 4 (beta = 6.0 .. 8.5) + Failure Movies!
    for s in [2, 3, 4]:
        for sec in ["X_err", "Z_err"]:
            tag = f"rottor_L2r1_s{s}_{sec}"
            shots = 500 if s == 4 else 700
            for b in [6.00, 6.50, 7.00, 7.50, 8.00, 8.50]:
                mov = (s in (2, 4) and b == 7.50 and sec == "X_err")
                add_ffs(tag, "uniform", "uf", b, shots, 3, save_movie=mov)

    # 3. Planar `sq_L2r1` s=2, 3, 4, 5 (Up to N=189,988!) & Proven 2-Bridge `sq_L3r2` s=2, 3 (N=22,160!)
    for s in [3, 4, 5]:
        for sec in ["X_err", "Z_err"]:
            tag = f"sq_L2r1_s{s}_{sec}"
            for b in [5.00, 5.50, 6.00]:
                add_kmc(tag, "uniform", "uf", b, 16 if s == 5 else 24, 60_000_000)
            for b in [6.50, 7.00, 7.50, 8.00]:
                shots = 350 if s == 5 else (500 if s == 4 else 700)
                mov = (s == 4 and b == 7.50 and sec == "X_err")
                add_ffs(tag, "uniform", "uf", b, shots, 3, save_movie=mov)

    for s in [2, 3]:
        for sec in ["X_err", "Z_err"]:
            tag = f"sq_L3r2_s{s}_{sec}"
            for b in [5.00, 5.50]:
                add_kmc(tag, "uniform", "uf", b, 24, 60_000_000)
            for b in [6.00, 6.50, 7.00, 7.50, 8.00]:
                add_ffs(tag, "uniform", "uf", b, 550 if s == 3 else 750, 3)

    # 4. 3D Toric L=6, 7 at beta=2.00, 2.25 via FFS + Cubic L=7, 9 at beta=5.0, 5.5, 6.0 + Caha FFS at beta=8.0, 9.0
    for L in [5, 6, 7]:
        for b in [2.00, 2.25]:
            add_ffs(f"toric3d_L{L}_X_err", "uniform", "sweep_uf", b, 700, 3)
    for L in [7, 9, 11]:
        for b in [5.00, 5.50, 6.00]:
            add_ffs(f"cubic_L{L}_X_err", "uniform", "uf", b, 800, 3)
    for sp in ["caha_n11_code_100", "caha_n15_code_10", "caha_n19_code_129"]:
        if os.path.exists(os.path.join(DATA_DIR, f"{sp}_X_err.code")):
            for b in [8.00, 9.00]:
                add_ffs(f"{sp}_X_err", "uniform", "uf", b, 600, 3)

    mpath = os.path.join(WORK_DIR, "manifest_v6b.txt")
    with open(mpath, "w") as f:
        for ln in tasks:
            f.write(ln + "\n")
    print(f"Wrote {len(tasks)} targeted V6b tasks to {mpath}!")


if __name__ == "__main__":
    main()
