#!/usr/bin/env python3
"""
build_v8_rigorous.py
Builds and verifies the V8 Multi-Scale & Provably-Contractive 2x2 Benchmark Suite on SubMIT:
  1. 5-Scale 3D Welded Family (`s = 0, 1, 2, 3, 4` consecutive R_X steps of the 2x2 four-arm cross):
     - Boundaryless Torus (`rottor_rx_s0`..`s4`: N = 4, 38, 332, 3,100, 29,704; K = 2)
     - Planar Patch (`sq_rx_s0`..`s4`: N = 2, 19, 166, 1,550, 14,852; K = 1)
     - Uniform Planar Patch (`sq_rx_iso_s0`..`s4`: N = 2, 19, 219, 2,571, 30,507; K = 1)
  2. Two-Cycle Full (X, Z) Quantum Crossover on the Provably Contractive (`c_cycle = 2/5`) Uniform 2x2 Cell:
     - `sq_iso_L2r1_s2` (Cycle 1, N = 323, K = 1) vs `sq_iso_L2r1_s4` (Cycle 2, N = 47,387, K = 1)
  3. Extended Cold Points (`beta = 6.65, 6.75, 7.00`) on Boundaryless Torus Cycle 2 (`rottor_L2r1_s4`, N = 50,296, K = 2).
"""

from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys

V7_DATA = "/work/submit/harshila/6s980_qmem_v7/data"
WORK_DIR = "/work/submit/harshila/6s980_qmem_v8"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v8")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    seed_planar_11,
    seed_rot_torus_422,
    apply_sheaf_step,
    sheaf_to_css_with_seams,
)


def export_one_sector(comp, tag, sec):
    N, HX, HZ, *_ = sheaf_to_css_with_seams(comp)
    checks, stabs = (HZ, HX) if sec == "X_err" else (HX, HZ)
    fin_p = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
    if not os.path.exists(fin_p):
        raw_p = os.path.join(DATA_DIR, f"{tag}_{sec}.raw")
        with open(raw_p, "w") as f:
            f.write(f"{N} {len(checks)} {len(stabs)}\n")
            for r in checks:
                f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
            for r in stabs:
                f.write(str(len(r)) + " " + " ".join(map(str, r)) + "\n")
        subprocess.run([ENGINE, "prep_logicals", raw_p, fin_p], check=True)
        os.remove(raw_p)
    vout = subprocess.run([ENGINE, "verify", fin_p], capture_output=True, text=True, check=True).stdout.strip()
    print(f"  Built & verified: {vout}", flush=True)
    return fin_p


def build_chain(args):
    chain_type, seed_kind, steps_list, aniso, sectors = args
    comp = seed_planar_11() if seed_kind == "planar" else seed_rot_torus_422()
    cyl = set()
    built = []
    if 0 in steps_list:
        for sec in sectors:
            if len(comp.faces) > 0 and (sec == "X_err" or seed_kind == "torus"):
                export_one_sector(comp, f"{chain_type}_s0", sec)
                built.append(f"{chain_type}_s0_{sec}")
    max_s = max(steps_list)
    for s in range(1, max_s + 1):
        itype = "X" if ("rx" in chain_type or s % 2 == 1) else "Z"
        comp, new_cyl = apply_sheaf_step(comp, 2, [1], itype, prev_cyl=(cyl if aniso else None))
        cyl = new_cyl
        if s in steps_list:
            for sec in sectors:
                export_one_sector(comp, f"{chain_type}_s{s}", sec)
                built.append(f"{chain_type}_s{s}_{sec}")
    return built


def kmc_line(tag, b, trials, cap):
    cp = os.path.join(DATA_DIR, f"{tag}.code")
    return f"kmc__{tag}__uniform__uf__b{b:.2f}|kmc|{cp}|uniform|uf|{b:.2f}|{trials}|{cap}"


def ffs_line(tag, b, shots, reps=6):
    cp = os.path.join(DATA_DIR, f"{tag}.code")
    return f"ffs__{tag}__uniform__uf__b{b:.2f}|ffs|{cp}|uniform|uf|{b:.2f}|{shots}|{reps}|"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    chains = [
        ("rottor_rx",   "torus",  [0, 1, 2, 3, 4], True,  ["X_err"]),
        ("sq_rx",       "planar", [0, 1, 2, 3, 4], True,  ["X_err"]),
        ("sq_rx_iso",   "planar", [0, 1, 2, 3, 4], False, ["X_err"]),
        ("sq_iso_L2r1", "planar", [2, 4],          False, ["X_err", "Z_err"]),
    ]

    with ProcessPoolExecutor(max_workers=4) as ex:
        for res in ex.map(build_chain, chains):
            print("Completed chain:", res, flush=True)

    for tag in ["rottor_L2r1_s2_X_err", "rottor_L2r1_s2_Z_err", "rottor_L2r1_s4_X_err", "rottor_L2r1_s4_Z_err"]:
        src = os.path.join(V7_DATA, f"{tag}.code")
        dst = os.path.join(DATA_DIR, f"{tag}.code")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        vout = subprocess.run([ENGINE, "verify", dst], capture_output=True, text=True, check=True).stdout.strip()
        print(f"  Linked & verified: {vout}", flush=True)

    tasks = []

    # 1. 5-Scale Torus Pure R_X^s (s = 0..4: N = 4, 38, 332, 3100, 29704)
    for s in range(5):
        tag = f"rottor_rx_s{s}_X_err"
        trials = 160 if s <= 2 else (96 if s == 3 else 48)
        cap = 40000000 if s <= 2 else (120000000 if s == 3 else 160000000)
        betas = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
        if s == 4:
            betas = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.50]
        for b in betas:
            tasks.append(kmc_line(tag, b, trials, cap))
        if s in (3, 4):
            for b in [2.50, 2.75, 3.00]:
                shots = 1600 if s == 3 else 900
                tasks.append(ffs_line(tag, b, shots, 6))

    # 2. 5-Scale Planar Pure R_X^s (thin & uniform, s = 0..4)
    for pref in ("sq_rx", "sq_rx_iso"):
        for s in range(5):
            tag = f"{pref}_s{s}_X_err"
            trials = 160 if s <= 2 else (96 if s == 3 else 48)
            cap = 40000000 if s <= 2 else (120000000 if s == 3 else 160000000)
            betas = [1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
            if s == 4:
                betas = [1.25, 1.50, 1.75, 2.00, 2.25, 2.40, 2.50, 2.65]
            for b in betas:
                tasks.append(kmc_line(tag, b, trials, cap))
            if s in (3, 4):
                for b in [2.65, 2.75, 3.00]:
                    shots = 1600 if s == 3 else 900
                    tasks.append(ffs_line(tag, b, shots, 6))

    # 3. Provably-Contractive Uniform 2x2 Two-Cycle Quantum Crossover (`sq_iso_L2r1_s2` vs `sq_iso_L2r1_s4`)
    for s, trials, cap in [(2, 96, 80000000), (4, 36, 180000000)]:
        for sec in ("X_err", "Z_err"):
            tag = f"sq_iso_L2r1_s{s}_{sec}"
            for b in [5.00, 5.50, 6.00, 6.25, 6.50, 6.75, 7.00]:
                if s == 4 and sec == "Z_err" and b >= 6.00:
                    if b in (6.00, 6.25, 6.50):
                        tasks.append(ffs_line(tag, b, 900, 6))
                    continue
                tasks.append(kmc_line(tag, b, trials, cap))
                if s == 4 and sec == "X_err" and b >= 6.50:
                    tasks.append(ffs_line(tag, b, 900, 6))

    # 4. Extended Cold Torus Cycle 1 vs Cycle 2 (`rottor_L2r1_s2` vs `rottor_L2r1_s4` at beta = 6.65, 6.75, 7.00)
    for b in [6.65, 6.75, 7.00]:
        tasks.append(kmc_line("rottor_L2r1_s2_X_err", b, 96, 100000000))
        tasks.append(kmc_line("rottor_L2r1_s4_X_err", b, 32, 220000000))
        tasks.append(ffs_line("rottor_L2r1_s4_X_err", b, 900, 6))

    man_p = os.path.join(WORK_DIR, "manifest_v8.txt")
    with open(man_p, "w") as f:
        for t in tasks:
            f.write(t + "\n")
    print(f"Wrote {len(tasks)} tasks to {man_p}", flush=True)


if __name__ == "__main__":
    main()
