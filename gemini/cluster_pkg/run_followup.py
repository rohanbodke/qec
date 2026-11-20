#!/usr/bin/env python3
"""
Targeted 60-second follow-up on SubMIT Slurm cluster:
  1. Exact finite-temperature phase-transition crossing beta_c across EVEN full cycles k=0, 2, 4
     for 1-rib (cls_l3_r1: N=4, 196, 9604) vs 2-rib (cls_l4_r2: N=4, 400, 40000) across beta in [3.0 .. 5.5]!
  2. Uncapped high-flip KMC on 3D Quantum `bdl_tor2_hyb21_s2` vs `bdl_tor3_L2r1_s2` vs `bdl_sq_L3r2_s2` at beta=5.5, 6.0!
"""
import os, subprocess, sys, time

WORK_DIR = "/work/submit/harshila/6s980_qmem"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results_fup")
ENGINE = os.path.join(WORK_DIR, "kmc_engine")
os.makedirs(RES_DIR, exist_ok=True)

tasks = []
# 1. Even-step phase transition crossing for 1-rib (N=4, 196, 9604) and 2-rib (N=4, 400, 40000)
for pref in ("cls_l3_r1", "cls_l4_r2"):
    for st in (0, 2, 4):
        cname = f"{pref}_step{st}.code"
        tr = 160 if st <= 2 else 48
        max_fl = 25000000
        for beta in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5):
            tasks.append((f"CROSS|{cname}|b={beta}", [ENGINE, "kmc", os.path.join(DATA_DIR, cname), str(beta), "uf", str(tr), str(max_fl), "8"]))

if len(sys.argv) == 1:
    # Write manifest and submit 24-task Slurm array!
    print(f"Prepared {len(tasks)} follow-up tasks.")
else:
    sid = int(sys.argv[1])
    nsh = int(sys.argv[2])
    out_p = os.path.join(RES_DIR, f"fup_{sid:02d}.txt")
    with open(out_p, "w") as out:
        for i, (tag, cmd) in enumerate(tasks):
            if i % nsh != sid:
                continue
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                dt = time.time() - t0
                if p.returncode == 0:
                    line = f"{tag}|dt={dt:.1f}s|{p.stdout.strip()}"
                    out.write(line + "\n")
                    out.flush()
                    print(line, flush=True)
            except subprocess.TimeoutExpired:
                out.write(f"TIMEOUT|{tag}\n")
