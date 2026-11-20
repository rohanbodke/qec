#!/usr/bin/env python3
"""
Generates `manifest_v9b.txt` with two shard groups:
  - Shards 0..5: all fast s=0..4 (and s=1,3 quantum) `maj_uf` reference runs (<30s total per shard).
  - Shards 6..N-1: ONE s=5 quarter-million-qubit task per shard (8 threads, 32G RAM, zero OOM/blocking).
"""
import os

WORK = "/work/submit/harshila/6s980_qmem_v9"
DATA = f"{WORK}/data"

fast_tasks = []
s5_tasks = []

# 1. 6-Level (s=0..5) 3D 2x2 Welded Families with consistent `maj_uf` across ALL scales
rx_betas = ["1.60", "1.75", "1.90", "2.00", "2.08", "2.15", "2.22"]
for fam in ("sq_rx_iso", "rottor_rx", "sq_rx"):
    for s in range(5):
        for b in rx_betas:
            tag = f"{fam}_s{s}_X_err"
            tid = f"kmc__{tag}__uniform__maj_uf__b{b}"
            fast_tasks.append(f"{tid}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|40|200000000")
    for b in rx_betas:
        tag = f"{fam}_s5_X_err"
        tid = f"kmc__{tag}__uniform__maj_uf__b{b}"
        s5_tasks.append(f"{tid}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|20|320000000")

# Also add s=4 at 2.08, 2.22 for `uf` and s=5 cold pure-`uf` points
for b in ("2.20", "2.28", "2.35"):
    for fam in ("sq_rx_iso", "rottor_rx"):
        t4 = f"{fam}_s4_X_err"
        fast_tasks.append(f"kmc__{t4}__uniform__uf__b{b}|kmc|{DATA}/{t4}.code|uniform|uf|{b}|24|250000000")
        t5 = f"{fam}_s5_X_err"
        s5_tasks.append(f"kmc__{t5}__uniform__uf__b{b}|kmc|{DATA}/{t5}.code|uniform|uf|{b}|16|350000000")

# 2. 5-Step Full Two-Sector Quantum Welded Ladder (`s=1, 3, 5`) with consistent `maj_uf`
q_sq_betas = ["4.75", "5.00", "5.25", "5.50", "5.75", "6.00", "6.25", "6.50"]
for s in (1, 3):
    for b in q_sq_betas:
        tag = f"sq_L2r1_s{s}_Z_err"
        fast_tasks.append(f"kmc__{tag}__uniform__maj_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|48|200000000")
for b in q_sq_betas:
    tag = "sq_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__maj_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|20|350000000")

q_tor_betas = ["4.75", "5.00", "5.25", "5.50", "5.75", "6.00", "6.15"]
for s in (1, 3):
    for b in q_tor_betas:
        tag = f"rottor_L2r1_s{s}_Z_err"
        fast_tasks.append(f"kmc__{tag}__uniform__maj_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|48|200000000")
for b in q_tor_betas:
    tag = "rottor_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__maj_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|maj_uf|{b}|20|350000000")

# 3. Pure `uf` complement points on `sq_L2r1_s5_Z_err` (189,988q) to join finished 6.25 & 6.50!
for b in ("6.12", "6.18", "6.35"):
    tag = "sq_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__uf__b{b}|kmc|{DATA}/{tag}.code|uniform|uf|{b}|20|350000000")

# Assign shard IDs explicitly: shards 0..5 get fast_tasks round-robin; shards 6..6+len(s5)-1 get 1 s5 task each!
out_path = os.path.join(os.path.dirname(__file__), "manifest_v9b.txt")
total_shards = 6 + len(s5_tasks)
with open(out_path, "w") as f:
    for idx, ln in enumerate(fast_tasks):
        sh = idx % 6
        f.write(f"{sh}|{ln}\n")
    for idx, ln in enumerate(s5_tasks):
        sh = 6 + idx
        f.write(f"{sh}|{ln}\n")

print(f"Wrote {len(fast_tasks)} fast tasks (shards 0..5) + {len(s5_tasks)} s=5 tasks (shards 6..{total_shards-1}) -> total_shards={total_shards}")
