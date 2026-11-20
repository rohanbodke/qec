#!/usr/bin/env python3
import os

WORK = "/work/submit/harshila/6s980_qmem_v9"
DATA = f"{WORK}/data"

fast_tasks = []
s5_tasks = []

rx_betas_fast = ["1.60", "1.75", "1.90", "2.00", "2.08", "2.15", "2.22", "2.28"]
rx_betas_s5   = ["2.00", "2.08", "2.15", "2.22", "2.28"]
for fam in ("sq_rx_iso", "rottor_rx", "sq_rx"):
    for s in range(5):
        for b in rx_betas_fast:
            tag = f"{fam}_s{s}_X_err"
            fast_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|36|250000000")
    for b in rx_betas_s5:
        tag = f"{fam}_s5_X_err"
        s5_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|18|320000000")

# Boundaryless 379,976-qubit torus 3-level quantum ladder (s=1, 3, 5) with exact zero-bailout gain_uf
q_tor_betas = ["5.50", "5.75", "6.00", "6.15", "6.30"]
for s in (1, 3):
    for b in q_tor_betas:
        tag = f"rottor_L2r1_s{s}_Z_err"
        fast_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|48|250000000")
for b in q_tor_betas:
    tag = "rottor_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|16|350000000")

# Pure `uf` zero-bailout complement points on 189,988-qubit sq_L2r1_s5_Z_err
for b in ("6.12", "6.18"):
    tag = "sq_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__uf__b{b}|kmc|{DATA}/{tag}.code|uniform|uf|{b}|18|350000000")

out_path = os.path.join(os.path.dirname(__file__), "manifest_v9d.txt")
total_shards = 4 + len(s5_tasks)
with open(out_path, "w") as f:
    for idx, ln in enumerate(fast_tasks):
        f.write(f"{idx % 4}|{ln}\n")
    for idx, ln in enumerate(s5_tasks):
        f.write(f"{4 + idx}|{ln}\n")

print(f"Wrote {len(fast_tasks)} fast + {len(s5_tasks)} s=5 -> total_shards={total_shards}")
