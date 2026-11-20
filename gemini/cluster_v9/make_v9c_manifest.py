#!/usr/bin/env python3
import os

WORK = "/work/submit/harshila/6s980_qmem_v9"
DATA = f"{WORK}/data"

fast_tasks = []
s5_tasks = []

rx_betas = ["1.60", "1.75", "1.90", "2.00", "2.08", "2.15", "2.22"]
for fam in ("sq_rx_iso", "rottor_rx", "sq_rx"):
    for s in range(5):
        for b in rx_betas:
            tag = f"{fam}_s{s}_X_err"
            fast_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|40|200000000")
    for b in rx_betas:
        tag = f"{fam}_s5_X_err"
        s5_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|20|300000000")

q_tor_betas = ["5.00", "5.25", "5.50", "5.75", "6.00"]
for s in (1, 3):
    for b in q_tor_betas:
        tag = f"rottor_L2r1_s{s}_Z_err"
        fast_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|48|200000000")
for b in q_tor_betas:
    tag = "rottor_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__gain_uf__b{b}|kmc|{DATA}/{tag}.code|uniform|gain_uf|{b}|16|300000000")

for b in ("6.12", "6.18"):
    tag = "sq_L2r1_s5_Z_err"
    s5_tasks.append(f"kmc__{tag}__uniform__uf__b{b}|kmc|{DATA}/{tag}.code|uniform|uf|{b}|20|300000000")

out_path = os.path.join(os.path.dirname(__file__), "manifest_v9c.txt")
total_shards = 4 + len(s5_tasks)
with open(out_path, "w") as f:
    for idx, ln in enumerate(fast_tasks):
        f.write(f"{idx % 4}|{ln}\n")
    for idx, ln in enumerate(s5_tasks):
        f.write(f"{4 + idx}|{ln}\n")

print(f"Wrote {len(fast_tasks)} fast + {len(s5_tasks)} s=5 -> total_shards={total_shards}")
