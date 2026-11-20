#!/usr/bin/env python3
"""
Builds and generates manifest for Unified Uniform 2x2 (`aniso=False`) Quantum Benchmarks:
  1. `rottor_iso_L2r1` (Uniform 2x2 Boundaryless Torus, K=2): s=2 (646q) vs s=4 (94,774q)
  2. `sq_iso_L2r1` (Uniform 2x2 Planar 3-Level Quantum, K=1): s=1 (19q) -> s=3 (3,379q) -> s=5 (~453k q)
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
sys.path.insert(0, WORK_DIR)
from build_v9_quarter_million import build_chain_job, DATA_DIR


def main():
    specs = [
        ("rottor_iso_L2r1", "torus",  ["X","Z","X","Z"],     [2, 4],    False, ["X_err", "Z_err"], False),
        ("sq_iso_L2r1",     "planar", ["X","Z","X","Z","X"], [1, 3, 5], False, ["X_err", "Z_err"], False),
    ]
    with ProcessPoolExecutor(max_workers=2) as ex:
        for nm in ex.map(build_chain_job, specs):
            print(f"Built unified chain: {nm}", flush=True)

    fast = []
    heavy = []
    # 1. Uniform 2x2 Torus Cycle 1 (s=2) vs Cycle 2 (s=4, 94,774q)
    for b in ("5.50", "6.00", "6.25", "6.50", "6.65", "6.80"):
        fast.append(f"kmc__rottor_iso_L2r1_s2_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/rottor_iso_L2r1_s2_X_err.code|uniform|uf|{b}|120|60000000")
        heavy.append(f"kmc__rottor_iso_L2r1_s4_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/rottor_iso_L2r1_s4_X_err.code|uniform|uf|{b}|20|300000000")
    for b in ("5.50", "6.00"):
        fast.append(f"kmc__rottor_iso_L2r1_s2_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/rottor_iso_L2r1_s2_Z_err.code|uniform|uf|{b}|80|80000000")
        heavy.append(f"kmc__rottor_iso_L2r1_s4_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/rottor_iso_L2r1_s4_Z_err.code|uniform|uf|{b}|16|250000000")

    # 2. Uniform 2x2 Planar 3-Level Quantum (s=1, 3, 5)
    for s, tr in [(1, 160), (3, 64)]:
        for b in ("4.50", "5.00", "5.50", "5.75", "6.00", "6.25", "6.50"):
            fast.append(f"kmc__sq_iso_L2r1_s{s}_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s{s}_Z_err.code|uniform|uf|{b}|{tr}|150000000")
    for b in ("4.50", "5.00"):
        fast.append(f"kmc__sq_iso_L2r1_s3_X_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s3_X_err.code|uniform|uf|{b}|48|150000000")
    for b in ("6.25", "6.35", "6.50"):
        heavy.append(f"kmc__sq_iso_L2r1_s5_Z_err__uniform__uf__b{b}|kmc|{DATA_DIR}/sq_iso_L2r1_s5_Z_err.code|uniform|uf|{b}|16|350000000")
    heavy.append(f"kmc__sq_iso_L2r1_s5_X_err__uniform__uf__b5.00|kmc|{DATA_DIR}/sq_iso_L2r1_s5_X_err.code|uniform|uf|5.00|12|200000000")

    man = os.path.join(WORK_DIR, "manifest_v10u.txt")
    with open(man, "w") as f:
        for i, ln in enumerate(fast):
            f.write(f"{i % 2}|{ln}\n")
        for i, ln in enumerate(heavy):
            f.write(f"{2 + i}|{ln}\n")
    print(f"Wrote {len(fast)} fast (0..1) + {len(heavy)} heavy (2..{1+len(heavy)}) -> total_shards={2+len(heavy)}", flush=True)


if __name__ == "__main__":
    main()
