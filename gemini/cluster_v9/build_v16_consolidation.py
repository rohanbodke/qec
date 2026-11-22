#!/usr/bin/env python3
"""
build_v16_consolidation.py
Builds and verifies all V16 codes, exact Cycle-1 parent-face maps (.c1map),
detailed-balance-preserving zero-energy kinetic masks (.qmask), code specs
(`v16_code_specs.json`), and load-balanced SLURM manifest (`manifest_v16.txt`).

Suites:
  1. Suite V16-1: Withheld Growth-Compatible Comparator Family C3/C2
     (`v16_sq_C3C2`, `v16_sq_C2C3`, `v16_tor_C3C2`, `v16_tor_C2C3`) across beta in [4.50, 7.00].
  2. Suite V16-2: Uncensored High-Precision Span & Lifetime Campaign (n_X = 32)
     on C2/R3 (`pkg_CC`, `pkg_RC`, `pkg_CR`) and L=3 (`L3_CC`, `L3_RC`, `L3_CR`, `L3_RR`)
     recording per-trial stop times, censor flags, coarse arrival times (A2, A6, A10),
     and unbiased first-excursion return indicators.
  3. Suite V16-3: Operational Final-Readout Validation (`final_readout`, n_X = 48, n_Z = 32)
     evolving passively WITHOUT intermediate stopping across prescribed physical readout grids.
  4. Suite V16-4: Detailed-Balance-Preserving Zero-Energy Kinetic Causal Test (`kinetic_zero_e`)
     scaling dE == 0 transitions by mu in {0.01, 0.03, 0.10, 0.30, 1.00, 3.00} on non-seam
     flat-region d_Z=2 qubits (`flat`) vs exact-cardinality d_Z=2 seam controls (`seamctrl`)
     on `sq_L3_RC`, `sq_L3_CR`, `tor_L3_RC`, and `tor_L3_CR` at beta = 6.50.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import os
import random
import sys
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v16")

sys.path.insert(0, WORK_DIR)
from build_v7_rescue import (
    seed_planar_11,
    sheaf_to_css_with_seams,
)
from build_v9_quarter_million import (
    push_X_logical,
    push_Z_logical,
    verify_and_write_code,
)
import build_v9_quarter_million as b9
b9.ENGINE = ENGINE

from build_v15_selective_hierarchy import (
    seed_torus_1face,
    apply_step_with_parent_faces,
)


def build_new_code(spec):
    import build_v9_quarter_million as b9_loc
    b9_loc.ENGINE = ENGINE
    name, seed_kind, sched, aniso = spec
    t0 = time.time()
    if seed_kind == "planar":
        comp = seed_planar_11()
        lx, lz = [{0, 1}], [{0}]
    elif seed_kind == "tor1":
        comp = seed_torus_1face()
        lx, lz = [{0}, {1}], [{0}, {1}]
    else:
        raise ValueError(seed_kind)

    cyl = set()
    pf_maps = []
    f_c1 = 0
    for step_idx, (L, rb, it) in enumerate(sched, 1):
        pcomp = comp
        comp, cyl, meta, c2pf = apply_step_with_parent_faces(
            pcomp, L, rb, it, prev_cyl=(cyl if aniso else None)
        )
        pf_maps.append(c2pf)
        N, HX, HZ, seam_qs, *_ = sheaf_to_css_with_seams(comp)
        lx = [push_X_logical(pcomp, comp, meta, HZ, x) for x in lx]
        lz = [push_Z_logical(pcomp, comp, meta, HX, z) for z in lz]
        if step_idx == 2:
            f_c1 = len(comp.faces)

    verify_and_write_code(N, HX, HZ, lx, lz, name, ["X_err", "Z_err"])

    if len(sched) == 4:
        m3, m4 = pf_maps[2], pf_maps[3]
        c1_map = [m3[m4[f4]] for f4 in range(len(HZ))]
        c1_p = os.path.join(DATA_DIR, f"{name}_X_err.c1map")
        with open(c1_p, "w") as f:
            f.write(f"{len(c1_map)}\n")
            f.write(" ".join(map(str, c1_map)) + "\n")

    qdeg = [0] * N
    for r in HX:
        for q in r:
            qdeg[q] += 1
    for r in HZ:
        for q in r:
            qdeg[q] += 1

    return {
        "name": name,
        "seed": seed_kind,
        "sched": [(L, list(rb), it) for (L, rb, it) in sched],
        "aniso": aniso,
        "N": N,
        "K": len(lx),
        "M_X": len(HX),
        "M_Z": len(HZ),
        "F_C1": f_c1,
        "seam_qubit_fraction": len(seam_qs) / float(N),
        "max_wX": max(len(r) for r in HX),
        "max_wZ": max(len(r) for r in HZ),
        "max_deg": max(qdeg),
        "pushed_wLX": [len(x) for x in lx],
        "pushed_wLZ": [len(z) for z in lz],
        "build_sec": round(time.time() - t0, 2),
    }


def build_qmasks_for_l3(spec):
    tag, seed_kind, sched = spec
    comp = seed_planar_11() if seed_kind == "planar" else seed_torus_1face()
    cyl = set()
    for L, rb, it in sched:
        comp, cyl, _, _ = apply_step_with_parent_faces(comp, L, rb, it, prev_cyl=cyl)
    N, HX, HZ, seam_qs, *_ = sheaf_to_css_with_seams(comp)

    # Verify N and M_Z match the existing .code file on disk:
    code_p = os.path.join(DATA_DIR, f"{tag}_X_err.code")
    with open(code_p) as f:
        hdr = [int(x) for x in f.readline().split()]
    assert hdr[0] == N and hdr[1] == len(HZ), (tag, hdr, N, len(HZ))

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1

    flat_qs = sorted(q for q in range(N) if (q not in seam_qs and dz[q] == 2))
    seam_d2_qs = sorted(q for q in seam_qs if dz[q] == 2)
    assert len(seam_d2_qs) >= len(flat_qs), (tag, len(flat_qs), len(seam_d2_qs))

    rng = random.Random(20261122 + N)
    ctrl_qs = sorted(rng.sample(seam_d2_qs, len(flat_qs)))

    flat_mask = [0] * N
    for q in flat_qs:
        flat_mask[q] = 1
    ctrl_mask = [0] * N
    for q in ctrl_qs:
        ctrl_mask[q] = 1

    p_flat = os.path.join(DATA_DIR, f"{tag}_flat.qmask")
    p_ctrl = os.path.join(DATA_DIR, f"{tag}_seamctrl.qmask")
    with open(p_flat, "w") as f:
        f.write(f"{N}\n" + " ".join(map(str, flat_mask)) + "\n")
    with open(p_ctrl, "w") as f:
        f.write(f"{N}\n" + " ".join(map(str, ctrl_mask)) + "\n")

    return {
        "tag": tag,
        "N": N,
        "flat_dz2_count": len(flat_qs),
        "seam_dz2_pool": len(seam_d2_qs),
        "ctrl_dz2_count": len(ctrl_qs),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    C3 = [(3, [1], "X"), (3, [1], "Z")]
    R3 = [(3, [1, 2], "X"), (3, [1, 2], "Z")]
    C2 = [(2, [1], "X"), (2, [1], "Z")]

    new_specs = [
        ("v16_sq_C3C2",  "planar", C3 + C2, True),
        ("v16_sq_C2C3",  "planar", C2 + C3, True),
        ("v16_tor_C3C2", "tor1",   C3 + C2, True),
        ("v16_tor_C2C3", "tor1",   C2 + C3, True),
    ]

    v16_meta = {"codes": {}, "qmasks": {}}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for info in ex.map(build_new_code, new_specs):
            v16_meta["codes"][info["name"]] = info
            print(f"Built {info['name']:15s}: N={info['N']:7,d} MZ={info['M_Z']:6,d} seam={info['seam_qubit_fraction']:.4f} ({info['build_sec']}s)", flush=True)

    mask_specs = [
        ("v15_sq_L3_RC",  "planar", R3 + C3),
        ("v15_sq_L3_CR",  "planar", C3 + R3),
        ("v15_tor_L3_RC", "tor1",   R3 + C3),
        ("v15_tor_L3_CR", "tor1",   C3 + R3),
    ]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for minfo in ex.map(build_qmasks_for_l3, mask_specs):
            v16_meta["qmasks"][minfo["tag"]] = minfo
            print(f"Mask {minfo['tag']:15s}: flat_dz2={minfo['flat_dz2_count']:7,d} seam_pool={minfo['seam_dz2_pool']:7,d}", flush=True)

    meta_p = os.path.join(WORK_DIR, "v16_code_specs.json")
    with open(meta_p, "w") as f:
        json.dump(v16_meta, f, indent=2)

    weighted_tasks = []

    def add_span_off(tag, b, trials, off, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_X_err.code")
        mp = os.path.join(DATA_DIR, f"{tag}_X_err.c1map")
        weighted_tasks.append(
            (cost, f"spanoff__{tag}_X_err__b{b:.2f}__o{off}|span_off|{cp}|uniform|{mp}|{b:.2f}|{trials}|{off}|{cap}")
        )

    def add_kmc_off(tag, sec, b, trials, off, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        weighted_tasks.append(
            (cost, f"kmcoff__{tag}_{sec}__b{b:.2f}__o{off}|kmc_off|{cp}|uniform|uf|{b:.2f}|{trials}|{off}|{cap}")
        )

    def add_final_readout(tag, sec, b, trials, off, lg_min, lg_max, n_grid, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        weighted_tasks.append(
            (cost, f"frout__{tag}_{sec}__b{b:.2f}__o{off}|final_readout|{cp}|{b:.2f}|{trials}|{off}|{lg_min:.2f}|{lg_max:.2f}|{n_grid}|{cap}")
        )

    def add_kinetic(tag, mkind, mu, b, trials, off, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_X_err.code")
        qp = os.path.join(DATA_DIR, f"{tag}_{mkind}.qmask")
        weighted_tasks.append(
            (cost, f"kin__{tag}__{mkind}__mu{mu:.2f}__b{b:.2f}__o{off}|kinetic_zero_e|{cp}|{qp}|{mu:.4f}|{b:.2f}|{trials}|{off}|{cap}")
        )

    # 1. Suite V16-1: New C3C2 & C2C3 architectures across beta in [4.50..7.00]
    for tag in ["v16_sq_C3C2", "v16_sq_C2C3", "v16_tor_C3C2", "v16_tor_C2C3"]:
        for b in [4.50, 5.00, 5.50, 5.75, 6.00, 6.25]:
            add_span_off(tag, b, 24, 0, 300000000, 12.0 if b <= 5.75 else 25.0)
            add_kmc_off(tag, "Z_err", b, 16, 0, 100000000, 10.0)
        for b in [6.50, 6.75, 7.00]:
            for sh in range(8):
                add_span_off(tag, b, 4, 4 * sh, 1400000000, 45.0 if b < 7.00 else 80.0)
            for sh in range(2):
                add_kmc_off(tag, "Z_err", b, 8, 8 * sh, 300000000, 25.0)

    # 2. Suite V16-2A: Uncensored n_X = 32 Span & Lifetime on C2/R3 (Planar & Torus, plus Rotated Torus at 6.50..7.00)
    for tag in [
        "v15_sq_pkg_CC", "v15_sq_pkg_RC", "v15_sq_pkg_CR",
        "v15_tor_pkg_CC", "v15_tor_pkg_RC", "v15_tor_pkg_CR",
    ]:
        for b in [6.00, 6.25]:
            for sh in range(2):
                add_span_off(tag, b, 16, 16 * sh, 400000000, 20.0)
        for b in [6.50, 6.75, 7.00]:
            for sh in range(8):
                add_span_off(tag, b, 4, 4 * sh, 1600000000, 55.0 if b < 7.00 else 110.0)
            for sh in range(2):
                add_kmc_off(tag, "Z_err", b, 8, 8 * sh, 350000000, 30.0)

    for tag in ["v15_rottor_pkg_RC", "v15_rottor_pkg_CR"]:
        for b in [6.50, 6.75, 7.00]:
            for sh in range(8):
                add_span_off(tag, b, 4, 4 * sh, 1600000000, 65.0 if b < 7.00 else 120.0)

    # 2B. Suite V16-2B: n_X = 32 Span & Return Statistics on L=3 (Planar & Torus)
    for tag in [
        "v15_sq_L3_CC", "v15_sq_L3_RC", "v15_sq_L3_CR", "v15_sq_L3_RR",
        "v15_tor_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CR", "v15_tor_L3_RR",
    ]:
        for b in [6.00, 6.50, 6.75, 7.00]:
            for sh in range(4):
                add_span_off(tag, b, 8, 8 * sh, 400000000, 35.0 if b <= 6.50 else 65.0)

    # 3. Suite V16-3: True Operational Final-Readout Validation (`final_readout`, beta = 6.50)
    # 3A. 1-cycle modules (fast: 64 X trials, 64 Z trials up to t = 10^6.6)
    for tag in ["v15_sq_C2", "v15_sq_C3", "v15_sq_R3", "v15_tor_C2", "v15_tor_C3", "v15_tor_R3"]:
        add_final_readout(tag, "X_err", 6.50, 64, 0, 2.00, 6.60, 47, 500000000, 8.0)
        add_final_readout(tag, "Z_err", 6.50, 64, 0, 2.00, 6.60, 47, 500000000, 8.0)

    # 3B. Unit-patch C2/R3 + Growth-compatible C3/C2 (48 X trials, 32 Z trials up to t = 10^6.6 = 3.98e6)
    for tag in [
        "v15_sq_pkg_CC", "v15_sq_pkg_RC", "v15_sq_pkg_CR", "v16_sq_C3C2", "v16_sq_C2C3",
        "v15_tor_pkg_CC", "v15_tor_pkg_RC", "v15_tor_pkg_CR", "v16_tor_C3C2", "v16_tor_C2C3",
    ]:
        for sh in range(8):
            add_final_readout(tag, "X_err", 6.50, 6, 6 * sh, 2.00, 6.60, 47, 900000000, 45.0)
        for sh in range(4):
            add_final_readout(tag, "Z_err", 6.50, 8, 8 * sh, 2.00, 6.60, 47, 500000000, 20.0)

    # 3C. Matched L=3 CC, RC, CR, RR (48 X trials, 32 Z trials up to t = 10^5.4 = 2.51e5)
    for tag in [
        "v15_sq_L3_CC", "v15_sq_L3_RC", "v15_sq_L3_CR", "v15_sq_L3_RR",
        "v15_tor_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CR", "v15_tor_L3_RR",
    ]:
        for sh in range(8):
            add_final_readout(tag, "X_err", 6.50, 6, 6 * sh, 2.00, 5.40, 35, 600000000, 35.0)
        for sh in range(4):
            add_final_readout(tag, "Z_err", 6.50, 8, 8 * sh, 2.00, 5.40, 35, 300000000, 15.0)

    # 4. Suite V16-4: Detailed-Balance-Preserving Zero-Energy Kinetic Causal Test (beta = 6.50)
    for tag in ["v15_sq_L3_RC", "v15_sq_L3_CR", "v15_tor_L3_RC", "v15_tor_L3_CR"]:
        for mkind in ["flat", "seamctrl"]:
            for mu in [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]:
                for sh in range(3):
                    add_kinetic(tag, mkind, mu, 6.50, 8, 8 * sh, 500000000, 35.0)

    # Greedy min-heap load balancing across NUM_SHARDS = 144 workers:
    NUM_SHARDS = 144
    weighted_tasks.sort(key=lambda x: -x[0])
    heap = [(0.0, s) for s in range(NUM_SHARDS)]
    heapq.heapify(heap)
    assigned = [[] for _ in range(NUM_SHARDS)]
    for cost, tstr in weighted_tasks:
        c_tot, sh = heapq.heappop(heap)
        assigned[sh].append(tstr)
        heapq.heappush(heap, (c_tot + cost, sh))

    man_p = os.path.join(WORK_DIR, "manifest_v16.txt")
    with open(man_p, "w") as f:
        for sh in range(NUM_SHARDS):
            for tstr in assigned[sh]:
                f.write(f"{sh}|{tstr}\n")

    loads = [c for c, _ in heap]
    print(
        f"Wrote {man_p}: {len(weighted_tasks)} tasks across {NUM_SHARDS} shards "
        f"(est load min={min(loads):.1f}, max={max(loads):.1f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
