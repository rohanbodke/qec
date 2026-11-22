#!/usr/bin/env python3
"""
update_manifest_v17.py
Generates `manifest_v17.txt` with exact 0.10-decade common readout grids (10^2.0, 10^2.1, ...)
stopping cleanly past t_0.80 and t_0.50 (avoiding giant-cluster percolation OOMs), preserving
all completed tasks in `results_v17/*.txt`.
"""

import glob
import heapq
import os

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results_v17")


def load_done():
    done = set()
    for p in glob.glob(os.path.join(RES_DIR, "*.txt")):
        with open(p) as f:
            for raw in f:
                if "|" in raw and "KINETIC_READOUT" in raw:
                    done.add(raw.split("|", 1)[0].strip())
    return done


def ng_for(lg0, lg1):
    return int(round((lg1 - lg0) * 10.0)) + 1


def main():
    done = load_done()
    weighted_tasks = []

    def add_kread(tid_prefix, tag, sec, mkind, mu, b, trials, off, lg0, lg1, cap, cost):
        tid = f"{tid_prefix}__{tag}_{sec}__{mkind}__mu{mu:.2f}__b{b:.2f}__o{off}"
        if tid in done:
            return
        ng = ng_for(lg0, lg1)
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        qp = "none" if mkind == "none" else os.path.join(DATA_DIR, f"{tag}_{mkind}.qmask")
        weighted_tasks.append(
            (cost, f"{tid}|kinetic_readout|{cp}|{qp}|{mu:.4f}|{b:.2f}|{trials}|{off}|{lg0:.2f}|{lg1:.2f}|{ng}|{cap}")
        )

    # -------------------------------------------------------------------------
    # Suite V17-1: Full 3x3 Operational Matrix at beta = 6.50
    # (Combined with V16's 48 X + 32 Z trials -> n_X >= 96, n_Z >= 64 everywhere,
    #  plus >= 48 full ternary crossing/healing trajectories per code!)
    # -------------------------------------------------------------------------
    for tag in ["v15_sq_C2", "v15_sq_C3", "v15_sq_R3", "v15_tor_C2", "v15_tor_C3", "v15_tor_R3"]:
        add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 64, 0, 2.00, 5.60, 300000000, 3.0)
        add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 64, 0, 2.00, 5.60, 300000000, 3.0)

    for tag in ["v16_sq_C2C3", "v16_tor_C2C3"]:
        for sh in range(6):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 4.40, 200000000, 3.0)
        for sh in range(2):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 4.40, 200000000, 2.0)

    for tag in ["v15_sq_pkg_CR", "v15_tor_pkg_CR"]:
        for sh in range(6):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 5.00, 300000000, 5.0)
        for sh in range(2):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 5.00, 200000000, 3.0)

    for tag in ["v15_sq_pkg_CC", "v16_sq_C3C2", "v15_sq_pkg_RC", "v15_tor_pkg_CC", "v16_tor_C3C2", "v15_tor_pkg_RC"]:
        for sh in range(8):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 5.80, 500000000, 9.0)
        for sh in range(2):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 5.80, 300000000, 4.0)

    for tag in ["v15_sq_L3_CC", "v15_sq_L3_RC", "v15_tor_L3_CC", "v15_tor_L3_RC"]:
        for sh in range(8):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 4.40, 250000000, 4.0)
        for sh in range(2):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 4.80, 250000000, 4.0)

    for tag in ["v15_sq_L3_CR", "v15_sq_L3_RR", "v15_tor_L3_CR", "v15_tor_L3_RR"]:
        for sh in range(6):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 4.70, 300000000, 7.0)
        for sh in range(2):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 4.70, 250000000, 4.0)

    # -------------------------------------------------------------------------
    # Suite V17-2: Cold Operational Readout at beta = 7.00 (Torus & Planar)
    # -------------------------------------------------------------------------
    for tag in ["v15_tor_C2", "v15_tor_C3", "v15_tor_R3"]:
        add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 64, 0, 2.00, 6.80, 400000000, 4.0)
        add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 64, 0, 2.00, 6.80, 400000000, 3.0)

    for tag in ["v15_tor_pkg_CC", "v16_tor_C3C2", "v15_tor_pkg_RC"]:
        for sh in range(8):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 7.00, 800000000, 16.0)
        for sh in range(2):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 7.00, 400000000, 5.0)

    for sh in range(6):
        add_kread("c70", "v15_tor_pkg_CR", "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 6.00, 400000000, 6.0)
    for sh in range(2):
        add_kread("c70", "v15_tor_pkg_CR", "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 6.00, 300000000, 4.0)

    for tag in ["v15_tor_L3_CC", "v15_tor_L3_RC"]:
        for sh in range(6):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 4.80, 250000000, 5.0)
        for sh in range(2):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 4.80, 250000000, 4.0)

    # Planar beta = 7.00 (note: sq_pkg_CC, sq_C3C2, sq_pkg_RC already have all 12 X shards = 96 trials done!)
    for tag in ["v15_sq_pkg_CC", "v16_sq_C3C2", "v15_sq_pkg_RC"]:
        for sh in range(12):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 7.00, 1000000000, 18.0)
        for sh in range(2):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 7.20, 400000000, 5.0)

    for sh in range(6):
        add_kread("c70", "v15_sq_pkg_CR", "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 6.20, 400000000, 8.0)
    for sh in range(2):
        add_kread("c70", "v15_sq_pkg_CR", "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 6.20, 300000000, 4.0)

    for tag in ["v15_sq_L3_CC", "v15_sq_L3_RC"]:
        for sh in range(6):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 4.80, 250000000, 5.0)
        for sh in range(2):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 4.80, 250000000, 4.0)

    # -------------------------------------------------------------------------
    # Suite V17-3: Decisive Causal Marginal-Benefit Test on R3C3 vs C3C3
    # Pre-declared mu in {1.0, 0.3, 0.1, 0.03, 0.01} (n_X = 48 passive trajectories)
    # -------------------------------------------------------------------------
    for tag in ["v15_sq_L3_RC", "v15_sq_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CC"]:
        for mkind in ["fineflat", "fineseam"]:
            for mu, lg_end in [(0.30, 4.40), (0.10, 4.50), (0.03, 4.70), (0.01, 4.80)]:
                for sh in range(6):
                    add_kread("kin", tag, "X_err", mkind, mu, 6.50, 8, 8 * sh, 2.00, lg_end, 300000000, 5.0)

    for tag in ["v15_sq_L3_CC", "v15_sq_L3_CR", "v15_tor_L3_CC", "v15_tor_L3_CR"]:
        for mkind in ["allflat", "allseam"]:
            for mu, lg_end in [(0.10, 4.60), (0.01, 4.80)]:
                for sh in range(6):
                    add_kread("kin", tag, "X_err", mkind, mu, 6.50, 8, 8 * sh, 2.00, lg_end, 300000000, 6.0)

    NUM_SHARDS = 64
    weighted_tasks.sort(key=lambda x: -x[0])
    heap = [(0.0, s) for s in range(NUM_SHARDS)]
    heapq.heapify(heap)
    assigned = [[] for _ in range(NUM_SHARDS)]
    for cost, tstr in weighted_tasks:
        c_tot, sh = heapq.heappop(heap)
        assigned[sh].append(tstr)
        heapq.heappush(heap, (c_tot + cost, sh))

    man_p = os.path.join(WORK_DIR, "manifest_v17.txt")
    with open(man_p, "w") as f:
        for sh in range(NUM_SHARDS):
            for tstr in assigned[sh]:
                f.write(f"{sh}|{tstr}\n")

    loads = [c for (c, _) in heap]
    print(
        f"Already done={len(done)}, remaining={len(weighted_tasks)} across {NUM_SHARDS} shards "
        f"(load {min(loads):.1f}..{max(loads):.1f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
