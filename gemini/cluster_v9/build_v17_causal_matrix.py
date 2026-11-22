#!/usr/bin/env python3
"""
build_v17_causal_matrix.py
Builds exact fine-C3 interior vs matched fine-C3 seam-control masks (.qmask) for R3C3 and C3C3,
records exact mask metadata in `v17_mask_specs.json`, and generates the load-balanced SLURM
manifest `manifest_v17.txt` across three pre-declared suites:

  Suite V17-1: High-Precision Full 3x3 Coarse-by-Fine Operational Matrix (n_X = 128, n_Z = 64)
    at beta = 6.50 on planar (K = 1) and one-face torus (K = 2) seeds + 1-cycle modules,
    decomposing passive recovery into no-crossing survival ('2') + post-crossing healing ('1').

  Suite V17-2: Cold Operational Readout & Marginal Coarse-Bridge Benefit G_B(beta)
    at beta = 6.75 (planar) and beta = 7.00 (torus & planar) across R3C2, C3C2, C2C2, C2R3,
    and contrasting fine-C3 codes R3C3, C3C3, C3R3, R3R3 (n_X = 96, n_Z = 48).

  Suite V17-3: Pre-Declared Causal Marginal-Benefit Test on R3C3 vs C3C3 at beta = 6.50:
    Measuring G_{C3}(mu) = t_0.80(R3C3; mu) / t_0.80(C3C3; mu) (primary) and
    tau_Q(R3C3; mu) / tau_Q(C3C3; mu) (supporting) across pre-declared mu in {1.0, 0.3, 0.1, 0.03, 0.01}
    under identical fine-C3 interior flat-region rate scaling (`fineflat`) vs equal-cardinality
    fine-C3 degree-2 seam controls (`fineseam`), plus all-flat checks (`allflat`, `allseam`).
"""

from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import os
import random
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
DATA_DIR = os.path.join(WORK_DIR, "data")
sys.path.insert(0, WORK_DIR)

from build_v7_rescue import seed_planar_11, sheaf_to_css_with_seams
from build_v15_selective_hierarchy import seed_torus_1face, apply_step_with_parent_faces


def build_fine_and_all_qmasks(spec):
    tag, seed_kind, sched = spec
    comp = seed_planar_11() if seed_kind == "planar" else seed_torus_1face()
    cyl = set()
    c1_edges = set()
    c1_desc3 = set()
    c1_desc4 = set()
    for s_idx, (L, rb, it) in enumerate(sched, 1):
        pcomp = comp
        comp, cyl, meta, _ = apply_step_with_parent_faces(pcomp, L, rb, it, prev_cyl=cyl)
        if s_idx == 2:
            c1_edges = set(range(len(comp.edges)))
        elif s_idx == 3:
            n_xe3 = meta["n_xe"]
            for e2 in c1_edges:
                for eid in meta["edge_sub_eids"][e2]:
                    c1_desc3.add(eid)
                    c1_desc3.add(eid + n_xe3)
        elif s_idx == 4:
            n_xe4 = meta["n_xe"]
            for e3 in c1_desc3:
                for eid in meta["edge_sub_eids"][e3]:
                    c1_desc4.add(eid)
                    c1_desc4.add(eid + n_xe4)

    e2qs = []
    qi = 0
    for e_idx, inc_faces in enumerate(comp.edge_faces):
        d = len(inc_faces)
        col = comp.edge_colors[e_idx]
        if d == 0:
            e2qs.append([])
        elif d == 2 or col == "blue":
            e2qs.append([qi])
            qi += 1
        elif col == "red":
            e2qs.append(list(range(qi, qi + d - 1)))
            qi += d - 1

    N, HX, HZ, seam_qs, *_ = sheaf_to_css_with_seams(comp)
    assert qi == N

    code_p = os.path.join(DATA_DIR, f"{tag}_X_err.code")
    with open(code_p) as f:
        hdr = [int(x) for x in f.readline().split()]
    assert hdr[0] == N and hdr[1] == len(HZ), (tag, hdr, N, len(HZ))

    dz = [0] * N
    for r in HZ:
        for q in r:
            dz[q] += 1

    q_from_c1 = {q for e4 in c1_desc4 for q in e2qs[e4]}

    fine_flat = sorted(q for q in range(N) if (q not in seam_qs and dz[q] == 2 and q not in q_from_c1))
    fine_seam_pool = sorted(q for q in seam_qs if (dz[q] == 2 and q not in q_from_c1))
    all_flat = sorted(q for q in range(N) if (q not in seam_qs and dz[q] == 2))
    all_seam_pool = sorted(q for q in seam_qs if dz[q] == 2)

    rng_fine = random.Random(20261123 + N + (1000 if seed_kind == "tor1" else 0))
    fine_seam_ctrl = sorted(rng_fine.sample(fine_seam_pool, len(fine_flat))) if fine_flat else []

    rng_all = random.Random(20261122 + N + (1000 if seed_kind == "tor1" else 0))
    all_seam_ctrl = sorted(rng_all.sample(all_seam_pool, len(all_flat))) if all_flat else []

    def write_mask(mname, qlist):
        arr = [0] * N
        for q in qlist:
            arr[q] = 1
        mp = os.path.join(DATA_DIR, f"{tag}_{mname}.qmask")
        with open(mp, "w") as out:
            out.write(f"{N}\n" + " ".join(map(str, arr)) + "\n")

    if fine_flat:
        write_mask("fineflat", fine_flat)
        write_mask("fineseam", fine_seam_ctrl)
    write_mask("allflat", all_flat)
    write_mask("allseam", all_seam_ctrl)

    return {
        "tag": tag,
        "seed": seed_kind,
        "N": N,
        "M_Z": len(HZ),
        "fineflat_count": len(fine_flat),
        "fineseam_pool": len(fine_seam_pool),
        "fineseam_ctrl_count": len(fine_seam_ctrl),
        "allflat_count": len(all_flat),
        "coarse_desc_flat_count": len(all_flat) - len(fine_flat),
        "allseam_pool": len(all_seam_pool),
        "allseam_ctrl_count": len(all_seam_ctrl),
        "rng_seed_fine": 20261123 + N + (1000 if seed_kind == "tor1" else 0),
        "rng_seed_all": 20261122 + N + (1000 if seed_kind == "tor1" else 0),
    }


def main():
    C3 = [(3, [1], "X"), (3, [1], "Z")]
    R3 = [(3, [1, 2], "X"), (3, [1, 2], "Z")]

    mask_specs = [
        ("v15_sq_L3_RC",  "planar", R3 + C3),
        ("v15_sq_L3_CC",  "planar", C3 + C3),
        ("v15_sq_L3_CR",  "planar", C3 + R3),
        ("v15_tor_L3_RC", "tor1",   R3 + C3),
        ("v15_tor_L3_CC", "tor1",   C3 + C3),
        ("v15_tor_L3_CR", "tor1",   C3 + R3),
    ]

    meta_out = {"masks": {}}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for info in ex.map(build_fine_and_all_qmasks, mask_specs):
            meta_out["masks"][info["tag"]] = info
            print(
                f"Mask {info['tag']:15s}: fineflat={info['fineflat_count']:7,d} "
                f"fineseam_pool={info['fineseam_pool']:7,d} allflat={info['allflat_count']:7,d}",
                flush=True,
            )

    with open(os.path.join(WORK_DIR, "v17_mask_specs.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    weighted_tasks = []

    def add_kread(tid_prefix, tag, sec, mkind, mu, b, trials, off, lg0, lg1, ng, cap, cost):
        cp = os.path.join(DATA_DIR, f"{tag}_{sec}.code")
        qp = "none" if mkind == "none" else os.path.join(DATA_DIR, f"{tag}_{mkind}.qmask")
        tid = f"{tid_prefix}__{tag}_{sec}__{mkind}__mu{mu:.2f}__b{b:.2f}__o{off}"
        weighted_tasks.append(
            (cost, f"{tid}|kinetic_readout|{cp}|{qp}|{mu:.4f}|{b:.2f}|{trials}|{off}|{lg0:.2f}|{lg1:.2f}|{ng}|{cap}")
        )

    # -------------------------------------------------------------------------
    # Suite V17-1: Full 3x3 Operational Matrix at beta = 6.50 (n_X=128, n_Z=64)
    # -------------------------------------------------------------------------
    for tag in ["v15_sq_C2", "v15_sq_C3", "v15_sq_R3", "v15_tor_C2", "v15_tor_C3", "v15_tor_R3"]:
        for sh in range(2):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 64, 64 * sh, 2.00, 6.60, 47, 600000000, 8.0)
        add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 64, 0, 2.00, 6.60, 47, 600000000, 8.0)

    c2_two_cyc = [
        "v15_sq_pkg_CC", "v16_sq_C2C3", "v15_sq_pkg_CR", "v16_sq_C3C2", "v15_sq_pkg_RC",
        "v15_tor_pkg_CC", "v16_tor_C2C3", "v15_tor_pkg_CR", "v16_tor_C3C2", "v15_tor_pkg_RC",
    ]
    for tag in c2_two_cyc:
        for sh in range(16):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 6.60, 47, 1200000000, 38.0)
        for sh in range(8):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 6.60, 47, 600000000, 14.0)

    l3_two_cyc = [
        "v15_sq_L3_CC", "v15_sq_L3_RC", "v15_sq_L3_CR", "v15_sq_L3_RR",
        "v15_tor_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CR", "v15_tor_L3_RR",
    ]
    for tag in l3_two_cyc:
        for sh in range(16):
            add_kread("m65", tag, "X_err", "none", 1.00, 6.50, 8, 8 * sh, 2.00, 5.60, 37, 600000000, 26.0)
        for sh in range(4):
            add_kread("m65", tag, "Z_err", "none", 1.00, 6.50, 16, 16 * sh, 2.00, 5.60, 37, 400000000, 14.0)

    # -------------------------------------------------------------------------
    # Suite V17-2: Cold Operational Readout at beta = 6.75 and beta = 7.00
    # -------------------------------------------------------------------------
    # 2A. Torus at beta = 7.00 (n_X = 96, n_Z = 48)
    for tag in ["v15_tor_C2", "v15_tor_C3", "v15_tor_R3"]:
        for sh in range(2):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 48, 48 * sh, 2.00, 7.60, 57, 800000000, 10.0)
        add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 48, 0, 2.00, 7.60, 57, 800000000, 8.0)

    for tag in ["v15_tor_pkg_CC", "v16_tor_C3C2", "v15_tor_pkg_RC", "v15_tor_pkg_CR"]:
        for sh in range(12):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 7.60, 57, 1800000000, 55.0)
        for sh in range(3):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 7.60, 57, 600000000, 15.0)

    for tag in ["v15_tor_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CR", "v15_tor_L3_RR"]:
        for sh in range(12):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 6.00, 41, 600000000, 24.0)
        for sh in range(3):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 6.00, 41, 400000000, 12.0)

    # 2B. Planar at beta = 6.75 (n_X = 96, n_Z = 48)
    for tag in ["v15_sq_C2", "v15_sq_C3", "v15_sq_R3"]:
        for sh in range(2):
            add_kread("c675", tag, "X_err", "none", 1.00, 6.75, 48, 48 * sh, 2.00, 7.20, 53, 800000000, 9.0)
        add_kread("c675", tag, "Z_err", "none", 1.00, 6.75, 48, 0, 2.00, 7.20, 53, 800000000, 8.0)

    for tag in ["v15_sq_pkg_CC", "v16_sq_C3C2", "v15_sq_pkg_RC", "v15_sq_pkg_CR"]:
        for sh in range(12):
            add_kread("c675", tag, "X_err", "none", 1.00, 6.75, 8, 8 * sh, 2.00, 7.20, 53, 1600000000, 45.0)
        for sh in range(3):
            add_kread("c675", tag, "Z_err", "none", 1.00, 6.75, 16, 16 * sh, 2.00, 7.20, 53, 600000000, 14.0)

    for tag in ["v15_sq_L3_CC", "v15_sq_L3_RC", "v15_sq_L3_CR", "v15_sq_L3_RR"]:
        for sh in range(12):
            add_kread("c675", tag, "X_err", "none", 1.00, 6.75, 8, 8 * sh, 2.00, 5.80, 39, 600000000, 24.0)
        for sh in range(3):
            add_kread("c675", tag, "Z_err", "none", 1.00, 6.75, 16, 16 * sh, 2.00, 5.80, 39, 400000000, 12.0)

    # 2C. Planar at beta = 7.00 (n_X = 96, n_Z = 48)
    for tag in ["v15_sq_pkg_CC", "v16_sq_C3C2", "v15_sq_pkg_RC", "v15_sq_pkg_CR"]:
        for sh in range(12):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 7.50, 56, 2200000000, 65.0)
        for sh in range(3):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 7.50, 56, 600000000, 15.0)

    for tag in ["v15_sq_L3_CC", "v15_sq_L3_RC"]:
        for sh in range(8):
            add_kread("c70", tag, "X_err", "none", 1.00, 7.00, 8, 8 * sh, 2.00, 6.00, 41, 600000000, 24.0)
        for sh in range(3):
            add_kread("c70", tag, "Z_err", "none", 1.00, 7.00, 16, 16 * sh, 2.00, 6.00, 41, 400000000, 12.0)

    # -------------------------------------------------------------------------
    # Suite V17-3: Decisive Causal Marginal-Benefit Test on R3C3 vs C3C3
    # Pre-declared mu in {1.0, 0.3, 0.1, 0.03, 0.01} (mu = 1.0 identical to m65)
    # -------------------------------------------------------------------------
    for tag in ["v15_sq_L3_RC", "v15_sq_L3_CC", "v15_tor_L3_RC", "v15_tor_L3_CC"]:
        for mkind in ["fineflat", "fineseam"]:
            for mu in [0.30, 0.10, 0.03, 0.01]:
                for sh in range(8):
                    add_kread("kin", tag, "X_err", mkind, mu, 6.50, 8, 8 * sh, 2.00, 5.60, 37, 600000000, 24.0)

    # Secondary check: all-flat vs all-seam on C3C3 and C3R3 at mu in {0.10, 0.01} (n_X = 64)
    for tag in ["v15_sq_L3_CC", "v15_sq_L3_CR", "v15_tor_L3_CC", "v15_tor_L3_CR"]:
        for mkind in ["allflat", "allseam"]:
            for mu in [0.10, 0.01]:
                for sh in range(8):
                    add_kread("kin", tag, "X_err", mkind, mu, 6.50, 8, 8 * sh, 2.00, 5.60, 37, 600000000, 24.0)

    NUM_SHARDS = 96
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
        f"Wrote {man_p}: total_tasks={len(weighted_tasks)} shards={NUM_SHARDS} "
        f"load_min={min(loads):.1f} load_max={max(loads):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
