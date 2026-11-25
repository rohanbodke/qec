#!/usr/bin/env python3
"""
parse_v19_rebalance_and_decoders.py
Zero-trust parser & 10,000-draw paired/independent bootstrap analyzer for the V19 campaign:
  1. Direction 1 (Sector-Rebalanced & Palindromic Hierarchies):
     - L=2 4-step Welding-Order Quartet (`XZXZ`, `ZXZX`, `XZZX`, `ZXXZ`) on Planar (K=1) and
       symmetric 1-face Torus (K=2) across beta in {5.50, 6.00, 6.50, 7.00}, including empirical
       Arrhenius slopes d(ln t_0.80)/d(beta) and sector-bottleneck ratios t_80^X / t_80^Z.
     - 3D-continuable coarse-L=3 palindromic codes (`C3_C2ZX`, `RX3_C2ZX`, `RZ3_C2ZX`, `R3_C2ZX`)
       versus standard `(X,Z,X,Z)` codes (`C3_C2`, `RX3_C2`, `R3_C2`) at beta = 6.50 and 7.00.
     - Ultra-compact 3-step balanced `(X, Z, X)` family (`N = 1,942..10,580`), testing the
       mirror-image bottleneck prediction (`X2_Z3r2_X2` vs matched-size `X3r2_Z2_X2`).
  2. Direction 2 (Thermal Basin Crossing vs. Classical Decoder Suboptimality):
     - Paired trajectory-by-trajectory comparison of `UF`, `Cool-UF`, `Hier-UF`, and `BP-OSD`
       on IDENTICAL thermal BKL trajectories (exact common-random-numbers paired bootstrap CIs!).
     - Calibration of `UF` and `BP-OSD` on 1-step `C2` (N=19, beta=7.00) against the exact
       analytical 8,192-state Markov maximum-likelihood curve P^opt(t).
     - Microscopic failure decomposition at t_0.80^UF: exact zero-syndrome wrong ground states ('G'),
       T=0 downhill quench-settled wrong-basin states ('Q'), excited failures rescued by BP-OSD,
       and excited failures unrecovered by all four decoders.
     - Unrelaxed i.i.d. bit-flip error-rate sweeps (`iid_readout`) contrasting dense-syndrome
       decoder thresholds p_0.80 across `UF`, `Hier-UF`, `Cool-UF`, and `BP-OSD`.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V19_RAW = ROOT / "cluster_v19_results.txt"
V19_SPECS = ROOT / "v19_code_specs.json"
LOC_QUOT = ROOT / "local_quotient_and_pruning_data.json"
OUT_JSON = ROOT / "parsed_v19_rebalance_and_decoders.json"

DECS = ("uf", "cool", "hier", "bposd")
PASS_CHARS = ("1", "2")
FAIL_CHARS = ("0", "Q", "G")


def parse_kv(body):
    out = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def crossing_time_log(x_grid, p_curve, thresh=0.80, log_x=True):
    for g in range(len(x_grid)):
        if p_curve[g] < thresh:
            if g == 0:
                return float(x_grid[0]), True
            p0, p1 = float(p_curve[g - 1]), float(p_curve[g])
            if p0 <= p1:
                return float(x_grid[g]), True
            frac = (p0 - thresh) / (p0 - p1)
            if log_x:
                lg0, lg1 = math.log10(x_grid[g - 1]), math.log10(x_grid[g])
                return float(10.0 ** (lg0 + frac * (lg1 - lg0))), True
            return float(x_grid[g - 1] + frac * (x_grid[g] - x_grid[g - 1])), True
    return float(x_grid[-1]), False


def eval_paired_arm(x_map, z_map, grid_spec, rng, B=5000):
    lg0_s, lg1_s, ng_s = grid_spec.split(":")
    lg0, lg1_full, ng_full = float(lg0_s), float(lg1_s), int(ng_s)
    step = (lg1_full - lg0) / max(1, ng_full - 1)

    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    ng = min(
        [ng_full]
        + [len(x_map[k]["uf"]) for k in x_keys]
        + [len(z_map[k]["uf"]) for k in z_keys]
    )
    lg1 = lg0 + step * (ng - 1)
    t_grid = np.array([10.0 ** (lg0 + step * i) for i in range(ng)], dtype=np.float64)

    nx, nz = len(x_keys), len(z_keys)
    cap_x = sum(1 for k in x_keys if "C" in x_map[k]["uf"][:ng])
    cap_z = sum(1 for k in z_keys if "C" in z_map[k]["uf"][:ng])

    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))

    dec_stats = {}
    b_t80_by_dec = {}
    pass_mats = {}

    for d in DECS:
        X_pass = np.array(
            [[1.0 if ch in PASS_CHARS else 0.0 for ch in x_map[k][d][:ng]] for k in x_keys],
            dtype=np.float64,
        )
        Z_pass = np.array(
            [[1.0 if ch in PASS_CHARS else 0.0 for ch in z_map[k][d][:ng]] for k in z_keys],
            dtype=np.float64,
        )
        pass_mats[d] = (X_pass, Z_pass)
        px = X_pass.mean(axis=0)
        pz = Z_pass.mean(axis=0)
        pq = px * pz

        t80_q, cr_q = crossing_time_log(t_grid, pq, 0.80, True)
        t70_q, cr_70 = crossing_time_log(t_grid, pq, 0.70, True)
        t90_q, cr_90 = crossing_time_log(t_grid, pq, 0.90, True)
        t80_x, cr_x = crossing_time_log(t_grid, px, 0.80, True)
        t80_z, cr_z = crossing_time_log(t_grid, pz, 0.80, True)

        b_pq = X_pass[ix].mean(axis=1) * Z_pass[iz].mean(axis=1)
        b_t80 = np.array([crossing_time_log(t_grid, b_pq[b], 0.80, True)[0] for b in range(B)])
        b_t80_by_dec[d] = b_t80

        dec_stats[d] = {
            "t_80": t80_q,
            "t_80_crossed": cr_q,
            "t_80_ci95": [float(np.percentile(b_t80, 2.5)), float(np.percentile(b_t80, 97.5))],
            "t_70": t70_q,
            "t_70_crossed": cr_70,
            "t_90": t90_q,
            "t_90_crossed": cr_90,
            "t_80_X": t80_x,
            "t_80_X_crossed": cr_x,
            "t_80_Z": t80_z,
            "t_80_Z_crossed": cr_z,
            "p_X": [float(v) for v in px],
            "p_Z": [float(v) for v in pz],
            "p_Q": [float(v) for v in pq],
        }

    # Exact trajectory-paired decoder ratios vs baseline UF (common random numbers!):
    paired_vs_uf = {}
    for d in ("cool", "hier", "bposd"):
        pt = float(dec_stats[d]["t_80"] / max(1e-30, dec_stats["uf"]["t_80"]))
        br = b_t80_by_dec[d] / np.maximum(1e-30, b_t80_by_dec["uf"])
        paired_vs_uf[d] = {
            "ratio": pt,
            "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
        }

    # Microscopic failure-mode anatomy near t_80(UF) in the limiting sector(s):
    idx_uf = min(range(ng), key=lambda i: abs(math.log10(t_grid[i]) - math.log10(dec_stats["uf"]["t_80"])))
    window = sorted({max(0, min(ng - 1, idx_uf + dg)) for dg in (-1, 0, 1)})
    tot_uf_fail = 0
    cnt_G = 0
    cnt_Q = 0
    cnt_rescued_bp = 0
    cnt_excited_both_fail = 0
    for sec_map in (x_map, z_map):
        for k in sec_map:
            s_uf = sec_map[k]["uf"]
            s_bp = sec_map[k]["bposd"]
            for g in window:
                c_u = s_uf[g]
                if c_u in FAIL_CHARS:
                    tot_uf_fail += 1
                    if c_u == "G":
                        cnt_G += 1
                    elif c_u == "Q":
                        cnt_Q += 1
                    elif s_bp[g] in PASS_CHARS:
                        cnt_rescued_bp += 1
                    else:
                        cnt_excited_both_fail += 1

    denom = max(1, tot_uf_fail)
    anatomy = {
        "total_uf_sector_fails_near_t80": tot_uf_fail,
        "frac_zero_syn_G": cnt_G / denom,
        "frac_quench_settled_Q": cnt_Q / denom,
        "frac_settled_wrong_basin_GQ": (cnt_G + cnt_Q) / denom,
        "frac_rescued_by_bposd": cnt_rescued_bp / denom,
        "frac_excited_unrescued": cnt_excited_both_fail / denom,
    }

    return {
        "n_X": nx,
        "n_Z": nz,
        "capped_X": cap_x,
        "capped_Z": cap_z,
        "grid_spec": f"{lg0:.4f}:{lg1:.4f}:{ng}",
        "t_grid": [float(v) for v in t_grid],
        "decoders": dec_stats,
        "paired_decoder_ratio_vs_uf": paired_vs_uf,
        "failure_anatomy_near_t80": anatomy,
        "_b_t80_uf": b_t80_by_dec["uf"],
        "_b_t80_bposd": b_t80_by_dec["bposd"],
    }


def eval_iid_arm(x_map, z_map, pgrid_spec, rng, B=5000):
    p0_s, p1_s, ng_s = pgrid_spec.split(":")
    p0, p1, ng = float(p0_s), float(p1_s), int(ng_s)
    p_grid = np.array([p0 + (p1 - p0) * i / max(1, ng - 1) for i in range(ng)], dtype=np.float64)

    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    nx, nz = len(x_keys), len(z_keys)
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))

    out = {"n_X": nx, "n_Z": nz, "p_grid": [float(v) for v in p_grid], "decoders": {}, "paired_ratio_vs_uf": {}}
    b_p80 = {}
    for d in DECS:
        Xp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in x_map[k][d][:ng]] for k in x_keys])
        Zp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in z_keys_str(z_map, k, d, ng)] for k in z_keys])
        px = Xp.mean(axis=0)
        pz = Zp.mean(axis=0)
        pq = px * pz
        p80_q, cr_q = crossing_time_log(p_grid, pq, 0.80, False)
        p80_x, _ = crossing_time_log(p_grid, px, 0.80, False)
        p80_z, _ = crossing_time_log(p_grid, pz, 0.80, False)

        bpq = Xp[ix].mean(axis=1) * Zp[iz].mean(axis=1)
        bp = np.array([crossing_time_log(p_grid, bpq[b], 0.80, False)[0] for b in range(B)])
        b_p80[d] = bp
        out["decoders"][d] = {
            "p_80_Q": p80_q,
            "p_80_crossed": cr_q,
            "p_80_ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
            "p_80_X": p80_x,
            "p_80_Z": p80_z,
            "curve_Q": [float(v) for v in pq],
            "curve_X": [float(v) for v in px],
            "curve_Z": [float(v) for v in pz],
        }

    for d in ("cool", "hier", "bposd"):
        pt = float(out["decoders"][d]["p_80_Q"] / max(1e-12, out["decoders"]["uf"]["p_80_Q"]))
        br = b_p80[d] / np.maximum(1e-12, b_p80["uf"])
        out["paired_ratio_vs_uf"][d] = {
            "ratio": pt,
            "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
        }
    return out


def z_keys_str(z_map, k, d, ng):
    return z_map[k][d][:ng]


def indep_ratio(a_num, a_den, key="_b_t80_uf", dec="uf"):
    pt = float(a_num["decoders"][dec]["t_80"] / max(1e-30, a_den["decoders"][dec]["t_80"]))
    br = a_num[key] / np.maximum(1e-30, a_den[key])
    return {"ratio": pt, "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}


def main():
    with open(V19_SPECS) as f:
        code_specs = json.load(f)

    paired_trajs = defaultdict(dict)  # (tag, sec, b_s) -> {gtr: {d: bitstr}}
    paired_grids = {}
    iid_trajs = defaultdict(dict)     # (tag, sec) -> {gtr: {d: bitstr}}
    iid_grids = {}
    errors = []

    with open(V19_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            kv = parse_kv(body)
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^([a-z0-9]+)__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                if not m:
                    errors.append((tid, "UNPARSED_PAIRED"))
                    continue
                _, tag, sec, b_s, off_s = m.groups()
                off = int(off_s)
                g_str = kv["grid"]
                if (tag, b_s) not in paired_grids or int(g_str.split(":")[2]) > int(paired_grids[(tag, b_s)].split(":")[2]):
                    paired_grids[(tag, b_s)] = g_str
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                ntr = len(split_d["uf"])
                for i in range(ntr):
                    gtr = off + i
                    if gtr not in paired_trajs[(tag, sec, b_s)] or len(split_d["uf"][i]) > len(paired_trajs[(tag, sec, b_s)][gtr]["uf"]):
                        paired_trajs[(tag, sec, b_s)][gtr] = {d: split_d[d][i] for d in DECS}
            elif body.startswith("IID_READOUT"):
                m = re.match(r"^([a-z0-9]+)__(.+)_(X_err|Z_err)__iid__o(\d+)$", tid)
                if not m:
                    errors.append((tid, "UNPARSED_IID"))
                    continue
                _, tag, sec, off_s = m.groups()
                off = int(off_s)
                iid_grids[tag] = kv["pgrid"]
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                ntr = len(split_d["uf"])
                for i in range(ntr):
                    gtr = off + i
                    iid_trajs[(tag, sec)][gtr] = {d: split_d[d][i] for d in DECS}

    print(f"Loaded V19 results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112401)
    B = 10000
    arms = {}

    for tag, b_s in sorted({(t, b) for (t, _, b) in paired_trajs}):
        xm = paired_trajs.get((tag, "X_err", b_s), {})
        zm = paired_trajs.get((tag, "Z_err", b_s), {})
        if not xm:
            continue
        # Note: 1-step X-welded C2 (`v19_sq_C2_1step`) has HX = 6, HZ = 12 -> both X_err and Z_err exist!
        if not zm:
            continue
        key = f"{tag}__b{b_s}"
        arms[key] = eval_paired_arm(xm, zm, paired_grids[(tag, b_s)], rng, B=B)
        arms[key]["N"] = code_specs[tag]["N"]
        arms[key]["K"] = code_specs[tag]["K"]
        u = arms[key]["decoders"]["uf"]
        bp = arms[key]["decoders"]["bposd"]
        anat = arms[key]["failure_anatomy_near_t80"]
        r_bp = arms[key]["paired_decoder_ratio_vs_uf"]["bposd"]
        print(
            f"  {key:30s}: N={arms[key]['N']:7,d} n=({arms[key]['n_X']},{arms[key]['n_Z']}) cap=({arms[key]['capped_X']},{arms[key]['capped_Z']}) "
            f"| t80_UF={u['t_80']:10.1f}{'' if u['t_80_crossed'] else '+'} (X={u['t_80_X']:9.1f}, Z={u['t_80_Z']:9.1f}) "
            f"| BPOSD/UF={r_bp['ratio']:.3f} [{r_bp['ci95'][0]:.3f},{r_bp['ci95'][1]:.3f}] "
            f"| settled_GQ={anat['frac_settled_wrong_basin_GQ']:.2f}"
        )

    iid_arms = {}
    for tag in sorted({t for (t, _) in iid_trajs}):
        xm = iid_trajs.get((tag, "X_err"), {})
        zm = iid_trajs.get((tag, "Z_err"), {})
        if not xm or not zm:
            continue
        iid_arms[tag] = eval_iid_arm(xm, zm, iid_grids[tag], rng, B=B)
        iid_arms[tag]["N"] = code_specs[tag]["N"]
        du = iid_arms[tag]["decoders"]["uf"]["p_80_Q"]
        dc = iid_arms[tag]["decoders"]["cool"]["p_80_Q"]
        dh = iid_arms[tag]["decoders"]["hier"]["p_80_Q"]
        db = iid_arms[tag]["decoders"]["bposd"]["p_80_Q"]
        rb = iid_arms[tag]["paired_ratio_vs_uf"]["bposd"]
        print(
            f"  [IID] {tag:24s}: p80 UF={du:.4f} | Hier={dh:.4f} | Cool={dc:.4f} | BPOSD={db:.4f} "
            f"(BPOSD/UF = {rb['ratio']:.2f}x [{rb['ci95'][0]:.2f}, {rb['ci95'][1]:.2f}])"
        )

    # Key architectural contrasts (Direction 1):
    contrasts = {}
    contrast_specs = [
        # 3-step mirror-image bottleneck test & comparison vs C2C2:
        ("3s_X2Z3r2X2_over_X3r2Z2X2_b65", "v19_sq_3s_X2Z3r2X2__b6.50", "v19_sq_3s_X3r2Z2X2__b6.50"),
        ("3s_X2Z3r2X2_over_X2Z3r1X2_b65", "v19_sq_3s_X2Z3r2X2__b6.50", "v19_sq_3s_X2Z3r1X2__b6.50"),
        ("3s_X2Z3r2X2_over_C2XZXZ_b65",   "v19_sq_3s_X2Z3r2X2__b6.50", "v19_sq_C2_XZXZ__b6.50"),
        ("3s_X3r2Z3r2X2_over_C2XZXZ_b65", "v19_sq_3s_X3r2Z3r2X2__b6.50", "v19_sq_C2_XZXZ__b6.50"),
        ("3s_X2Z3r2X2_over_X3r2Z2X2_b70", "v19_sq_3s_X2Z3r2X2__b7.00", "v19_sq_3s_X3r2Z2X2__b7.00"),
        # 4-step Welding-Order Quartet vs standard XZXZ (Planar & Torus, beta=6.5 & 7.0):
        ("sq_ZXZX_over_XZXZ_b65",         "v19_sq_C2_ZXZX__b6.50",     "v19_sq_C2_XZXZ__b6.50"),
        ("sq_XZZX_over_XZXZ_b65",         "v19_sq_C2_XZZX__b6.50",     "v19_sq_C2_XZXZ__b6.50"),
        ("sq_ZXXZ_over_XZXZ_b65",         "v19_sq_C2_ZXXZ__b6.50",     "v19_sq_C2_XZXZ__b6.50"),
        ("sq_ZXXZ_over_XZXZ_b70",         "v19_sq_C2_ZXXZ__b7.00",     "v19_sq_C2_XZXZ__b7.00"),
        ("tor_ZXZX_over_XZXZ_b65",        "v19_tor_C2_ZXZX__b6.50",    "v19_tor_C2_XZXZ__b6.50"),
        ("tor_XZZX_over_XZXZ_b65",        "v19_tor_C2_XZZX__b6.50",    "v19_tor_C2_XZXZ__b6.50"),
        ("tor_ZXXZ_over_XZXZ_b65",        "v19_tor_C2_ZXXZ__b6.50",    "v19_tor_C2_XZXZ__b6.50"),
        ("sq_XZZX_over_XZXZ_b70",         "v19_sq_C2_XZZX__b7.00",     "v19_sq_C2_XZXZ__b7.00"),
        ("tor_XZZX_over_XZXZ_b70",        "v19_tor_C2_XZZX__b7.00",    "v19_tor_C2_XZXZ__b7.00"),
        # 3D-continuable Coarse-L=3 palindromic (XZZX) vs standard (XZXZ) & mirror-image coarse bridge test:
        ("RZ3C2ZX_over_RX3C2ZX_b60",      "v19_sq_RZ3_C2ZX__b6.00",    "v19_sq_RX3_C2ZX__b6.00"),
        ("RZ3C2ZX_over_C3C2ZX_b60",       "v19_sq_RZ3_C2ZX__b6.00",    "v19_sq_C3_C2ZX__b6.00"),
        ("RZ3C2ZX_over_RX3C2ZX_b65",      "v19_sq_RZ3_C2ZX__b6.50",    "v19_sq_RX3_C2ZX__b6.50"),
        ("C3C2ZX_over_C3C2_b65",          "v19_sq_C3_C2ZX__b6.50",     "v19_sq_C3_C2__b6.50"),
        ("RX3C2ZX_over_RX3C2_b65",        "v19_sq_RX3_C2ZX__b6.50",    "v19_sq_RX3_C2__b6.50"),
        ("RZ3C2ZX_over_RX3C2_b65",        "v19_sq_RZ3_C2ZX__b6.50",    "v19_sq_RX3_C2__b6.50"),
        ("RX3C2ZX_over_R3C2_b65",         "v19_sq_RX3_C2ZX__b6.50",    "v19_sq_R3_C2__b6.50"),
        ("C3C2ZX_over_C3C2_b70",          "v19_sq_C3_C2ZX__b7.00",     "v19_sq_C3_C2__b7.00"),
        ("RX3C2ZX_over_RX3C2_b70",        "v19_sq_RX3_C2ZX__b7.00",    "v19_sq_RX3_C2__b7.00"),
        ("RX3C2ZX_over_R3C2_b70",         "v19_sq_RX3_C2ZX__b7.00",    "v19_sq_R3_C2__b7.00"),
    ]
    for cname, k_num, k_den in contrast_specs:
        if k_num in arms and k_den in arms:
            contrasts[cname] = {
                "uf": indep_ratio(arms[k_num], arms[k_den], "_b_t80_uf", "uf"),
                "bposd": indep_ratio(arms[k_num], arms[k_den], "_b_t80_bposd", "bposd"),
            }
            ru = contrasts[cname]["uf"]
            print(f"  [{cname:30s}]: UF ratio = {ru['ratio']:8.2f}x [{ru['ci95'][0]:.2f}, {ru['ci95'][1]:.2f}]")

    # Strip internal bootstrap arrays before JSON dump:
    clean_arms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()}
    out_doc = {
        "errors": errors,
        "arms": clean_arms,
        "iid_arms": iid_arms,
        "contrasts": contrasts,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
