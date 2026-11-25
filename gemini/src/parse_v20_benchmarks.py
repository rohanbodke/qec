#!/usr/bin/env python3
"""
parse_v20_benchmarks.py
Zero-trust parser and 10,000-draw bootstrap analyzer for the V20 3D cross-family benchmark
campaign (`cluster_v20_results.txt`) under standard unit stabilizer coupling (J = 1),
comparing:
  1. 2D Toric Code (`L = 8, 16, 32, 64`)
  2. Single-Scale 3D Welded Surface Codes (`L = 2, 3, 4, 6, 8`, 1-cycle Michnicki-type)
  3. 3D-Volume-Admissible Multiscale Welded Codes (`C3 C2`, `RX3 C2`, `C3 C2^ZX`, `RZ3 C2^ZX`)
  4. Haah's 3D Cubic Code 1 (`L = 5, 7, 9, 11, 13, 15, 17`, strictly Z^3 nearest-neighbor)
under identical continuous-time BKL Glauber dynamics and paired 4-decoder readout (`UF`,
`Cool-UF`, `Hier-UF`, `BP-OSD`).
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V20_RAW = ROOT / "cluster_v20_results.txt"
V20_SPECS = ROOT / "v20_code_specs.json"
V19_SPECS = ROOT / "v19_code_specs.json"
OUT_JSON = ROOT / "parsed_v20_benchmarks.json"

DECS = ("uf", "cool", "hier", "bposd")
PASS_CHARS = ("1", "2")


def parse_kv(body):
    out = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def crossing_time_log(x_grid, p_curve, thresh=0.80):
    for g in range(len(x_grid)):
        if p_curve[g] < thresh:
            if g == 0:
                return float(x_grid[0]), True
            p0, p1 = float(p_curve[g - 1]), float(p_curve[g])
            if p0 <= p1:
                return float(x_grid[g]), True
            frac = (p0 - thresh) / (p0 - p1)
            lg0, lg1 = math.log10(x_grid[g - 1]), math.log10(x_grid[g])
            return float(10.0 ** (lg0 + frac * (lg1 - lg0))), True
    return float(x_grid[-1]), False


def eval_arm(x_map, z_map, grid_spec, rng, B=10000):
    lg0_s, lg1_s, ng_s = grid_spec.split(":")
    lg0, lg1_full, ng_full = float(lg0_s), float(lg1_s), int(ng_s)
    step = (lg1_full - lg0) / max(1, ng_full - 1)

    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    ng = min([ng_full] + [len(x_map[k]["uf"]) for k in x_keys] + [len(z_map[k]["uf"]) for k in z_keys])
    t_grid = np.array([10.0 ** (lg0 + step * i) for i in range(ng)], dtype=np.float64)

    nx, nz = len(x_keys), len(z_keys)
    cap_x = sum(1 for k in x_keys if "C" in x_map[k]["uf"][:ng])
    cap_z = sum(1 for k in z_keys if "C" in z_map[k]["uf"][:ng])

    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))

    dec_stats = {}
    b_by_dec = {}
    for d in DECS:
        Xp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in x_map[k][d][:ng]] for k in x_keys], dtype=np.float64)
        Zp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in z_map[k][d][:ng]] for k in z_keys], dtype=np.float64)
        px = Xp.mean(axis=0)
        pz = Zp.mean(axis=0)
        pq = px * pz

        t80, cr = crossing_time_log(t_grid, pq, 0.80)
        t80_x, cr_x = crossing_time_log(t_grid, px, 0.80)
        t80_z, cr_z = crossing_time_log(t_grid, pz, 0.80)

        b_pq = Xp[ix].mean(axis=1) * Zp[iz].mean(axis=1)
        b_t80 = np.array([crossing_time_log(t_grid, b_pq[b], 0.80)[0] for b in range(B)])
        b_by_dec[d] = b_t80

        dec_stats[d] = {
            "t_80": t80,
            "t_80_crossed": cr,
            "t_80_ci95": [float(np.percentile(b_t80, 2.5)), float(np.percentile(b_t80, 97.5))],
            "t_80_X": t80_x,
            "t_80_X_crossed": cr_x,
            "t_80_Z": t80_z,
            "t_80_Z_crossed": cr_z,
        }

    t80_bp = dec_stats["bposd"]["t_80"]
    g_cross = max(0, min(ng - 1, math.ceil((math.log10(max(t_grid[0], t80_bp)) - lg0) / max(1e-12, step)) + 1))
    cap_pre_x = sum(1 for k in x_keys if "C" in x_map[k]["uf"][:g_cross + 1])
    cap_pre_z = sum(1 for k in z_keys if "C" in z_map[k]["uf"][:g_cross + 1])

    paired_bp_uf = b_by_dec["bposd"] / np.maximum(1e-30, b_by_dec["uf"])
    return {
        "n_X": nx,
        "n_Z": nz,
        "capped_X": cap_x,
        "capped_Z": cap_z,
        "capped_pre_t80_X": cap_pre_x,
        "capped_pre_t80_Z": cap_pre_z,
        "decoders": dec_stats,
        "paired_bposd_vs_uf": {
            "ratio": float(dec_stats["bposd"]["t_80"] / max(1e-30, dec_stats["uf"]["t_80"])),
            "ci95": [float(np.percentile(paired_bp_uf, 2.5)), float(np.percentile(paired_bp_uf, 97.5))],
        },
        "_b_uf": b_by_dec["uf"],
        "_b_bposd": b_by_dec["bposd"],
    }


def indep_ratio(a_num, a_den, dec="bposd"):
    k = "_b_bposd" if dec == "bposd" else "_b_uf"
    pt = float(a_num["decoders"][dec]["t_80"] / max(1e-30, a_den["decoders"][dec]["t_80"]))
    br = a_num[k] / np.maximum(1e-30, a_den[k])
    return {"ratio": pt, "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}


def main():
    with open(V20_SPECS) as f:
        specs = json.load(f)
    if V19_SPECS.exists():
        with open(V19_SPECS) as f:
            s19 = json.load(f)
        for k, v in s19.items():
            if k not in specs:
                specs[k] = {
                    "tag": k,
                    "N": v["N"],
                    "K": v["K"],
                    "max_w": 7,
                    "d_tot": 4.63,
                }

    trajs = defaultdict(dict)
    grids = {}
    betas = {}
    errors = []

    with open(V20_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if not body.startswith("PAIRED_READOUT"):
                continue
            m = re.match(r"^v20__(.+)_(X_err|Z_err)__(b[qr][0-9.]+)__b([0-9.]+)__o(\d+)$", tid)
            if not m:
                errors.append((tid, "UNPARSED"))
                continue
            tag, sec, mode_lbl, b_s, off_s = m.groups()
            off = int(off_s)
            kv = parse_kv(body)
            arm_key = f"{tag}__{mode_lbl}"
            grids[arm_key] = kv["grid"]
            betas[arm_key] = float(b_s)
            split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
            ntr = len(split_d["uf"])
            for i in range(ntr):
                trajs[(arm_key, sec)][off + i] = {d: split_d[d][i] for d in DECS}

    print(f"Loaded V20 benchmark results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112420)
    arms = {}
    for arm_key in sorted(grids.keys()):
        tag, mode_lbl = arm_key.split("__", 1)
        xm = trajs.get((arm_key, "X_err"), {})
        zm = trajs.get((arm_key, "Z_err"), {})
        if not xm or not zm:
            continue
        ev = eval_arm(xm, zm, grids[arm_key], rng, B=10000)
        sp = specs[tag]
        ev["tag"] = tag
        ev["mode"] = mode_lbl
        ev["N"] = sp["N"]
        ev["K"] = sp["K"]
        ev["max_w"] = sp.get("max_w", 7)
        ev["d_tot"] = sp.get("d_tot", 4.63)
        ev["beta"] = betas[arm_key]
        arms[arm_key] = ev

        u = ev["decoders"]["uf"]
        bp = ev["decoders"]["bposd"]
        p_u = "" if u["t_80_crossed"] else "+"
        p_b = "" if bp["t_80_crossed"] else "+"
        print(
            f"  {arm_key:32s}: N={sp['N']:7,d} K={sp['K']:2d} beta={ev['beta']:.3f} "
            f"n=({ev['n_X']},{ev['n_Z']}) pre_cap=({ev['capped_pre_t80_X']},{ev['capped_pre_t80_Z']}) "
            f"| UF={u['t_80']:9.1f}{p_u} | BPOSD={bp['t_80']:9.1f}{p_b} [{bp['t_80_ci95'][0]:.3g},{bp['t_80_ci95'][1]:.3g}]"
        )

    contrasts = {}
    c_specs = [
        ("mich_L2_over_L8_b65",  "v20_mich1c_L2__br6.50", "v20_mich1c_L8__br6.50"),
        ("tor2d_L8_over_L64_b65","v20_tor2d_L8__bq6.50",  "v20_tor2d_L64__bq6.50"),
        ("haah_L5_over_L17_b325","v20_haah_L5__bq6.50",   "v20_haah_L17__bq6.50"),
        ("haah_L7_over_L17_b325","v20_haah_L7__bq6.50",   "v20_haah_L17__bq6.50"),
        ("haah_L5_over_L17_b40", "v20_haah_L5__br4.00",   "v20_haah_L17__br4.00"),
        ("haah_L7_over_L17_b40", "v20_haah_L7__br4.00",   "v20_haah_L17__br4.00"),
        ("haah_L7_over_L17_b378","v20_haah_L7__bq7.55",   "v20_haah_L17__bq7.55"),
    ]
    for cname, k_num, k_den in c_specs:
        if k_num in arms and k_den in arms:
            contrasts[cname] = {
                "bposd": indep_ratio(arms[k_num], arms[k_den], "bposd"),
                "uf": indep_ratio(arms[k_num], arms[k_den], "uf"),
            }
            rb = contrasts[cname]["bposd"]
            print(f"  [{cname:24s}]: BPOSD ratio = {rb['ratio']:8.2f}x [{rb['ci95'][0]:.2f}, {rb['ci95'][1]:.2f}]")

    clean_arms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()}
    with open(OUT_JSON, "w") as f:
        json.dump({"errors": errors, "specs": specs, "arms": clean_arms, "contrasts": contrasts}, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
