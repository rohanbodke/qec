#!/usr/bin/env python3
"""
parse_v21_3d_local.py
Zero-trust parser and 10,000-draw bootstrap analyzer for the V21 Strictly 3D-Local (`L = 3` at
every step, `dim_eff < 3`) welded memory campaign (`cluster_v21_results.txt`).
Evaluates all 4 paired decoders (`UF`, `Cool-UF`, `Hier-UF`, `BP-OSD`), microscopic failure
anatomy near `t_0.80^UF`, unrelaxed single-shot i.i.d. error thresholds `p_0.80`, and exact
bootstrap confidence intervals across all pure-L = 3 contrasts.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V21_RAW = ROOT / "cluster_v21_results.txt"
V21_SPECS = ROOT / "v21_code_specs.json"
OUT_JSON = ROOT / "parsed_v21_3d_local.json"

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


def crossing_linear(x_grid, p_curve, thresh=0.80):
    for g in range(len(x_grid)):
        if p_curve[g] < thresh:
            if g == 0:
                return float(x_grid[0]), True
            p0, p1 = float(p_curve[g - 1]), float(p_curve[g])
            if p0 <= p1:
                return float(x_grid[g]), True
            frac = (p0 - thresh) / (p0 - p1)
            return float(x_grid[g - 1] + frac * (x_grid[g] - x_grid[g - 1])), True
    return float(x_grid[-1]), False


def eval_thermal_arm(x_map, z_map, grid_spec, rng, B=10000):
    lg0_s, lg1_s, ng_s = grid_spec.split(":")
    lg0, lg1_full, ng_full = float(lg0_s), float(lg1_s), int(ng_s)
    step = (lg1_full - lg0) / max(1, ng_full - 1)

    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    ng = min([ng_full] + [len(x_map[k]["uf"]) for k in x_keys] + [len(z_map[k]["uf"]) for k in z_keys])
    t_grid = np.array([10.0 ** (lg0 + step * i) for i in range(ng)], dtype=np.float64)

    nx, nz = len(x_keys), len(z_keys)
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

        # No-prior-failure curve for UF/BPOSD return analysis
        Xnc = np.array([[1.0 if ch == "2" else 0.0 for ch in x_map[k][d][:ng]] for k in x_keys], dtype=np.float64)
        Znc = np.array([[1.0 if ch == "2" else 0.0 for ch in z_map[k][d][:ng]] for k in z_keys], dtype=np.float64)
        pq_nc = Xnc.mean(axis=0) * Znc.mean(axis=0)
        t80_nc, cr_nc = crossing_time_log(t_grid, pq_nc, 0.80)
        b_pq_nc = Xnc[ix].mean(axis=1) * Znc[iz].mean(axis=1)
        b_t80_nc = np.array([crossing_time_log(t_grid, b_pq_nc[b], 0.80)[0] for b in range(B)])
        ret_rat = b_t80 / np.maximum(1e-30, b_t80_nc)

        dec_stats[d] = {
            "t_80": t80,
            "t_80_crossed": cr,
            "t_80_ci95": [float(np.percentile(b_t80, 2.5)), float(np.percentile(b_t80, 97.5))],
            "t_80_X": t80_x,
            "t_80_X_crossed": cr_x,
            "t_80_Z": t80_z,
            "t_80_Z_crossed": cr_z,
            "t_80_nocross": t80_nc,
            "t_80_nocross_crossed": cr_nc,
            "t_80_nocross_ci95": [float(np.percentile(b_t80_nc, 2.5)), float(np.percentile(b_t80_nc, 97.5))],
            "return_ratio": float(t80 / max(1e-30, t80_nc)),
            "return_ratio_ci95": [float(np.percentile(ret_rat, 2.5)), float(np.percentile(ret_rat, 97.5))],
            "p_Q": [float(v) for v in pq],
            "p_Q_nocross": [float(v) for v in pq_nc],
        }

    t80_bp = dec_stats["bposd"]["t_80"]
    g_cross = max(0, min(ng - 1, round((math.log10(max(t_grid[0], t80_bp)) - lg0) / max(1e-12, step))))
    cap_pre_x = sum(1 for k in x_keys if "C" in x_map[k]["uf"][:g_cross + 1])
    cap_pre_z = sum(1 for k in z_keys if "C" in z_map[k]["uf"][:g_cross + 1])

    # Failure anatomy near g_cross on limiting sector
    n_gq, n_ex, n_bp = 0, 0, 0
    for m_sec in (x_map, z_map):
        for k in m_sec:
            u_ch = m_sec[k]["uf"][g_cross]
            b_ch = m_sec[k]["bposd"][g_cross]
            if u_ch not in PASS_CHARS:
                if b_ch in PASS_CHARS:
                    n_bp += 1
                elif u_ch in ("G", "Q"):
                    n_gq += 1
                else:
                    n_ex += 1
    tot_f = max(1, n_gq + n_ex + n_bp)

    paired_bp_uf = b_by_dec["bposd"] / np.maximum(1e-30, b_by_dec["uf"])
    paired_cool_uf = b_by_dec["cool"] / np.maximum(1e-30, b_by_dec["uf"])

    return {
        "n_X": nx,
        "n_Z": nz,
        "t_grid": [float(v) for v in t_grid],
        "capped_pre_t80_X": cap_pre_x,
        "capped_pre_t80_Z": cap_pre_z,
        "decoders": dec_stats,
        "paired_bposd_vs_uf": {
            "ratio": float(dec_stats["bposd"]["t_80"] / max(1e-30, dec_stats["uf"]["t_80"])),
            "ci95": [float(np.percentile(paired_bp_uf, 2.5)), float(np.percentile(paired_bp_uf, 97.5))],
        },
        "paired_cool_vs_uf": {
            "ratio": float(dec_stats["cool"]["t_80"] / max(1e-30, dec_stats["uf"]["t_80"])),
            "ci95": [float(np.percentile(paired_cool_uf, 2.5)), float(np.percentile(paired_cool_uf, 97.5))],
        },
        "failure_anatomy_near_t80": {
            "tot_uf_failures": n_gq + n_ex + n_bp,
            "frac_settled_wrong_basin_GQ": n_gq / tot_f,
            "frac_excited_unrescued": n_ex / tot_f,
            "frac_rescued_by_bposd": n_bp / tot_f,
        },
        "_b_uf": b_by_dec["uf"],
        "_b_bposd": b_by_dec["bposd"],
    }


def eval_iid_arm(x_map, z_map, pgrid_spec, rng, B=10000):
    p0_s, p1_s, ng_s = pgrid_spec.split(":")
    p0, p1, ng = float(p0_s), float(p1_s), int(ng_s)
    p_grid = np.linspace(p0, p1, ng)
    x_keys = sorted(x_map.keys())
    z_keys = sorted(z_map.keys())
    nx, nz = len(x_keys), len(z_keys)
    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))

    dec_stats = {}
    b_by_dec = {}
    for d in DECS:
        Xp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in x_map[k][d][:ng]] for k in x_keys], dtype=np.float64)
        Zp = np.array([[1.0 if ch in PASS_CHARS else 0.0 for ch in z_map[k][d][:ng]] for k in z_keys], dtype=np.float64)
        pq = Xp.mean(axis=0) * Zp.mean(axis=0)
        p80, cr = crossing_linear(p_grid, pq, 0.80)
        b_pq = Xp[ix].mean(axis=1) * Zp[iz].mean(axis=1)
        b_p80 = np.array([crossing_linear(p_grid, b_pq[b], 0.80)[0] for b in range(B)])
        b_by_dec[d] = b_p80
        dec_stats[d] = {
            "p_80": p80,
            "p_80_crossed": cr,
            "p_80_ci95": [float(np.percentile(b_p80, 2.5)), float(np.percentile(b_p80, 97.5))],
            "curve_Q": [float(v) for v in pq],
        }

    paired_bp_uf = b_by_dec["bposd"] / np.maximum(1e-30, b_by_dec["uf"])
    return {
        "n_X": nx,
        "n_Z": nz,
        "p_grid": [float(v) for v in p_grid],
        "decoders": dec_stats,
        "paired_bposd_vs_uf": {
            "ratio": float(dec_stats["bposd"]["p_80"] / max(1e-30, dec_stats["uf"]["p_80"])),
            "ci95": [float(np.percentile(paired_bp_uf, 2.5)), float(np.percentile(paired_bp_uf, 97.5))],
        },
    }


def indep_ratio(a_num, a_den, dec="bposd"):
    k = "_b_bposd" if dec == "bposd" else "_b_uf"
    pt = float(a_num["decoders"][dec]["t_80"] / max(1e-30, a_den["decoders"][dec]["t_80"]))
    br = a_num[k] / np.maximum(1e-30, a_den[k])
    return {"ratio": pt, "ci95": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]}


def main():
    with open(V21_SPECS) as f:
        specs = json.load(f)

    th_trajs = defaultdict(dict)
    th_grids = {}
    th_betas = {}
    iid_trajs = defaultdict(dict)
    iid_grids = {}
    errors = []

    with open(V21_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^v21__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                if not m:
                    errors.append((tid, "UNPARSED_TH"))
                    continue
                tag, sec, b_s, off_s = m.groups()
                off = int(off_s)
                kv = parse_kv(body)
                akey = f"{tag}__b{b_s}"
                th_grids[akey] = kv["grid"]
                th_betas[akey] = float(b_s)
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                for i in range(len(split_d["uf"])):
                    th_trajs[(akey, sec)][off + i] = {d: split_d[d][i] for d in DECS}
            elif body.startswith("IID_READOUT"):
                m = re.match(r"^v21_iid__(.+)_(X_err|Z_err)__o(\d+)$", tid)
                if not m:
                    errors.append((tid, "UNPARSED_IID"))
                    continue
                tag, sec, off_s = m.groups()
                off = int(off_s)
                kv = parse_kv(body)
                iid_grids[tag] = kv["pgrid"]
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                for i in range(len(split_d["uf"])):
                    iid_trajs[(tag, sec)][off + i] = {d: split_d[d][i] for d in DECS}

    print(f"Loaded V21 3D-local results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112423)
    arms = {}
    for akey in sorted(th_grids.keys()):
        tag, _ = akey.split("__b", 1)
        xm = th_trajs.get((akey, "X_err"), {})
        zm = th_trajs.get((akey, "Z_err"), {})
        if not xm or not zm:
            continue
        ev = eval_thermal_arm(xm, zm, th_grids[akey], rng, B=10000)
        sp = specs[tag]
        ev["tag"] = tag
        ev["N"] = sp["N"]
        ev["K"] = sp["K_target"]
        ev["K_tot"] = sp["K_tot"]
        ev["dim_eff"] = sp["dim_eff"]
        ev["beta"] = th_betas[akey]
        arms[akey] = ev

        u = ev["decoders"]["uf"]
        bp = ev["decoders"]["bposd"]
        pu = "" if u["t_80_crossed"] else "+"
        pb = "" if bp["t_80_crossed"] else "+"
        print(
            f"  {akey:36s}: N={sp['N']:7,d} b={ev['beta']:.2f} n=({ev['n_X']},{ev['n_Z']}) "
            f"pre_cap=({ev['capped_pre_t80_X']},{ev['capped_pre_t80_Z']}) "
            f"| UF={u['t_80']:9.1f}{pu} | BP={bp['t_80']:9.1f}{pb} [{bp['t_80_ci95'][0]:.3g},{bp['t_80_ci95'][1]:.3g}]"
        )

    iid_arms = {}
    for tag in sorted(iid_grids.keys()):
        xm = iid_trajs.get((tag, "X_err"), {})
        zm = iid_trajs.get((tag, "Z_err"), {})
        if not xm or not zm:
            continue
        ev = eval_iid_arm(xm, zm, iid_grids[tag], rng, B=10000)
        ev["N"] = specs[tag]["N"]
        iid_arms[tag] = ev
        u80 = ev["decoders"]["uf"]["p_80"] * 100
        c80 = ev["decoders"]["cool"]["p_80"] * 100
        b80 = ev["decoders"]["bposd"]["p_80"] * 100
        rr = ev["paired_bposd_vs_uf"]
        print(f"  [IID] {tag:28s}: UF={u80:.2f}% Cool={c80:.2f}% BP-OSD={b80:.2f}% (BP/UF={rr['ratio']:.2f}x [{rr['ci95'][0]:.2f},{rr['ci95'][1]:.2f}])")

    contrasts = {}
    c_list = [
        ("C3_pruneX_vs_full_b65",  "v21_L3_C3_pruneX__b6.50",        "v21_L3_C3_full__b6.50"),
        ("C3_pruneZ_vs_full_b65",  "v21_L3_C3_pruneZ__b6.50",        "v21_L3_C3_full__b6.50"),
        ("C3_pruneBoth_vs_full_b65","v21_L3_C3_pruneBoth__b6.50",    "v21_L3_C3_full__b6.50"),
        ("1c_R3_vs_C3_b65",        "v21_L3_R3__b6.50",               "v21_L3_C3_full__b6.50"),
        ("1c_R3ZX_vs_R3_b65",      "v21_L3_R3ZX__b6.50",             "v21_L3_R3__b6.50"),
        ("3s_midZ2_vs_outX2_b65",  "v21_3s_X3r1_Z3r2_X3r1__b6.50",   "v21_3s_X3r2_Z3r1_X3r1__b6.50"),
        ("3s_midZ2_vs_outX2_b70",  "v21_3s_X3r1_Z3r2_X3r1__b7.00",   "v21_3s_X3r2_Z3r1_X3r1__b7.00"),
        ("3s_midZ2_vs_1b_b65",     "v21_3s_X3r1_Z3r2_X3r1__b6.50",   "v21_3s_X3r1_Z3r1_X3r1__b6.50"),
        ("3s_midZ2_vs_R3R3_b65",   "v21_3s_X3r1_Z3r2_X3r1__b6.50",   "v21_4s_R3_R3__b6.50"),
        ("4s_C3R3_vs_R3R3_b65",    "v21_4s_C3_R3__b6.50",            "v21_4s_R3_R3__b6.50"),
        ("4s_C3RX3_vs_C3R3_b65",   "v21_4s_C3_RX3__b6.50",           "v21_4s_C3_R3__b6.50"),
        ("4s_C3RZ3_vs_C3R3_b65",   "v21_4s_C3_RZ3__b6.50",           "v21_4s_C3_R3__b6.50"),
        ("4s_C3RX3_vs_C3RZ3_b65",  "v21_4s_C3_RX3__b6.50",           "v21_4s_C3_RZ3__b6.50"),
        ("4s_C3RX3_vs_R3R3_b65",   "v21_4s_C3_RX3__b6.50",           "v21_4s_R3_R3__b6.50"),
        ("tor_C3RX3_vs_R3R3_b65",  "v21_tor_4s_C3_RX3__b6.50",       "v21_tor_4s_R3_R3__b6.50"),
        ("4s_RX3RX3_vs_R3R3_b65",  "v21_4s_RX3_RX3__b6.50",          "v21_4s_R3_R3__b6.50"),
        ("4s_RX3RX3_vs_C3RX3_b70", "v21_4s_RX3_RX3__b7.00",          "v21_4s_C3_RX3__b7.00"),
        ("4s_RX3RX3_vs_R3R3_b70",  "v21_4s_RX3_RX3__b7.00",          "v21_4s_R3_R3__b7.00"),
        ("4s_ZXZX_vs_XZXZ_b65",    "v21_4s_R3ZX_R3ZX__b6.50",        "v21_4s_R3_R3__b6.50"),
        ("4s_ZXXZ_vs_XZXZ_b65",    "v21_4s_R3ZX_R3__b6.50",          "v21_4s_R3_R3__b6.50"),
        ("4s_ZXXZ_vs_XZXZ_b70",    "v21_4s_R3ZX_R3__b7.00",          "v21_4s_R3_R3__b7.00"),
        ("4s_R3R3ZX_vs_R3R3_b65",  "v21_4s_R3_R3ZX__b6.50",          "v21_4s_R3_R3__b6.50"),
        ("4s_R3R3ZX_vs_R3R3_b70",  "v21_4s_R3_R3ZX__b7.00",          "v21_4s_R3_R3__b7.00"),
        ("4s_C3R3ZX_vs_R3R3_b65",  "v21_4s_C3_R3ZX__b6.50",          "v21_4s_R3_R3__b6.50"),
        ("4s_C3R3ZX_vs_R3R3_b70",  "v21_4s_C3_R3ZX__b7.00",          "v21_4s_R3_R3__b7.00"),
        ("4s_C3R3ZX_vs_C3R3_b65",  "v21_4s_C3_R3ZX__b6.50",          "v21_4s_C3_R3__b6.50"),
        ("4s_RZ3R3ZX_vs_R3R3_b65", "v21_4s_RZ3_R3ZX__b6.50",         "v21_4s_R3_R3__b6.50"),
        ("4s_RZ3R3ZX_vs_RX3R3ZX_b60", "v21_4s_RZ3_R3ZX__b6.00",      "v21_4s_RX3_R3ZX__b6.00"),
        ("4s_RZ3R3ZX_vs_RX3R3ZX_b65", "v21_4s_RZ3_R3ZX__b6.50",      "v21_4s_RX3_R3ZX__b6.50"),
        ("tor_R3R3ZX_vs_R3R3_b65", "v21_tor_4s_R3_R3ZX__b6.50",      "v21_tor_4s_R3_R3__b6.50"),
    ]
    for cname, kn, kd in c_list:
        if kn in arms and kd in arms:
            contrasts[cname] = {
                "bposd": indep_ratio(arms[kn], arms[kd], "bposd"),
                "uf": indep_ratio(arms[kn], arms[kd], "uf"),
            }
            rb = contrasts[cname]["bposd"]
            print(f"  [{cname:28s}]: BP={rb['ratio']:8.2f}x [{rb['ci95'][0]:.2f}, {rb['ci95'][1]:.2f}]")

    clean_arms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()}
    with open(OUT_JSON, "w") as f:
        json.dump({
            "errors": errors,
            "specs": specs,
            "arms": clean_arms,
            "iid_arms": iid_arms,
            "contrasts": contrasts,
        }, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
