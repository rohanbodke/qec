#!/usr/bin/env python3
"""
parse_v22_color_welded.py
Zero-trust parser and 10,000-draw bootstrap analyzer for the V22 3D-Local Welded
4.8.8 Flag Color-Code campaign (`cluster_v22_results.txt`) + unwelded 2D 4.8.8 controls.
Outputs `parsed_v22_color_welded.json`.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parse_v21_3d_local import (
    DECS,
    PASS_CHARS,
    crossing_linear,
    crossing_time_log,
    eval_iid_arm,
    eval_thermal_arm,
    indep_ratio,
    parse_kv,
)

V22_RAW = ROOT / "cluster_v22_results.txt"
V22_SPECS = ROOT / "v22_code_specs.json"
OUT_JSON = ROOT / "parsed_v22_color_welded.json"


def main():
    with open(V22_SPECS) as f:
        specs = json.load(f)

    th_trajs = defaultdict(dict)
    th_grids = {}
    th_betas = {}
    iid_trajs = defaultdict(dict)
    iid_grids = {}
    errors = []

    with open(V22_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^(v22_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
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
                m = re.match(r"^v22_iid__(v22_.+)_(X_err|Z_err)__o(\d+)$", tid)
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

    print(f"Loaded V22 color-welded results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112512)
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
        ev["K"] = sp["K"]
        ev["L_blk"] = sp["L_blk"]
        ev["dim_F"] = sp["dim_F"]
        ev["seam_mode"] = sp["seam_mode"]
        ev["beta"] = th_betas[akey]
        arms[akey] = ev

        u = ev["decoders"]["uf"]
        bp = ev["decoders"]["bposd"]
        pu = "" if u["t_80_crossed"] else "+"
        pb = "" if bp["t_80_crossed"] else "+"
        print(
            f"  {akey:34s}: N={sp['N']:7,d} b={ev['beta']:.2f} n=({ev['n_X']},{ev['n_Z']}) "
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
        print(f"  [IID] {tag:26s}: UF={u80:.2f}% Cool={c80:.2f}% BP-OSD={b80:.2f}% (BP/UF={rr['ratio']:.2f}x [{rr['ci95'][0]:.2f},{rr['ci95'][1]:.2f}])")

    # Cross-campaign references (V21) re-bootstrapped directly from cluster_v21_results.txt
    v21_raw = ROOT / "cluster_v21_results.txt"
    v21_targets = {
        "v21_4s_R3_R3__b6.50",
        "v21_4s_C3_R3ZX__b5.00",
        "v21_4s_C3_R3ZX__b6.50",
        "v21_4s_R3_R3ZX__b6.50",
        "v21_tor_4s_R3_R3ZX__b6.50",
    }
    v21_th = defaultdict(dict)
    v21_grids = {}
    if v21_raw.exists():
        with open(v21_raw) as f:
            for raw in f:
                line = raw.strip()
                if not line or "|" not in line:
                    continue
                tid, body = [x.strip() for x in line.split("|", 1)]
                if not body.startswith("PAIRED_READOUT"):
                    continue
                m = re.match(r"^v21__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                if not m:
                    continue
                tag, sec, b_s, off_s = m.groups()
                akey = f"{tag}__b{b_s}"
                if akey not in v21_targets:
                    continue
                off = int(off_s)
                kv = parse_kv(body)
                v21_grids[akey] = kv["grid"]
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                for i in range(len(split_d["uf"])):
                    v21_th[(akey, sec)][off + i] = {d: split_d[d][i] for d in DECS}

    rng_v21 = np.random.default_rng(2026112599)
    all_arms = dict(arms)
    for akey in sorted(v21_grids.keys()):
        xm = v21_th.get((akey, "X_err"), {})
        zm = v21_th.get((akey, "Z_err"), {})
        if xm and zm:
            all_arms[akey] = eval_thermal_arm(xm, zm, v21_grids[akey], rng_v21, B=10000)

    contrasts = {}
    c_list = [
        ("cc2d_L3_vs_L12_b65",          "v22_cc2d_L3__b6.50",            "v22_cc2d_L12__b6.50"),
        ("cc1s_L3r2_vs_sd_b50",         "v22_cc1s_L3r2__b5.00",          "v22_cc1s_L3r2_sd__b5.00"),
        ("cc1s_L3r2_vs_sd_b65",         "v22_cc1s_L3r2__b6.50",          "v22_cc1s_L3r2_sd__b6.50"),
        ("cc2s_R3_vs_sd_b50",           "v22_cc2s_R3__b5.00",            "v22_cc2s_R3_sd__b5.00"),
        ("cc2s_R3_vs_sd_b65",           "v22_cc2s_R3__b6.50",            "v22_cc2s_R3_sd__b6.50"),
        ("cc2s_R3_vs_cc2d_L12_b50",     "v22_cc2s_R3__b5.00",            "v22_cc2d_L12__b5.00"),
        ("cc2s_R3_vs_cc2d_L12_b65",     "v22_cc2s_R3__b6.50",            "v22_cc2d_L12__b6.50"),
        ("cc2s_RX3_vs_cc2d_L12_b65",    "v22_cc2s_RX3__b6.50",           "v22_cc2d_L12__b6.50"),
        ("cc2s_RX3_vs_RZ3_b65",         "v22_cc2s_RX3__b6.50",           "v22_cc2s_RZ3__b6.50"),
        ("cc2s_RX3_vs_R3_b65",          "v22_cc2s_RX3__b6.50",           "v22_cc2s_R3__b6.50"),
        ("cc2s_R3_vs_C3_b65",           "v22_cc2s_R3__b6.50",            "v22_cc2s_C3__b6.50"),
        ("cc3s_midZ2_vs_out1b_b50",     "v22_cc3s_X3r1Z3r2X3r1__b5.00",  "v22_cc3s_X3r1Z3r1X3r1__b5.00"),
        ("cc3s_midZ2_vs_out1b_b65",     "v22_cc3s_X3r1Z3r2X3r1__b6.50",  "v22_cc3s_X3r1Z3r1X3r1__b6.50"),
        ("cc3s_midZ2_vs_cc2d_L12_b65",  "v22_cc3s_X3r1Z3r2X3r1__b6.50",  "v22_cc2d_L12__b6.50"),
        ("cc3s_R3X3r2_vs_cc2d_L12_b65", "v22_cc3s_R3X3r2__b6.50",        "v22_cc2d_L12__b6.50"),
        ("cc2s_R3_vs_bdl_R3R3_b65",     "v22_cc2s_R3__b6.50",            "v21_4s_R3_R3__b6.50"),
        ("cc2s_RX3_vs_bdl_R3R3_b65",    "v22_cc2s_RX3__b6.50",           "v21_4s_R3_R3__b6.50"),
        ("cc3s_midZ2_vs_bdl_R3R3_b65",  "v22_cc3s_X3r1Z3r2X3r1__b6.50",  "v21_4s_R3_R3__b6.50"),
        ("cc2s_RX3_vs_C3R3ZX_b50",      "v22_cc2s_RX3__b5.00",           "v21_4s_C3_R3ZX__b5.00"),
        ("cc2s_mix32_vs_C3R3ZX_b50",    "v22_cc2s_mix32__b5.00",         "v21_4s_C3_R3ZX__b5.00"),
        ("cc2s_mix32_vs_torR3R3ZX_b65", "v22_cc2s_mix32__b6.50",         "v21_tor_4s_R3_R3ZX__b6.50"),
    ]
    for cname, kn, kd in c_list:
        if kn in all_arms and kd in all_arms:
            contrasts[cname] = {
                "bposd": indep_ratio(all_arms[kn], all_arms[kd], "bposd"),
                "uf": indep_ratio(all_arms[kn], all_arms[kd], "uf"),
            }
            rb = contrasts[cname]["bposd"]
            print(f"  [{cname:30s}]: BP={rb['ratio']:8.2f}x [{rb['ci95'][0]:.2f}, {rb['ci95'][1]:.2f}]")

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
