#!/usr/bin/env python3
"""
parse_v23_extensionA.py
Zero-trust parser and 10,000-draw bootstrap analyzer for Campaign V23 (Extension A:
BDL T-junction dispersed bridges + open-boundary `css_dual` planar 4.8.8 welded
color codes + beta=7.0 cc3s_121 reference). Outputs `parsed_v23_extensionA.json`.
"""

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parse_v21_3d_local import (
    DECS,
    eval_iid_arm,
    eval_thermal_arm,
    indep_ratio,
    parse_kv,
)

V23_RAW = ROOT / "cluster_v23_results.txt"
V23_SPECS = ROOT / "v23_code_specs.json"
OUT_JSON = ROOT / "parsed_v23_extensionA.json"


def main():
    with open(V23_SPECS) as f:
        specs = json.load(f)

    th_trajs = defaultdict(dict)
    th_grids = {}
    th_betas = {}
    iid_trajs = defaultdict(dict)
    iid_grids = {}
    errors = []

    with open(V23_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^(v23_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
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
                m = re.match(r"^v23_iid__(v23_.+)_(X_err|Z_err)__o(\d+)$", tid)
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

    print(f"Loaded V23 Extension A results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112516)
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
        ev["beta"] = th_betas[akey]
        arms[akey] = ev

        u = ev["decoders"]["uf"]
        bp = ev["decoders"]["bposd"]
        pu = "" if u["t_80_crossed"] else "+"
        pb = "" if bp["t_80_crossed"] else "+"
        print(
            f"  {akey:32s}: N={sp['N']:7,d} K={sp['K']} b={ev['beta']:.2f} n=({ev['n_X']},{ev['n_Z']}) "
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
        ev["K"] = specs[tag]["K"]
        iid_arms[tag] = ev
        u80 = ev["decoders"]["uf"]["p_80"] * 100
        b80 = ev["decoders"]["bposd"]["p_80"] * 100
        rr = ev["paired_bposd_vs_uf"]
        print(f"  [IID] {tag:24s}: UF={u80:.2f}% BP-OSD={b80:.2f}% (BP/UF={rr['ratio']:.2f}x [{rr['ci95'][0]:.2f},{rr['ci95'][1]:.2f}])")

    # Cross-campaign references from cluster_v22_results.txt
    v22_raw = ROOT / "cluster_v22_results.txt"
    v22_targets = {
        "v22_cc1s_L3r1__b6.50",
        "v22_cc1s_L3r2__b6.50",
        "v22_cc2s_C3__b6.50",
        "v22_cc2s_RX3__b6.50",
        "v22_cc3s_X3r1Z3r1X3r1__b6.50",
        "v22_cc3s_X3r1Z3r2X3r1__b6.50",
    }
    v22_th, v22_grids = defaultdict(dict), {}
    if v22_raw.exists():
        with open(v22_raw) as f:
            for raw in f:
                line = raw.strip()
                if not line or "|" not in line:
                    continue
                tid, body = [x.strip() for x in line.split("|", 1)]
                if not body.startswith("PAIRED_READOUT"):
                    continue
                m = re.match(r"^(v22_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                if not m:
                    continue
                tag, sec, b_s, off_s = m.groups()
                akey = f"{tag}__b{b_s}"
                if akey not in v22_targets:
                    continue
                off = int(off_s)
                kv = parse_kv(body)
                v22_grids[akey] = kv["grid"]
                split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
                for i in range(len(split_d["uf"])):
                    v22_th[(akey, sec)][off + i] = {d: split_d[d][i] for d in DECS}

    rng_ref = np.random.default_rng(2026112598)
    all_arms = dict(arms)
    for akey in sorted(v22_grids.keys()):
        xm = v22_th.get((akey, "X_err"), {})
        zm = v22_th.get((akey, "Z_err"), {})
        if xm and zm:
            all_arms[akey] = eval_thermal_arm(xm, zm, v22_grids[akey], rng_ref, B=10000)

    contrasts = {}
    c_list = [
        ("disp_2s_vs_cc2s_C3_b65",      "v23_disp_2s_L3__b6.50",      "v22_cc2s_C3__b6.50"),
        ("disp_2s_RX3_vs_cc2s_RX3_b65", "v23_disp_2s_RX3__b6.50",     "v22_cc2s_RX3__b6.50"),
        ("disp_2s_vs_disp_1s_b65",      "v23_disp_2s_L3__b6.50",      "v23_disp_1s_L3__b6.50"),
        ("disp_3s_vs_disp_1s_b65",      "v23_disp_3s_L3__b6.50",      "v23_disp_1s_L3__b6.50"),
        ("disp_3s_vs_disp_1s_b70",      "v23_disp_3s_L3__b7.00",      "v23_disp_1s_L3__b7.00"),
        ("disp_3s_vs_cc3s_111_b65",     "v23_disp_3s_L3__b6.50",      "v22_cc3s_X3r1Z3r1X3r1__b6.50"),
        ("disp_3s_vs_cc3s_121_b65",     "v23_disp_3s_L3__b6.50",      "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("disp_3s_vs_cc3s_121_b70",     "v23_disp_3s_L3__b7.00",      "v23_cc3s_121_ref__b7.00"),
        ("cc3s_121_vs_cc1s_L3r2_b65",   "v22_cc3s_X3r1Z3r2X3r1__b6.50","v22_cc1s_L3r2__b6.50"),
        ("plan_RX3_vs_plan_2dL12_b65",  "v23_plan_cc2s_RX3__b6.50",   "v23_plan_cc2d_L12__b6.50"),
        ("plan_disp2s_vs_2dL12_b65",    "v23_plan_disp_2s_L3__b6.50", "v23_plan_cc2d_L12__b6.50"),
        ("plan_R3_vs_plan_2dL12_b65",   "v23_plan_cc2s_R3__b6.50",    "v23_plan_cc2d_L12__b6.50"),
        ("plan_RX3_vs_tor_RX3_b65",     "v23_plan_cc2s_RX3__b6.50",   "v22_cc2s_RX3__b6.50"),
    ]
    for cname, kn, kd in c_list:
        if kn in all_arms and kd in all_arms:
            contrasts[cname] = {
                "bposd": indep_ratio(all_arms[kn], all_arms[kd], "bposd"),
                "uf": indep_ratio(all_arms[kn], all_arms[kd], "uf"),
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
