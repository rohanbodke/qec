#!/usr/bin/env python3
"""
parse_v24b_xxz_clincher.py
Zero-trust parser and 10,000-draw bootstrap analyzer for Campaign V24b
(3-step pure-L=3 XXZ2 N=110,400 clincher, planar 3-step XXZ, and beta=7.0 extensions).
Outputs `parsed_v24b_xxz_clincher.json`.
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
    eval_iid_arm,
    eval_thermal_arm,
    indep_ratio,
    parse_kv,
)
from parse_v24_degree3_balanced import load_ref_arms

V24B_RAW = ROOT / "cluster_v24b_results.txt"
V24B_SPECS = ROOT / "v24b_code_specs.json"
OUT_JSON = ROOT / "parsed_v24b_xxz_clincher.json"


def main():
    with open(V24B_SPECS) as f:
        specs = json.load(f)

    th_trajs = defaultdict(dict)
    th_grids = {}
    th_betas = {}
    iid_trajs = defaultdict(dict)
    iid_grids = {}
    errors = []

    with open(V24B_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^(v24b_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
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
                m = re.match(r"^v24b_iid__(v24b_.+)_(X_err|Z_err)__o(\d+)$", tid)
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

    print(f"Loaded V24b XXZ Clincher results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112525)
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
        px = "" if bp["t_80_X_crossed"] else "+"
        pz = "" if bp["t_80_Z_crossed"] else "+"
        rat = bp["t_80_X"] / max(1e-12, bp["t_80_Z"])
        print(
            f"  {akey:32s}: N={sp['N']:7,d} K={sp['K']} b={ev['beta']:.2f} n=({ev['n_X']},{ev['n_Z']}) "
            f"pre=({ev['capped_pre_t80_X']},{ev['capped_pre_t80_Z']}) | UF={u['t_80']:9.2e}{pu} "
            f"| BP Q={bp['t_80']:9.2e}{pb} X={bp['t_80_X']:9.2e}{px} Z={bp['t_80_Z']:9.2e}{pz} X/Z={rat:5.1f}x"
        )

    print("\n=== ARRHENIUS SLOPES (beta 5.5 -> 6.5) IN V24b ===")
    for tag in sorted(specs.keys()):
        k55, k65 = f"{tag}__b5.50", f"{tag}__b6.50"
        if k55 in arms and k65 in arms:
            b55 = arms[k55]["decoders"]["bposd"]
            b65 = arms[k65]["decoders"]["bposd"]
            dq = math.log(b65["t_80"] / b55["t_80"])
            dx = math.log(b65["t_80_X"] / b55["t_80_X"])
            dz = math.log(b65["t_80_Z"] / b55["t_80_Z"])
            print(f"  {tag:24s}: dlnQ/db={dq:.2f} dlnX/db={dx:.2f} dlnZ/db={dz:.2f}")

    print("\n=== PAIRED I.I.D. THRESHOLDS ===")
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

    v22_refs = load_ref_arms(
        ROOT / "cluster_v22_results.txt",
        r"^(v22_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$",
        {"v22_cc3s_R3X3r2__b6.50", "v22_cc3s_X3r1Z3r2X3r1__b6.50"},
        2026112583,
    )
    v23_refs = load_ref_arms(
        ROOT / "cluster_v23_results.txt",
        r"^(v23_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$",
        {"v23_plan_cc2s_RX3__b6.50"},
        2026112584,
    )
    v24_refs = load_ref_arms(
        ROOT / "cluster_v24_results.txt",
        r"^(v24_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$",
        {"v24_3s_X2_X2_Z1__b6.50", "v24_plan_2s_X5r4_Z3r1__b6.50"},
        2026112585,
    )
    all_arms = dict(arms)
    all_arms.update(v22_refs)
    all_arms.update(v23_refs)
    all_arms.update(v24_refs)

    print("\n=== V24b VS V22/V23/V24 CONTRASTS (beta=6.5) ===")
    contrasts = {}
    c_list = [
        ("XXZ2_vs_R3X3r2_sameN_b65",        "v24b_3s_X2_X2_Z2__b6.50",      "v22_cc3s_R3X3r2__b6.50"),
        ("XXZ2_vs_XZX121_b65",              "v24b_3s_X2_X2_Z2__b6.50",      "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("XXZ2_vs_XXZ1_b65",                "v24b_3s_X2_X2_Z2__b6.50",      "v24_3s_X2_X2_Z1__b6.50"),
        ("XXZjog1_vs_XZX121_b65",           "v24b_3s_X2_X2_Zjog1__b6.50",   "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("plan_3s_XXZ1_vs_plan_RX3_b65",    "v24b_plan_3s_X2_X2_Z1__b6.50", "v23_plan_cc2s_RX3__b6.50"),
        ("plan_3s_XXZ2_vs_plan_RX3_b65",    "v24b_plan_3s_X2_X2_Z2__b6.50", "v23_plan_cc2s_RX3__b6.50"),
        ("plan_3s_XXZ2_vs_plan_2s_X5_b65",  "v24b_plan_3s_X2_X2_Z2__b6.50", "v24_plan_2s_X5r4_Z3r1__b6.50"),
    ]
    for cname, kn, kd in c_list:
        if kn in all_arms and kd in all_arms:
            contrasts[cname] = {
                "bposd": indep_ratio(all_arms[kn], all_arms[kd], "bposd"),
                "uf": indep_ratio(all_arms[kn], all_arms[kd], "uf"),
            }
            rb = contrasts[cname]["bposd"]
            print(f"  [{cname:32s}]: BP={rb['ratio']:8.2f}x [{rb['ci95'][0]:.2f}, {rb['ci95'][1]:.2f}]")

    clean_arms = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()}
    with open(OUT_JSON, "w") as f:
        json.dump({
            "errors": errors,
            "specs": specs,
            "arms": clean_arms,
            "iid_arms": iid_arms,
            "contrasts": contrasts,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
