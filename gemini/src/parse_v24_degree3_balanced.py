#!/usr/bin/env python3
"""
parse_v24_degree3_balanced.py
Zero-trust parser and 10,000-draw bootstrap analyzer for Campaign V24 (degree-3
X/Z-balanced welded 4.8.8 color codes). Outputs `parsed_v24_degree3_balanced.json`.
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

V24_RAW = ROOT / "cluster_v24_results.txt"
V24_SPECS = ROOT / "v24_code_specs.json"
OUT_JSON = ROOT / "parsed_v24_degree3_balanced.json"


def load_ref_arms(raw_path, prefix_pat, target_keys, rng_seed):
    if not raw_path.exists():
        return {}
    trajs, grids = defaultdict(dict), {}
    with open(raw_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if not body.startswith("PAIRED_READOUT"):
                continue
            m = re.match(prefix_pat, tid)
            if not m:
                continue
            tag, sec, b_s, off_s = m.groups()
            akey = f"{tag}__b{b_s}"
            if akey not in target_keys:
                continue
            off = int(off_s)
            kv = parse_kv(body)
            grids[akey] = kv["grid"]
            split_d = {d: kv[f"bits_{d}"].split(",") for d in DECS}
            for i in range(len(split_d["uf"])):
                trajs[(akey, sec)][off + i] = {d: split_d[d][i] for d in DECS}
    rng = np.random.default_rng(rng_seed)
    out = {}
    for akey in sorted(grids.keys()):
        xm = trajs.get((akey, "X_err"), {})
        zm = trajs.get((akey, "Z_err"), {})
        if xm and zm:
            out[akey] = eval_thermal_arm(xm, zm, grids[akey], rng, B=10000)
    return out


def main():
    with open(V24_SPECS) as f:
        specs = json.load(f)

    th_trajs = defaultdict(dict)
    th_grids = {}
    th_betas = {}
    iid_trajs = defaultdict(dict)
    iid_grids = {}
    errors = []

    with open(V24_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                errors.append((tid, body))
                continue
            if body.startswith("PAIRED_READOUT"):
                m = re.match(r"^(v24_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
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
                m = re.match(r"^v24_iid__(v24_.+)_(X_err|Z_err)__o(\d+)$", tid)
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

    print(f"Loaded V24 Degree-3 Balanced results (errors={len(errors)})")
    for e in errors[:10]:
        print("  ERR:", e)

    rng = np.random.default_rng(2026112524)
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
            f"  {akey:30s}: N={sp['N']:6,d} K={sp['K']} b={ev['beta']:.2f} n=({ev['n_X']},{ev['n_Z']}) "
            f"pre=({ev['capped_pre_t80_X']},{ev['capped_pre_t80_Z']}) | UF={u['t_80']:9.2e}{pu} "
            f"| BP Q={bp['t_80']:9.2e}{pb} X={bp['t_80_X']:9.2e}{px} Z={bp['t_80_Z']:9.2e}{pz} X/Z={rat:6.1f}x"
        )

    print("\n=== ARRHENIUS SLOPES (beta 5.5 -> 6.5) IN V24 ===")
    for tag in sorted(specs.keys()):
        k55, k65 = f"{tag}__b5.50", f"{tag}__b6.50"
        if k55 in arms and k65 in arms:
            b55 = arms[k55]["decoders"]["bposd"]
            b65 = arms[k65]["decoders"]["bposd"]
            dq = math.log(b65["t_80"] / b55["t_80"])
            dx = math.log(b65["t_80_X"] / b55["t_80_X"])
            dz = math.log(b65["t_80_Z"] / b55["t_80_Z"])
            print(
                f"  {tag:24s}: dlnQ/db={dq:.2f} dlnX/db={dx:.2f} dlnZ/db={dz:.2f} "
                f"(Q65={b65['t_80']:.2e}, X65={b65['t_80_X']:.2e}, Z65={b65['t_80_Z']:.2e})"
            )

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
        {
            "v22_cc2s_RX3__b6.50",
            "v22_cc2s_R3__b6.50",
            "v22_cc3s_X3r1Z3r2X3r1__b6.50",
        },
        2026112581,
    )
    v23_refs = load_ref_arms(
        ROOT / "cluster_v23_results.txt",
        r"^(v23_.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$",
        {"v23_plan_cc2s_RX3__b6.50"},
        2026112582,
    )
    all_arms = dict(arms)
    all_arms.update(v22_refs)
    all_arms.update(v23_refs)

    print("\n=== V24 VS V22/V23 CONTRASTS (beta=6.5) ===")
    contrasts = {}
    c_list = [
        ("X4r3_Z3r1_vs_cc2s_RX3_b65",   "v24_2s_X4r3_Z3r1__b6.50",      "v22_cc2s_RX3__b6.50"),
        ("X4r3_Z3r1_vs_cc2s_R3_b65",    "v24_2s_X4r3_Z3r1__b6.50",      "v22_cc2s_R3__b6.50"),
        ("X4r3_Z3jog1_vs_cc2s_RX3_b65", "v24_2s_X4r3_Z3jog1__b6.50",    "v22_cc2s_RX3__b6.50"),
        ("X4r3_Z3r2_vs_cc2s_R3_b65",    "v24_2s_X4r3_Z3r2__b6.50",      "v22_cc2s_R3__b6.50"),
        ("X5r4_Z3r1_vs_cc2s_RX3_b65",   "v24_2s_X5r4_Z3r1__b6.50",      "v22_cc2s_RX3__b6.50"),
        ("X5r4_Z3r2_vs_cc2s_R3_b65",    "v24_2s_X5r4_Z3r2__b6.50",      "v22_cc2s_R3__b6.50"),
        ("XXZ_121_vs_XZX_121_b65",      "v24_3s_X1_X2_Z1__b6.50",       "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("XXZ_211_vs_XZX_121_b65",      "v24_3s_X2_X1_Z1__b6.50",       "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("XXZ_221_vs_XZX_121_b65",      "v24_3s_X2_X2_Z1__b6.50",       "v22_cc3s_X3r1Z3r2X3r1__b6.50"),
        ("plan_X4r3_vs_plan_RX3_b65",   "v24_plan_2s_X4r3_Z3r1__b6.50", "v23_plan_cc2s_RX3__b6.50"),
        ("plan_X5r4_vs_plan_RX3_b65",   "v24_plan_2s_X5r4_Z3r1__b6.50", "v23_plan_cc2s_RX3__b6.50"),
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
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
