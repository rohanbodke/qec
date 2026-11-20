#!/usr/bin/env python3
"""Parse cluster_v2_results.txt into structured summary tables."""
import os
from collections import defaultdict

BASE = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_PATH = os.path.join(BASE, "cluster_v2_results.txt")
OUT_PATH = os.path.join(BASE, "parsed_v2_summary.txt")

stats = {}
phase = defaultdict(dict)
new_q = defaultdict(dict)
coupled = defaultdict(dict)
dec5_th = defaultdict(lambda: defaultdict(dict))
dec5_kmc = defaultdict(lambda: defaultdict(dict))
anneal_inf = defaultdict(dict)
anneal_heal = defaultdict(dict)
goldilocks = defaultdict(lambda: defaultdict(dict))
timeouts = []

def kv_map(tokens):
    d = {}
    for t in tokens:
        if "<=" in t:
            k, v = t.split("<=", 1)
            d[k] = v
        elif "=" in t:
            k, v = t.split("=", 1)
            d[k] = v
    return d

with open(IN_PATH) as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("TIMEOUT|", "ERR|")):
            timeouts.append(line)
            continue
        parts = line.split("|")
        suite, label, dt_str, body = parts[0], parts[1], parts[2], parts[3]
        toks = body.split()
        head = toks[0]
        kv = kv_map(toks[1:])

        if suite == "STATS" and head == "STATS":
            stats[label] = kv
        elif suite == "PHASE" and head == "KMC":
            phase[label][float(kv["beta"])] = kv
        elif suite == "NEW_Q" and head == "KMC":
            new_q[label][float(kv["beta"])] = kv
        elif suite == "COUPLED" and head == "KMC_COUPLED":
            coupled[label][float(kv["beta"])] = kv
        elif suite == "DEC5_TH" and head == "THRESHOLD":
            dec5_th[label][kv["dec"]][float(kv["p"])] = kv
        elif suite == "DEC5_KMC" and head == "KMC":
            dec5_kmc[label][kv["dec"]][float(kv["beta"])] = kv
        elif suite == "ANNEAL_INF" and head == "ANNEAL":
            anneal_inf[(label, int(kv["stages"]))][float(kv["T"])] = kv
        elif suite == "ANNEAL_HEAL" and head == "ANNEAL":
            anneal_heal[(label, int(kv["stages"]))][float(kv["p_init"])] = kv
        elif suite == "GOLDILOCKS" and head == "HYB_NOISY":
            goldilocks[label][float(kv["p_act"])][float(kv["tau_ec"])] = kv

lines = []
lines.append("=== 0. TIMEOUT / ERR SUMMARY ===")
lines.append(f"Total timeouts/errors: {len(timeouts)}")
for t in timeouts:
    lines.append("  " + t)

lines.append("\n=== 1. NEW V2 CODE STATS (N, K, Check Weights, Distance, Energy Barrier) ===")
for lbl in sorted(stats.keys()):
    s = stats[lbl]
    lines.append(
        f"  {lbl:32s} | N={s['N']:>5s} M={s['M_check']:>5s} K={s['K']:>2s} "
        f"max_cw={s['max_cw']:>2s} mean_cw={s['mean_cw']:>4s} qdeg={s['max_q_deg']:>2s} "
        f"d<={s['d']:>3s} Delta<={s['Delta']:>2s}"
    )

lines.append("\n=== 2. UNCAPPED 5-LEVEL PHASE CROSSING (Step 0, 2, 4 up to N=40,000!) ===")
for lbl in sorted(phase.keys()):
    row = [f"{b}:{phase[lbl][b]['mean_t']}(c={phase[lbl][b]['capped']})" for b in sorted(phase[lbl])]
    lines.append(f"  {lbl:24s} | " + " | ".join(row))

lines.append("\n=== 3. NEW QUANTUM ARCHITECTURES KMC LIFETIME (RP^2 K=1, Sec 3.2 ell=3, hyb21 vs hyb12, Torus L3r2 N=8608) ===")
for lbl in sorted(new_q.keys()):
    row = [f"b={b:.0f}:{new_q[lbl][b]['mean_t']}(c={new_q[lbl][b]['capped']})" for b in sorted(new_q[lbl])]
    lines.append(f"  {lbl:30s} | " + " | ".join(row))

lines.append("\n=== 4. COUPLED DEPOLARIZING DAVIES BATH (X, Y, Z simultaneous: y_scale=0.0 vs 1.0!) ===")
for lbl in sorted(coupled.keys()):
    row = [f"b={b:.0f}:{coupled[lbl][b]['mean_t']}(c={coupled[lbl][b]['capped']})" for b in sorted(coupled[lbl])]
    lines.append(f"  {lbl:26s} | " + " | ".join(row))

lines.append("\n=== 5. 5-DECODER ACCURACY + MICROSECOND SPEED BENCHMARK (including NOVEL `rg_soft`!) ===")
for lbl in sorted(dec5_th.keys()):
    lines.append(f"  -- Code: {lbl} --")
    for dec in ("bposd", "rg_soft", "uf", "rg_strict", "rg_relaxed"):
        if dec not in dec5_th[lbl]:
            continue
        row = [f"p={p:.2f}:{dec5_th[lbl][dec][p]['P_L']}({dec5_th[lbl][dec][p]['decode_us']}us)" for p in sorted(dec5_th[lbl][dec])]
        lines.append(f"    {dec:11s} | " + " | ".join(row))
    for dec in ("bposd", "rg_soft", "uf", "rg_strict", "rg_relaxed"):
        if dec in dec5_kmc[lbl]:
            rowk = [f"b={b:.0f}:{dec5_kmc[lbl][dec][b]['mean_t']}" for b in sorted(dec5_kmc[lbl][dec])]
            lines.append(f"    {dec:11s} KMC | " + " | ".join(rowk))

lines.append("\n=== 6A. BDL OUTLOOK #2 SOLVED: SIMULATED ANNEALING (10-stage b=1.5->6.0) vs COLD QUENCH (b=6.0) FROM T=inf ===")
for (lbl, stg) in sorted(anneal_inf.keys()):
    mode_s = "Quench(b=6)" if stg == 1 else "Anneal(1.5->6)"
    row = [
        f"T={T:.0e}:syn={anneal_inf[(lbl,stg)][T]['final_syn']}/Pf={anneal_inf[(lbl,stg)][T]['P_fail_uf']}/trap={anneal_inf[(lbl,stg)][T]['frac_trapped']}"
        for T in sorted(anneal_inf[(lbl, stg)])
    ]
    lines.append(f"  {lbl:26s} {mode_s:15s} | " + " | ".join(row))

lines.append("\n=== 6B. AUTONOMOUS THERMAL SELF-HEALING OF CORRUPTED CODEWORDS (T=5e4) ===")
for (lbl, stg) in sorted(anneal_heal.keys()):
    mode_s = "Quench(b=6)" if stg == 1 else "Anneal(1.5->6)"
    row = [
        f"p0={p0:.2f}:Pf={anneal_heal[(lbl,stg)][p0]['P_fail_uf']}(syn={anneal_heal[(lbl,stg)][p0]['final_syn']},trap={anneal_heal[(lbl,stg)][p0]['frac_trapped']})"
        for p0 in sorted(anneal_heal[(lbl, stg)])
    ]
    lines.append(f"  {lbl:26s} {mode_s:15s} | " + " | ".join(row))

lines.append("\n=== 7. GOLDILOCKS NOISY HYBRID ACTIVE-PASSIVE QEC (T_storage=1000, beta=5.0) ===")
for lbl in sorted(goldilocks.keys()):
    lines.append(f"  -- Code: {lbl} --")
    for pa in sorted(goldilocks[lbl]):
        row = [f"tau={tau:g}:{goldilocks[lbl][pa][tau]['P_fail']}" for tau in sorted(goldilocks[lbl][pa])]
        lines.append(f"    p_act={pa:.3f} | " + " | ".join(row))

text = "\n".join(lines) + "\n"
with open(OUT_PATH, "w") as f:
    f.write(text)
print(text)
