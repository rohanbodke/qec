#!/usr/bin/env python3
"""Parse cluster_v3_results.txt into structured summary tables."""
import os
from collections import defaultdict

WS = "/Users/harshilavlani/Downloads/6.S980_gemini"
IN_PATH = os.path.join(WS, "cluster_v3_results.txt")
OUT_PATH = os.path.join(WS, "parsed_v3_summary.txt")

stats = {}
kmc_uni = defaultdict(dict)
kmc_jgrad = defaultdict(dict)
mfqec = defaultdict(dict)
singleshot = defaultdict(dict)
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
        if line.startswith(("ERR|", "TIMEOUT|")):
            timeouts.append(line)
            continue
        parts = line.split("|", 3)
        suite, label, dt_s, body = parts[0], parts[1], parts[2], parts[3]
        toks = body.split()
        head = toks[0]
        kv = kv_map(toks[1:])

        if suite == "STATS" and head == "STATS":
            stats[label] = kv
        elif suite == "KMC_UNI" and head == "KMC":
            kmc_uni[label][float(kv["beta"])] = kv
        elif suite == "KMC_JGRAD" and head == "KMC":
            kmc_jgrad[label][float(kv["beta"])] = kv
        elif suite == "MFQEC" and head == "MFQEC":
            mfqec[label][float(kv["p_gate"])] = kv
        elif suite == "SINGLESHOT" and head == "SINGLESHOT":
            singleshot[label][float(kv["p_data"])] = kv

lines = []
lines.append(f"=== 0. TIMEOUT / ERR COUNT: {len(timeouts)} ===")
for t in timeouts:
    lines.append("  " + t)

lines.append("\n=== 1. V3 ULTRA-COMPACT 3D-LOCAL & NON-LOCAL CODE STATS (with Meta-Checks M_meta!) ===")
for lbl in sorted(stats.keys()):
    s = stats[lbl]
    n_int, k_int = int(s["N"]), max(1, int(s["K"]))
    lines.append(
        f"  {lbl:38s} | N={s['N']:>4s} K={s['K']:>2s} N/K={n_int/k_int:5.0f} "
        f"M={s['M']:>4s} M_meta={s['M_meta']:>3s} cw={s['max_cw']}({s['mean_cw']}) "
        f"qdeg={s['qdeg']} d<={s['d']:>3s} Delta<={s['Delta']:>2s}"
    )

lines.append("\n=== 2A. UNIFORM KMC LIFETIME & FREE-ENERGY N-SCALING (Beating the Michnicki Trap!) ===")
for lbl in sorted(kmc_uni.keys()):
    row = [f"b={b:.0f}:{kmc_uni[lbl][b]['mean_t']}(c={kmc_uni[lbl][b]['capped']})" for b in sorted(kmc_uni[lbl])]
    lines.append(f"  {lbl:38s} | " + " | ".join(row))

lines.append("\n=== 2B. GRADED CHECK COUPLINGS (J_rib=1.8 vs Uniform J=1.0 — Zero Extra Qubits!) ===")
for lbl in sorted(kmc_jgrad.keys()):
    row_g = [f"b={b:.0f}:Jgrad={kmc_jgrad[lbl][b]['mean_t']}(vs Uni={kmc_uni[lbl].get(b,{}).get('mean_t','?')})" for b in sorted(kmc_jgrad[lbl])]
    lines.append(f"  {lbl:38s} | " + " | ".join(row_g))

lines.append("\n=== 3. DIGITAL GATE-BASED MEASUREMENT-FREE QEC (Hong-Guo-Lucas Nat. Comm. 2025: 150 Sweeps, p_env=0.002) ===")
for lbl in sorted(mfqec.keys()):
    row = [f"pg={pg:g}:Pf={mfqec[lbl][pg]['P_fail']}(syn={mfqec[lbl][pg]['end_syn']})" for pg in sorted(mfqec[lbl])]
    lines.append(f"  {lbl:27s} | " + " | ".join(row))

lines.append("\n=== 4. 20-ROUND SINGLE-SHOT ACTIVE QEC UNDER NOISY SYNDROME MEASUREMENTS (p_data = q_meas) ===")
for lbl in sorted(singleshot.keys()):
    row = [f"pq={pq:.3f}:{singleshot[lbl][pq]['P_fail']}" for pq in sorted(singleshot[lbl])]
    lines.append(f"  {lbl:25s} | " + " | ".join(row))

txt = "\n".join(lines) + "\n"
with open(OUT_PATH, "w") as f:
    f.write(txt)
print(txt)
