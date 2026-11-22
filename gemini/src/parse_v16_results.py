#!/usr/bin/env python3
"""
parse_v16_results.py
Unified statistical parser combining `cluster_v15_results.txt` and `cluster_v16_results.txt`
with `v15_code_specs.json` and `v16_code_specs.json` into `parsed_v16_data.json`:

  1. Independent-sector bootstrap (B = 2,000 resamples of n_X X-trials and n_Z Z-trials separately).
  2. Strict zero-censoring check (`capped_X == 0` and either `capped_Z == 0` or `max(T_X) < min(T_Z_capped)`),
     exclusion of censored points from log-lifetime slope fits, and strict suppression of any ratio whose
     denominator is censored.
  3. True operational final-readout fidelity P_id(t) = p_X(t) * p_Z(t) across prescribed physical readout
     grids (without intermediate stopping), plus operational storage times t_0.80 and t_0.50 with 95% CIs.
  4. Trajectory-paired recovery-cluster arrival times (A_2, A_6, A_10) with SEs, true conditional post-arrival
     residual survival E[T_X - A_k | A_k < T_X], and unbiased first-excursion return probabilities P_ret,k.
  5. Detailed-balance-preserving zero-energy kinetic causal intervention (`flat` vs matched `seamctrl` across mu).
"""

from collections import defaultdict
import json
import math
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V15_RAW = os.path.join(ROOT, "cluster_v15_results.txt")
V16_RAW = os.path.join(ROOT, "cluster_v16_results.txt")
V15_SPECS = os.path.join(ROOT, "v15_code_specs.json")
V16_SPECS = os.path.join(ROOT, "v16_code_specs.json")
OUT_PATH = os.path.join(ROOT, "parsed_v16_data.json")


def parse_kv(body):
    kv = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1.0 + z * z / n
    cen = (p + z * z / (2.0 * n)) / den
    rad = (z / den) * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return [max(0.0, cen - rad), min(1.0, cen + rad)]


def crossing_time_log10(t_grid, p_curve, thresh):
    """Interpolates first time t where p_curve <= thresh (returns None if p_curve[-1] > thresh)."""
    for g in range(len(t_grid)):
        if p_curve[g] <= thresh:
            if g == 0:
                return float(t_grid[0])
            p0, p1 = float(p_curve[g - 1]), float(p_curve[g])
            if p0 <= p1:
                return float(t_grid[g])
            frac = (p0 - thresh) / (p0 - p1)
            lg0, lg1 = math.log10(t_grid[g - 1]), math.log10(t_grid[g])
            return float(10.0 ** (lg0 + frac * (lg1 - lg0)))
    return None


def compute_joint_stats(tx, cx, tz, cz, span_recs, rng, B=2000):
    tx = np.asarray(tx, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.int32)
    tz = np.asarray(tz, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.int32)

    cap_x = int(cx.sum())
    cap_z = int(cz.sum())
    nx, nz = len(tx), len(tz)

    # Exact zero-censoring test:
    if cap_x == 0 and cap_z == 0:
        zero_censor = True
    elif cap_x == 0 and cap_z > 0:
        # Conservative check: even if unknown which Z trials were capped (in V15 legacy lines),
        # check max(tx) < min(tz_capped) if cz flags known, else max(tx) < min(tz_all):
        if int((cz == 1).sum()) == cap_z and cap_z < nz and np.any(cz == 1):
            min_z_cap = float(np.min(tz[cz == 1]))
        else:
            min_z_cap = float(np.min(tz))
        zero_censor = bool(float(np.max(tx)) < min_z_cap)
    else:
        zero_censor = False

    tau_q = float(np.minimum(tx[:, None], tz[None, :]).mean())

    ix = rng.integers(0, nx, size=(B, nx))
    iz = rng.integers(0, nz, size=(B, nz))
    bx = tx[ix]
    bz = tz[iz]
    b_tau = np.empty(B, dtype=np.float64)
    for b in range(B):
        b_tau[b] = np.minimum(bx[b, :, None], bz[b, None, :]).mean()

    se_q = float(b_tau.std(ddof=1))
    ci_lo, ci_hi = float(np.percentile(b_tau, 2.5)), float(np.percentile(b_tau, 97.5))

    span_summary = None
    if span_recs:
        # Each rec: (t_fail, cap, a2, a6, a10, ret6, ret10)
        unc = [r for r in span_recs if r[1] == 0]
        a2s = np.array([r[2] if r[2] >= 0 else r[0] for r in span_recs], dtype=np.float64)
        a6s = np.array([r[3] if r[3] >= 0 else r[0] for r in span_recs], dtype=np.float64)
        a10s = np.array([r[4] if r[4] >= 0 else r[0] for r in span_recs], dtype=np.float64)

        res6 = np.array([r[0] - r[3] for r in unc if r[3] >= 0 and r[3] < r[0]], dtype=np.float64)
        res10 = np.array([r[0] - r[4] for r in unc if r[4] >= 0 and r[4] < r[0]], dtype=np.float64)
        ret6_vals = [r[5] for r in unc if r[5] in (0, 1)]
        ret10_vals = [r[6] for r in unc if r[6] in (0, 1)]

        def m_se(arr):
            if len(arr) == 0:
                return None, None
            m = float(arr.mean())
            se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return m, se

        m_a2, se_a2 = m_se(a2s)
        m_a6, se_a6 = m_se(a6s)
        m_a10, se_a10 = m_se(a10s)
        m_res6, se_res6 = m_se(res6)
        m_res10, se_res10 = m_se(res10)

        k6, n6 = sum(ret6_vals), len(ret6_vals)
        k10, n10 = sum(ret10_vals), len(ret10_vals)

        span_summary = {
            "n_total": len(span_recs),
            "n_uncensored": len(unc),
            "mean_A2": m_a2, "se_A2": se_a2,
            "mean_A6": m_a6, "se_A6": se_a6,
            "mean_A10": m_a10, "se_A10": se_a10,
            "cond_residual_T_minus_A6": m_res6, "se_residual_6": se_res6, "n_cond_6": int(len(res6)),
            "cond_residual_T_minus_A10": m_res10, "se_residual_10": se_res10, "n_cond_10": int(len(res10)),
            "first_ret6_prob": (k6 / n6 if n6 > 0 else None),
            "first_ret6_counts": [k6, n6],
            "first_ret6_wilson95": wilson_ci(k6, n6),
            "first_ret10_prob": (k10 / n10 if n10 > 0 else None),
            "first_ret10_counts": [k10, n10],
            "first_ret10_wilson95": wilson_ci(k10, n10),
        }

    return {
        "tau_Q": tau_q,
        "se_Q": se_q,
        "ci95_Q": [ci_lo, ci_hi],
        "zero_censor_exact": zero_censor,
        "is_censored": (not zero_censor),
        "tau_X": float(tx.mean()),
        "se_X": float(tx.std(ddof=1) / math.sqrt(nx)) if nx > 1 else 0.0,
        "median_X": float(np.median(tx)),
        "trials_X": nx,
        "capped_X": cap_x,
        "max_TX": float(np.max(tx)),
        "tau_Z": float(tz.mean()),
        "se_Z": float(tz.std(ddof=1) / math.sqrt(nz)) if nz > 1 else 0.0,
        "trials_Z": nz,
        "capped_Z": cap_z,
        "min_TZ": float(np.min(tz)),
        "span_v16": span_summary,
        "_boot_Q": b_tau,
    }


def fit_slope_uncensored(points, beta_min=5.5, beta_max=7.25):
    """Weighted least-squares slope d(ln tau_Q)/d(beta) strictly over UNCENSORED points in [beta_min, beta_max]."""
    pts = [p for p in points if (not p["is_censored"]) and beta_min - 1e-9 <= p["beta"] <= beta_max + 1e-9]
    if len(pts) < 3:
        return None
    x = np.array([p["beta"] for p in pts], dtype=np.float64)
    y = np.array([math.log(p["tau_Q"]) for p in pts], dtype=np.float64)
    # Propagate relative SE: sigma(ln tau) = se_Q / tau_Q
    sig = np.array([max(p["se_Q"] / max(p["tau_Q"], 1e-30), 0.03) for p in pts], dtype=np.float64)
    w = 1.0 / (sig * sig)
    W = np.sum(w)
    xb = np.sum(w * x) / W
    yb = np.sum(w * y) / W
    sxx = np.sum(w * (x - xb) ** 2)
    slope = float(np.sum(w * (x - xb) * (y - yb)) / sxx)
    # Bootstrap slope directly from stored _boot_Q vectors across the same uncensored points:
    B = len(pts[0]["_boot_Q"])
    b_mat = np.log(np.maximum(np.vstack([p["_boot_Q"] for p in pts]), 1e-30))  # (M, B)
    w_col = w[:, None]
    yb_b = np.sum(w_col * b_mat, axis=0) / W
    b_slopes = np.sum(w_col * ((x - xb)[:, None]) * (b_mat - yb_b[None, :]), axis=0) / sxx
    se_slope = float(b_slopes.std(ddof=1))
    return {
        "slope": slope,
        "se": se_slope,
        "ci95": [float(np.percentile(b_slopes, 2.5)), float(np.percentile(b_slopes, 97.5))],
        "betas_used": [p["beta"] for p in pts],
        "n_points": len(pts),
        "_boot_slope": b_slopes,
    }


def main():
    with open(V15_SPECS) as f:
        specs = json.load(f)
    with open(V16_SPECS) as f:
        v16_meta = json.load(f)
    specs.update(v16_meta["codes"])

    # V15 baseline storage: (tag, sec) -> {beta: {"tx": [...], "cx": [...]}}
    v15_sec = defaultdict(dict)
    cadence_data = defaultdict(dict)
    with open(V15_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                continue
            kv = parse_kv(body)
            if tid.startswith("kmc__") or tid.startswith("span__"):
                m = re.match(r"^(?:kmc|span)__(.+)_(X_err|Z_err)__b([0-9.]+)$", tid)
                if not m:
                    continue
                tag, sec, b = m.group(1), m.group(2), float(m.group(3))
                tl = [float(x) for x in kv["t_list"].split(",") if x]
                cap = int(kv["capped"])
                # Mark top `cap` sorted entries as capped in V15 legacy lines:
                idx_s = np.argsort(tl)
                cflags = [0] * len(tl)
                for k in range(cap):
                    cflags[int(idx_s[len(tl) - 1 - k])] = 1
                v15_sec[(tag, sec)][b] = {"t": tl, "c": cflags, "span_recs": None}
            elif tid.startswith("cadence__"):
                m = re.match(r"^cadence__(.+)_X_err__b([0-9.]+)$", tid)
                if m:
                    tag, b = m.group(1), float(m.group(2))
                    cadence_data[tag][b] = {
                        "etas": [0.02, 0.05, 0.20, 0.50],
                        "means": [float(kv["m0"]), float(kv["m05"]), float(kv["m20"]), float(kv["m50"])],
                        "sems": [float(kv["s0"]), float(kv["s05"]), float(kv["s20"]), float(kv["s50"])],
                    }

    # V16 sharded pooling:
    v16_span = defaultdict(lambda: defaultdict(dict))   # tag -> beta -> {off: [recs]}
    v16_kmc = defaultdict(lambda: defaultdict(dict))    # (tag, sec) -> beta -> {off: [(t, c)]}
    v16_frout = defaultdict(lambda: defaultdict(dict))  # (tag, sec, b) -> {off: (grid_spec, [bitstrings])}
    v16_kin = defaultdict(dict)                         # (tag, mkind, mu, b) -> {off: [(t, c)]}
    v16_errors = []

    with open(V16_RAW) as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            tid, body = [x.strip() for x in line.split("|", 1)]
            if body.startswith("ERROR") or body == "TIMEOUT":
                v16_errors.append((tid, body))
                continue
            kv = parse_kv(body)
            if tid.startswith("spanoff__"):
                m = re.match(r"^spanoff__(.+)_X_err__b([0-9.]+)__o(\d+)$", tid)
                tag, b, off = m.group(1), float(m.group(2)), int(m.group(3))
                recs = []
                for item in kv["tr_records"].split(","):
                    if not item:
                        continue
                    p = item.split(":")
                    recs.append((float(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6])))
                v16_span[tag][b][off] = recs
            elif tid.startswith("kmcoff__"):
                m = re.match(r"^kmcoff__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                tag, sec, b, off = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
                tcs = []
                for item in kv["tc_list"].split(","):
                    if item:
                        ts, cs = item.split(":")
                        tcs.append((float(ts), int(cs)))
                v16_kmc[(tag, sec)][b][off] = tcs
            elif tid.startswith("frout__"):
                m = re.match(r"^frout__(.+)_(X_err|Z_err)__b([0-9.]+)__o(\d+)$", tid)
                tag, sec, b, off = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
                v16_frout[(tag, sec, b)][off] = (kv["grid"], [s for s in kv["bits"].split(",") if s])
            elif tid.startswith("kin__"):
                m = re.match(r"^kin__(.+)__(flat|seamctrl)__mu([0-9.]+)__b([0-9.]+)__o(\d+)$", tid)
                tag, mkind, mu, b, off = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), int(m.group(5))
                tcs = []
                for item in kv["tc_list"].split(","):
                    if item:
                        ts, cs = item.split(":")
                        tcs.append((float(ts), int(cs)))
                v16_kin[(tag, mkind, mu, b)][off] = tcs

    # Merge V16 sharded runs (superseding V15 wherever V16 ran the clean sharded version):
    merged_sec = defaultdict(dict)
    for k, bmap in v15_sec.items():
        for b, d in bmap.items():
            merged_sec[k][b] = d

    for tag, bmap in v16_span.items():
        for b, omap in bmap.items():
            recs = []
            for off in sorted(omap):
                recs.extend(omap[off])
            merged_sec[(tag, "X_err")][b] = {
                "t": [r[0] for r in recs],
                "c": [r[1] for r in recs],
                "span_recs": recs,
            }

    for (tag, sec), bmap in v16_kmc.items():
        for b, omap in bmap.items():
            tcs = []
            for off in sorted(omap):
                tcs.extend(omap[off])
            merged_sec[(tag, sec)][b] = {
                "t": [x[0] for x in tcs],
                "c": [x[1] for x in tcs],
                "span_recs": None,
            }

    rng = np.random.default_rng(20261122)
    codes_out = {}
    boot_store = {}
    slopes_out = {}

    for tag in sorted(specs):
        xm = merged_sec.get((tag, "X_err"), {})
        zm = merged_sec.get((tag, "Z_err"), {})
        common_b = sorted(set(xm) & set(zm))
        if not common_b:
            continue
        pts = []
        boot_store[tag] = {}
        for b in common_b:
            st = compute_joint_stats(xm[b]["t"], xm[b]["c"], zm[b]["t"], zm[b]["c"], xm[b]["span_recs"], rng)
            st["beta"] = b
            pts.append(st)
        sl = fit_slope_uncensored(pts, beta_min=5.5, beta_max=7.25)
        for st in pts:
            boot_store[tag][st["beta"]] = st.pop("_boot_Q")
        if sl is not None:
            slopes_out[tag] = {k: v for k, v in sl.items() if k != "_boot_slope"}
            slopes_out[tag]["_boot"] = sl["_boot_slope"]
        codes_out[tag] = {
            "spec": {k: v for k, v in specs[tag].items() if k not in ("wX_hist", "wZ_hist", "deg_hist")},
            "slope_uncensored": ({k: v for k, v in slopes_out[tag].items() if k != "_boot"} if tag in slopes_out else None),
            "points": pts,
        }

    # Strict censoring-aware pairwise ratio helper:
    def ratio_series(tag_num, tag_den):
        p_num = {p["beta"]: p for p in codes_out.get(tag_num, {}).get("points", [])}
        p_den = {p["beta"]: p for p in codes_out.get(tag_den, {}).get("points", [])}
        out_r = []
        for b in sorted(set(p_num) & set(p_den)):
            pn, pd = p_num[b], p_den[b]
            # NEVER report a ratio or lower bound when denominator is censored!
            if pd["is_censored"]:
                out_r.append({
                    "beta": b,
                    "status": "omitted_denominator_censored",
                    "ratio": None,
                    "ci95": None,
                })
                continue
            r_pt = pn["tau_Q"] / pd["tau_Q"]
            br = boot_store[tag_num][b] / np.maximum(boot_store[tag_den][b], 1e-30)
            ci = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
            out_r.append({
                "beta": b,
                "status": ("lower_bound_numerator_censored" if pn["is_censored"] else "uncensored"),
                "ratio": float(r_pt),
                "ci95": ci,
                "lcb95": ci[0],
            })
        return out_r

    comparisons = {
        "1cyc_planar_R3_over_C3": ratio_series("v15_sq_R3", "v15_sq_C3"),
        "1cyc_torus_R3_over_C3": ratio_series("v15_tor_R3", "v15_tor_C3"),
        "1cyc_planar_C2_over_C3": ratio_series("v15_sq_C2", "v15_sq_C3"),
        "1cyc_torus_C2_over_C3": ratio_series("v15_tor_C2", "v15_tor_C3"),
        "L3_planar_CR_over_RC": ratio_series("v15_sq_L3_CR", "v15_sq_L3_RC"),
        "L3_planar_RC_over_CR": ratio_series("v15_sq_L3_RC", "v15_sq_L3_CR"),
        "L3_planar_CR_over_RR": ratio_series("v15_sq_L3_CR", "v15_sq_L3_RR"),
        "L3_torus_CR_over_RC": ratio_series("v15_tor_L3_CR", "v15_tor_L3_RC"),
        "L3_torus_RC_over_CR": ratio_series("v15_tor_L3_RC", "v15_tor_L3_CR"),
        "L3_torus_CR_over_RR": ratio_series("v15_tor_L3_CR", "v15_tor_L3_RR"),
        "pkg_planar_RC_over_CR": ratio_series("v15_sq_pkg_RC", "v15_sq_pkg_CR"),
        "pkg_planar_RC_over_1cyc_R3": ratio_series("v15_sq_pkg_RC", "v15_sq_R3"),
        "pkg_planar_CR_over_1cyc_R3": ratio_series("v15_sq_pkg_CR", "v15_sq_R3"),
        "pkg_planar_RC_over_CC": ratio_series("v15_sq_pkg_RC", "v15_sq_pkg_CC"),
        "pkg_torus_RC_over_CR": ratio_series("v15_tor_pkg_RC", "v15_tor_pkg_CR"),
        "pkg_torus_RC_over_1cyc_R3": ratio_series("v15_tor_pkg_RC", "v15_tor_R3"),
        "pkg_torus_CR_over_1cyc_R3": ratio_series("v15_tor_pkg_CR", "v15_tor_R3"),
        "pkg_rottor_RC_over_CR": ratio_series("v15_rottor_pkg_RC", "v15_rottor_pkg_CR"),
        "C3C2_planar_C3C2_over_C2C3": ratio_series("v16_sq_C3C2", "v16_sq_C2C3"),
        "C3C2_torus_C3C2_over_C2C3": ratio_series("v16_tor_C3C2", "v16_tor_C2C3"),
        "rib2_value_planar_R3C2_over_C3C2": ratio_series("v15_sq_pkg_RC", "v16_sq_C3C2"),
        "rib2_value_torus_R3C2_over_C3C2": ratio_series("v15_tor_pkg_RC", "v16_tor_C3C2"),
    }

    # Slope-difference comparisons WITH uncertainties on BOTH two-cycle and single-cycle quantities:
    def slope_diff(tag_a, tag_b):
        if tag_a not in slopes_out or tag_b not in slopes_out:
            return None
        sa, sb = slopes_out[tag_a], slopes_out[tag_b]
        diff_b = sa["_boot"] - sb["_boot"]
        return {
            "diff": float(sa["slope"] - sb["slope"]),
            "se": float(diff_b.std(ddof=1)),
            "ci95": [float(np.percentile(diff_b, 2.5)), float(np.percentile(diff_b, 97.5))],
            "betas_a": sa["betas_used"],
            "betas_b": sb["betas_used"],
        }

    slope_comparisons = {
        "L3_planar": {"two_cycle": slope_diff("v15_sq_L3_RC", "v15_sq_L3_CR"), "one_cycle": slope_diff("v15_sq_R3", "v15_sq_C3")},
        "L3_torus": {"two_cycle": slope_diff("v15_tor_L3_RC", "v15_tor_L3_CR"), "one_cycle": slope_diff("v15_tor_R3", "v15_tor_C3")},
        "C3C2_planar": {"two_cycle": slope_diff("v16_sq_C3C2", "v16_sq_C2C3"), "one_cycle": slope_diff("v15_sq_C3", "v15_sq_C2")},
        "C3C2_torus": {"two_cycle": slope_diff("v16_tor_C3C2", "v16_tor_C2C3"), "one_cycle": slope_diff("v15_tor_C3", "v15_tor_C2")},
        "pkg_planar": {"two_cycle": slope_diff("v15_sq_pkg_RC", "v15_sq_pkg_CR"), "one_cycle": slope_diff("v15_sq_R3", "v15_sq_C2")},
        "pkg_torus": {"two_cycle": slope_diff("v15_tor_pkg_RC", "v15_tor_pkg_CR"), "one_cycle": slope_diff("v15_tor_R3", "v15_tor_C2")},
        "pkg_rottor": {"two_cycle": slope_diff("v15_rottor_pkg_RC", "v15_rottor_pkg_CR"), "one_cycle": slope_diff("v15_rottor_R3", "v15_rottor_C2")},
    }

    # Operational Final-Readout parsing (`final_readout`):
    final_readout_out = {}
    frout_tags = sorted({tag for (tag, _, _) in v16_frout})
    for tag in frout_tags:
        betas_f = sorted({b for (t0, _, b) in v16_frout if t0 == tag})
        final_readout_out[tag] = {}
        for b in betas_f:
            if (tag, "X_err", b) not in v16_frout or (tag, "Z_err", b) not in v16_frout:
                continue
            gx_map = v16_frout[(tag, "X_err", b)]
            gz_map = v16_frout[(tag, "Z_err", b)]
            gspec = next(iter(gx_map.values()))[0]
            lg0, lg1, n_g = [float(x) for x in gspec.split(":")]
            n_g = int(n_g)
            t_grid = np.array([10.0 ** (lg0 + (lg1 - lg0) * i / (n_g - 1)) for i in range(n_g)], dtype=np.float64)

            xbits, zbits = [], []
            for off in sorted(gx_map):
                xbits.extend(gx_map[off][1])
            for off in sorted(gz_map):
                zbits.extend(gz_map[off][1])

            cap_fx = sum(1 for s in xbits if "C" in s)
            cap_fz = sum(1 for s in zbits if "C" in s)
            X_mat = np.array([[1.0 if ch == "1" else 0.0 for ch in s] for s in xbits], dtype=np.float64)
            Z_mat = np.array([[1.0 if ch == "1" else 0.0 for ch in s] for s in zbits], dtype=np.float64)

            px = X_mat.mean(axis=0)
            pz = Z_mat.mean(axis=0)
            pid = px * pz

            # Count healed crossings ('01' pattern in X_mat rows):
            healed_trials_x = sum(1 for s in xbits if "01" in s.replace("C", ""))

            B = 2000
            ix = rng.integers(0, len(X_mat), size=(B, len(X_mat)))
            iz = rng.integers(0, len(Z_mat), size=(B, len(Z_mat)))
            b_pid = X_mat[ix].mean(axis=1) * Z_mat[iz].mean(axis=1)  # (B, n_g)
            ci_lo = np.percentile(b_pid, 2.5, axis=0)
            ci_hi = np.percentile(b_pid, 97.5, axis=0)

            t50 = crossing_time_log10(t_grid, pid, 0.50)
            t75 = crossing_time_log10(t_grid, pid, 0.75)
            t80 = crossing_time_log10(t_grid, pid, 0.80)
            t85 = crossing_time_log10(t_grid, pid, 0.85)
            b_t50_v = [v for v in (crossing_time_log10(t_grid, b_pid[k], 0.50) for k in range(B)) if v is not None]
            b_t75_v = [v for v in (crossing_time_log10(t_grid, b_pid[k], 0.75) for k in range(B)) if v is not None]
            b_t80_v = [v for v in (crossing_time_log10(t_grid, b_pid[k], 0.80) for k in range(B)) if v is not None]
            b_t85_v = [v for v in (crossing_time_log10(t_grid, b_pid[k], 0.85) for k in range(B)) if v is not None]

            final_readout_out[tag][f"{b:.2f}"] = {
                "beta": b,
                "n_X": len(X_mat),
                "n_Z": len(Z_mat),
                "capped_X": cap_fx,
                "capped_Z": cap_fz,
                "healed_crossing_trials_X": healed_trials_x,
                "t_grid": t_grid.tolist(),
                "p_X": px.tolist(),
                "p_Z": pz.tolist(),
                "P_id": pid.tolist(),
                "P_id_ci95_lo": ci_lo.tolist(),
                "P_id_ci95_hi": ci_hi.tolist(),
                "t_85": t85,
                "t_85_ci95": ([float(np.percentile(b_t85_v, 2.5)), float(np.percentile(b_t85_v, 97.5))] if len(b_t85_v) >= int(0.90 * B) else None),
                "t_80": t80,
                "t_80_ci95": ([float(np.percentile(b_t80_v, 2.5)), float(np.percentile(b_t80_v, 97.5))] if len(b_t80_v) >= int(0.90 * B) else None),
                "t_75": t75,
                "t_75_ci95": ([float(np.percentile(b_t75_v, 2.5)), float(np.percentile(b_t75_v, 97.5))] if len(b_t75_v) >= int(0.90 * B) else None),
                "t_50": t50,
                "t_50_ci95": ([float(np.percentile(b_t50_v, 2.5)), float(np.percentile(b_t50_v, 97.5))] if len(b_t50_v) >= int(0.90 * B) else None),
            }

    # Kinetic Zero-Energy Causal Intervention parsing (`kinetic_zero_e`):
    kinetic_out = defaultdict(lambda: defaultdict(list))
    kin_boot = {}
    for (tag, mkind, mu, b) in sorted(v16_kin):
        omap = v16_kin[(tag, mkind, mu, b)]
        tcs = []
        for off in sorted(omap):
            tcs.extend(omap[off])
        ts = np.array([x[0] for x in tcs], dtype=np.float64)
        cs = np.array([x[1] for x in tcs], dtype=np.int32)
        B = 2000
        bt = ts[rng.integers(0, len(ts), size=(B, len(ts)))].mean(axis=1)
        kin_boot[(tag, mkind, mu, b)] = bt
        kinetic_out[tag][mkind].append({
            "mu": mu,
            "beta": b,
            "trials": len(ts),
            "capped": int(cs.sum()),
            "tau_X": float(ts.mean()),
            "se_X": float(ts.std(ddof=1) / math.sqrt(len(ts))) if len(ts) > 1 else 0.0,
            "ci95_X": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
        })

    # Compute RC/CR ratio across mu under `flat` vs `seamctrl`:
    kinetic_ratios = {}
    for prefix, rc_tag, cr_tag in [
        ("planar_L3", "v15_sq_L3_RC", "v15_sq_L3_CR"),
        ("torus_L3", "v15_tor_L3_RC", "v15_tor_L3_CR"),
    ]:
        kinetic_ratios[prefix] = {}
        for mkind in ("flat", "seamctrl"):
            rows = []
            for mu in [0.01, 0.03, 0.10, 0.30, 1.00, 3.00]:
                k_rc = (rc_tag, mkind, mu, 6.50)
                k_cr = (cr_tag, mkind, mu, 6.50)
                if k_rc in kin_boot and k_cr in kin_boot:
                    b_r = kin_boot[k_rc] / np.maximum(kin_boot[k_cr], 1e-30)
                    r_pt = float(kin_boot[k_rc].mean() / kin_boot[k_cr].mean())
                    rows.append({
                        "mu": mu,
                        "RC_over_CR": r_pt,
                        "ci95": [float(np.percentile(b_r, 2.5)), float(np.percentile(b_r, 97.5))],
                    })
            kinetic_ratios[prefix][mkind] = rows

    out_doc = {
        "v16_errors": v16_errors,
        "qmasks": v16_meta["qmasks"],
        "codes": codes_out,
        "comparisons": comparisons,
        "slope_comparisons": slope_comparisons,
        "final_readout": final_readout_out,
        "kinetic_zero_e": dict(kinetic_out),
        "kinetic_rc_over_cr": kinetic_ratios,
        "cadence": dict(cadence_data),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"Wrote {OUT_PATH} (errors={len(v16_errors)}, codes={len(codes_out)})")


if __name__ == "__main__":
    main()
