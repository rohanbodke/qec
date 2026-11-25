# Selective Simplification and Sector Rebalancing of Finite Welded Quantum Memories

Companion CSS code generators, exact stabilizer-quotient and low-temperature resistor-network solvers, algebraic and barrier certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation and paired four-decoder engines, frozen local-interface predictions, cross-family finite-size benchmarks, raw fixed-time readout trajectories, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Starting from the proof-motivated recursive sheet-welding framework of Balasubramanian, Davydova, and Lin (BDL Section 6.2), we formulate finite simplification as an explicit operational optimization problem under independent single-qubit Pauli $X/Z$ Glauber heat-bath dynamics $r(\Delta E) = (1 + e^{\beta \Delta E})^{-1}$ and single-shot readout decoding:

$$\min_{\mathcal{A} \in \mathcal{F}} N(\mathcal{A}) \quad \text{subject to} \quad P_{\mathrm{rec}}^{\mathcal{A}}(t_*; \beta) \ge 0.80,$$

preserving exact CSS commutation $H_X H_Z^T = 0 \pmod 2$ and the seed's rank-derived logical qubit count $K$.

1. **Compact Baseline Anchor, Scale Asymmetry, and Null Baselines (`src/reanalyze_existing_v16_v17.py`):**
   - At inverse temperature $\beta = 6.5$, all-compact one-bridge $C_2 C_2$ ($25{,}148$ qubits) removes $96.46\%$ of the physical qubits of uniform two-bridge $R_3 R_3$ ($709{,}408$ qubits) while extending $t_{0.80}$ by $13.04\times$ ($95\%$ bootstrap CI $[8.17, 18.86]$) on planar seeds and $13.45\times$ ($[8.61, 17.01]$) on periodic tori, exceeding all six fine-$C_3$/fine-$R_3$ two-cycle codes by at least $5.99\times$ ($[3.85, 9.48]$) planar and $9.85\times$ ($[5.93, 12.64]$) torus across $p \in \{0.70, 0.75, 0.80, 0.85, 0.90\}$.

2. **Exact Stabilizer Quotients, Bridge-Arm Pruning, and Protected-Subsystem Test (`src/verify_local_quotients_and_algebra.py`):**
   - Quotienting physical $X$ errors by $\mathrm{rowspace}(H_X)$ ($\dim = N - \mathrm{rank}\,H_X$) yields exact reversible identity $T_{\mathrm{clean}} = Z/(2\kappa) = Z/(2\Gamma\bar{h})$, rigorous energy-truncated capacity brackets on one-step $C_3$ and $R_3$, and exact rational low-temperature prefactors.
   - In two-step $C_2$ at $\beta = 6.5$, $Z$-step and doubly pruned variants ($N = 228, 222$) introduce eight unintended logical modes ($K_{\mathrm{total}} = 9$) and collapse $62\text{--}68\times$ via an exact barrier-$(1, 1)$ $Z$ sequence $0 \to 1 \to 1 \to 1 \to 0$ even as a $K_{\mathrm{target}} = 1$ subsystem.

3. **Post-Crossing Return, Matched $L=4$ Placement, and Prospective Interface-Guided Simplification (`src/freeze_interface_predictor.py`, `src/parse_v18_rework.py`):**
   - Freezing a parameter-free collared $3 \times 3$ local interface predictor (`frozen_interface_predictions.json`, SHA-256 `36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`) before running held-out two-cycle simulations prospectively identifies strictly better sector-selective simplifications: $C_2 R_{X3}$ ($99{,}580$ qubits, $-27.1\%$ qubits) increases $\beta = 6.5$ storage over $C_2 R_3$ by $1.98\times$ ($[1.31, 3.53]$), while coarse $R_{X3} C_2$ ($116{,}164$ qubits, $\rho \approx 40{,}965 < 36^3$) reaches $t_{0.80} = 2.63 \times 10^7$ at $\beta = 7.0$, exceeding $R_3 C_2$ by $5.01\times$ ($[1.66, 10.4]$).

4. **Cross-Scale Sector Rebalancing and Palindromic Welding (`src/verify_rebalanced_algebra.py`, `cluster_v9/build_v19_rebalance_and_decoders.py`):**
   - Three-step $(X, Z, X)$ hierarchies ($N = 1{,}942\text{--}10{,}580$) flip the bottleneck to $Z$: $X_2 R_{3Z} X_2$ ($4{,}566$ qubits) beats matched-size $R_{3X} Z_2 X_2$ ($4{,}556$ qubits) by $3.53\times$ ($[1.63, 6.41]$) at $\beta = 6.5$ (matching $C_2 R_{X3}$ at $21.8\times$ fewer qubits), while $R_3 X_2$ ($10{,}580$ qubits) matches $C_2 C_2$ at $57.9\%$ fewer qubits.
   - Ending a four-step $L = 2$ hierarchy on a single-qubit $X$-weld cuts $18.6\%$ of the qubits in palindromic $C_2 C_2^{ZX}$ (`XZZX`, $N = 20{,}476$) and raises the fine module's static barriers from $(2, 2)$ to $(3, 2)$, extending $\beta = 6.5$ storage by $101\times$ ($[55.9, 205]$; $1.47 \times 10^7$) on planar seeds and $\ge 667\times$ ($\ge 1.00 \times 10^8$) on periodic tori, while continuable $C_3 C_2^{ZX}$ ($87{,}652$ qubits) and $R_{Z3} C_2^{ZX}$ ($100{,}900$ qubits) exceed $3.3 \times 10^7$ and reverse the coarse bridge winner.

5. **Physical Basin Crossing versus Single-Shot Readout Decoding (`cluster_v9/kmc_engine_v19_decoders.cpp`, `src/parse_v19_rebalance_and_decoders.py`):**
   - Paired evaluation of `UF`, `Cool-UF`, `Hier-UF`, and `BP-OSD` on identical error states shows that `BP-OSD` boosts unrelaxed single-shot bit-flip thresholds $p_{0.80}$ by $2.34\text{--}3.03\times$ over `UF`, yet agrees with `UF` within a few percent under cold Glauber storage as globally quenchable wrong-basin states rise to $67\text{--}86\%$.

6. **Finite-Size Performance Across Code Families (`cluster_v9/build_v20_benchmarks.py`, `src/parse_v20_benchmarks.py`):**
   - Benchmarking against 2D toric ($L = 8\text{--}64$), single-scale 3D welded ($L = 2\text{--}8$), Haah's 3D Cubic Code 1 ($L = 5\text{--}17$), and 4D toric ($L = 3\text{--}5$) codes under `BP-OSD` separates local check degree ($\bar{d}_{\mathrm{tot}} = 8$ on cubic and 4D toric codes vs. $4.0\text{--}4.7$ on surface and welded codes) from multiscale geometry.
   - While raw $J = 1$ unit check couplings favor weight-$8$, degree-$8$ cubic codes by doubling per-qubit energy ($\beta_{\mathrm{q}} = \beta\bar{d}_{\mathrm{tot}}/4 = 13.0$), comparing at equal Hamiltonian coupling per physical qubit ($\beta_{\mathrm{q}} = 6.5$ and $7.55$) shows single-scale welded ($2{,}111 \to 202$) and cubic ($2{,}933 \to 1{,}602$ across $L = 7 \to 17$, ratio $1.83\times$ $[1.30, 2.71]$) codes turning downward with $N$, whereas rebalanced welded hierarchies climb to $N = 87{,}652$ and exceed Haah $L = 7$ by $12.4\times$ ($[7.50, 20.7]$) planar and $22.6\times$ ($[14.6, 40.5]$) torus at $\beta_{\mathrm{q}} = 6.5$ ($126\text{--}701\times$ at $\beta_{\mathrm{q}} = 7.55$).

## Reproducing All Suites

```bash
# 1. Existing-data reanalysis, exact stabilizer quotients, and rebalanced algebra/barrier certificates
python3 src/reanalyze_existing_v16_v17.py
python3 src/verify_local_quotients_and_algebra.py
python3 src/verify_rebalanced_algebra.py

# 2. Build V18 (20 pairs), V19 (29 pairs), and V20 (25 cross-family benchmark pairs)
python3 cluster_v9/build_v18_rework.py
python3 cluster_v9/build_v19_rebalance_and_decoders.py
python3 cluster_v9/build_v20_benchmarks.py

# 3. Parse V18 (424 shards), V19 (1,656 shards), and V20 (502 benchmark shards) with 10,000-draw bootstrap CIs
python3 src/parse_v18_rework.py
python3 src/parse_v19_rebalance_and_decoders.py
python3 src/parse_v20_benchmarks.py

# 4. Render Figures 1-7 and compile REVTeX 4.2 manuscripts
python3 src/render_v23_prx_figures.py
python3 src/render_v24_prx_figures.py
python3 src/render_v25_benchmarks.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key Files

- `paper/main.tex`, `paper/main.pdf`, `paper/references.bib`: 16-page *Physical Review X*-formatted manuscript with Appendices A–D.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `frozen_interface_predictions.json`, `src/freeze_interface_predictor.py`: Pre-simulation frozen collared $3 \times 3$ interface predictor (`SHA-256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`).
- `cluster_v9/kmc_engine_v19_decoders.cpp`, `cluster_v9/build_v19_rebalance_and_decoders.py`, `cluster_v9/build_v20_benchmarks.py`: Paired four-decoder C++20 engine, rebalanced code builder, and five-family benchmark builder.
- `cluster_v19_results.txt`, `cluster_v20_results.txt`, `src/parse_v19_rebalance_and_decoders.py`, `src/parse_v20_benchmarks.py`: Raw trajectory logs (`1,656` + `502` tasks, `0` errors) and $10{,}000$-draw bootstrap parsers.
- `src/render_v23_prx_figures.py`, `src/render_v24_prx_figures.py`, `src/render_v25_benchmarks.py`, `figures/`: Publication figure generators and vector graphics.
