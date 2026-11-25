# Selective Simplification and Sector Rebalancing of Finite Welded Quantum Memories

Companion CSS code generators, exact stabilizer-quotient and low-temperature resistor-network solvers, algebraic and barrier certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation and paired four-decoder engines, frozen local-interface predictions, raw fixed-time readout trajectories, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Starting from the proof-motivated recursive sheet-welding framework of Balasubramanian, Davydova, and Lin (BDL Section 6.2), we formulate finite simplification as an explicit operational optimization problem under independent single-qubit Pauli $X/Z$ Glauber heat-bath dynamics $r(\Delta E) = (1 + e^{\beta \Delta E})^{-1}$ and single-shot readout decoding:

$$\min_{\mathcal{A} \in \mathcal{F}} N(\mathcal{A}) \quad \text{subject to} \quad P_{\mathrm{rec}}^{\mathcal{A}}(t_*; \beta) \ge 0.80,$$

preserving exact CSS commutation $H_X H_Z^T = 0 \pmod 2$ and the seed's rank-derived logical qubit count $K$.

1. **Compact Baseline Anchor, Scale Asymmetry, and Null Baselines (`src/reanalyze_existing_v16_v17.py`):**
   - At inverse temperature $\beta = 6.5$, all-compact one-bridge $C_2 C_2$ ($25{,}148$ qubits) removes $96.46\%$ of the physical qubits of uniform two-bridge $R_3 R_3$ ($709{,}408$ qubits) while extending $t_{0.80}$ by $13.04\times$ ($95\%$ bootstrap CI $[8.17, 18.86]$) on planar seeds and $13.45\times$ ($[8.61, 17.01]$) on periodic tori, exceeding all six fine-$C_3$/fine-$R_3$ two-cycle codes by at least $5.99\times$ ($[3.85, 9.48]$) planar and $9.85\times$ ($[5.93, 12.64]$) torus across $p \in \{0.70, 0.75, 0.80, 0.85, 0.90\}$.
   - Fine-module choice drives $36\text{--}109\times$ variation in $t_{0.80}$ versus $\le 2.4\times$ across coarse modules, inverted relative to standalone $1 \times 1$ lifetimes ($R_3 > C_2$ by $8.6\times$) and retaining $8.6\text{--}29\times$ beyond $1/N$ and $16.80\times$ ($[11.61, 31.07]$) beyond exact clean-state $X$ departure rates $\Gamma_X^{(0)}$ on $C_3 C_2$ versus $C_3 C_3$.

2. **Exact Stabilizer Quotients, Bridge-Arm Pruning, and Protected-Subsystem Test (`src/verify_local_quotients_and_algebra.py`):**
   - Quotienting physical $X$ errors by $\mathrm{rowspace}(H_X)$ ($\dim = N - \mathrm{rank}\,H_X$) yields exact reversible identity $T_{\mathrm{clean}} = Z/(2\kappa) = Z/(2\Gamma\bar{h})$, rigorous energy-truncated capacity brackets on one-step $C_3$ ($[261{,}455, 261{,}464]$) and $R_3$ ($[94.074, 94.076]\times 10^6$), and exact rational low-temperature prefactors ($\frac{10}{3}e^{-2\beta}$, $\frac{423}{119}e^{-2\beta}$, $\frac{258{,}892}{56{,}235}e^{-2\beta}$, $14e^{-3\beta}$, and joint mobility crossover $\frac{3(\lambda+1)}{\lambda+3}e^{-2\beta}$ at $\lambda = \mu e^\beta$).
   - In the complete two-step $C_2$ module at $\beta = 6.5$ ($n_X = 2{,}048, n_Z = 1{,}024$), $X$-step pruning ($N = 248, K = 1$) gives $t_{0.80}$ ratio $0.894$ ($[0.776, 1.036]$), whereas $Z$-step and doubly pruned variants ($N = 228, 222$) introduce eight extra logical modes ($K_{\mathrm{total}} = 9$) and collapse $62\text{--}68\times$ via an exact barrier-$(1, 1)$ $Z$ sequence $0 \to 1 \to 1 \to 1 \to 0$ even as a $K_{\mathrm{target}} = 1$ subsystem.

3. **Post-Crossing Return, Matched $L=4$ Placement, and Prospective Interface-Guided Simplification (`src/freeze_interface_predictor.py`, `src/parse_v18_rework.py`):**
   - Resource-matched $L = 4$ offset ($1+3$) versus centered ($2+2$) placements confirm return-driven compression both standalone ($N = 2{,}926$: paired $t_{0.80}$ ratio $1.66$ $[0.89, 3.12]$ vs. $6.19$ $[3.68, 9.48]$ no-prior-failure) and composed under coarse $C_2$ ($N = 261{,}476$: $1.25$ $[1.08, 1.75]$ vs. $1.47$ $[1.26, 1.69]$).
   - Freezing a parameter-free collared $3 \times 3$ local interface predictor (`frozen_interface_predictions.json`, SHA-256 `36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`) before running held-out two-cycle simulations prospectively identifies strictly better sector-selective simplifications: $C_2 R_{X3}$ ($99{,}580$ qubits, $-27.1\%$ qubits) increases $\beta = 6.5$ storage over $C_2 R_3$ by $1.98\times$ ($[1.31, 3.53]$), while coarse $R_{X3} C_2$ ($116{,}164$ qubits, $\rho \approx 40{,}965 < 36^3$) reaches $t_{0.80} = 2.63 \times 10^7$ at $\beta = 7.0$, exceeding $R_3 C_2$ by $5.01\times$ ($[1.66, 10.4]$).

4. **Cross-Scale Sector Rebalancing and Palindromic Welding (`src/verify_rebalanced_algebra.py`, `cluster_v9/build_v19_rebalance_and_decoders.py`):**
   - Three-step $(X, Z, X)$ hierarchies ($N = 1{,}942\text{--}10{,}580$) trade away unused $Z$-sector margin and flip the limiting bottleneck to $Z$: placing two bridges in the limiting middle $Z$ step ($X_2 R_{3Z} X_2$, $N = 4{,}566$) beats matched-size outer-$X$ placement ($R_{3X} Z_2 X_2$, $N = 4{,}556$) by $3.53\times$ ($[1.63, 6.41]$) at $\beta = 6.5$ ($t_{0.80} = 55{,}602$, matching $C_2 R_{X3}$ at $21.8\times$ fewer qubits), while $R_3 X_2$ ($10{,}580$ qubits) reaches $1.34 \times 10^5$, matching $C_2 C_2$ at $57.9\%$ fewer qubits.
   - Across the four-step $L = 2$ welding-order quartet (`XZXZ`, `ZXZX`, `ZXXZ`, `XZZX`) sharing the exact same $11{,}188$-face cell complex, ending on a single-qubit $X$-weld cuts $18.6\%$ of the qubits in palindromic $C_2 C_2^{ZX}$ (`XZZX`, $N = 20{,}476$) and raises the fine module's static barriers from $(2, 2)$ to $(3, 2)$, extending $\beta = 6.5$ storage by $101\times$ ($[55.9, 205]$; $t_{0.80} = 1.47 \times 10^7$) on planar seeds and $\ge 667\times$ ($\ge 1.00 \times 10^8$) on periodic tori.
   - Under continuable coarse $L = 3$ modules ($\rho < 36^3 = 46{,}656$), $C_3 C_2^{ZX}$ ($87{,}652$ qubits) reaches $3.32 \times 10^7$ at $\beta = 6.5$ ($313\times$ $C_3 C_2$; $\ge 1.58 \times 10^8$ at $\beta = 7.0$), and $R_{Z3} C_2^{ZX}$ ($100{,}900$ qubits) reverses the coarse bridge winner ($9.49\times$ $[1.52, 18.2]$ over $R_{X3} C_2^{ZX}$ at $\beta = 6.0$; $\ge 3.98 \times 10^7$ at $\beta = 6.5$).

5. **Physical Basin Crossing versus Single-Shot Readout Decoding (`cluster_v9/kmc_engine_v19_decoders.cpp`, `src/parse_v19_rebalance_and_decoders.py`):**
   - Paired evaluation of baseline Union-Find (`UF`), $T = 0$ strictly downhill cellular-automaton pre-cooling (`Cool-UF`), multiscale staged Union-Find (`Hier-UF`), and damped normalized min-sum belief propagation with ordered-statistics decoding (`BP-OSD`) on identical error states shows that `BP-OSD` increases unrelaxed single-shot bit-flip thresholds $p_{0.80}$ by $2.34\text{--}3.03\times$ over `UF` ($2.18\% \to 5.67\%$ on $C_2 C_2$; $2.33\% \to 7.06\%$ on $C_2 C_2^{ZX}$), yet agrees with `UF` within a few percent under cold Glauber storage (achieving $94.5\text{--}96.8\%$ of exact $8{,}192$-state maximum-likelihood $t_{0.80}^{\mathrm{opt}}$ on one-step $C_2$).
   - As $\beta$ increases from $5.5$ to $7.0$, `BP-OSD`-rescued failures vanish ($0\text{--}3\%$) while globally quenchable wrong-basin states rise to $67\text{--}86\%$ on $C_2 C_2^{ZX}$ and $79\%$ on $X_2 R_{3Z} X_2$: cold passive welded memories fail through activated thermal saddle crossing rather than single-shot Union-Find cluster percolation.

## Reproducing All Suites

```bash
# 1. Existing-data reanalysis, deduplication, single-cohort healing, and null baselines
python3 src/reanalyze_existing_v16_v17.py

# 2. Exact/bounded 1-step stabilizer quotients, rational prefactors, subsystem completion, and matched L=4 algebra
python3 src/verify_local_quotients_and_algebra.py

# 3. Rebalanced cell-complex, seam-multiplicity, spectral-radius, and exact C2^ZX barrier certificates
python3 src/verify_rebalanced_algebra.py

# 4. Build V18 (20 CSS code pairs) and V19 (29 CSS code pairs) and verify exact CSS commutation & ranks
python3 cluster_v9/build_v18_rework.py
python3 cluster_v9/build_v19_rebalance_and_decoders.py

# 5. Parse V18 (424 shards) and V19 (1,656 paired-decoder & i.i.d. shards) with 10,000-draw bootstrap CIs
python3 src/parse_v18_rework.py
python3 src/parse_v19_rebalance_and_decoders.py

# 6. Render Figures 1-6 and compile REVTeX 4.2 manuscripts
python3 src/render_v23_prx_figures.py
python3 src/render_v24_prx_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key Files

- `paper/main.tex`, `paper/main.pdf`, `paper/references.bib`: 14-page *Physical Review X*-formatted manuscript with Appendices A–D.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `frozen_interface_predictions.json`, `src/freeze_interface_predictor.py`, `src/eval_coarse_uf.cpp`: Pre-simulation frozen collared $3 \times 3$ interface predictor and scalar baselines (`SHA-256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`).
- `src/verify_rebalanced_algebra.py`, `rebalanced_algebra_certificates.json`: Quartet cell-complex invariance, seam-multiplicity counts, thin-bridge eigenvalues, and exact $(3, 2)$ barrier certificate on $C_2^{ZX}$.
- `cluster_v9/kmc_engine_v19_decoders.cpp`, `cluster_v9/build_v19_rebalance_and_decoders.py`, `v19_code_specs.json`: Paired four-decoder (`UF`, `Cool-UF`, `Hier-UF`, `BP-OSD`) C++20 engine and rebalanced hierarchy generator.
- `cluster_v19_results.txt`, `src/parse_v19_rebalance_and_decoders.py`, `parsed_v19_rebalance_and_decoders.json`: Raw paired-decoder and i.i.d. logs (`1,656` completed tasks, `0` errors) and $10{,}000$-draw bootstrap analysis.
- `src/render_v23_prx_figures.py`, `src/render_v24_prx_figures.py`, `figures/fig_v23_*.{pdf,png}`, `figures/fig_v24_*.{pdf,png}`: Publication figure generators and vector graphics.
