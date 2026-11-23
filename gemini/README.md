# Selective Simplification of Finite Welded Quantum Memories

Companion CSS code generators, exact stabilizer-quotient and low-temperature resistor-network solvers, algebraic and barrier certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation engine, frozen local-interface predictions, raw fixed-time readout trajectories, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Starting from the proof-motivated recursive sheet-welding framework of Balasubramanian, Davydova, and Lin (BDL Section 6.2), we formulate finite simplification as an explicit operational optimization problem under independent single-qubit Pauli $X/Z$ Glauber heat-bath dynamics $r(\Delta E) = (1 + e^{\beta \Delta E})^{-1}$ and a fixed global Union-Find cluster decoder:

$$\min_{\mathcal{A} \in \mathcal{F}} N(\mathcal{A}) \quad \text{subject to} \quad P_{\mathrm{rec}}^{\mathcal{A}}(t_*; \beta) \ge 0.80,$$

preserving exact CSS commutation $H_X H_Z^T = 0 \pmod 2$ and the seed's rank-derived logical qubit count $K$.

1. **Compact Baseline Anchor, Scale Asymmetry, and Null Baselines (`src/reanalyze_existing_v16_v17.py`):**
   - At inverse temperature $\beta = 6.5$, all-compact one-bridge $C_2 C_2$ ($25{,}148$ qubits) removes $96.46\%$ of the physical qubits of uniform two-bridge $R_3 R_3$ ($709{,}408$ qubits) while extending $t_{0.80}$ by $13.04\times$ ($95\%$ bootstrap CI $[8.17, 18.86]$) on planar seeds and $13.45\times$ ($[8.61, 17.01]$) on periodic tori, exceeding all six fine-$C_3$/fine-$R_3$ two-cycle codes by at least $5.99\times$ ($[3.85, 9.48]$) planar and $9.85\times$ ($[5.93, 12.64]$) torus across $p \in \{0.70, 0.75, 0.80, 0.85, 0.90\}$.
   - Fine-module choice drives $36\text{--}109\times$ variation in $t_{0.80}$ versus $\le 2.4\times$ across coarse modules, inverted relative to standalone $1 \times 1$ lifetimes ($R_3 > C_2$ by $8.6\times$) and retaining $8.6\text{--}29\times$ beyond $1/N$ and $16.80\times$ ($[11.61, 31.07]$) beyond exact clean-state $X$ departure rates $\Gamma_X^{(0)}$ on $C_3 C_2$ versus $C_3 C_3$. At $\beta = 7.0$, $R_3 C_2$ ($133{,}864$ qubits) buys $3.40\times$ ($[1.19, 7.21]$) longer planar $t_{0.80}$ than $C_2 C_2$ with resolved torus temperature growth $G_{C_2}(7.0)/G_{C_2}(6.5) = 5.88$ ($[1.58, 11.73]$) while passing the necessary thin-bridge periodic-continuation screen $\rho(T_{\mathrm{blk}}) \le L_{\mathrm{blk}}^3$.

2. **Exact Stabilizer Quotients, Bridge-Arm Pruning, and Protected-Subsystem Test (`src/verify_local_quotients_and_algebra.py`):**
   - Quotienting physical $X$ errors by $\mathrm{rowspace}(H_X)$ ($\dim = N - \mathrm{rank}\,H_X$) yields exact reversible identity $T_{\mathrm{clean}} = Z/(2\kappa) = Z/(2\Gamma\bar{h})$, rigorous energy-truncated capacity brackets on one-step $C_3$ ($[261{,}455, 261{,}464]$) and $R_3$ ($[94.074, 94.076]\times 10^6$), and exact rational low-temperature prefactors ($\frac{10}{3}e^{-2\beta}$, $\frac{423}{119}e^{-2\beta}$, $\frac{258{,}892}{56{,}235}e^{-2\beta}$, $14e^{-3\beta}$, and joint mobility crossover $\frac{3(\lambda+1)}{\lambda+3}e^{-2\beta}$ at $\lambda = \mu e^\beta$).
   - On one-step $C_2$ at $\beta = 7.0$, three-arm bridge deletion ($N = 19 \to 18, K = 1$) retains $93.85\%$ of $t_{0.80}^{\mathrm{opt}, X}$ while $Z$ limits $t_{0.80}^{\mathrm{opt}, Q} = 46.67$. In the complete two-step $C_2$ module at $\beta = 6.5$ ($n_X = 2{,}048, n_Z = 1{,}024$), $X$-step pruning ($N = 248, K = 1$) gives $t_{0.80}$ ratio $0.894$ ($[0.776, 1.036]$), whereas $Z$-step and doubly pruned variants ($N = 228, 222$) introduce eight extra logical modes ($K_{\mathrm{total}} = 9$) and collapse $62\text{--}68\times$ via an exact barrier-$(1, 1)$ $Z$ sequence $0 \to 1 \to 1 \to 1 \to 0$ even as a $K_{\mathrm{target}} = 1$ subsystem.

3. **Post-Crossing Return, Matched $L=4$ Placement, and Prospective Interface-Guided Simplification (`src/freeze_interface_predictor.py`, `src/parse_v18_rework.py`):**
   - Single-cohort V17 trajectories show substantial post-crossing returns ($P_{\mathrm{healed}}(t_{0.80}) = 0.391$ on planar $C_3 C_2$ at $\beta = 6.5$), and resource-matched $L = 4$ offset ($1+3$) versus centered ($2+2$) placements confirm return-driven compression both standalone ($N = 2{,}926$: paired $t_{0.80}$ ratio $1.66$ $[0.89, 3.12]$ vs. $6.19$ $[3.68, 9.48]$ no-prior-failure) and composed under coarse $C_2$ ($N = 261{,}476$: $1.25$ $[1.08, 1.75]$ vs. $1.47$ $[1.26, 1.69]$).
   - Freezing a parameter-free collared $3 \times 3$ local interface predictor (`frozen_interface_predictions.json`, SHA-256 `36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`) before running held-out two-cycle simulations prospectively identifies strictly better sector-selective simplifications: deleting the second parallel bridge from $R_3$'s nonlimiting $Z$ step ($C_2 R_{X3}$, $99{,}580$ qubits, $-27.1\%$ qubits) increases $\beta = 6.5$ storage over $C_2 R_3$ by $1.98\times$ ($[1.31, 3.53]$; $52{,}827$ vs. $26{,}710$), whereas cutting the limiting $X$ bridge ($C_2 R_{Z3}$, $131{,}820$ qubits) drops storage to $2{,}208$ ($0.083\times$). At $\beta = 7.0$, coarse $R_{X3} C_2$ ($116{,}164$ qubits) reaches $t_{0.80} = 2.63 \times 10^7$ ($[1.16, 4.27] \times 10^7$), exceeding $R_3 C_2$ by $5.01\times$ ($[1.66, 10.4]$) and $C_3 C_2$ by $16.2\times$ ($[4.35, 28.2]$) while revealing three-dimensional coarse $X$-bridge cleaning beyond a flat 2D collar.

## Reproducing All Suites

```bash
# 1. Existing-data reanalysis, deduplication, single-cohort healing, and null baselines (Suite A)
python3 src/reanalyze_existing_v16_v17.py

# 2. Exact/bounded 1-step stabilizer quotients, rational prefactors, subsystem completion, and matched L=4 algebra (Suites B, C, D-alg, E-alg, F-alg)
python3 src/verify_local_quotients_and_algebra.py

# 3. Build all 20 V18 CSS code pairs (40 X/Z .code files) and verify exact CSS commutation & ranks
python3 cluster_v9/build_v18_rework.py

# 4. Verify frozen pre-simulation SHA-256 hash and parse all 424 V18 whole-memory/pruning/matched-geometry shards
python3 src/parse_v18_rework.py

# 5. Render Figures 1-4 and compile REVTeX 4.2 manuscripts
python3 src/render_v23_prx_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key Files

- `paper/main.tex`, `paper/main.pdf`, `paper/references.bib`: 11-page *Physical Review X*-formatted manuscript with Appendices A–D.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `frozen_interface_predictions.json`, `src/freeze_interface_predictor.py`, `src/eval_coarse_uf.cpp`: Pre-simulation frozen collared $3 \times 3$ interface predictor and scalar baselines (`SHA-256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`).
- `src/reanalyze_existing_v16_v17.py`, `reanalyzed_v16_v17_suiteA.json`: Deduplicated V15/V16/V17 reanalysis, threshold sensitivity $p \in \{0.70,\dots,0.90\}$, single-cohort healing, and size/birth-rate null checks.
- `src/verify_local_quotients_and_algebra.py`, `local_quotient_and_pruning_data.json`: Exact 1-step $C_2$ eigensolves, truncated-capacity bounds on $C_3$ and $R_3$, exact rational resistor networks, non-symmetric counterexample, Poulin subsystem completion, and matched $L=4$ certificates.
- `cluster_v9/build_v7_rescue.py`, `cluster_v9/build_v18_rework.py`, `cluster_v9/kmc_engine_v17.cpp`, `v18_code_specs.json`: Deterministic CSS hierarchy builders and multithreaded C++20 BKL + Union-Find thermal engine.
- `cluster_v18_results.txt`, `src/parse_v18_rework.py`, `parsed_v18_rework_data.json`: Raw trajectory logs (`424` completed shards, `0` errors) and $10{,}000$-draw paired/independent bootstrap parser.
- `src/render_v23_prx_figures.py`, `figures/fig_v23_*.{pdf,png}`: Publication figure generator and vector graphics.
