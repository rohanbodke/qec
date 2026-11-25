# Selective Simplification of Three-Dimensional Welded Quantum Memories

Companion 3D-local CSS code generators and Euclidean $\mathbb{R}^3$ coordinate verifiers, exact stabilizer-quotient and low-temperature resistor-network solvers, algebraic and barrier certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation and paired four-decoder engines, zero-parameter collared local-interface predictions, 3D cross-family benchmarks, raw fixed-time readout trajectories, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Starting from the three-dimensional recursive sheet-welding framework of Balasubramanian, Davydova, and Lin (BDL, arXiv:2605.10943), we enforce **three-dimensional Euclidean locality** throughout under unit stabilizer coupling ($J = 1$), independent single-qubit Pauli $X/Z$ Glauber heat-bath dynamics $r(\Delta E) = (1 + e^{\beta \Delta E})^{-1}$, and single-shot readout decoding:

$$\min_{\mathcal{A} \in \mathcal{F}_{\mathrm{3D}}} N(\mathcal{A}) \quad \text{subject to} \quad P_{\mathrm{rec}}^{\mathcal{A}}(t_*; \beta) \ge 0.80,$$

preserving exact CSS commutation $H_X H_Z^T = 0 \pmod 2$, the seed's rank-derived logical qubit count $K$, and 3D Euclidean packing dimension below $3$.

- **Attribution:** Alternating four-step $(X, Z, X, Z)$ ($XZXZ$) welding with two parallel interior bridges per cell is BDL's original template; **sector-selective bridge deletions ($R_{X3}, R_{Z3}$)**, **three-step $(X, Z, X)$ hierarchies**, and **symmetric four-step $(X, Z, Z, X)$ ($XZZX$) welding order** are introduced in this work and preserve BDL's exact 3D cell complex and $\mathbb{R}^3$ embedding whenever $L \ge 3$.

1. **3D Euclidean Packing Cutoff and Pure-$L = 3$ Baseline (`src/verify_3d_local_embeddings_and_L3.py`):**
   - An exact thin-bridge face recurrence proves that $L = 2$ violates 3D packing ($d_{\mathrm{pack}}(2, 1) = \log_2 \rho(T_{2,1}) \approx 3.30 > 3$, with 3D box density $N/L_{\mathrm{blk}}^3$ diverging $3.97 \to 3.79 \to 6.14$ across $s = 2, 3, 4$), establishing $L = 3$ ($d_{\mathrm{pack}}(3, 1) \approx 2.72$, $d_{\mathrm{pack}}(3, 2) \approx 2.80 < 3$, constructive $\mathbb{R}^3$ box density $0.80\text{--}1.34$, check diameter $R_{\max} \le 3.79$) as the minimal 3D-local scale.
   - Within BDL's alternating $(X, Z, X, Z)$ pure-$L = 3$ baseline $\{C_3, R_3\}^2$ at $\beta = 6.5$, fine-$C_3$ codes fail early ($1.85\text{--}3.19 \times 10^3$) via $2 \times 2$ flat-patch diffusion, whereas fine $R_3$ reaches $t_{0.80}^{\mathrm{BP}} = 1.11 \times 10^4$ ($R_3 R_3$, $709{,}408$ qubits) and coarse simplification $C_3 R_3$ ($591{,}208$ qubits, $-16.7\%$) raises storage $1.69\times$ ($[1.42, 1.98]$; $1.88 \times 10^4$).

2. **Exact Stabilizer Quotients, $L = 3$ Bridge-Arm Pruning, and Matched $L = 4$ Placement (`src/verify_local_quotients_and_algebra.py`, `src/verify_3d_local_embeddings_and_L3.py`):**
   - Quotienting physical $X$ errors by $\mathrm{rowspace}(H_X)$ ($\dim = N - \mathrm{rank}\,H_X$) yields the exact reversible identity $T_{\mathrm{clean}} = Z/(2\kappa) = Z/(2\Gamma\bar{h})$, rigorous energy-truncated capacity brackets on one-step $C_3$ vs. $R_3$, and exact rational low-temperature prefactors.
   - On two-step $L = 3$ $C_3$ ($N = 1{,}036$), cutting one bridge arm in the $Z$ step ($N = 971, 963$) spawns $11$ unintended logical modes ($K_{\mathrm{tot}} = 12$) and collapses $K_{\mathrm{target}} = 1$ subsystem storage $27.5\text{--}35.7\times$ via a certified barrier-$(1, 1)$ path, while 3D-local $L = 4$ matched placements ($N = 2{,}926$) isolate post-crossing returns ($6.19\times$ first-failure vs. $1.66\times$ operational ratio).

3. **Zero-Parameter Collared $3 \times 3$ Interface-Guided Selective Cuts (`src/eval_L3_collared_predictions.py`, `frozen_interface_predictions.json`):**
   - Probing $L = 3$ modules inside restoring $3 \times 3$ neighbor seams (`SHA-256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`) predicts with zero fitted parameters that removing the nonlimiting fine-$Z$ bridge in $C_3 R_{X3}$ ($431{,}188$ qubits, $-27.1\%$ vs. $C_3 R_3$) increases $\beta = 6.5$ storage ($2.89\times$ predicted vs. $2.25\times$ $[1.43, 2.65]$ observed; $23.5\times$ over $C_3 R_{Z3}$), while uniform $R_{X3} R_{X3}$ ($470{,}332$ qubits, $-33.7\%$ vs. $R_3 R_3$) boosts storage $4.71\times$ ($[3.57, 5.59]$) at $\beta = 6.5$ and $5.58\times$ ($[4.13, 12.3]$; $1.09 \times 10^6$) at $\beta = 7.0$.

4. **Novel 3D-Local Sequences: Three-Step $(X, Z, X)$ and Symmetric $(X, Z, Z, X)$ (`cluster_v9/build_v21_3d_local.py`, `src/parse_v21_3d_local.py`):**
   - Pure-$L = 3$ three-step $(X, Z, X)$ hierarchies ($N = 16{,}798\text{--}22{,}160, L_{\mathrm{blk}} = 27$) flip the bottleneck to $Z$: placing two bridges in the limiting middle $Z$ step ($X_{3,1} Z_{3,2} X_{3,1}$, $19{,}174$ qubits) beats outer-$X$ placement by $13.5\times$ ($[8.24, 22.0]$) at $\beta = 6.5$ ($2.84 \times 10^4$) and $18.0\times$ ($[10.2, 30.0]$; $1.29 \times 10^5$) at $\beta = 7.0$, outperforming $709{,}408$-qubit $R_3 R_3$ by $2.56\times$ at $37\times$ fewer qubits.
   - Switching four-step welding on the **identical $298{,}896$-face 3D complex** from BDL's $(X, Z, X, Z)$ to symmetric $(X, Z, Z, X)$ ($R_3 R_3^{ZX}$, $504{,}352$ qubits, $-28.9\%$) boosts $\beta = 6.5$ storage by $585\times$ ($[481, 706]$; $6.49 \times 10^6$) on planar seeds and $1{,}085\times$ ($[370, 1{,}439]$; $1.24 \times 10^7$) on periodic tori; simplified $C_3 R_3^{ZX}$ ($422{,}296$ qubits, $-40.5\%$) reaches $9.50 \times 10^5$ ($\beta = 6.5$) and $2.13 \times 10^7$ ($\beta = 7.0$), while $R_{Z3} R_3^{ZX}$ ($464{,}104$ qubits) reaches $7.39 \times 10^6$ and reverses the coarse bridge winner over $R_{X3} R_3^{ZX}$ by $9.45\text{--}10.5\times$.

5. **Paired Four-Decoder Calibration and $J = 1$ 3D Benchmarks (`cluster_v9/kmc_engine_v19_decoders.cpp`, `src/parse_v20_benchmarks.py`):**
   - Paired evaluation of `UF`, `Cool-UF`, `Hier-UF`, and `BP-OSD` on pure-$L = 3$ codes boosts unrelaxed i.i.d. thresholds $p_{0.80}$ by $2.15\text{--}2.77\times$ ($2.46\% \to 6.24\%$ on $C_3 R_3^{ZX}$) yet agrees with `UF` within statistical uncertainty under cold Glauber storage ($76\% \to 0\%$ BP-OSD rescued share from $\beta = 4.5 \to 6.5$).
   - Under identical $J = 1, \beta = 6.5$ dynamics across 2D and 3D-local families, 2D toric degrades ($1{,}220 \to 170$), single-scale 3D welded collapses $29.4\times$ ($6{,}145 \to 209$ across $L = 2 \to 8$), pure-$L = 3$ symmetric welded hierarchies jump $85.6\text{--}1{,}085\times$ over $R_3 R_3$, and $\mathbb{Z}^3$-local Haah's Cubic Code ($L = 7, N = 686, t_{0.80}^{\mathrm{BP}} \ge 10^9$; $L = 11, \ge 3.98 \times 10^8$) outperforms finite welded sheet codes at $190\text{--}735\times$ smaller $N$ ($\Delta E_1 = 4$ vs. $2$) while turning downward across $L = 5 \to 17$ at fixed $\beta \in \{3.25, 3.78, 4.00\}$.

## Reproducing All Suites

```bash
# 1. Constructive R^3 geometry, L=3 pruning algebra, exact quotients, and collared L=3 predictions
python3 src/verify_3d_local_embeddings_and_L3.py
python3 src/eval_L3_collared_predictions.py
python3 src/verify_local_quotients_and_algebra.py

# 2. Build V21 strictly 3D-local pure-L=3 campaign and V20 cross-family benchmark manifests
python3 cluster_v9/build_v21_3d_local.py
python3 cluster_v9/build_v20_benchmarks.py
python3 cluster_v9/build_v20b_j1.py

# 3. Parse V21 (648 shards, 0 errors) and V20 (630 shards, 0 errors) with 10,000-draw bootstrap CIs
python3 src/parse_v21_3d_local.py
python3 src/parse_v20_benchmarks.py

# 4. Render Figures 1-7 and compile REVTeX 4.2 manuscripts
python3 src/render_v26_3d_local_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key Files

- `paper/main.tex`, `paper/main.pdf`, `paper/references.bib`: *Physical Review X*-formatted strictly 3D-local manuscript with Appendices A–D.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `verified_3d_local_L3_geometry.json`, `src/verify_3d_local_embeddings_and_L3.py`, `src/eval_L3_collared_predictions.py`: Constructive $\mathbb{R}^3$ coordinate embeddings, $L = 3$ pruning/barrier certificates, and zero-parameter collared $3 \times 3$ pure-$L = 3$ predictions.
- `frozen_interface_predictions.json`, `src/freeze_interface_predictor.py`: Pre-registered collared $3 \times 3$ patch curves (`SHA-256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`).
- `cluster_v21_results.txt`, `v21_code_specs.json`, `parsed_v21_3d_local.json`, `cluster_v9/build_v21_3d_local.py`, `src/parse_v21_3d_local.py`: Raw trajectory logs (`648` tasks, `0` errors) and $10{,}000$-draw bootstrap parser for the pure-$L = 3$ 3D-local campaign.
- `cluster_v20_results.txt`, `parsed_v20_benchmarks.json`, `src/parse_v20_benchmarks.py`: Cross-family $J = 1$ benchmarks across 2D Toric, 1-cycle 3D Welded, and $\mathbb{Z}^3$-local Haah's Cubic Code.
- `src/render_v26_3d_local_figures.py`, `figures/fig_v26_*.{pdf,png}`: Publication figure generator and vector graphics.
