# Selective Simplification of Three-Dimensional Welded Quantum Memories

Companion 3D-local CSS code generators and Euclidean $\mathbb{R}^3$ coordinate verifiers, exact stabilizer-quotient and low-temperature resistor-network solvers, algebraic and barrier certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation and paired four-decoder engines, zero-parameter collared local-interface predictions, 3D cross-family benchmarks, 3D-local welded 4.8.8 flag color-code hierarchies, raw fixed-time readout trajectories, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Starting from the three-dimensional recursive sheet-welding framework of Balasubramanian, Davydova, and Lin (BDL, arXiv:2605.10943), we enforce **three-dimensional Euclidean locality** throughout under unit stabilizer coupling ($J = 1$), independent single-qubit Pauli $X/Z$ Glauber heat-bath dynamics $r(\Delta E) = (1 + e^{\beta \Delta E})^{-1}$, and single-shot readout decoding:

$$\min_{\mathcal{A} \in \mathcal{F}_{\mathrm{3D}}} N(\mathcal{A}) \quad \text{subject to} \quad P_{\mathrm{rec}}^{\mathcal{A}}(t_*; \beta) \ge 0.80,$$

preserving exact CSS commutation $H_X H_Z^T = 0 \pmod 2$, the seed's rank-derived logical qubit count $K$, and 3D Euclidean packing dimension below $3$.

- **Clear Attribution (Prior Art vs. Ours):**
  - *Prior-art baselines:* Kitaev's 2D toric code, Michnicki's single-scale 1-cycle 3D welded surface code, BDL's alternating $(X, Z, X, Z)$ ($XZXZ$) four-step two-bridge template ($R_3 R_3$), and Haah's $\mathbb{Z}^3$-local Cubic Code.
  - *Our four 3D-preserving contributions:* (i) **sector-selective single-bridge cuts ($R_{X3}, R_{Z3}$)**, (ii) **three-step $(X, Z, X)$ hierarchies**, (iii) **symmetric four-step $(X, Z, Z, X)$ ($XZZX$) welding order**, and (iv) **3D-local welded 4.8.8 flag color-code sheet hierarchies** ($\deg_{X,Z} \ge 3$, zero $\Delta E = 0$ sheet hops). Because BDL's $X$-weld and $Z$-weld share the exact same 3D complex and $\mathbb{R}^3$ map, bridge deletion only removes cylinder faces, and flag color-code sheets merely decorate each square face locally with eight corner-flag qubits ($N = 8F$), all four constructions strictly preserve 3D Euclidean locality at $L \ge 3$.

1. **3D Euclidean Packing Cutoff and Pure-$L = 3$ Baseline (`src/verify_3d_local_embeddings_and_L3.py`):**
   - An exact thin-bridge face recurrence proves that $L = 2$ violates 3D packing ($d_{\mathrm{pack}}(2, 1) = \log_2 \rho(T_{2,1}) \approx 3.30 > 3$, with 3D box density $N/L_{\mathrm{blk}}^3$ diverging $3.97 \to 3.79 \to 6.14$ across $s = 2, 3, 4$), establishing $L = 3$ ($d_{\mathrm{pack}}(3, 1) \approx 2.72$, $d_{\mathrm{pack}}(3, 2) \approx 2.80 < 3$, constructive $\mathbb{R}^3$ box density $0.80\text{--}1.34$, check diameter $R_{\max} \le 3.79$) as the minimal 3D-local scale.
   - Within BDL's alternating $(X, Z, X, Z)$ pure-$L = 3$ baseline $\{C_3, R_3\}^2$ at $\beta = 6.5$, fine-$C_3$ codes fail early ($1.85\text{--}3.19 \times 10^3$) via $2 \times 2$ flat-patch diffusion, whereas fine $R_3$ reaches $t_{0.80}^{\mathrm{BP}} = 1.11 \times 10^4$ ($R_3 R_3$, $709{,}408$ qubits) and coarse simplification $C_3 R_3$ ($591{,}208$ qubits, $-16.7\%$) raises storage $1.69\times$ ($[1.42, 1.98]$; $1.88 \times 10^4$).

2. **Exact Stabilizer Quotients, $L = 3$ Bridge-Arm Pruning, and Matched $L = 4$ Placement (`src/verify_local_quotients_and_algebra.py`, `src/verify_3d_local_embeddings_and_L3.py`):**
   - Quotienting physical $X$ errors by $\mathrm{rowspace}(H_X)$ ($\dim = N - \mathrm{rank}\,H_X$) yields the exact reversible identity $T_{\mathrm{clean}} = Z/(2\kappa) = Z/(2\Gamma\bar{h})$, rigorous energy-truncated capacity brackets on one-step $C_3$ vs. $R_3$, and exact rational low-temperature prefactors.
   - On two-step $L = 3$ $C_3$ ($N = 1{,}036$), cutting one bridge arm in the $Z$ step ($N = 971, 963$) spawns $11$ unintended logical modes ($K_{\mathrm{tot}} = 12$) and collapses $K_{\mathrm{target}} = 1$ subsystem storage $27.5\text{--}35.7\times$ via a certified barrier-$(1, 1)$ path, while 3D-local $L = 4$ matched placements ($N = 2{,}926$) isolate post-crossing returns ($6.19\times$ first-failure vs. $1.66\times$ operational ratio).

3. **Zero-Parameter Collared $3 \times 3$ Interface-Guided Selective Cuts (`src/eval_L3_collared_predictions.py`, `frozen_interface_predictions.json`):**
   - Probing $L = 3$ modules inside restoring $3 \times 3$ neighbor seams (`frozen_sha256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`) predicts with zero fitted parameters that removing the nonlimiting fine-$Z$ bridge in $C_3 R_{X3}$ ($431{,}188$ qubits, $-27.1\%$ vs. $C_3 R_3$) increases $\beta = 6.5$ storage ($2.89\times$ predicted vs. $2.25\times$ $[1.43, 2.65]$ observed; $23.5\times$ over $C_3 R_{Z3}$), while uniform $R_{X3} R_{X3}$ ($470{,}332$ qubits, $-33.7\%$ vs. $R_3 R_3$) boosts storage $4.71\times$ ($[3.57, 5.59]$) at $\beta = 6.5$ and $5.58\times$ ($[4.13, 12.3]$; $1.09 \times 10^6$) at $\beta = 7.0$.

4. **Novel 3D-Local Surface Sequences: Three-Step $(X, Z, X)$ and Symmetric $(X, Z, Z, X)$ (`cluster_v9/build_v21_3d_local.py`, `src/parse_v21_3d_local.py`):**
   - Pure-$L = 3$ three-step $(X, Z, X)$ hierarchies ($N = 16{,}798\text{--}22{,}160, L_{\mathrm{blk}} = 27$) flip the bottleneck to $Z$: placing two bridges in the limiting middle $Z$ step ($X_{3,1} Z_{3,2} X_{3,1}$, $19{,}174$ qubits) beats outer-$X$ placement by $13.5\times$ ($[8.24, 22.0]$) at $\beta = 6.5$ ($2.84 \times 10^4$) and $18.0\times$ ($[10.2, 30.0]$; $1.29 \times 10^5$) at $\beta = 7.0$, outperforming $709{,}408$-qubit $R_3 R_3$ by $2.56\times$ at $37\times$ fewer qubits.
   - Switching four-step welding on the **identical $298{,}896$-face 3D complex** from BDL's $(X, Z, X, Z)$ to symmetric $(X, Z, Z, X)$ ($R_3 R_3^{ZX}$, $504{,}352$ qubits, $-28.9\%$) boosts $\beta = 6.5$ storage by $585\times$ ($[481, 706]$; $6.49 \times 10^6$) on planar seeds and $1{,}085\times$ ($[370, 1{,}439]$; $1.24 \times 10^7$) on periodic tori; simplified $C_3 R_3^{ZX}$ ($422{,}296$ qubits, $-40.5\%$) reaches $9.50 \times 10^5$ ($\beta = 6.5$) and $2.13 \times 10^7$ ($\beta = 7.0$), while $R_{Z3} R_3^{ZX}$ ($464{,}104$ qubits) reaches $7.39 \times 10^6$ and reverses the coarse bridge winner over $R_{X3} R_3^{ZX}$ by $9.45\text{--}10.5\times$.

5. **Paired Four-Decoder Calibration and $J = 1$ Cross-Family Benchmarks (`cluster_v9/kmc_engine_v19_decoders.cpp`, `src/parse_v20_benchmarks.py`):**
   - Paired evaluation of `UF`, `Cool-UF`, `Hier-UF`, and `BP-OSD` on pure-$L = 3$ surface codes boosts unrelaxed i.i.d. thresholds $p_{0.80}$ by $2.15\text{--}2.77\times$ ($2.46\% \to 6.24\%$ on $C_3 R_3^{ZX}$) yet agrees with `UF` within statistical uncertainty under cold Glauber storage ($76\% \to 0\%$ BP-OSD rescued share from $\beta = 4.5 \to 6.5$).
   - Under identical $J = 1, \beta = 6.5$ dynamics across prior-art baselines, 2D toric degrades ($1{,}220 \to 170$), single-scale 3D welded collapses $29.4\times$ ($6{,}145 \to 209$ across $L = 2 \to 8$), and $\mathbb{Z}^3$-local Haah's Cubic Code ($L = 7, N = 686, t_{0.80}^{\mathrm{BP}} \ge 10^9$; $L = 11, \ge 3.98 \times 10^8$) outperforms all surface-sheet codes via $\Delta E_1 = 4$ cube checks (while turning downward across $L = 5 \to 17$ at fixed $\beta \le 4.00$).

6. **3D-Local Welded 4.8.8 Flag Color-Code Sheet Hierarchies (`src/verify_3d_color_welded_algebra.py`, `cluster_v9/build_v22_color_welded.py`, `src/parse_v22_color_welded.py`, `564` tasks, `0` errors, `0` pre-$t_{0.80}$ caps):**
   - Placing eight corner-flag qubits $(v, e, f)$ per face on BDL's 3D welded complex ($N = 8F, K = 4$ on periodic 1-face torus seeds) raises minimum check degree from $\Delta E_1 = 2$ to $\deg_{X,Z} \ge 3$ everywhere, eliminates $\Delta E = 0$ flat-sheet hops, and slashes $\beta = 6.5$ clean birth rates $> 6 \times 10^5$-fold relative to $422\mathrm{k}$-qubit $C_3 R_3^{ZX}$.
   - Asymmetric $X/Z$ seam welding lifts 1-step barriers from $(3, 3)$ to $(3, 4)$ and boosts 2-step storage **$61.6\times$** ($[40.6, 85.0]$) at identical $N = 5{,}088$ (`cc2s_R3`: $6.16 \times 10^6$ vs. self-dual `cc2s_R3_sd`: $1.00 \times 10^5$).
   - Unwelded 2D 4.8.8 color tori still degrade **$7.89\times$** (`[4.06, 12.1]`) across $L = 3 \to 12$ ($6.02 \times 10^5 \to 7.63 \times 10^4$). Pure-$L = 3$ 2-step welded `cc2s_RX3` ($4{,}128$ qubits, $\dim_F = 2.843 < 3, R_{\max} = 3.01$) reaches **$8.91 \times 10^6$** ($[5.72, 13.5]\times 10^6$) at $\beta = 6.5$ (**$117\times$** 2D $L = 12$, **$29.5\times$** opposite-cut `cc2s_RZ3`, **$803\times$** $[509, 1{,}244]$ BDL `R_3 R_3` at $172\times$ fewer qubits, and **$18.8\times$** $C_3 R_3^{ZX}$ at warmer $\beta = 5.0$). Mixed $(3, 2)$ `cc2s_mix32` ($2{,}208$ qubits) reaches $3.69 \times 10^7$ ($\beta = 6.5$), while 3-step `cc3s_X3r1Z3r2X3r1` ($81{,}888$ qubits, $\dim_F = 2.802 < 3$) reaches $1.41 \times 10^6$ ($7.61\times$ `1-1-1`, $127\times$ `R_3 R_3`). Paired `BP-OSD` lifts unrelaxed color-code $p_{0.80}$ by $1.63\text{--}3.17\times$ ($1.97\% \to 4.60\%$ on `cc2s_R3`) while agreeing with `UF` under cold Glauber dynamics (`BP/UF` $= 0.95\text{--}1.00$ at $\beta = 6.5$).

7. **Extension A & Regimes 1/3: Scaling Anatomy, T-Junction Dispersion (`jog1`), and Open-Boundary (`css_dual`) Color Codes (`src/verify_3d_extensionA_and_regimes.py`, `cluster_v9/build_v23_extensionA.py`, `src/parse_v23_extensionA.py`, `270` tasks, `0` errors, `0` pre-$t_{0.80}$ caps):**
   - **Same-parity scaling & $E = 2$ bulk dimer entropy (Regime 1):** Holding limiting-sector parity fixed (`1s L_{3,2}` $240 \to$ `3s 1-2-1` $81{,}888$), $\beta = 6.5$ storage grows **$2.49\times$** (`[1.49, 4.04]`). Exact adjacent-pair enumeration isolates why 3-step color codes lag 2-step `cc2s_RX3`: degree-$3$ 4.8.8 sheets carry **$118{,}600$** limiting-sector $E = 2$ dimer traps on 3s `1-2-1` ($0.268$ occupied dimers at $\beta = 6.5$) vs. **$5{,}832$** on 2s `cc2s_RX3` ($0.013$ dimers) and zero $E \le 2$ pairs on degree-$4$ Haah. Splitting $d = 3$ seams symmetrically half-$X$/half-$Z$ breaks topological order ($K \to 8, 12$; $\min \deg \to 2$).
   - **BDL `jog1` T-junction dispersion (Extension A1):** Shifting transverse bridges by one unit removes all $d = 4$ seam pillars (`d4_edge_count = 0`, $w_{\max} \le (14, 18)$, $R_{\max} \le 4.53$) across single-bridge `v23_disp_1s/2s/3s_L3`. Reinforced 2s `v23_disp_2s_RX3` ($4{,}272$ qubits, $8$ residual $d = 4$ seams) reaches **$1.11 \times 10^7$** (`[5.47, 15.4]\times 10^6$) at $\beta = 6.5$ (`1.24x [0.58, 2.13]` vs. `cc2s_RX3`), while single-bridge `v23_disp_3s_L3` ($80{,}648$ qubits, $127{,}876$ $E = 2$ traps) reaches $1.11 \times 10^5$ ($\beta = 6.5$) and $6.85 \times 10^5$ ($\beta = 7.0$), confirming $r = 2$ limiting-step bridges and $E = 2$ dimer entropy govern 3-step lifetimes. Paired `BP-OSD` lifts i.i.d. $p_{0.80}$ on `v23_disp_2s_L3` from $2.09\%$ to $5.16\%$ ($2.47\times$).
   - **Open-boundary `css_dual` planar color codes (Regime 3 / Extension A2):** Alternating smooth/rough boundary checks lock logical degeneracy at invariant **$K = 2$** across every step without periodic wraparound—in sharp contrast to periodic Haah $\mathbb{T}^3$, which oscillates **$K(L) \in [2, 62]$** across $L = 3\text{--}17$. Open planar 2s `v23_plan_cc2s_RX3` ($4{,}128$ qubits, $K = 2$) reaches **$4.15 \times 10^6$** (`[2.51, 6.71]\times 10^6$) at $\beta = 6.5$, outperforming open 2D $L = 12$ ($1.08 \times 10^4$) by **$383\times$** (`[210, 636]`) with `BP-OSD` i.i.d. threshold $5.51\%$ ($2.89\times$ `UF`).

## Reproducing All Suites

```bash
# 1. Constructive R^3 geometry, pruning algebra, quotients, and color-welded algebraic certificates
python3 src/verify_3d_local_embeddings_and_L3.py
python3 src/verify_3d_color_welded_algebra.py
python3 src/verify_3d_extensionA_and_regimes.py
python3 src/eval_L3_collared_predictions.py
python3 src/verify_local_quotients_and_algebra.py

# 2. Build V21 (surface), V20 (benchmarks), V22 (4.8.8 color-welded), and V23 (Extension A) SubMIT campaigns
python3 cluster_v9/build_v21_3d_local.py
python3 cluster_v9/build_v20_benchmarks.py
python3 cluster_v9/build_v20b_j1.py
python3 cluster_v9/build_v22_color_welded.py
python3 cluster_v9/build_v23_extensionA.py

# 3. Parse V21 (648 tasks), V20 (630 tasks), V22 (564 tasks), and V23 (270 tasks) with 10,000-draw bootstrap CIs
python3 src/parse_v21_3d_local.py
python3 src/parse_v20_benchmarks.py
python3 src/parse_v22_color_welded.py
python3 src/parse_v23_extensionA.py

# 4. Render Figures 1-9 and compile REVTeX 4.2 manuscripts
python3 src/render_v26_3d_local_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```

## Key Files

- `paper/main.tex`, `paper/main.pdf`, `paper/references.bib`: *Physical Review X*-formatted strictly 3D-local manuscript (`10` pages, Sections I–VI, Figures 1–4, and Appendices A–C).
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `verified_3d_local_L3_geometry.json`, `src/verify_3d_local_embeddings_and_L3.py`, `src/eval_L3_collared_predictions.py`: Constructive $\mathbb{R}^3$ coordinate embeddings, $L = 3$ pruning/barrier certificates, and zero-parameter collared $3 \times 3$ pure-$L = 3$ predictions.
- `verified_3d_color_welded_geometry.json`, `src/verify_3d_color_welded_algebra.py`: Flag 4.8.8 color-welded geometry, constructive $\mathbb{R}^3$ diameter certificates, and 1-step BFS barrier certificates.
- `frozen_interface_predictions.json`, `src/freeze_interface_predictor.py`: Pre-registered collared $3 \times 3$ patch curves (`frozen_sha256: 36048c6919f8633d9e7487a5b886c96deca5ce18f32b1ac9cc0f828be23a0ff4`).
- `cluster_v21_results.txt`, `v21_code_specs.json`, `parsed_v21_3d_local.json`, `cluster_v9/build_v21_3d_local.py`, `src/parse_v21_3d_local.py`: Raw trajectory logs (`648` tasks, `0` errors) and bootstrap parser for the pure-$L = 3$ surface campaign.
- `cluster_v22_results.txt`, `v22_code_specs.json`, `parsed_v22_color_welded.json`, `cluster_v9/build_v22_color_welded.py`, `src/parse_v22_color_welded.py`, `cluster_v9/slurm_worker_v22.sh`: Raw trajectory logs (`564` tasks, `0` errors) and bootstrap parser for the V22 3D-local welded 4.8.8 color-code campaign.
- `verified_v23_extensionA.json`, `v23_code_specs.json`, `cluster_v23_results.txt`, `parsed_v23_extensionA.json`, `cluster_v9/build_v23_extensionA.py`, `cluster_v9/slurm_worker_v23.sh`, `src/verify_3d_extensionA_and_regimes.py`, `src/parse_v23_extensionA.py`: V23 Extension A (`270` tasks, `0` errors) geometries, exact dimer-trap / Haah $K(L)$ certificates, and bootstrap parser for T-junction dispersion and open-boundary planar hierarchies.
- `cluster_v20_results.txt`, `parsed_v20_benchmarks.json`, `src/parse_v20_benchmarks.py`: Cross-family $J = 1$ benchmarks across 2D Toric, Michnicki 1-cycle 3D Welded, and $\mathbb{Z}^3$-local Haah's Cubic Code.
- `src/render_v26_3d_local_figures.py`, `figures/fig_v26_*.{pdf,png}`, `figures/fig_v29_*.{pdf,png}`: Publication figure generator and vector graphics for Figures 1–4 of the main paper plus supporting plots.
