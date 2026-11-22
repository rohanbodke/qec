# The Thermal Value of Welding Across Scales

Companion generator code, standalone algebraic and continuation-growth certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation engine, raw continued fixed-time readout trajectories, kinetic mask specifications, whole-trajectory bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Building on the welded cell layout of Balasubramanian, Davydova, and Lin (BDL Section 6.2), we study how much interior bridge structure can be removed from finite two-cycle welded memories, how the remaining cells should be ordered across scales, and when fine-scale error dynamics make extra coarse welding worthwhile across `34` bounded-degree CSS codes up to $709{,}408$ physical qubits:

1. **All-One-Bridge Simplification, Nine-Code Operational Matrix, and Constituent Misranking:**
   - At planar inverse temperature $\beta = 6.5$, an ordered hierarchy using one bridge cross in both cycles ($C_3 C_2$, $107{,}140$ qubits) increases the passive $80\%$-recovery readout time $t_{0.80}$ by $9.1\times$ ($95\%$ CI $[6.0, 14.5]$) and mean first-failure time $\tau_Q$ by $7.6\times$ over the uniform two-bridge reference ($R_3 R_3$, $709{,}408$ qubits) using $85\%$ fewer qubits ($5.5\times$ in torus $t_{0.80}$). Reversing those two one-bridge modules ($C_2 C_3$, $95{,}896$ qubits) reduces $t_{0.80}$ by $37\times$ planar ($21\times$ torus).
   - Across all nine two-cycle combinations $\{C_2, C_3, R_3\} \times \{C_2, C_3, R_3\}$, the fine inner cell dominates operational storage ($36\text{--}109\times$ variation across fine cells versus $\le 2.4\times$ across coarse cells), and standalone module lifetime inverts fine-constituent rankings: isolated $R_3$ outlives isolated $C_2$ by $8.6\times$ in planar $t_{0.80}$ ($6.3\times$ torus), yet fine $C_2$ outperforms fine $R_3$ by $4.1\text{--}16\times$ in every coarse context with $5.3\text{--}5.5\times$ fewer qubits.

2. **Cold Marginal Coarse-Bridge Benefit, Healing Decomposition, and Continuation Screen:**
   - Measuring $G_B(\beta) = t_{0.80}(R_3 B; \beta)/t_{0.80}(C_3 B; \beta)$ directly separates scale interaction from separable cell effects $\log t_{0.80}(A, B) = c + u_A + v_B$. At $\beta = 6.5$, stronger post-crossing healing in $C_3 C_2$ ($P_{\mathrm{healed}} = 0.39$ vs. $0.17$) offsets $R_3 C_2$'s $2.5\times$ longer first-crossing survival, leaving $G_{C_2}(6.5)$ unresolved. At $\beta = 7.0$, $G_{C_2}(7.0)$ rises to $3.24\times$ ($[1.24, 6.49]$) planar and $5.90\times$ ($[2.34, 9.77]$) torus---overtaking all-compact $C_2 C_2$ in planar $t_{0.80}$ ($3.40\times$, $[1.17, 7.28]$) and torus $\tau_Q$ ($1.87\times$, $[1.15, 3.12]$)---while $G_{C_3}(7.0)$ remains below one ($0.79\text{--}0.81$), giving non-separable interaction ratios $G_{C_2}/G_{C_3} = 4.1\text{--}7.3\times$.
   - Proposition 1 proves the exact thin-bridge face recurrence $\binom{S_{q+1}}{B_{q+1}} = \bigl(\begin{smallmatrix}2L^2 & 2L\\ 2rL & r\end{smallmatrix}\bigr)\binom{S_q}{B_q}$ with Tanner diameter $D_q = O(\prod_t L_t)$, showing that uniform $C_2$ repetition exceeds $\rho(T_{\mathrm{blk}}) \le L_{\mathrm{blk}}^3$ ($\rho = 9{,}281 > 16^3$) whereas $C_3 C_2$ ($\rho = 37{,}636 < 36^3$) and $R_3 C_2$ ($\rho = 45{,}930 < 36^3$) satisfy this necessary periodic-continuation screen.

3. **Controlled Zero-Energy Kinetic Test on $R_3 C_3$ versus $C_3 C_3$:**
   - Multiplying equal-energy ($\Delta E = 0$) flip rates by $\mu \in \{1, 0.3, 0.1, 0.03, 0.01\}$ on fine-$C_3$ flat-sheet qubits while keeping each code's Hamiltonian and Gibbs equilibrium unchanged increases $t_{0.80}$ at $\mu = 0.01$ by $7.0\text{--}8.1\times$ in both $R_3 C_3$ and $C_3 C_3$ ($3.0\text{--}4.8\times$ above equal-size fine-$C_3$ degree-$2$ seam controls), while $G_{C_3}(\mu) = t_{0.80}(R_3 C_3; \mu)/t_{0.80}(C_3 C_3; \mu)$ remains below one ($0.59\text{--}0.85$): suppressing flat-patch mobility alone does not make extra coarse welding worthwhile on larger $3 \times 3$ fine cells.

## Key Files

- `paper/main.tex`, `paper/main.pdf`: 9-page *Physical Review X*-formatted manuscript.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `src/reproducibility_certificates.py`, `certificates_v21.json`: Standalone verification of Proposition 1, bounded edge/vertex incidences, periodic-continuation volume screens, SHA-256 check-matrix hashes for all `34` codes, and Markov-generator detailed balance under zero-energy kinetic scaling.
- `cluster_v9/build_v16_consolidation.py`, `cluster_v9/build_v17_causal_matrix.py`, `cluster_v9/kmc_engine_v17.cpp`: Code and kinetic-mask builders (`v15_code_specs.json`, `v16_code_specs.json`, `v17_mask_specs.json`) and multithreaded C++20 BKL simulation engine (`kmc_off`, `span_off`, `final_readout`, `kinetic_readout`).
- `cluster_v15_results.txt`, `cluster_v16_results.txt`, `cluster_v17_results.txt`, `src/parse_v16_results.py`, `src/parse_v17_results.py`, `parsed_v16_data.json`, `parsed_v17_data.json`: Raw trajectory logs (including per-trajectory continued fixed-time ternary survival/healing bitstrings), whole-trajectory independent-sector bootstrap parsers, and compiled tables.
- `src/render_v22_prx_figures.py`, `figures/fig_v22_*.pdf`: Publication figure scripts and vector PDFs.
