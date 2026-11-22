# The Thermal Value of Welding Across Scales

Companion code, standalone verification scripts, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation engine, raw trajectory logs, bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Building on the welded cell layout of Balasubramanian, Davydova, and Lin (BDL Section 6.2), we study whether additional interior bridges are more useful at fine or coarse scales, and when their extra qubits are justified across `34` bounded-degree CSS codes up to $709{,}408$ physical qubits:

1. **Placement Reversal Across First-Failure and Fixed-Time Readout:**
   - Pairing the two-bridge $3 \times 3$ module $R_3$ ($1{,}480$ qubits) with the one-bridge $3 \times 3$ module $C_3$ ($1{,}036$ qubits, whose single cross leaves $1 \times 1$, $1 \times 2$, $2 \times 1$, and $2 \times 2$ sheet regions) favors fine placement ($C_3 R_3$, $591{,}208$ qubits) over coarse placement ($R_3 C_3$, $499{,}088$ qubits) at $\beta = 6.5$ by $10.3\times$ in both mean first-failure time $\tau_Q$ and the $80\%$-recovery storage time $t_{0.80}$ under passive fixed-time readout $P_{\mathrm{id}}(t) = p_X(t)\,p_Z(t)$.
   - Replacing $C_3$ with the $2 \times 2$ one-bridge module $C_2$ ($254$ qubits, four unit faces) reverses the preferred placement across planar, one-face torus, two-face rotated torus $[[4, 2, 2]]$, and isotropic codes: coarse placement ($R_3 C_2$, $133{,}864$ qubits) exceeds fine placement ($C_2 R_3$, $136{,}536$ qubits) by $7.80\times$ in $\tau_Q$ ($4.50\times$ in $t_{0.80}$) at planar $\beta = 6.5$, $19.6\times$ at planar $\beta = 6.75$, and $31.5\times$ on uncensored periodic tori at $\beta = 7.0$.
   - Pairing the two one-bridge modules ($C_3$ and $C_2$) likewise favors placing $C_2$ at the fine scale ($C_3 C_2$, $107{,}140$ qubits) over $C_2 C_3$ ($95{,}896$ qubits) by $30.6\text{--}141\times$.

2. **Thin-Bridge Face Recurrence, Volume Screen, and Second-Bridge Comparison:**
   - Proposition 1 proves directly from the cell-complex generator that thin-bridge face counts obey $\binom{S_{q+1}}{B_{q+1}} = \bigl(\begin{smallmatrix}2L^2 & 2L\\ 2rL & r\end{smallmatrix}\bigr)\binom{S_q}{B_q}$ across arbitrary module sequences, giving the three-dimensional volume-growth condition $\rho(T_{\mathrm{blk}}) \le L_{\mathrm{blk}}^3$.
   - All-compact $C_2 C_2$ ($25{,}148$ qubits) exceeds this bound ($\rho = 9{,}281 > 16^3$) and is overtaken by $R_3 C_2$ at $\beta = 6.75$ on planar sheets and $\beta = 7.0$ on tori ($1.87\times$). Both $C_3 C_2$ ($\rho = 37{,}636 < 36^3$) and $R_3 C_2$ ($\rho = 45{,}930 < 36^3$) satisfy the bound, and keeping the second coarse bridge in $R_3 C_2$ increases cold lifetimes by $2.8\text{--}4.0\times$ over $C_3 C_2$ for $25\%$ more qubits.

3. **Equilibrium-Preserving Zero-Energy Kinetic Test:**
   - Multiplying zero-energy ($\Delta E = 0$) flip rates by $\mu \in [0.01, 3.0]$ on flat-sheet qubits while keeping $H$ and the Boltzmann–Gibbs equilibrium state unchanged increases $\tau_X(R_3 C_3)$ at $\beta = 6.5$ by $6.57\times$ ($3.56\times$ planar and $4.40\times$ torus above equal-size seam controls).

## Key Files

- `paper/main.tex`, `paper/main.pdf`: 9-page *Physical Review X*-formatted manuscript.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `src/reproducibility_certificates.py`, `certificates_v21.json`: Standalone verification of Proposition 1 across multi-step mixed schedules, volume-growth bounds, SHA-256 check-matrix hashes, and Markov-generator detailed balance under zero-energy kinetic scaling.
- `cluster_v9/build_v16_consolidation.py`, `cluster_v9/kmc_engine_v16.cpp`: Code generator, exact $\mathbb{F}_2$ verifier, `.c1map` / `.qmask` builders, and multithreaded C++20 BKL engine (`kmc_off`, `span_off`, `final_readout`, `kinetic_zero_e`).
- `cluster_v15_results.txt`, `cluster_v16_results.txt`, `src/parse_v16_results.py`, `parsed_v16_data.json`: Raw SubMIT trajectory logs with per-run stop times and censor flags, independent-sector bootstrap parser, and compiled statistics.
- `src/render_v21_prx_figures.py`, `figures/fig_v21_*.pdf`: Publication figure scripts and vector PDFs.
