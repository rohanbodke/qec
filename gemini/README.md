# The Thermal Value of Welding Across Scales

Companion code, exact algebraic/combinatorial certificates, C++20 continuous-time Bortz–Kalos–Lebowitz (BKL) simulation engine, raw per-trajectory stop times and censor flags, statistical bootstrap parsers, and REVTeX 4.2 manuscript sources for **MIT 6.S980 (Quantum Error Correction)**.

## Overview

Building on the explicit welded cell architecture introduced by Balasubramanian, Davydova, and Lin (BDL Section 6.2), we study how local cell structure determines whether additional interior welding is more useful at fine or coarse scales, and when retaining that structure justifies its resource cost across `34` bounded-degree CSS codes up to $709{,}408$ physical qubits:

1. **Scale-Placement Reversal Verified by Operational Final Readout and First-Passage Statistics:**
   - Pairing the two-rib $3 \times 3$ module $R_3$ ($1{,}480$ qubits) with the single-rib $3 \times 3$ module $C_3$ ($1{,}036$ qubits, whose single cross at $(1, 1)$ partitions each sheet into $1 \times 1, 1 \times 2, 2 \times 1, 2 \times 2$ regions) favors fine-scale placement ($C_3 R_3$, $591{,}208$ qubits) over coarse-scale placement ($R_3 C_3$, $499{,}088$ qubits) at $\beta = 6.5$ by $10.3\times$ in first-detected failure time $\tau_Q$ and $10.3\times$ in one-shot $80\%$-recovery readout time $t_{0.80}$ (passive evolution without intermediate stopping).
   - Replacing $C_3$ by the $2 \times 2$ single-rib module $C_2$ ($254$ qubits, four $1 \times 1$ unit faces) reverses the preferred placement across planar, one-face torus, two-face rotated torus $[[4, 2, 2]]$, and isotropic codes: coarse placement ($R_3 C_2$, $133{,}864$ qubits) exceeds fine placement ($C_2 R_3$, $136{,}536$ qubits) by $7.80\times$ in $\tau_Q$ ($4.50\times$ in $t_{0.80}$) at planar $\beta = 6.5$, $19.6\times$ at planar $\beta = 6.75$, and $31.5\times$ on uncensored periodic tori at $\beta = 7.0$, turning a loss relative to single-cycle $R_3$ ($0.46\times$ in $C_2 R_3$) into a $3.55\text{--}10.6\times$ gain ($2.70\times$ in $t_{0.80}$).
   - Between two single-rib modules ($C_3$ vs. $C_2$), placing $C_2$ at the fine scale ($C_3 C_2$, $107{,}140$ qubits) exceeds $C_2 C_3$ ($95{,}896$ qubits) by $30.6\text{--}141\times$.

2. **Proved Thin-Bridge Two-Type Face Recurrence, Spectral Volume Screen, and `C3C2` Comparator:**
   - Proposition 1 proves directly from the cell-complex generator that thin-bridge face counts satisfy $\binom{S_{q+1}}{B_{q+1}} = \bigl(\begin{smallmatrix}2L^2 & 2L\\ 2rL & r\end{smallmatrix}\bigr)\binom{S_q}{B_q}$ at every depth and across module changes, yielding the three-dimensional spectral volume condition $\rho(T_{\mathrm{blk}}) \le L_{\mathrm{blk}}^3$.
   - All-compact $C_2 C_2$ ($25{,}148$ qubits) violates this screen ($\rho = 9{,}281 > 16^3 = 4{,}096$) and is overtaken by $R_3 C_2$ at $\beta = 6.75$ planar and $\beta = 7.0$ torus ($1.87\times$). Both $C_3 C_2$ ($\rho = 37{,}636 < 36^3 = 46{,}656$) and $R_3 C_2$ ($\rho = 45{,}930 < 46{,}656$) satisfy the screen, and adding the second coarse rib in $R_3 C_2$ gains $2.8\text{--}4.0\times$ over $C_3 C_2$ at $25\%$ higher qubit count.

3. **Detailed-Balance-Preserving Zero-Energy Kinetic Intervention:**
   - Rescaling $\Delta E = 0$ transition rates by $\mu \in [0.01, 3.0]$ on non-seam $d_Z = 2$ flat-region qubits while keeping $H$ and the Boltzmann–Gibbs distribution invariant raises $\tau_X(R_3 C_3)$ at $\beta = 6.5$ by $6.57\times$ ($3.56\times$ planar and $4.40\times$ torus above equal-cardinality $d_Z = 2$ seam controls).

## Key Files

- `paper/main.tex`, `paper/main.pdf`: 10-page *Physical Review X*-formatted manuscript.
- `proposal_improved.tex`, `proposal_improved.pdf`: 2-page project summary.
- `src/reproducibility_certificates.py`, `certificates_v21.json`: Standalone verification of Proposition 1 across multi-step mixed schedules, spectral volume screens, SHA-256 check-matrix hashes, $Z$-cochain wall bounds ($W_{\mathrm{wall}} \le 2V_{\mathrm{phys}}$), check-wire marginals, and 64-state Markov-generator Gibbs invariance under zero-energy kinetic scaling.
- `cluster_v9/build_v16_consolidation.py`, `cluster_v9/kmc_engine_v16.cpp`: Code generator, exact GF(2) verifier, `.c1map` / `.qmask` builders, and multithreaded C++20 BKL engine (`kmc_off`, `span_off`, `final_readout`, `kinetic_zero_e`).
- `cluster_v15_results.txt`, `cluster_v16_results.txt`, `src/parse_v16_results.py`, `parsed_v16_data.json`: Raw SubMIT trajectory logs with per-trial physical stop times and censor flags, independent-sector bootstrap parser, and compiled statistics.
- `src/render_v21_prx_figures.py`, `figures/fig_v21_*.pdf`: Publication figure scripts and vector PDFs.
