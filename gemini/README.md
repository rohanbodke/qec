# Selective Multi-Scale Protection in Recursively Welded Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the explicit recursive welded CSS code constructors, standalone algebraic and mixed-hierarchy certificates, C++20 continuous-time BKL heat-bath simulation and coarse-excursion/readout engines, MIT SubMIT SLURM scripts, raw cluster benchmark logs, PRX-style figure generation scripts, and compiled APS REVTeX 4.2 manuscripts investigating whether the additional local bridge structure in BDL Section 6.2 recursive sheet-and-bridge codes ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943)) must be repeated at every scale or can be retained selectively at chosen scales.

---

## Repository Map

* **Manuscripts**
  * `paper/main.tex` & `paper/main.pdf` — Ten-page APS REVTeX 4.2 paper (*Selective Multi-Scale Protection in Recursively Welded Quantum Memories*), formatted in *Physical Review X* style with four clean figures, two quantitative specification/lifetime tables, and a three-part technical appendix.
  * `proposal_improved.tex` & `proposal_improved.pdf` — Two-page APS REVTeX 4.2 project summary.
  * `paper/references.bib` & `references.bib` — Bibliography including BDL (2026), Hayden Part I (2026), Stengele et al. (2026), Bergamaschi et al. (2025), Gu et al. (2026), Williamson (2026), and Qiu, Zoller, & Li (2020).

* **CSS Code Builders & Standalone Certificates**
  * `cluster_v9/build_v15_selective_hierarchy.py` — Builds and verifies all `30` single-cycle (`C2`, `C3`, `R3`, isotropic `C3_iso`, `R3_iso`) and two-cycle (`CC`, `RC`, `CR`, `RR`) CSS codes up to `709,408` physical qubits across open planar ($K = 1$), boundaryless one-face torus ($K = 2$), and four-face rotated torus ($K = 2$) seeds (`v15_code_specs.json`), plus exact Cycle-1 parent-face maps (`.c1map`).
  * `src/reproducibility_certificates.py` & `certificates_v20.json` — Standalone Python verification suite certifying:
    1. Exact $2 \times 2$ ($2^{12}$ states) and $3 \times 3$ ($2^{24}$ states) single-rib cell truth tables separating per-wall minority $W_{\mathrm{wall}} \le 2V$ from rounded wall-majority quotient $C \le \frac{2}{5}V$ ($C \le \frac{2}{7}V$ for $L=3$).
    2. Physical $19$-qubit interior parity coboundary bounds, multi-cell coarse lift consistency, and $3 \times 3$ torus check-pocket counts ($10$ one-rib vs. $37$ two-rib; $14$ vs. $53$ on full-cycle $C_3$ vs. $R_3$).
    3. Fixed-rule T-bridge and 1-step $X+Z$ cross ablations.
    4. Mixed-hierarchy aggregate 3D ball-packing bounds at $L = 3$ ($m_C = 24 < 27$, $m_R = 30 > 27$, while periodic $(C_3, R_3)$ blocks satisfy $m_C m_R = 720 < 27^2 = 729$ and $518{,}400 < 531{,}441$).
    5. Exact CSS check-leg wire rank preservation ($K(w) = K(1)$), core syndrome marginals at $\beta_{\mathrm{eff}}(w) = 2\,\operatorname{arctanh}[\tanh^w(\beta/2)]$, and $16$-state CTMC matrix inverses.

* **Continuous-Time BKL Monte Carlo Engine & Cluster Data**
  * `cluster_v9/kmc_engine_v15.cpp` (extending `kmc_engine_v14.cpp` & `kmc_engine_v13.cpp`) — Rejection-free continuous-time BKL heat-bath engine supporting connected Union-Find coarse-cell excursion tracking (`span_kmc`), paired checkpoint-cadence sweeps (`paired_cadence`), paired `uf` vs. `rg_dec` readout, and inhomogeneous check couplings $H(\lambda; U)$.
  * `cluster_v9/run_shard_v15.py` & `cluster_v9/slurm_worker_v15.sh` — SubMIT SLURM shard dispatcher and worker script.
  * `cluster_v15_results.txt` (plus `cluster_v14_results.txt` & `cluster_v13_results.txt`) — Complete raw SubMIT simulation outputs (`570/570` zero-error tasks in `v15` across all `30` selective-hierarchy codes, `608` intervention/readout tasks in `v14`, and `343` wire/baseline tasks in `v13`).
  * `src/parse_v15_results.py` & `parsed_v15_data.json` — Computes exact joint two-sector lifetimes $\tau_Q = \int_0^\infty S_X(t)S_Z(t)\,dt$, censoring brackets, $2{,}000$-resample bootstrap CIs, pairwise `RC/CR` and `CR/RR` ratios, cold-regime slopes $\hat{\Delta}_{\mathrm{eff}}$, coarse excursion arrival/survival metrics, and paired cadence summaries.

* **Figures**
  * `src/render_v20_prx_figures.py` — Generates the four *Physical Review X* figures (`figures/fig_v20_1_modules_and_claim1.{pdf,png}` through `figures/fig_v20_4_reliability_and_scope.{pdf,png}`) directly from `parsed_v15_data.json`, `parsed_v14_data.json`, `parsed_v13_data.json`, and `certificates_v20.json`.

---

## Reproducing Certificates, Parsed Data, Figures, and PDFs

```bash
# 1. Run exact algebraic, mixed-hierarchy volume, and dynamical certificates (~43s)
python3 src/reproducibility_certificates.py

# 2. Parse raw SubMIT cluster logs into summary JSON
python3 src/parse_v13_results.py
python3 src/parse_v14_results.py
python3 src/parse_v15_results.py

# 3. Render all 4 PRX figures (PDF + 300 DPI PNG)
python3 src/render_v20_prx_figures.py

# 4. Compile main manuscript and 2-page project summary
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && cd ..
pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
