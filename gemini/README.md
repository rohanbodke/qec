# Thermal Protection in Compact Recursively Welded Codes (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the explicit recursive welded CSS code constructors, standalone algebraic and dynamical certificates, C++20 continuous-time BKL heat-bath simulation and paired readout/intervention engines, MIT SubMIT SLURM scripts, raw cluster benchmark logs, figure generation scripts, and compiled APS REVTeX 4.2 manuscripts investigating what finite-size thermal protection survives when BDL Section 6.2 recursive sheet-and-bridge codes ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943)) are simplified to compact cells on bounded-degree graphs.

---

## Repository Map

* **Manuscripts**
  * `paper/main.tex` & `paper/main.pdf` — Ten-page APS REVTeX 4.2 paper (*Thermal Protection in Compact Recursively Welded Codes*), formatted in *Physical Review X* style with five clean figures and a three-part technical appendix.
  * `proposal_improved.tex` & `proposal_improved.pdf` — Two-page APS REVTeX 4.2 project summary.
  * `paper/references.bib` & `references.bib` — Bibliography including BDL (2026), Hayden Part I (2026), Stengele et al. (2026), Bergamaschi et al. (2025), Gu et al. (2026), Williamson (2026), and Qiu, Zoller, & Li (2020).

* **CSS Code Builders & Standalone Certificates**
  * `src/bdl_css_builder.py` — Deterministic cellular sheaf CSS builder (`seed_planar_11` $[[2,1,(2,1)]]$, `seed_rot_torus_422` $[[4,2,2]]$, alternating odd $X$-weld and even $Z$-weld steps, and $\ell = 8$ cubic-grid comparison).
  * `cluster_v9/build_v13_referee_upgrade.py` & `cluster_v9/build_v14_intervention.py` — Generate serialized CSS code instances, hierarchical `.rgdec` maps, scale-selective check-coupling intervention sets ($U_{\mathrm{macro}}, U_{\mathrm{sub}}, U_{\mathrm{C2,all}}$), and exact matched-$\beta_{\mathrm{eff}}$ wire-subdivided CSS codes (`intervention_metadata_v14.json`).
  * `src/reproducibility_certificates.py` & `certificates_v19.json` — Standalone 14-second Python verification suite certifying:
    1. Exact $2 \times 2$ ($2^{12}$ states) and $3 \times 3$ ($2^{24}$ states) auxiliary truth tables separating per-wall minority $W_{\mathrm{wall}} \le 2V$ from rounded wall-majority quotient $C \le \frac{2}{5}V$ ($C \le \frac{2}{7}V$ for $L=3$).
    2. Exact physical $19$-qubit interior parity coboundary $V \le V_{\mathrm{phys}} \le \frac{11}{5}V$, multi-cell simultaneous four-wall coarse lift $(H_Z^{\mathrm{fine}})^T(\pi^* y)$, and $3 \times 3$ periodic-torus pocket ranks ($\dim\ker H_Z^T = 37$ two-rib vs. $10$ one-rib).
    3. Fixed-rule T-bridge and 1-step $X+Z$ cross MILP dressed distances.
    4. Physical check-incidence qubit bounds ($3F_q \le I_q \le 4N_q \implies N_q \ge \frac{3}{4}F_0(2L^2+2L)^q$), diameter recurrences, and `sq_bdl_8_8_r2` collar degree anatomy.
    5. Exact CSS check-leg wire rank preservation ($K(w) = K(1)$), core syndrome marginals at $\beta_{\mathrm{eff}}(w) = 2\,\operatorname{arctanh}[\tanh^w(\beta/2)]$, and $16$-state CTMC heat-bath matrix inverses on $[[4,1,2]]$.

* **Continuous-Time BKL Monte Carlo Engine & Cluster Data**
  * `cluster_v9/kmc_engine_v14.cpp` (extending `kmc_engine_v13.cpp`) — Rejection-free continuous-time BKL heat-bath engine supporting inhomogeneous check couplings $H(\lambda; U)$, Cycle-1 coarse syndrome excursion tracking, paired `uf` vs. `rg_dec` vs. `rg_strict` readout, paired checkpoint-cadence sweeps ($\eta \in [0, 0.50]$), and matched-weight random vs. metastable thermal coarse-flow diagnostics.
  * `cluster_v9/run_shard_v14.py` & `cluster_v9/slurm_worker_v14.sh` — SubMIT SLURM shard dispatcher and worker script.
  * `cluster_v14_results.txt` & `cluster_v13_results.txt` — Complete raw SubMIT simulation outputs (`608` zero-error tasks in `v14` plus `343` baseline/wire tasks in `v13`).
  * `src/parse_v14_results.py`, `parsed_v14_data.json`, & `parsed_v14_summary.txt` — Parses raw trajectories into sector-resolved lifetimes, exact censoring brackets, bootstrap confidence intervals, complete-cycle gains $G_r(\beta)$, scale-selective sensitivity ratios, paired readout/cadence ratios, and matched-$\beta_{\mathrm{eff}}$ wire exponents.

* **Figures**
  * `src/render_v19_focused_figures.py` — Generates the five publication figures (`figures/fig_v19_1_construction.{pdf,png}` through `figures/fig_v19_5_implementation.{pdf,png}`) directly from `parsed_v14_data.json` and `parsed_v13_data.json`.

---

## Reproducing Certificates, Parsed Data, Figures, and PDFs

```bash
# 1. Run exact algebraic, topological, and dynamical certificates (~14s)
python3 src/reproducibility_certificates.py

# 2. Parse raw SubMIT cluster logs into summary JSON & tables
python3 src/parse_v13_results.py
python3 src/parse_v14_results.py

# 3. Render all 5 figures (PDF + 300 DPI PNG)
python3 src/render_v19_focused_figures.py

# 4. Compile main manuscript and 2-page project summary
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && cd ..
pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
