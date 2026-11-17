# Manifest: manuscript items → scripts → outputs → inputs (2026-11-16)

Numbers and labels are those of the paper, *Sharp Asymptotic Theory of Maximum Likelihood
Estimation for Gaussian Processes with a Matérn Kernel* (Qaqish, Banerjee and Li).
"report §X" means the section of `results/summary/report.txt` headed X. Some report
headings keep older table numbers (for example "Table 4: coverage at the largest n" is
Table 3 of the paper); the Item column gives the paper's number.

**Pipeline.**
- Summaries: `python -m matern_sim.aggregate --out results` → `results/summary/{replicates,summary,joint,nugget,theory}.csv`.
- Report: `python -m matern_sim.report --out results > results/summary/report.txt`.
- Figures: `PYTHONPATH=repro/pylib SOURCE_DATE_EPOCH=1794268800 python -m matern_sim.plots --out results --fig paper/figures`.
- All three: `repro/regenerate.sh OUTDIR`, which also runs the Figure E1 check and compares every output with the frozen copy.

**Inputs** (post-refit; `results/*_prerefit` is never read, `repro/guard.py` checks this):
- `fits`: `results/fits/<tag>/<variant>/<law>/repNNNN.npz` (fit, curvatures at the fit, κ_pair);
- `kappa`: `results/kappa/...` (κ̃, eq. A.54, and the oracle nugget fit);
- `hess0`: `results/hess0/...` (Hessian at θ0);
- `theory`: `results/theory/d*_nu*_G*.npz` (I_n(θ0), D_k);
- `score0`: `audit/out/score0{,_local}/ok/...` (score at θ0).

All of these are in the inputs archive `matern_inputs_2026-11-16.tar.gz` (README §7), together with `results/kappa/fit_hosts.csv`, which gives the processor family and thread count of every fit. `replicates.csv` (78 MB) is also in the data archive; every other file under `results/summary/` and `paper/figures/` is in the repository.

## Tables and figures

| Item | Label | File / produced by | Output | Inputs |
|---|---|---|---|---|
| Tables 1, 2 | `tab:rates`, `tab:parameterizations` | theory (no simulation) | — | — |
| Figures 1–6 | `fig:sim-d{3,4,5}-nu{05,15}` | `matern_mle_log_d{d}_nu{ν}.pdf`, `plots.fig_setting(raw=False)` | `paper/figures/` | `replicates.csv`, `summary.csv`, `theory.csv` |
| Figure 7 | `fig:sim-laws` | `matern_mle_laws.pdf`, `plots.fig_laws` (1000 paired bootstrap resamples, fixed seed) | `paper/figures/` | `replicates.csv` |
| Table 3 | `tab:sim-coverage` | `report.coverage` (report §"Table 4: coverage at the largest n") | `report.txt` | `summary.csv` ← fits, kappa, theory |
| Table 4 | `tab:sim-joint-steps` | `audit/joint_diagnostic.py`, `audit/score_exact_mc.py` | `audit/out/followup/task1_norms_coverage_d5_gauss.csv`, `task1_score_exact_mc_d5.csv` | fits, kappa, theory, score0 |
| Table 5 | `tab:sim-joint` | `report.joint`; exact counts and ranges: `audit/appendixE_q3_q4.py` (q4) | `audit/out/appendixE/q4_table_sim_joint.csv`, `q3_q4.txt` | `joint.csv` (`variant=joint`, `vector=log_vector`, `method=fisher_full_khat`) |
| Tables 6–8 | `tab:coordinate-expansions`, `tab:joint-tail-remainders`, `tab:forward-remainder-orders` | theory | — | — |
| Table E1 | `tab:sim-blocks` | `matern_sim/symmetry.py`; `audit/appendixE_q1.py` | `audit/out/appendixE/q1_block_multiplicities.csv`, `q1.txt` | none (deterministic) |
| Table E2 | `tab:sim-boundary` | `report.boundary` (report §"Boundary solutions") | `report.txt` | `replicates.csv` ← fits |
| Table E3 | `tab:sim-joint-chain` | `audit/joint_diagnostic.py`, `audit/score_exact_mc.py`; Log (fit), Log (truth) columns: `audit/appendixE_q3_q4.py` | `audit/out/followup/task1_*`, `audit/out/appendixE/q3_q4.txt` | fits, kappa, theory, score0 |
| Table E4 | `tab:sim-slopes` | `report.slopes` (report §"Table 3: least-squares slopes") | `report.txt` | `replicates.csv`, `theory.csv` |
| Table E5 | `tab:sim-se` | `report.eight`; counts: `audit/appendixE_q6.py` (b); TeX body: `audit/appendixE_table_e5_tex.py` | `audit/out/appendixE/q6b_table_E5_counts.csv`, `table_E5_body.tex`, `table_E5_cells.csv` | `summary.csv` ← fits, kappa, hess0, theory |
| Figures E1–E3 | `fig:sim-se`, `fig:sim-se-d4`, `fig:sim-se-d5` | `matern_mle_se_d{3,4,5}.pdf`, `plots.fig_se_dim` (statistics: `plots.se_curves`) | `paper/figures/` | `summary.csv` |
| Combined SE figure (not in the paper) | — | `matern_mle_se.pdf`, `plots.fig_se` (the three dimensions of Figures E1–E3 in one figure; same `se_curves`) | `paper/figures/` | `summary.csv` |
| Figure E1 check | — | `audit/appendixE_fig_e1_by_dim.py` (identical arrays: combined = per-dimension = pre-relabel; only text change is the row-3 label) | `fig_e1_array_check.txt`; frozen pre-relabel arrays `fig_e1_arrays_before_relabel.npz`, `fig_e1_texts_before_relabel.json` | `summary.csv` |
| Figures E4–E9 | `fig:sim-raw-d{3,4,5}-nu{05,15}` | `matern_mle_d{d}_nu{ν}.pdf`, `plots.fig_setting(raw=True)` | `paper/figures/` | `replicates.csv`, `summary.csv`, `theory.csv` |

## Numbers in the text (Appendix E)

| Item | Script | Output (`audit/out/...`) |
|---|---|---|
| E.1.1 folded entries (AE1) | `audit/appendixE_q2.py` | `appendixE/q2_folded_entries.txt` |
| E.1.2 Σ m_ρ dim B_ρ = n on all grids (AE2) | `audit/appendixE_q1.py` | `appendixE/q1_block_multiplicities.csv` |
| E.2.3 data generation, eigenvalue clipping | `audit/field_generation_check.py`, `audit/field_clip_scan.py` | `followup/task3_*` |
| E.2.4 restart rule, KKT, evaluation counts | `audit/audit_fits.py`, `audit/refit_claims.py`, `audit/refit_compare.py` | `refit_claims.txt`, `refit_compare.txt` |
| E.2.4 initialization audit, 899/912 runs (AE3) | `audit/multistart.py` (regenerated data, AMD login node), `audit/appendixE_q3_q4.py` (q3) | `ms_local/multistart_{A..E}.csv`, `appendixE/q3_*.csv`, `appendixE/q3_q4.txt` |
| E.3.1 gradient and log-Bessel checks | `tests/`, `audit/dlogK_check.py` | `test_validation.txt`, `appendixE/dlogK_check.txt` |
| E.3.1 cumulant dense references | `audit/appendixE_q7_kappa_ref.py` | `appendixE/kappa_dense_refs.txt` |
| E.3.1 Hessian chain-rule check (FE3) | `audit/hessian_chart_check.py` | `hessian_chart_check.txt` |
| E.3.1 finite-difference step | `audit/hess_step_check.py` | `hess_step_check.txt`, `appendixE/hess_step_check_rerun.txt` |
| E.4.1 validity rule, joint ranges, uncorrected 66.25% | `matern_sim/aggregate.joint_wald`; `audit/joint_validity_compare.py`; `audit/appendixE_q3_q4.py` (q4); `audit/refit_claims.py` | `appendixE/q3_q4.txt`, `refit_claims.txt` |
| §4.2 oracle vs fitted joint, paired SE | `audit/joint_paired.py` | `joint_paired.txt` |
| E.5.1 Fisher vs Hessian example 100%/80% (FE4) | `audit/appendixE_final_summaries.py` (2b) | `appendixE/final_2b_d5_nu05_n1024_lognu.csv`, `final_summaries.txt` |
| E.5.1 invalid shares 1.5%/7.5%/0.5%, uncorrected 0.75/2.0/0.5%, indefinite 53 and 2.25%, 98 (FE5, FE6) | `audit/appendixE_final_summaries.py` (2c); `audit/appendixE_q6_perfit.py` | `appendixE/final_2c_cells.csv`, `q6c_perfit_small_n.csv`, `q6c_perfit.txt` |
| E.5.1 failure counts 148 = 50 + 98, 199 = 197 + 2, 51 joint-only, 3 Fisher | `audit/appendixE_q6_perfit.py`, `audit/appendixE_q6.py` (c) | `appendixE/q6c_perfit_small_n.csv`, `q6c_per_fit_flags.csv`, `q6c_failure_types.csv`, `q6.txt` |
| E.5.1 median SEs within 2% (Gaussian), 4% (three full versions) (FE7) | `audit/appendixE_final_summaries.py` (2d) | `appendixE/final_2d_median_se_largest_n.csv` |
| E.5.1 block coverage ranges, d = 3 | `audit/appendixE_q6.py` (a) | `appendixE/q6a_block_coverage_max.csv` |
| E.5.1 H(θ0) 36.5%, reported-chart comparison | report §"Eight SEs …"; `audit/hess_reported_chart.py`, `audit/hess_reported_summary.py` | `hess_reported_chart.csv`, `hess_reported_summary.txt` |
| Processor provenance of every post-fit pass | `audit/postfit_processor_audit.py` | `postfit_processor.txt`, `postfit_processor.csv.gz` |

`repro/regenerate.sh` reruns the pipeline rows (aggregate, report, figures and the Figure E1 check) from the saved inputs and compares the results with the frozen copies.
The `audit/*` rows are rerun with the commands in their docstrings and write to `audit/out/`. They read the same saved inputs, except the initialization audit and the Hessian checks, which regenerate data. Apart from the two frozen Figure E1 references in `audit/out/appendixE/`, the audit outputs are not in the repository.
