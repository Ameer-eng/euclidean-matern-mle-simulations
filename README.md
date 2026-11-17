# Simulations for "Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with a Matérn Kernel"

Code, Slurm scripts, run manifests, Monte Carlo summaries and figures for
Section 4 and Appendix E of

> Ameer Qaqish, Sudipto Banerjee and Didong Li. *Sharp Asymptotic Theory of Maximum
> Likelihood Estimation for Gaussian Processes with a Matérn Kernel.*

The study fits the full Matérn model (variance, range, smoothness and nugget) by
maximum likelihood on regular grids in d = 3, 4, 5 with n up to about 10⁵, under four
measurement-error laws. It checks the asymptotic theory and the feasible Wald
intervals of the paper.

| path | content |
|---|---|
| `matern_sim/` | simulation package: exact blockwise likelihood, fitting, post-fit passes, summaries, report, figures (§2) |
| `tests/` | validation against dense computations |
| `audit/` | scripts behind individual numbers in the text (see `repro/MANIFEST.md`) |
| `slurm/`, `manifests/` | Slurm scripts and the array manifests and plans used for the runs |
| `repro/` | regeneration of the summaries, report and figures from the saved per-fit outputs; `MANIFEST.md` maps every table, figure and in-text number to its script |
| `results/summary/` | Monte Carlo summaries (`summary.csv`, `joint.csv`, `nugget.csv`, `theory.csv`) and `report.txt`, which prints every Monte Carlo number quoted in the paper |
| `paper/figures/` | the figures of Section 4 and Appendix E, as PDF |

The per-fit outputs and the per-replicate table `replicates.csv` are too large for
this repository and are archived separately (§6).

## 1. What is simulated and why

Equation, theorem and table numbers refer to the paper.

| Paper result | What is checked | Output |
|---|---|---|
| Thm 3.1: the limits of the signal estimators (ν, c or σ², l) do not depend on the error law | RMSE and IQR/1.349 against n should coincide across laws | `summary.csv` (`rmse`, `iqr_sd`, `sd_law_*` vs `sd_gauss_*`); Figures 1–7, E4–E9 |
| Thm 3.1, eqs. (3.10a) and (3.15): only the nugget limit changes; √n·SD(log τ̂²) → √(Eξ⁴−1) | per-law nugget SD against the limit and against the exact finite-n sandwich | `nugget.csv` (`sqrt_n_sd_logtau2`, `limit_sd`, `sqrt_n_sd_theory_*`) |
| Eq. (3.10b): the nugget error is asymptotically uncorrelated with the signal errors | correlations → 0 | `nugget.csv` (`corr_tau_*`) |
| Eq. (3.15): the nugget MLE has the same limit as when the signal is known | √n·RMSE(log t̂ − log t̃_oracle) → 0 | `nugget.csv` (`sqrt_n_rmse_joint_minus_oracle`) |
| Cor 3.6: feasible Wald intervals with the fourth-cumulant correction of eq. (3.22) | coverage of 95% intervals (expected information vs observed Hessian; full inverse vs identifiable block for d = 3); Table 3 | `summary.csv` (`cov_*`) |
| Cor 3.6 without the correction | the same intervals with κ₄ estimate := 0 (large-n prediction for the nugget: about 99.8% coverage under `unif`, about 69% under `mixnorm`) | `cov_*_none` |
| Lemma A.12: residual estimator κ̃₄ (eq. A.52) | bias and SD of κ̃₄; the `ktrue` intervals isolate the error from estimating κ₄; an earlier adjacent-pair estimator is kept as `kpair` | `nugget.csv` (`kappa_hat_*`, `kappa_pair_*`), `cov_fisher_full_ktrue`, `cov_fisher_full_kpair` |
| Cor 3.4, eq. (3.20), and Cor 3.6, eq. (3.23): joint χ² for the log vectors | coverage and mean of the quadratic statistic; Tables 4, 5 and E3 | `joint.csv` (`vector=log_vector`) |
| App. D.4: the joint limit fails on the (σ², l, ν, τ²) scale for 4 ≤ d ≤ 8 and ν₀ ≠ 1/2 | joint χ² coverage on that scale against the log vector | `joint.csv` (`vector=raw_public`) |
| Which of eight standard errors is most accurate (Fisher vs Hessian, at θ̂ vs θ₀, full inverse vs identifiable block for d = 3); App. E.4.6 and E.5.1 | coverage, robust SD of error/SE (`madz_*`), SE spread (`se_cv_*`), share of negative variances (`n_se_*`); Table E5, Figures E1–E3 | `summary.csv` (`*_{fisher,hess,fisher0,hess0}_{full,block}_khat`) |
| Cor 3.7: known smoothness | c at rate r_n⁻¹ | variant `knownnu` (implemented; not run for the paper) |

The truth, grids, optimisation box and L-BFGS-B settings are those of Section 4.1
and Appendix E.2: (σ², l, τ²) = (1, 0.4, 0.25), ν₀ ∈ {1/2, 3/2}, d ∈ {3, 4, 5}, and
G ∈ {5, 7, 10, 14, 22, 32, 40, 46} (d = 3), {3, 4, 6, 8, 10, 13, 18} (d = 4),
{3, 4, 5, 6, 8, 10} (d = 5), with n = Gᵈ. All of this is in `matern_sim/config.py`.

### Error laws (`matern_sim/noise.py`)

All four laws are standardized (Eξ = 0, Eξ² = 1), sub-Gaussian, and have Eξ⁴ − 1 > 0, so they satisfy Assumption 4. The latent field is always Gaussian.

| law | definition | Eξ⁴ | κ₄/τ₀⁴ | limiting SD of √n(log τ̂² − log τ₀²) |
|---|---|---|---|---|
| `gauss` | N(0,1) | 3 | 0 | 1.414 |
| `unif` | Uniform(−√3, √3) | 1.8 | −1.2 | 0.894 |
| `mixnorm` | 0.9·N(0,s²) + 0.1·N(0,9s²), s² = 1/1.8 | 8.33 | 5.33 | 2.708 |
| `bern` | (B − 0.1)/0.3, B ~ Bernoulli(0.1) | 8.11 | 5.11 | 2.667 |

`mixnorm` and `bern` have nearly the same fourth moment but very different shapes:
`bern` is skewed and discrete. If the two give the same nugget variance, that
supports the claim that only Eξ⁴ enters.

### Design choices

* **400 replicates** per (d, ν₀, G, law). The Monte Carlo SE of a 95% coverage
  estimate is then about 1.1%.
* **Common random numbers.** Replicate r uses the same latent field for all four
  laws; only the error draws differ, so between-law differences reflect the
  noise and not field-to-field variation. Seeds:
  `SeedSequence((ROOT_SEED, d, round(10ν₀), G, r, 0))` for the field and
  `(..., r, 1, law_id)` for the errors (`ROOT_SEED = 20261105`, PCG64).
* **Data generation (App. E.2.3).** The field is drawn in block coordinates from
  the signal blocks S_ρ(θ₀) = V_ρΛ_ρV_ρᵀ, and the errors are drawn iid on the grid
  and rotated by Qᵀ. (Drawing Z_ρ = L_ρΞ_ρ in block coordinates would be exact only
  for Gaussian errors.)
* **κ̃₄, the residual estimator of eq. (A.52).** With Â = t̂C(θ̂)⁻¹, r̂ = ÂY and
  v̂ᵢ = t̂Âᵢᵢ, κ̃₄ = Σᵢ(r̂ᵢ⁴ − 6v̂ᵢr̂ᵢ² + 3v̂ᵢ²) / Σᵢⱼ Âᵢⱼ⁴, all in the **grid** basis
  (fourth powers are not invariant under Q). `kappa.py` computes it with one
  blockwise inverse at θ̂: r̂ and diag Â from the blocks, and Σᵢⱼ Âᵢⱼ⁴ from one row
  of C⁻¹ per hyperoctahedral orbit of grid points (C⁻¹ is group invariant, so a
  row's fourth-power sum is constant on its orbit), each row assembled from the
  parity blocks. Cost ≈ one gradient evaluation per fit. The earlier
  adjacent-pair estimator is still stored per fit and reported as `kpair`.
* **κ̃₄ is computed after the fits**, by `kappa_task.py`, which regenerates each
  replicate's data from its seeds and checks that ℓ(θ̂) equals the stored value
  (relative 1e-8). **Exact regeneration needs the CPU family and the thread count
  of the fit**: for ν₀ = 3/2 the blocks of S(θ₀) are badly conditioned, and the
  signs of some eigenvectors (hence the field draw) depend on OpenBLAS's
  rounding, which differs between AVX2 and AVX-512 kernels and, for large blocks,
  with the number of threads. `fit_hosts.py` reads each fit's node and thread
  count from the task logs; the planner pins every κ̃₄ job to that family
  (`--constraint`, plus `--exclude` of the AVX-512 nodes for the AVX2 families)
  and thread count. A fit that still does not reproduce gets a marker in
  `results/kappa_mismatch/` and is left out of the khat coverage.
* **Stored per fit:** θ̂, the score at θ̂, the 4×4 expected information and the
  observed negative Hessian at θ̂ (central differences of the analytic gradient
  in the optimiser's Fisher-scaled coordinates), the pair estimate of κ₄ and t̃,
  and the oracle nugget fit. In `results/kappa/` (same layout as `results/fits/`):
  κ̃₄ and its numerator and denominator. Everything else is computed afterwards.
* **Stored per setting** (`results/theory/`): I_n(θ₀) and the matrix
  Dk with Dk_pq = Σᵢ(C⁻¹C_pC⁻¹)ᵢᵢ(C⁻¹C_qC⁻¹)ᵢᵢ, so that Cov(score) = I + (κ₄/4)Dk
  holds **exactly** at finite n (eq. A.11). The benchmark line for each law is
  the sandwich I⁻¹(I + κ₄Dk/4)I⁻¹. At ν₀ = 1/2 the nugget is still far from its
  limit at n ≤ 10⁵, so the limit formula J⁻¹ + κ₄e_te_tᵀ of eq. (3.10a) alone would
  be a poor benchmark there.

## 2. Implementation (`matern_sim/`)

| module | content |
|---|---|
| `symmetry.py` | Hyperoctahedral block decomposition of App. E.1.1: per-axis parity bases, then Young's orthogonal form for S_k × S_{d−k} and the orbit-wise projectors P^ρ₁₁, P^ρᵢ₁. Q is never formed; `forward`/`inverse` apply Qᵀ/Q. |
| `kernel.py` | Matérn tables K, ∂_νK and ∂_{log a}K indexed by the integer squared lag D (r = √D/G). ∂_ν log 𝖪_ν uses the trapezoidal integral ratio. |
| `assemble.py` | Parallel numba kernels. Only the parity-block rows at **one representative per H-orbit** are assembled; every other row is a permuted copy (B[x,y] = B[rep, h⁻¹y]). Also: projection onto each irrep block, row-wise quadratic diagonals for Dk, and the in-place symmetrisation and trace products. |
| `likelihood.py` | Blockwise log-likelihood and analytic gradient, expected information (App. E.1.2), Dk, signal eigendecomposition, field draws, oracle profile; plus a dense reference implementation. |
| `fit.py` | L-BFGS-B in x = diag(√I_φ)(φ − φ₀), φ = (η − 2L_nν, ν, log a, log t); finite-difference Hessian; oracle nugget fit; adjacent-pair estimator. |
| `kappa.py` | residual fourth-cumulant estimator κ̃₄ (eq. A.52) in the grid basis; dense reference. |
| `kappa_task.py` | κ̃₄ for stored fits: regenerates the data, checks ℓ(θ̂), writes `results/kappa/`. Same arguments as `run_task`, plus `--family`/`--cpus`. |
| `fit_hosts.py` | node, CPU family and thread count of every fit, from the task logs (`results/kappa/fit_hosts.csv`). |
| `noise.py`, `config.py` | error laws; the whole design. |
| `run_task.py` | one task: replicates [r0, r1) of one (d, ν₀, G) for several laws. Skips existing outputs and writes each fit atomically. |
| `refit_task.py` | restart rule (App. E.2.4): fits with KKT residual > 10⁻³ are refitted from the stopping point, three far starts and a data-based start; the best is kept, κ̃ is recomputed, and the originals are kept in `results/*_prerefit/`. |
| `make_manifest.py` | writes array manifests and a plan, sized from benchmark timings; stage `kappa` groups the fits lacking κ̃₄ by (setting, CPU family, threads). |
| `hess0_task.py` | −∇²ℓ(θ₀) for stored fits (central differences of the exact gradient, batched over all data sets of a design since θ₀ is common); regenerates the data like `kappa_task` and checks the oracle-nugget criterion; writes `results/hess0/`. |
| `aggregate.py` | writes the CSV summaries listed in §1. Joint regions use a covariance only if it is finite, symmetric to a relative 1e-8 and positive definite (Cholesky), never clipped or jittered (App. E.4.1); `joint.csv` reports `n_valid`, `coverage_given_valid`, `prob_valid_and_cover` and `n_invalid_*`. |
| `report.py` | prints every Monte Carlo number quoted in Section 4 and Appendix E (`results/summary/report.txt`). |
| `plots.py` | Figures 1–7 and E1–E9 into `paper/figures/` (plus `matern_mle_se.pdf`, the three dimensions of Figures E1–E3 in one figure). Needs matplotlib, which the simulation itself does not use. |

**Validation** (App. E.3.1; `python -m tests.test_validation`, or `sbatch slurm/validate.sbatch`).
For seven designs with n ≤ 343, including odd G and parameters far from θ₀:
* Q round trip: 2e-13;
* log-likelihood vs dense Cholesky: 6e-16 relative;
* gradient vs dense trace formula: 1e-13;
* Fisher information and Dk vs dense: 2e-15 and 4e-15;
* gradient vs finite differences: 1e-7, which is the accuracy of the finite differences themselves.

`python -m tests.test_kappa` checks κ̃₄ on eight designs (d = 2–5, odd and even G)
at four parameter values: numerator, Σᵢⱼ Âᵢⱼ⁴ and κ̃₄ agree with a dense inverse to
3e-14, and at the true covariance κ̃₄ is unbiased for every law (300 replicates,
n = 512: |z| ≤ 1.5). `python -m tests.test_joint_validity` checks the validity rule
for joint regions. Regenerated data were checked against stored fits on every
CPU family (ℓ(θ̂) equal to 1e-12 on the fit's family and thread count).

The kernel equals the exponential covariance at ν = 1/2 to 3e-16, and ∂_ν log 𝖪_ν
matches its ν = 1/2 closed form to 1e-15. Block counts and largest blocks reproduce
Table E1 exactly: (10, 4200), (10, 6348), (20, 2160), (36, 600).

## 3. Installation

The runs used CPython 3.12.4 with the versions pinned in `requirements.lock.txt`
(numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, numba 0.68.1). The figures also need
matplotlib, kept outside the simulation environment:

```bash
git clone https://github.com/<owner>/euclidean-matern-mle-simulations.git && cd euclidean-matern-mle-simulations
module load python/3.12.4                       # or any CPython 3.12.4
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install --target repro/pylib matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 \
    fonttools==4.68.0 kiwisolver==1.5.1 pillow==13.0.0 pyparsing==3.3.3 packaging==26.3
source slurm/env.sh                             # activates .venv and pins the thread counts
python -m tests.test_validation && python -m tests.test_kappa && python -m tests.test_joint_validity
```

`slurm/env.sh` is sourced by every job. It loads `python/3.12.4` with Lmod (as on
the UNC Longleaf cluster, where the runs were made), activates `.venv`, sets every
thread count to `$SLURM_CPUS_PER_TASK` and sets `OPENBLAS_THREAD_TIMEOUT=4` (§4). On
a machine without environment modules, delete the two `module` lines.

## 4. Running the campaign (Slurm)

All commands run from the repository root.

```bash
sbatch slurm/validate.sbatch                                  # optional re-check on a compute node

# 1. timings (done once; sizes every later array; -> results/bench/*.json)
python -m matern_sim.make_manifest bench && slurm/submit.sh manifests/bench.plan

# 2. theory: I_n(theta0) and Dk on the MC grids and beyond (grey regions)
python -m matern_sim.make_manifest theory && slurm/submit.sh manifests/theory.plan

# 3. main campaign: 4 laws x 400 reps, joint fit
python -m matern_sim.make_manifest fit --name main            # prints tasks, walltimes, core-hours
slurm/submit.sh manifests/fit_main.plan --throttle 60

# 4. optional, not run for the paper: known-smoothness fits (Cor. 3.7) for two laws
python -m matern_sim.make_manifest fit --name knownnu --variants knownnu --laws gauss,mixnorm
slurm/submit.sh manifests/fit_knownnu.plan

# 4b. restart rule (App. E.2.4): refit every joint fit whose KKT residual
#     (max |projected gradient| in the scaled coordinates) exceeds 1e-3 from the
#     stopping point, the three far starts and a data-based start; keep the best.
#     Originals -> results/{fits,kappa}_prerefit/.  Run on the CPU family of the fit
#     (data check as in kappa_task); `--list` shows the flagged fits.
python -m matern_sim.refit_task --list
python -m matern_sim.refit_task            # or --family amd_avx512 --cpus 1 on such a node

# 5. residual kappa estimate (eq. A.52) for every fit that lacks it
python -m matern_sim.make_manifest kappa --name k1 --skip-done --target-hours 2 --safety 3
slurm/submit.sh manifests/kappa_k1.plan

# 6. observed Hessian at theta0 for every fit (for the eight-SE comparison)
python -m matern_sim.make_manifest hess0 --name h1
slurm/submit.sh manifests/hess0_h1.plan

# 7. summaries (after the arrays finish; or chain with --dependency=afterany:<ids>)
sbatch slurm/aggregate.sbatch          # -> results/summary/*.csv
python -m matern_sim.report --out results > results/summary/report.txt
PYTHONPATH=repro/pylib SOURCE_DATE_EPOCH=1794268800 python -m matern_sim.plots --out results --fig paper/figures
```

The manifests and plans used for the paper are in `manifests/`.

* **Resubmitting is safe.** Finished fits are skipped, so after a timeout or a
  node failure just rerun `slurm/submit.sh` with the same plan.
* **Adding replicates later:** `make_manifest fit --name more --rep-start 400 --nrep 200`.
* **A subset of dimensions:** `--dims 4,5`.
* **Progress:** `squeue -u $USER`, `ls results/fits/<tag>/joint/<law> | wc -l`, `logs/`.
* Set `SBATCH_ACCOUNT` / `SBATCH_PARTITION` in the environment to change the account or partition.

### Parallelism

* **Inside a fit:** parallel numba kernels (assembly, projection, traces,
  symmetrisation) plus multithreaded OpenBLAS (Cholesky, inverse, matrix
  products, eigendecomposition). `slurm/env.sh` sets every thread count to
  `$SLURM_CPUS_PER_TASK`.
* **Across fits:** Slurm job arrays (replicate chunks per task).
* **Serial by nature:** the L-BFGS-B iterations and the few seconds of
  per-task set-up.

Time per gradient evaluation at n ≈ 10⁵, measured on 1 and 4 cores:

| setting | 1 thread | 4 threads |
|---|---|---|
| d=5, G=10 | 2.3 s | 0.74 s |
| d=4, G=18 | 9.3 s | 2.8 s |
| d=3, G=46 | 45 s | 13.8 s |

That is roughly 80% parallel efficiency; peak memory is 5.4 GB for d=3. For
d=3 the time is dominated by the Cholesky factorisation and inverse of the
6348-row blocks. Because efficiency drops as threads are added,
`make_manifest` uses 1 CPU per fit task for n ≤ 3k and 4 otherwise, and covers
the volume with more array tasks. On a cluster that caps the CPUs per user, work
per core is what counts: 4-core tasks do 15–35% more per core than 8-core ones.
To trade core-hours for wall time, edit `cpus_for` in `make_manifest.py`.

**Thread oversubscription.** Inside a Slurm job the process sees only its own
cores. numpy's and scipy's OpenBLAS pools each spin about 0.1 s after every call,
which starves numba's threads. With 4–8 CPUs per task this slowed fits 3×
(d=3, G=46) to 600× (d=5, G=5). `slurm/env.sh` sets `OPENBLAS_THREAD_TIMEOUT=4`,
which fixes it without changing any numbers. Benchmarks run outside Slurm don't
show the problem; to reproduce it, pin the process with `taskset -c 0-3`.

**Automatic completion.** `slurm/sweep.sbatch` runs after all fit jobs in flight
have ended (an `afterany` dependency). It queues every fit that is still missing,
using `--skip-done`, and schedules itself again after those jobs (at most 3
sweeps). Then it switches to the κ̃₄ phase: it queues `kappa_task` for every fit
without κ̃₄ (at most 2 sweeps; the last also takes fits whose node is unknown),
and finally submits `slurm/aggregate.sbatch`. Each round is logged in
`logs/sweep.log`; `ALL FITS DONE` means the fits are complete, and `kappa phase
over` reports the number of fits whose data could not be regenerated. Start it
once: `sbatch --dependency=afterany:<id>:<id>... slurm/sweep.sbatch 1 3`
(`slurm/sweep.sbatch 1 2 kappa` runs only the κ̃₄ phase).

**Reruns.** Every fit is saved as it finishes, so a task that times out loses only
the fit in progress. To queue the missing fits by hand:
`python -m matern_sim.make_manifest fit --name rerunN --only d3G46,... --skip-done`,
then `slurm/submit.sh manifests/fit_rerunN.plan`. Only include settings that have
no tasks still queued or running, or the same replicates run twice.
`python -m tests.profile_grad <d> <G>` prints the stage-by-stage timing.

## 5. Compute budget

Measured with `--bench` on 4 threads of an AMD EPYC 7713 (`results/bench/*.json`).
Each replicate = 4 laws × (about 44 gradient evaluations + one information
matrix). Plan: `manifests/fit_main.plan`, made with `--target-hours 3`.

| setting | gradient (4 thr) | information (4 thr) | wall time per replicate | array tasks | CPUs | memory |
|---|---|---|---|---|---|---|
| d=3, G=46 | 13.6 s | 53 s | ≤ 44 min | 200 × 4 reps | 8 | 10 GB |
| d=3, G=40 | 4.9 s | 16 s | ≤ 16 min | 74 | 8 | 6 GB |
| d=4, G=18 | 2.8 s | 5.3 s | ≤ 9 min | 40 | 8 | 4 GB |
| d=3, G=32 | 1.1 s | 2.9 s | ≤ 3.4 min | 16 | 8 | 4 GB |
| d=5, G=10 | 0.8 s | 0.9 s | ≤ 2.4 min | 12 | 8 | 4 GB |
| 16 smaller settings | ≤ 0.3 s | ≤ 0.4 s | ≤ 1 min | 34 | 1–8 | 4 GB |

* **Total.** 376 array tasks, each requesting ≤ 6.5 h. The planner's 8,100
  core-hours is an upper bound: it charges the 4-thread time to all 8 CPUs.
  Expect roughly **4,500–5,000 core-hours**, of which d=3, G=46 is about 57%,
  G=40 about 21% and d=4, G=18 about 11%. The κ̃₄ and Hessian-at-θ₀ passes come
  on top.
* **Wall time** is set by how many cores Slurm grants at once: about 1 day at
  ~200 cores and about 2 days at ~100, longer on a busy queue.
* **Theory stage:** 62 short tasks, minutes in total.
* **Known-smoothness campaign (optional, 2 laws):** about 30% of the main campaign.

## 6. Reproducing the paper's numbers

* **Without any computation.** `results/summary/report.txt` prints every Monte
  Carlo number quoted in Section 4 and Appendix E, and `paper/figures/` holds the
  figures. `repro/MANIFEST.md` maps each table, figure and in-text number to the
  script that produces it, its output and its inputs.
* **From the saved per-fit outputs (no fitting).** `repro/regenerate.sh OUTDIR`
  rebuilds the summaries, the report and the figures from the saved outputs in
  about 10–25 minutes on 2–4 cores, checks that no pre-refit file is read
  (`repro/guard.py`), and compares everything with the copies in this repository
  (`repro/compare_regenerated.py`): CSVs and `report.txt` byte for byte, PDFs
  apart from their creation dates. It needs the inputs archive (§7). See
  `repro/REPRODUCE.md`.
* **From scratch.** §4 above; the seeds make every data set a function of
  (d, ν₀, G, replicate, law). Bitwise reproduction of a data set needs the same
  software versions, CPU family and thread count as the original fit; these are
  recorded per fit in `results/kappa/fit_hosts.csv` (in the inputs archive), and
  `repro/REPRODUCE.md` explains why.
* **Audits.** The scripts in `audit/` behind individual numbers in the text are
  listed in `repro/MANIFEST.md` with their commands. They write to `audit/out/`.

## 7. Data availability

Not in this repository, because of their size:

| file | size | content | restore to |
|---|---|---|---|
| `matern_inputs_2026-11-16.tar.gz` (+ `inputs_SHA256SUMS.txt.gz`, `SHA256SUMS`) | 85 MB (1.4 GB unpacked, 268,863 files) | post-refit per-fit outputs `results/{fits,kappa,hess0,theory}`, `results/kappa/fit_hosts.csv`, and the score-at-θ₀ files `audit/out/score0{,_local}` | repository root: `tar -xzf matern_inputs_2026-11-16.tar.gz` |
| `replicates.csv` | 78 MB | one row per fit (written by `matern_sim.aggregate`) | `results/summary/` |

Archive: *DOI to be added.*

## 8. Licence and citation

Code (`matern_sim/`, `audit/` scripts, `repro/`, `slurm/`, `manifests/`, `tests/`) is
released under the MIT licence (`LICENSE`). The simulation outputs and figures
(`results/`, `audit/out/`, `paper/figures/` and the data archive) are released under
CC BY 4.0 (`LICENSE-DATA`). If you use this code or these results, please cite the paper;
`CITATION.cff` has the details.
