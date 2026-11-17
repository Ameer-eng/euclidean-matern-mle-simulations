# Reproducing the simulation results (frozen 2026-11-16)

Everything runs from the repository root. The per-fit outputs are not in the repository; they are in the inputs archive described in README §7.

## 0. Installation

```bash
module load python/3.12.4                       # or any CPython 3.12.4
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt  # numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, numba 0.68.1
.venv/bin/pip install --target repro/pylib matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 \
    fonttools==4.68.0 kiwisolver==1.5.1 pillow==13.0.0 pyparsing==3.3.3 packaging==26.3
# inputs: download matern_inputs_2026-11-16.tar.gz, inputs_SHA256SUMS.txt.gz and
# SHA256SUMS from the data archive (README §7) into the repository root, then
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf matern_inputs_2026-11-16.tar.gz
zcat inputs_SHA256SUMS.txt.gz | sha256sum -c --quiet
```

`slurm/env.sh` loads the module and `.venv` and pins the thread counts. Off Longleaf, delete its two `module` lines and use an equivalent Python 3.12.4.
The inputs archive restores `results/{fits,kappa,hess0,theory}`, `results/kappa/fit_hosts.csv` and `audit/out/score0{,_local}`. It was built by `repro/make_inputs_archive.sh`.

## A. Regenerate tables and figures from the saved outputs (no fitting)

```bash
repro/regenerate.sh repro/regen          # about 10-25 min, 2-4 cores, < 4 GB
cat repro/regen/comparison.txt
```

- **Environment.** `slurm/env.sh` loads `python/3.12.4` and `.venv`, whose versions are in `requirements.lock.txt`: numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, numba 0.68.1. matplotlib 3.11.2 and its dependencies are in `repro/pylib`; they are used for the figures only. The processor family and thread count of every fit are in `results/kappa/fit_hosts.csv` (inputs archive); those of every post-fit pass are summarized by `audit/postfit_processor_audit.py` (§C).
- **Inputs.** The script reads only the post-refit saved outputs, `results/{fits,kappa,hess0,theory}`, through symlinks. They are packed in `matern_inputs_2026-11-16.tar.gz` (268,863 files, 85 MB compressed, about 1.4 GB unpacked); its SHA-256 is in `SHA256SUMS` and the per-file checksums are in `inputs_SHA256SUMS.txt.gz`.
- **Pre-refit guard.** `repro/guard.py` records every data file opened and fails if a path contains `prerefit`. The opened files are listed in `OUT/opened_files.tsv`.
- **Determinism.** The summaries are deterministic, and the figure bootstrap uses fixed seeds. The CSVs and `report.txt` are expected to match the frozen copies in `results/summary/` byte for byte. `replicates.csv` is compared only if its frozen copy (data archive) has been placed in `results/summary/`; otherwise it is reported as skipped. The PDFs are expected to match except for their creation and modification dates; `SOURCE_DATE_EPOCH` is fixed, and the comparison removes those dates.
- **Figure E1.** `plots.py --only se` writes the combined `matern_mle_se.pdf` and the per-dimension `matern_mle_se_d{3,4,5}.pdf`. `regenerate.sh` then runs `audit/appendixE_fig_e1_by_dim.py`, which asserts that all of them plot identical arrays, equal to the arrays frozen from the combined figure before its row-3 label was changed to "invalid variance (%)" on 2026-11-16.
- **Audits.** The audit scripts behind individual manuscript numbers are listed in `repro/MANIFEST.md`, keyed to the paper's labels.

## B. Regenerate data and rerun fits

This part is optional and expensive. The main fit campaign alone took roughly 4,500–5,000 core-hours (README §4); the κ̃ and Hessian-at-θ0 passes come on top. Section 3 of the README gives the full order: theory, fits, restart rule, κ̃, Hessian at θ0, then summaries.

**Data.** Every data set is a function of its seed tuple:
- `SeedSequence([20261105, d, round(10ν0), G, rep, 0])` for the field;
- `SeedSequence([20261105, d, round(10ν0), G, rep, 1, law_id])` for the errors;
- PCG64 generator.

**Bitwise reproduction** of a data set, and hence of its fit, requires the same:
1. software versions (`requirements.lock.txt`, Python 3.12.4);
2. **CPU family**, one of `amd`, `amd_avx512`, `intel`, `intel_avx512`;
3. **thread count**: 1, 4 or 8, from `OPENBLAS_NUM_THREADS = NUMBA_NUM_THREADS = SLURM_CPUS_PER_TASK`.

Items 2 and 3 matter because the eigendecomposition used to generate the field (`numpy.linalg.eigh` through OpenBLAS) chooses its kernels by CPU family and blocks by thread count.

The family and thread count of every fit are recorded in `results/kappa/fit_hosts.csv`:

| family | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| amd | 11,200 | 4,709 | 1,691 |
| amd_avx512 | 17,600 | 13,420 | 10,433 |
| intel | — | 5,738 | 2,040 |
| intel_avx512 | — | 369 | — |

On Longleaf, select a family with `sbatch --constraint=amd`, `"amd&avx512"`, `intel` or `"intel&avx512"`, and set `--cpus-per-task` to the recorded count. `kappa_task` and `hess0_task` take `--family F --cpus C` and handle only the fits made on F with C threads. `kappa_task` refuses to run on a node of another family. `refit_task` selects fits by `--family`/`--cpus`, and records a data mismatch so that the fit is retried on a node of a different family.

**On a different family or thread count**, regenerated data usually agree with the stored data to rounding. They can, however, give a *different realization of the same law*. This happened for 2 of the 143 refit data sets: both were made on amd_avx512 with 1 thread and regenerated on an amd node, and the relative criterion difference was 2.4e−3. The likely cause is that eigenvectors are determined only up to sign, or up to rotation within eigenspaces, and different kernels can return different choices. Such a data set is equally valid but is not the stored one. Every task that reuses data therefore checks the stored criterion value; `kappa_task` uses a relative tolerance of 1e−8.

**Fits.** `make_manifest fit` followed by `slurm/submit.sh` (README §3, steps 3–7). The production settings are in `matern_sim/config.py`:
- BOX: η ∈ [−15, 40], ν ∈ [0.1, 4], l ∈ [0.02, 2], τ² ∈ [1e−3, 10];
- L-BFGS-B: ftol 1e−15, gtol 1e−6, maxiter 2000, maxfun 4000, maxls 50;
- HESS_STEP = 1e−4;
- ROOT_SEED = 20261105.

**Restart rule**: `python -m matern_sim.refit_task` (README step 4b). The originals are kept in `results/{fits,kappa}_prerefit`.

## C. Processor provenance of the calculations that regenerate data

Two kinds of later calculation regenerate a replicate's data from its seeds. They are documented separately (`audit/postfit_processor_audit.py` → `audit/out/postfit_processor.txt`, per file in `postfit_processor.csv.gz`; it reads the inputs archive).

**Matched-processor post-fit passes**, which feed the manuscript's summaries. Each one checks the regenerated data against the stored fit before computing anything:

| Pass | Files | Same family and threads as the fit | Data check (relative), max |
|---|---|---|---|
| κ̃ (`kappa_task`, `results/kappa`) | 67,200 | 67,135; the other 65 (44 on amd with 1 thread for amd_avx512 fits, 21 with the processor not recorded) passed the check | 1.1e−10 (those 65: 4.7e−12) |
| Hessian at θ0 (`hess0_task`, `results/hess0`) | 67,200 | 67,200 | 4.4e−10 |
| score at θ0, Slurm (`audit/out/score0/ok`) | 9,561 | 9,561 | 0 |
| score at θ0, AMD login node, 8 threads (`audit/out/score0_local/ok`) | 57,639 | 1,691; the others passed the check on a different family or thread count | 1.0e−9 |

The 9,561 data sets whose local regeneration failed the check are recorded in `audit/out/score0_local/mismatch/*.json` and were recomputed on matched Slurm nodes (`score0/ok`). Together the two `ok` sets cover all 67,200 fits once.

**Initialization audit** (`audit/multistart.py` → `audit/out/ms_local/`, 152 data sets × 6 starts). It ran on the AMD login node whatever the fit's family, and refits from scratch. All 152 regenerated data sets passed the 1e−8 relative check against the stored criterion, and 16 differ from it by more than 1e−9 in absolute value (at most 2.0e−4 absolute and 2.5e−9 relative; these have |ℓ| between 2.1×10⁴ and 1.0×10⁵). Its results describe these regenerated data sets; it feeds only the E.2.4 multistart sentence.

## D. Licence and citation

`LICENSE` (MIT, code), `LICENSE-DATA` (CC BY 4.0, outputs and figures) and `CITATION.cff` at the repository root.
