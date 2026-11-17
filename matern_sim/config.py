"""Simulation design (edit here; everything else reads from this file)."""

# True parameter (Section 4.1): sigma^2 = 1, l = 0.4, tau^2 = 0.25.
TRUTH = dict(sigma2=1.0, ell=0.4, tau2=0.25)
NU0 = (0.5, 1.5)

# Monte Carlo grids: G points per axis, n = G^d (Section 4.1).
GRIDS = {
    3: (5, 7, 10, 14, 22, 32, 40, 46),
    4: (3, 4, 6, 8, 10, 13, 18),
    5: (3, 4, 5, 6, 8, 10),
}
# Grids where only I_n(theta0) and the kurtosis matrix are computed
# (benchmark curves beyond the Monte Carlo range; grey regions of the figures).
THEORY_GRIDS = {
    3: (5, 7, 10, 14, 18, 22, 26, 32, 36, 40, 46),
    4: (3, 4, 6, 8, 10, 13, 16, 18, 20, 22),
    5: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
}

# Error laws (see noise.py) and replicates per (d, nu0, G, law).
LAWS = ("gauss", "unif", "mixnorm", "bern")
NREP = 400

# Optimisation box Theta (Section 4.1), in public units.
BOX = dict(eta=(-15.0, 40.0), nu=(0.1, 4.0), ell=(0.02, 2.0), tau2=(1e-3, 10.0))

# L-BFGS-B settings (Appendix D.3).
LBFGS = dict(ftol=1e-15, gtol=1e-6, maxiter=2000, maxfun=4000, maxls=50)

# Step for the finite-difference observed Hessian, in the optimiser's
# Fisher-scaled coordinates (one unit ~ one asymptotic SD).
HESS_STEP = 1e-4

# Seed root.  Replicate r of (d, nu0, G) draws its latent field from
# SeedSequence((ROOT_SEED, d, round(10 nu0), G, r, 0)) -- shared by all laws
# (common random numbers) -- and its errors from
# SeedSequence((ROOT_SEED, d, round(10 nu0), G, r, 1, law_id)).
ROOT_SEED = 20261105
