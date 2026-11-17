"""AUDIT (2026-11-09) small validation computations for Q8:
 (1) Monte Carlo moments of the four standardized error laws (noise.sample);
 (2) kernel used by the likelihood (kernel.matern_tables) vs closed forms at nu = 1/2, 3/2,
     and vs the Fourier transform of exp(eta)/(a + 4 pi^2 |w|^2)^(nu + d/2) (d = 3,
     convention exp(2 pi i w.x));
 (3) common latent field across laws and errors generated on the grid then rotated."""
import math
import numpy as np
from scipy.integrate import quad
from matern_sim import config, noise
from matern_sim.kernel import matern_tables, log_A, theta_from_public
from matern_sim.likelihood import GridModel
from matern_sim.kappa_task import field_blocks, data_blocks
from matern_sim.run_task import truth

print("(1) error-law moments, 2e7 draws each (MC SE in brackets); exact from noise.MOMENTS")
rng = np.random.default_rng(12345)
for law in ["gauss", "unif", "mixnorm", "bern"]:
    x = noise.sample(law, rng, 20_000_000)
    m = [x.mean(), (x**2).mean(), (x**3).mean(), (x**4).mean()]
    se4 = (x**4).std() / math.sqrt(len(x))
    print(f"  {law:8s} mean {m[0]:+.4f}  E xi^2 {m[1]:.4f}  E xi^3 {m[2]:+.4f}  E xi^4 {m[3]:.4f} [{se4:.4f}]"
          f"   exact m3,m4 = {noise.MOMENTS[law].get('m3', 0):.4f}, {noise.MOMENTS[law]['m4']:.4f};"
          f"  kappa4(P,t0=0.25) = {noise.kappa4(law, 0.25):+.4f}")

print("\n(2) kernel checks at sigma2=1, l=0.4 (a=l^-2)")
G, Dmax = 10, 300
r = np.sqrt(np.arange(Dmax + 1)) / G
for d in (3, 4, 5):
    for nu in (0.5, 1.5):
        th = theta_from_public(d, 1.0, 0.4, nu, 0.25)
        K = matern_tables(th, d, G, Dmax, which=("K",))[0]
        u = r / 0.4
        ref = np.exp(-u) if nu == 0.5 else (1 + u) * np.exp(-u)
        print(f"  d={d} nu={nu}: max |K - closed form| = {np.max(np.abs(K - ref)):.1e}; K(0) = {K[0]:.15f};"
              f" eta = {th[0]:.6f} = log A_d + log s2 + nu log a: {log_A(d, nu) + nu * math.log(0.4**-2):.6f}")
d = 3
for nu in (0.5, 1.5):
    th = theta_from_public(d, 1.0, 0.4, nu, 0.25); eta, a = th[0], math.exp(th[2])
    f = lambda w: math.exp(eta) / (a + 4 * math.pi**2 * w * w) ** (nu + d / 2)
    errs = []
    for rr in (0.05, 0.1, 0.3, 0.7):
        # radial FT in d=3: K(r) = (2/r) int_0^inf f(w) w sin(2 pi w r) dw
        val = 2 / rr * quad(lambda w: f(w) * w, 0, np.inf, weight="sin", wvar=2 * math.pi * rr, limit=500)[0]
        Kc = matern_tables(th, d, 1, int(round((rr) ** 2 * 1e4)), which=("K",))[0]
        u = rr / 0.4
        ref = math.exp(-u) if nu == 0.5 else (1 + u) * math.exp(-u)
        errs.append(abs(val - ref))
    print(f"  d=3 nu={nu}: max |FT of exp(eta)/(a+4pi^2|w|^2)^(nu+3/2) - K| over r in (.05,.1,.3,.7) = {max(errs):.1e}")

print("\n(3) common field across laws; errors on the grid then rotated (d=3, G=5, nu0=0.5, rep 7)")
d, nu0, G, rep = 3, 0.5, 5, 7
m = GridModel(d, G); th0 = truth(d, nu0); eig = m.signal_eigen(th0)
Zf = field_blocks(eig, d, nu0, G, rep)
tau0 = math.sqrt(config.TRUTH["tau2"])
ys = {}
for law in ["gauss", "unif", "mixnorm", "bern"]:
    Z = data_blocks(m, Zf, d, nu0, G, rep, law)
    ss = [config.ROOT_SEED, d, int(round(10 * nu0)), G, rep]
    rng_e = np.random.Generator(np.random.PCG64(np.random.SeedSequence(ss + [1, noise.LAW_ID[law]])))
    eps = tau0 * noise.sample(law, rng_e, m.n)
    y = m.sym.inverse(Z)
    ys[law] = (y, eps)
    # Z must equal Q^T(Q Zf + eps) block by block
    Ze = m.sym.forward(eps)
    print(f"  {law:8s}: max |Z - (Zf + Q^T eps)| = {max(np.max(np.abs(Z[k] - Zf[k] - Ze[k])) for k in Z):.1e}")
F = {law: y - eps for law, (y, eps) in ys.items()}
print("  max |field(law) - field(gauss)| on the grid:",
      max(np.max(np.abs(F[l] - F["gauss"])) for l in F))
x = ys["bern"][1] / tau0
print(f"  bern errors on the grid take {len(np.unique(np.round(x, 12)))} distinct values (iid two-point law);"
      f" rotated components take {len(np.unique(np.round(np.concatenate([v.ravel() for v in m.sym.forward(x).values()]), 12)))}")
