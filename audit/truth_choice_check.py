"""AUDIT: App. D.2.2 ratios of the information SD of tau2_hat to sqrt(2) tau2 n^-1/2,
d=3, nu0=1/2, n=32768 (G=32), sigma2=1, for the candidate (l, tau2) values."""
import math
import numpy as np
from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel
m = GridModel(3, 32)
for l, t2 in [(0.1, 0.01), (0.2, 0.01), (0.2, 0.1), (0.4, 0.1), (0.4, 0.5), (0.4, 0.25)]:
    th = theta_from_public(3, 1.0, l, 0.5, t2)
    I = m.fisher(th)
    full = math.sqrt(m.n * np.linalg.inv(I)[3, 3] / 2)
    blk = math.sqrt(m.n * np.linalg.inv(I[np.ix_([0, 1, 3], [0, 1, 3])])[2, 2] / 2)
    print(f"(l, tau2) = ({l}, {t2}): ratio full-info {full:.2f}, block-info {blk:.2f}")
