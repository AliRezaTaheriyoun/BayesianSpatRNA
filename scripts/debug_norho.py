import torch
import nabr_svg as nb
from main import QUICK

# disable adaptive weights: rho = 1
nb.NABRChain._compute_rho = lambda self, gamma: torch.ones(2, self.p, device=self.dev)

cfg = nb.NABRConfig(**QUICK)
cfg.kernels = ("linear",)
cfg.n_chains = 1
nb.fit("sim_out", "fit_norho", cfg)