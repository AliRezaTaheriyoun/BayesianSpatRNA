import torch
import nabr_svg as nb
from main import QUICK

nb.NABRChain._compute_rho = lambda self, gamma: torch.ones(2, self.p, device=self.dev)
cfg = nb.NABRConfig(**QUICK)
cfg.n_chains = 1
cfg.combine = "genewise"
nb.fit("sim_out", "fit_floor", cfg)