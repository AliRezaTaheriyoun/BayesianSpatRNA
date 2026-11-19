#!/usr/bin/env python3
"""
main.py -- simulate -> fit NABR -> plot.

Requirements: numpy, scipy, pandas, networkx, matplotlib, torch (CUDA GPU strongly recommended);
anndata optional (--h5ad).

Examples
  # smoke test end-to-end (small, fast; works on CPU)
  python main.py --stage all --quick --sim-bins 20000 --sim-genes 4000

  # full simulation only (~101k bins x 18k genes; a few minutes, ~3-4 GB RAM)
  python main.py --stage simulate

  # quick fit on the full simulation, then plots
  python main.py --stage fit --quick
  python main.py --stage plot

  # paper settings (5 kernels x 5 chains x 2000 iterations; GPU, hours)
  python main.py --stage fit
  # reproduce the paper's shared alpha_k exactly
  python main.py --stage fit --alpha-mode shared --fit-dir fit_shared
"""
from __future__ import annotations

import argparse

from nabr_svg import KERNELS, NABRConfig, fit
from plots import make_all_plots
from sim_data import SimConfig, simulate

QUICK = dict(agg_factor=4, n_hvg=1500, n_iter=400, burn_in=200, thin=5, n_chains=2, verbose_every=50)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["simulate", "fit", "plot", "all"], default="all")
    ap.add_argument("--sim-dir", default="sim_out")
    ap.add_argument("--fit-dir", default="fit_out")
    ap.add_argument("--quick", action="store_true", help="small/fast fit settings for debugging")
    # simulation
    ap.add_argument("--sim-bins", type=int, default=101_000)
    ap.add_argument("--sim-genes", type=int, default=18_000)
    ap.add_argument("--sim-seed", type=int, default=2026)
    ap.add_argument("--h5ad", action="store_true")
    # fit (None = keep config / quick default)
    ap.add_argument("--device", default=None)
    ap.add_argument("--alpha-mode", choices=["gene", "shared", "none"], default=None)
    ap.add_argument("--analysis-unit", choices=["aggregate", "subsample", "bin"], default=None)
    ap.add_argument("--agg-factor", type=int, default=None)
    ap.add_argument("--n-subsample", type=int, default=None)
    ap.add_argument("--n-hvg", type=int, default=None, help="0 = keep all genes passing detection")
    ap.add_argument("--n-chains", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--burn-in", type=int, default=None)
    ap.add_argument("--thin", type=int, default=None)
    ap.add_argument("--combine", choices=["paper", "genewise"], default=None)
    ap.add_argument("--kernels", default=None, help=f"comma list from {','.join(KERNELS)}")
    ap.add_argument("--fit-seed", type=int, default=None)
    a = ap.parse_args()

    if a.stage in ("simulate", "all"):
        n_mod = max(1, min(260, a.sim_genes // 70))             # keep ~half the genes in modules
        cfg = SimConfig(n_bins=a.sim_bins, n_genes=a.sim_genes, seed=a.sim_seed, n_modules=n_mod,
                        n_informative_modules=max(1, min(25, n_mod // 10)))
        simulate(a.sim_dir, cfg, write_h5ad=a.h5ad)

    if a.stage in ("fit", "all"):
        cfg = NABRConfig(**(QUICK if a.quick else {}))
        over = dict(device=a.device, alpha_mode=a.alpha_mode, analysis_unit=a.analysis_unit,
                    agg_factor=a.agg_factor, n_subsample=a.n_subsample, n_chains=a.n_chains,
                    n_iter=a.n_iter, burn_in=a.burn_in, thin=a.thin, combine=a.combine, seed=a.fit_seed)
        for k, v in over.items():
            if v is not None:
                setattr(cfg, k, v)
        if a.n_hvg is not None:
            cfg.n_hvg = None if a.n_hvg == 0 else a.n_hvg
        if a.kernels:
            cfg.kernels = tuple(k.strip() for k in a.kernels.split(","))
        if cfg.burn_in >= cfg.n_iter:
            raise SystemExit("burn-in must be smaller than n-iter")
        fit(a.sim_dir, a.fit_dir, cfg)

    if a.stage in ("plot", "all"):
        make_all_plots(a.sim_dir, a.fit_dir if a.stage != "simulate" else None)


if __name__ == "__main__":
    main()
