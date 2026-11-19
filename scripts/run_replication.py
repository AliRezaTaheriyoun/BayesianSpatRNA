#!/usr/bin/env python3
"""
run_replication.py -- fit nabr_svg on the paper-replication data with paper-faithful settings.

It does NOT edit nabr_svg.py; it monkeypatches three things at runtime:
  1. build_units : use the TRUE Dirichlet compositions w_ik (bins.csv: w_<type>) as the model's w_i,
                   and NO library-size term (the paper's data-generating model has none).
  2. kernel_basis: the paper's printed exponential kernel exp(-x / (2 l^2)) (monotone) instead of the
                   symmetric Gaussian bump used in nabr_svg.py. (names gauss40/gauss60 = exponential here)
  3. (optional --no-rho) switch off the adaptive weights rho.
Settings: alpha_mode = "shared" (paper's alpha_k), all genes kept, lambda lower bound 0, rho ON.

BEFORE RUNNING: in nabr_svg.py, update_lambda must read
        lo, hi, tau, lam = 0.0, self.lam_u, self.tau_lam, self.lam
(revert the 0.5 * self.lam_u floor from the debug_floor experiment). This script checks it.

Examples
  python sim_paper.py --outdir sim_paper_small --n-subnets 20 --n-informative 2 --side 20
  python run_replication.py --sim-dir sim_paper_small --fit-dir fit_paper_small --kernels linear \
         --n-iter 200 --burn-in 100 --thin 5

  python sim_paper.py --outdir sim_paper
  python run_replication.py --sim-dir sim_paper --fit-dir fit_paper --kernels linear \
         --n-chains 1 --n-iter 1000 --burn-in 500 --thin 5
"""
from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import nabr_svg as nb

# (pattern, network, dropout) -> (recall, precision, F1) of the PROPOSED method, paper Table 1 (dropout 0.1)
# and Supplementary Table S3 (dropout 0.5); mean over 50 replicates.
PAPER = {
    ("linear", "star", 0.1): (0.992, 1.000, 0.996), ("linear", "scalefree", 0.1): (0.991, 1.000, 0.995),
    ("exponential", "star", 0.1): (0.933, 1.000, 0.959), ("exponential", "scalefree", 0.1): (0.979, 1.000, 0.989),
    ("periodic", "star", 0.1): (0.996, 1.000, 0.998), ("periodic", "scalefree", 0.1): (0.979, 1.000, 0.981),
    ("linear", "star", 0.5): (0.975, 1.000, 0.985), ("linear", "scalefree", 0.5): (0.965, 1.000, 0.978),
    ("exponential", "star", 0.5): (0.873, 1.000, 0.913), ("exponential", "scalefree", 0.5): (0.937, 1.000, 0.958),
    ("periodic", "star", 0.5): (0.983, 1.000, 0.988), ("periodic", "scalefree", 0.5): (0.977, 1.000, 0.980),
}


def build_units_given(X, bins, cfg, rng, comp_names):
    """Use the true compositions; constant library size -> log c_i = 0 (paper's model has no c_i)."""
    W = bins[[f"w_{c}" for c in comp_names]].to_numpy(float)
    n = X.shape[0]
    return dict(Y=X.tocsr(), W=W, xy=bins[["x_um", "y_um"]].to_numpy(float),
                lib=np.ones(n), n_bins_per_unit=np.ones(n))


def kernel_basis_paper(xy: np.ndarray, name: str) -> np.ndarray:
    """Paper's kernels as printed: linear, exp(-x/(2 l^2)) x2, cos(2 pi x / l) x2; l = 40%/60% quantile of |x|."""
    s = (xy - xy.mean(0)) / xy.std(0)
    a = np.abs(s)
    q40, q60 = np.quantile(a, 0.4, axis=0), np.quantile(a, 0.6, axis=0)
    if name == "linear":
        B = s
    elif name == "gauss40":
        B = np.exp(-s / (2 * q40 ** 2))
    elif name == "gauss60":
        B = np.exp(-s / (2 * q60 ** 2))
    elif name == "cos40":
        B = np.cos(2 * np.pi * s / q40)
    elif name == "cos60":
        B = np.cos(2 * np.pi * s / q60)
    else:
        raise ValueError(name)
    B = (B - B.mean(0)) / B.std(0)
    return B.astype(np.float32)


def _auc(score, y):
    n1, n0 = y.sum(), (~y).sum()
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(sim_dir, fit_dir, kernels):
    with open(f"{sim_dir}/meta.json") as fh:
        c = json.load(fh)["config"]
    with open(f"{fit_dir}/fit_summary.json") as fh:
        m = json.load(fh)["metrics"]
    key = (c["pattern"], c["network"], round(float(c["dropout"]), 2))
    print("\n" + "=" * 70)
    print(f"Scenario: pattern={key[0]}, network={key[1]}, dropout={key[2]}")
    print(f"This run : recall={m['recall']:.3f}  precision={m['precision']:.3f}  F1={m['f1']:.3f}  "
          f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    if key in PAPER:
        r, p, f = PAPER[key]
        print(f"Paper    : recall={r:.3f}  precision={p:.3f}  F1={f:.3f}  (mean of 50 replicates, full 5x5 chains)")
    r_ = pd.read_csv(f"{fit_dir}/results.csv", keep_default_na=False)
    sv = r_["is_SV"].astype(str).str.lower().eq("true").to_numpy()
    num = lambda col: pd.to_numeric(r_[col], errors="coerce").to_numpy(float)
    absb = np.max([np.maximum(np.abs(num(f"beta1_{k}")), np.abs(num(f"beta2_{k}"))) for k in kernels], axis=0)
    print(f"Ranking  : AUROC pip = {_auc(num('pip'), sv):.3f} | AUROC max|beta| = {_auc(absb, sv):.3f}")
    o = np.argsort(-absb)
    n_sv = int(sv.sum())
    for k in sorted({max(n_sv // 4, 1), max(n_sv // 2, 1), n_sv}):
        print(f"           top {k:4d} by max|beta|: precision {sv[o[:k]].mean():.2f}")
    print(f"Genes called by PIP/BFDR: {int(r_['SV_called'].astype(str).str.lower().eq('true').sum())} "
          f"(true SV = {n_sv}, analysed = {len(r_)})")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", default="sim_paper")
    ap.add_argument("--fit-dir", default="fit_paper")
    ap.add_argument("--kernels", default="linear,gauss40,gauss60,cos40,cos60")
    ap.add_argument("--n-chains", type=int, default=1)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--combine", choices=["paper", "genewise"], default="paper")
    ap.add_argument("--alpha-mode", choices=["shared", "gene", "none"], default="shared")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-rho", action="store_true", help="switch off adaptive weights (ablation)")
    ap.add_argument("--my-kernels", action="store_true", help="keep nabr_svg.py's Gaussian-bump kernels")
    ap.add_argument("--allow-floor", action="store_true", help="run even if the lambda floor edit is present")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    if "0.5 * self.lam_u" in inspect.getsource(nb.NABRChain.update_lambda) and not a.allow_floor:
        raise SystemExit("update_lambda in nabr_svg.py still has the 0.5 * self.lam_u floor.\n"
                         "Change it back to:  lo, hi, tau, lam = 0.0, self.lam_u, self.tau_lam, self.lam\n"
                         "(or pass --allow-floor to run anyway).")

    nb.build_units = build_units_given
    if not a.my_kernels:
        nb.kernel_basis = kernel_basis_paper
    if a.no_rho:
        import torch
        nb.NABRChain._compute_rho = lambda self, gamma: torch.ones(2, self.p, device=self.dev)

    kernels = tuple(k.strip() for k in a.kernels.split(","))
    cfg = nb.NABRConfig(alpha_mode=a.alpha_mode, n_hvg=None, min_detect_frac=0.0, kernels=kernels,
                        n_chains=a.n_chains, n_iter=a.n_iter, burn_in=a.burn_in, thin=a.thin,
                        combine=a.combine, device=a.device, seed=a.seed, verbose_every=50)
    if cfg.burn_in >= cfg.n_iter:
        raise SystemExit("burn-in must be smaller than n-iter")
    nb.fit(a.sim_dir, a.fit_dir, cfg)
    report(a.sim_dir, a.fit_dir, kernels)

    if not a.no_plots:
        try:
            from plots import make_all_plots
            make_all_plots(a.sim_dir, a.fit_dir)
        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
