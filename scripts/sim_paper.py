#!/usr/bin/env python3
"""
sim_paper.py -- replicate the BASIC simulation of Wu, Li, Ma, Wu (AOAS 2025), Sec. 3.1 + Supp. S3.1.

Paper design (what is implemented)
  * n = 1,024 spots on a 32 x 32 lattice, p = 5,000 genes, K = 6 cell types.
  * Three regions; compositions w_i ~ Dirichlet(1,1,1,1,1,1) / (3,5,7,9,11,13) / (18,16,14,12,10,8).
  * ZINB model (1):  Y_ij = 0 w.p. pi (dropout, pi = 0.1 or 0.5), else NB(mu_ij, phi = 10)
        log mu_ij = mu0_j + b1_j K(x_i1) + b2_j K(x_i2) + sum_k alpha_k w_ik
        mu0_j ~ N(2, 0.5^2),  alpha_k ~ N(0, 3.5^2)  (ONE alpha_k per cell type, shared by all genes)
        NO library-size term.
  * K(x) in {linear x, exponential exp(-x/2), periodic cos(2 pi x)}.
  * 100 disconnected sub-networks x 50 genes (star: 1 TF + 49 targets; or scale-free).
    First 10 sub-networks are informative: ALL their genes are SV -> 500 SV genes.
  * Star:  TF (b1, b2) = (1, -1); targets linear (0.15, -0.25), periodic (0.2, -0.3), exponential (0.4, -0.5).
  * Scale-free: hub |b| ~ U(1, 1.2); regular |b| ~ U(a*lo, a*hi) with (lo, hi) = linear (0.1, 0.2),
    periodic (0.2, 0.3), exponential (0.4, 0.5); a = 2 (large) or 1 (small); random signs.

NOT specified in the paper (my choices, change if you can read Figure S1 / the code):
  * region layout (default: three vertical bands)           --layout vbands|hbands|diag
  * coordinate scaling before K(.) (default: standardized)   --coord-scale standardized|unit
  * scale-free generator: Barabasi-Albert with m = 1         --ba-m
  * scale-free strength for the main table: large (a = 2)    --strength large|small

Files written (same layout as sim_data.py, so nabr_svg.py and plots.py can read them):
  counts.npz, bins.csv (incl. w_<type> true compositions), genes.csv, edges.csv, meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

# placeholder names for cell types 1..6 (same names as sim_data.py so plots.py colours work)
NAMES = ["Epithelial", "Stroma", "Adipose", "Immune", "Vascular", "SmoothMuscle"]
DIRICHLET = {0: [1, 1, 1, 1, 1, 1], 1: [3, 5, 7, 9, 11, 13], 2: [18, 16, 14, 12, 10, 8]}
STAR_TARGET = {"linear": (0.15, -0.25), "periodic": (0.20, -0.30), "exponential": (0.40, -0.50)}
SF_RANGE = {"linear": (0.1, 0.2), "periodic": (0.2, 0.3), "exponential": (0.4, 0.5)}


@dataclass
class PaperSimConfig:
    side: int = 32
    n_subnets: int = 100
    subnet_size: int = 50
    n_informative: int = 10
    pattern: str = "linear"          # linear | exponential | periodic
    network: str = "star"            # star | scalefree
    dropout: float = 0.1             # 0.1 (low) or 0.5 (high)
    strength: str = "large"          # scale-free only: large (a=2) | small (a=1)
    phi: float = 10.0
    mu0_mean: float = 2.0
    mu0_sd: float = 0.5
    alpha_sd: float = 3.5
    coord_scale: str = "standardized"
    layout: str = "vbands"
    ba_m: int = 1
    eta_max: float = 12.0            # numerical safety cap on log-mean
    seed: int = 2026


def spatial_fn(pattern: str, x: np.ndarray) -> np.ndarray:
    if pattern == "linear":
        return x
    if pattern == "exponential":
        return np.exp(-x / 2.0)
    if pattern == "periodic":
        return np.cos(2 * np.pi * x)
    raise ValueError(pattern)


def make_regions(side: int, layout: str):
    r, c = np.divmod(np.arange(side * side), side)
    if layout == "vbands":
        reg = (c * 3) // side
    elif layout == "hbands":
        reg = (r * 3) // side
    elif layout == "diag":
        reg = np.minimum(((r + c) * 3) // (2 * side - 1), 2)
    else:
        raise ValueError(layout)
    return r, c, reg.astype(int)


def make_network(cfg: PaperSimConfig, rng):
    size, p = cfg.subnet_size, cfg.n_subnets * cfg.subnet_size
    module = np.repeat(np.arange(cfg.n_subnets), size)
    hub = np.zeros(p, bool)
    edges = []
    for m in range(cfg.n_subnets):
        g = np.arange(m * size, (m + 1) * size)
        if cfg.network == "star":
            h = int(g[0])
            edges += [(h, int(t)) for t in g[1:]]
        else:
            G = nx.barabasi_albert_graph(size, cfg.ba_m, seed=int(rng.integers(2**31 - 1)))
            edges += [(int(g[a]), int(g[b])) for a, b in G.edges()]
            deg = dict(G.degree())
            h = int(g[max(deg, key=deg.get)])
        hub[h] = True
    return edges, module, hub


def make_effects(cfg: PaperSimConfig, rng, module, hub):
    p = len(module)
    b1, b2 = np.zeros(p), np.zeros(p)
    inf = module < cfg.n_informative
    if cfg.network == "star":
        t1, t2 = STAR_TARGET[cfg.pattern]
        b1[inf], b2[inf] = t1, t2
        b1[inf & hub], b2[inf & hub] = 1.0, -1.0
    else:
        a = 2.0 if cfg.strength == "large" else 1.0
        lo, hi = SF_RANGE[cfg.pattern]
        for b in (b1, b2):
            mag = rng.uniform(a * lo, a * hi, p)
            mag[hub] = rng.uniform(1.0, 1.2, int(hub.sum()))
            sign = rng.choice([-1.0, 1.0], p)
            b[inf] = (sign * mag)[inf]
    return b1, b2, inf


def simulate_paper(outdir: str, cfg: PaperSimConfig | None = None) -> None:
    cfg = cfg or PaperSimConfig()
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    n, K = cfg.side ** 2, len(NAMES)

    # ---- spots, regions, compositions
    row, col, reg = make_regions(cfg.side, cfg.layout)
    W = np.zeros((n, K))
    for r_id in range(3):
        idx = np.where(reg == r_id)[0]
        W[idx] = rng.dirichlet(DIRICHLET[r_id], size=len(idx))

    # ---- coordinates entering K(.)
    if cfg.coord_scale == "standardized":
        x1, x2 = (col - col.mean()) / col.std(), (row - row.mean()) / row.std()
    else:
        x1, x2 = col / (cfg.side - 1.0), row / (cfg.side - 1.0)
    K1, K2 = spatial_fn(cfg.pattern, x1), spatial_fn(cfg.pattern, x2)

    # ---- genes, network, effects
    edges, module, hub = make_network(cfg, rng)
    p = len(module)
    b1, b2, inf = make_effects(cfg, rng, module, hub)
    is_sv = (b1 != 0) | (b2 != 0)
    mu0 = rng.normal(cfg.mu0_mean, cfg.mu0_sd, p)
    alpha = rng.normal(0.0, cfg.alpha_sd, K)          # shared by ALL genes (paper's alpha_k)

    # ---- counts from the ZINB model (1); no library-size term
    eta = mu0[None, :] + (W @ alpha)[:, None] + np.outer(K1, b1) + np.outer(K2, b2)
    n_clip = int((eta > cfg.eta_max).sum())
    np.minimum(eta, cfg.eta_max, out=eta)
    mu = np.exp(eta)
    lam = rng.gamma(shape=cfg.phi, scale=mu / cfg.phi)     # Gamma-Poisson = NB(mu, phi)
    Y = rng.poisson(lam).astype(np.int32)
    Y[rng.random(Y.shape) < cfg.dropout] = 0
    X = sp.csr_matrix(Y)
    X.eliminate_zeros()

    # ---- tables
    names = np.array([f"G{j:05d}" for j in range(p)])
    comp = W.argmax(1)
    bins = pd.DataFrame({
        "bin_id": [f"spot_{i:05d}" for i in range(n)], "row": row, "col": col,
        "x_um": col * 100.0, "y_um": row * 100.0,
        "comp_code": comp, "compartment": np.array(NAMES)[comp], "region": reg + 1})
    for k, nm in enumerate(NAMES):
        bins[f"w_{nm}"] = W[:, k]
    bins["true_logdepth"] = 0.0
    bins["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    bins["n_genes"] = np.diff(X.indptr)

    deg = np.bincount(np.array(edges).ravel(), minlength=p)
    genes = pd.DataFrame({
        "gene": names, "mu0": mu0, "dropout": cfg.dropout, "phi": cfg.phi,
        "module": module, "module_kind": cfg.network, "is_hub": hub, "module_noise": False,
        "degree": deg, "is_SV": is_sv, "is_ctSV": False, "ct_type": "",
        "pattern": np.where(is_sv, cfg.pattern, ""), "b1": b1, "b2": b2,
        "marker_of": "", "off_in": "", "off_in_adipose": False,
        "eval_class": np.where(is_sv, "SV:" + cfg.pattern, "null"),
        "total_counts": np.asarray(X.sum(axis=0)).ravel(),
        "detect_frac": X.getnnz(axis=0) / n})
    for k, nm in enumerate(NAMES):
        genes[f"alpha_{nm}"] = alpha[k]
    edges_df = pd.DataFrame({"gene_a": names[[a for a, _ in edges]], "gene_b": names[[b for _, b in edges]]})

    sp.save_npz(os.path.join(outdir, "counts.npz"), X, compressed=False)
    bins.to_csv(os.path.join(outdir, "bins.csv"), index=False)
    genes.to_csv(os.path.join(outdir, "genes.csv"), index=False)
    edges_df.to_csv(os.path.join(outdir, "edges.csv"), index=False)
    meta = dict(H=cfg.side, W=cfg.side, bin_um=100.0, compartments=NAMES, n_bins=n, n_genes=p,
                nnz=int(X.nnz), kind="paper_replication", config=asdict(cfg))
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- sanity report
    per_gene = X.sum(0).A1 / n
    print(f"[sim_paper] {n} spots x {p} genes | pattern={cfg.pattern} network={cfg.network} "
          f"dropout={cfg.dropout} | {int(is_sv.sum())} SV genes in {cfg.n_informative} informative sub-networks")
    print(f"[sim_paper] alpha_k = {np.round(alpha, 2).tolist()}  (shared by all genes)")
    print(f"[sim_paper] mean count per gene per spot: median {np.median(per_gene):.2f}, "
          f"mean {per_gene.mean():.2f}; zeros {1 - X.nnz / (n * p):.1%}; "
          f"log-mean entries capped at {cfg.eta_max}: {n_clip}")
    print(bins.groupby("region")[["total_counts", "n_genes"]].median().to_string())
    print(f"[sim_paper] done in {time.time() - t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="sim_paper")
    ap.add_argument("--pattern", choices=["linear", "exponential", "periodic"], default="linear")
    ap.add_argument("--network", choices=["star", "scalefree"], default="star")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--strength", choices=["large", "small"], default="large")
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--n-subnets", type=int, default=100)
    ap.add_argument("--subnet-size", type=int, default=50)
    ap.add_argument("--n-informative", type=int, default=10)
    ap.add_argument("--layout", choices=["vbands", "hbands", "diag"], default="vbands")
    ap.add_argument("--coord-scale", choices=["standardized", "unit"], default="standardized")
    ap.add_argument("--ba-m", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    simulate_paper(a.outdir, PaperSimConfig(
        side=a.side, n_subnets=a.n_subnets, subnet_size=a.subnet_size, n_informative=a.n_informative,
        pattern=a.pattern, network=a.network, dropout=a.dropout, strength=a.strength,
        layout=a.layout, coord_scale=a.coord_scale, ba_m=a.ba_m, seed=a.seed))
