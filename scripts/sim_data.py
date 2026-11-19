#!/usr/bin/env python3
"""
sim_data.py -- synthetic Visium-HD-like spatial transcriptome (16 um x 16 um bins)
with known ground-truth spatially variable (SV) genes.

What is simulated
  * ~101,000 bins on a 16 um grid inside an irregular tissue outline.
  * 6 compartments ("bin types"): Epithelial, Stroma, Adipose, Immune, Vascular, SmoothMuscle.
    Adipose bins have low RNA content and ~70% of genes switched off -> far fewer detected genes.
  * 18,000 genes from a zero-inflated negative binomial (ZINB) with gene-specific dispersion
    (the NABR model assumes one global dispersion -> mild, realistic misspecification).
  * Gene network: block-wise modules (star = TF->targets, or scale-free) + singletons + random edges.
  * Ground truth classes (column `eval_class` in genes.csv):
      SV:<pattern>        global SV gene, spatial term independent of bin type
                          (linear / gaussian / periodic / hotspot); mostly clustered in modules
      ctSV                spatial pattern only inside one compartment (reported separately)
      marker (null)       compartment marker, NO spatial term -> looks spatial only because the
                          compartments are spatial. A method that adjusts for bin type should NOT call it.
      adipose-off (null)  switched off in adipose (also compartment-driven), null
      null                everything else

Outputs (in outdir):
  counts.npz  CSR (bins x genes, int32)      bins.csv   per-bin metadata
  genes.csv   per-gene truth table           edges.csv  gene network edge list
  meta.json   grid shape, compartments, config
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, gaussian_filter

COMPARTMENTS = ["Epithelial", "Stroma", "Adipose", "Immune", "Vascular", "SmoothMuscle"]

# (hub-gene magnitude range, member-gene magnitude range) on the log-mean scale,
# spatial functions evaluated on standardized coordinates (mean 0, sd 1).
MAGNITUDE = {
    "linear": ((0.50, 0.70), (0.20, 0.45)),
    "gaussian": ((0.90, 1.20), (0.50, 0.90)),
    "periodic": ((0.60, 0.80), (0.30, 0.50)),
    "hotspot": ((1.20, 1.60), (0.70, 1.10)),
}


@dataclass
class SimConfig:
    n_bins: int = 101_000
    n_genes: int = 18_000
    bin_um: float = 16.0
    seed: int = 2026
    # expression
    base_logmean_mu: float = -3.4          # gives ~1,000-1,500 UMIs / non-adipose bin
    base_logmean_sd: float = 1.3
    phi: float = 10.0                      # NB size (variance = mu + mu^2/phi)
    phi_gene_logsd: float = 0.3
    dropout_a: float = 2.0                 # per-gene dropout prob ~ Beta(a, b), mean 0.2
    dropout_b: float = 8.0
    depth_logsd: float = 0.35
    technical_depth_gradient: float = 0.3  # smooth non-biological depth trend (size factors fix it)
    depth_by_type: dict = field(default_factory=lambda: {
        "Epithelial": 1.3, "Stroma": 1.0, "Adipose": 0.25,
        "Immune": 0.9, "Vascular": 0.8, "SmoothMuscle": 0.9})
    off_frac_by_type: dict = field(default_factory=lambda: {
        "Epithelial": 0.25, "Stroma": 0.30, "Adipose": 0.70,
        "Immune": 0.35, "Vascular": 0.35, "SmoothMuscle": 0.35})
    n_markers_per_type: int = 150
    marker_effect: tuple = (1.0, 2.5)
    type_noise_sd: float = 0.15
    # network / truth
    n_modules: int = 260
    module_size: tuple = (20, 50)
    n_informative_modules: int = 25
    informative_noise_frac: float = 0.2    # null genes inside informative modules (noisy network)
    n_random_edges: int = 300
    n_singleton_sv: int = 200              # SV genes with no network support
    n_ctsv: int = 100
    pattern_probs: dict = field(default_factory=lambda: {
        "linear": 0.2, "gaussian": 0.25, "periodic": 0.2, "hotspot": 0.35})
    chunk: int = 128                       # genes simulated per chunk (memory control)


# ----------------------------------------------------------------------------- tissue
def _smooth_noise(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    return (f - f.mean()) / (f.std() + 1e-12)


def make_tissue(cfg: SimConfig, rng):
    side = int(np.ceil(np.sqrt(cfg.n_bins / 0.70)))
    H = W = side
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((xx - W / 2) / (0.50 * W)) ** 2 + ((yy - H / 2) / (0.47 * H)) ** 2)
    score = -r + 0.08 * _smooth_noise(rng, (H, W), side / 12)
    keep = np.argpartition(-score.ravel(), cfg.n_bins - 1)[: cfg.n_bins]   # exactly n_bins
    mask = np.zeros(H * W, bool)
    mask[keep] = True
    mask = mask.reshape(H, W)

    c = {name: i for i, name in enumerate(COMPARTMENTS)}
    lab = np.full((H, W), c["Stroma"], np.int16)
    adip = (xx / W + 0.12 * _smooth_noise(rng, (H, W), side / 8)) > 0.64          # fat lobe
    epi = (_smooth_noise(rng, (H, W), side / 30) > 0.9) & ~adip                     # ducts/islands
    imm_f = _smooth_noise(rng, (H, W), side / 70)
    near_epi = binary_dilation(epi, iterations=6)
    immune = (((imm_f > 1.3) & near_epi) | (imm_f > 2.3)) & ~epi & ~adip           # aggregates
    vf = _smooth_noise(rng, (H, W), side / 20)
    vessel = np.abs(vf) < 0.035                                                     # thin curves
    smc = binary_dilation(vessel, iterations=2) & ~vessel & ~adip                  # vessel walls
    lab[adip] = c["Adipose"]
    lab[epi] = c["Epithelial"]
    lab[immune] = c["Immune"]
    lab[smc] = c["SmoothMuscle"]
    lab[vessel] = c["Vascular"]

    rows, cols = np.nonzero(mask)
    tech = _smooth_noise(rng, (H, W), side / 6)[rows, cols]
    bins = pd.DataFrame({
        "bin_id": [f"bin_{i:06d}" for i in range(len(rows))],
        "row": rows, "col": cols,
        "x_um": cols * cfg.bin_um, "y_um": rows * cfg.bin_um,
        "comp_code": lab[rows, cols].astype(int)})
    bins["compartment"] = np.array(COMPARTMENTS)[bins["comp_code"].to_numpy()]
    return bins, tech, (H, W)


# ----------------------------------------------------------------------------- spatial programs
def _draw_pattern(rng, cfg) -> str:
    names = list(cfg.pattern_probs)
    pr = np.array([cfg.pattern_probs[k] for k in names], float)
    return str(names[rng.choice(len(names), p=pr / pr.sum())])


def _shape_params(rng, pattern: str) -> dict:
    """Shape shared by all genes in one module (same spatial program)."""
    use_b2 = bool(rng.random() > 0.3)
    if pattern == "linear":
        return {"use_b2": use_b2}
    if pattern == "gaussian":
        return {"use_b2": use_b2, "l": rng.choice([0.4, 0.7, 1.0]),
                "c1": rng.uniform(-0.6, 0.6), "c2": rng.uniform(-0.6, 0.6)}
    if pattern == "periodic":
        ph = 0.0 if rng.random() < 0.5 else rng.uniform(0, 2 * np.pi)
        return {"use_b2": use_b2, "l": rng.choice([0.8, 1.2, 1.6]), "ph1": ph, "ph2": ph}
    if pattern == "hotspot":
        return {"use_b2": False, "l": rng.choice([0.25, 0.4]),
                "c1": rng.uniform(-1.0, 1.0), "c2": rng.uniform(-1.0, 1.0)}
    raise ValueError(pattern)


def _gene_spec(rng, pattern, shape, hub, s1, s2) -> dict:
    lo, hi = MAGNITUDE[pattern][0 if hub else 1]
    b1 = s1 * rng.uniform(lo, hi)
    b2 = s2 * rng.uniform(lo, hi) if shape.get("use_b2", False) else 0.0
    spec = dict(pattern=pattern, b1=float(b1), b2=float(b2))
    spec.update({k: float(v) for k, v in shape.items() if k != "use_b2"})
    return spec


def spatial_effect(spec: dict, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    pat, b1, b2 = spec["pattern"], spec["b1"], spec["b2"]
    if pat == "linear":
        f = b1 * sx + b2 * sy
    elif pat == "gaussian":
        l2 = 2 * spec["l"] ** 2
        f = b1 * np.exp(-(sx - spec["c1"]) ** 2 / l2) + b2 * np.exp(-(sy - spec["c2"]) ** 2 / l2)
    elif pat == "periodic":
        f = (b1 * np.cos(2 * np.pi * sx / spec["l"] + spec["ph1"])
             + b2 * np.cos(2 * np.pi * sy / spec["l"] + spec["ph2"]))
    elif pat == "hotspot":
        f = b1 * np.exp(-((sx - spec["c1"]) ** 2 + (sy - spec["c2"]) ** 2) / (2 * spec["l"] ** 2))
    else:
        raise ValueError(pat)
    return f - f.mean()


# ----------------------------------------------------------------------------- genes + network
def make_genes_and_network(cfg: SimConfig, rng):
    p, K = cfg.n_genes, len(COMPARTMENTS)
    perm = rng.permutation(p)
    edges = set()
    module = np.full(p, -1)
    kind = np.array([""] * p, dtype=object)
    is_hub = np.zeros(p, bool)
    modules, pos = [], 0
    for m in range(cfg.n_modules):
        size = int(rng.integers(cfg.module_size[0], cfg.module_size[1] + 1))
        if pos + size > p:
            break
        g = perm[pos:pos + size]
        pos += size
        if rng.random() < 0.5:
            k, hub = "star", int(g[0])
            for t in g[1:]:
                edges.add((min(hub, int(t)), max(hub, int(t))))
        else:
            k = "scalefree"
            G = nx.barabasi_albert_graph(size, int(rng.integers(1, 3)), seed=int(rng.integers(2**31 - 1)))
            for a, b in G.edges():
                ga, gb = int(g[a]), int(g[b])
                edges.add((min(ga, gb), max(ga, gb)))
            deg = dict(G.degree())
            hub = int(g[max(deg, key=deg.get)])
        module[g] = m
        kind[g] = k
        is_hub[hub] = True
        modules.append(dict(id=m, kind=k, genes=g, hub=hub))
    singletons = perm[pos:]
    for _ in range(cfg.n_random_edges):
        a, b = (int(v) for v in rng.choice(p, 2, replace=False))
        edges.add((min(a, b), max(a, b)))

    # ---- global SV genes: informative modules + singletons
    spec, ct_spec = {}, {}
    module_noise = np.zeros(p, bool)
    n_inf = min(cfg.n_informative_modules, len(modules))
    for mi in rng.choice(len(modules), n_inf, replace=False):
        M = modules[mi]
        pat = _draw_pattern(rng, cfg)
        shape = _shape_params(rng, pat)
        s1m, s2m = rng.choice([-1, 1]), rng.choice([-1, 1])
        for gidx in M["genes"]:
            gidx = int(gidx)
            hub = gidx == M["hub"]
            if not hub and rng.random() < cfg.informative_noise_frac:
                module_noise[gidx] = True
                continue
            if M["kind"] == "star":          # TF and targets move together
                s1, s2 = s1m, s2m
            else:                            # scale-free: mixed directions
                s1, s2 = rng.choice([-1, 1]), rng.choice([-1, 1])
            spec[gidx] = _gene_spec(rng, pat, shape, hub, s1, s2)
    free = rng.permutation(singletons)
    n_ssv = min(cfg.n_singleton_sv, len(free))
    for gidx in free[:n_ssv]:
        pat = _draw_pattern(rng, cfg)
        spec[int(gidx)] = _gene_spec(rng, pat, _shape_params(rng, pat), rng.random() < 0.5,
                                     rng.choice([-1, 1]), rng.choice([-1, 1]))
    # ---- cell-type-specific SV genes (only in large compartments)
    ct_types = [COMPARTMENTS.index(n) for n in ("Epithelial", "Stroma", "Adipose")]
    n_ct = min(cfg.n_ctsv, len(free) - n_ssv)
    for gidx in free[n_ssv:n_ssv + n_ct]:
        pat = str(rng.choice(["linear", "hotspot"]))
        ct_spec[int(gidx)] = (int(rng.choice(ct_types)),
                              _gene_spec(rng, pat, _shape_params(rng, pat), True,
                                         rng.choice([-1, 1]), rng.choice([-1, 1])))

    # ---- expression parameters
    mu0 = rng.normal(cfg.base_logmean_mu, cfg.base_logmean_sd, p)
    alpha = rng.normal(0, cfg.type_noise_sd, (p, K))
    marker_of = np.array([""] * p, dtype=object)
    for k, name in enumerate(COMPARTMENTS):
        mk = rng.choice(p, cfg.n_markers_per_type, replace=False)
        alpha[mk, k] += rng.uniform(*cfg.marker_effect, size=len(mk))
        for g in mk:
            marker_of[g] = (marker_of[g] + ";" + name).strip(";")
    off_in = np.array([""] * p, dtype=object)
    for k, name in enumerate(COMPARTMENTS):
        cand = np.where(alpha[:, k] < 0.5)[0]                    # never switch off own markers
        n_off = min(int(cfg.off_frac_by_type[name] * p), len(cand))
        off = rng.choice(cand, n_off, replace=False)
        alpha[off, k] = -30.0                                     # effectively zero expression
        for g in off:
            off_in[g] = (off_in[g] + ";" + name).strip(";")
    dropout = rng.beta(cfg.dropout_a, cfg.dropout_b, p)
    phi = cfg.phi * np.exp(rng.normal(0, cfg.phi_gene_logsd, p))

    # ---- truth table
    pattern = np.array([""] * p, dtype=object)
    b1 = np.zeros(p)
    b2 = np.zeros(p)
    is_sv = np.zeros(p, bool)
    is_ct = np.zeros(p, bool)
    ct_type = np.array([""] * p, dtype=object)
    for j, s in spec.items():
        is_sv[j], pattern[j], b1[j], b2[j] = True, s["pattern"], s["b1"], s["b2"]
    for j, (k, s) in ct_spec.items():
        is_ct[j], ct_type[j], pattern[j], b1[j], b2[j] = True, COMPARTMENTS[k], s["pattern"], s["b1"], s["b2"]
    edges_arr = np.array(sorted(edges), dtype=int).reshape(-1, 2)
    degree = np.bincount(edges_arr.ravel(), minlength=p)
    is_marker = np.array([m != "" for m in marker_of])
    off_adip = np.array(["Adipose" in s for s in off_in])
    eval_class = np.where(is_sv, np.char.add("SV:", pattern.astype(str)),
                 np.where(is_ct, "ctSV",
                 np.where(is_marker, "marker (null)",
                 np.where(off_adip, "adipose-off (null)", "null"))))
    names = np.array([f"G{j:05d}" for j in range(p)])
    genes = pd.DataFrame({
        "gene": names, "mu0": mu0, "dropout": dropout, "phi": phi,
        "module": module, "module_kind": kind, "is_hub": is_hub, "module_noise": module_noise,
        "degree": degree, "is_SV": is_sv, "is_ctSV": is_ct, "ct_type": ct_type,
        "pattern": pattern, "b1": b1, "b2": b2,
        "marker_of": marker_of, "off_in": off_in, "off_in_adipose": off_adip,
        "eval_class": eval_class})
    for k, name in enumerate(COMPARTMENTS):
        genes[f"alpha_{name}"] = alpha[:, k]
    edges_df = pd.DataFrame({"gene_a": names[edges_arr[:, 0]], "gene_b": names[edges_arr[:, 1]]})
    return genes, edges_df, spec, ct_spec, alpha


# ----------------------------------------------------------------------------- counts
def simulate_counts(cfg: SimConfig, rng, bins, tech, genes, spec, ct_spec, alpha):
    n, p = len(bins), cfg.n_genes
    types = bins["comp_code"].to_numpy()
    depth = np.array([cfg.depth_by_type[c] for c in COMPARTMENTS])
    logdepth = (np.log(depth[types]) + rng.normal(0, cfg.depth_logsd, n)
                + cfg.technical_depth_gradient * tech)
    x, y = bins["x_um"].to_numpy(float), bins["y_um"].to_numpy(float)
    sx, sy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    mu0 = genes["mu0"].to_numpy()
    pi = genes["dropout"].to_numpy()
    phi = genes["phi"].to_numpy()

    blocks, t0 = [], time.time()
    for ci, start in enumerate(range(0, p, cfg.chunk)):
        stop = min(p, start + cfg.chunk)
        c = stop - start
        eta = logdepth[:, None] + mu0[None, start:stop] + alpha[start:stop][:, types].T
        for jj in range(c):
            j = start + jj
            if j in spec:
                eta[:, jj] += spatial_effect(spec[j], sx, sy)
            if j in ct_spec:
                k, s = ct_spec[j]
                inside = types == k
                eta[inside, jj] += spatial_effect(s, sx[inside], sy[inside])
        np.minimum(eta, 9.0, out=eta)
        mu = np.exp(eta)
        ph = phi[None, start:stop]
        lam = rng.gamma(shape=ph, scale=mu / ph)            # Gamma-Poisson = NB(mu, phi)
        cnt = rng.poisson(lam).astype(np.int32)
        cnt[rng.random((n, c)) < pi[None, start:stop]] = 0  # zero inflation
        blocks.append(sp.csc_matrix(cnt))
        if (ci + 1) % 20 == 0 or stop == p:
            print(f"  simulated genes {stop:>6d}/{p}  ({time.time() - t0:.0f}s)", flush=True)
    X = sp.hstack(blocks, format="csr")
    X.eliminate_zeros()
    return X, logdepth


# ----------------------------------------------------------------------------- driver
def simulate(outdir: str, cfg: SimConfig | None = None, write_h5ad: bool = False) -> None:
    cfg = cfg or SimConfig()
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    print(f"[sim] tissue with {cfg.n_bins:,} bins ...")
    bins, tech, (H, W) = make_tissue(cfg, rng)
    print(f"[sim] genes/network ({cfg.n_genes:,} genes) ...")
    genes, edges, spec, ct_spec, alpha = make_genes_and_network(cfg, rng)
    print("[sim] counts ...")
    X, logdepth = simulate_counts(cfg, rng, bins, tech, genes, spec, ct_spec, alpha)

    bins["true_logdepth"] = logdepth
    bins["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    bins["n_genes"] = np.diff(X.indptr)
    genes["total_counts"] = np.asarray(X.sum(axis=0)).ravel()
    genes["detect_frac"] = X.getnnz(axis=0) / X.shape[0]

    sp.save_npz(os.path.join(outdir, "counts.npz"), X, compressed=False)
    bins.to_csv(os.path.join(outdir, "bins.csv"), index=False)
    genes.to_csv(os.path.join(outdir, "genes.csv"), index=False)
    edges.to_csv(os.path.join(outdir, "edges.csv"), index=False)
    meta = dict(H=int(H), W=int(W), bin_um=cfg.bin_um, compartments=COMPARTMENTS,
                n_bins=int(X.shape[0]), n_genes=int(X.shape[1]), nnz=int(X.nnz), config=asdict(cfg))
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    summ = bins.groupby("compartment").agg(n_bins=("bin_id", "size"),
                                           median_UMIs=("total_counts", "median"),
                                           median_genes=("n_genes", "median"))
    print(summ.to_string())
    print(genes["eval_class"].value_counts().to_string())
    print(f"[sim] nnz={X.nnz:,} (density {X.nnz / (X.shape[0] * X.shape[1]):.3%}); "
          f"{len(edges):,} edges; done in {time.time() - t0:.0f}s -> {outdir}")

    if write_h5ad:
        try:
            import anndata as ad
            adata = ad.AnnData(X=X.astype(np.float32), obs=bins.set_index("bin_id"),
                               var=genes.set_index("gene"))
            adata.obsm["spatial"] = bins[["x_um", "y_um"]].to_numpy()
            adata.write_h5ad(os.path.join(outdir, "sim.h5ad"))
            print("[sim] wrote sim.h5ad")
        except Exception as exc:  # anndata optional
            print(f"[sim] h5ad not written: {exc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="sim_out")
    ap.add_argument("--n-bins", type=int, default=101_000)
    ap.add_argument("--n-genes", type=int, default=18_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--h5ad", action="store_true")
    a = ap.parse_args()
    simulate(a.outdir, SimConfig(n_bins=a.n_bins, n_genes=a.n_genes, seed=a.seed), a.h5ad)
