#!/usr/bin/env python3
"""
nabr_svg.py -- Network-Assisted Bayesian Regularization (NABR) for spatially variable genes.

Implements
  Wu M., Li Y., Ma S., Wu M. (2025). Joint identification of spatially variable genes via a
  network-assisted Bayesian regularization approach. Ann. Appl. Stat. 19(4):2705-2723.
  arXiv:2407.05241

Model (paper Eq. 1-3)
  Y_ij ~ r_ij * delta_0 + (1 - r_ij) * NB(c_i * mu_ij, phi)                 ZINB, size factor c_i
  log mu_ij = mu0_j + beta1_j K(x_i1) + beta2_j K(x_i2) + sum_k alpha_k w_ik
  beta = gamma o 1(|gamma| > lambda * rho)                                 hard-thresholded Gaussian
  gamma ~ N(0, sigma2 (L + eps I)^-1),  L = (sgn(b~) sgn(b~)^T) o (I - D^-1/2 A D^-1/2)
  sigma2 ~ IG(3.5, 0.5); lambda ~ U(0, q90|b~|); r_ij ~ Bern(pi_j); pi_j ~ Beta(1, 1)
  phi ~ Ga(10, 0.1); mu0_j ~ N(0, 3^2); alpha_k ~ N(0, 3^2)
MCMC (Sec 2.4, steps a-h): Gibbs r; MH mu0, alpha, phi, lambda; pCN-Langevin for gamma;
group-wise rho; Gibbs sigma2. One run per kernel (linear, 2 Gaussian, 2 cosine), combined with
posterior model probabilities, max over the two coordinates, BFDR 0.05; repeated over chains and
genes called in >= 80% of chains are reported.

Documented deviations (all switchable in NABRConfig)
  1. alpha_mode="gene" (default): alpha_jk gene-specific, reference compartment fixed at 0.
     The paper's alpha_k is shared across genes (alpha_mode="shared" reproduces it), which cannot
     absorb gene-specific compartment expression (markers, genes absent from adipose).
  2. Gaussian kernel exp(-x^2/(2 l^2)) (CTSV convention); the arXiv text prints exp(-x/(2 l^2)).
  3. Kernel columns centred and scaled to unit SD (reparameterisation; mu0 absorbs the mean).
  4. gamma: block-wise MH (block = connected sub-network, both coordinates) using the full
     pCN-Langevin ratio (target and proposal densities). The paper prints only the likelihood
     ratio, which is exact for plain pCN but not once the gradient drift is added.
  5. pi_j sampled from its Beta full conditional.
  6. Proposal scales adapt during burn-in only.
  7. Derivative "spike" of the smoothed threshold (paper S1) clipped to +/- grad_spike_cap.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, truncnorm

KERNELS = ("linear", "gauss40", "gauss60", "cos40", "cos60")


@dataclass
class NABRConfig:
    # analysis units
    analysis_unit: str = "aggregate"   # "aggregate" | "subsample" | "bin"
    agg_factor: int = 2                # f x f bins per pseudo-spot (2 -> 32 um)
    n_subsample: int = 30_000
    # gene filtering
    min_detect_frac: float = 0.01
    n_hvg: int | None = 5000           # paper: top 5,000 HVGs
    hvg_subsample: int = 30_000
    # model
    alpha_mode: str = "gene"           # "gene" | "shared" (paper) | "none"
    kernels: tuple = KERNELS
    combine: str = "paper"             # "paper" (global model weights) | "genewise"
    eps_L: float = 1e-3
    eps_smooth: float = 1e-4
    grad_spike_cap: float = 10.0
    # priors (paper S2)
    a_pi: float = 1.0
    b_pi: float = 1.0
    a_phi: float = 10.0
    b_phi: float = 0.1                 # Gamma(shape, rate)
    a_gamma: float = 3.5
    b_gamma: float = 0.5
    sigma_mu0: float = 3.0
    sigma_alpha: float = 3.0
    lambda_quantile: float = 0.9
    # MCMC (paper: 2000 iter, 1000 burn-in, thin 10, 5 chains, 80%)
    n_iter: int = 2000
    burn_in: int = 1000
    thin: int = 10
    n_chains: int = 5
    chain_agreement: float = 0.8
    bfdr: float = 0.05
    target_accept: float = 0.30
    adapt_every: int = 25
    # misc
    device: str = "auto"
    seed: int = 1
    verbose_every: int = 100


# ============================================================================ data preparation
def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_sim(sim_dir):
    X = sp.load_npz(os.path.join(sim_dir, "counts.npz")).tocsr()
    bins = pd.read_csv(os.path.join(sim_dir, "bins.csv"))
    genes = pd.read_csv(os.path.join(sim_dir, "genes.csv"), keep_default_na=False)
    edges = pd.read_csv(os.path.join(sim_dir, "edges.csv"))
    with open(os.path.join(sim_dir, "meta.json")) as fh:
        meta = json.load(fh)
    return X, bins, genes, edges, meta


def select_genes(X, cfg: NABRConfig, rng) -> np.ndarray:
    """Detection filter + simple binned-dispersion HVG selection (paper keeps top 5,000 HVGs)."""
    n, p = X.shape
    det = X.getnnz(axis=0) / n
    keep = det >= cfg.min_detect_frac
    if cfg.n_hvg is None or keep.sum() <= cfg.n_hvg:
        return np.where(keep)[0]
    sub = rng.choice(n, min(n, cfg.hvg_subsample), replace=False)
    Xs = X[sub]
    lib = np.asarray(Xs.sum(axis=1)).ravel().astype(float)
    sf = lib / np.median(lib[lib > 0])
    sf[sf == 0] = 1.0
    Xn = (sp.diags(1.0 / sf) @ Xs).tocsc()
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel() - mean ** 2
    idx = np.where(keep & (mean > 0))[0]
    lm = np.log10(mean[idx])
    disp = np.log10(var[idx] / mean[idx] + 1e-12)
    grp = pd.qcut(lm, 20, labels=False, duplicates="drop")
    z = np.full(len(idx), -np.inf)
    for g in np.unique(grp):
        m = grp == g
        z[m] = (disp[m] - disp[m].mean()) / (disp[m].std() + 1e-12)
    top = idx[np.argsort(-z)[: cfg.n_hvg]]
    return np.sort(top)


def build_units(X, bins, cfg: NABRConfig, rng, comp_names):
    """Analysis units. 'aggregate' turns f x f bins into pseudo-spots with compositions w_ik
    (the paper's setting); 'subsample'/'bin' keep 16 um bins with one-hot w_ik."""
    n_bins, K = X.shape[0], len(comp_names)
    code = pd.Categorical(bins["compartment"], categories=comp_names).codes.astype(int)
    lib_bins = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    x, y = bins["x_um"].to_numpy(float), bins["y_um"].to_numpy(float)
    if cfg.analysis_unit == "aggregate":
        f = int(cfg.agg_factor)
        r, c = bins["row"].to_numpy() // f, bins["col"].to_numpy() // f
        _, unit = np.unique(r * (c.max() + 1) + c, return_inverse=True)
        unit = unit.ravel()
        nU = int(unit.max()) + 1
        M = sp.csr_matrix((np.ones(n_bins), (unit, np.arange(n_bins))), shape=(nU, n_bins))
        Y = (M @ X).tocsr()
        nb = np.asarray(M.sum(axis=1)).ravel()
        Wm = np.zeros((nU, K))
        np.add.at(Wm, (unit, code), 1.0)
        Wm /= nb[:, None]
        xy = np.c_[M @ x, M @ y] / nb[:, None]
        lib = M @ lib_bins
    elif cfg.analysis_unit in ("subsample", "bin"):
        if cfg.analysis_unit == "subsample":
            idx = np.sort(rng.choice(n_bins, min(cfg.n_subsample, n_bins), replace=False))
        else:
            idx = np.arange(n_bins)
        Y, Wm = X[idx], np.eye(K)[code[idx]]
        xy, lib, nb = np.c_[x[idx], y[idx]], lib_bins[idx], np.ones(len(idx))
    else:
        raise ValueError(cfg.analysis_unit)
    return dict(Y=Y, W=Wm, xy=xy, lib=lib, n_bins_per_unit=nb)


def kernel_basis(xy: np.ndarray, name: str) -> np.ndarray:
    """Paper's 5 spatialization functions on standardized coordinates; l = 40%/60% quantiles of |x|."""
    s = (xy - xy.mean(0)) / xy.std(0)
    a = np.abs(s)
    q40, q60 = np.quantile(a, 0.4, axis=0), np.quantile(a, 0.6, axis=0)
    if name == "linear":
        B = s
    elif name == "gauss40":
        B = np.exp(-s ** 2 / (2 * q40 ** 2))
    elif name == "gauss60":
        B = np.exp(-s ** 2 / (2 * q60 ** 2))
    elif name == "cos40":
        B = np.cos(2 * np.pi * s / q40)
    elif name == "cos60":
        B = np.cos(2 * np.pi * s / q60)
    else:
        raise ValueError(name)
    B = (B - B.mean(0)) / B.std(0)          # deviation 3
    return B.astype(np.float32)


def build_network(gene_names, edges: pd.DataFrame):
    idx = pd.Series(np.arange(len(gene_names)), index=gene_names)
    a, b = edges["gene_a"].map(idx), edges["gene_b"].map(idx)
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(int).to_numpy(), b[ok].astype(int).to_numpy()
    a, b = a[a != b], b[a != b]
    p = len(gene_names)
    A = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(p, p)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    d = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.zeros(p)
    dinv[d > 0] = 1.0 / np.sqrt(d[d > 0])
    Ltil = (sp.eye(p) - sp.diags(dinv) @ A @ sp.diags(dinv)).toarray()   # isolated node -> 1
    ncomp, comp = connected_components(A, directed=False)
    return A, Ltil, comp, int(ncomp), d


def build_precisions(Ltil: np.ndarray, beta_tilde: np.ndarray, eps: float, dev):
    """P_d = (s_d s_d^T) o Ltil + eps I for d = 1, 2, with Cholesky factors (block-diagonal)."""
    P, C = [], []
    for d in range(2):
        s = np.where(beta_tilde[d] >= 0, 1.0, -1.0)
        Pd = s[:, None] * Ltil * s[None, :]
        Pd[np.diag_indices_from(Pd)] += eps
        Pt = torch.tensor(Pd, dtype=torch.float64, device=dev)
        C.append(torch.linalg.cholesky(Pt).float())
        P.append(Pt.float())
        del Pd, Pt
    return P, C


@torch.no_grad()
def nb_irls(Y, logc, X, phi: float = 10.0, n_iter: int = 15):
    """Vectorized per-gene NB GLM (fixed phi): the rough estimate beta~ used by the prior."""
    n, p = Y.shape
    q = X.shape[1]
    beta = torch.zeros(q, p, device=Y.device)
    beta[0] = torch.log((Y.sum(0) + 0.5) / torch.exp(logc).sum())
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, q * q)
    I = torch.eye(q, device=Y.device)
    for _ in range(n_iter):
        eta0 = (X @ beta).clamp(-15, 12)
        mu = torch.exp(eta0 + logc[:, None])
        w = mu / (1 + mu / phi)
        wz = w * eta0 + (Y - mu) / (1 + mu / phi)
        XtWX = (XX.T @ w).T.reshape(p, q, q)
        dm = XtWX.diagonal(dim1=1, dim2=2).mean(1)
        XtWX = XtWX + (1e-6 + 1e-4 * dm)[:, None, None] * I
        XtWz = (X.T @ wz).T
        beta = torch.linalg.solve(XtWX, XtWz.unsqueeze(-1)).squeeze(-1).T.clamp(-10, 10)
    return beta


# ============================================================================ MCMC chain
class NABRChain:
    """One MCMC chain for one spatial kernel."""

    @torch.no_grad()
    def __init__(self, data, B, prec, cfg: NABRConfig, init, seed: int):
        self.cfg = cfg
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.Y, self.Yzero, self.ynz = data["Y"], data["Yzero"], data["ynz"]
        self.logc, self.W, self.ref = data["logc"], data["W"], data["ref"]
        self.dev = self.Y.device
        self.n, self.p = self.Y.shape
        self.K = self.W.shape[1]
        self.Wsum = self.W.sum(0).tolist()
        self.B = B
        self.P, self.Lc = prec["P"], prec["chol"]
        self.LcT = [L.T.contiguous() for L in self.Lc]
        self.comp, self.ncomp = prec["comp"], prec["ncomp"]
        self.mode = cfg.alpha_mode
        f32 = dict(device=self.dev, dtype=torch.float32)

        g0 = init["gamma"]
        self.gamma = g0 + 0.05 * g0.abs().mean() * torch.randn(2, self.p, **f32)
        self.mu0 = init["mu0"] + 0.01 * torch.randn(self.p, **f32)
        if self.mode == "gene":
            self.alpha = init["alpha"].clone()
        elif self.mode == "shared":
            self.alpha = torch.zeros(self.K, **f32)
        else:
            self.alpha = None
        self.phi = float(init["phi"])
        self.lam_u = max(float(init["lam_u"]), 1e-6)
        self.lam = 0.5 * self.lam_u
        self.sigma2 = max(float(self._quad(self.gamma).double().sum()) / (2 * self.p), 1e-3)
        self.pi = torch.full((self.p,), 0.1, **f32)
        self.R = torch.zeros_like(self.Yzero)
        self.rho = self._compute_rho(self.gamma)

        # proposal scales (adapted during burn-in)
        self.tau_mu0 = torch.full((self.p,), 0.05, **f32)
        if self.mode == "gene":
            self.tau_alpha = torch.full((self.p, self.K), 0.05, **f32)
        else:
            self.tau_alpha = torch.full((self.K,), 0.002, **f32)
        self.tau_gamma = torch.full((self.ncomp,), 0.05, **f32)
        self.tau_phi = 0.2
        self.tau_lam = 0.05 * self.lam_u

        self._win = self._zero_counters()
        self._tot = self._zero_counters()
        self._tot_n = 0
        # posterior accumulators
        self.M = 0
        self.sum_t = torch.zeros(2, self.p, **f32)
        self.sum_gt = torch.zeros(2, self.p, **f32)
        self.sum_mu0 = torch.zeros(self.p, **f32)
        self.sum_alpha = None if self.alpha is None else torch.zeros_like(self.alpha)
        self.sum_phi = 0.0
        self.sum_R = torch.zeros(self.n, self.p, **f32)
        self.trace = {k: [] for k in ("phi", "lam", "sigma2", "loglik", "n_active")}

    # ---------------------------------------------------------------- helpers
    def _zero_counters(self):
        z = lambda *s: torch.zeros(*s, device=self.dev)
        return {"mu0": z(self.p),
                "alpha": z(self.p, self.K) if self.mode == "gene" else z(self.K),
                "gamma": z(self.ncomp), "phi": 0.0, "lam": 0.0}

    def _count(self, key, val):
        self._win[key] = self._win[key] + val
        self._tot[key] = self._tot[key] + val

    def beta_of(self, gamma, lam=None, rho=None):
        lam = self.lam if lam is None else lam
        rho = self.rho if rho is None else rho
        return gamma * (gamma.abs() > lam * rho).to(gamma.dtype)

    def eta(self, mu0=None, alpha=None, beta=None):
        """log(c_i mu_ij), shape (n, p)."""
        mu0 = self.mu0 if mu0 is None else mu0
        beta = self.beta_of(self.gamma) if beta is None else beta
        e = self.B @ beta
        e += self.logc[:, None]
        e += mu0[None, :]
        if self.mode == "shared":
            e += (self.W @ (self.alpha if alpha is None else alpha))[:, None]
        elif self.mode == "gene":
            e += self.W @ (self.alpha if alpha is None else alpha).T
        return e

    def _lae(self, e, phi=None):
        phi = self.phi if phi is None else phi
        return torch.logaddexp(e, torch.tensor(math.log(phi), device=self.dev, dtype=e.dtype))

    def ll_gene(self, e, phi=None):
        """Per-gene NB log-lik over r_ij = 0 entries, mu-dependent part only (float64)."""
        phi = self.phi if phi is None else phi
        t = self.Y * e - (self.Y + phi) * self._lae(e, phi)
        t.masked_fill_(self.R, 0.0)
        return t.sum(0, dtype=torch.float64)

    def ll_phi_total(self, e, phi):
        """Full NB log-lik (minus lgamma(y+1)) over r_ij = 0 entries, for the phi update."""
        t = self.Y * e + phi * math.log(phi) - (self.Y + phi) * self._lae(e, phi)
        t.masked_fill_(self.R, 0.0)
        out = float(t.sum(dtype=torch.float64))
        out += float((torch.lgamma(self.ynz + phi) - math.lgamma(phi)).sum(dtype=torch.float64))
        return out

    def _quad(self, v):
        """Per-gene contribution to sum_d v_d^T P_d v_d (P block-diagonal)."""
        return v[0] * (self.P[0] @ v[0]) + v[1] * (self.P[1] @ v[1])

    def _comp_sum(self, x):
        return torch.zeros(self.ncomp, dtype=torch.float64, device=self.dev).index_add_(0, self.comp, x.double())

    def _Sigma_mult(self, g):
        out = torch.empty_like(g)
        for d in range(2):
            out[d] = torch.cholesky_solve(g[d][:, None], self.Lc[d]).squeeze(1)
        return self.sigma2 * out

    def _Sigma_half_noise(self):
        z = torch.randn(2, self.p, device=self.dev)
        out = torch.empty_like(z)
        for d in range(2):
            out[d] = torch.linalg.solve_triangular(self.LcT[d], z[d][:, None], upper=True).squeeze(1)
        return math.sqrt(self.sigma2) * out

    def _compute_rho(self, gamma):
        """Step (f): rho_j = 1 / sqrt(max_{l in sub-network(j)} |gamma_l|)."""
        mx = torch.zeros(2, self.ncomp, device=self.dev, dtype=gamma.dtype)
        idx = self.comp.unsqueeze(0).expand(2, -1)
        mx = mx.scatter_reduce(1, idx, gamma.abs(), reduce="amax", include_self=False)
        return 1.0 / torch.sqrt(mx[:, self.comp].clamp_min(1e-8))

    # ---------------------------------------------------------------- steps (a)-(h)
    def update_R(self, e):                                                    # (a)
        log_nb0 = self.phi * (math.log(self.phi) - self._lae(e))
        logit = (torch.log(self.pi) - torch.log1p(-self.pi))[None, :] - log_nb0
        self.R = (torch.rand_like(e) < torch.sigmoid(logit)) & self.Yzero
        nr = self.R.sum(0).to(torch.float32)
        self.pi = torch.distributions.Beta(self.cfg.a_pi + nr,
                                           self.cfg.b_pi + self.n - nr).sample().clamp(1e-6, 1 - 1e-6)

    def update_mu0(self, e):                                                  # (b)
        prop = self.mu0 + self.tau_mu0 * torch.randn_like(self.mu0)
        e_new = e + (prop - self.mu0)[None, :]
        ll_new = self.ll_gene(e_new)
        s2 = self.cfg.sigma_mu0 ** 2
        logr = ll_new - self.llg - ((prop ** 2 - self.mu0 ** 2) / (2 * s2)).double()
        acc = torch.log(torch.rand_like(logr)) < logr
        self.mu0 = torch.where(acc, prop, self.mu0)
        self.llg = torch.where(acc, ll_new, self.llg)
        self._count("mu0", acc.float())
        return torch.where(acc[None, :], e_new, e)

    def update_alpha(self, e):                                                # (c)
        if self.mode == "none":
            return e
        s2 = self.cfg.sigma_alpha ** 2
        if self.mode == "shared":
            for k in range(self.K):
                if self.Wsum[k] == 0:
                    continue
                old = float(self.alpha[k])
                prop = old + float(self.tau_alpha[k]) * float(self.rng.standard_normal())
                e_new = e + (prop - old) * self.W[:, k:k + 1]
                ll_new = self.ll_gene(e_new)
                logr = float((ll_new - self.llg).sum()) - (prop ** 2 - old ** 2) / (2 * s2)
                if math.log(self.rng.random() + 1e-300) < logr:
                    self.alpha[k] = prop
                    self.llg, e = ll_new, e_new
                    self._win["alpha"][k] += 1
                    self._tot["alpha"][k] += 1
            return e
        for k in range(self.K):                                               # gene-specific
            if k == self.ref or self.Wsum[k] == 0:
                continue
            old = self.alpha[:, k]
            prop = old + self.tau_alpha[:, k] * torch.randn_like(old)
            e_new = e + self.W[:, k:k + 1] * (prop - old)[None, :]
            ll_new = self.ll_gene(e_new)
            logr = ll_new - self.llg - ((prop ** 2 - old ** 2) / (2 * s2)).double()
            acc = torch.log(torch.rand_like(logr)) < logr
            self.alpha[:, k] = torch.where(acc, prop, old)
            self.llg = torch.where(acc, ll_new, self.llg)
            e = torch.where(acc[None, :], e_new, e)
            self._win["alpha"][:, k] += acc.float()
            self._tot["alpha"][:, k] += acc.float()
        return e

    def update_phi(self, e):                                                  # (d)
        tau, phi = self.tau_phi, self.phi
        prop = -1.0
        while prop <= 0:
            prop = phi + tau * float(self.rng.standard_normal())
        a, b = self.cfg.a_phi, self.cfg.b_phi
        lp = lambda v: (a - 1) * math.log(v) - b * v
        logr = (self.ll_phi_total(e, prop) - self.ll_phi_total(e, phi) + lp(prop) - lp(phi)
                + norm.logcdf(phi / tau) - norm.logcdf(prop / tau))
        if math.log(self.rng.random() + 1e-300) < logr:
            self.phi = prop
            self.llg = self.ll_gene(e)
            self._count("phi", 1.0)
        return e

    def grad_gamma(self, e, gamma):
        lae = self._lae(e)
        s = self.phi * (self.Y * torch.exp(-lae) - torch.exp(e - lae))      # d ll / d eta
        s.masked_fill_(self.R, 0.0)
        gb = self.B.T @ s                                                    # d ll / d beta
        thr = self.lam * self.rho
        ind = (gamma.abs() > thr).to(gamma.dtype)
        eps = self.cfg.eps_smooth
        spike = gamma * (2 * gamma / eps) / (math.pi * (1 + ((gamma ** 2 - thr ** 2) / eps) ** 2))
        spike = spike.clamp(-self.cfg.grad_spike_cap, self.cfg.grad_spike_cap)
        return gb * (ind + spike)

    def update_gamma(self, e):                                                # (e) pCN-Langevin
        tau = self.tau_gamma[self.comp]
        a = torch.sqrt(1 - tau ** 2)
        g = self.grad_gamma(e, self.gamma)
        m = a * self.gamma + (1 - a) * self._Sigma_mult(g)
        prop = m + tau * self._Sigma_half_noise()
        e_new = e + self.B @ (self.beta_of(prop) - self.beta_of(self.gamma))
        ll_new = self.ll_gene(e_new)
        g_new = self.grad_gamma(e_new, prop)
        m_new = a * prop + (1 - a) * self._Sigma_mult(g_new)
        s2 = self.sigma2
        prior_old = -0.5 * self._quad(self.gamma) / s2
        prior_new = -0.5 * self._quad(prop) / s2
        q_fwd = -0.5 * self._quad(prop - m) / (s2 * tau ** 2)
        q_rev = -0.5 * self._quad(self.gamma - m_new) / (s2 * tau ** 2)
        per_gene = (ll_new - self.llg) + (prior_new - prior_old + q_rev - q_fwd).double()
        logr = self._comp_sum(per_gene)
        acc_c = torch.log(torch.rand(self.ncomp, device=self.dev, dtype=torch.float64)) < logr
        acc = acc_c[self.comp]
        self.gamma = torch.where(acc[None, :], prop, self.gamma)
        self.llg = torch.where(acc, ll_new, self.llg)
        self._count("gamma", acc_c.float())
        return torch.where(acc[None, :], e_new, e)

    def update_rho(self, e):                                                  # (f)
        beta_old = self.beta_of(self.gamma)
        self.rho = self._compute_rho(self.gamma)
        beta_new = self.beta_of(self.gamma)
        if not torch.equal(beta_old, beta_new):
            e = e + self.B @ (beta_new - beta_old)
            self.llg = self.ll_gene(e)
        return e

    def update_sigma2(self):                                                  # (g)
        quad = float(self._quad(self.gamma).double().sum())
        shape, rate = self.cfg.a_gamma + self.p, self.cfg.b_gamma + 0.5 * quad
        self.sigma2 = 1.0 / self.rng.gamma(shape, 1.0 / rate)

    def update_lambda(self, e):                                               # (h)
        lo, hi, tau, lam = 0.0, self.lam_u, self.tau_lam, self.lam
        prop = float(truncnorm.rvs((lo - lam) / tau, (hi - lam) / tau, loc=lam, scale=tau,
                                   random_state=self.rng))
        beta_old = self.beta_of(self.gamma)
        beta_new = self.beta_of(self.gamma, lam=prop)
        Z = lambda c: max(norm.cdf((hi - c) / tau) - norm.cdf((lo - c) / tau), 1e-300)
        if torch.equal(beta_old, beta_new):
            e_new, ll_new, dll = e, self.llg, 0.0
        else:
            e_new = e + self.B @ (beta_new - beta_old)
            ll_new = self.ll_gene(e_new)
            dll = float((ll_new - self.llg).sum())
        logr = dll + math.log(Z(lam)) - math.log(Z(prop))
        if math.log(self.rng.random() + 1e-300) < logr:
            self.lam, self.llg, e = prop, ll_new, e_new
            self._count("lam", 1.0)
        return e

    # ---------------------------------------------------------------- bookkeeping
    def _adapt(self):
        w, t = float(self.cfg.adapt_every), self.cfg.target_accept
        self.tau_mu0 = (self.tau_mu0 * torch.exp(self._win["mu0"] / w - t)).clamp(1e-4, 2.0)
        if self.alpha is not None:
            self.tau_alpha = (self.tau_alpha * torch.exp(self._win["alpha"] / w - t)).clamp(1e-5, 2.0)
        self.tau_gamma = (self.tau_gamma * torch.exp(self._win["gamma"] / w - t)).clamp(1e-4, 0.99)
        self.tau_phi = float(min(max(self.tau_phi * math.exp(self._win["phi"] / w - t), 1e-3), 50.0))
        self.tau_lam = float(min(max(self.tau_lam * math.exp(self._win["lam"] / w - t), 1e-6), self.lam_u))
        self._win = self._zero_counters()

    def _record(self):
        t = (self.gamma.abs() > self.lam * self.rho).float()
        self.sum_t += t
        self.sum_gt += self.gamma * t
        self.sum_mu0 += self.mu0
        if self.alpha is not None:
            self.sum_alpha += self.alpha
        self.sum_phi += self.phi
        self.sum_R += self.R.float()
        self.M += 1

    def _trace(self):
        act = ((self.gamma.abs() > self.lam * self.rho).any(0)).sum().item()
        self.trace["phi"].append(self.phi)
        self.trace["lam"].append(self.lam)
        self.trace["sigma2"].append(self.sigma2)
        self.trace["loglik"].append(float(self.llg.sum()))
        self.trace["n_active"].append(act)

    def zinb_loglik(self, mu0, alpha, beta, phi, Rhat):
        """Per-gene ZINB log-lik at posterior estimates (for posterior model probabilities)."""
        e = self.eta(mu0=mu0, alpha=alpha, beta=beta)
        lognb = (torch.lgamma(self.Y + phi) - math.lgamma(phi) - torch.lgamma(self.Y + 1)
                 + phi * math.log(phi) + self.Y * e - (self.Y + phi) * self._lae(e, phi))
        ll = torch.where(self.Yzero,
                         torch.log(Rhat + (1 - Rhat) * torch.exp(lognb) + 1e-30),
                         torch.log1p(-Rhat.clamp(max=1 - 1e-6)) + lognb)
        return ll.sum(0, dtype=torch.float64)

    @torch.no_grad()
    def run(self, label=""):
        cfg, t0 = self.cfg, time.time()
        for it in range(cfg.n_iter):
            e = self.eta()
            self.update_R(e)                     # (a)
            self.llg = self.ll_gene(e)
            e = self.update_mu0(e)               # (b)
            e = self.update_alpha(e)             # (c)
            e = self.update_phi(e)               # (d)
            e = self.update_gamma(e)             # (e)
            e = self.update_rho(e)               # (f)
            self.update_sigma2()                 # (g)
            e = self.update_lambda(e)            # (h)
            self._tot_n += 1
            if it < cfg.burn_in and (it + 1) % cfg.adapt_every == 0:
                self._adapt()
            if it >= cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                self._record()
            self._trace()
            if it == 9:
                spi = (time.time() - t0) / 10
                print(f"    [{label}] {spi:.2f} s/iter -> ~{spi * cfg.n_iter / 60:.1f} min for this run",
                      flush=True)
            if (it + 1) % cfg.verbose_every == 0:
                tn = self._tot_n
                print(f"    [{label}] it {it + 1:>5d}  phi={self.phi:7.3f}  lam={self.lam:.4f}  "
                      f"sig2={self.sigma2:.4f}  active={self.trace['n_active'][-1]:>5d}  "
                      f"acc(mu0={self._tot['mu0'].mean().item() / tn:.2f}, "
                      f"gamma={self._tot['gamma'].mean().item() / tn:.2f}, "
                      f"phi={self._tot['phi'] / tn:.2f}, lam={self._tot['lam'] / tn:.2f})", flush=True)
        return self.summary()

    @torch.no_grad()
    def summary(self):
        M = max(self.M, 1)
        pip = self.sum_t / M
        beta_hat = self.sum_gt / self.sum_t.clamp_min(1.0)       # paper: mean of gamma when selected
        mu0_hat = self.sum_mu0 / M
        alpha_hat = None if self.sum_alpha is None else self.sum_alpha / M
        phi_hat = self.sum_phi / M
        R_hat = self.sum_R / M
        ll = self.zinb_loglik(mu0_hat, alpha_hat, beta_hat, phi_hat, R_hat)
        del self.sum_R, R_hat
        tn = max(self._tot_n, 1)
        acc = {k: (float(v.mean()) / tn if torch.is_tensor(v) else v / tn) for k, v in self._tot.items()}
        return dict(pip=pip.cpu().numpy(), beta_hat=beta_hat.cpu().numpy(), ll_gene=ll.cpu().numpy(),
                    phi_hat=phi_hat, acc=acc, traces={k: np.asarray(v) for k, v in self.trace.items()})


# ============================================================================ post-processing
def combine_kernels(pips: np.ndarray, lls: np.ndarray, how: str):
    """pips (S,2,p), lls (S,p) -> combined PIP~_j = max_d sum_s w_s PIP_s,dj, and weights."""
    if how == "paper":
        tot = lls.sum(1)
        w = np.exp(tot - tot.max())
        w /= w.sum()
        pip = np.tensordot(w, pips, axes=1)
    elif how == "genewise":
        w = np.exp(lls - lls.max(0, keepdims=True))
        w /= w.sum(0, keepdims=True)
        pip = (w[:, None, :] * pips).sum(0)
    else:
        raise ValueError(how)
    return pip.max(0), w


def bfdr_select(pip: np.ndarray, level: float) -> np.ndarray:
    q = 1 - pip
    order = np.argsort(q, kind="stable")
    cm = np.cumsum(q[order]) / np.arange(1, len(q) + 1)
    ok = np.where(cm <= level)[0]
    sel = np.zeros(len(q), bool)
    if len(ok):
        sel[order[: ok.max() + 1]] = True
    return sel


def evaluate(res: pd.DataFrame, genes: pd.DataFrame) -> dict:
    if "is_SV" not in res.columns:
        return {}
    ev = res[~res["is_ctSV"].astype(bool)]
    y = ev["is_SV"].astype(bool).to_numpy()
    c = ev["SV_called"].astype(bool).to_numpy()
    tp, fp, fn = int((y & c).sum()), int((~y & c).sum()), int((y & ~c).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    n_true_all = int(genes["is_SV"].astype(bool).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                n_true_SV_total=n_true_all, n_true_SV_analyzed=int(y.sum()),
                recall_vs_all_true_SV=tp / max(n_true_all, 1),
                call_rate_by_class=res.groupby("eval_class")["SV_called"].mean().to_dict(),
                n_by_class=res["eval_class"].value_counts().to_dict())


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ============================================================================ driver
def fit(sim_dir: str, out_dir: str, cfg: NABRConfig) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    dev = get_device(cfg.device)
    t_all = time.time()
    print(f"[NABR] device={dev}  alpha_mode={cfg.alpha_mode}  unit={cfg.analysis_unit}")
    X, bins, genes, edges, meta = load_sim(sim_dir)
    comp_names = meta["compartments"]
    rng = np.random.default_rng(cfg.seed)

    gsel = select_genes(X, cfg, rng)
    units = build_units(X, bins, cfg, rng, comp_names)
    keep_u = units["lib"] > 0
    Yd = units["Y"][keep_u][:, gsel].toarray().astype(np.float32)
    gk = Yd.sum(0) > 0
    gsel, Yd = gsel[gk], Yd[:, gk]
    Wm = units["W"][keep_u].astype(np.float32)
    xy, lib = units["xy"][keep_u], units["lib"][keep_u]
    logc = np.log(lib / np.median(lib)).astype(np.float32)
    n, p = Yd.shape
    K = Wm.shape[1]
    ref = int(np.argmax(Wm.sum(0)))
    gene_names = genes["gene"].to_numpy()[gsel]
    print(f"[NABR] units n={n:,}  genes p={p:,}  zeros={np.mean(Yd == 0):.1%}  "
          f"reference type={comp_names[ref]}  ~{n * p * 4 / 1e9:.2f} GB per dense n x p tensor")
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "logc": logc,
                  **{f"w_{c}": Wm[:, k] for k, c in enumerate(comp_names)}}
                 ).to_csv(os.path.join(out_dir, "units.csv"), index=False)

    Y = torch.tensor(Yd, device=dev)
    del Yd
    data = dict(Y=Y, Yzero=(Y == 0), ynz=Y[Y > 0].contiguous(),
                logc=torch.tensor(logc, device=dev), W=torch.tensor(Wm, device=dev), ref=ref)

    A, Ltil, comp, ncomp, deg = build_network(gene_names, edges)
    sizes = np.bincount(comp)
    print(f"[NABR] network: {int(A.nnz // 2):,} edges, {int((sizes > 1).sum())} sub-networks "
          f"(largest {sizes.max()}), {int((sizes == 1).sum()):,} singletons")
    comp_t = torch.tensor(comp, device=dev, dtype=torch.long)

    S, C = len(cfg.kernels), cfg.n_chains
    pip_all = np.zeros((C, S, 2, p))
    ll_all = np.zeros((C, S, p))
    beta_all = np.zeros((C, S, 2, p))
    traces, accs = {}, {}
    Wsum = Wm.sum(0)
    nonref = [k for k in range(K) if k != ref and Wsum[k] > 0]

    for s, kname in enumerate(cfg.kernels):
        print(f"[NABR] kernel {kname} ({s + 1}/{S})")
        B = torch.tensor(kernel_basis(xy, kname), device=dev)
        cols = [torch.ones(n, 1, device=dev)]
        if cfg.alpha_mode != "none":
            cols.append(data["W"][:, nonref])
        cols.append(B)
        coef = nb_irls(Y, data["logc"], torch.cat(cols, 1))
        beta_tilde = coef[-2:].cpu().numpy()
        P, Lc = build_precisions(Ltil, beta_tilde, cfg.eps_L, dev)
        prec = dict(P=P, chol=Lc, comp=comp_t, ncomp=ncomp)
        lam_u = float(np.quantile(np.abs(beta_tilde), cfg.lambda_quantile))
        init = dict(gamma=coef[-2:].clone(), lam_u=lam_u, phi=10.0)
        if cfg.alpha_mode == "gene":
            a0 = torch.zeros(p, K, device=dev)
            a0[:, nonref] = coef[1:1 + len(nonref)].T
            init.update(mu0=coef[0].clone(), alpha=a0)
        elif cfg.alpha_mode == "shared":
            init.update(mu0=torch.log((Y.sum(0) + 0.5) / torch.exp(data["logc"]).sum()))
        else:
            init.update(mu0=coef[0].clone())
        print(f"    rough beta~: lambda_u={lam_u:.4f}; |beta~| median={np.median(np.abs(beta_tilde)):.4f}")

        for c in range(C):
            t0 = time.time()
            chain = NABRChain(data, B, prec, cfg, init, seed=cfg.seed * 7919 + 101 * c + s)
            out = chain.run(label=f"chain {c + 1}/{C} {kname}")
            pip_all[c, s], ll_all[c, s], beta_all[c, s] = out["pip"], out["ll_gene"], out["beta_hat"]
            for k, v in out["traces"].items():
                traces[f"c{c}_{kname}_{k}"] = v
            accs[f"c{c}_{kname}"] = out["acc"]
            print(f"    chain {c + 1} done in {(time.time() - t0) / 60:.1f} min; "
                  f"genes with PIP>0.5: {(out['pip'].max(0) > 0.5).sum()}", flush=True)
            del chain
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del P, Lc, prec, B, coef
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    pip_chain = np.zeros((C, p))
    sel_chain = np.zeros((C, p), bool)
    weights = []
    for c in range(C):
        pip_c, w_c = combine_kernels(pip_all[c], ll_all[c], cfg.combine)
        pip_chain[c], sel_chain[c] = pip_c, bfdr_select(pip_c, cfg.bfdr)
        weights.append(w_c if w_c.ndim == 1 else w_c.mean(1))
    call_freq = sel_chain.mean(0)
    final = call_freq >= cfg.chain_agreement - 1e-9

    res = pd.DataFrame({"gene": gene_names, "net_degree": deg.astype(int), "component": comp,
                        "component_size": sizes[comp], "pip": pip_chain.mean(0),
                        "call_freq": call_freq, "SV_called": final})
    for s, k in enumerate(cfg.kernels):
        res[f"pip_{k}"] = pip_all[:, s].max(axis=1).mean(0)
        res[f"beta1_{k}"] = beta_all[:, s, 0].mean(0)
        res[f"beta2_{k}"] = beta_all[:, s, 1].mean(0)
    truth = [c for c in ("is_SV", "is_ctSV", "eval_class", "pattern", "module", "module_kind",
                         "is_hub", "module_noise", "marker_of", "off_in", "detect_frac") if c in genes.columns]
    res = res.merge(genes[["gene"] + truth], on="gene", how="left")
    res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "diagnostics.npz"), **traces)

    metrics = evaluate(res, genes)
    summary = dict(config=asdict(cfg), n_units=n, n_genes=p, n_called=int(final.sum()),
                   kernel_weights=np.array(weights).tolist(), kernels=list(cfg.kernels),
                   acceptance=accs, metrics=metrics, runtime_min=(time.time() - t_all) / 60)
    with open(os.path.join(out_dir, "fit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=_jsonable)
    print(f"[NABR] called {int(final.sum())} SV genes of {p}; "
          f"metrics: {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, default=_jsonable)}")
    print(f"[NABR] total {(time.time() - t_all) / 60:.1f} min -> {out_dir}")
    return res
