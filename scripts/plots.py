#!/usr/bin/env python3
"""plots.py -- figures for the synthetic data, ground truth, and the NABR fit.

Every figure is wrapped in try/except so one failure does not stop the others;
the traceback is printed for debugging.
"""
from __future__ import annotations

import json
import os
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

COMP_COLORS = {"Epithelial": "#d1495b", "Stroma": "#edae49", "Adipose": "#f3e3b5",
               "Immune": "#00798c", "Vascular": "#30638e", "SmoothMuscle": "#7b5ea7"}
BG = "#d9d9d9"


# ----------------------------------------------------------------------------- helpers
def _load(sim_dir):
    X = sp.load_npz(os.path.join(sim_dir, "counts.npz")).tocsc()
    bins = pd.read_csv(os.path.join(sim_dir, "bins.csv"))
    genes = pd.read_csv(os.path.join(sim_dir, "genes.csv"), keep_default_na=False)
    edges = pd.read_csv(os.path.join(sim_dir, "edges.csv"))
    with open(os.path.join(sim_dir, "meta.json")) as fh:
        meta = json.load(fh)
    return X, bins, genes, edges, meta


def _grid(bins, values, shape):
    img = np.full(shape, np.nan)
    img[bins["row"].to_numpy(), bins["col"].to_numpy()] = values
    return img


def _nan_smooth(img, sigma):
    m = ~np.isnan(img)
    num = gaussian_filter(np.where(m, img, 0.0), sigma)
    den = gaussian_filter(m.astype(float), sigma)
    out = num / np.maximum(den, 1e-9)
    out[~m] = np.nan
    return out


def _expr_img(X, bins, shape, sf, j, sigma=1.5):
    v = np.log1p(X[:, j].toarray().ravel() / sf)
    img = _grid(bins, v, shape)
    return _nan_smooth(img, sigma) if sigma > 0 else img


def _show(ax, img, title, cmap="magma", vmin=0.0, vmax=None):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(BG)
    if vmax is None:
        vmax = np.nanpercentile(img, 99.5) if np.isfinite(img).any() else 1.0
    im = ax.imshow(img, cmap=cm, vmin=vmin, vmax=max(vmax, vmin + 1e-9), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {path}")


def _safe(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception:
        print(f"[plots] {fn.__name__} failed:")
        traceback.print_exc()


def _gene_title(row, extra=""):
    return f"{row['gene']}  {row.get('eval_class', '')}{extra}"


# ----------------------------------------------------------------------------- data figures
def plot_tissue(bins, meta, out):
    comps, shape = meta["compartments"], (meta["H"], meta["W"])
    img = _grid(bins, bins["comp_code"].to_numpy(float), shape)
    cmap = ListedColormap([COMP_COLORS[c] for c in comps])
    cmap.set_bad(BG)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax[0].imshow(img, cmap=cmap, vmin=-0.5, vmax=len(comps) - 0.5, interpolation="nearest")
    ax[0].set_title(f"Bin types ({len(bins):,} bins of {meta['bin_um']:.0f} um)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].legend(handles=[Patch(fc=COMP_COLORS[c], ec="k", label=c) for c in comps],
                 loc="lower left", fontsize=8, framealpha=0.9)
    cnt = bins["compartment"].value_counts().reindex(comps).fillna(0)
    ax[1].barh(comps, cnt.values, color=[COMP_COLORS[c] for c in comps], edgecolor="k")
    for i, v in enumerate(cnt.values):
        ax[1].text(v, i, f" {int(v):,} ({v / len(bins):.1%})", va="center", fontsize=8)
    ax[1].set_xlabel("number of bins")
    ax[1].invert_yaxis()
    _save(fig, os.path.join(out, "01_tissue_bin_types.png"))


def plot_qc(bins, meta, out):
    comps, shape = meta["compartments"], (meta["H"], meta["W"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    im = _show(ax[0, 0], _grid(bins, np.log10(bins["total_counts"] + 1), shape), "log10 UMIs per bin",
               cmap="viridis", vmin=None)
    fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    im = _show(ax[0, 1], _grid(bins, bins["n_genes"].to_numpy(float), shape), "genes detected per bin",
               cmap="viridis", vmin=None)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    for a, col, lab in [(ax[1, 0], "total_counts", "UMIs per bin"), (ax[1, 1], "n_genes", "genes per bin")]:
        data = [bins.loc[bins["compartment"] == c, col].to_numpy() for c in comps]
        bp = a.boxplot(data, positions=np.arange(len(comps)), showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], comps):
            patch.set_facecolor(COMP_COLORS[c])
        a.set_xticks(np.arange(len(comps)))
        a.set_xticklabels(comps, rotation=30, ha="right")
        a.set_yscale("log")
        a.set_ylabel(lab)
    _save(fig, os.path.join(out, "02_qc_depth_and_detection.png"))


def plot_truth_examples(X, bins, genes, meta, sf, out):
    shape = (meta["H"], meta["W"])
    picks = []
    for pat in ["linear", "gaussian", "periodic", "hotspot"]:
        cand = genes[(genes["eval_class"] == f"SV:{pat}") & (genes["is_hub"].astype(bool))]
        if cand.empty:
            cand = genes[genes["eval_class"] == f"SV:{pat}"]
        if not cand.empty:
            picks.append(cand.sort_values("mu0").iloc[-1])
    for cls, extra in [("marker (null)", "Epithelial"), ("marker (null)", "Adipose"),
                       ("adipose-off (null)", None), ("ctSV", None)]:
        cand = genes[genes["eval_class"] == cls]
        if extra:
            cand = cand[cand["marker_of"].str.contains(extra)]
        if not cand.empty:
            picks.append(cand.sort_values("mu0").iloc[-1])
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for a in axes.ravel():
        a.axis("off")
    for a, row in zip(axes.ravel(), picks):
        j = int(row.name)
        extra = f"\nmarker of {row['marker_of']}" if row["marker_of"] else ""
        if row["eval_class"] == "ctSV":
            extra = f"\nspatial only in {row['ct_type']}"
        _show(a, _expr_img(X, bins, shape, sf, j), _gene_title(row, extra))
        a.axis("on")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Ground truth examples: log1p(normalized counts), Gaussian-smoothed for display (sigma = 1.5 bins)")
    _save(fig, os.path.join(out, "03_truth_examples.png"))


# ----------------------------------------------------------------------------- result figures
def _curves(scores, labels):
    o = np.argsort(-scores, kind="stable")
    l = labels[o].astype(float)
    tp, fp = np.cumsum(l), np.cumsum(1 - l)
    P, N = max(l.sum(), 1), max(len(l) - l.sum(), 1)
    prec, rec, fpr = tp / np.arange(1, len(l) + 1), tp / P, fp / N
    x, y = np.r_[0, fpr], np.r_[0, rec]
    auroc = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))
    ap = float(np.sum(prec * l) / P)
    return prec, rec, fpr, auroc, ap


def plot_results(res, summ, out):
    ev = res[~res["is_ctSV"].astype(bool)]
    y = ev["is_SV"].astype(bool).to_numpy()
    s = ev["pip"].to_numpy()
    classes = sorted([c for c in res["eval_class"].unique() if c.startswith("SV:")]) + \
        [c for c in ["ctSV", "marker (null)", "adipose-off (null)", "null"] if c in set(res["eval_class"])]
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))

    a = ax[0, 0]
    rng = np.random.default_rng(0)
    for i, c in enumerate(classes):
        v = res.loc[res["eval_class"] == c, "pip"].to_numpy()
        a.scatter(i + rng.uniform(-0.3, 0.3, len(v)), v, s=4, alpha=0.4,
                  color="tab:red" if c.startswith("SV:") else "tab:gray")
        a.text(i, 1.04, f"n={len(v)}", ha="center", fontsize=7)
    a.set_xticks(range(len(classes)))
    a.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    a.set_ylabel("combined PIP (mean over chains)")
    a.set_ylim(-0.03, 1.1)
    a.set_title("PIP by ground-truth class")

    prec, rec, fpr, auroc, ap = _curves(s, y)
    ax[0, 1].plot(rec, prec)
    ax[0, 1].set_xlabel("recall")
    ax[0, 1].set_ylabel("precision")
    ax[0, 1].set_title(f"Precision-recall (AP = {ap:.3f})")
    m = summ.get("metrics", {})
    if m:
        ax[0, 1].scatter([m["recall"]], [m["precision"]], color="red", zorder=3,
                         label=f"final calls: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
        ax[0, 1].legend(fontsize=8, loc="lower left")
    ax[0, 2].plot(fpr, rec)
    ax[0, 2].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0, 2].set_xlabel("false positive rate")
    ax[0, 2].set_ylabel("true positive rate")
    ax[0, 2].set_title(f"ROC (AUROC = {auroc:.3f})")

    rate = res.groupby("eval_class")["SV_called"].mean().reindex(classes)
    ax[1, 0].bar(range(len(classes)), rate.values,
                 color=["tab:red" if c.startswith("SV:") else "tab:gray" for c in classes])
    ax[1, 0].set_xticks(range(len(classes)))
    ax[1, 0].set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax[1, 0].set_ylabel("fraction called SV")
    ax[1, 0].set_title("Call rate by class (red = should be called)")

    sv = res[res["is_SV"].astype(bool)].copy()
    sv["net"] = np.where(sv["component_size"] > 1, "in network", "singleton")
    tab = sv.groupby(["pattern", "net"])["SV_called"].mean().unstack()
    tab.plot.bar(ax=ax[1, 1], rot=0)
    ax[1, 1].set_ylabel("recall")
    ax[1, 1].set_title("Recall by true pattern and network support")
    ax[1, 1].set_ylim(0, 1.05)

    w = np.array(summ.get("kernel_weights", []))
    if w.size:
        im = ax[1, 2].imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax[1, 2].set_xticks(range(len(summ["kernels"])))
        ax[1, 2].set_xticklabels(summ["kernels"])
        ax[1, 2].set_yticks(range(w.shape[0]))
        ax[1, 2].set_yticklabels([f"chain {i + 1}" for i in range(w.shape[0])])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                ax[1, 2].text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax[1, 2], shrink=0.7)
        ax[1, 2].set_title(f"Kernel model weights (combine = {summ['config']['combine']})")
    fig.suptitle(f"NABR results: {summ.get('n_called', '?')} genes called of {summ.get('n_genes', '?')} analyzed "
                 f"(alpha_mode = {summ['config']['alpha_mode']}, unit = {summ['config']['analysis_unit']})")
    _save(fig, os.path.join(out, "04_results_summary.png"))


def plot_network(res, edges, out):
    r = res.copy()
    r["module"] = pd.to_numeric(r["module"], errors="coerce")
    mods = r[r["module"] >= 0].groupby("module").agg(n=("gene", "size"), sv=("is_SV", "mean"))
    inf = mods[(mods["sv"] > 0.5) & (mods["n"] >= 8)].sort_values("n", ascending=False).index[:3]
    non = mods[(mods["sv"] == 0) & (mods["n"] >= 8)].sort_values("n", ascending=False).index[:2]
    sub = r[r["module"].isin(list(inf) + list(non))].set_index("gene")
    if sub.empty:
        print("[plots] network: no suitable modules among analyzed genes")
        return
    names = set(sub.index)
    G = nx.Graph()
    G.add_nodes_from(names)
    e = edges[edges["gene_a"].isin(names) & edges["gene_b"].isin(names)]
    G.add_edges_from(zip(e["gene_a"], e["gene_b"]))
    pos = nx.spring_layout(G, seed=1, k=0.15)
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.6)
    sc = None
    for truth, shape, lab in [(True, "o", "true SV"), (False, "s", "true null")]:
        nodes = [g for g in G.nodes if bool(sub.loc[g, "is_SV"]) == truth]
        if not nodes:
            continue
        sc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_shape=shape, node_size=70,
                                    node_color=[sub.loc[g, "pip"] for g in nodes], cmap="viridis",
                                    vmin=0, vmax=1, label=lab,
                                    edgecolors=["red" if sub.loc[g, "SV_called"] else "none" for g in nodes],
                                    linewidths=1.4)
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.6, label="PIP")
    ax.legend(loc="upper left", fontsize=9, title="circle = SV, square = null\nred ring = called")
    ax.set_title("Gene network: 3 informative + 2 null modules (node colour = PIP)")
    ax.axis("off")
    _save(fig, os.path.join(out, "05_network_modules.png"))


def plot_called_maps(X, bins, genes, meta, sf, res, out):
    shape = (meta["H"], meta["W"])
    col_of = pd.Series(np.arange(len(genes)), index=genes["gene"])
    r = res[~res["is_ctSV"].astype(bool)]
    tp = r[r["is_SV"].astype(bool) & r["SV_called"]].sort_values("pip", ascending=False).head(4)
    fp = r[~r["is_SV"].astype(bool)].sort_values("pip", ascending=False).head(4)
    fn = r[r["is_SV"].astype(bool) & ~r["SV_called"]].sort_values("pip").head(4)
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    for a in axes.ravel():
        a.axis("off")
    for i, (df, lab) in enumerate([(tp, "TP"), (fp, "top null (FP if called)"), (fn, "FN")]):
        for jj, (_, row) in enumerate(df.iterrows()):
            a = axes[i, jj]
            _show(a, _expr_img(X, bins, shape, sf, int(col_of[row["gene"]])),
                  f"{lab}: {row['gene']}  PIP={row['pip']:.2f}\n{row['eval_class']}")
            a.axis("on")
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Expression maps (log1p normalized, smoothed): true positives / highest-PIP nulls / missed SV genes")
    _save(fig, os.path.join(out, "06_called_gene_maps.png"))


def plot_diagnostics(fit_dir, summ, out):
    d = np.load(os.path.join(fit_dir, "diagnostics.npz"))
    kernels = summ["kernels"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, key in zip(ax.ravel(), ["phi", "lam", "sigma2", "loglik", "n_active"]):
        for k in kernels:
            name = f"c0_{k}_{key}"
            if name in d.files:
                a.plot(d[name], lw=0.7, label=k)
        a.axvline(summ["config"]["burn_in"], color="k", ls="--", lw=0.7)
        a.set_title(f"chain 1: {key}")
        if key in ("sigma2",):
            a.set_yscale("log")
    ax[0, 0].legend(fontsize=7)
    acc = pd.DataFrame(summ["acceptance"]).T
    ax[1, 2].axis("off")
    ax[1, 2].table(cellText=np.round(acc.to_numpy(float), 2), rowLabels=acc.index, colLabels=acc.columns,
                   loc="center", fontsize=7)
    ax[1, 2].set_title("mean acceptance rates")
    _save(fig, os.path.join(out, "07_mcmc_diagnostics.png"))


# ----------------------------------------------------------------------------- driver
def make_all_plots(sim_dir, fit_dir=None, fig_dir=None):
    fig_dir = fig_dir or os.path.join(fit_dir or sim_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    X, bins, genes, edges, meta = _load(sim_dir)
    sf = bins["total_counts"].to_numpy(float)
    sf = np.maximum(sf / np.median(sf), 1e-3)
    _safe(plot_tissue, bins, meta, fig_dir)
    _safe(plot_qc, bins, meta, fig_dir)
    _safe(plot_truth_examples, X, bins, genes, meta, sf, fig_dir)
    if fit_dir and os.path.exists(os.path.join(fit_dir, "results.csv")):
        res = pd.read_csv(os.path.join(fit_dir, "results.csv"), keep_default_na=False)
        for c in ("is_SV", "is_ctSV", "SV_called", "is_hub"):
            if c in res.columns:
                res[c] = res[c].astype(str).str.lower().eq("true")
        with open(os.path.join(fit_dir, "fit_summary.json")) as fh:
            summ = json.load(fh)
        _safe(plot_results, res, summ, fig_dir)
        _safe(plot_network, res, edges, fig_dir)
        _safe(plot_called_maps, X, bins, genes, meta, sf, res, fig_dir)
        _safe(plot_diagnostics, fit_dir, summ, fig_dir)
    print(f"[plots] figures in {fig_dir}")
