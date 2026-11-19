#!/usr/bin/env python3
"""
make_slide_assets.py -- figures, tables and numbers for slides.tex, built from outputs you already have.

Run from the scripts folder (where the fit_* and sim_* folders are):
    python make_slide_assets.py
Creates
    slides/figs/*.pdf|png
    slides/tables/runs_ours.tex, runs_paper.tex, numbers.tex   (numbers used in the slide text)
and copies slides.tex into slides/ if it is in the current folder.
Missing runs are skipped; the slides then show a 'missing' box or ?? instead of failing.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = "slides"
FIG = os.path.join(OUT, "figs")
TAB = os.path.join(OUT, "tables")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

RUNS = [
    dict(key="A", dir="fit_out", rho="on", note=r"5 kernels, 2 chains", label="A: ρ on, 5 kernels, 2 chains"),
    dict(key="B", dir="fit_dbg", rho="on", note=r"linear kernel, 1 chain", label="B: ρ on, linear kernel"),
    dict(key="C", dir="fit_norho", rho="off", note=r"linear kernel, 1 chain", label="C: ρ off, linear kernel"),
    dict(key="D", dir="fit_norho_all", rho="off", note=r"5 kernels, gene-wise weights", label="D: ρ off, 5 kernels"),
    dict(key="E", dir="fit_floor", rho="off", note=r"as D, with $\lambda \geq 0.5\,\lambda_u$",
         label="E: ρ off, 5 kernels, λ floor"),
    dict(key="F", dir="fit_paper", rho="on", note=r"linear kernel, 1 chain, 1{,}000 it.", label="F: ρ on"),
    dict(key="G", dir="fit_paper_norho", rho="off", note=r"linear kernel, 1 chain, 1{,}000 it.", label="G: ρ off"),
]

# (pattern, network, dropout) -> (recall, precision, F1) of the proposed method; paper Table 1 and Table S3
PAPER = {
    ("linear", "star", 0.1): (0.992, 1.000, 0.996), ("linear", "scalefree", 0.1): (0.991, 1.000, 0.995),
    ("exponential", "star", 0.1): (0.933, 1.000, 0.959), ("exponential", "scalefree", 0.1): (0.979, 1.000, 0.989),
    ("periodic", "star", 0.1): (0.996, 1.000, 0.998), ("periodic", "scalefree", 0.1): (0.979, 1.000, 0.981),
    ("linear", "star", 0.5): (0.975, 1.000, 0.985), ("linear", "scalefree", 0.5): (0.965, 1.000, 0.978),
    ("exponential", "star", 0.5): (0.873, 1.000, 0.913), ("exponential", "scalefree", 0.5): (0.937, 1.000, 0.958),
    ("periodic", "star", 0.5): (0.983, 1.000, 0.988), ("periodic", "scalefree", 0.5): (0.977, 1.000, 0.980),
}
ORDER = ["SV:linear", "SV:gaussian", "SV:periodic", "SV:hotspot", "ctSV",
         "marker (null)", "adipose-off (null)", "null"]
CLASSKEY = {"SV:linear": "svlinear", "SV:gaussian": "svgaussian", "SV:periodic": "svperiodic",
            "SV:hotspot": "svhotspot", "ctSV": "ctsv", "marker (null)": "marker",
            "adipose-off (null)": "adip", "null": "null"}
PATS = ["linear", "gaussian", "periodic", "hotspot", "ALL"]
NUM: dict[str, str] = {}


# ----------------------------------------------------------------------------- helpers
def put(key, val, nd=3):
    if isinstance(val, (bool, np.bool_)):
        s = "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        s = f"{int(val):,}"
    elif isinstance(val, (float, np.floating)):
        s = f"{float(val):.{nd}f}" if np.isfinite(val) else "--"
    elif val is None:
        s = "--"
    else:
        s = str(val)
    NUM[key] = s


def put_pct(key, val):
    NUM[key] = f"{100 * float(val):.0f}\\%" if val is not None and np.isfinite(val) else "--"


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def auc(score, y):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bool(r, c):
    if c in r.columns:
        return r[c].astype(str).str.lower().eq("true")
    return pd.Series(False, index=r.index)


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:
        print(f"[assets] {fn.__name__} failed:")
        traceback.print_exc()


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[assets] wrote {p}")


def copy_first(cands, name):
    for c in cands:
        if os.path.exists(c):
            shutil.copy(c, os.path.join(FIG, name))
            print(f"[assets] copied {c} -> figs/{name}")
            return c
    print(f"[assets] no source found for figs/{name}")
    return None


# ----------------------------------------------------------------------------- loading
def load_run(spec):
    d = spec["dir"]
    fr, fs = os.path.join(d, "results.csv"), os.path.join(d, "fit_summary.json")
    if not (os.path.exists(fr) and os.path.exists(fs)):
        print(f"[assets] run {spec['key']}: '{d}' not found -> skipped")
        return None
    r = pd.read_csv(fr, keep_default_na=False)
    with open(fs) as fh:
        s = json.load(fh)
    for c in ("is_SV", "is_ctSV", "SV_called", "is_hub"):
        r[c] = _bool(r, c)
    ks = list(s["config"]["kernels"])
    num = lambda c: pd.to_numeric(r[c], errors="coerce").fillna(0).to_numpy(float)
    r["absb"] = np.max([np.maximum(np.abs(num(f"beta1_{k}")), np.abs(num(f"beta2_{k}"))) for k in ks], axis=0)
    r["pip"] = pd.to_numeric(r["pip"], errors="coerce").fillna(0)
    r["eval_class"] = r["eval_class"].astype(str) if "eval_class" in r.columns else "null"
    R = dict(spec)
    R.update(res=r, summ=s, kernels=ks, ev=r[~r["is_ctSV"]].reset_index(drop=True))
    return R


def metrics(R):
    ev, m = R["ev"], R["summ"]["metrics"]
    y = ev["is_SV"].to_numpy()
    n1 = int(y.sum())
    o = np.argsort(-ev["absb"].to_numpy(), kind="stable")
    return dict(called=int(R["res"]["SV_called"].sum()), tp=int(m.get("tp", 0)), fp=int(m.get("fp", 0)),
                fn=int(m.get("fn", 0)), prec=float(m.get("precision", np.nan)), rec=float(m.get("recall", np.nan)),
                f1=float(m.get("f1", np.nan)), aucpip=auc(ev["pip"].to_numpy(), y),
                aucbeta=auc(ev["absb"].to_numpy(), y),
                topk=float(y[o[:n1]].mean()) if n1 else float("nan"), ntrue=n1,
                nunits=int(R["summ"].get("n_units", 0)), ngenes=int(R["summ"].get("n_genes", 0)),
                runtime=float(R["summ"].get("runtime_min", np.nan)))


def lam_post(R):
    p = os.path.join(R["dir"], "diagnostics.npz")
    if not os.path.exists(p):
        return float("nan")
    z = np.load(p)
    keys = sorted(k for k in z.files if k.startswith("c0_") and k.endswith("_lam"))
    if not keys:
        return float("nan")
    t, b = z[keys[0]], int(R["summ"]["config"]["burn_in"])
    return float(t[b:].mean()) if len(t) > b else float(t.mean())


def hub_counts(R):
    r = R["res"]
    sv, hub, c = r["is_SV"], r["is_hub"], r["SV_called"]
    return dict(tp=int((sv & c).sum()), nsv=int(sv.sum()),
                hub=int((~sv & hub & c).sum()), nhub=int((~sv & hub).sum()),
                non=int((~sv & ~hub & c).sum()), nnon=int((~sv & ~hub).sum()))


def _auc_by_pattern(R, score):
    ev = R["ev"]
    sv, s = ev["is_SV"].to_numpy(), ev[score].to_numpy()
    out = []
    for p in PATS:
        pos = sv if p == "ALL" else (ev["eval_class"] == f"SV:{p}").to_numpy()
        m = pos | ~sv
        out.append(auc(s[m], pos[m]))
    return out


# ----------------------------------------------------------------------------- tables
def write_table(runs, name, extra_rows=()):
    L = [r"\begin{tabular}{c c l r r r r r r}", r"\toprule",
         r"Run & $\rho$ & Setting & Called & Prec. & Recall & F1 & AUC$_{\mathrm{PIP}}$ & AUC$_{|\hat\beta|}$ \\",
         r"\midrule"]
    for R in runs:
        M = R["m"]
        L.append(f"{R['key']} & {R['rho']} & {R['note']} & {M['called']:,} & {f2(M['prec'])} & {f2(M['rec'])} & "
                 f"{f2(M['f1'])} & {f3(M['aucpip'])} & {f3(M['aucbeta'])} \\\\")
    if extra_rows:
        L.append(r"\midrule")
        L += list(extra_rows)
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[assets] wrote tables/{name}")


def write_numbers():
    p = os.path.join(TAB, "numbers.tex")
    with open(p, "w") as fh:
        fh.write("\\makeatletter\n")
        for k, v in sorted(NUM.items()):
            fh.write(f"\\@namedef{{res@{k}}}{{{v}}}\n")
        fh.write("\\makeatother\n")
    print(f"[assets] wrote tables/numbers.tex ({len(NUM)} numbers)")


# ----------------------------------------------------------------------------- figures
def fig_callrate(runs):
    if not runs:
        return
    classes = [c for c in ORDER if any((R["res"]["eval_class"] == c).any() for R in runs)]
    x, w = np.arange(len(classes)), 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    nulls = [i for i, c in enumerate(classes) if "null" in c]
    if nulls:
        ax.axvspan(min(nulls) - 0.5, len(classes) - 0.5, color="0.93", zorder=0)
        ax.text(min(nulls) - 0.4, 1.02, "nulls: should NOT be called", fontsize=9, va="bottom")
    for i, R in enumerate(runs):
        rate = R["res"].groupby("eval_class")["SV_called"].mean().reindex(classes)
        ax.bar(x + (i - (len(runs) - 1) / 2) * w, rate.to_numpy(), width=w, label=R["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel("fraction called (BFDR 0.05)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, ncol=len(runs), loc="lower center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    _save(fig, "callrate.pdf")


def fig_auc_pattern(runs):
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x, w = np.arange(len(PATS)), 0.8 / len(runs)
    for ax, score, title in zip(axes, ["pip", "absb"], ["ranked by PIP", "ranked by posterior |β|"]):
        for i, R in enumerate(runs):
            ax.bar(x + (i - (len(runs) - 1) / 2) * w, _auc_by_pattern(R, score), width=w, label=R["label"])
        ax.axhline(0.5, ls="--", color="0.4", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(PATS)
        ax.set_title(title)
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("AUROC (true SV vs nulls)")
    axes[0].legend(fontsize=9, loc="lower left")
    _save(fig, "auc_pattern.pdf")


def fig_topk(groups):
    groups = [(t, rs) for t, rs in groups if rs]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.2), squeeze=False)
    for ax, (title, rs) in zip(axes[0], groups):
        for R in rs:
            ev = R["ev"]
            y = ev["is_SV"].to_numpy()
            n1 = int(y.sum())
            if n1 == 0:
                continue
            o = np.argsort(-ev["absb"].to_numpy(), kind="stable")
            kmax = min(len(y), 2 * n1)
            prec = np.cumsum(y[o][:kmax]) / np.arange(1, kmax + 1)
            ax.plot(np.arange(1, kmax + 1) / n1, prec, lw=2, label=R["label"])
        ax.axvline(1.0, ls="--", color="0.4", lw=1)
        ax.set_xlabel("k / number of true SV genes")
        ax.set_ylabel("precision of top-k genes (by |β|)")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(fontsize=9)
    _save(fig, "topk.pdf")


def fig_hub(runs):
    if not runs:
        return
    cats = ["true SV called", "null hubs called", "other nulls called"]
    x, w = np.arange(3), 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, R in enumerate(runs):
        h = hub_counts(R)
        vals, tots = [h["tp"], h["hub"], h["non"]], [h["nsv"], h["nhub"], h["nnon"]]
        bars = ax.bar(x + (i - (len(runs) - 1) / 2) * w, vals, width=w, label=R["label"])
        for b, v, t in zip(bars, vals, tots):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}/{t}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("number of genes")
    ax.legend()
    _save(fig, "hub_ablation.pdf")


def fig_mechanism(R):
    if R is None:
        return
    r = R["res"]
    lam = lam_post(R)
    thr = lam ** (2 / 3) if np.isfinite(lam) else float("nan")
    groups = [("true SV, called", r[r["is_SV"] & r["SV_called"]]["absb"].to_numpy()),
              ("null hubs, called", r[~r["is_SV"] & r["is_hub"] & r["SV_called"]]["absb"].to_numpy()),
              ("other nulls, called", r[~r["is_SV"] & ~r["is_hub"] & r["SV_called"]]["absb"].to_numpy())]
    vals = [v for _, v in groups if len(v)]
    top = max(max(v.max() for v in vals) if vals else 1.0, 0.3)
    bins = np.linspace(0, top * 1.05, 45)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, v in groups:
        if len(v):
            ax.hist(v, bins=bins, alpha=0.6, label=f"{name} (n={len(v)})")
    if np.isfinite(thr):
        ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"λ^(2/3) = {thr:.3f}")
    ax.set_xlabel("posterior |β| (mean over iterations where selected)")
    ax.set_ylabel("genes")
    ax.legend(fontsize=9)
    _save(fig, "mechanism.pdf")


def fig_traces(runs):
    runs = [R for R in runs if os.path.exists(os.path.join(R["dir"], "diagnostics.npz"))]
    if not runs:
        return
    fields = [("phi", "dispersion φ"), ("lam", "threshold λ"), ("sigma2", "σ²"), ("n_active", "genes above threshold")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    b = 0
    for R in runs:
        z = np.load(os.path.join(R["dir"], "diagnostics.npz"))
        for ax, (f, t) in zip(axes, fields):
            keys = sorted(k for k in z.files if k.startswith("c0_") and k.endswith("_" + f))
            if keys:
                ax.plot(z[keys[0]], lw=1, label=R["label"])
            ax.set_title(t)
        b = int(R["summ"]["config"]["burn_in"])
    for ax in axes:
        ax.axvline(b, color="0.4", ls="--", lw=1)
        ax.set_xlabel("iteration")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8)
    _save(fig, "traces_paper.pdf")


def fig_paper_design(sim_dir="sim_paper"):
    if not os.path.exists(os.path.join(sim_dir, "bins.csv")):
        print("[assets] sim_paper not found -> no paper_design figure")
        return
    b = pd.read_csv(os.path.join(sim_dir, "bins.csv"))
    with open(os.path.join(sim_dir, "meta.json")) as fh:
        meta = json.load(fh)
    H, W = meta["H"], meta["W"]

    def grid(v):
        img = np.full((H, W), np.nan)
        img[b["row"].to_numpy(), b["col"].to_numpy()] = np.asarray(v, float)
        return img

    wc = [c for c in b.columns if c.startswith("w_")]
    panels = [("region (Dirichlet parameters)", b["region"], "Set2"),
              ("composition: cell type 1", b[wc[0]], "viridis"),
              ("composition: cell type 6", b[wc[-1]], "viridis"),
              ("log10 total counts", np.log10(b["total_counts"] + 1), "magma")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (t, v, cm) in zip(axes, panels):
        im = ax.imshow(grid(v), cmap=cm, interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    _save(fig, "paper_design.pdf")


# ----------------------------------------------------------------------------- numbers about the data
def data_numbers(runs):
    for sim, key in (("sim_out", "fit"), ("sim_paper", "paper")):
        p = os.path.join(sim, "meta.json")
        if os.path.exists(p):
            with open(p) as fh:
                m = json.load(fh)
            put(f"{key}-nbins", int(m["n_bins"]))
            put(f"{key}-ngenes", int(m["n_genes"]))
    if "B" in runs and os.path.exists("sim_out/genes.csv"):
        g = pd.read_csv("sim_out/genes.csv", keep_default_na=False)[["gene", "total_counts"]]
        R = runs["B"]
        tc = pd.to_numeric(R["res"][["gene"]].merge(g, on="gene", how="left")["total_counts"], errors="coerce")
        if R["m"]["nunits"]:
            put("ours-cpg", float(np.nanmedian(tc)) / R["m"]["nunits"], nd=2)
    if os.path.exists("sim_paper/genes.csv") and os.path.exists("sim_paper/meta.json"):
        g = pd.read_csv("sim_paper/genes.csv", keep_default_na=False)
        with open("sim_paper/meta.json") as fh:
            m = json.load(fh)
        put("paper-cpg", float(np.median(pd.to_numeric(g["total_counts"]))) / m["n_bins"], nd=1)
        put("paper-nsv", int(g["is_SV"].astype(str).str.lower().eq("true").sum()))
        c = m["config"]
        scen = (c["pattern"], c["network"], round(float(c["dropout"]), 2))
        put("paper-scen", f"{c['pattern']} pattern, {c['network']} network, dropout {c['dropout']}")
        if scen in PAPER:
            rr, pp, ff = PAPER[scen]
            put("paper-rec", rr)
            put("paper-prec", pp)
            put("paper-f1", ff)
            return f"Paper (Table 1/S3) & on & 5 kernels $\\times$ 5 chains, 50 replicates & -- & " \
                   f"{pp:.2f} & {rr:.2f} & {ff:.2f} & -- & -- \\\\"
    return None


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = {}
    for spec in RUNS:
        R = load_run(spec)
        if R is not None:
            R["m"] = metrics(R)
            runs[spec["key"]] = R
    for k, R in runs.items():
        M = R["m"]
        for f in ("called", "tp", "fp", "fn", "ntrue", "nunits", "ngenes"):
            put(f"{k}-{f}", M[f])
        for f in ("prec", "rec", "f1", "aucpip", "aucbeta", "topk"):
            put(f"{k}-{f}", M[f])
        put(f"{k}-runtime", M["runtime"], nd=1)
        if R["dir"].startswith("fit_paper"):
            h = hub_counts(R)
            put(f"{k}-nullhub", h["hub"])
            put(f"{k}-nnullhub", h["nhub"])
            put(f"{k}-nonhub", h["non"])
            put(f"{k}-nnonhub", h["nnon"])
            lam = lam_post(R)
            put(f"{k}-lam", lam, nd=4)
            put(f"{k}-lam23", lam ** (2 / 3) if np.isfinite(lam) else float("nan"))
            r = R["res"]
            fp_hub = r[~r["is_SV"] & r["is_hub"] & r["SV_called"]]["absb"]
            put(f"{k}-fphub", float(fp_hub.median()) if len(fp_hub) else float("nan"))
        else:
            rate = R["res"].groupby("eval_class")["SV_called"].mean()
            for c, v in rate.items():
                if c in CLASSKEY:
                    put_pct(f"{k}-rate-{CLASSKEY[c]}", v)
            for p, v in zip(PATS, _auc_by_pattern(R, "absb")):
                put(f"{k}-aucb-{p}", v)
            for p, v in zip(PATS, _auc_by_pattern(R, "pip")):
                put(f"{k}-aucp-{p}", v)

    paper_row = None
    try:
        paper_row = data_numbers(runs)
    except Exception:
        traceback.print_exc()

    ours = [runs[k] for k in "ABCDE" if k in runs]
    paper = [runs[k] for k in "FG" if k in runs]
    write_table(ours, "runs_ours.tex")
    write_table(paper, "runs_paper.tex", [paper_row] if paper_row else ())

    _safe(fig_callrate, [runs[k] for k in "ABDE" if k in runs])
    _safe(fig_auc_pattern, [runs[k] for k in "ADE" if k in runs])
    _safe(fig_topk, [("Our synthetic data", [runs[k] for k in "ADE" if k in runs]), ("Paper design", paper)])
    _safe(fig_hub, paper)
    _safe(fig_mechanism, runs.get("F"))
    _safe(fig_traces, paper)
    _safe(fig_paper_design)

    order = ("fit_norho_all", "fit_floor", "fit_dbg", "fit_norho", "fit_out")
    src = copy_first(["sim_full/figures/01_tissue_bin_types.png"]
                     + [f"{d}/figures/01_tissue_bin_types.png" for d in order], "tissue.png")
    copy_first(["sim_full/figures/02_qc_depth_and_detection.png"]
               + [f"{d}/figures/02_qc_depth_and_detection.png" for d in order], "qc.png")
    copy_first(["sim_full/figures/03_truth_examples.png"]
               + [f"{d}/figures/03_truth_examples.png" for d in order], "truth.png")
    copy_first(["fit_paper/figures/05_network_modules.png"], "network_paper.png")
    copy_first(["fit_paper/figures/04_results_summary.png"], "results_paper.png")
    copy_first(["fit_norho_all/figures/06_called_gene_maps.png", "fit_floor/figures/06_called_gene_maps.png"],
               "maps_ours.png")
    meta_src = "sim_full/meta.json" if (src or "").startswith("sim_full") else "sim_out/meta.json"
    if os.path.exists(meta_src):
        with open(meta_src) as fh:
            m = json.load(fh)
        put("fig-nbins", int(m["n_bins"]))
        put("fig-ngenes", int(m["n_genes"]))

    write_numbers()
    if os.path.exists("slides.tex"):
        shutil.copy("slides.tex", os.path.join(OUT, "slides.tex"))
        print("[assets] copied slides.tex -> slides/slides.tex")
    print(f"[assets] runs found: {', '.join(runs) or 'none'}")
    print("[assets] next:  cd slides && pdflatex slides.tex && pdflatex slides.tex")


if __name__ == "__main__":
    main()
