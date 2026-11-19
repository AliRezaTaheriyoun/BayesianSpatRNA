import pandas as pd, numpy as np
from scipy.stats import rankdata

r = pd.read_csv("fit_norho_all/results.csv", keep_default_na=False)
r = r[r["is_ctSV"].astype(str) != "True"].reset_index(drop=True)
ks = ["linear", "gauss40", "gauss60", "cos40", "cos60"]
num = lambda c: pd.to_numeric(r[c], errors="coerce").abs().to_numpy(float)
absb = np.max([np.maximum(num(f"beta1_{k}"), num(f"beta2_{k}")) for k in ks], axis=0)
pip = pd.to_numeric(r["pip"]).to_numpy(float)
sv = r["is_SV"].astype(str).eq("True").to_numpy()
for pat in ["linear", "gaussian", "periodic", "hotspot", "ALL"]:
    pos = sv if pat == "ALL" else (r["eval_class"] == f"SV:{pat}").to_numpy()
    m = pos | ~sv
    y = pos[m]; n1, n0 = y.sum(), (~y).sum()
    auc = lambda s: (rankdata(s[m])[y].sum() - n1*(n1+1)/2) / (n1*n0)
    print(f"{pat:9s} n={n1:3d}  AUROC pip={auc(pip):.3f}  max|beta|={auc(absb):.3f}")