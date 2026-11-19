import pandas as pd, numpy as np
from scipy.stats import rankdata

r = pd.read_csv("fit_norho/results.csv", keep_default_na=False)
r = r[r["is_ctSV"].astype(str) != "True"]
sv = r["is_SV"].astype(str).eq("True")
absb = np.maximum(r["beta1_linear"].abs(), r["beta2_linear"].abs()).to_numpy(float)
pip = r["pip"].to_numpy(float)
for pat in ["linear", "gaussian", "periodic", "hotspot"]:
    m = (r["eval_class"] == f"SV:{pat}") | ~sv
    y = (r["eval_class"] == f"SV:{pat}")[m].to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    auc = lambda s: (rankdata(s[m.to_numpy()])[y].sum() - n1*(n1+1)/2) / (n1*n0)
    print(f"{pat:9s} n={n1:3d}  AUROC pip={auc(pip):.3f}  |beta|={auc(absb):.3f}")