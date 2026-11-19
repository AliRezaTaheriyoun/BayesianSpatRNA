import pandas as pd, numpy as np
from scipy.stats import rankdata

for d in ["fit_dbg", "fit_norho"]:
    r = pd.read_csv(f"{d}/results.csv", keep_default_na=False)
    r = r[r["is_ctSV"].astype(str) != "True"]
    y = r["is_SV"].astype(str).eq("True").to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    auc = lambda s: (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    absb = np.maximum(r["beta1_linear"].abs(), r["beta2_linear"].abs()).to_numpy(float)
    print(d, "AUROC pip:", round(auc(r["pip"].to_numpy(float)), 3),
          " AUROC |beta|:", round(auc(absb), 3))