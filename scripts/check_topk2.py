import sys, pandas as pd, numpy as np
d = sys.argv[1]
r = pd.read_csv(f"{d}/results.csv", keep_default_na=False)
r = r[r["is_ctSV"].astype(str) != "True"].reset_index(drop=True)
ks = ["linear", "gauss40", "gauss60", "cos40", "cos60"]
num = lambda c: pd.to_numeric(r[c], errors="coerce").abs().to_numpy(float)
s = np.max([np.maximum(num(f"beta1_{k}"), num(f"beta2_{k}")) for k in ks], axis=0)
sv = r["is_SV"].astype(str).eq("True").to_numpy()
o = np.argsort(-s)
print(d)
for k in (25, 50, 100, 132):
    print(f"  top {k:3d}: precision = {sv[o[:k]].mean():.2f}, found {sv[o[:k]].sum()} of {sv.sum()}")