import pandas as pd, numpy as np
r = pd.read_csv("fit_paper/results.csv", keep_default_na=False)
b = lambda c: r[c].astype(str).str.lower().eq("true")
r["sv"], r["called"], r["hub"] = b("is_SV"), b("SV_called"), b("is_hub")
for c in ["module", "pip", "beta1_linear", "beta2_linear"]:
    r[c] = pd.to_numeric(r[c])
r["absb"] = np.maximum(r.beta1_linear.abs(), r.beta2_linear.abs())
fp = r[~r.sv & r.called]
print("false positives:", len(fp), "in", fp.module.nunique(), "of", r[~r.sv].module.nunique(), "null sub-networks")
print(fp.groupby("module").size().sort_values(ascending=False).head(10).to_string())
print("hubs among FP:", int(fp.hub.sum()))
print("median |beta|: FP", round(fp.absb.median(), 3), "| TP", round(r[r.sv & r.called].absb.median(), 3),
      "| uncalled null", round(r[~r.sv & ~r.called].absb.median(), 3))