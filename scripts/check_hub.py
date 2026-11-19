import sys, numpy as np, pandas as pd
d = sys.argv[1]
r = pd.read_csv(f"{d}/results.csv", keep_default_na=False)
b = lambda c: r[c].astype(str).str.lower().eq("true")
r["sv"], r["called"], r["hub"] = b("is_SV"), b("SV_called"), b("is_hub")
for c in ["pip", "beta1_linear", "beta2_linear"]:
    r[c] = pd.to_numeric(r[c])
r["absb"] = np.maximum(r.beta1_linear.abs(), r.beta2_linear.abs())
z = np.load(f"{d}/diagnostics.npz")
lam = z[[k for k in z.files if k.endswith("_lam")][0]][500:].mean()
print("lambda:", round(lam, 4), "-> lambda^(2/3) =", round(lam ** (2 / 3), 4))
nh = r[~r.sv & r.hub]
print("null hubs called:", int(nh.called.sum()), "of", len(nh),
      "| median |beta| of called null hubs:", round(nh[nh.called].absb.median(), 3))
print("null non-hub genes called:", int((~r.sv & ~r.hub & r.called).sum()), "of", int((~r.sv & ~r.hub).sum()))
print("true hubs called:", int((r.sv & r.hub & r.called).sum()), "of", int((r.sv & r.hub).sum()))