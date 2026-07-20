"""Exact symm1-vs-symm0 deltas from the as-shipped pipeline npzs."""
import numpy as np

h1 = list(np.load("sqdship_symm1.npz", allow_pickle=True)["hist"])
h0 = list(np.load("sqdship_symm0.npz", allow_pickle=True)["hist"])
print("it b | dE(symm1-symm0) uHa |  dS2      | dims symm1 / symm0")
for a, b in zip(h1, h0):
    assert a["iter"] == b["iter"] and a["batch"] == b["batch"]
    de = (a["e_tot"] - b["e_tot"]) * 1e6
    ds = a["s2"] - b["s2"]
    d1 = a["dim_a"] * a["dim_b"]
    d0 = b["dim_a"] * b["dim_b"]
    print(f"{a['iter']}  {a['batch']} | {de:+12.3f}        | {ds:+.2e} "
          f"| {d1:>9,} / {d0:>9,} ({d1/d0:.2f}x)")
b1 = np.load("sqdship_symm1.npz", allow_pickle=True)
b0 = np.load("sqdship_symm0.npz", allow_pickle=True)
print(f"BEST dE {float(b1['best_e']-b0['best_e'])*1e6:+.3f} uHa | "
      f"dS2 {float(b1['best_s2']-b0['best_s2']):+.2e}")
print(f"BEST symm1: E {float(b1['best_e']):.8f} S2 {float(b1['best_s2']):.5f}")
print(f"BEST symm0: E {float(b0['best_e']):.8f} S2 {float(b0['best_s2']):.5f}")
