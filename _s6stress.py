"""LP sensitivity stress test (review round 16, section 3G): inflate the
per-moment uncertainty boxes 10x and 100x and re-derive the sector-weight
bounds for all eight audited states.

The production LP (_s6moments.py lp_bounds) constrains each measured moment
m_k inside a box m_k +/- tol_k with tol_k = 1e-7 * max(1, |m_k|) — i.e. the
tolerance IS a per-moment uncertainty interval, and the bounds are an
uncertainty propagation through the sector-decomposition constraints. This
script re-runs the identical LP (function copied verbatim, one `infl`
parameter added) at inflation factors f in {1, 10, 100, 1000} on the moments
stored in s6moments.npz. Gate: f = 1 must reproduce the stored bounds.

Headline: the set-draw (symm on) triplet ceiling w1_max and the [4Fe-4S]
(symm on) triplet floor w1_min vs f.

Out: s6stress.npz  Sentinel: S6STRESS_DONE (exit 1 on any gate failure)
"""
import sys

import numpy as np
from scipy.optimize import linprog


def lp_bounds(moms, smax, infl=1.0):
    """Verbatim from _s6moments.py lp_bounds, with the per-moment tolerance
    box inflated by `infl` (infl = 1 -> the production LP)."""
    lam = np.array([S * (S + 1) for S in range(smax + 1)], float)
    rows, lo, hi = [np.ones_like(lam)], [1.0], [1.0]
    for k, m in enumerate(moms, start=1):
        if np.isnan(m):
            continue
        tol = infl * 1e-7 * max(1.0, abs(m))
        rows.append(lam ** k)
        lo.append(m - tol)
        hi.append(m + tol)
    Aub = np.vstack([np.vstack(rows), -np.vstack(rows)])
    bub = np.concatenate([np.array(hi), -np.array(lo)])
    out, ok = {}, True
    for name, cvec in [("w0", np.eye(smax + 1)[0]),
                       ("w1", np.eye(smax + 1)[1]),
                       ("wge3", (lam >= 12) * 1.0)]:
        pair = []
        for sgn in (1.0, -1.0):
            r = linprog(sgn * cvec, A_ub=Aub, b_ub=bub,
                        bounds=[(0, 1)] * (smax + 1), method="highs")
            ok &= bool(r.success)
            pair.append(sgn * r.fun if r.success else np.nan)
        out[name] = (pair[0], pair[1])  # (min, max)
    return out, ok


d = np.load("s6moments.npz", allow_pickle=True)
labels = [str(x) for x in d["labels"]]
files = [str(x) for x in d["files"]]
M = np.stack([np.asarray(d[k], float) for k in ("m1", "m2", "m3", "m4")], 1)
smax = np.asarray(d["smax"], int)
stored = {"w0": np.asarray(d["w0_bounds"], float),
          "w1": np.asarray(d["w1_bounds"], float),
          "wge3": np.asarray(d["wge3_bounds"], float)}

FACT = [1.0, 10.0, 100.0, 1000.0]
fails = []
res = np.full((len(labels), len(FACT), 3, 2), np.nan)  # state, f, {w0,w1,wge3}, {lo,hi}

for i, lab in enumerate(labels):
    moms = M[i]
    for j, f in enumerate(FACT):
        out, ok = lp_bounds(moms, int(smax[i]), infl=f)
        if not ok:
            fails.append(f"{lab}: LP failed at infl {f}")
        for q, name in enumerate(("w0", "w1", "wge3")):
            res[i, j, q] = out[name]
    # gate: f = 1 reproduces the stored production bounds
    for q, name in enumerate(("w0", "w1", "wge3")):
        dmax = np.abs(res[i, 0, q] - stored[name][i]).max()
        if not (dmax < 1e-9):
            fails.append(f"{lab}: f=1 {name} bound mismatch {dmax:.2e}")

print("state | factor | w0 [lo,hi] | w1 [lo,hi] | w>=3 [lo,hi]")
for i, lab in enumerate(labels):
    for j, f in enumerate(FACT):
        w0, w1, wg = res[i, j]
        print(f"{lab:32s} x{int(f):<5d} "
              f"w0 [{w0[0]:.6f},{w0[1]:.6f}]  "
              f"w1 [{w1[0]:.2e},{w1[1]:.2e}]  "
              f"w>=3 [{wg[0]:.2e},{wg[1]:.2e}]")

i_sd = labels.index("[2Fe-2S] set-draw, symm on")
i_f4 = labels.index("[4Fe-4S] hardware, symm on")
print("\nHEADLINE set-draw-on triplet ceiling w1_max: "
      + ", ".join(f"x{int(f)}: {res[i_sd, j, 1, 1]:.3e}"
                  for j, f in enumerate(FACT)))
print("HEADLINE [4Fe-4S]-on triplet floor w1_min:   "
      + ", ".join(f"x{int(f)}: {res[i_f4, j, 1, 0]:.6f}"
                  for j, f in enumerate(FACT)))
print("HEADLINE set-draw-on singlet window at x100: "
      f"[{res[i_sd, 2, 0, 0]:.4f}, {res[i_sd, 2, 0, 1]:.4f}]")

np.savez_compressed("s6stress.npz",
                    labels=np.array(labels), files=np.array(files),
                    factors=np.array(FACT), bounds=res,
                    quantities=np.array(["w0", "w1", "wge3"]))
print(f"saved s6stress.npz ({len(labels)} states x {len(FACT)} factors)")
if fails:
    print("FAILS:", fails)
print("S6STRESS_DONE", flush=True)
sys.exit(1 if fails else 0)
