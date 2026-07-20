"""Formalize the sampling wall (gap-2 hardening): the unique-determinant
yield of ANY fixed sampling distribution is E[U(M)] = sum_i 1-(1-p_i)^M.
Instantiate it with the MEASURED distributions from this study, validate the
closed form against the actually-sampled unique counts, and report the local
exchange rate gamma = dlogU/dlogM at each operating point ("10x more shots
buys only 10^gamma more configurations").

Distributions:
  soup   -- the converged [2Fe-2S] aufbau-HCI 170k-det vector (sqdship_vec.npz),
            validated against the measured 38,118 uniques @ 1e6 shots
            (sqdship_counts.npz)
  lucj   -- the exact noiseless LUCJ(fe2s2) distribution's coupon curve
            (lucj_stats.npz, written by _lucjfe.py stage L2), validated
            against its own sampled uniq
  n2     -- empirical (shots, uniq) pairs from the N2 QSCI leg (n2_qsci.npz)
            + slope between successive shot decades

Output: couponlaw.npz {dist: (mgrid, eu)} + printed table. Gate: closed-form
E[U] must match measured uniq within 3% wherever a measurement exists.
Sentinel: COUPONLAW_DONE (exit 1 on gate failure)
"""
import os
import sys

import numpy as np

FAILS = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def eu_curve(p, mgrid):
    p = np.asarray(p, float)
    p = p[p > 0]
    p = p / p.sum()
    lp = np.log1p(-np.minimum(p, 1 - 1e-16))
    return np.array([len(p) - np.exp(M * lp).sum() for M in mgrid])


def slope(mgrid, eu, m0):
    lm, lu = np.log(mgrid), np.log(np.maximum(eu, 1e-12))
    j = np.searchsorted(mgrid, m0)
    j = min(max(j, 1), len(mgrid) - 2)
    return (lu[j + 1] - lu[j - 1]) / (lm[j + 1] - lm[j - 1])


mgrid = np.unique(np.round(np.logspace(3, 10, 36)).astype(np.int64))
out = {}

# ---- soup distribution -------------------------------------------------------
d = np.load("sqdship_vec.npz")
p = d["c"] ** 2
eu = eu_curve(p, mgrid)
out["soup_mgrid"], out["soup_eu"] = mgrid, eu
meas = len(np.load("sqdship_counts.npz")["a"])
pred = float(np.interp(np.log(1e6), np.log(mgrid), eu))
gate("soup: closed form vs measured uniq @1e6",
     abs(pred - meas) / meas < 0.03, f"pred {pred:,.0f} vs meas {meas:,}")
g = slope(mgrid, eu, 1_000_000)
sat = eu[-1] / len(p[p > 0])
print(f"soup: support {len(p):,}; gamma@1e6 = {g:.3f}; "
      f"E[U] @1e6/1e7/1e8 = {np.interp(np.log([1e6,1e7,1e8]), np.log(mgrid), eu)}")
out["soup_gamma"] = g

# ---- LUCJ distribution (if the big job has finished stage L2) ----------------
if os.path.exists("lucj_stats.npz"):
    s = np.load("lucj_stats.npz")
    out["lucj_mgrid"], out["lucj_eu"] = s["mgrid"], s["eu"]
    shots, nuniq = int(s["shots"]), int(s["nuniq"])
    pred = float(np.interp(np.log(shots), np.log(s["mgrid"]), s["eu"]))
    gate("lucj: closed form vs measured uniq",
         abs(pred - nuniq) / nuniq < 0.03,
         f"pred {pred:,.0f} vs meas {nuniq:,} @ {shots:.0e}")
    g = slope(s["mgrid"], s["eu"], shots)
    print(f"lucj: <S2>vec {float(s['s2vec']):.4f}; gamma@{shots:.0e} = {g:.3f}")
    out["lucj_gamma"] = g
else:
    print("lucj_stats.npz not present yet -- skipped (rerun after _lucjfe)")

# ---- N2 empirical (shots, uniq) + closed form from the sampled HCI vector ----
for tag, qpath, hpath in [("n2eq", "n2_qsci.npz", "n2_hci.npz"),
                          ("n2r200", "n2_r200/n2_qsci.npz",
                           "n2_r200/n2_hci.npz")]:
    if not os.path.exists(qpath):
        print(f"{tag}: {qpath} missing -- skipped")
        continue
    rows = np.asarray(np.load(qpath, allow_pickle=True)["rows"], float)
    rows = rows.reshape(-1, 4)          # (N, uniq, ndets, E)
    N, uniq = rows[:, 0], rows[:, 1]
    o = np.argsort(N)
    N, uniq = N[o], uniq[o]
    gs = np.diff(np.log(uniq)) / np.diff(np.log(N))
    print(f"{tag}: shots {N.astype(int).tolist()}")
    print(f"{tag}: uniq  {uniq.astype(int).tolist()}")
    print(f"{tag}: gamma between decades = {np.round(gs, 3).tolist()}")
    out[f"{tag}_shots"], out[f"{tag}_uniq"] = N, uniq
    out[f"{tag}_gamma"] = gs
    if os.path.exists(hpath):
        h = np.load(hpath, allow_pickle=True)
        for vec_tag, key in [("deep", "deep_coef"), ("ck", "ck_coef")]:
            eu = eu_curve(np.asarray(h[key]) ** 2, mgrid)
            pred = np.interp(np.log(N), np.log(mgrid), eu)
            rel = np.abs(pred - uniq) / uniq
            print(f"{tag}/{vec_tag}: closed-form pred "
                  f"{np.round(pred).astype(int).tolist()} "
                  f"(rel dev {np.round(rel, 3).tolist()})")
            if rel.max() < 0.03:
                gate(f"{tag}: closed form ({vec_tag} vector) vs measured "
                     f"uniq at all {len(N)} shot counts", True,
                     f"max dev {rel.max():.3f}")
                out[f"{tag}_mgrid"], out[f"{tag}_eu"] = mgrid, eu
                break
        else:
            gate(f"{tag}: closed form matches some sampled vector",
                 False, "neither deep nor ck within 3%")

np.savez("couponlaw.npz", **out)
print(f"\n{len(FAILS)} gate failures", flush=True)
print("COUPONLAW_DONE", flush=True)
sys.exit(1 if FAILS else 0)
