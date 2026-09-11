"""Adversarial number audit: every quantitative claim in paper_sqd_spin_audit.md
re-derived from the raw on-disk archives. PASS/FAIL per claim; exit 1 on any FAIL.
"""
import os
import sys
from math import comb

import numpy as np

FAILS = []
N_CHECKS = [0]


def check(name, ok, detail=""):
    N_CHECKS[0] += 1
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- manuscript sources: main file + optional Supporting Information ----
# For journal submission the manuscript splits into a main file and an SI file.
# Prose pins run against the concatenation: a required phrase counts as present
# if it survives anywhere in the manuscript, and a banned phrase must be absent
# from both halves (x not in main+SI is exactly x not in main and x not in SI).
# Gates about the main file's own layout -- the abstract in particular -- read
# _MD_MAIN instead. With no SI file on disk the concatenation is the main file,
# so an unsplit manuscript audits exactly as it did before.
_SI_JOIN = "\n\n%%AUDIT-SI-BOUNDARY%%\n\n"


def _read_ms(main, si):
    _m = open(main, encoding="utf-8").read()
    _s = ""
    if os.path.exists(si):
        _s = open(si, encoding="utf-8").read()
    return _m, _s, (_m + _SI_JOIN + _s) if _s else _m


# The four filenames are overridable so one audit engine serves both the frozen
# preprint (the default: the version of record on ChemRxiv) and a journal
# main+SI pair, without either version having to overwrite the other.
_MS_MD = os.environ.get("AUDIT_MD", "paper_sqd_spin_audit.md")
_MS_TX = os.environ.get("AUDIT_TEX", "paper_sqd_spin_audit.tex")
_MD_MAIN, _MD_SI, _MD_ALL = _read_ms(
    _MS_MD, os.environ.get("AUDIT_MD_SI", _MS_MD[:-3] + "_si.md"))
_TX_MAIN, _TX_SI, _TX_ALL = _read_ms(
    _MS_TX, os.environ.get("AUDIT_TEX_SI", _MS_TX[:-4] + "_si.tex"))
print(f"   sources: {_MS_MD} / {_MS_TX}")
print(f"   manuscript: main {len(_MD_MAIN.split())} words"
      + (f" + SI {len(_MD_SI.split())} words" if _MD_SI else " (no SI file)"))


# ============ A. flagship tables (paper 2.4 / 2.5) ============
E2 = -116.6056091
E4 = -327.2396369
PAPER = {
 "fe2s2bs": [(1011, -116.456202, 149.4, 4.978, 4.977),
             (25734, -116.569067, 36.5, 4.893, 4.921),
             (65124, -116.584165, 21.4, 4.779, 4.863),
             (155030, -116.592307, 13.3, 4.659, 4.809)],
 "fe2s2":   [(1003, -116.060455, 545.2, 3.033, 3.015),
             (5053, -116.155646, 450.0, 2.050, 2.073),
             (24570, -116.430693, 174.9, 4.672, 4.461),
             (68825, -116.579203, 26.4, 4.824, 4.876)],
 "fe2s2deep": [(170448, -116.591986, 13.6, 4.669, 4.810)],
 "fe4s4":  [(1106, -326.184317, 1055.3, 5.293, 5.270),
            (5905, -326.385018, 854.6, 6.053, 5.914),
            (17993, -326.445889, 793.7, 6.464, 6.302),
            (40435, -326.484822, 754.8, 7.004, 9.284)],
}
maxsymm = {}
for tag, want in PAPER.items():
    ref = E4 if tag == "fe4s4" else E2
    rs = {r["ndets"]: r for r in
          np.load(f"s2audit_{tag}.npz", allow_pickle=True)["results"]}
    for nd, e0, err, s0, s1 in want:
        r = rs.get(nd)
        if r is None:
            check(f"{tag} D={nd} row exists", False)
            continue
        ok = (close(r["e_pyci"][0], e0, 5e-7)
              and close((r["e_pyci"][0] - ref) * 1e3, err, 0.05)
              and close(r["s2_pyci"][0], s0, 5e-4)
              and close(r["s2_pyci"][1], s1, 5e-4))
        check(f"{tag} D={nd} table row", ok,
              f"E {r['e_pyci'][0]:.6f} S2 {r['s2_pyci'][0]:.3f}/{r['s2_pyci'][1]:.3f}")
    # B. SYMM no-op at EVERY checkpoint of this arm
    for r in rs.values():
        if "e_symm" not in r:
            continue
        de = abs(r["e_symm"][0] - r["e_pyci"][0])
        check(f"{tag} D={r['ndets']} symm dE0<1uHa", de < 1e-6, f"{de*1e6:.3f} uHa")
        maxsymm[tag] = max(maxsymm.get(tag, 0), int(r["ndets_symm"]))
        G = np.asarray(r["g_symm"]); es = np.asarray(r["e_symm"])
        if len(es) > 1 and es[1] - es[0] < 2e-6:
            ev = np.linalg.eigvalsh(G[:2, :2])
            span_ok = (close(ev[0], r["s2_pyci"][0], 2e-3)
                       and close(ev[1], r["s2_pyci"][0], 2e-3))
            check(f"{tag} D={r['ndets']} S2-block ~ identity*soup", span_ok,
                  f"span [{ev[0]:.3f},{ev[1]:.3f}] vs r0 {r['s2_pyci'][0]:.3f}")
check("symm reach: fe2s2deep 319,189 dets", maxsymm.get("fe2s2deep") == 319189,
      str(maxsymm.get("fe2s2deep")))
check("symm reach: fe2s2bs 310,058", maxsymm.get("fe2s2bs") == 310058,
      str(maxsymm.get("fe2s2bs")))
check("symm reach: fe4s4 80,755", maxsymm.get("fe4s4") == 80755,
      str(maxsymm.get("fe4s4")))

# ============ C. as-shipped (paper 2.6) ============
h1 = list(np.load("sqdship_symm1.npz", allow_pickle=True)["hist"])
h0 = list(np.load("sqdship_symm0.npz", allow_pickle=True)["hist"])
b1 = np.load("sqdship_symm1.npz", allow_pickle=True)
b0 = np.load("sqdship_symm0.npz", allow_pickle=True)
# paper table = per-iteration batch MEANS (symm on)
want_it = {1: (59000, -116.513, 92.8, 4.90), 2: (114000, -116.542, 63.3, 4.87),
           3: (158000, -116.553, 52.6, 4.85), 4: (193000, -116.560, 45.7, 4.837)}
for it, (wd, we, werr, ws) in want_it.items():
    rows = [r for r in h1 if r["iter"] == it]
    md = np.mean([r["dim_a"] * r["dim_b"] for r in rows])
    me = np.mean([r["e_tot"] for r in rows])
    ms = np.mean([r["s2"] for r in rows])
    ok = (close(md, wd, 1500) and close(me, we, 5e-4)
          and close((me - E2) * 1e3, werr, 0.5) and close(ms, ws, 5e-3))
    check(f"sqdship it{it} mean row", ok,
          f"dim {md:,.0f} E {me:.6f} err {(me-E2)*1e3:+.2f} S2 {ms:.3f}")
check("sqdship best E +45.5 mHa", close((float(b1["best_e"]) - E2) * 1e3, 45.502, 0.01),
      f"{(float(b1['best_e'])-E2)*1e3:+.3f}")
check("sqdship best S2 4.834", close(float(b1["best_s2"]), 4.834, 5e-4))
for a, b in zip(h1, h0):
    if abs(a["e_tot"] - b["e_tot"]) > 1e-9 or abs(a["s2"] - b["s2"]) > 1e-9:
        check("symm1==symm0 per batch (<1nHa)", False,
              f"it{a['iter']} b{a['batch']}")
        break
else:
    check("symm1==symm0 per batch (<1nHa, dS2<1e-9)", True)
r41 = [r for r in h1 if r["iter"] == 4]
r40 = [r for r in h0 if r["iter"] == 4]
ratios = [(a["dim_a"] * a["dim_b"]) / (b["dim_a"] * b["dim_b"])
          for a, b in zip(r41, r40)]
check("symm dim ratio 4.00x @it4", all(close(x, 4.0, 0.01) for x in ratios),
      f"{[f'{x:.3f}' for x in ratios]}")
check("it4 dims 194,481 vs 48,600 (b0)",
      r41[0]["dim_a"] * r41[0]["dim_b"] == 194481
      and r40[0]["dim_a"] * r40[0]["dim_b"] == 48600)
cnt = np.load("sqdship_counts.npz")
check("1M shots -> 38,118 uniq", len(cnt["n"]) == 38118 and cnt["n"].sum() == 1000000,
      f"{len(cnt['n'])} uniq / {cnt['n'].sum()} shots")
vec = np.load("sqdship_vec.npz")
check("stage1 vec E -116.591986", close(float(vec["e"]), -116.591986, 5e-7))
ratio = (float(b1["best_e"]) - E2) / (float(vec["e"]) - E2)
check("matched-cost inversion 3.3x", close(ratio, 3.34, 0.05), f"{ratio:.2f}x")

# ============ D. diazene round-3 spin-clean (paper 2.1-2.2) ============
def arm_rows(res, key):
    a = np.asarray(res[key], float).ravel()
    assert len(a) % 7 == 0
    return a.reshape(7, -1)  # D grid 64..4096

for f, label in [("fairfight_dzsinglet.npz", "seed7"),
                 ("fairfight_dzs11.npz", "seed11"),
                 ("fairfight_dzs23.npz", "seed23")]:
    rows = list(np.load(f, allow_pickle=True)["results"])
    r090 = next(r for r in rows if "r090" in r["file"])["results"]
    cip = arm_rows(r090, "cipsi_hf")[-1]   # D=4096
    exa = arm_rows(r090, "exact")[-1]
    check(f"{label} r090@4096 CIPSI beats exact-sampling",
          cip[1] < exa[1], f"{cip[1]:+.2f} vs {exa[1]:+.2f} mHa")
    if label == "seed7":
        check("r090@4096 CIPSI +0.71", close(cip[1], 0.71, 0.02), f"{cip[1]:+.3f}")
        check("r090@4096 exact +1.93", close(exa[1], 1.93, 0.02), f"{exa[1]:+.3f}")
        lucj = arm_rows(r090, "lucj")[-1]
        lopt = arm_rows(r090, "lucj_opt")[-1]
        check("r090 LUCJ +64 mHa, 181 uniq",
              close(lucj[1], 64.3, 1.0) and int(lucj[5]) == 181,
              f"{lucj[1]:+.1f} mHa, {int(lucj[5])} uniq")
        check("r090 LUCJ-opt +35 mHa (50x CIPSI)",
              close(lopt[1], 35.3, 1.0) and lopt[1] / cip[1] > 40,
              f"{lopt[1]:+.1f} = {lopt[1]/cip[1]:.0f}x CIPSI")
        check("r090 exact-sampling 2,869 uniq", int(exa[5]) == 2869, f"{int(exa[5])}")
        i180 = next(r for r in rows if "i180" in r["file"])["results"]
        ci = arm_rows(i180, "cipsi_hf")[-1]
        check("i180@4096 CIPSI +0.29 (converged control)",
              close(ci[1], 0.285, 0.02), f"{ci[1]:+.4f}")
        li = arm_rows(i180, "lucj")[-1]
        check("i180 LUCJ 345 uniq", int(li[5]) == 345, f"{int(li[5])}")

# round-1 narrative (paper 2.2): classical trap ~+21 @4096; exact escapes
rows1 = list(np.load("fairfight_dz.npz", allow_pickle=True)["results"])
r090a = next(r for r in rows1 if "r090" in r["file"])["results"]
cl = [arm_rows(r090a, k)[-1][1] for k in ("cipsi_hf", "hci_hf", "hci_cisd")]
check("round1 classical all trapped ~+21 @4096",
      all(close(x, 21, 2.5) for x in cl), f"{[f'{x:+.1f}' for x in cl]}")
ex1 = arm_rows(r090a, "exact")
print(f"   round1 exact trajectory (D, Evar-FCI): "
      f"{[(int(r[0]), round(r[1], 2)) for r in ex1]}")
check("round1 exact escapes trap (@4096 << classical)",
      ex1[-1][1] < 3.0 and cl[0] > 20, f"exact @4096 {ex1[-1][1]:+.2f}")

# ============ F. N2 cross-validation (paper 2.1) ============
def hci_interp(hrows, nd):
    """log-ndets piecewise-linear interpolation of the HCI ladder energy."""
    h = hrows[np.argsort(hrows[:, 1])]
    return float(np.interp(np.log(nd), np.log(h[:, 1]), h[:, 2]))

q = np.asarray(np.load("n2_qsci.npz")["rows"], float)
print(f"   n2 qsci rows (N, uniq, ndets, E): "
      f"{[(f'{r[0]:.0e}', int(r[1]), int(r[2]), round(r[3], 6)) for r in q]}")
r10 = [q[i + 1][2] / q[i][2] for i in range(len(q) - 1)]
check("N2 10x samples -> only ~2.5-3.6x dets",
      all(2.0 < x < 4.0 for x in r10), f"{[f'{x:.2f}' for x in r10]}")
hci = np.asarray(np.load("n2_hci.npz")["rows"], float)
gaps = [(e - hci_interp(hci, nd)) * 1e3 for N, uq, nd, e in q]
check("N2 QSCI ~ HCI at matched dets (parity, no sampling win)",
      all(g > -0.6 for g in gaps), f"gaps {[f'{g:+.2f}' for g in gaps]} mHa")
q2 = np.asarray(np.load("n2_r200/n2_qsci.npz")["rows"], float)
h2 = np.asarray(np.load("n2_r200/n2_hci.npz")["rows"], float)
edges = [(e - hci_interp(h2, nd)) * 1e3 for N, uq, nd, e in q2]
check("N2 R=2.0 nominal edge ~ -0.9 mHa (proxy artifact per paper)",
      -1.3 < min(edges) < -0.4, f"edges {[f'{g:+.2f}' for g in edges]} mHa")

# ============ G. arithmetic ============
s2 = comb(20, 15) ** 2
s4 = comb(36, 27) ** 2
check("fe2s2 sector 240,374,016 (2.4e8)", s2 == 240374016, f"{s2:,}")
check("fe4s4 sector ~8.8e15", close(s4, 8.86e15, 0.03e15), f"{s4:.3e}")
check("155k = 0.065% of fe2s2 sector", close(155030 / s2 * 100, 0.065, 0.002),
      f"{155030/s2*100:.4f}%")
check("40k ~ 5e-12 of fe4s4 sector", close(40435 / s4, 4.6e-12, 1e-12),
      f"{40435/s4:.2e}")

# ============ H. coupon law (paper: sampling-wall section) ============
def eu_at(p, M):
    p = np.asarray(p, float)
    p = p[p > 0]
    p = p / p.sum()
    return float(len(p) - np.exp(M * np.log1p(-np.minimum(p, 1 - 1e-16))).sum())


pv = np.load("sqdship_vec.npz")["c"] ** 2
pred6 = eu_at(pv, 1_000_000)
meas6 = len(np.load("sqdship_counts.npz")["a"])
check("coupon: soup closed form vs measured @1e6 (<3%)",
      abs(pred6 - meas6) / meas6 < 0.03, f"{pred6:,.0f} vs {meas6:,}")
pred8 = eu_at(pv, 100_000_000)
check("coupon: soup E[U]@1e8 ~131,667 of 170,448",
      close(pred8, 131667, 700), f"{pred8:,.0f}")
cl_ = np.load("couponlaw.npz", allow_pickle=True)
check("coupon: soup gamma@1e6 ~0.44",
      close(float(cl_["soup_gamma"]), 0.441, 0.02),
      f"{float(cl_['soup_gamma']):.3f}")
for tag, hp in [("n2eq", "n2_hci.npz"), ("n2r200", "n2_r200/n2_hci.npz")]:
    hh = np.load(hp, allow_pickle=True)
    N_ = cl_[f"{tag}_shots"]
    uq_ = cl_[f"{tag}_uniq"]
    dev = max(abs(eu_at(np.asarray(hh["deep_coef"]) ** 2, M) - u) / u
              for M, u in zip(N_, uq_))
    check(f"coupon: {tag} closed form (deep vector) <3% at all shots",
          dev < 0.03, f"max dev {dev:.3f}")

# ============ I. cost model (paper: price-independence) ============
cm = np.load("costmodel.npz", allow_pickle=True)
dz_gaps = [float(cm[k]) for k in cm.files if k.startswith("fairfight_dz")]
check("cost: sampled worse at matched D on all 6 point-seed combos",
      len(dz_gaps) == 6 and all(1.0 < g < 1.4 for g in dz_gaps),
      f"{[f'{g:+.2f}' for g in dz_gaps]} mHa")
infl = [float(cm[k]) for k in cm.files if k.startswith(("n2eq:", "n2r200:"))]
check("cost: N2 det inflation at matched accuracy in [0.85, 1.40] (parity)",
      all(0.85 <= x <= 1.40 for x in infl),
      f"{[f'{x:.2f}' for x in infl]}")

# ============ J. their own archive (zenodo.15324153) ============
IBB = "_ibm_data/sqd_data_repository-main"


def their_txt(path, skip=2):
    r = np.loadtxt(path, skiprows=skip)
    return r.reshape(-1, r.shape[-1])


E2_THEIR, E4_THEIR = -116.6056091, -327.2396369
check("their fe2s2 DMRG reference == our exact reference (basis-invariant)",
      True, f"{E2_THEIR} (identical by construction; their "
      f"classical_methods_energies.txt line)")
for L, stem, want in [("2Fe-2S", "Fe2S2", (20, 30)),
                      ("4Fe-4S", "Fe4S4", (36, 54))]:
    with open(f"{IBB}/integrals/{L}/fcidump_{stem}_MO.txt") as fh:
        head = fh.readline()
    flat = head.replace(" ", "")
    ok = f"NORB={want[0]}," in flat and f"NELEC={want[1]}," in flat
    check(f"their {L} FCIDUMP header NORB={want[0]} NELEC={want[1]}", ok,
          head.strip()[:40])
hwA = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_hardware_energetics/"
                "energy-variance_data_SQD_eigenstate_A.txt")
unA = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_on_uniform_distribution/"
                "energy-variance_data_SQD_eigenstate_A_uniform.txt")
dA = (hwA[:, 1].min() - unA[:, 1].min()) * 1e3
check("their fe2s2 eigenstate A: uniform beats hardware by ~1.9 mHa",
      close(dA, 1.90, 0.05), f"{dA:+.2f} mHa")
hwB = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_hardware_energetics/"
                "energy-variance_data_SQD_eigenstate_B.txt")
unB = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_on_uniform_distribution/"
                "energy-variance_data_SQD_eigenstate_B_uniform.txt")
dB = (hwB[:, 1].min() - unB[:, 1].min()) * 1e3
check("their fe2s2 eigenstate B: uniform beats hardware by ~12.2 mHa",
      close(dB, 12.17, 0.1), f"{dB:+.2f} mHa")
hwC = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_hardware_energetics/"
                "energy-variance_data_SQD_eigenstate_C.txt")
unC = their_txt(f"{IBB}/experiments/2Fe-2S/sqd_on_uniform_distribution/"
                "energy-variance_data_SQD_eigenstate_C_uniform.txt")


# evidence structure, verified row-by-row BEFORE counting (paper 2.7): the
# B hardware file is a strict subset of C's, the B/C uniform files are
# byte-identical (md5 check below), and A shares no rows with C -> TWO
# independent streams, A and B/C (hardware = C = B-union-C). An earlier
# draft counted the three files as 13 per-file points; corrected to 9
# independent comparisons (Appendix A event #6).
def _rs(r):
    return set(map(tuple, r))


check("archive structure: hw B strict subset of hw C (C adds 7 rows)",
      _rs(hwB) < _rs(hwC) and len(_rs(hwC) - _rs(hwB)) == 7,
      f"C extra {len(_rs(hwC) - _rs(hwB))}")
check("archive structure: hw A row-disjoint from hw C",
      not (_rs(hwA) & _rs(hwC)))


def _stream_stats(hw, un):
    dims = sorted(set(hw[:, 2]) & set(un[:, 2]))
    out = []
    for d in dims:
        h = hw[np.isclose(hw[:, 2], d), 1]
        u = un[np.isclose(un[:, 2], d), 1]
        out.append(((u.mean() - h.mean()) * 1e3,
                    h.std(ddof=1) * 1e3, u.std(ddof=1) * 1e3, u, h))
    return out


_gA = _stream_stats(hwA, unA)
_gBC = _stream_stats(hwC, unB)  # B/C stream: hardware = C, null = B (== C)
_all = _gA + _gBC
check("batch-resolved: uniform MEAN beats hardware MEAN at all 9 (stream, "
      "dimension) conditions (repeated measures on 2 independent streams)",
      len(_all) == 9 and all(g[0] < 0 for g in _all),
      f"{sum(1 for g in _all if g[0] < 0)}/{len(_all)}")
_mA = [-g[0] for g in _gA]
_mBC = [-g[0] for g in _gBC]
check("uniform-mean margins: A 4.8-14.9 mHa, B/C 18.4-26.7 mHa",
      close(min(_mA), 4.80, 0.05) and close(max(_mA), 14.87, 0.05)
      and close(min(_mBC), 18.38, 0.05) and close(max(_mBC), 26.74, 0.05),
      f"A [{min(_mA):.2f},{max(_mA):.2f}] BC [{min(_mBC):.2f},{max(_mBC):.2f}]")
_sds = [s for g in _all for s in (g[1], g[2])]
check("per-group sample standard deviations (ddof=1) span 3.7-16.3 mHa",
      close(min(_sds), 3.66, 0.06) and close(max(_sds), 16.33, 0.06),
      f"[{min(_sds):.2f},{max(_sds):.2f}]")
from scipy import stats as _st  # noqa: E402
_mwu2 = [_st.mannwhitneyu(g[3], g[4], alternative="two-sided").pvalue
         for g in _all]
check("uniform advantage individually significant (TWO-sided MWU p<0.05, "
      "per condition) at 7/9; only the two smallest-A dims are not",
      sum(1 for p in _mwu2 if p < 0.05) == 7
      and _mwu2[0] >= 0.05 and _mwu2[1] >= 0.05
      and all(p < 0.05 for p in _mwu2[2:]),
      f"sig {sum(1 for p in _mwu2 if p < 0.05)}/9, "
      f"p[0:2]=[{_mwu2[0]:.3f},{_mwu2[1]:.3f}]")
_md_stats = _MD_ALL
check("no global sign test cited (pseudoreplication guard): 'sign test' and "
      "'2^-9' absent from the paper; per-condition tests two-sided",
      "sign test" not in _md_stats and "2⁻⁹" not in _md_stats
      and "two-sided Mann–Whitney" in _md_stats
      and "9 independent" not in _md_stats)
_ust = np.load("ibmuniform_stats.npz")
check("ibmuniform_stats.npz regenerated: two-sided count 7/9 stored, legacy "
      "sign test stored under a legacy key only",
      int(_ust["n_un_ahead_sig_two_sided"]) == 7
      and "sign_test_p_legacy" in _ust.files
      and "sign_test_p" not in _ust.files
      and int(_ust["n_un_ahead_mean"]) == 9)
import hashlib  # noqa: E402
h1_ = hashlib.md5(open(f"{IBB}/experiments/2Fe-2S/sqd_on_uniform_distribution/"
                       "energy-variance_data_SQD_eigenstate_B_uniform.txt",
                       "rb").read()).hexdigest()
h2_ = hashlib.md5(open(f"{IBB}/experiments/2Fe-2S/sqd_on_uniform_distribution/"
                       "energy-variance_data_SQD_eigenstate_C_uniform.txt",
                       "rb").read()).hexdigest()
check("their fe2s2 uniform B and C files are byte-identical (footnote)",
      h1_ == h2_, h1_[:8])
check("their fe2s2 hardware best (A) ~ +184 mHa above their own DMRG",
      close((hwA[:, 1].min() - E2_THEIR) * 1e3, 183.91, 0.5),
      f"{(hwA[:,1].min()-E2_THEIR)*1e3:+.2f}")
hw4 = their_txt(f"{IBB}/experiments/4Fe-4S/sqd_hardware_energetics/"
                "energy-variance_data_SQD.txt")
un4 = their_txt(f"{IBB}/experiments/4Fe-4S/sqd_on_uniform_distribution/"
                "energy-variance_data_SQD_uniform.txt")
check("their fe4s4: hardware DOES beat uniform (~1.06 Ha; 36-orbital space)",
      close((hw4[:, 1].min() - un4[:, 1].min()) * 1e3, -1057.98, 2),
      f"{(hw4[:,1].min()-un4[:,1].min())*1e3:+.1f} mHa")
check("their fe4s4 hardware best remains ~ +653 mHa above their DMRG",
      close((hw4[:, 1].min() - E4_THEIR) * 1e3, 653.33, 0.5),
      f"{(hw4[:,1].min()-E4_THEIR)*1e3:+.2f}")
d100 = np.loadtxt(f"{IBB}/experiments/4Fe-4S/sqd_hardware_energetics/"
                  "SQD_energies_100_batches_d-100M.txt")
best100 = d100[:, 1].min()
check("their fe4s4 SQD best at d=1e8 ~ +594.8 mHa vs their DMRG",
      close((best100 - E4_THEIR) * 1e3, 594.81, 0.5),
      f"{(best100-E4_THEIR)*1e3:+.2f}")
E_HCI4 = -326.793854099389
check("their fe4s4 classical HCI(best) ~ +445.8 -> SQD@1e8 trails HCI by "
      "~149 mHa on their own numbers",
      close((best100 - E_HCI4) * 1e3, 149.03, 0.5),
      f"{(best100-E_HCI4)*1e3:+.2f}")

# ============ L. their hardware samples through their pipeline ============
if os.path.exists("ibmraw_counts.npz"):
    dr = np.load("ibmraw_counts.npz")
    check("fe2s2 hw: 2,457,600 shots reconstructed (their 300x8192)",
          int(dr["tot"]) == 2457600, f"{int(dr['tot']):,}")
    frac = int(dr["insec_shots"]) / int(dr["tot"])
    check("fe2s2 hw: measured in-sector shot fraction ~0.445%",
          close(frac, 4.450e-3, 5e-5), f"{frac:.3e}")
if os.path.exists("ibmship_symm1.npz"):
    h1 = np.load("ibmship_symm1.npz", allow_pickle=True)
    check("ibmship symm1 BEST: +247.5 mHa, S2 ~0.235 (their hw samples, "
          "their pipeline, dim 500x500)",
          close(float(h1["best_e"]), -116.358076, 5e-6)
          and close(float(h1["best_s2"]), 0.235, 0.005),
          f"E {float(h1['best_e']):.6f} S2 {float(h1['best_s2']):.3f}")
if os.path.exists("ibmship_symm0.npz"):
    h0 = np.load("ibmship_symm0.npz", allow_pickle=True)
    check("ibmship symm0 BEST: +257.1 mHa, S2 ~0.231",
          close(float(h0["best_e"]), -116.348532, 5e-6)
          and close(float(h0["best_s2"]), 0.231, 0.005),
          f"E {float(h0['best_e']):.6f} S2 {float(h0['best_s2']):.3f}")
if os.path.exists("lucjopt_stats.npz"):
    lo = np.load("lucjopt_stats.npz")
    check("their 'samples' file: 58,644,960 entries, ALL unique (=> the "
          "deduplicated subspace-defining configuration set, not raw shots)",
          int(lo["tot"]) == 58644960 and int(lo["nuniq"]) == 58644960,
          f"tot {int(lo['tot']):,} uniq {int(lo['nuniq']):,}")
    check("their config set: 100% in-sector (noiseless provenance)",
          int(lo["insec_uniq"]) == int(lo["nuniq"]),
          f"{int(lo['insec_uniq']):,}")
    # v5.9.7: v5.9.5-v5.9.6 said the set held NO Hartree-Fock string. It holds
    # exactly one, row 0. The O1 log prints hf_weight at .4f, and 1/58.6M reads
    # as 0.0000; the claim was never pinned, so 368/0 could not catch it.
    # Re-derived three ways from IBM's raw files by _hfcheck.py.
    check("their config set: the Hartree-Fock string appears EXACTLY ONCE "
          "(hf_weight x tot == 1; row 0, as their load_samples.py labels it)",
          round(float(lo["hf_weight"]) * int(lo["tot"])) == 1,
          f"hf_weight {float(lo['hf_weight']):.6e} x {int(lo['tot']):,} = "
          f"{float(lo['hf_weight']) * int(lo['tot']):.6f}")
# ... and the manuscript must say so: the v5.9.5-v5.9.6 phrasing is forbidden
check("HF string stated as present once, never as absent (md + tex)",
      "none is the Hartree" not in _MD_ALL
      and "none is the Hartree" not in _TX_ALL
      and "string appears exactly once, as row" in _MD_ALL
      and "string appears exactly once, as row" in _TX_ALL,
      "md ok" if "string appears exactly once, as row" in _MD_ALL
      else "md missing corrected phrase")
if os.path.exists("lucjopt_state.npz"):
    ls = np.load("lucjopt_state.npz")
    check("THEIR optimized ansatz state (their params): exact singlet "
          "S2 ~3e-4", abs(float(ls["s2vec"])) < 5e-4,
          f"{float(ls['s2vec']):.6f}")
    check("raw-draw hypothesis refuted: closed form from their own state "
          "predicts ~5.4M distinct at 58.6M draws (file has 58.6M distinct)",
          close(float(np.interp(np.log(58644960), np.log(ls["mgrid"]),
                                ls["eu"])), 5.4e6, 0.2e6),
          f"{float(np.interp(np.log(58644960), np.log(ls['mgrid']), ls['eu'])):,.0f}")

if os.path.exists("ibm4raw_counts.npz"):
    d4 = np.load("ibm4raw_counts.npz")
    check("fe4s4 hw: 3,163,742 recorded outcomes, every one unit-count",
          int(d4["tot"]) == 3163742 and int(d4["tot"]) == 3163742,
          f"{int(d4['tot']):,}")
    check("fe4s4 hw: measured in-sector fraction 5.354e-04",
          close(int(d4["insec_shots"]) / int(d4["tot"]), 5.354e-4, 5e-6),
          f"{int(d4['insec_shots'])/int(d4['tot']):.3e}")
if os.path.exists("ibm4ship_symm1.npz"):
    f41 = np.load("ibm4ship_symm1.npz", allow_pickle=True)
    check("ibm4ship symm1 BEST: spin-PURE TRIPLET S2=2.000 at +1438.0 mHa "
          "(their fe4s4 hw data; mitigated manifold permits singlet, solver "
          "picks triplet)",
          close(float(f41["best_e"]), -325.801643, 5e-6)
          and close(float(f41["best_s2"]), 2.000, 0.005),
          f"E {float(f41['best_e']):.6f} S2 {float(f41['best_s2']):.3f}")

if os.path.exists("ibm4ship_symm0.npz"):
    f40 = np.load("ibm4ship_symm0.npz", allow_pickle=True)
    check("ibm4ship symm0 BEST: S2=3.00 at +2148.7 mHa (mitigation off; "
          "on fe4s4 the mitigation acts -- purifying into the WRONG state)",
          close(float(f40["best_e"]), -325.090909, 5e-6)
          and close(float(f40["best_s2"]), 3.003, 0.005),
          f"E {float(f40['best_e']):.6f} S2 {float(f40['best_s2']):.3f}")

# ============ K. noiseless LUCJ legs (paper 2.7-2.8) ============
if os.path.exists("lucj2_stats.npz"):
    s = np.load("lucj2_stats.npz")
    check("lucj2 (their template): 1e6 shots -> 5,338 unique",
          int(s["nuniq"]) == 5338, f"{int(s['nuniq'])}")
    pred = float(np.interp(np.log(1e6), np.log(s["mgrid"]), s["eu"]))
    check("lucj2: coupon closed form ~5,344 (4th validation, <1%)",
          abs(pred - int(s["nuniq"])) / int(s["nuniq"]) < 0.01,
          f"pred {pred:,.0f}")
    check("lucj2: HF det is top weight ~0.558",
          bool(s["hf_is_top"]) and close(float(s["top_w"][0]), 0.5579, 0.001),
          f"{float(s['top_w'][0]):.4f}")
if os.path.exists("lucj2ship_symm1.npz"):
    d21 = np.load("lucj2ship_symm1.npz", allow_pickle=True)
    check("lucj2ship symm1 BEST: +228.9 mHa, low-mean S2 ~0.11 "
          "(low ⟨S²⟩ but a spin mixture, wrong energy — the noiseless jaw)",
          close(float(d21["best_e"]), -116.376680, 5e-6)
          and close(float(d21["best_s2"]), 0.108, 0.005),
          f"E {float(d21['best_e']):.6f} S2 {float(d21['best_s2']):.3f}")
if os.path.exists("lucj2ship_symm0.npz"):
    d20 = np.load("lucj2ship_symm0.npz", allow_pickle=True)
    check("lucj2ship symm0 BEST: +236.7 mHa, S2 ~0.12 (mitigation buys a "
          "modest 7.7 mHa ONLY on the low-⟨S²⟩ noiseless input; "
          "exact no-op on realistic soup input)",
          close(float(d20["best_e"]), -116.368941, 5e-6)
          and close(float(d20["best_s2"]), 0.123, 0.005),
          f"E {float(d20['best_e']):.6f} S2 {float(d20['best_s2']):.3f}")
if os.path.exists("lucj_stats.npz"):
    s1 = np.load("lucj_stats.npz", allow_pickle=True)
    check("lucj run-1 (naive localized-basis ref, appendix note): ansatz "
          "S^2 = 0 exact singlet", abs(float(s1["s2vec"])) < 1e-6,
          f"{float(s1['s2vec']):.2e}")
    check("lucj run-1: 149 unique dets @1e6 (reference-choice observation)",
          int(s1["nuniq"]) == 149, f"{int(s1['nuniq'])}")

# ===== L2. set-draws leg: their 58.6M configuration set (paper 2.8) =====
if os.path.exists("lucjoptship_symm1.npz") \
        and os.path.exists("lucjoptship_symm0.npz"):
    o1 = np.load("lucjoptship_symm1.npz", allow_pickle=True)
    o0 = np.load("lucjoptship_symm0.npz", allow_pickle=True)
    check("lucjoptship symm1 BEST: essentially the triplet S2=2.008 at "
          "+501.8 mHa (their curated set; triplet attractor)",
          close(float(o1["best_e"]), -116.103827, 5e-6)
          and close(float(o1["best_s2"]), 2.008, 0.005),
          f"E {float(o1['best_e']):.6f} S2 {float(o1['best_s2']):.3f}")
    check("lucjoptship symm1 err vs their DMRG = +501.8 mHa",
          close((float(o1["best_e"]) - E2_THEIR) * 1e3, 501.8, 0.5),
          f"{(float(o1['best_e'])-E2_THEIR)*1e3:+.2f}")
    check("lucjoptship symm0 BEST: S2=1.089 (near-even singlet-triplet "
          "mixture) at +377.8 mHa",
          close(float(o0["best_e"]), -116.227841, 5e-6)
          and close(float(o0["best_s2"]), 1.089, 0.005),
          f"E {float(o0['best_e']):.6f} S2 {float(o0['best_s2']):.3f}")
    check("lucjoptship: mitigation COSTS energy on this input class: "
          "symm1 ends 124 mHa above symm0",
          close((float(o1["best_e"]) - float(o0["best_e"])) * 1e3,
                124.0, 0.5),
          f"{(float(o1['best_e'])-float(o0['best_e']))*1e3:+.2f} mHa")
    it1 = sorted(float(h["s2"]) for h in o1["hist"] if h["iter"] == 1)
    check("lucjoptship symm1 iteration-1 spin lottery: batch S2 = "
          "{0.97, 3.00, 4.00}",
          len(it1) == 3 and close(it1[0], 0.97, 0.02)
          and close(it1[1], 3.00, 0.02) and close(it1[2], 4.00, 0.02),
          f"{[f'{x:.2f}' for x in it1]}")

# ===== L3. raw-counts ladder at their published construction (paper 2.8) =====
LADDER = [  # (npz, best_e, err_mHa, s2, dim, their_pub_best10_mHa, margin)
    ("sqdraw_results_aws/sqdraw_n500_symm1_s17.npz",
     -116.368271, 237.338, 0.1422, 2.5e5, 266.3, 28.9),
    ("sqdraw_results_aws/sqdraw_n1000_symm1_s17.npz",
     -116.390277, 215.332, 0.1080, 1e6, 238.8, 23.5),
    ("sqdraw_results_aws/sqdraw_n2000_symm1_s17.npz",
     -116.438440, 167.169, 0.3565, 4e6, 216.7, 49.5),
]
if all(os.path.exists(p) for p, *_ in LADDER):
    s2_by_n, pub_by_n = [], []
    for p, we, werr, ws2, dim, pub, marg in LADDER:
        z = np.load(p, allow_pickle=True)
        n = int(z["n"])
        check(f"sqdraw n{n} knobs: spb=3N, 10 batches, 5 iterations, "
              "max_dim=N (their published construction)",
              int(z["spb"]) == 3 * n and int(z["nbatch"]) == 10
              and int(z["maxit"]) == 5 and int(z["seed"]) == 17,
              f"spb {int(z['spb'])} nb {int(z['nbatch'])} it {int(z['maxit'])}")
        e, s2v = float(z["best_e"]), float(z["best_s2"])
        check(f"sqdraw n{n} table row: E {we} err +{werr:.1f} S2 {ws2}",
              close(e, we, 5e-6) and close((e - E2) * 1e3, werr, 0.05)
              and close(s2v, ws2, 5e-4),
              f"E {e:.6f} err {(e-E2)*1e3:+.3f} S2 {s2v:.4f}")
        rows = hwA[np.isclose(hwA[:, 2], dim)]
        check(f"their eigenstate-A file: 10 published batches at dim {dim:.0e}",
              rows.shape[0] == 10, f"{rows.shape[0]} rows")
        pub_err = (rows[:, 1].min() - E2_THEIR) * 1e3
        check(f"their published best-of-10 at dim {dim:.0e} = +{pub:.1f} mHa",
              close(pub_err, pub, 0.05), f"{pub_err:+.2f}")
        check(f"ours beats their published at dim {dim:.0e} by {marg:.1f} mHa",
              close(pub_err - (e - E2) * 1e3, marg, 0.05),
              f"{pub_err-(e-E2)*1e3:+.2f}")
        s2_by_n.append(s2v)
        pub_by_n.append(pub_err)
    check("ladder: our 1e6 run ~ matches their published 4e6 value (<2 mHa)",
          abs((LADDER[1][1] - E2) * 1e3 - pub_by_n[2]) < 2.0,
          f"{(LADDER[1][1]-E2)*1e3:+.1f} vs {pub_by_n[2]:+.1f}")
    check("ladder: converged S2 NON-monotone, largest dim most contaminated",
          s2_by_n[0] > s2_by_n[1] < s2_by_n[2]
          and s2_by_n[2] == max(s2_by_n),
          f"{[f'{x:.3f}' for x in s2_by_n]}")
    check("ladder best +167.2 mHa = two orders of magnitude from chemical "
          "accuracy (1.6 mHa)", (LADDER[2][1] - E2) * 1e3 / 1.6 > 100,
          f"{(LADDER[2][1]-E2)*1e3/1.6:.0f}x")
    h2k = list(np.load(LADDER[2][0], allow_pickle=True)["hist"])

    def band(it, key):
        v = [float(r[key]) for r in h2k if r["iter"] == it]
        return min(v), max(v)

    s3, e3 = band(3, "s2"), band(3, "err_mHa")
    s4, e4_ = band(4, "s2"), band(4, "err_mHa")
    s5, e5 = band(5, "s2"), band(5, "err_mHa")
    check("n2000 it3: batch S2 0.12-0.20 at +189-195 mHa (low-spin band)",
          0.115 <= s3[0] and s3[1] <= 0.205
          and close(e3[0], 188.7, 0.5) and close(e3[1], 194.9, 0.5),
          f"S2 [{s3[0]:.3f},{s3[1]:.3f}] E [{e3[0]:+.1f},{e3[1]:+.1f}]")
    check("n2000 it4: jumps to higher-spin lower-E basin S2 0.33-0.45 "
          "at +177-185 mHa",
          0.32 <= s4[0] and s4[1] <= 0.46
          and close(e4_[0], 176.5, 0.5) and close(e4_[1], 185.4, 0.5),
          f"S2 [{s4[0]:.3f},{s4[1]:.3f}] E [{e4_[0]:+.1f},{e4_[1]:+.1f}]")
    best5 = min((float(r["err_mHa"]), float(r["s2"])) for r in h2k
                if r["iter"] == 5)
    check("n2000 it5 converges in new basin: best +167.2 at S2 0.357",
          close(best5[0], 167.169, 0.05) and close(best5[1], 0.3565, 5e-4),
          f"{best5[0]:+.3f} S2 {best5[1]:.4f}")
    check("n2000 final two iterations buy ~21 mHa (abandoning low-spin band)",
          close(e3[0] - best5[0], 21.5, 0.5), f"{e3[0]-best5[0]:.2f}")
    # ablations (desktop venue)
    if all(os.path.exists(p) for p in
           ("sqdraw_n1000_symm1_s43.npz", "sqdraw_n1000_symm0_s17.npz",
            "sqdraw_n500_symm1_s17.npz")):
        a17 = np.load(LADDER[1][0], allow_pickle=True)
        a43 = np.load("sqdraw_n1000_symm1_s43.npz", allow_pickle=True)
        de = abs(float(a43["best_e"]) - float(a17["best_e"])) * 1e3
        ds = abs(float(a43["best_s2"]) - float(a17["best_s2"]))
        check("ladder ablation: seed 43 reproduces 1e6 point to 2.6 mHa, "
              "dS2 0.001", close(de, 2.56, 0.05) and ds < 0.002,
              f"dE {de:.2f} dS2 {ds:.4f}")
        a0 = np.load("sqdraw_n1000_symm0_s17.npz", allow_pickle=True)
        rat = float(a0["best_s2"]) / float(a17["best_s2"])
        de0 = (float(a17["best_e"]) - float(a0["best_e"])) * -1e3
        check("ladder ablation: symmetrize OFF ~doubles S2 (0.208 vs 0.108) "
              "and costs 6.4 mHa",
              close(float(a0["best_s2"]), 0.2082, 5e-4)
              and 1.75 < rat < 2.1 and close(de0, 6.40, 0.05),
              f"S2 {float(a0['best_s2']):.4f} ({rat:.2f}x) dE {de0:+.2f}")
        d5 = np.load("sqdraw_n500_symm1_s17.npz", allow_pickle=True)
        a5 = np.load(LADDER[0][0], allow_pickle=True)
        dxb = abs(float(d5["best_e"]) - float(a5["best_e"])) * 1e3
        check("ladder ablation: independent second-hardware n500 run same "
              "regime, D = 6.2 mHa",
              close(dxb, 6.21, 0.05)
              and 0.1 < float(d5["best_s2"]) < 0.2,
              f"dE {dxb:.2f} S2 {float(d5['best_s2']):.4f}")

# ===== L4. flagship-dimension leg: converged S2 at 5.625e7 (paper 2.8) =====
FLAG = [  # (npz, err_mHa, s2) — idealized exact-marginal construction rungs
    ("flagsolve_n300.npz", 234.787, 3.8082),
    ("flagsolve_n500.npz", 186.777, 3.7918),
    ("flagsolve_n600.npz", 173.190, 3.8012),
    ("flagsolve_n1000.npz", 129.785, 3.7711),
    ("flagsolve_n2000.npz", 80.002, 3.3930),
]
_flag_files = ([p for p, *_ in FLAG]
               + ["flagsolve_n7500_box2.npz", "flagsolve_n2000_box1.npz"])
if all(os.path.exists(p) for p in _flag_files):
    for p, werr, ws2 in FLAG:
        z = np.load(p, allow_pickle=True)
        e, s2v, n = float(z["e"]), float(z["s2"]), int(z["n"])
        check(f"flagsolve n{n} rung: err +{werr:.1f} mHa, their S2 {ws2}",
              close((e - E2) * 1e3, werr, 0.05) and close(s2v, ws2, 5e-4)
              and close((e - E2) * 1e3, float(z["err_mHa"]), 1e-6),
              f"err {(e-E2)*1e3:+.3f} S2 {s2v:.4f}")
    zf = np.load("flagsolve_n7500_box2.npz", allow_pickle=True)
    ef, s2f = float(zf["e"]), float(zf["s2"])
    check("flagship: n=7500, ndets = 56,250,000 = their largest published "
          "[2Fe-2S] dim",
          int(zf["n"]) == 7500 and int(zf["ndets"]) == 56_250_000,
          f"n {int(zf['n'])} ndets {int(zf['ndets']):,}")
    check("flagship table row: E -116.591965, err +13.6 mHa",
          close(ef, -116.591965, 5e-6) and close((ef - E2) * 1e3, 13.645, 0.05),
          f"E {ef:.6f} err {(ef-E2)*1e3:+.3f}")
    check("flagship: their S2 = 1.287 (near-even singlet-triplet mixture, "
          "parent exact singlet)",
          close(s2f, 1.2874, 5e-4) and 1.0 < s2f < 2.0, f"{s2f:.4f}")
    check("flagship: full amplitude matrix archived (7500 x 7500)",
          zf["amps"].shape == (7500, 7500), f"{zf['amps'].shape}")
    rows75 = hwA[np.isclose(hwA[:, 2], 5.625e7)]
    check("their eigenstate-A: single published batch at 5.625e7",
          rows75.shape[0] == 1, f"{rows75.shape[0]} rows")
    pub75 = (rows75[:, 1].min() - E2_THEIR) * 1e3
    check("their published at 5.625e7 = +183.9 mHa = their best published "
          "[2Fe-2S] energy anywhere",
          close(pub75, 183.909, 0.05)
          and np.isclose(rows75[:, 1].min(), hwA[:, 1].min()),
          f"{pub75:+.2f}")
    check("flagship converged beats their published at matched dim by "
          "170.3 mHa",
          close(pub75 - (ef - E2) * 1e3, 170.26, 0.05),
          f"{pub75-(ef-E2)*1e3:+.2f}")
    check("flagship wall 63,846 s (17.7 h) at 32 threads",
          close(float(zf["wall_s"]), 63846, 5) and int(zf["nt"]) == 32,
          f"{float(zf['wall_s']):,.0f}s NT{int(zf['nt'])}")
    zd = np.load("flagsolve_n2000.npz", allow_pickle=True)
    zb = np.load("flagsolve_n2000_box1.npz", allow_pickle=True)
    de_uHa = abs(float(zd["e"]) - float(zb["e"])) * 1e6
    ds2 = abs(float(zd["s2"]) - float(zb["s2"]))
    check("n2000 identity instability: converged runs 22 uHa apart in E",
          close(de_uHa, 22.2, 0.5), f"{de_uHa:.1f} uHa")
    check("n2000 identity instability: same runs differ 0.13 in S2 "
          "(3.393 desktop / 3.518 cloud)",
          close(ds2, 0.1252, 5e-4) and close(float(zd["s2"]), 3.3930, 5e-4)
          and close(float(zb["s2"]), 3.5182, 5e-4), f"dS2 {ds2:.4f}")
    check("n2000 cloud run matches PyCI anchor S2 3.5192 to ~1e-3",
          abs(float(zb["s2"]) - 3.5192) < 1.5e-3, f"{float(zb['s2']):.4f}")
    _ANCH = {300: 234.8, 500: 186.8, 600: 173.2, 1000: 129.8, 2000: 80.0}
    _devs = [abs(float(np.load(p, allow_pickle=True)["err_mHa"])
                 - _ANCH[int(np.load(p, allow_pickle=True)["n"])])
             for p, *_ in FLAG] + [abs(float(zb["err_mHa"]) - 80.0)]
    check("flagsolve rungs certified vs PyCI anchors: dE range "
          "0.002-0.023 mHa", max(_devs) <= 0.025, f"max {max(_devs):.3f}")

# ===== N. scale extrapolation: S2 decay per doubling (paper 2.4 ii) =====
import csv as _csv  # noqa: E402
import math as _math  # noqa: E402
_rows = list(_csv.DictReader(open("s2audit_master.csv")))


def _rates(armname, last_n):
    pts = sorted((int(float(r["ndets"])), float(r["S2_r0"]))
                 for r in _rows if r["arm"] == armname)
    out = []
    for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
        out.append((s1 - s0) / _math.log2(d1 / d0))
    return pts, out[-last_n:]


_bspts, _bsr = _rates("fe2s2bs", 3)
check("scale: BS-arm S2 decay per doubling constant in [0.08, 0.10] over "
      "final 3 intervals (paper: 0.085-0.097)",
      all(-0.10 <= r <= -0.08 for r in _bsr),
      f"{[f'{r:+.3f}' for r in _bsr]}")
_dppts, _dpr = _rates("fe2s2deep", 3)
check("scale: deep-aufbau-arm S2 decay per doubling in [0.10, 0.14] "
      "(paper: 0.106-0.133)",
      all(-0.14 <= r <= -0.10 for r in _dpr),
      f"{[f'{r:+.3f}' for r in _dpr]}")
_fast = max(abs(r) for r in _bsr + _dpr)
_dlast, _slast = max(_bspts[-1], _dppts[-1])
_SECTOR = 15504.0 ** 2  # C(20,15)^2 = 2.4e8
_d_s2lt1 = _dlast * 2 ** ((_slast - 1.0) / _fast)
_d_s2eq0 = _dlast * 2 ** (_slast / _fast)
check("scale: at fastest measured rate, S2<1 needs ~28 doublings, "
      "D ~ 3e13, >=5 orders beyond the 2.4e8 sector",
      1e13 < _d_s2lt1 < 1e14 and _d_s2lt1 / _SECTOR > 1e5
      and 27 < (_slast - 1.0) / _fast < 29,
      f"D {_d_s2lt1:.2e} ratio {_d_s2lt1/_SECTOR:.1e}")
check("scale: S2~0 needs ~35 doublings, D ~ 6e15, >=7 orders beyond sector",
      1e15 < _d_s2eq0 < 1e16 and _d_s2eq0 / _SECTOR > 1e7
      and 34 < _slast / _fast < 36,
      f"D {_d_s2eq0:.2e} ratio {_d_s2eq0/_SECTOR:.1e}")

# ===== M. successor paper (arXiv:2511.00224) quoted values =====
# Literature transcriptions (verified against the arXiv v1 full text on
# 2026-07-10) + internal arithmetic. These cannot be re-derived from our
# archives; the checks pin the quoted constants and their consistency.
SUC_BEST4, SUC_EXTRAP, SUC_DMRG = -326.912, -327.118, -327.239
check("successor 2511.00224: extrapolation gap -327.118 vs their DMRG "
      "-327.239 = 0.121 Ha, consistent with their 'about 0.12 E_h'",
      0.10 < abs(SUC_EXTRAP - SUC_DMRG) < 0.14,
      f"{abs(SUC_EXTRAP - SUC_DMRG):.3f} Ha")
check("successor fe4s4 DMRG ref (-327.239) matches benchmark dump ref "
      "(-327.2396369) to <1 mHa (paper: 'to the milli-Hartree')",
      abs(SUC_DMRG - E4) < 1e-3, f"{abs(SUC_DMRG - E4)*1e3:.2f} mHa")
check("successor ordering: raw D=1e10 best above extrapolated above "
      "their DMRG (all remain ABOVE the reference)",
      SUC_BEST4 > SUC_EXTRAP > SUC_DMRG,
      f"{SUC_BEST4} > {SUC_EXTRAP} > {SUC_DMRG}")

# ===== N. flagship three-root certification (gram7500, Section 2.8) =====
import re as _re
g3 = np.load("gram7500.npz")
gc = {t: np.load(f"gram7500_{t}.npz") for t in ("ck1", "ck2", "ck3")}
_G = g3["gram"]; _Er = g3["e_roots"]; _Eex = float(g3["e_exact"])

check("gram7500 final == ck3 (energies, gram, bitwise)",
      np.array_equal(_Er, gc["ck3"]["e_roots"])
      and np.array_equal(_G, gc["ck3"]["gram"]))
check("gram7500 e_exact == master E2", close(_Eex, E2, 1e-9))
if os.path.exists("flagsolve_n7500_box2.npz"):  # Zenodo companion, 413 MB
    fs75 = np.load("flagsolve_n7500_box2.npz")
    check("warm provenance: e_bank == flagsolve_n7500_box2 energy",
          close(float(g3["e_bank"]), float(fs75["e"]), 1e-12),
          f"{float(g3['e_bank']):.12f}")
    check("warm provenance: s2_bank == flagsolve_n7500_box2 S2 (1.287)",
          close(float(g3["s2_bank"]), float(fs75["s2"]), 1e-12)
          and close(float(g3["s2_bank"]), 1.287, 5e-4))
check("gate G3: warm <S2> vs banked < 1e-14",
      abs(float(g3["s2_warm"]) - float(g3["s2_bank"])) < 1e-14,
      f"{abs(float(g3['s2_warm'])-float(g3['s2_bank'])):.1e}")
check("delivered residual 3.5e-4 (paper: 3.5x10^-4)",
      close(float(g3["warm_resid"]) * 1e4, 3.5, 0.06),
      f"{float(g3['warm_resid']):.3e}")
check("ladder config: tol (1e-6,1e-7,1e-8), max_cycle (150,150,200), "
      "nroots=3, n=7500, NT=64",
      [float(gc[t]["tol"]) for t in ("ck1", "ck2", "ck3")] == [1e-6, 1e-7, 1e-8]
      and [int(gc[t]["max_cycle"]) for t in ("ck1", "ck2", "ck3")] == [150, 150, 200]
      and int(g3["nroots"]) == 3 and int(g3["n"]) == 7500 and int(g3["nt"]) == 64)
check("certified ground: -116.591966, err +13.643 mHa",
      round(float(_Er[0]), 6) == -116.591966
      and close((float(_Er[0]) - _Eex) * 1e3, 13.643, 1e-3),
      f"{float(_Er[0]):.9f}")
check("root1 gap +0.285 mHa, err +13.927",
      close((_Er[1] - _Er[0]) * 1e3, 0.285, 1e-3)
      and close((_Er[1] - _Eex) * 1e3, 13.927, 1e-3))
check("root2 gap +6.52 mHa, err +20.163",
      close((_Er[2] - _Er[0]) * 1e3, 6.52, 5e-3)
      and close((_Er[2] - _Eex) * 1e3, 20.163, 1e-3))
check("certified S2 diag (1.371, 2.357, 4.732)",
      all(round(float(_G[i, i]), 3) == v
          for i, v in enumerate((1.371, 2.357, 4.732))))
check("invariant S2 spectrum {0.038, 2.357, 6.066}",
      all(round(float(x), 3) == v for x, v in
          zip(gc["ck3"]["gram_eigs"], (0.038, 2.357, 6.066))))
check("all eigenstate S2 >= 1.37 (none below)", float(_G.diagonal().min()) > 1.37)
check("residuals (6.0, 8.4, 9.1)e-5, all < 1e-4",
      all(round(float(r) * 1e5, 1) == v for r, v in
          zip(g3["resid"], (6.0, 8.4, 9.1))) and float(max(g3["resid"])) < 1e-4)
_dS2r0 = abs(float(gc["ck3"]["gram"][0, 0]) - float(gc["ck2"]["gram"][0, 0]))
_dfloor = abs(float(gc["ck3"]["gram_eigs"][0]) - float(gc["ck2"]["gram_eigs"][0]))
check("ladder stability: last-stage dS2(root0) ~ 5e-5, dfloor ~ 6e-4",
      3e-5 < _dS2r0 < 7e-5 and 4e-4 < _dfloor < 7e-4,
      f"{_dS2r0:.1e} {_dfloor:.1e}")
_rot_s2 = float(_G[0, 0]) - float(g3["s2_bank"])
_rot_e = (float(_Er[0]) - float(g3["e_bank"])) * 1e6
check("residual-rotation: dS2 +0.084, dE -1.9 uHa (certified lower)",
      close(_rot_s2, 0.084, 1e-3) and -2.0 < _rot_e < -1.8,
      f"dS2 {_rot_s2:+.4f} dE {_rot_e:+.2f} uHa")
_w, _V = np.linalg.eigh(_G)
_v0 = _V[:, 0]; _wts = _v0 ** 2
_eray = float(np.sum(_wts * _Er))
check("floor direction: weights 0.78/0.22 (root1 < 1e-6), |G02| = 2.50",
      close(float(_wts[0]), 0.78, 5e-3) and close(float(_wts[2]), 0.22, 5e-3)
      and float(_wts[1]) < 1e-6 and close(abs(float(_G[0, 2])), 2.50, 5e-3))
check("floor direction Rayleigh +1.44 mHa above minimum",
      close((_eray - float(_Er[0])) * 1e3, 1.44, 5e-3),
      f"{(_eray-float(_Er[0]))*1e3:+.3f}")
check("ground-pair span floor == root0 S2 to 9 digits",
      abs(float(np.linalg.eigvalsh(_G[:2, :2])[0]) - float(_G[0, 0])) < 1e-9)
_log = open("gram7500.run.log").read()
check("log gates: G1/G2/G3 PASS, root0<=banked PASS, campaign DONE",
      "G1 PASS warm strings bit-identical" in _log
      and "G2 vs banked npz energy: dE 0.00 uHa [PASS]" in _log
      and "G3 warm <S2> 1.2874 vs banked 1.2874 [PASS" in _log
      and "GATE root0 <= banked: drift -1.86 uHa [PASS]" in _log
      and "GRAM7500_CAMPAIGN_DONE" in _log)
check("log: all three roots carry per-root converged flags",
      all(f"root {i} converged" in _log for i in range(3)))
_tmax = max(float(x) for x in _re.findall(r"\[\s*([0-9.]+)s rss", _log))
_rss = max(float(x) for x in _re.findall(r"rss\s+([0-9.]+)G", _log))
check("campaign 11.3 h, peak RSS 41.8 GB",
      close(_tmax / 3600, 11.3, 0.05) and close(_rss, 41.8, 0.05),
      f"{_tmax/3600:.2f} h rss {_rss:.1f}G")

# ============ O. review-round-2 hardening (v5.2) ============
# Language guards (the codemod's banned list, enforced forever), the
# residual/gap precision-limit disclosure, the structural-mechanism equation,
# the abstract restructure, and the package version matrix.
print("\n-- section O: review-round-2 hardening (v5.2) --")
_mdO = _MD_ALL
_txO = _TX_ALL

for _b in ("pins the exact", "is an exact singlet", "exact singlet's marginals",
           "of an exact singlet", "fully converged", "nine digits",
           "infinite-shot", "no one had run", "multi-spin zoo",
           "convergence-certified", "necessary condition for containing",
           "not by the subspace"):
    check(f"language guard: {_b!r} absent from md and tex",
          _b not in _mdO and _b not in _txO)

_ratioO = float(g3["resid"][0]) / float(_Er[1] - _Er[0])
check("precision limit: residual/gap ratio ~0.21 re-derived from archive",
      0.20 < _ratioO < 0.22, f"{_ratioO:.4f}")
check("ratio disclosed in both sources (21% of the 0.285 mHa gap)",
      "21% of the 0.285 mHa" in _mdO and "21\\% of the 0.285~mHa" in _txO)
check("Davis-Kahan caveat + stability-carries-certification present",
      "Davis–Kahan-type bound" in _mdO and "Davis--Kahan-type bound" in _txO
      and _mdO.count("stability and invariance") >= 1
      and _txO.count("stability and invariance") >= 1)
check("structural-mechanism equation present in both sources",
      "[PHP, PS²P] ≠ 0" in _mdO and "[PHP,\\, PS^2P] \\neq 0" in _txO)
_absO = _MD_MAIN.split("## Abstract\n\n", 1)[1].split("\n\n---", 1)[0]
check("journal abstract <= 300 words",
      len(_absO.split()) <= 300, f"{len(_absO.split())} words")
check("front matter (review round 7): Principal findings box present; "
      "Appendix B extended summary removed; References section added",
      "## Principal findings" in _mdO
      and "\\section*{Principal findings}" in _txO
      and "Extended summary of findings" not in _mdO
      and "Extended summary of findings" not in _txO
      and "## References" in _mdO
      and ("\\section*{References}" in _txO or "\\section{References}" in _txO
           or "\\begin{thebibliography}" in _txO))
check("compressed-block caveat present (not full spin-sector projector "
      "weights)",
      "not full spin-sector projector weights" in _mdO
      and "not full spin-sector projector weights" in _txO)
check("ansatz spin identity stated as measured value 3.3e-4 (never 'exact "
      "singlet')",
      "3.3×10⁻⁴" in _mdO and "3.3\\times10^{-4}" in _txO)

# version matrix: earliest-public 0.3.0 vs current 0.12.1, identical inputs
_v03 = np.load("verm_0_3_0.npz")
_v12 = np.load("verm_0_12_1.npz")
check("version matrix ran at earliest public (0.3.0) and current (0.12.1)",
      str(_v03["version"]) == "0.3.0" and str(_v12["version"]) == "0.12.1")
check("spin_sq penalty default is unset (None) in BOTH versions",
      str(_v03["spin_default"]) == "None"
      and str(_v12["spin_default"]) == "None")
check("spin_sq wires to pyscf fix_spin in both versions' source",
      bool(_v03["wires_fix_spin"]) and bool(_v12["wires_fix_spin"]))
for _leg in ("v1", "v2", "v3"):
    _a, _b = _v03[_leg], _v12[_leg]
    # dims compared as an unordered pair: 0.12.1's closure-off construction
    # swaps the alpha/beta half ordering relative to 0.3.0 (physically
    # equivalent; identical E and S2 — the one observed version difference)
    check(f"version agreement {_leg}: same dims (up to alpha/beta order), "
          f"|dE| < 10 uHa, |dS2| < 1e-3",
          {int(_a[2]), int(_a[3])} == {int(_b[2]), int(_b[3])}
          and abs(_a[0] - _b[0]) < 1e-5 and abs(_a[1] - _b[1]) < 1e-3,
          f"dE {abs(_a[0]-_b[0])*1e6:.3f} uHa, dS2 {abs(_a[1]-_b[1]):.2e}, "
          f"dims {int(_a[2])}x{int(_a[3])} vs {int(_b[2])}x{int(_b[3])}")
check("V1 reproducer core: both versions report the contaminated <S^2> "
      "~3.50 at N=60 through their own diagnostics",
      close(float(_v03["v1"][1]), 3.50, 0.01)
      and close(float(_v12["v1"][1]), 3.50, 0.01),
      f"{float(_v03['v1'][1]):.4f} / {float(_v12['v1'][1]):.4f}")
check("paper states the 0.1.0-not-retrievable finding and the 0.3.0 matrix",
      "not publicly retrievable" in _mdO and "0.3.0" in _mdO
      and "not publicly retrievable" in _txO and "0.3.0" in _txO)

# ============ P. stage-4 certification (v5.3) ============
# Residual push (ratio 0.21 -> 0.011), the 4th root, and the S^4 second
# moments: every number quoted in the new Section 2.8 paragraph, Methods
# stage-4 paragraph, and Limitations updates, re-derived from the four
# checkpoints of the stage-4 run (gram7500_s4ck{0..3}.npz).
print("\n-- section P: stage-4 certification (residual push, root 3, S^4) --")
_cksP = [np.load(f"gram7500_s4ck{_i}.npz") for _i in range(4)]
_c0P, _c3P = _cksP[0], _cksP[3]
_E3P = np.asarray(_c3P["e_rayleigh"], float)
_r3P = np.asarray(_c3P["resid"], float)
_G3P = np.asarray(_c3P["gram"], float)
_V3P = np.diagonal(np.asarray(_c3P["gram4"], float)) - np.diagonal(_G3P) ** 2


def _swP(m, q):
    """{S=0,1,2} sector solve exactly as in Methods."""
    w2 = (q - 2.0 * m) / 24.0
    w1 = (6.0 * m - q) / 8.0
    return np.array([1.0 - w1 - w2, w1, w2])


check("s4 warm gate: ck0 Rayleigh energies == banked certified roots "
      "(<1e-8 Ha)",
      np.abs(np.asarray(_c0P["e_rayleigh"]) -
             np.asarray(_c0P["e_cert"])).max() < 1e-8,
      f"max {np.abs(np.asarray(_c0P['e_rayleigh'])-np.asarray(_c0P['e_cert'])).max():.1e}")
check("s4 warm gate: ck0 S2-Gram == banked certified Gram (<1e-6)",
      np.abs(np.asarray(_c0P["gram"]) -
             np.asarray(_c0P["gram_cert"])).max() < 1e-6)
_gap0P = float(_c0P["e_rayleigh"][1] - _c0P["e_rayleigh"][0])
_rat0P = float(_c0P["resid"][0]) / _gap0P
check("s4 ck0 re-derives the disclosed 21% starting ratio",
      0.20 < _rat0P < 0.22, f"{_rat0P:.4f}")
_gap3P = float(_E3P[1] - _E3P[0])
check("ground gap 0.285 mHa at the final stage",
      round(_gap3P * 1e3, 3) == 0.285, f"{_gap3P*1e3:.5f}")
check("final measured ground residual quotes as 3.2e-6",
      close(float(_r3P[0]), 3.2e-6, 0.03), f"{float(_r3P[0]):.3e}")
_rat3P = float(_r3P[0]) / _gap3P
check("final ratio quotes as 1.1% of the ground gap",
      round(_rat3P * 100, 1) == 1.1, f"{_rat3P*100:.3f}%")
check("tightening factor quotes as 18.6-fold",
      round(_rat0P / _rat3P, 1) == 18.6, f"{_rat0P/_rat3P:.3f}")
check("ground <S^2> = 1.3711 at ALL four checkpoints (4th decimal frozen)",
      all(round(float(np.asarray(_z["gram"])[0, 0]), 4) == 1.3711
          for _z in _cksP),
      " ".join(f"{float(np.asarray(_z['gram'])[0,0]):.5f}" for _z in _cksP))
check("four-root <S^2> ladder quotes as 1.371, 2.363, 4.738, 11.590",
      [round(float(_G3P[_i, _i]), 3) for _i in range(4)]
      == [1.371, 2.363, 4.738, 11.590],
      " ".join(f"{float(_G3P[_i,_i]):.4f}" for _i in range(4)))
_errP = (_E3P - float(_c3P["e_exact"])) * 1e3
check("root-3 energy quotes as +26.19 mHa",
      round(float(_errP[3]), 2) == 26.19, f"{float(_errP[3]):+.4f}")
check("root 3 sits 12.5 mHa above the ground root",
      round(float(_errP[3] - _errP[0]), 1) == 12.5,
      f"{float(_errP[3]-_errP[0]):.4f}")
check("Var(S^2) quotes as 6.92, 3.54, 7.41, 4.41",
      [round(float(_v), 2) for _v in _V3P] == [6.92, 3.54, 7.41, 4.41],
      " ".join(f"{float(_v):.4f}" for _v in _V3P))
check("'every low root carries Var(S^2) >= 3.5' (min over 4 roots)",
      float(_V3P.min()) >= 3.5, f"min {float(_V3P.min()):.4f}")
_w0P = _swP(float(_G3P[0, 0]),
            float(np.asarray(_c3P["gram4"])[0, 0]))
check("ground-root sector solve quotes as (0.82, -0.07, 0.25), infeasible",
      [round(float(_w), 2) for _w in _w0P] == [0.82, -0.07, 0.25]
      and float(_w0P.min()) < -1e-3,
      " ".join(f"{float(_w):+.5f}" for _w in _w0P))
check("ground-root sector solve identical to 4 decimals across all four "
      "checkpoints",
      len({tuple(round(float(_w), 4) for _w in
                 _swP(float(np.asarray(_z["gram"])[0, 0]),
                      float(np.asarray(_z["gram4"])[0, 0])))
           for _z in _cksP}) == 1)
_w1P = _swP(float(_G3P[1, 1]), float(np.asarray(_c3P["gram4"])[1, 1]))
check("root 1 is the only feasible root: weights >= 0, quote (0.18, 0.63, "
      "0.18)",
      float(_w1P.min()) >= 0.0
      and [round(float(_w), 2) for _w in _w1P] == [0.18, 0.63, 0.18],
      " ".join(f"{float(_w):+.5f}" for _w in _w1P))
check("roots 0, 2, 3 all infeasible in {S=0,1,2} (negative weight)",
      all(float(_swP(float(_G3P[_i, _i]),
                     float(np.asarray(_c3P["gram4"])[_i, _i])).min()) < -1e-3
          for _i in (0, 2, 3)))
_evP = np.linalg.eigvalsh(_G3P[:3, :3])
_evC = np.linalg.eigvalsh(np.asarray(_c0P["gram_cert"], float))
check("three-root invariant spectrum re-derives to {0.037, 2.363, 6.072}, "
      "within 0.007 of the certified spectrum",
      [round(float(_x), 3) for _x in _evP] == [0.037, 2.363, 6.072]
      and float(np.abs(_evP - _evC).max()) < 7e-3,
      f"{' '.join(f'{float(_x):.5f}' for _x in _evP)} "
      f"(max dev {float(np.abs(_evP-_evC).max()):.5f})")
check("roots 0-2 final residuals quote as (3.2, 4.1, 2.6)e-6; root 3 as "
      "3.3e-5",
      [round(float(_r) * 1e6, 1) for _r in _r3P[:3]] == [3.2, 4.1, 2.6]
      and close(float(_r3P[3]), 3.3e-5, 0.02),
      " ".join(f"{float(_r):.2e}" for _r in _r3P))
_mxP = 0.0
for _ln in open("gram7500s4_live.log", encoding="utf-8", errors="replace"):
    _mP = _re.match(r"\[\s*([0-9.]+)s ", _ln)
    if _mP:
        _mxP = max(_mxP, float(_mP.group(1)))
check("stage-4 wall quotes as 14.8 h (log elapsed span)",
      round(_mxP / 3600, 1) == 14.8, f"{_mxP/3600:.3f} h")
check("stage-4 log: all drift gates PASS, sentinel DONE, no PARTIAL",
      (_logP := open("gram7500s4_live.log", encoding="utf-8",
                     errors="replace").read()).count(
          "GATE root0 vs certified: drift -0.00 uHa [PASS]") == 3
      and "GRAM7500S4_DONE" in _logP and "PARTIAL" not in _logP
      and "[FAIL]" not in _logP)
check("md quotes the stage-4 numbers (1.3711, 18.6-fold, 1.1%, 11.59, Var "
      "row, sector row)",
      all(_s in _mdO for _s in
          ("1.3711", "18.6-fold", "1.1% of the 0.285 mHa", "⟨S²⟩ = 11.59",
           "6.92, 3.54, 7.41, 4.41", "(0.82, −0.07, 0.25)",
           "provably carries spin character beyond S = 2")))
check("tex quotes the stage-4 numbers (1.3711, 18.6-fold, 1.1%, 11.59, Var "
      "row, sector row)",
      all(_s in _txO for _s in
          ("1.3711", "18.6-fold", "1.1\\% of the 0.285~mHa",
           "\\ssq = 11.59", "6.92, 3.54, 7.41, 4.41",
           "(0.82, -0.07, 0.25)",
           "provably carries spin character beyond $S = 2$")))

print("\n-- section Q: v5.4 (S^4 on eight states, four-root table, "
      "review-round-3 language) --")
_dQ = np.load("s4states.npz", allow_pickle=True)
_labQ = [str(x) for x in _dQ["labels"]]
_s2Q = np.asarray(_dQ["s2"], float)
_s2tQ = np.asarray(_dQ["s2_theirs"], float)
_s4Q = np.asarray(_dQ["s4"], float)
_varQ = np.asarray(_dQ["var"], float)
_wQ = np.asarray(_dQ["w"], float)


def _iQ(sub):
    for _j, _L in enumerate(_labQ):
        if sub in _L:
            return _j
    raise KeyError(sub)


check("s4states.npz: the eight final pipeline states are present",
      len(_labQ) == 8, f"{len(_labQ)} states")
check("s4states.npz gate: each computed <S^2> matches the shipped "
      "spin_square to 1e-6 (all eight arms)",
      float(np.abs(_s2Q - _s2tQ).max()) < 1e-6,
      f"max |dev| {float(np.abs(_s2Q-_s2tQ).max()):.1e}")
_iT = _iQ("[4Fe-4S] hardware, symm on")
check("[4Fe-4S] hw symm-on is a MEASURED pure triplet: <S^2>=2.000, Var "
      "quotes 3e-6, weights (0,1,0)",
      round(float(_s2Q[_iT]), 3) == 2.000
      and close(float(_varQ[_iT]), 3e-6, 2e-6)
      and [round(float(_x), 3) for _x in _wQ[_iT]] == [0.0, 1.0, 0.0],
      f"<S2> {float(_s2Q[_iT]):.5f} Var {float(_varQ[_iT]):.2e} "
      f"w {tuple(round(float(_x),4) for _x in _wQ[_iT])}")
_iTo = _iQ("[4Fe-4S] hardware, symm off")
check("[4Fe-4S] hw symm-off is a broad high-spin mixture: Var quotes 9.0, "
      "<S^2>=3.00 (the 'from Var=9.0 unmitigated' datum)",
      round(float(_varQ[_iTo]), 0) == 9.0
      and round(float(_s2Q[_iTo]), 2) == 3.00,
      f"<S2> {float(_s2Q[_iTo]):.4f} Var {float(_varQ[_iTo]):.4f}")
_iS = _iQ("[2Fe-2S] set-draw, symm on")
check("[2Fe-2S] set-draw symm-on: <S^2>=2.008, <S^4> quotes 12.05, Var "
      "quotes 8.0, weights (0.67,0.00,0.33), NO triplet weight",
      round(float(_s2Q[_iS]), 3) == 2.008
      and round(float(_s4Q[_iS]), 2) == 12.05
      and round(float(_varQ[_iS]), 1) == 8.0
      and [round(float(_x), 2) for _x in _wQ[_iS]] == [0.67, 0.0, 0.33]
      and abs(float(_wQ[_iS][1])) < 1e-2,
      f"<S2> {float(_s2Q[_iS]):.5f} <S4> {float(_s4Q[_iS]):.4f} "
      f"Var {float(_varQ[_iS]):.4f} w {tuple(round(float(_x),4) for _x in _wQ[_iS])}")
_iSo = _iQ("[2Fe-2S] set-draw, symm off")
check("[2Fe-2S] set-draw symm-off: <S^2> quotes 1.089, Var quotes 1.5",
      round(float(_s2Q[_iSo]), 3) == 1.089
      and round(float(_varQ[_iSo]), 1) == 1.5,
      f"<S2> {float(_s2Q[_iSo]):.5f} Var {float(_varQ[_iSo]):.4f}")
check("scalar-mean fallacy in our own data: set-draw-on and [4Fe-4S]-on both "
      "round to <S^2>=2.0, but their Var differ by >6 orders of magnitude",
      round(float(_s2Q[_iS]), 1) == 2.0 and round(float(_s2Q[_iT]), 1) == 2.0
      and float(_varQ[_iS]) / float(_varQ[_iT]) > 1e6,
      f"Var ratio {float(_varQ[_iS])/float(_varQ[_iT]):.1e}")
_i5 = [_iQ(_L) for _L in (
    "[2Fe-2S] hardware, symm on", "[2Fe-2S] hardware, symm off",
    "[2Fe-2S] template, symm on", "[2Fe-2S] template, symm off",
    "[2Fe-2S] set-draw, symm off")]
check("the other five [2Fe-2S] arms are low-spin mixtures, Var in [0.6, 1.52]",
      all(0.6 <= float(_varQ[_j]) <= 1.52 for _j in _i5),
      " ".join(f"{float(_varQ[_j]):.3f}" for _j in _i5))
_nEigQ = int(sum(1 for _j in range(8) if float(_varQ[_j]) < 1e-4))
check("exactly one of the eight final states is a spin eigenstate "
      "(Var < 1e-4) — the [4Fe-4S] triplet, the wrong one",
      _nEigQ == 1 and float(_varQ[_iT]) < 1e-4, f"{_nEigQ} eigenstate(s)")
_G4dQ = np.diagonal(np.asarray(_c3P["gram4"], float))
check("four-root <S^4> table row quotes as 8.795, 9.127, 29.857, 138.740",
      [round(float(_G4dQ[_j]), 3) for _j in range(4)]
      == [8.795, 9.127, 29.857, 138.740],
      " ".join(f"{float(_G4dQ[_j]):.4f}" for _j in range(4)))
_dEgQ = (_E3P - _E3P[0]) * 1e3
check("four-root DeltaE-above-ground table row quotes as 0, +0.285, +6.52, "
      "+12.54",
      round(float(_dEgQ[1]), 3) == 0.285
      and round(float(_dEgQ[2]), 2) == 6.52
      and round(float(_dEgQ[3]), 2) == 12.54,
      " ".join(f"{float(_x):.4f}" for _x in _dEgQ))

_posMD = ("No audited execution returned the named singlet", "A scalar mean does not identify these "
          "states", "upper pair's per-root attributions shift by", "singlet direction at all",
          "positive semidefinite",
          "no energy advantage over the corresponding uniform-random control",
          "reconstructed from this small in-sector fraction",
          "now confirmed by its second moment (Var(S²) = 3×10⁻⁶",
          "but this is not a triplet", "singlet–quintet mixture",
          "triplet eigenstate, to tol. (S = 1)",
          "exactly one of the eight is a spin eigenstate to measured tolerance",
          "without labeling that root converged", "_s4states.py",
          "0.85) infeasible",
          "In this ablation, the samples do not carry the accuracy",
          "it erases the state-specific meaning assigned to it",
          "into the wrong sector",
          "fig_flagship_s4.png",
          "we could not retrieve the stated 0.1.0 through PyPI or a tagged "
          "GitHub release")
check("md carries every v5.4 review-round-3 correction and new-result phrase",
      all(_s in _mdO for _s in _posMD),
      "; ".join(_s[:28] for _s in _posMD if _s not in _mdO) or "all present")
_posTX = ("No audited execution returned the named singlet", "A scalar mean does not identify these "
          "states", "upper pair's per-root attributions shift by", "singlet direction at all",
          "positive semidefinite",
          "no energy advantage over the corresponding uniform-random control",
          "reconstructed from this small in-sector fraction",
          "now confirmed by its second moment",
          "this is not a triplet", "singlet--quintet mixture",
          "triplet eigenstate, to tol.\\ ($S=1$)",
          "exactly one of the eight is a spin eigenstate to measured tolerance",
          "without labeling that root converged", "\\_s4states.py",
          "0.85)$ infeasible",
          "In this ablation, the samples do not carry the accuracy",
          "it erases the state-specific meaning assigned to it",
          "into the wrong sector",
          "fig\\_flagship\\_s4.png",
          "we could not retrieve the stated 0.1.0 through PyPI or a tagged "
          "GitHub release", "\\includegraphics[width=\\textwidth]{fig_flagship_s4.png}")
check("tex carries every v5.4 review-round-3 correction and new-result phrase",
      all(_s in _txO for _s in _posTX),
      "; ".join(_s[:28] for _s in _posTX if _s not in _txO) or "all present")
_neg = ("neither singlets nor any spin eigenstate", "leaves every",
        "white noise", "manufactured from the remainder",
        "a spin-pure state at all", "exists on neither PyPI nor GitHub",
        "essentially the triplet", "(⟨S²⟩ > 2)",
        "catastroph", "wrong state", "wrong spin state")
check("the reviewer's flagged overstatements are gone from BOTH sources "
      "(negative gate)",
      all(_s not in _mdO and _s not in _txO for _s in _neg),
      "; ".join(_s[:24] for _s in _neg
                if _s in _mdO or _s in _txO) or "all absent")

print("\n-- section R: v5.4.2 (S^6/S^8 moments + LP sector bounds, review "
      "round 6 — measure the objection) --")
_dR = np.load("s6moments.npz", allow_pickle=True)
_labR = [str(x) for x in _dR["labels"]]
_fR = [str(x) for x in _dR["files"]]
_m1R = np.asarray(_dR["m1"], float)
_m2R = np.asarray(_dR["m2"], float)
_m3R = np.asarray(_dR["m3"], float)
_smaxR = np.asarray(_dR["smax"], int)
_w1bR = np.asarray(_dR["w1_bounds"], float)     # (min, max) triplet weight
_wge3R = np.asarray(_dR["wge3_bounds"], float)  # (min, max) sum_{S>=3}


def _iR(sub):
    for _j, _L in enumerate(_labR):
        if sub in _L:
            return _j
    raise KeyError(sub)


check("s6moments.npz: eight states, LP run over the FULL physical sector grid "
      "(S_max = 5 for [2Fe-2S], 9 for [4Fe-4S])",
      len(_labR) == 8
      and all(_smaxR[_j] == 5 for _j in range(8) if "2Fe" in _labR[_j])
      and all(_smaxR[_j] == 9 for _j in range(8) if "4Fe" in _labR[_j]),
      f"{len(_labR)} states, S_max {sorted(set(int(_x) for _x in _smaxR))}")
# cross-kernel: the independent S^6 kernel reproduces the shipped-gauged
# <S^2>,<S^4> of _s4states.py bit-for-bit on all eight states
_pmR = {str(_f): (float(_s2), float(_s4)) for _f, _s2, _s4
        in zip(_dQ["files"], _s2Q, _s4Q)}
_d1R = max(abs(_m1R[_j] - _pmR[_fR[_j]][0]) for _j in range(8))
_d2R = max(abs(_m2R[_j] - _pmR[_fR[_j]][1]) for _j in range(8))
check("independent S^6 kernel reproduces _s4states.py <S^2> and <S^4> on all "
      "eight states to <1e-8 (two-kernel agreement)",
      _d1R < 1e-8 and _d2R < 1e-8,
      f"max |d<S2>| {_d1R:.1e}, max |d<S4>| {_d2R:.1e}")
_m3tab = {"[2Fe-2S] hardware, symm on": 8.50,
          "[2Fe-2S] hardware, symm off": 6.95,
          "[2Fe-2S] template, symm on": 3.96,
          "[2Fe-2S] template, symm off": 4.29,
          "[2Fe-2S] set-draw, symm on": 72.30,
          "[2Fe-2S] set-draw, symm off": 9.92,
          "[4Fe-4S] hardware, symm on": 8.00,
          "[4Fe-4S] hardware, symm off": 144.73}
check("eight <S^6> values quote as the cross-instrument table row "
      "(8.50/6.95/3.96/4.29/72.30/9.92/8.00/144.73)",
      all(round(float(_m3R[_iR(_k)]), 2) == _v for _k, _v in _m3tab.items()),
      " ".join(f"{float(_m3R[_j]):.2f}" for _j in range(8)))
_iSR = _iR("[2Fe-2S] set-draw, symm on")
check("set-draw-on: LP triplet weight w1 < 1e-4 AND beyond-S=2 weight = 0 over "
      "all sectors — the 2:1 singlet-quintet reading is MEASURED, not assumed",
      float(_w1bR[_iSR][1]) < 1e-4 and float(_wge3R[_iSR][1]) < 1e-4,
      f"w1<=[{_w1bR[_iSR][0]:.5f},{_w1bR[_iSR][1]:.5f}] "
      f"wge3<=[{_wge3R[_iSR][0]:.5f},{_wge3R[_iSR][1]:.5f}]")
# the objection, measured: reviewer's {0,1,3} 60%-triplet alternative
# reproduces (m1,m2) exactly but predicts m3 = 120; the state measures 72.30
_altm3 = (3 / 5) * 2 ** 3 + (1 / 15) * 12 ** 3
check("review-round-6 objection answered by measurement: the 60%-triplet "
      "{0,1,3} alternative predicts <S^6>=120, the state measures 72.30",
      round(_altm3, 1) == 120.0
      and round(float(_m3R[_iSR]), 1) == 72.3
      and abs(_altm3 - float(_m3R[_iSR])) > 40.0,
      f"alt {_altm3:.1f} vs measured {float(_m3R[_iSR]):.2f}")
_iTR = _iR("[4Fe-4S] hardware, symm on")
check("[4Fe-4S]-on pure triplet now RIGOROUS: LP triplet weight w1 = 1.000 "
      "(both bounds), consistent with Chebyshev cap Var/4 < 1e-6",
      round(float(_w1bR[_iTR][0]), 3) == 1.000
      and round(float(_w1bR[_iTR][1]), 3) == 1.000
      and float(_varQ[_iT]) / 4 < 1e-6,
      f"w1 [{_w1bR[_iTR][0]:.4f},{_w1bR[_iTR][1]:.4f}]")
_iToR = _iR("[4Fe-4S] hardware, symm off")
check("[4Fe-4S]-off carries a CERTIFIED >=5% weight in sectors S>=3 (LP lower "
      "bound on sum_{S>=3}); triplet w1 ~ 0.45",
      float(_wge3R[_iToR][0]) >= 0.05
      and round(float(_w1bR[_iToR][0]), 2) == 0.45,
      f"wge3>=[{_wge3R[_iToR][0]:.4f},{_wge3R[_iToR][1]:.4f}] "
      f"w1 [{_w1bR[_iToR][0]:.4f},{_w1bR[_iToR][1]:.4f}]")
check("the three symmetrize-on [2Fe-2S] arms match their table triplet-weight "
      "cells: hw-on and set-draw-on < 1e-4, template-on < 2e-4",
      float(_w1bR[_iR("[2Fe-2S] hardware, symm on")][1]) < 1e-4
      and float(_w1bR[_iR("[2Fe-2S] set-draw, symm on")][1]) < 1e-4
      and float(_w1bR[_iR("[2Fe-2S] template, symm on")][1]) < 2e-4,
      f"hw {float(_w1bR[_iR('[2Fe-2S] hardware, symm on')][1]):.1e} "
      f"tmpl {float(_w1bR[_iR('[2Fe-2S] template, symm on')][1]):.1e} "
      f"set {float(_w1bR[_iR('[2Fe-2S] set-draw, symm on')][1]):.1e}")
_posR_md = ("the state measures ⟨S⁶⟩ = 72.30",
            "excluded by the third",
            "admits S ≤ 2 (feasible, not unique)",
            "triplet w₁ (LP, all S)",
            "`_s6moments.py`", "72.30")
check("md carries every S^6/LP measurement phrase",
      all(_s in _mdO for _s in _posR_md),
      "; ".join(_s[:24] for _s in _posR_md if _s not in _mdO) or "all present")
_posR_tx = ("the state measures $\\langle S^6\\rangle = 72.30$",
            "excluded by the third",
            "admits $S\\le2$ (feasible, not unique)",
            "$w_1$ (LP, all $S$)",
            "\\_s6moments.py", "72.30")
check("tex carries every S^6/LP measurement phrase",
      all(_s in _txO for _s in _posR_tx),
      "; ".join(_s[:24] for _s in _posR_tx if _s not in _txO) or "all present")

# ============ S. review-round-7 fold-in (v5.4.3) ============
# The moment-ladder attribution (<S^6> ALONE closes the triplet bound), the
# full pinned singlet-quintet decomposition, <S^8>, the independent high-spin
# gauge (S=3/4/5), and the References / Appendix-B-removal front-matter change.
print("\n-- section S: v5.4.3 (moment-ladder attribution + high-spin gauge, "
      "review round 7) --")
from scipy.optimize import linprog as _lpS  # noqa: E402
_m4S = np.asarray(_dR["m4"], float)
_iSS = _iR("[2Fe-2S] set-draw, symm on")
_SgS = np.arange(0, int(_smaxR[_iSS]) + 1)
_lamS = (_SgS * (_SgS + 1)).astype(float)
_momS = [float(_m1R[_iSS]), float(_m2R[_iSS]), float(_m3R[_iSS]),
         float(_m4S[_iSS])]


def _maxw1S(nmom):
    _A = [np.ones_like(_lamS)]
    _b = [1.0]
    for _k in range(1, nmom + 1):
        _A.append(_lamS ** _k)
        _b.append(_momS[_k - 1])
    _c = np.zeros(len(_lamS))
    _c[1] = -1.0
    _r = _lpS(_c, A_eq=np.array(_A), b_eq=np.array(_b),
              bounds=[(0, 1)] * len(_lamS), method="highs")
    return -float(_r.fun)


_lad4, _lad6, _lad8 = _maxw1S(2), _maxw1S(3), _maxw1S(4)
check("set-draw-on moment ladder: two-moment triplet ceiling ~0.86 COLLAPSES "
      "to 8.4e-5 at <S^6> and 3.0e-5 at <S^8> — the third moment alone closes "
      "the <1e-4 bound (reviewer round-7 point #1)",
      0.84 < _lad4 < 0.88 and _lad6 < 1e-4 and _lad8 < 5e-5 and _lad6 > _lad8,
      f"through S4 {_lad4:.3f}, S6 {_lad6:.2e}, S8 {_lad8:.2e}")
check("set-draw-on full decomposition PINNED: w0 = 0.665 (singlet), "
      "w2 = 0.335 (quintet), triplet < 4e-5, combined S>=3 < 2e-5 (round-7 #3)",
      round(float(_dR["w0_bounds"][_iSS][0]), 3) == 0.665
      and round(float(_dR["w0_bounds"][_iSS][1]), 3) == 0.665
      and float(_w1bR[_iSS][1]) < 4e-5 and float(_wge3R[_iSS][1]) < 2e-5,
      f"w0 [{_dR['w0_bounds'][_iSS][0]:.4f},{_dR['w0_bounds'][_iSS][1]:.4f}] "
      f"w1<={_w1bR[_iSS][1]:.2e} wge3<={_wge3R[_iSS][1]:.2e}")
check("set-draw-on <S^8> measured 434.1 (fourth moment, archived per arm; "
      "reviewer round-7 point #2)",
      round(float(_m4S[_iSS]), 0) == 434.0,
      f"{float(_m4S[_iSS]):.2f}")
_s6src = open("_s6moments.py", encoding="utf-8").read()
check("high-spin gauge G0b present in _s6moments.py: pure S=3/4/5 states via "
      "pyscf contract_ss (independent of the S+ kernel), hard-asserted "
      "(reviewer round-7 point #4)",
      "G0b" in _s6src and "contract_ss" in _s6src
      and "G0b FAIL" in _s6src and "S = 3, 4, 5" in _s6src)
check("both sources disclose the high-spin gauge (S=3/4/5 via contract_ss; "
      "tex escapes the underscore)",
      "S = 3, 4, 5" in _mdO and "contract_ss" in _mdO
      and "contract\\_ss" in _txO and "S = 3/4/5" in _txO)
check("md + tex carry the round-7 ladder / decomposition / calibrated phrasing",
      "collapses the certified triplet ceiling to 8.4×10⁻⁵" in _mdO
      and "w₀ = 0.665 (singlet) and w₂ = 0.335 (quintet)" in _mdO
      and "numerically certified" in _mdO and "numerically certified" in _txO
      and "8.4\\times10^{-5}" in _txO)
check("front-matter References list present with the audited identifiers",
      "## References" in _mdO
      and "arXiv:2405.05068" in _mdO
      and "10.5281/zenodo.21359923" in _mdO)

# ============ T. review-round-8 internal-consistency (v5.4.3, no number changed) ============
# Round 8 audited the paper against its own thesis and found prose that used the
# scalar ⟨S²⟩ mean as a state label — the exact fallacy the paper indicts — plus a
# few protocol-specific findings stated too broadly. These lock the corrections in
# both sources so they cannot regress.
print("\n-- section T: v5.4.3 review-round-8 internal-consistency locks --")
check("round-8 #1: the spin–energy vise no longer calls the low-⟨S²⟩/high-variance "
      "hardware+template arms 'near-singlet'/'nearly clean' (the scalar-mean fallacy "
      "the paper itself indicts); md+tex",
      all(_s not in _mdO for _s in ("come out near-singlet", "already nearly clean",
                                    "near-singlet noiseless", "the near-singlet arms"))
      and all(_s not in _txO for _s in ("come out near-singlet", "already nearly clean",
                                        "near-singlet noiseless", "the near-singlet arms")),
      "fallacy phrasings absent")
check("round-8 #1b: the vise instead reports those arms as spin mixtures with nonzero "
      "variance (low scalar ⟨S²⟩, Var 0.64–1.36); md+tex",
      "low scalar" in _mdO and "0.64" in _mdO
      and "low scalar" in _txO and "0.64" in _txO,
      "variance-aware framing present")
check("round-8 #2: §2.11 heading and the eight-state table caption read 'numerically "
      "certified', not 'rigorous'; md+tex",
      "numerically certified sector bounds" in _mdO
      and "rigorous sector bounds" not in _mdO
      and "numerically certified sector bounds" in _txO
      and "rigorous sector bounds" not in _txO
      and "rigorous over all physical sectors" not in _txO,
      "certified language consistent")
check("round-8 #3: the intro scopes 'not spin eigenstates' with the single measured "
      "triplet exception ([4Fe-4S]-on is a spin eigenstate); md+tex",
      "with a single measured exception" in _mdO
      and "that exception is a triplet" in _mdO
      and "with a single measured exception" in _txO
      and "that exception is a triplet" in _txO,
      "eigenstate claim scoped")
check("round-8 #4: the N₂ QSCI-vs-HCI sentence drops 'without overtaking it' "
      "(contradicted by the −0.6 mHa crossing it then reports); md+tex",
      "without overtaking it" not in _mdO and "with only a nominal" in _mdO
      and "without overtaking it" not in _txO and "with only a nominal" in _txO,
      "overtaking contradiction removed")
check("round-8 #5: the spin-clean claim is scoped to the rebuilt optimized-circuit "
      "leg; raw hardware + CCSD template named as distinct preparations; md+tex",
      "not covered by this reconstruction" in _mdO and "distinct preparations" in _mdO
      and "not covered by this reconstruction" in _txO,
      "state-prep breadth scoped")
check("round-8 #6: the 58.6M-file claim is hedged — 'cannot be a raw multinomial shot "
      "record and is consistent with the deduplicated union'; md+tex",
      "cannot be a raw multinomial shot record" in _mdO
      and "cannot be a raw multinomial shot record" in _txO,
      "data-semantics hedged")
check("round-8 #7: Table 8's [4Fe-4S]-on reading is softened to 'triplet eigenstate, "
      "to tol.' (Var-qualified), not the unqualified 'pure triplet'; md+tex",
      "triplet eigenstate, to tol. (S = 1)" in _mdO
      and "triplet eigenstate, to tol.\\ ($S=1$)" in _txO
      and "pure triplet (S = 1)" not in _mdO,
      "table-cell purity qualified")

# ============ U. review-round-10 successor-scoping lock (v5.4.3) ============
# Round 10 (strongly positive) asked only that the inheritance argument to the
# 10^9-10^10 RIKEN-IBM successor state plainly what it does and does not
# establish, so a defender cannot dismiss the paper by pointing at Fugaku.
# One scoping sentence was added to Section 3; this locks it in both sources.
print("\n-- section U: v5.4.3 review-round-10 successor-scoping lock --")
check("round-10: Section 3 states the successor inheritance is an argument from "
      "shared mechanism, not a measurement, and that the paper does not rest on "
      "it; md+tex",
      "the paper does not rest on it" in _mdO
      and "not a spin measurement of the successor's own states" in _mdO
      and "the paper does not rest on it" in _txO
      and "not a spin measurement of the successor's own states" in _txO,
      "successor-scoping sentence present")

# ============ V. review-round-11 mechanism-hierarchy locks (v5.4.3) ============
# Round 11 (conceptual review, zero defects) yielded three adoptions: (a) the
# Discussion's "Every observation ... is a face of that noncommutation" was
# literally false — the Section 2.1 sampling wall is distribution concentration,
# not broken commutation — now scoped to spin-identity findings with the two
# mechanisms explicitly separated; (b) the representable/present/returned
# three-condition hierarchy adopted (completion achieves #1, the Gram audit
# exhibits #2, only #3 supports a named-state energy); (c) Recommendation 1 now
# scopes spin moments as the FIRST identity certificate, not the complete one.
print("\n-- section V: v5.4.3 review-round-11 mechanism-hierarchy locks --")
check("round-11: 'Every observation' overclaim scoped to spin-identity findings, "
      "sampling wall named a distinct mechanism, the three-condition hierarchy "
      "(representable/present/returned) present, and Recommendation 1 scopes spin "
      "moments as the first-not-complete certificate; md+tex",
      "Every observation in this audit" not in _mdO
      and "Every observation in this audit" not in _txO
      and "Every spin-identity finding" in _mdO
      and "Every spin-identity finding" in _txO
      and "distribution concentration, not broken commutation" in _mdO
      and "distribution concentration, not broken commutation" in _txO
      and "Only the third supports a named-state energy" in _mdO
      and "Only the third supports a named-state energy" in _txO
      and "neither delivers the third" in _mdO
      and "neither delivers the third" in _txO
      and "first identity certificate, not the complete one" in _mdO
      and "first identity certificate, not the complete one" in _txO
      and "passes the certificate declared in advance" in _mdO
      and "passes the certificate declared in advance" in _txO,
      "hierarchy + certificate locks present")

# ============ W. independent-implementation flagship cross-check (v5.5) ============
# Section 2.10 closing paragraph + Methods + Limitations: the tb7500 second-kernel
# run (PyCI-built operator, in-house block Davidson; no PySCF selected-CI
# component) vs the certified flagship triple. The 2 KB result archive ships in
# the zip, so these checks run on reduced clones too.
_tbf = "tb7500_result_n7500.npz"
if os.path.exists(_tbf):
    _tb = np.load(_tbf, allow_pickle=True)
    _tbe = np.asarray(_tb["e"], float)
    _tbs2 = np.asarray(_tb["s2"], float)
    _tbrn = np.asarray(_tb["resid"], float)
    _tbrq = np.asarray(_tb["rq_warm"], float)
    _EC = np.array([-116.591966494, -116.591681685, -116.585446380])
    _SC = np.array([1.37111705, 2.36349461, 4.73801314])
    check("W: archived e_cert/s2_cert equal the stage-4 certified triple",
          np.allclose(np.asarray(_tb["e_cert"], float), _EC, atol=5e-10)
          and np.allclose(np.asarray(_tb["s2_cert"], float), _SC, atol=5e-9),
          f"{np.asarray(_tb['e_cert'])} {np.asarray(_tb['s2_cert'])}")
    check("W: nnz_tri equals the quoted exact combinatorial count",
          int(_tb["nnz_tri"]) == 127975471122, f"{int(_tb['nnz_tri']):,}")
    check("W: paper quotes the count with separators in md and tex",
          "127,975,471,122" in _mdO and "127{,}975{,}471{,}122" in _txO)
    _de = np.abs(_tbe - _EC)
    check("W: energies within (0.4, 0.6, 0.8)e-9 Ha as quoted, all <= 8e-10",
          _de.max() <= 8e-10
          and [round(float(x) * 1e9, 1) for x in _de] == [0.4, 0.6, 0.8],
          f"{_de}")
    check("W: S2 ladder quoted to 6 decimals matches archive",
          [round(float(x), 6) for x in _tbs2] == [1.371065, 2.362748, 4.737714],
          f"{_tbs2}")
    _ds = np.abs(_tbs2 - _SC)
    check("W: |dS2| = (5.2e-5, 7.5e-4, 3.0e-4) as quoted and max <= 7.5e-4",
          _ds.max() <= 7.5e-4
          and [float(f"{x:.1e}") for x in _ds] == [5.2e-5, 7.5e-4, 3.0e-4],
          f"{_ds}")
    check("W: iterations = 38 and max residual quotes 1.25e-5",
          int(_tb["iters"]) == 38 and float(f"{_tbrn.max():.2e}") == 1.25e-5,
          f"iters {int(_tb['iters'])}, rn {_tbrn}")
    _gap = float(_EC[1] - _EC[0])
    check("W: residual fraction of ground gap quotes 4.4%",
          round(100 * float(_tbrn.max()) / _gap, 1) == 4.4,
          f"{100 * float(_tbrn.max()) / _gap:.3f}%")
    _sin = float(_tbrn.max()) / _gap
    _ebound = float(_tbrn.max()) ** 2 / _gap
    _sbound = _sin ** 2 * float(_SC[1] - _SC[0])
    check("W: Davis-Kahan numbers as quoted (sin 0.044, E 5.5e-7, S2 1.9e-3)",
          round(_sin, 3) == 0.044
          and float(f"{_ebound:.1e}") == 5.5e-7
          and float(f"{_sbound:.1e}") == 1.9e-3,
          f"sin {_sin:.4f}, E {_ebound:.2e}, S2 {_sbound:.2e}")
    check("W: containment factors ~700x (energy) and 2.5x (spin)",
          650 <= _ebound / _de.max() <= 750
          and round(float(_sbound / _ds.max()), 1) == 2.5,
          f"E {_ebound/_de.max():.0f}x, S2 {_sbound/_ds.max():.2f}x")
    # The quoted RQ agreements (1.4e-14, 2.0e-13, 1.3e-13) are vs the FULL-
    # precision stage-4 energies (the warm file's e_roots); vs the 9-decimal
    # rounded constants the difference is the rounding itself (~1e-10). Primary
    # source for the quoted values = the shipped run log's G3 gate lines.
    import re as _reW
    _tblog = open("tb7500_live.log", encoding="utf-8", errors="replace").read()
    _g3 = _reW.findall(r"warm root\d: RQ \S+ vs cert \S+ \|d\|=(\S+)", _tblog)
    check("W: run log G3 lines carry |d| = (1.42e-14, 1.99e-13, 1.28e-13), "
          "2 s.f. = the quoted (1.4e-14, 2.0e-13, 1.3e-13)",
          len(_g3) == 3
          and [float(x) for x in _g3] == [1.42e-14, 1.99e-13, 1.28e-13]
          and [float(f"{float(x):.1e}") for x in _g3]
          == [1.4e-14, 2.0e-13, 1.3e-13], f"log |d| {_g3}")
    check("W: archived rq_warm consistent with certified energies at the "
          "9-decimal rounding scale (ties npz to log)",
          np.abs(_tbrq - _EC).max() < 1e-9, f"{np.abs(_tbrq - _EC)}")
    check("W: run archived as PARTIAL-by-residual (E and S2 gates inside, "
          "residual gate outside 1e-7) -- matches the scoped prose",
          bool(_tb["gates_ok"]) is False and _de.max() < 5e-6
          and _ds.max() < 5e-3 and _tbrn.max() > 1e-7)
    for _phr, _phrx in (("1.371065, 2.362748, 4.737714",
                         "1.371065, 2.362748, 4.737714"),
                        ("38 iterations", "38 iterations"),
                        ("4.4%", r"4.4\%")):
        check(f"W: md+tex carry '{_phr}'", _phr in _mdO and _phrx in _txO)
    check("W: md+tex carry the residual and both bounds",
          "1.25×10⁻⁵" in _mdO and r"1.25\times10^{-5}" in _txO
          and "5.5×10⁻⁷" in _mdO and r"5.5\times10^{-7}" in _txO
          and "1.9×10⁻³" in _mdO and r"1.9\times10^{-3}" in _txO)
    check("W: companion tb7500_ck.npz listed in md and tex data availability",
          "`tb7500_ck.npz` (three-root, independent implementation)" in _mdO
          and r"\texttt{tb7500\_ck.npz} (three-root, independent implementation)"
          in _txO)
    check("W: two-builds claim present in md and tex",
          "twice, in two independent builds on two fresh cloud instances" in _mdO
          and "twice, in two independent builds on two fresh cloud instances"
          in _txO)
else:
    print("[skip] W. tb7500_result_n7500.npz not present")

print("\n-- section X: v5.6 penalty-on audit (Section 2.6 — measure the "
      "rebuttal) --")
if os.path.exists("sqdpen_frontier.npz") and os.path.exists("sqdpen_theirs.npz"):
    _fr = np.load("sqdpen_frontier.npz", allow_pickle=True)["frontier"].item()
    _it4 = {round(p["lam"], 3): p for p in _fr["it4"]["pts"]}
    _it1 = {round(p["lam"], 3): p for p in _fr["it1"]["pts"]}
    # paper Table (it4 operating point): (lam, E, err, S2)
    for _lam, _e, _err, _s2 in [(0.0, -116.560107, 45.5, 4.834),
                                (0.1, -116.533031, 72.6, 4.165),
                                (0.2, -116.494963, 110.6, 3.902),
                                (0.5, -116.341427, 264.2, 3.441)]:
        _p = _it4[_lam]
        check(f"X: penalty table it4 lam={_lam} (E, err, S2)",
              close(_p["e_h"], _e, 5e-6) and close(_p["err_mHa"], _err, 0.05)
              and close(_p["s2"], _s2, 5e-4),
              f"E {_p['e_h']:.6f} err {_p['err_mHa']:.3f} S2 {_p['s2']:.4f}")
    _end4, _end1 = _fr["it4"]["pts"][-1], _fr["it1"]["pts"][-1]
    check("X: it4 singlet endpoint +2895 mHa / S2 0 (paper 2.9 Ha, E -113.711)",
          close(_end4["err_mHa"], 2894.7, 1.5) and _end4["s2"] < 1e-3
          and round(_end4["err_mHa"] / 1e3, 1) == 2.9
          and close(_end4["e_h"], -113.711, 5e-4),
          f"err {_end4['err_mHa']:.1f} S2 {_end4['s2']:.4f} E {_end4['e_h']:.6f}")
    check("X: it1 singlet floor +3034 mHa (paper 3.03 Ha)",
          close(_end1["err_mHa"], 3034.5, 1.5)
          and close(_end1["err_mHa"] / 1e3, 3.03, 0.01),
          f"err {_end1['err_mHa']:.1f}")
    check("X: lam=0 eigsh vs pyci agree <1 uHa on both spaces",
          abs(_fr["it4"]["e_pyci"] - _it4[0.0]["e_h"]) * 1e6 < 1.0
          and abs(_fr["it1"]["e_pyci"] - _it1[0.0]["e_h"]) * 1e6 < 1.0,
          f"it4 {abs(_fr['it4']['e_pyci']-_it4[0.0]['e_h'])*1e6:.3f} uHa")
    _errs = [_it4[l]["err_mHa"] for l in (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0)]
    check("X: penalized true energy rises monotonically in lambda (PSD)",
          all(_errs[i] < _errs[i + 1] for i in range(len(_errs) - 1)),
          f"{[round(e,1) for e in _errs]}")
    _th = list(np.load("sqdpen_theirs.npz", allow_pickle=True)["rows"])
    _conv = [r["converged"] for r in _th]
    check("X: all 5 their-kernel penalty arms report converged=False",
          len(_th) == 5 and all(c is not None and not c for c in _conv),
          f"{_conv}")
    _cert = {}
    for _tag in ("it1", "it4"):
        for _p in _fr[_tag]["pts"]:
            _cert[(_tag, round(_p["lam"], 3))] = _p["e_h"] + _p["lam"] * _p["s2"]
    _gaps = [r["pen_rq"] - _cert[(r["tag"], round(r["shift"], 3))] for r in _th]
    check("X: their-kernel penalized RQ sits 0.27-0.33 Ha above certified min",
          all(0.26 <= g <= 0.34 for g in _gaps),
          f"{[round(g,3) for g in _gaps]}")
    for _lam, _err, _s2 in [(0.2, 110.4, 3.93), (0.1, 72.6, 4.18)]:
        _f = f"sqdpen_loop_l{int(round(_lam*100)):02d}.npz"
        if os.path.exists(_f):
            _z = np.load(_f, allow_pickle=True)
            _e, _s = float(_z["best_e"]), float(_z["best_s2"])
            check(f"X: penalized protocol lam={_lam} best (err, S2)",
                  close((_e - E2) * 1e3, _err, 0.6) and close(_s, _s2, 0.02),
                  f"err {(_e-E2)*1e3:.3f} S2 {_s:.4f}")
    if os.path.exists("sqdpen_s3b.npz"):
        _s3b = list(np.load("sqdpen_s3b.npz", allow_pickle=True)["rows"])
        check("X: seeded, their kernel confirms our penalized minimum "
              "(conv=True, overlap 1, dRQ=0)",
              len(_s3b) == 2
              and all(bool(r["conv"]) and abs(r["overlap"] - 1.0) < 1e-6
                      and abs(r["theirs_rq"] - r["ours_rq"]) < 1e-6
                      for r in _s3b),
              f"{[(bool(r['conv']), round(float(r['overlap']),6)) for r in _s3b]}")
    _pen_md = ("run with the penalty on", "2.9 Hartree above the reference",
               "penalty-matched rerun of the published protocol",
               "no competitive singlet", "converged = False` on all five",
               "the deficit is upstream in the subspace",
               "the package's own kernel certifies the frontier")
    _pen_tx = ("run with the penalty on", "2.9~Hartree above the reference",
               "penalty-matched rerun of the published protocol",
               "no competitive singlet", "converged = False} on all five",
               "the deficit is upstream in the subspace",
               "the package's own kernel certifies the frontier")
    check("X: md carries the v5.6 penalty-on phrases",
          all(s in _mdO for s in _pen_md),
          "; ".join(s[:26] for s in _pen_md if s not in _mdO) or "all present")
    check("X: tex carries the v5.6 penalty-on phrases",
          all(s in _txO for s in _pen_tx),
          "; ".join(s[:26] for s in _pen_tx if s not in _txO) or "all present")
    # negative gate: the softened over-claim must be gone from both sources
    check("X: the retracted 'could target the singlet directly' phrase is gone",
          "could target the singlet directly" not in _mdO
          and "could target the singlet directly" not in _txO)
    # -- X2: the flagship's stated SQUARED form at its stated lambda = 0.2
    #    (v5.9 correction: the flagship Methods state the soft constraint
    #    H + lam*[S^2 - s(s+1)]^2 at lam = 0.2 — verified from raw arXiv
    #    source, all versions; the earlier penalty-off claim was wrong).
    if os.path.exists("sqdpen_sq_it4.npz"):
        _sqB = np.load("sqdpen_sq_it4.npz", allow_pickle=True)["blk"].item()
        _sq0, _sqP = _sqB["pts"][0], _sqB["pts"][-1]
        check("X2: squared-run lam=0 anchor reproduces the shipped record",
              close(_sq0["e_h"], -116.560107, 5e-6)
              and close(_sq0["s2"], 4.834, 5e-3),
              f"E {_sq0['e_h']:.6f} S2 {_sq0['s2']:.4f}")
        check("X2: squared lam=0.2 point (paper: +528 mHa at S2 = 3.010, "
              "in-subspace Var 0.009)",
              abs(_sqP["lam"] - 0.2) < 1e-9
              and close(_sqP["e_h"], -116.077625, 3e-4)
              and 527.4 <= _sqP["err_mHa"] <= 528.6
              and close(_sqP["s2"], 3.010, 5e-3)
              and _sqP["var_s2"] < 0.02
              and bool(_sqP["eigsh_conv"]) and _sqP["resid"] < 5e-5
              and "squared" in str(_sqB.get("form", "")),
              f"err {_sqP['err_mHa']:+.3f} S2 {_sqP['s2']:.4f} "
              f"Var {_sqP['var_s2']:.4f} resid {_sqP['resid']:.1e}")
        check("X2: 11.6-fold increase over the unpenalized error (paper "
              "phrase), re-derived",
              11.4 <= _sqP["err_mHa"] / _sq0["err_mHa"] <= 11.8
              and "11.6-fold" in _mdO and "11.6-fold" in _txO,
              f"ratio {_sqP['err_mHa']/_sq0['err_mHa']:.2f}")
        check("X2: algebraic physical-variance floor (S2-2)(6-S2) ~ 3.0 "
              "(adjacent-eigenvalue bound) exceeds 2.9",
              (_sqP["s2"] - 2.0) * (6.0 - _sqP["s2"]) > 2.9,
              f"{(_sqP['s2']-2.0)*(6.0-_sqP['s2']):.4f}")
        _sq_md = ("The flagship's stated form, at the flagship's stated "
                  "strength",
                  "conservation of total spin by a soft constraint in the "
                  "eigenstate solver",
                  "employed at λ = 0.2", "+528 mHa", "3.010",
                  "projected square", "literal square", "leakage operator",
                  "P(S²)²P", "PS²(I − P)S²P", "+2,884 mHa",
                  "Penalty provenance at production scale",
                  "reviewer-prompted correction")
        _sq_tx = ("The flagship's stated form, at the flagship's stated "
                  "strength",
                  "conservation of total spin by a soft constraint in the "
                  "eigenstate solver",
                  "employed at $\\lambda=0.2$", "$+528$~mHa", "3.010",
                  "projected square", "literal square", "leakage operator",
                  "P(S^2)^2P", "PS^2(I-P)S^2P", "$+2{,}884$~mHa",
                  "Penalty provenance at production scale",
                  "reviewer-prompted correction")
        check("X2: md carries the squared-form/provenance phrases",
              all(s in _mdO for s in _sq_md),
              "; ".join(s[:30] for s in _sq_md if s not in _mdO)
              or "all present")
        check("X2: tex carries the squared-form/provenance phrases",
              all(s in _txO for s in _sq_tx),
              "; ".join(s[:30] for s in _sq_tx if s not in _txO)
              or "all present")
        _sq_neg = ("leave the penalty off entirely",
                   "leaves the penalty off entirely",
                   "attributed to the flagship",
                   "sole spin-handling mechanism",
                   "switched off where the utility-scale claims live",
                   "mitigates only by determinant completion")
        check("X2: the retracted penalty-off claims are gone from BOTH "
              "sources (negative gate)",
              all(s not in _mdO and s not in _txO for s in _sq_neg),
              "; ".join(s[:30] for s in _sq_neg
                        if s in _mdO or s in _txO) or "all absent")
        # -- X2b (v5.9.1): the LITERAL compression P(S^2)^2P of the stated
        #    operator (review round 12: _sqdpen_sq.py's ssop-twice is the
        #    PROJECTED square (PS^2P)^2, not the literal P(S^2)^2P; the two
        #    differ by the PSD leakage PS^2(I-P)S^2P = the noncommutation).
        #    The literal square reaches the singlet floor at the stated
        #    lam = 0.2 where the projected square stays contaminated.
        if os.path.exists("sqdpen_true_it4.npz"):
            _tB = np.load("sqdpen_true_it4.npz",
                          allow_pickle=True)["blk"].item()
            _tr = _tB["true"]
            check("X2b: literal square P(S^2)^2P at lam=0.2 reaches the singlet "
                  "floor (paper: S2 = 0.004 at +2,884 mHa)",
                  abs(_tr["lam"] - 0.2) < 1e-9 and _tr["s2"] < 0.02
                  and 2860.0 <= _tr["err_mHa"] <= 2910.0
                  and _tr["var_true"] < 0.1
                  and bool(_tr["conv"]) and _tr["resid"] < 5e-7,
                  f"S2 {_tr['s2']:.4f} err {_tr['err_mHa']:+.1f} "
                  f"var {_tr['var_true']:.4f} resid {_tr['resid']:.1e}")
            check("X2b: the leakage makes the literal square the stronger "
                  "penalty (its S2 << the projected square's at the same lam)",
                  _tr["s2"] < _tB["projsq"]["s2"] - 2.0
                  and _tB["projsq"]["s2"] > 3.0,
                  f"true S2 {_tr['s2']:.4f} vs projsq S2 "
                  f"{_tB['projsq']['s2']:.4f}")
            check("X2b: unpenalized <S^4> ~ 47 (the literal-square penalty "
                  "sees the full fourth moment, not the in-subspace proxy ~23)",
                  46.0 <= _tB["unpen_s4"] <= 48.0
                  and "⟨S⁴⟩ ≈ 47" in _mdO
                  and "\\langle S^4\\rangle \\approx 47" in _txO,
                  f"<S^4> {_tB['unpen_s4']:.3f}")
            check("X2b: both sources frame the two compressions and the "
                  "leakage = [PHP, PS^2P] != 0",
                  "in either compression" in _mdO
                  and "in either compression" in _txO
                  and "leakage operator" in _mdO and "leakage operator" in _txO)
        else:
            print("[skip] X2b. sqdpen_true_it4.npz not present")
        check("X2b: retracted 'solved exactly that operator' over-claim gone "
              "(negative gate)",
              "solved exactly that operator" not in _mdO
              and "solved exactly that operator" not in _txO)
    else:
        print("[skip] X2. sqdpen_sq_it4.npz not present")
else:
    print("[skip] X. sqdpen_frontier.npz / sqdpen_theirs.npz not present")

print("\n-- section Y: v5.7 completion-mechanism table (Section 2.6 — the "
      "degeneracy mechanism, measured per checkpoint) --")
if os.path.exists("degen.npz") and os.path.exists("degen_dense.npz"):
    _dgY = [dict(r) for r in np.load("degen.npz", allow_pickle=True)["recs"]]
    _ddY = np.load("degen_dense.npz", allow_pickle=True)
    _okY = [r for r in _dgY if int(r["n_orig"]) != 1106]
    check("Y: 14 checkpoints re-solved; completed reach 310,058 / 319,189 / "
          "80,755",
          len(_dgY) == 14
          and {310058, 319189, 80755} <= {int(r["n_comp"]) for r in _dgY},
          f"{len(_dgY)} rows, 13 iterative + 1 dense-adjudicated")
    _deY = max([abs(float(r["dE"])) for r in _okY] + [abs(float(_ddY["dE"]))])
    check("Y: |dE0| <= 8e-13 Ha at every checkpoint (dense at the flagged row)",
          _deY <= 8e-13, f"max {_deY:.2e}")
    _spY = max(float(r["split"]) for r in _okY)
    check("Y: iterative pair splitting <= 1.9e-12 Ha on the 13 resolved rows "
          "(tol 1e-9); dense row splits 0",
          _spY <= 1.9e-12 and float(_ddY["split"]) == 0.0, f"max {_spY:.2e}")
    _p2Y = max(float(r["p_lo"]) for r in _okY)
    check("Y: projector spectra: p1 >= 1-1e-13, p2 <= 0.109 (pair = state + "
          "its image)",
          all(float(r["p_hi"]) >= 1 - 1e-13 for r in _okY) and _p2Y <= 0.109
          and float(np.asarray(_ddY["p_spec"], float)[0]) > 1 - 1e-12
          and float(np.asarray(_ddY["p_spec"], float)[1]) < 1e-12,
          f"max p2 {_p2Y:.4f}")
    _waY = max([abs(float(r["w_added"])) for r in _okY]
               + [abs(float(_ddY["w_add"]))])
    check("Y: localized-member weight on added dets < 1e-14 everywhere",
          _waY < 1e-14, f"max {_waY:.2e}")
    _cpY = max([float(r["coup"]) for r in _okY] + [float(_ddY["coup"])])
    check("Y: coupling ||P+ H v|| <= 2.5e-9 Ha everywhere",
          _cpY <= 2.5e-9, f"max {_cpY:.2e}")
    _s2refY = {}
    for _tY in ("fe2s2bs", "fe2s2deep", "fe4s4"):
        for _rY in np.load(f"s2audit_{_tY}.npz", allow_pickle=True)["results"]:
            if "e_symm" in _rY:
                _s2refY[int(_rY["ndets"])] = float(
                    np.asarray(_rY["s2_pyci"])[0])
    _dblkY = max(max(abs(float(_b) - _s2refY[int(r["n_orig"])])
                     for _b in np.asarray(r["s2_block"], float).ravel())
                 for r in _okY)
    _sprY = max(abs(np.asarray(r["s2_block"], float)[-1]
                    - np.asarray(r["s2_block"], float)[0]) for r in _okY)
    check("Y: S2 blocks = [x,x] at the arm's uncompleted root-0 value "
          "(identity to 2e-8; match to 2e-3)",
          _dblkY < 2e-3 and _sprY <= 2e-8,
          f"max |blk-r0| {_dblkY:.1e}, max spread {_sprY:.1e}")
    check("Y: dense adjudication (fe4s4 1106->2097): asym E0 matches stored "
          "record, S2 block [5.2930,5.2930]",
          abs(float(_ddY["E_asym"]) - (-326.1843174466)) < 1e-9
          and bool(np.allclose(np.asarray(_ddY["s2blk"], float), 5.2930,
                               atol=5e-5)),
          f"E_asym {float(_ddY['E_asym']):.10f}")
    check("Y: iterative artifact on record: op.solve split 9.31e-2 on the "
          "same space (one Ritz value per degenerate pair)",
          abs(float(_ddY["iter_split"]) - 9.311e-2) < 2e-4,
          f"{float(_ddY['iter_split']):.4e}")
    _rowY = {int(r["n_comp"]): r for r in _dgY}
    check("Y: headline row 170,448 -> 319,189: split 1.4e-14, S2 block 4.6688",
          float(_rowY[319189]["split"]) <= 2e-14
          and bool(np.allclose(np.asarray(_rowY[319189]["s2_block"], float),
                               4.6688, atol=5e-5)),
          f"split {float(_rowY[319189]['split']):.1e}")
    check("Y: headline row 40,435 -> 80,755: split 2.3e-13, S2 block 7.0044",
          float(_rowY[80755]["split"]) <= 3e-13
          and bool(np.allclose(np.asarray(_rowY[80755]["s2_block"], float),
                               7.0044, atol=5e-5)),
          f"split {float(_rowY[80755]['split']):.1e}")
    _mechmdY = ("The completion mechanism at each checkpoint.",
                "‖P₊Hv‖ ≤ 2.5×10⁻⁹ Ha",
                "splitting ≤ 1.9×10⁻¹² Ha against a solver tolerance of 10⁻⁹",
                "carrying < 10⁻¹⁴ of its weight on the added determinants",
                "One checkpoint required adjudication (†).",
                "dynamically disconnected from that support")
    check("Y: md carries the v5.7 mechanism phrases",
          all(_s in _mdO for _s in _mechmdY),
          "; ".join(_s[:30] for _s in _mechmdY if _s not in _mdO)
          or "all present")
    _mechtxY = ("The completion mechanism at each checkpoint.",
                "$\\|P_+ H v\\| \\le 2.5\\times10^{-9}$~Ha",
                "One checkpoint required adjudication ($\\dagger$).",
                "dynamically disconnected from that support")
    check("Y: tex carries the v5.7 mechanism phrases",
          all(_s in _txO for _s in _mechtxY),
          "; ".join(_s[:30] for _s in _mechtxY if _s not in _txO)
          or "all present")
else:
    print("[skip] Y. degen.npz / degen_dense.npz not present")

print("\n-- section Z: v5.7.1 LP sensitivity stress (Section 2.11 — inflated "
      "moment-uncertainty boxes) --")
if os.path.exists("s6stress.npz"):
    _dZ = np.load("s6stress.npz", allow_pickle=True)
    _labZ = [str(x) for x in _dZ["labels"]]
    _facZ = np.asarray(_dZ["factors"], float)
    _bZ = np.asarray(_dZ["bounds"], float)   # (state, factor, {w0,w1,wge3}, {lo,hi})
    check("Z: stress grid = 8 states x factors {1,10,100,1000}",
          len(_labZ) == 8 and list(_facZ) == [1.0, 10.0, 100.0, 1000.0]
          and _bZ.shape == (8, 4, 3, 2), f"{_bZ.shape}")
    _w1pZ = np.asarray(_dR["w1_bounds"], float)
    _w0pZ = np.asarray(_dR["w0_bounds"], float)
    _g3pZ = np.asarray(_dR["wge3_bounds"], float)
    _labRZ = [str(x) for x in _dR["labels"]]
    _permZ = [_labZ.index(_l) for _l in _labRZ]
    _d1Z = max(np.abs(_bZ[_permZ, 0, 1] - _w1pZ).max(),
               np.abs(_bZ[_permZ, 0, 0] - _w0pZ).max(),
               np.abs(_bZ[_permZ, 0, 2] - _g3pZ).max())
    check("Z: factor-1 stress LP reproduces the production s6moments bounds",
          _d1Z < 1e-9, f"max |d| {_d1Z:.1e}")
    _isdZ = _labZ.index("[2Fe-2S] set-draw, symm on")
    _if4Z = _labZ.index("[4Fe-4S] hardware, symm on")
    _ceilZ = _bZ[_isdZ, :, 1, 1]
    check("Z: set-draw triplet ceiling 3.1e-5 -> 3.7e-5 (10x) -> 1.0e-4 "
          "(100x), 5.4e-4 at 1000x",
          abs(_ceilZ[0] - 3.086e-5) < 2e-7 and abs(_ceilZ[1] - 3.713e-5) < 2e-7
          and abs(_ceilZ[2] - 9.986e-5) < 5e-7
          and abs(_ceilZ[3] - 5.399e-4) < 2e-6,
          " ".join(f"{_x:.3e}" for _x in _ceilZ))
    _w0wZ = _bZ[_isdZ, 2, 0]
    check("Z: set-draw singlet weight confined to [0.6653, 0.6654] at 100x",
          0.66530 <= _w0wZ[0] and _w0wZ[1] <= 0.66545
          and (_w0wZ[1] - _w0wZ[0]) < 1e-4,
          f"[{_w0wZ[0]:.6f}, {_w0wZ[1]:.6f}]")
    check("Z: [4Fe-4S] mitigated triplet floor > 0.99998 at 100x "
          "(wrong-sector purity robust)",
          _bZ[_if4Z, 2, 1, 0] > 0.99998, f"{_bZ[_if4Z, 2, 1, 0]:.6f}")
    _stressmdZ = ("moment-constrained sector bounds",
                  "inflating every box 10× and 100×",
                  "confines its singlet weight to [0.6653, 0.6654]",
                  "The accuracy ordering does not depend on pricing the "
                  "samples.",
                  "no quantum computer sampling a fixed prepared state")
    check("Z: md carries the v5.7.1 stress/language phrases",
          all(_s in _mdO for _s in _stressmdZ),
          "; ".join(_s[:30] for _s in _stressmdZ if _s not in _mdO)
          or "all present")
    _stresstxZ = ("moment-constrained sector bounds",
                  "inflating every box $10\\times$ and $100\\times$",
                  "confines its singlet weight to $[0.6653, 0.6654]$",
                  "The accuracy ordering does not depend on pricing the "
                  "samples.",
                  "no quantum computer sampling a fixed prepared state")
    check("Z: tex carries the v5.7.1 stress/language phrases",
          all(_s in _txO for _s in _stresstxZ),
          "; ".join(_s[:30] for _s in _stresstxZ if _s not in _txO)
          or "all present")
    check("Z: the dropped rhetoric is gone from both sources "
          "(negative gate: 'present or future', 'price-independent')",
          all(_s not in _mdO and _s not in _txO
              for _s in ("present or future", "price-independent")))
else:
    print("[skip] Z. s6stress.npz not present")

print("\n-- section AA: v5.8 threshold-scan plateau (Sections 2.3/2.4, "
      "Methods) + claim table (Table 11) + four-senses terminology note --")
if os.path.exists("thrscan.npz"):
    _dA = np.load("thrscan.npz", allow_pickle=True)
    _rowsA = list(_dA["rows"])
    _nvA = int(_dA["n_vectors"])
    _nrA = sum(int(r["nroots"]) for r in _rowsA)
    _miA = float(_dA["max_intra"]); _mnA = float(_dA["min_inter"])
    _winA = float(_dA["window"]); _identA = bool(_dA["all_identical"])
    check("AA: threshold scan covers 49 stored root vectors (138 roots)",
          _nvA == 49 and _nrA == 138, f"{_nvA} vectors, {_nrA} roots")
    check("AA: separation margin — max intra split 1.876e-12 Ha, min inter "
          "gap 4.79e-6 Ha, 2.55e6-fold window (the quoted 1.9e-12/4.8uHa/2.6e6)",
          abs(_miA - 1.876e-12) < 1e-14 and abs(_mnA - 4.7925e-6) < 1e-8
          and abs(_winA - 2.5549e6) < 5e3,
          f"intra {_miA:.3e}, inter {_mnA:.3e}, window {_winA:.4e}")
    _intraA = [g for r in _rowsA for g in r["gaps"] if g < 2e-6]
    _interA = [g for r in _rowsA for g in r["gaps"] if g >= 2e-6]
    check("AA: margin recomputed from raw per-row gaps matches the stored "
          "aggregates (independent of _thrscan bookkeeping)",
          abs(max(_intraA) - _miA) < 1e-18 and abs(min(_interA) - _mnA) < 1e-15,
          f"intra {max(_intraA):.3e}, inter {min(_interA):.3e}")
    _diffA = [r for r in _rowsA if not r["identical"]]
    _pA = _diffA[0]["partitions"] if len(_diffA) == 1 else None
    check("AA: 0.2/2 uHa + residual-adaptive reproduce the production "
          "partition exactly; exactly one 20-uHa reassignment, at the "
          "unreported fe2s2 D=5053 checkpoint",
          not _identA and len(_diffA) == 1
          and "fe2s2 D=5053" in _diffA[0]["label"]
          and _pA["0.2uHa"] == _pA["2uHa_prod"]
          and _pA["adaptive"] == _pA["2uHa_prod"]
          and _pA["20uHa"] != _pA["2uHa_prod"]
          and [list(x) for x in _pA["20uHa"]] == [[0, 2], [2, 3]],
          f"{len(_diffA)} differ: "
          + (";".join(d["label"] for d in _diffA) or "none"))
    _s2A = _diffA[0]["s2_diag"] if _diffA else None
    check("AA: the two merged D=5053 roots are both mid-mixture (<S2> ~ "
          "2.05, 2.07) — the wandering point, never a reported cluster",
          _s2A is not None and abs(_s2A[0] - 2.050) < 5e-3
          and abs(_s2A[1] - 2.073) < 5e-3,
          f"{[round(x, 3) for x in _s2A[:2]] if _s2A else None}")
    _thrmdA = (
        "changes none of the paper's reported clusters or block spectra",
        "That convention sits on a measured plateau",
        "49 stored root vectors (138 roots)",
        # numbering-agnostic: the JCTC variant renumbers Results, so this
        # pin checks the prose, not the section number (the tex pin below
        # is already agnostic via a LaTeX cross-reference)
        "the wandering point of Section",
        "merge into a cluster the paper never reported as one")
    check("AA: md carries the v5.8 threshold-scan phrases",
          all(_s in _mdO for _s in _thrmdA),
          "; ".join(_s[:30] for _s in _thrmdA if _s not in _mdO)
          or "all present")
    _thrtxA = (
        "changes none of the paper's reported clusters or block spectra",
        "That convention sits on a measured plateau",
        "49 stored root vectors (138 roots)",
        "the wandering point of Section~\\ref{sec:fe2s2}",
        "merge into a cluster the paper never reported as one")
    check("AA: tex carries the v5.8 threshold-scan phrases",
          all(_s in _txO for _s in _thrtxA),
          "; ".join(_s[:30] for _s in _thrtxA if _s not in _txO)
          or "all present")
    _termmdA = ("Terminology: four senses of", "*residual-certified*",
                "*cross-kernel reproduced*", "*artifact-verified*")
    check("AA: md carries the four-senses terminology note",
          all(_s in _mdO for _s in _termmdA),
          "; ".join(_s[:26] for _s in _termmdA if _s not in _mdO)
          or "all present")
    _termtxA = ("Terminology: four senses of", "\\emph{residual-certified}",
                "\\emph{cross-kernel reproduced}", "\\emph{artifact-verified}")
    check("AA: tex carries the four-senses terminology note",
          all(_s in _txO for _s in _termtxA),
          "; ".join(_s[:26] for _s in _termtxA if _s not in _txO)
          or "all present")
    _clmdA = (
        "What each audited source claimed, and what we measured.",
        "upper bounds for the ground-state energy",
        "beyond sizes amenable to exact diagonalization",
        "closure under spin inversion",
        "accuracy comparable to some all-classical approximation methods")
    check("AA: md carries the Table 11 claim-alignment verbatim quotes",
          all(_s in _mdO for _s in _clmdA),
          "; ".join(_s[:34] for _s in _clmdA if _s not in _mdO)
          or "all present")
    _cltxA = _clmdA + ("\\label{tab:claims}",)
    check("AA: tex carries the Table 11 claim-alignment quotes and label",
          all(_s in _txO for _s in _cltxA),
          "; ".join(_s[:34] for _s in _cltxA if _s not in _txO)
          or "all present")
    check("AA: the banned 'leaves every' overstatement is absent from both "
          "sources (v5.8 reword held)",
          "leaves every" not in _mdO and "leaves every" not in _txO)
else:
    print("[skip] AA. thrscan.npz not present")

# ---- final gate: stated check-counts equal this run's actual count ----
# On a clone without the 413 MB Zenodo companion, sections L4 + the two
# warm-provenance checks (19 checks) are skipped by design; the stated
# version-of-record count still refers to the full run.
import re as _reO  # noqa: E402
_wantO = N_CHECKS[0] + 1  # including this very check
if not os.path.exists("flagsolve_n7500_box2.npz"):
    _wantO += 19  # the companion-gated checks a reduced clone skips
_hdr = _reO.search(r"— (\d+)\s*checks, zero failures", _mdO)
_dis = _reO.findall(r"\((\d+) checks, zero failures\)", _mdO)
_txd = _reO.findall(r"\((\d+) checks, zero failures\)", _txO)
check("stated check-count equals the audit's actual count in md header, "
      "md disclosure, and tex disclosure (self-referential gate)",
      _hdr is not None and int(_hdr.group(1)) == _wantO
      and _dis and all(int(x) == _wantO for x in _dis)
      and _txd and all(int(x) == _wantO for x in _txd),
      f"actual {_wantO}; header "
      f"{_hdr.group(1) if _hdr else 'MISSING'}, md {_dis}, tex {_txd}")

print(f"\n{'='*60}\n{N_CHECKS[0]} checks, {len(FAILS)} FAILURES"
      + (f": {FAILS}" if FAILS else " — ALL CLAIMS VERIFIED"))
sys.exit(1 if FAILS else 0)
