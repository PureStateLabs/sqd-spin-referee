"""HARDENED uniform-null-control statistics (reviewer response, v2).

The reviewer's objection to the original "uniform matches or beats hardware"
claim: it leaned on best-of-K, an order statistic. This script replaces it
with a full distributional comparison — means, Welch 95% CIs, Mann-Whitney U
per matched condition (one-sided retained for continuity, TWO-sided reported;
batches within a condition are disjoint shot groups) — built on the archive's
actual evidence structure, which we verified row-by-row:

  * hw eigenstate-B file is a strict SUBSET of the C file (all 59 B rows
    appear bit-identically in C; C carries 7 more), and the B and C uniform
    files are byte-identical -> B and C are ONE evidence stream, counted once
    (as-published per-file comparisons are also printed for completeness).
  * hw A shares zero rows with B|C, and at common dimensions its energies
    differ from C's -> A is a separate eigenstate; its rows must NEVER be
    pooled with C's at a shared dimension (that would mix eigenstates).

Design: two independent streams, each hardware file paired with ITS OWN
uniform null -- A (5 matched dims) and B/C (4 matched dims, hardware = C) --
9 (stream, dimension) conditions total. Dimensions WITHIN a stream re-process
the same released sample pool, so conditions are repeated measures on two
independent evidence streams, not nine independent experimental units; no
global significance level is attached across conditions (the direction is
reported descriptively; the legacy sign test is stored, labeled legacy).

Sign convention: d_mean = hw_mean - un_mean in mHa; positive means the
uniform null lands LOWER (better, since all energies here lie above the
exact reference) than the quantum-hardware samples at the same dimension.

Output: ibmuniform_stats.npz + printed table. Sentinel: IBMUNIFORM2_DONE
"""
import numpy as np
from scipy import stats

BASE = "_ibm_data/sqd_data_repository-main/experiments"
E2 = -116.6056091   # their DMRG reference, 2Fe-2S


def load(path):
    r = np.loadtxt(path, skiprows=2)
    return r.reshape(-1, r.shape[-1])


def welch_ci(a, b, conf=0.95):
    """CI on mean(a) - mean(b), Welch (unequal var/n)."""
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, 0.0, 0.0
    df = se**4 / ((va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1))
    t = stats.t.ppf(0.5 + conf / 2, df)
    return ma - mb, (ma - mb) - t * se, (ma - mb) + t * se


hw = {t: load(f"{BASE}/2Fe-2S/sqd_hardware_energetics/"
              f"energy-variance_data_SQD_eigenstate_{t}.txt")
      for t in "ABC"}
un = {t: load(f"{BASE}/2Fe-2S/sqd_on_uniform_distribution/"
              f"energy-variance_data_SQD_eigenstate_{t}_uniform.txt")
      for t in "ABC"}

# --- verify the evidence structure we claim (hard assertions) ---
rowset = lambda r: set(map(tuple, r))
assert rowset(hw["B"]) <= rowset(hw["C"]), "B hw not a subset of C hw!"
assert np.array_equal(un["B"], un["C"]), "B/C uniform files differ!"
assert not (rowset(hw["A"]) & rowset(hw["C"])), "A shares rows with C!"
n_extra_c = len(rowset(hw["C"]) - rowset(hw["B"]))
print(f"verified: hw B subset of hw C (C adds {n_extra_c} rows); "
      f"un B == un C; hw A disjoint from B|C")
print("=> two independent streams: A, and B/C (hardware = C = B-union-C)\n")

STREAMS = [("A", hw["A"], un["A"]), ("B/C", hw["C"], un["B"])]

rows = []
out = {}
print(f"{'strm':4s} {'dim':>10s} {'n_hw':>4s} {'n_un':>4s} "
      f"{'hw_mean_mHa':>11s} {'un_mean_mHa':>11s} "
      f"{'hw_sd':>5s} {'un_sd':>5s} {'d_mean':>7s} "
      f"{'95%CI':>16s} {'MWU_p1':>7s} {'MWU_p2':>7s} {'best_d':>7s}")
print("-" * 106)
n_un_ahead_mean = 0
n_un_ahead_sig = 0
n_matched = 0

for tag, H, U in STREAMS:
    dims = sorted(set(np.unique(H[:, 2]).astype(np.int64)) &
                  set(np.unique(U[:, 2]).astype(np.int64)))
    for d in dims:
        eh = H[H[:, 2].astype(np.int64) == d, 1]
        eu = U[U[:, 2].astype(np.int64) == d, 1]
        if len(eh) < 2 or len(eu) < 2:
            continue
        n_matched += 1
        dmean, lo, hi = welch_ci(eh, eu)  # hw - un (Ha)
        # one-sided MWU retained for continuity; TWO-sided is what the paper
        # reports (the direction was not pre-registered)
        mwu = stats.mannwhitneyu(eu, eh, alternative="less").pvalue
        mwu2 = stats.mannwhitneyu(eu, eh, alternative="two-sided").pvalue
        best_d = (eh.min() - eu.min()) * 1e3   # order statistic, label as such
        if eu.mean() <= eh.mean():
            n_un_ahead_mean += 1
        if mwu2 < 0.05:
            n_un_ahead_sig += 1
        print(f"{tag:4s} {d:10d} {len(eh):4d} {len(eu):4d} "
              f"{(eh.mean()-E2)*1e3:11.2f} {(eu.mean()-E2)*1e3:11.2f} "
              f"{eh.std(ddof=1)*1e3:5.1f} {eu.std(ddof=1)*1e3:5.1f} "
              f"{dmean*1e3:7.2f} [{lo*1e3:6.2f},{hi*1e3:6.2f}] "
              f"{mwu:7.4f} {mwu2:7.4f} {best_d:7.2f}")
        rows.append((0 if tag == "A" else 1, d, len(eh), len(eu),
                     (eh.mean() - E2) * 1e3, (eu.mean() - E2) * 1e3,
                     eh.std(ddof=1) * 1e3, eu.std(ddof=1) * 1e3,
                     dmean * 1e3, lo * 1e3, hi * 1e3, mwu, best_d, mwu2))

R = np.array(rows, float)
out["rows"] = R
out["n_matched"] = n_matched
out["n_un_ahead_mean"] = n_un_ahead_mean
out["n_un_ahead_sig"] = n_un_ahead_sig

print("-" * 106)
print(f"matched (stream, dimension) conditions: {n_matched} "
      f"(repeated measures on 2 independent streams)")
print(f"uniform mean <= hardware mean: {n_un_ahead_mean}/{n_matched} "
      f"(descriptive; no global test attached -- within-stream conditions "
      f"share the released sample pool)")
print(f"uniform below hardware at p<0.05 (two-sided MWU, per condition): "
      f"{n_un_ahead_sig}/{n_matched}")
d_means = R[:, 8]
sign_p = stats.binomtest(int((d_means >= 0).sum()), n_matched, 0.5,
                         alternative="greater").pvalue
print(f"[legacy, not cited: sign test across conditions p = {sign_p:.2e}; "
      f"invalid as a global level for the reason above]")
out["sign_test_p_legacy"] = sign_p
out["n_un_ahead_sig_two_sided"] = n_un_ahead_sig
a_adv = R[R[:, 0] == 0, 8]
bc_adv = R[R[:, 0] == 1, 8]
sd_all = np.concatenate([R[:, 6], R[:, 7]])
print(f"uniform mean advantage: A stream {a_adv.min():.1f}-{a_adv.max():.1f} "
      f"mHa over {len(a_adv)} dims; B/C stream {bc_adv.min():.1f}-"
      f"{bc_adv.max():.1f} mHa over {len(bc_adv)} dims")
print(f"per-group batch SD range: {sd_all.min():.1f}-{sd_all.max():.1f} mHa")
out["a_adv"] = a_adv
out["bc_adv"] = bc_adv
out["sd_range"] = np.array([sd_all.min(), sd_all.max()])

# as-published per-file view (B and C listed separately) for completeness
print("\nas-published per-file comparisons (B and C files listed separately;"
      "\nNOT independent -- B/C share data as verified above):")
for t in "ABC":
    H, U = hw[t], un[t]
    dims = sorted(set(np.unique(H[:, 2]).astype(np.int64)) &
                  set(np.unique(U[:, 2]).astype(np.int64)))
    ahead = 0
    for d in dims:
        eh = H[H[:, 2].astype(np.int64) == d, 1]
        eu = U[U[:, 2].astype(np.int64) == d, 1]
        if eu.mean() <= eh.mean():
            ahead += 1
    print(f"  file {t}: uniform mean ahead at {ahead}/{len(dims)} "
          f"matched dims")

np.savez("ibmuniform_stats.npz", **out)
print("IBMUNIFORM2_DONE", flush=True)
