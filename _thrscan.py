"""_thrscan.py -- quasi-degeneracy threshold sensitivity scan (review R17).

The Gram instrument clusters quasi-degenerate roots by the chained rule
E_{i+1} - E_i < tol with tol = 2 uHa (Methods "Spin audit";
`_s2audit.spin_clusters`, `_paperaudit` section-B re-derivation,
`_s2fig`). Review round 17 asked whether the reported cluster structure
depends on that convention.

This scan re-derives every cluster partition from the archived root
energies alone, at tol in {0.2, 2, 20} uHa and under a residual-adaptive
criterion -- roots i, i+1 cluster when their gap is below the larger of
their per-root eigenvalue uncertainties, taken as the solve tolerance
where residuals are not archived and as the Kato--Temple-style bound
r_i^2 / gap_i where they are. A chained partition changes between
thresholds t1 < t2 iff some consecutive gap lies in [t1, t2), so the
decisive quantities are the two sides of the separation margin: the
largest split inside any production cluster vs the smallest gap between
distinct clusters, over every stored root vector the audit reads. Where
any partition differs, the affected block spectra are recomputed from
the stored Gram matrices at each threshold for adjudication.

Sources (read-only): s2audit_{fe2s2,fe2s2bs,fe2s2deep,fe4s4}.npz
(e_pyci/g_pyci and e_symm/g_symm at every checkpoint),
degen.npz (14 completion-mechanism records: pair split, residual, tol),
degen_dense.npz (the six dense eigenvalues at the adjudicated
checkpoint), gram7500.npz / gram7500_s4.npz (flagship three- and
four-root certified blocks with measured residuals).
Output: thrscan.npz + printed summary.  Desktop, seconds.
"""
import numpy as np

TOLS = [("0.2uHa", 2e-7), ("2uHa_prod", 2e-6), ("20uHa", 2e-5)]
PROD = 2e-6

rows = []
intra, inter = [], []   # consecutive gaps below / at-or-above PROD


def partition_fixed(es, tol):
    parts, i = [], 0
    while i < len(es):
        j = i + 1
        while j < len(es) and es[j] - es[j - 1] < tol:
            j += 1
        parts.append((i, j))
        i = j
    return tuple(parts)


def partition_adaptive(es, delta):
    """Chain roots i, i+1 when gap < max(delta_i, delta_{i+1})."""
    parts, i = [], 0
    while i < len(es):
        j = i + 1
        while (j < len(es)
               and es[j] - es[j - 1] < max(delta[j], delta[j - 1])):
            j += 1
        parts.append((i, j))
        i = j
    return tuple(parts)


def deltas(es, resid, tol):
    """Per-root eigenvalue uncertainty: r^2/gap if residuals archived,
    else the solve tolerance."""
    n = len(es)
    if resid is None:
        return [float(tol)] * n
    out = []
    for i in range(n):
        gaps = [abs(es[j] - es[i]) for j in range(n) if j != i]
        g = max(min(gaps), 1e-300)
        out.append(float(resid[i]) ** 2 / g)
    return out


def block_eigs(G, parts):
    if G is None:
        return []
    G = np.asarray(G, float)
    return [(i, j, [float(x) for x in np.linalg.eigvalsh(G[i:j, i:j])])
            for i, j in parts if j - i > 1]


def add_vec(label, family, es, G=None, resid=None, tol=1e-9, s2=None):
    es = np.asarray(es, float)
    gaps = np.diff(es)
    for g in gaps:
        (intra if g < PROD else inter).append(float(g))
    prod = partition_fixed(es, PROD)
    parts = {name: partition_fixed(es, t) for name, t in TOLS}
    parts["adaptive"] = partition_adaptive(es, deltas(es, resid, tol))
    same = all(p == prod for p in parts.values())
    if s2 is None and G is not None:
        s2 = np.diag(np.asarray(G, float))
    row = dict(label=label, family=family, nroots=int(len(es)),
               gaps=[float(g) for g in gaps], identical=bool(same),
               s2_diag=None if s2 is None else [float(x) for x in s2],
               partitions={k: [list(p) for p in v]
                           for k, v in parts.items()},
               blocks_prod=block_eigs(G, prod))
    if not same:
        row["blocks_by_tol"] = {k: block_eigs(G, v)
                                for k, v in parts.items()}
    rows.append(row)
    return row


# ---- trajectory + completion archives (the 2 uHa rule's home) -----------
for tag in ("fe2s2", "fe2s2bs", "fe2s2deep", "fe4s4"):
    rs = list(np.load(f"s2audit_{tag}.npz", allow_pickle=True)["results"])
    for r in rs:
        add_vec(f"{tag} D={int(r['ndets'])} orig", "trajectory",
                r["e_pyci"], r.get("g_pyci"), tol=1e-9,
                s2=r.get("s2_pyci"))
        if "e_symm" in r:
            row = add_vec(f"{tag} D={int(r['ndets_symm'])} completed",
                          "completion", r["e_symm"], r.get("g_symm"),
                          tol=1e-9, s2=r.get("s2_symm"))
            # reproduction gate: chained production rule must agree with
            # the audit's pair condition on the same stored vector
            es = np.asarray(r["e_symm"], float)
            pair_audit = len(es) > 1 and es[1] - es[0] < PROD
            pair_chain = row["partitions"]["2uHa_prod"][0][1] >= 2
            assert pair_audit == pair_chain, (tag, r["ndets"])

# ---- completion-mechanism re-solve records (v5.7 table) -----------------
for rec in np.load("degen.npz", allow_pickle=True)["recs"]:
    if int(rec["m_manifold"]) < 2:
        continue
    e0 = float(rec["E0"])
    add_vec(f"degen {rec['system']} {int(rec['n_orig'])}->"
            f"{int(rec['n_comp'])}", "mechanism",
            [e0, e0 + float(rec["split"])],
            resid=[float(rec["resid"])] * 2, tol=float(rec["tol"]))

dd = np.load("degen_dense.npz")
add_vec("degen fe4s4 1106->2097 dense (adjudicated)", "mechanism",
        dd["e_dense"], tol=1e-12)

# ---- flagship certified blocks (explicit manifolds, scanned anyway) -----
g3 = np.load("gram7500.npz")
add_vec("flagship n7500 three-root (stage 3)", "flagship",
        g3["e_rayleigh"], g3["gram"], resid=g3["resid"])
g4 = np.load("gram7500_s4.npz")
add_vec("flagship n7500 four-root (stage 4)", "flagship",
        g4["e_rayleigh"], g4["gram"], resid=g4["resid"])

# ---- aggregate + verdict ------------------------------------------------
max_intra = max(intra)
min_inter = min(inter)
window = min_inter / max_intra
all_same = all(r["identical"] for r in rows)
n_clustered = sum(1 for r in rows
                  if any(j - i > 1
                         for i, j in r["partitions"]["2uHa_prod"]))

print(f"vectors scanned      : {len(rows)}  "
      f"({sum(r['nroots'] for r in rows)} roots; "
      f"families: trajectory/completion/mechanism/flagship)")
print(f"clustered at 2 uHa   : {n_clustered} vectors carry a "
      f"multi-root cluster")
print(f"max intra-cluster gap: {max_intra:.3e} Ha")
print(f"min inter-cluster gap: {min_inter:.3e} Ha")
print(f"separation window    : {window:.2e}x")
print(f"partitions identical (0.2/2/20 uHa + adaptive): {all_same}")
for r in rows:
    if not r["identical"]:
        print("  DIFFERS:", r["label"], r["partitions"])
        print("    gaps:", [f"{g:.3e}" for g in r["gaps"]],
              " s2_diag:", r["s2_diag"])
        for k, b in r.get("blocks_by_tol", {}).items():
            print("    ", k, b)

np.savez("thrscan.npz",
         rows=np.array(rows, dtype=object),
         tols=np.array([t for _, t in TOLS]),
         max_intra=max_intra, min_inter=min_inter,
         window=window, all_identical=all_same,
         n_vectors=len(rows), n_clustered=n_clustered)
print("saved thrscan.npz")
print("THRSCAN_DONE" if all_same or rows else "THRSCAN_INCOMPLETE")
