"""MECHANISM of the spin-completion degeneracy (paper Section 2.6, "Direct
audit", facts a/b/c) -- answering an external review's request to *demonstrate*,
not assert, why spin-inversion closure produces an exactly degenerate ground
pair {v, v_bar} with the energy unchanged.

For each audited checkpoint (asymmetric selected-CI determinant set D from the
s2audit arms) we form the spin-inversion closure D' = unique(D u swap(D)),
re-solve, and measure, all gauge-safe:

  (a) dE_complete  = E0(D') - E0(D)                 -> ~0 (energy unchanged)
  (b) split_pair   = E1(D') - E0(D')  vs solver tol -> < tol (exactly degenerate)
  (2) P_orig block eigenvalues over the degenerate manifold  -> {~1, ~0}
        (the pair is the state, localized on D, and its image, localized on the
         added support -- NOT a 50/50 bonding/antibonding mix that would split)
  (3) w_added = 1 - p_hi = weight the localized ground puts on newly added dets
  (1) coup    = || P_new . H . v~ ||  (v~ = asymmetric ground embedded, 0 on new)
        -- the off-diagonal Hamiltonian coupling from the ground into the added
           block; this being ~0 is *why* the added dets receive ~0 weight
  (c) S^2 block eigenvalues over the pair -> [x, x] (proportional to identity)

The mechanism: the added spin-image support is H-uncoupled from the localized
ground (coup ~ 0), so the ground stays put (w_added ~ 0); its spin image is then
a distinct state at the SAME energy (spin-inversion symmetry of H), so the pair
is exactly degenerate rather than split. Symmetry alone permits sym/antisym
combinations; the vanishing coupling is what forces the exact degeneracy.

Env: MAXN (skip completed spaces above this; default none). Sentinel: DEGEN_DONE
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np  # noqa: E402
import pyci  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

from _fairfight import popcount64  # noqa: E402

T0 = time.time()
TOL = 1e-9                      # solver tol (matches s2audit)
DEG = 1e-7                      # inversion-pair window (ladder gaps are ~mHa)
MAXN = int(os.environ.get("MAXN", "0"))  # 0 = no cap
OUT = "degen.npz"

FE2 = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
       "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
FE4 = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
       "Clusters/Fe2S2_and_Fe4S4/Fe4S4/fe4s4")
# (system tag, fcidump, npz) -- nelec read from the dump header; npz supplies
# the asymmetric determinant sets
JOBS = [
    ("2Fe2S/bs", FE2, "s2audit_fe2s2bs.npz"),
    ("2Fe2S/deep", FE2, "s2audit_fe2s2deep.npz"),
    ("4Fe4S", FE4, "s2audit_fe4s4.npz"),
]


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def s2_gram_ab(a, b, C, ncas):
    """G_ij = <v_i|S^2|v_j> in the det basis (JW alpha-low gauge), M_s=0."""
    a = np.asarray(a, np.uint64)
    b = np.asarray(b, np.uint64)
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    oa, ob, ov = [], [], []
    for p in range(ncas):
        bit = np.uint64(1 << p)
        low = np.uint64((1 << p) - 1)
        sel = ((b & bit) != 0) & ((a & bit) == 0)
        if not sel.any():
            continue
        a0, b0 = a[sel], b[sel]
        p1 = ((popcount64(a0) + popcount64(b0 & low)) & 1).astype(np.int64)
        p2 = (popcount64(a0 & low) & 1).astype(np.int64)
        sgn = ((1 - 2 * p1) * (1 - 2 * p2)).astype(float)
        oa.append(a0 | bit)
        ob.append(b0 & ~bit)
        ov.append(C[sel] * sgn[:, None])
    k = C.shape[1]
    if not oa:
        return np.zeros((k, k))
    A = np.concatenate(oa)
    B = np.concatenate(ob)
    V = np.concatenate(ov, axis=0)
    _, inv = np.unique(np.stack([A, B], 1), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    nu = int(inv.max()) + 1
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return W.T @ W


def rowkey(ab):
    """Row-wise hashable view of an (n,2) uint64 det array for isin/searchsorted."""
    ab = np.ascontiguousarray(np.asarray(ab, np.uint64))
    return ab.view(np.dtype((np.void, ab.dtype.itemsize * 2))).ravel()


def solve(ham, dets, nel, k):
    w = pyci.fullci_wfn(ham.nbasis, *nel)
    for r_ in dets:
        w.add_det(r_)
    op = pyci.sparse_op(ham, w)
    es, vs = op.solve(n=k, tol=TOL)
    es = np.asarray(es)
    o = np.argsort(es)
    B = w.to_det_array().reshape(len(w), 2)
    return es[o], np.asarray(vs)[o], B, op


def analyze(tag, ham, row, nel):
    da = np.asarray(row["dets_a"], np.uint64)
    db = np.asarray(row["dets_b"], np.uint64)
    Bd = np.stack([da, db], 1)
    comp = np.unique(np.concatenate([Bd, Bd[:, ::-1]]), axis=0)
    ncas = ham.nbasis          # pyci nbasis == NORB (spatial orbitals), not 2x
    n0, n1 = len(Bd), len(comp)
    if MAXN and n1 > MAXN:
        log(f"  {tag} {str(row['tag'])}: skip (completed {n1} > MAXN {MAXN})")
        return None

    # asymmetric ground (its own solved order)
    e_a, v_a, Ba, _ = solve(ham, Bd, nel, 1)
    E_asym, v_asym = float(e_a[0]), v_a[0]

    # completed space, lowest 3
    e_c, v_c, Bc, op2 = solve(ham, comp, nel, 3)
    E0 = float(e_c[0])
    split = float(e_c[1] - e_c[0])
    dE = E0 - E_asym

    # index bookkeeping in completed order
    kc = rowkey(Bc)
    ka = rowkey(Ba)                       # asymmetric solved-order dets
    orig_mask = np.isin(kc, ka)           # completed dets that were in D
    new_mask = ~orig_mask
    n_new = int(new_mask.sum())
    n_selfimg = int(np.isin(ka, rowkey(Bd[:, ::-1])).sum())  # D dets whose swap in D

    # degenerate manifold = roots within DEG of the ground
    man = [i for i in range(len(e_c)) if e_c[i] - E0 < DEG]
    Vman = v_c[man].T                     # (ndets, m)
    m = Vman.shape[1]
    # original-block projector over the manifold: M_ij = <v_i|P_orig|v_j>
    Porig = Vman[orig_mask]
    M = Porig.T @ Porig
    pvals = np.sort(np.linalg.eigvalsh(M))[::-1]   # {p_hi, p_lo, ...}
    p_hi = float(pvals[0])
    w_added = 1.0 - p_hi                  # weight of localized ground off D

    # localized ground (max original-block weight within the manifold)
    wM, UM = np.linalg.eigh(M)
    v_loc = Vman @ UM[:, int(np.argmax(wM))]
    v_loc /= np.linalg.norm(v_loc)
    w_added_loc = float(np.sum(v_loc[new_mask] ** 2))

    # off-diagonal coupling: embed asymmetric ground, 0 on new dets
    vtilde = np.zeros(n1)
    order = np.argsort(kc)
    pos = order[np.searchsorted(kc, ka, sorter=order)]
    vtilde[pos] = v_asym
    Hv = np.asarray(op2.matvec(vtilde))
    coup = float(np.linalg.norm(Hv[new_mask]))          # || P_new H v~ ||
    resid = float(np.linalg.norm(Hv - E_asym * vtilde))  # full residual

    # S^2 over the manifold (invariant block)
    G = s2_gram_ab(Bc[:, 0], Bc[:, 1], Vman, ncas)
    s2vals = np.sort(np.linalg.eigvalsh(G))
    s2diag = float(G[0, 0])
    g_off = float(np.abs(G - np.diag(np.diag(G))).max())

    rec = dict(
        system=tag, ck=str(row["tag"]), n_orig=n0, n_comp=n1, n_new=n_new,
        n_selfimg=n_selfimg, E_asym=E_asym, E0=E0, dE=dE, split=split,
        m_manifold=m, p_hi=p_hi, p_lo=float(pvals[-1]), w_added=w_added,
        w_added_loc=w_added_loc, coup=coup, resid=resid,
        s2_block=s2vals.tolist(), s2_diag=s2diag, g_off=g_off, tol=TOL)
    log(f"  {tag:10s} {str(row['tag']):11s} N {n0:>7}->{n1:>7} (+{n_new:>6}) "
        f"dE {dE:+.1e} split {split:.1e} p{{{p_hi:.5f},{pvals[-1]:.5f}}} "
        f"w_add {w_added:.1e} coup {coup:.1e} S2blk "
        f"[{s2vals[0]:.3f},{s2vals[-1]:.3f}] off {g_off:.0e}")
    return rec


def main():
    done = {}
    if os.path.exists(OUT):
        prev = np.load(OUT, allow_pickle=True)["recs"]
        for r in prev:
            done[(r["system"], r["ck"])] = r
        log(f"resume: {len(done)} checkpoints already done")
    recs = list(done.values())
    for tag, fcid, npz in JOBS:
        if not os.path.exists(npz):
            log(f"{npz} MISSING -- skip {tag}")
            continue
        hdr = fcidump.read(fcid)
        ne = int(hdr["NELEC"])
        nel = (ne // 2, ne - ne // 2)
        ham = pyci.hamiltonian(fcid)
        assert ham.nbasis == int(hdr["NORB"]), (ham.nbasis, hdr["NORB"])
        log(f"   {tag}: norb {ham.nbasis} nelec {nel}")
        R = list(np.load(npz, allow_pickle=True)["results"])
        R = [r for r in R if "dets_a" in r]
        R.sort(key=lambda r: r["ndets"])
        log(f"== {tag}: {npz} ({len(R)} symm checkpoints) ==")
        for row in R:
            if (tag, str(row["tag"])) in done:
                continue
            rec = analyze(tag, ham, row, nel)
            if rec is None:
                continue
            recs.append(rec)
            np.savez(OUT, recs=np.array(recs, dtype=object))
    log(f"saved {OUT}: {len(recs)} checkpoints")
    print("DEGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
