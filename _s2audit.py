"""Det-space <S^2> + spin audits of the Li-Chan iron-sulfur FCIDUMPs under
the qsci-benchmarks HCI protocols.

Practice under audit: selected-CI / SQD pipelines target the M_s=0 sector and
follow the lowest root with no S^2 control. In [2Fe-2S]/[4Fe-4S] the
Heisenberg ladder packs all S states within ~tens of mHa — the same scale as
the accuracy claims.

Part A (validation, diazene r090): <S^2> = ||S+ v||^2 in the det basis
  (JW alpha-low gauge) must reproduce pyscf labels on Engine vectors and on
  pyci vectors (gauge equivalence VERIFIED 2026-07-09: exact match), and the
  3-root S^2 Gram must be ~diag(2,0,0).
Part B (FCID dump, THEIR eps-ladder): seed = aufbau (their run_hci.py) or
  bs (their run_hci_dorbs.py hardcoded broken-symmetry pair, fe2s2 only);
  at checkpoints solve lowest-3 (pyci, sorted) -> S^2 Gram over the roots
  (diag = per-root <S^2>; eigvals of quasi-degenerate blocks = the spin
  content the manifold actually CONTAINS, rotation-invariant); Engine
  re-solve cross-check where feasible (2*ncas<=62 and ndets<=ENGMAX).
Env: FCID, TAG, SEEDMODE (aufbau|bs), SYMM, DET_CAP, CKPTS, ENGMAX (70000),
  ENGINE (auto), NT, SKIPA, PARTA_ONLY.  Sentinel: S2AUDIT_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import pyci  # noqa: E402
from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

import _fairfight as ff  # noqa: E402
from _fairfight import popcount64  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def s2_gram_ab(a, b, C, ncas):
    """G_ij = <v_i|S^2|v_j> = <S+ v_i|S+ v_j> for M_s=0 vectors (S^2 =
    S-S+ + Sz(Sz+1), Sz terms vanish). Dets as separate (alpha, beta) uint64
    word arrays; global spin-orbital order: alpha block ascending then beta
    block ascending (pyscf/pyci/JW-alpha-low gauge). C: (ndets,) or
    (ndets, k) columns. Works for any ncas <= 63. Returns (k, k)."""
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
        # parities MUST be cast signed before 1-2*par: popcount64 preserves
        # uint64, and uint64(1)-uint64(2) wraps to 2^64-1 (shredded the
        # first matrix run -- S^2 ~ 1e77)
        # annihilate beta_p: occupied below global index ncas+p =
        # all alphas + betas below p
        p1 = ((popcount64(a0) + popcount64(b0 & low)) & 1).astype(np.int64)
        # create alpha_p on the result: alphas below p
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


def s2_expect_ab(a, b, c, ncas):
    return float(s2_gram_ab(a, b, c, ncas)[0, 0])


def _split(dets, ncas):
    dets = np.asarray(dets, np.int64)
    am = (np.int64(1) << ncas) - np.int64(1)
    return (dets & am).astype(np.uint64), ((dets >> ncas) & am).astype(np.uint64)


def s2_gram(dets, C, ncas):
    a, b = _split(dets, ncas)
    return s2_gram_ab(a, b, C, ncas)


def s2_expect(dets, c, ncas):
    return float(s2_gram(dets, c, ncas)[0, 0])


def spin_clusters(es, G, tol=2e-6):
    """Quasi-degenerate root clusters (|dE| < tol chained) and the eigvals of
    the S^2 block inside each: within a degenerate manifold the solver returns
    an arbitrary rotation, so per-root <S^2> is gauge; the block eigvals are
    the invariant spin content the manifold contains."""
    out = []
    i = 0
    while i < len(es):
        j = i + 1
        while j < len(es) and es[j] - es[j - 1] < tol:
            j += 1
        if j - i > 1:
            out.append((i, j, np.linalg.eigvalsh(G[i:j, i:j])))
        i = j
    return out


# ---------------- part A: validation on diazene r090 -------------------------
if os.environ.get("SKIPA", "0") != "1":
    d = np.load("dz_r090.00.npz")
    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    roots, all_dets = ff.exact_reference(d, nroots=3)
    want = [((m * m - 1) / 4) for _, m, _ in roots]
    eng = ff.Engine(d["h1"], d["h2"], float(d["ecore"]), ncas, na, nb)
    ws, V, dd = eng.solve_k(all_dets, k=3)
    G = s2_gram(dd, V, ncas)
    got = [float(G[i, i]) for i in range(3)]
    off = float(np.abs(G - np.diag(np.diag(G))).max())
    log("partA Engine: " + "  ".join(
        f"r{i}: E{ws[i]:.6f} S2 {got[i]:.4f} (pyscf {want[i]:.4f})"
        for i in range(3)) + f"  gram offdiag max {off:.1e}")
    if not all(abs(g - w) < 1e-2 for g, w in zip(got, want)) or off > 1e-6:
        import sys
        print(f"S2 GAUGE FAIL: got {got} want {want} offdiag {off:.2e}",
              file=sys.stderr, flush=True)
        raise SystemExit(1)
    log("partA OK")
    if os.environ.get("PARTA_ONLY", "0") == "1":
        raise SystemExit(0)

# ---------------- part B: dump + their protocol + spin resolution ------------
FCID = os.environ.get(
    "FCID", "_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
            "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
TAG = os.environ.get("TAG", "fe2s2")
SEEDMODE = os.environ.get("SEEDMODE", "aufbau")  # NOT "SEED" — that's
SYMM = os.environ.get("SYMM", "0") == "1"        # _fairfight's int RNG seed
DET_CAP = int(os.environ.get("DET_CAP", "60000"))
CKPTS = [int(x) for x in os.environ.get(
    "CKPTS", "1000,5000,15000,40000").split(",")]
ENGMAX = int(os.environ.get("ENGMAX", "30000"))  # 70k Engine solve + symm
# stacked RAM peaks killed the WSL VM (2026-07-09); certification at <=30k
# ckpts + prior 43k/68k evidence suffices

fd = fcidump.read(FCID)
NC = int(fd["NORB"])
nel = (int(fd["NELEC"]) // 2, int(fd["NELEC"]) // 2)
log(f"partB[{TAG}]: {FCID}: norb {NC} nelec {nel} ms2 {fd.get('MS2')} "
    f"seed {SEEDMODE} symm {SYMM} cap {DET_CAP}")
ham2 = pyci.hamiltonian(FCID)
USE_ENG = os.environ.get("ENGINE", "auto")
USE_ENG = (2 * NC <= 62) if USE_ENG == "auto" else USE_ENG == "1"
eng2 = None
if USE_ENG:
    eng2 = ff.Engine(fd["H1"], ao2mo.restore(1, fd["H2"], NC),
                     float(fd["ECORE"]), NC, *nel)
    log(f"partB: Engine built ({len(eng2.co)} offdiag paulis)")

wfn = pyci.fullci_wfn(ham2.nbasis, *nel)
results = []
PART = f"s2audit_{TAG}.partial.npz"
eps0 = 1e-1
if os.path.exists(PART):
    # WSL VM deaths are a proven failure mode: every checkpoint dumps the
    # det set, so a killed arm resumes from its last checkpoint (the ladder
    # state is exactly (det set, eps) -- deterministic continuation).
    results = list(np.load(PART, allow_pickle=True)["results"])
    last = results[-1]
    for r_ in np.stack([np.asarray(last["dets_a"], np.uint64),
                        np.asarray(last["dets_b"], np.uint64)], 1):
        wfn.add_det(r_)
    eps0 = float(last["eps"]) * 0.7943282347242815
    log(f"partB RESUME from ckpt {last['tag']}: ndets={len(wfn)} "
        f"-> eps {eps0:.3e}")
elif SEEDMODE == "bs":
    assert NC == 20, "bs seed pair is the fe2s2 run_hci_dorbs.py one"
    wfn.add_det(np.array([0b11111111111110000011, 0b11000001111111111111],
                         np.uint64))
    wfn.add_det(np.array([0b11000001111111111111, 0b11111111111110000011],
                         np.uint64))
else:
    wfn.add_hartreefock_det()
op = pyci.sparse_op(ham2, wfn)
e_vals, e_vecs = op.solve(n=1, tol=1e-9)
if not results:
    sa0 = wfn.to_det_array().reshape(len(wfn), 2)
    s2seed = s2_expect_ab(sa0[:, 0], sa0[:, 1], e_vecs[0], NC)
    log(f"partB: seed ({len(wfn)} dets) E {e_vals[0]:.6f} S2 {s2seed:.3f}")


def _fmt(es, s2p, cl):
    gaps = "  ".join(f"r{i}: dE {(es[i]-es[0])*1e3:+8.3f} mHa "
                     f"S2 {s2p[i]:.3f}" for i in range(len(s2p)))
    return gaps + "".join(
        f"  [r{i}..r{j-1} deg, S2-span {np.round(ev, 3).tolist()}]"
        for i, j, ev in cl)


def checkpoint(tag, eps_now):
    Bd = wfn.to_det_array().reshape(len(wfn), 2)
    esn, vsn = op.solve(n=3, tol=1e-9)
    o = np.argsort(np.asarray(esn))
    esn, vsn = np.asarray(esn)[o], np.asarray(vsn)[o]
    G = s2_gram_ab(Bd[:, 0], Bd[:, 1], np.asarray(vsn).T, NC)
    s2p = [float(G[i, i]) for i in range(len(esn))]
    row = {"tag": tag, "eps": eps_now, "ndets": len(wfn),
           "e_pyci": [float(x) for x in esn],
           "s2_pyci": s2p, "g_pyci": G,
           "dets_a": Bd[:, 0].copy(), "dets_b": Bd[:, 1].copy()}
    xchk = ""
    if USE_ENG and len(wfn) <= ENGMAX:
        dets64 = np.unique(Bd[:, 0].astype(np.int64)
                           | (Bd[:, 1].astype(np.int64) << NC))
        ws3, V3, dd3 = eng2.solve_k(dets64, k=3)
        Ge = s2_gram(dd3, V3, NC)
        row["e_eng"] = [float(x) for x in ws3]
        row["s2_eng"] = [float(Ge[i, i]) for i in range(V3.shape[1])]
        xchk = f" (eng d {(ws3[0]-esn[0])*1e6:+.1f} uHa)"
    log(f"partB ckpt {tag}: ndets={len(wfn):>7} E0 {esn[0]:.6f}{xchk}  "
        + _fmt(esn, s2p, spin_clusters(esn, G)))
    if SYMM:
        # IBM's stated mitigation: complete the det set under spin inversion
        # (alpha<->beta). Permits a singlet — does it PRODUCE one? The
        # S2-span of the (typically exactly degenerate) ground pair answers
        # invariantly: min-span ~0 means a singlet EXISTS in the manifold.
        uni = np.unique(np.concatenate([Bd, Bd[:, ::-1]]), axis=0)
        w2 = pyci.fullci_wfn(ham2.nbasis, *nel)
        for r_ in uni:
            w2.add_det(r_)
        op2 = pyci.sparse_op(ham2, w2)
        es2, vs2 = op2.solve(n=3, tol=1e-9)
        o2 = np.argsort(np.asarray(es2))
        es2, vs2 = np.asarray(es2)[o2], np.asarray(vs2)[o2]
        B2 = w2.to_det_array().reshape(len(w2), 2)
        G2 = s2_gram_ab(B2[:, 0], B2[:, 1], np.asarray(vs2).T, NC)
        s22 = [float(G2[i, i]) for i in range(len(es2))]
        row["ndets_symm"] = len(w2)
        row["e_symm"] = [float(x) for x in es2]
        row["s2_symm"] = s22
        row["g_symm"] = G2
        log(f"    symm ckpt : ndets={len(w2):>7} E0 {es2[0]:.6f}  "
            + _fmt(es2, s22, spin_clusters(es2, G2)))
    results.append(row)
    np.savez(PART, results=np.array(results, dtype=object))


eps, next_ck = eps0, 0
while next_ck < len(CKPTS) and CKPTS[next_ck] <= len(wfn):
    next_ck += 1
while eps >= 1e-6 and len(wfn) <= DET_CAP:
    added = True
    while added and len(wfn) <= DET_CAP:
        added = pyci.add_hci(ham2, wfn, e_vecs[0], eps=eps)
        op.update(ham2, wfn)
        e_vals, e_vecs = op.solve(n=1)
    log(f"partB: eps={eps:.3e} ndets={len(wfn):>7} E={e_vals[0]:.6f}")
    if next_ck < len(CKPTS) and len(wfn) >= CKPTS[next_ck]:
        checkpoint(f"eps={eps:.2e}", eps)
        while next_ck < len(CKPTS) and len(wfn) >= CKPTS[next_ck]:
            next_ck += 1
    eps *= 0.7943282347242815
if not results or results[-1]["ndets"] != len(wfn):
    checkpoint("final", eps)
np.savez(f"s2audit_{TAG}.npz", results=np.array(results, dtype=object))
if os.path.exists(PART):
    os.remove(PART)
log(f"saved s2audit_{TAG}.npz")
print("S2AUDIT_DONE", flush=True)
