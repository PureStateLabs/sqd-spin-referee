"""S3b addendum: seed THEIR penalized kernel with OUR converged penalized
ground state and let their code judge it.

S3 measured that pyscf kernel_fixed_space + fix_spin_(ss=0) fails to find the
penalized ground state on the captured it1 soup subspace at BOTH the shipped
default knobs and max_cycle=600/tol=1e-10 (conv=False, penalized Rayleigh
quotient 0.27-0.33 Ha above our certified minimum, both shifts). The failure
is structural (initial guess / preconditioner), so "run their solver longer"
cannot cross-certify the frontier. This closes the loop the other way:
re-derive our frontier vectors at lam = 0.1, 0.2 (same continuation as the
campaign; asserted to match the frontier npz), hand each to THEIR kernel as
ci0 with the SAME penalty, and record what their code does with it. If their
kernel keeps/refines the state (penRQ <= ours + eps, overlap ~ 1), their own
solver endorses our minimum -- 'you reimplemented it wrong' is closed.

Read-only wrt campaign outputs; writes sqdpen_s3b.npz. Sentinel: S3B_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo, fci  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402
from pyscf.fci import direct_spin1, selected_ci  # noqa: E402
from pyscf.fci.selected_ci import (  # noqa: E402
    _all_linkstr_index, _as_SCIvector, contract_ss, kernel_fixed_space,
)
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

T0 = time.time()
FCID = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
        "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
E_EXACT = -116.6056091
NC, NELEC = 20, (15, 15)
MAXCYC = int(os.environ.get("MAXCYC", "600"))


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
H2E = ao2mo.restore(1, direct_spin1.absorb_h1e(H1, ERI, NC, NELEC, .5), NC)

z = np.load("sqdpen_spaces.npz", allow_pickle=True)
cap = list(z["cap"])
it1 = min((r for r in cap if r["iter"] == 1), key=lambda r: r["e_tot"])
strs = (np.asarray(it1["strs_a"], dtype=np.int64),
        np.asarray(it1["strs_b"], dtype=np.int64))
na, nb = len(strs[0]), len(strs[1])
n = na * nb
link = _all_linkstr_index(strs, NC, NELEC)
log(f"it1 space {na}x{nb}={n:,}")


def hop(c):
    v = _as_SCIvector(np.asarray(c).reshape(na, nb), strs)
    return selected_ci.contract_2e(H2E, v, NC, NELEC, link).ravel()


def ssop(c):
    v = _as_SCIvector(np.asarray(c).reshape(na, nb), strs)
    return contract_ss(v, NC, NELEC).ravel()


def point(v):
    v = np.asarray(v).ravel()
    v = v / np.linalg.norm(v)
    hv, sv = hop(v), ssop(v)
    return float(v @ hv) + ECORE, float(v @ sv)


def solve_pen(lam, v0, tol=1e-9):
    def mv(c):
        out = hop(c)
        if lam:
            out = out + lam * ssop(c)
        return out
    A = LinearOperator((n, n), matvec=mv, dtype=float)
    th, vec = eigsh(A, k=1, which="SA", v0=v0, tol=tol, maxiter=5000,
                    ncv=min(n - 1, 64))
    vec = vec[:, 0]
    resid = float(np.linalg.norm(mv(vec) - th[0] * vec))
    return vec, float(th[0]) + ECORE, resid


# frontier anchor points (from the campaign log / npz) for identity asserts
fr = np.load("sqdpen_frontier.npz", allow_pickle=True)["frontier"].item()
anchors = {p["lam"]: p for p in fr["it1"]["pts"]}

# re-derive our frontier vectors by the same continuation
myci0 = fci.selected_ci.SelectedCI()
_, sv0 = kernel_fixed_space(myci0, H1, ERI, NC, NELEC, strs)
vec, th, resid = solve_pen(0.0, np.asarray(sv0).ravel())
e0, s20 = point(vec)
log(f"lam=0: E {e0:.6f} S2 {s20:.4f} resid {resid:.1e}")
assert abs(e0 - anchors[0.0]["e_h"]) < 5e-6

rows = []
for lam in [0.1, 0.2]:
    vec, th, resid = solve_pen(lam, vec)
    e, s2 = point(vec)
    a = anchors[lam]
    log(f"lam={lam}: ours E {e:.6f} S2 {s2:.4f} resid {resid:.1e} "
        f"(frontier {a['e_h']:.6f}/{a['s2']:.4f})")
    assert abs(e - a["e_h"]) < 1e-5 and abs(s2 - a["s2"]) < 1e-3, \
        "re-derived vector does not match the frontier point"
    ours_rq = e + lam * s2

    myci = fci.selected_ci.SelectedCI()
    myci = fci.addons.fix_spin_(myci, ss=0.0, shift=lam)
    ci0 = _as_SCIvector(vec.reshape(na, nb).copy(), strs)
    t1 = time.time()
    _, sv = kernel_fixed_space(myci, H1, ERI, NC, NELEC, strs, ci0=ci0,
                               max_cycle=MAXCYC, tol=1e-10)
    wall = time.time() - t1
    conv = bool(np.all(myci.converged)) \
        if hasattr(myci, "converged") else None
    w = np.asarray(sv).ravel()
    w = w / np.linalg.norm(w)
    e_t, s2_t = point(w)
    rq_t = e_t + lam * s2_t
    ovl = abs(float(w @ (vec / np.linalg.norm(vec))))
    log(f"lam={lam}: THEIR kernel seeded with ours -> E {e_t:.6f} "
        f"(err {(e_t-E_EXACT)*1e3:+.3f}) S2 {s2_t:.4f} conv={conv} "
        f"penRQ {rq_t:.6f} (ours {ours_rq:.6f}, d {rq_t-ours_rq:+.2e}) "
        f"overlap {ovl:.6f} [{wall:.0f}s]")
    rows.append({"lam": lam, "ours_e": e, "ours_s2": s2, "ours_rq": ours_rq,
                 "theirs_e": e_t, "theirs_s2": s2_t, "theirs_rq": rq_t,
                 "conv": conv, "overlap": ovl, "wall_s": wall,
                 "resid_ours": resid})

np.savez("sqdpen_s3b.npz", rows=np.array(rows, dtype=object))
print("S3B_DONE", flush=True)
