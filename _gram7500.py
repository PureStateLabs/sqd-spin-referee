"""Flagship multi-root S2-Gram audit at IBM's published dimension (Tier 3).

Review response: 'apply the rotation-invariant instrument to a converged
low-energy subspace' AT the flagship dimension 7500x7500 = 5.625e7. Block-
solves the lowest NROOTS of the projected Hamiltonian on the top-N exact-
marginal construction with pyscf's matrix-free machinery (the same
kernel_fixed_space path their solve_fermion calls), warm-started from the
banked box-2 ground vector, through a LADDER OF TOLERANCE CHECKPOINTS
(davidson tol 1e-6 -> 1e-7 -> 1e-8, i.e. residual ~1e-3 -> 3e-4 -> 1e-4).
At every checkpoint it reports, per root: Ritz value, explicit residual norm
||Hv - (v.Hv)v||, Rayleigh quotient, pairwise gaps, <S2> diagonal, and the
rotation-invariant S2-Gram spectrum over the manifold — reviewer items 1-5.

Warm-start disclosure: root 0 is seeded with the banked converged-to-3.5e-4
ground vector (methodologically clean here: the headline energy is already
banked cold-start; this run characterizes the manifold and answers 'does its
S2 drift under further polishing'). Excited roots start from lowest-hdiag
unit vectors orthogonalized against the seed.

Gates (all before the expensive block solve):
  G1 warm strings == freshly built top-N construction (bit-exact)
  G2 Rayleigh quotient of warm vector == banked npz energy to 5 uHa
     (validates integrals + strings + hop wiring end-to-end)
  G3 <S2> of warm vector == banked npz s2 to 1e-3 (validates Gram code)
After stage 1: root-0 energy must be <= banked + 5 uHa (variational).

Resume: if {OUT}_restart.npz exists, completed stages are skipped and the
block restarts from the saved vectors (watchdog-death resilience).

Env: N (7500), NROOTS (3), NT (8), WARM (flagsolve_n7500_box2.npz),
     STAGES ("1e-6,1e-7,1e-8"), STAGE_MAXCYC ("150,150,200"),
     WALL_BUDGET_S (0 = unlimited; skip next stage past this)
Out: gram{N}_ck{i}.npz per stage + gram{N}_restart.npz + gram{N}.npz
Sentinel: GRAM7500_DONE (or GRAM7500_PARTIAL if wall budget cut stages)
"""
import os
import resource
import time

import numpy as np

NT = int(os.environ.get("NT", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("PYSCF_MAX_MEMORY", os.environ.get("MAXMEM", "11000"))

from pyscf import ao2mo  # noqa: E402
from pyscf import fci  # noqa: E402
from pyscf.fci import direct_spin1, selected_ci  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
E_EXACT = -116.6056091
NC = 20
NELEC = (15, 15)
FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")

N = int(os.environ.get("N", "7500"))
NROOTS = int(os.environ.get("NROOTS", "3"))
WARM = os.environ.get("WARM", "flagsolve_n7500_box2.npz")
STAGES = [float(x) for x in
          os.environ.get("STAGES", "1e-6,1e-7,1e-8").split(",")]
STAGE_MAXCYC = [int(x) for x in
                os.environ.get("STAGE_MAXCYC", "150,150,200").split(",")]
WALL_BUDGET_S = float(os.environ.get("WALL_BUDGET_S", "0"))
OUT = f"gram{N}"

# SWAR popcount (verbatim _fairfight.py, inlined: that module drags in the
# full indrajala package, which the box venv doesn't have)
M1 = np.uint64(0x5555555555555555)
M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def popcount64(a):
    a = a - ((a >> np.uint64(1)) & M1)
    a = (a & M2) + ((a >> np.uint64(2)) & M2)
    a = (a + (a >> np.uint64(4))) & M4
    return (a * H01) >> np.uint64(56)


def log(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time()-T0:8.1f}s rss{rss:6.2f}G] {m}", flush=True)


def s2_gram_ab(a, b, C, ncas):
    """G_ij = <v_i|S^2|v_j> = <S+ v_i|S+ v_j> for M_s=0 vectors (verbatim
    from _s2audit.py / _gram2000.py). C: (ndets,) or (ndets, k)."""
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


# ---- construction: identical to _flagsolve.py / _gram2000.py ----
z = np.load("lucjopt_marginals.npz")
ra, rb = z["ra"], z["rb"]
strs = z["strs"]
sa_all = strs[ra] if int(ra.max()) < 32767 else ra
sb_all = strs[rb] if int(rb.max()) < 32767 else rb
sa = np.sort(sa_all[:N].astype(np.int64))
sb = np.sort(sb_all[:N].astype(np.int64))
na = nb = N
ci_strs = (sa, sb)
log(f"strings: top-{N}/{len(sa_all)} per spin, dim {N*N:,}")

fd = fcidump.read(FCID)
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
ECORE = float(fd.get("ECORE", 0.0))
log(f"THEIR MO integrals loaded (ecore {ECORE:.6f})")

# ---- warm vector + gates G1-G3 ----
w = np.load(WARM)
assert np.array_equal(np.asarray(w["ci_strs_a"], np.int64), sa), "G1a FAIL"
assert np.array_equal(np.asarray(w["ci_strs_b"], np.int64), sb), "G1b FAIL"
v0 = np.asarray(w["amps"], float).ravel()
v0 /= np.linalg.norm(v0)
e_bank, s2_bank = float(w["e"]), float(w["s2"])
log(f"G1 PASS warm strings bit-identical ({WARM}, banked E {e_bank:.9f} "
    f"S2 {s2_bank:.4f})")

# ---- hop (verbatim kernel_fixed_space internals, pyscf 2.13.1) ----
myci = fci.selected_ci.SelectedCI()
h2e = direct_spin1.absorb_h1e(H1, ERI, NC, NELEC, .5)
h2e = ao2mo.restore(1, h2e, NC)
link_index = selected_ci._all_linkstr_index(ci_strs, NC, NELEC)
hdiag = myci.make_hdiag(H1, ERI, ci_strs, NC, NELEC, compress=True)
log("h2e + link_index + hdiag built")


def hop(c):
    hc = myci.contract_2e(h2e, selected_ci._as_SCIvector(
        c.reshape(na, nb), ci_strs), NC, NELEC, link_index)
    return np.asarray(hc).ravel()


t0 = time.time()
hv0 = hop(v0)
e_ray0 = float(v0 @ hv0) + ECORE
r0 = float(np.linalg.norm(hv0 - (e_ray0 - ECORE) * v0))
log(f"warm hop: {time.time()-t0:,.0f}s  Rayleigh {e_ray0:.9f}  "
    f"resid {r0:.3e}")
de = abs(e_ray0 - e_bank)
tag = "PASS" if de < 5e-6 else "FAIL"
log(f"G2 vs banked npz energy: dE {de*1e6:.2f} uHa [{tag}]")
assert tag == "PASS", "G2 FAIL: hop/integrals/strings wiring mismatch"

a_words = np.repeat(sa.astype(np.uint64), nb)
b_words = np.tile(sb.astype(np.uint64), na)
t0 = time.time()
s2_warm = float(s2_gram_ab(a_words, b_words, v0, NC)[0, 0])
ds = abs(s2_warm - s2_bank)
tag = "PASS" if ds < 1e-3 else "FAIL"
log(f"G3 warm <S2> {s2_warm:.4f} vs banked {s2_bank:.4f} "
    f"[{tag}, {time.time()-t0:,.0f}s]")
assert tag == "PASS", "G3 FAIL: Gram code disagrees with banked S2"

# ---- initial block: warm root0 + lowest-hdiag unit guesses ----
RESTART = f"{OUT}_restart.npz"
stages_done = 0
if os.path.exists(RESTART):
    rz = np.load(RESTART)
    X = np.asarray(rz["vecs"], float)
    stages_done = int(rz["stages_done"])
    log(f"RESUME from {RESTART}: {X.shape[0]} vecs, "
        f"{stages_done} stages already done")
    assert X.shape == (NROOTS, na * nb)
else:
    hd = np.asarray(hdiag, float).ravel()
    order = np.argsort(hd)
    guesses = [v0]
    for idx in order:
        if len(guesses) == NROOTS:
            break
        g = np.zeros(na * nb)
        g[idx] = 1.0
        for u in guesses:
            g -= u * float(u @ g)
        nrm = np.linalg.norm(g)
        if nrm > 0.3:
            guesses.append(g / nrm)
    X = np.vstack(guesses)
    log(f"initial block: warm root0 + {NROOTS-1} lowest-hdiag guesses")

# ---- staged block solve with checkpoints ----
partial = False
e_roots = None
for si, (tol, mc) in enumerate(zip(STAGES, STAGE_MAXCYC)):
    if si < stages_done:
        log(f"stage {si+1} (tol {tol:g}) already done -- skip")
        continue
    if WALL_BUDGET_S and (time.time() - T0) > WALL_BUDGET_S:
        log(f"WALL BUDGET exceeded before stage {si+1} -- stopping here")
        partial = True
        break
    log(f"=== stage {si+1}/{len(STAGES)}: kernel_fixed_space nroots={NROOTS} "
        f"tol={tol:g} max_cycle={mc} NT={NT} ===")
    ci0 = selected_ci._as_SCIvector(X.copy(), ci_strs)
    t0 = time.time()
    e_roots, vecs = fci.selected_ci.kernel_fixed_space(
        myci, H1, ERI, NC, NELEC, ci_strs, ci0=ci0,
        nroots=NROOTS, tol=tol, max_cycle=mc, verbose=5)
    wall = time.time() - t0
    e_roots = np.atleast_1d(np.asarray(e_roots, float))
    if not isinstance(vecs, (list, tuple)):
        vecs = [vecs]
    X = np.vstack([np.asarray(v).ravel() for v in vecs])
    log(f"stage {si+1} solved in {wall:,.0f}s")

    # per-root diagnostics: Ritz, Rayleigh, explicit residual, gaps
    e_ray = np.zeros(NROOTS)
    resid = np.zeros(NROOTS)
    t1 = time.time()
    for i in range(NROOTS):
        hv = hop(X[i])
        e_ray[i] = float(X[i] @ hv) + ECORE
        resid[i] = float(np.linalg.norm(hv - (e_ray[i] - ECORE) * X[i]))
    log(f"residual hops: {time.time()-t1:,.0f}s")
    for i in range(NROOTS):
        log(f"  root {i}: Ritz {e_roots[i]:.9f}  Rayleigh {e_ray[i]:.9f}  "
            f"|r| {resid[i]:.3e}  err {(e_roots[i]-E_EXACT)*1e3:+9.3f} mHa"
            + ("" if i == 0 else
               f"  gap-to-prev {(e_roots[i]-e_roots[i-1])*1e3:+.3f} mHa"))
    if si == 0:
        d0 = e_roots[0] - e_bank
        tag = "PASS" if d0 < 5e-6 else "FAIL"
        log(f"GATE root0 <= banked: drift {d0*1e6:+.2f} uHa [{tag}]")

    t1 = time.time()
    G = s2_gram_ab(a_words, b_words, X.T, NC)
    ev = np.linalg.eigvalsh(G)
    log(f"Gram done in {time.time()-t1:,.0f}s")
    log(f"  per-root <S2> diag: {[f'{G[i,i]:.4f}' for i in range(NROOTS)]}")
    log(f"  Gram eigenvalues:   {[f'{x:.4f}' for x in ev]}")
    log(f"  min Gram eigenvalue (any-rotation floor): {ev[0]:.4f}")

    np.savez_compressed(
        f"{OUT}_ck{si+1}.npz", n=N, nroots=NROOTS, tol=tol, max_cycle=mc,
        e_roots=e_roots, e_rayleigh=e_ray, resid=resid, gram=G, gram_eigs=ev,
        wall_s=wall, nt=NT, e_exact=E_EXACT, e_bank=e_bank, s2_bank=s2_bank,
        s2_warm=s2_warm, warm_resid=r0)
    np.savez(RESTART, vecs=X, stages_done=si + 1, e_roots=e_roots, n=N,
             nroots=NROOTS)
    log(f"saved {OUT}_ck{si+1}.npz + restart")

if e_roots is not None and not partial:
    # consolidated final artifact = last checkpoint under the canonical name
    last = max(i + 1 for i in range(len(STAGES)) if i >= 0
               and os.path.exists(f"{OUT}_ck{i+1}.npz"))
    dat = dict(np.load(f"{OUT}_ck{last}.npz"))
    np.savez_compressed(f"{OUT}.npz", **dat)
    log(f"saved {OUT}.npz (= ck{last})")

print("GRAM7500_PARTIAL" if partial else "GRAM7500_DONE", flush=True)
