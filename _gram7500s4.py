"""Stage-4 flagship certification (review round 2, Tier 3c + 3a).

Three upgrades over the shipped gram7500 run, all warm from its converged
3-root block (gram7500_restart.npz):
  (i)  residual push: davidson tol 1e-10 => per-root residual target 1e-5
       (the review's ||r0||/(E1-E0) = 0.21 concern -> ~0.03);
  (ii) a 4th root: measures the gap ABOVE the certified block (the
       denominator a block-level Davis-Kahan bound actually needs) and its
       spin identity;
  (iii) S^2 second moments (Tier 3a): G4_ij = <S^2 v_i|S^2 v_j> via double
       S+ application => per-root Var(S^2) = G4_ii - G_ii^2 and constrained
       spin-sector weights (reviewer point: compressed Gram eigenvalues are
       not full sector weights — variances pin them).

Moment identity (M_s = 0):  W = S+ V (deduped M_s=1 image), G = W^T W;
W2 = S+ W (M_s=2);  G4 = W2^T W2 + 2 G.   Validated on H4/STO-3G FCI:
eigenstate Var < 1e-14, mixture Var = 4 s^2 c^2 analytic (gate G0 re-runs
this in-box before anything expensive).

Stages (per-stage checkpoint + restart; WALL_BUDGET_S guard between):
  s4ck0: S^4 moments on the certified vectors AS PUBLISHED (banked early)
  s4ck1: nroots=3 tol=1e-10 max_cycle=120  — residual push
  s4ck2: nroots=4 tol=1e-8  max_cycle=150  — bring in root 3
  s4ck3: nroots=4 tol=1e-10 max_cycle=150  — polish; final moments
Gates: G0 (H4 moment gauge), G1 (strings == construction, via warm-npz
strings of the ORIGINAL warm file check being inherited — here: construction
rebuilt identically), G2 (Rayleigh(warm root0) == certified e_roots[0] to
2 uHa), G3 (S2-Gram of warm block == certified gram to 1e-6 elementwise).

Env: N (7500), NT (8), WALL_BUDGET_S (0 = unlimited)
Out: gram7500_s4ck{0..3}.npz + gram7500_s4_restart.npz + gram7500_s4.npz
Sentinel: GRAM7500S4_DONE / GRAM7500S4_PARTIAL
"""
import os
import resource
import time

import numpy as np

NT = int(os.environ.get("NT", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("PYSCF_MAX_MEMORY", os.environ.get("MAXMEM", "11000"))

from pyscf import ao2mo  # noqa: E402
from pyscf import fci, gto, scf  # noqa: E402
from pyscf.fci import cistring, direct_spin1, selected_ci  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

T0 = time.time()
E_EXACT = -116.6056091
NC = 20
NELEC = (15, 15)
FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")

N = int(os.environ.get("N", "7500"))
WALL_BUDGET_S = float(os.environ.get("WALL_BUDGET_S", "0"))
WARM_RESTART = "gram7500_restart.npz"
WARM_SUMMARY = "gram7500.npz"
OUT = "gram7500_s4"
STAGEPLAN = [(3, 1e-10, 120), (4, 1e-8, 150), (4, 1e-10, 150)]

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


def splus_image(a, b, C, ncas):
    """S+ = sum_p a^dag_{p,alpha} a_{p,beta} applied to vectors C on dets
    (a, b); returns deduped (A, B, W). Phase logic verbatim from the shipped
    s2_gram_ab (_gram7500.py); only change: the image is returned."""
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
        return (np.zeros(0, np.uint64), np.zeros(0, np.uint64),
                np.zeros((0, k)))
    A = np.concatenate(oa)
    B = np.concatenate(ob)
    V = np.concatenate(ov, axis=0)
    pair = np.stack([A, B], 1)
    del A, B
    uniq, inv = np.unique(pair, axis=0, return_inverse=True)
    del pair
    inv = np.asarray(inv).reshape(-1)
    nu = uniq.shape[0]
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return uniq[:, 0].copy(), uniq[:, 1].copy(), W


def spin_moments(a, b, C, ncas):
    """(G, G4) at M_s=0: G = <v_i|S^2|v_j>, G4 = <S^2 v_i|S^2 v_j>."""
    A, B, W = splus_image(a, b, C, ncas)
    G = W.T @ W
    _, _, W2 = splus_image(A, B, W, ncas)
    G4 = W2.T @ W2 + 2.0 * G
    return G, G4


def sector_weights(m, q):
    """Constrained {S=0,1,2} solve: w1=(q-2m... ) — from 2w1+6w2=m,
    4w1+36w2=q: w2=(q-2m)/24, w1=(6m-q)/8, w0=1-w1-w2. Negative weight =>
    support beyond S=2 (reported, not hidden)."""
    w2 = (q - 2.0 * m) / 24.0
    w1 = (6.0 * m - q) / 8.0
    return 1.0 - w1 - w2, w1, w2


# ---- G0: H4/STO-3G moment gauge gate (exact answers) ----
mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
            basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
cis = fci.FCI(mol, mf.mo_coeff)
cis.nroots = 6
es_h4, vs_h4 = cis.kernel()
sa_h4 = np.asarray(cistring.make_strings(range(4), 2), np.uint64)
aw_h4 = np.repeat(sa_h4, len(sa_h4))
bw_h4 = np.tile(sa_h4, len(sa_h4))
vm_h4 = np.column_stack([np.asarray(v).ravel() for v in vs_h4])
Gh, G4h = spin_moments(aw_h4, bw_h4, vm_h4, 4)
var_h4 = np.diagonal(G4h) - np.diagonal(Gh) ** 2
ok = bool((np.abs(var_h4) < 1e-9).all())
i0 = int(np.argmin(np.diagonal(Gh)))
i1 = int(np.argmin(np.abs(np.diagonal(Gh) - 2.0)))
mix = (np.sqrt(0.7) * vm_h4[:, i0] + np.sqrt(0.3) * vm_h4[:, i1])
mix /= np.linalg.norm(mix)
Gm, G4m = spin_moments(aw_h4, bw_h4, mix, 4)
varm = float(G4m[0, 0] - Gm[0, 0] ** 2)
ok &= abs(float(Gm[0, 0]) - 0.6) < 1e-9 and abs(varm - 0.84) < 1e-9
log(f"G0 H4 moment gauge: eig |Var|max {np.abs(var_h4).max():.1e}, "
    f"mix mean {float(Gm[0,0]):.6f} Var {varm:.6f} "
    f"[{'PASS' if ok else 'FAIL'}]")
assert ok, "G0 FAIL: moment kernel does not reproduce exact spin answers"

# ---- construction: identical to _flagsolve.py / _gram7500.py ----
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

# ---- warm block + banked references ----
bank = np.load(WARM_SUMMARY)
e_cert = np.asarray(bank["e_roots"], float)
G_cert = np.asarray(bank["gram"], float)
rz = np.load(WARM_RESTART)
X = np.asarray(rz["vecs"], float)
assert X.shape == (3, na * nb), f"restart shape {X.shape}"
for i in range(3):
    X[i] /= np.linalg.norm(X[i])
log(f"warm block loaded: 3 certified roots, banked E {e_cert}")

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


a_words = np.repeat(sa.astype(np.uint64), nb)
b_words = np.tile(sb.astype(np.uint64), na)

t0 = time.time()
hv0 = hop(X[0])
e_ray0 = float(X[0] @ hv0) + ECORE
r0 = float(np.linalg.norm(hv0 - (e_ray0 - ECORE) * X[0]))
del hv0
de = abs(e_ray0 - e_cert[0])
tag = "PASS" if de < 2e-6 else "FAIL"
log(f"G2 Rayleigh(warm root0) {e_ray0:.9f} vs certified {e_cert[0]:.9f}: "
    f"dE {de*1e6:.2f} uHa, |r| {r0:.3e} [{tag}, {time.time()-t0:,.0f}s]")
assert tag == "PASS", "G2 FAIL: hop/integrals/strings wiring mismatch"


def diag_and_ck(X, tag_name, tol, mc, wall, do_moments=True):
    """Per-stage diagnostics + moments + checkpoint."""
    k = X.shape[0]
    e_ray = np.zeros(k)
    resid = np.zeros(k)
    t1 = time.time()
    for i in range(k):
        hv = hop(X[i])
        e_ray[i] = float(X[i] @ hv) + ECORE
        resid[i] = float(np.linalg.norm(hv - (e_ray[i] - ECORE) * X[i]))
        del hv
    log(f"residual hops: {time.time()-t1:,.0f}s")
    for i in range(k):
        log(f"  root {i}: Rayleigh {e_ray[i]:.9f}  |r| {resid[i]:.3e}  "
            f"err {(e_ray[i]-E_EXACT)*1e3:+9.3f} mHa"
            + ("" if i == 0 else
               f"  gap-to-prev {(e_ray[i]-e_ray[i-1])*1e3:+.3f} mHa"))
    G = G4 = None
    if do_moments:
        t1 = time.time()
        G, G4 = spin_moments(a_words, b_words, X.T, NC)
        ev = np.linalg.eigvalsh(G)
        log(f"moments done in {time.time()-t1:,.0f}s")
        log(f"  per-root <S2> diag:  {[f'{G[i,i]:.4f}' for i in range(k)]}")
        log(f"  Gram eigenvalues:    {[f'{x:.4f}' for x in ev]}")
        var = np.diagonal(G4) - np.diagonal(G) ** 2
        log(f"  per-root Var(S2):    {[f'{v:.4f}' for v in var]}")
        for i in range(k):
            w0, w1, w2 = sector_weights(G[i, i], G4[i, i])
            log(f"  root {i} sector solve (S=0,1,2): "
                f"w=({w0:+.4f}, {w1:+.4f}, {w2:+.4f})"
                + ("" if min(w0, w1, w2) > -1e-3 else "  [beyond-S=2 support]"))
    np.savez_compressed(
        f"{OUT}{tag_name}.npz", n=N, nroots=k, tol=tol, max_cycle=mc,
        e_rayleigh=e_ray, resid=resid,
        gram=(G if G is not None else np.zeros(0)),
        gram4=(G4 if G4 is not None else np.zeros(0)),
        wall_s=wall, nt=NT, e_exact=E_EXACT,
        e_cert=e_cert, gram_cert=G_cert)
    log(f"saved {OUT}{tag_name}.npz")
    return G


# ---- s4ck0: moments on the certified vectors AS PUBLISHED ----
done_ck0 = os.path.exists(f"{OUT}ck0.npz")
if done_ck0:
    log("s4ck0 already on disk -- skip")
    G_warm = np.asarray(np.load(f"{OUT}ck0.npz")["gram"], float)
else:
    log("=== s4ck0: S^2/S^4 moments of the certified 3-root block ===")
    G_warm = diag_and_ck(X, "ck0", 0.0, 0, 0.0, do_moments=True)
dG = float(np.abs(G_warm - G_cert).max())
tag = "PASS" if dG < 1e-6 else "FAIL"
log(f"G3 warm S2-Gram vs certified gram: max|dG| {dG:.2e} [{tag}]")
assert tag == "PASS", "G3 FAIL: Gram of warm block disagrees with banked"

# ---- staged solve ----
RESTART = f"{OUT}_restart.npz"
stages_done = 0
if os.path.exists(RESTART):
    rz2 = np.load(RESTART)
    X = np.asarray(rz2["vecs"], float)
    stages_done = int(rz2["stages_done"])
    log(f"RESUME: {X.shape[0]} vecs, {stages_done} stages already done")

partial = False
hd = np.asarray(hdiag, float).ravel()
hd_order = np.argsort(hd)
for si, (nr, tol, mc) in enumerate(STAGEPLAN):
    if si < stages_done:
        log(f"stage {si+1} (nroots={nr} tol={tol:g}) already done -- skip")
        continue
    if WALL_BUDGET_S and (time.time() - T0) > WALL_BUDGET_S:
        log(f"WALL BUDGET exceeded before stage {si+1} -- stopping here")
        partial = True
        break
    while X.shape[0] < nr:  # grow the block with orthogonalized hdiag guesses
        for idx in hd_order:
            g = np.zeros(na * nb)
            g[idx] = 1.0
            for u in X:
                g -= u * float(u @ g)
            nrm = np.linalg.norm(g)
            if nrm > 0.3:
                X = np.vstack([X, g / nrm])
                break
        log(f"block grown to {X.shape[0]} (lowest-hdiag guess, orthogonalized)")
    log(f"=== stage {si+1}/{len(STAGEPLAN)}: kernel_fixed_space nroots={nr} "
        f"tol={tol:g} max_cycle={mc} NT={NT} ===")
    ci0 = selected_ci._as_SCIvector(X.copy(), ci_strs)
    t0 = time.time()
    e_roots, vecs = fci.selected_ci.kernel_fixed_space(
        myci, H1, ERI, NC, NELEC, ci_strs, ci0=ci0,
        nroots=nr, tol=tol, max_cycle=mc, verbose=5)
    wall = time.time() - t0
    e_roots = np.atleast_1d(np.asarray(e_roots, float))
    if not isinstance(vecs, (list, tuple)):
        vecs = [vecs]
    X = np.vstack([np.asarray(v).ravel() for v in vecs])
    log(f"stage {si+1} solved in {wall:,.0f}s  Ritz {e_roots}")
    d0 = e_roots[0] - e_cert[0]
    log(f"GATE root0 vs certified: drift {d0*1e6:+.2f} uHa "
        f"[{'PASS' if d0 < 1e-6 else 'FAIL'}]")
    diag_and_ck(X, f"ck{si+1}", tol, mc, wall, do_moments=True)
    np.savez(RESTART, vecs=X, stages_done=si + 1, e_roots=e_roots, n=N)
    log("restart saved")

if not partial:
    last = max(i + 1 for i in range(len(STAGEPLAN))
               if os.path.exists(f"{OUT}ck{i+1}.npz"))
    dat = dict(np.load(f"{OUT}ck{last}.npz"))
    np.savez_compressed(f"{OUT}.npz", **dat)
    log(f"saved {OUT}.npz (= ck{last})")

print("GRAM7500S4_PARTIAL" if partial else "GRAM7500S4_DONE", flush=True)
