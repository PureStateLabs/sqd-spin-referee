"""True-form squared spin penalty: the LITERAL compression P(S^2)^2 P of the
flagship's stated operator H + lam*(S^2 - s(s+1))^2, NOT (P S^2 P)^2.

R23 fix (2026-07-19). Review round 12 (the strongest of the twelve) caught that
_sqdpen_sq.py forms the squared penalty as ssop(ssop(c)) -- it applies the
SELECTED-space projected S^2 (contract_ss = P S^2 P) twice, computing
(P S^2 P)^2. The flagship's stated operator (S^2)^2, compressed onto the fixed
determinant space P, is P (S^2)^2 P = P S^4 P. The two differ by

    P (S^2)^2 P  -  (P S^2 P)^2  =  P S^2 (I - P) S^2 P   >=  0,

the leakage of S^2 through the space complement (I - P). That leakage is
precisely the object this paper's central result is about: P is not
S^2-invariant, [PHP, PS^2P] != 0. So it cannot be assumed negligible -- we
build the true operator and measure it.

TRUE OPERATOR via the M_s = 0 ladder (same S+ machinery as the shipped moment
kernel _s4states.py / _s4gate.py, already gauged there against FCI). For c on
the fixed space P0, using S^2 = S- S+ on M_s = 0:
    W  = S+ c        (complete M_s=+1 image; sparse R1: P0 -> M1)
    W2 = S+ W        (complete M_s=+2 image; sparse R2: M1 -> M2)
    P0 (S^2)^2 P0 c  =  R1^T ( R2^T W2  +  2 W ).

Derivation: (S^2)^2 = S- S+ S- S+; on M_s=1, S+S- = S-S+ + 2, so
S-S+S-S+ c = S-(S+S-)(S+c) = S-(S-S+ + 2)W = S-^2 W2 + 2 S- W. Project with P0:
P0 S- W = R1^T W exactly (R1^T lowers-and-reprojects onto P0). And any M_s=1
vector supported OUTSIDE M1 is annihilated by P0 S- (d not in M1  <=>  S-|d> has
no P0 component), so the intermediate reprojection through M1 in R1^T R2^T loses
nothing: P0 S-^2 W2 = R1^T R2^T W2. Hence the identity above is EXACT, retaining
the full (I - P) leakage. Gauged three ways in G0 below.

Singlet target ss = 0 => k = s(s+1) = 0, penalty = (S^2)^2. (General k handled:
(S^2)^2 - 2k S^2 + k^2, but the flagship targets the singlet, k = 0.)

We run BOTH forms on the SAME it1 (245^2) and it4 (441^2) spaces _sqdpen_sq.py
used, at the flagship's stated lam = 0.2, plus a floor sweep confirming both
forms share the identical +2894.7 / +3034.5 mHa singlet floor (same null space:
PS^2P c = 0 => S^2 c = 0 exactly, since the first moment is exact and S^2 is
PSD). The projected-square arm reproduces _sqdpen_sq.py's frozen number as a
continuity check; the true arm is the number the flagship's stated operator
actually yields; their difference is the leakage operator, exhibited a third way.

Operators/spaces/ECORE/E_EXACT/anchors/SpaceOps imported verbatim from _sqdpen
(STAGES="" so import runs no stage). Idempotent, checkpoint per tag.

Env: NT(6) TAGS("it1,it4") LAM(0.2) FLOORMAX(64) TOL(1e-8) TOL2(0) NCV(64)
Out: sqdpen_true_{tag}.npz, sqdpen_true.npz  Sentinel: SQDPEN_TRUE_DONE
"""
import os

os.environ["STAGES"] = ""             # _sqdpen runs no stage on import
os.environ.setdefault("NT", "6")

import time                           # noqa: E402
import numpy as np                    # noqa: E402
import scipy.sparse as sp             # noqa: E402
import _sqdpen as base                # loads integrals, defines SpaceOps  # noqa: E402
from pyscf import fci                 # noqa: E402
from pyscf.fci.selected_ci import kernel_fixed_space  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence  # noqa: E402

T0 = time.time()
TAGS = os.environ.get("TAGS", "it1,it4").split(",")
LAM = float(os.environ.get("LAM", "0.2"))          # flagship's stated strength
# continuation sweep up to the stated lambda: shows the contaminated->singlet
# transition of the TRUE form and warm-starts each point from the previous
# (the singlet answer is far from the contaminated cold start, so ramping
# lambda is what keeps eigsh fast on the large S^4 spectral range).
LAMS = [float(x) for x in os.environ.get("LAMS", "0.05,0.1,0.2").split(",")]
TOL = float(os.environ.get("TOL", "1e-6"))
NCV = int(os.environ.get("NCV", "120"))
MAXIT = int(os.environ.get("MAXITER", "3000"))
ANCH = {"it1": (base.ANCHOR_E_IT1, base.ANCHOR_S2_IT1),
        "it4": (base.ANCHOR_E, base.ANCHOR_S2)}
# linear singlet floor per space (from the certified _sqdpen frontier), for ref
LIN_FLOOR = {"it1": 3034.5, "it4": 2894.7}

MB1 = np.uint64(0x5555555555555555)
MB2 = np.uint64(0x3333333333333333)
MB4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def popcount64(a):
    a = a - ((a >> np.uint64(1)) & MB1)
    a = (a & MB2) + ((a >> np.uint64(2)) & MB2)
    a = (a + (a >> np.uint64(4))) & MB4
    return (a * H01) >> np.uint64(56)


def splus_sparse(src_a, src_b, ncas):
    """Sparse S+ = sum_p a^dag_{p,alpha} a_{p,beta} as a matrix R with
    rows = deduped M_s+1 image dets, cols = source dets, R[t,s] = <t|S+|s>.
    Phase logic verbatim from splus_image (_s4states.py / _s4gate.py, gauged
    against FCI there and again in G0 here)."""
    src_a = np.asarray(src_a, np.uint64)
    src_b = np.asarray(src_b, np.uint64)
    n = len(src_a)
    idx = np.arange(n, dtype=np.int64)
    TA, TB, VAL, COL = [], [], [], []
    for p in range(ncas):
        bit = np.uint64(1 << p)
        low = np.uint64((1 << p) - 1)
        sel = ((src_b & bit) != 0) & ((src_a & bit) == 0)
        if not sel.any():
            continue
        a0, b0 = src_a[sel], src_b[sel]
        p1 = ((popcount64(a0) + popcount64(b0 & low)) & 1).astype(np.int64)
        p2 = (popcount64(a0 & low) & 1).astype(np.int64)
        sgn = ((1 - 2 * p1) * (1 - 2 * p2)).astype(float)
        TA.append(a0 | bit)
        TB.append(b0 & ~bit)
        VAL.append(sgn)
        COL.append(idx[sel])
    if not TA:
        return (sp.csr_matrix((0, n)), np.zeros(0, np.uint64),
                np.zeros(0, np.uint64))
    TA = np.concatenate(TA)
    TB = np.concatenate(TB)
    VAL = np.concatenate(VAL)
    COL = np.concatenate(COL)
    uniq, row = np.unique(np.stack([TA, TB], 1), axis=0, return_inverse=True)
    row = np.asarray(row).reshape(-1)
    R = sp.coo_matrix((VAL, (row, COL)),
                      shape=(uniq.shape[0], n)).tocsr()
    return R, uniq[:, 0].copy(), uniq[:, 1].copy()


class TrueS4:
    """Literal compression P (S^2)^2 P = P S^4 P on a fixed product space."""

    def __init__(self, strs_a, strs_b, ncas):
        sa = np.asarray(strs_a, np.uint64)
        sb = np.asarray(strs_b, np.uint64)
        self.na, self.nb = len(sa), len(sb)
        self.n = self.na * self.nb
        aa = np.repeat(sa, self.nb)
        bb = np.tile(sb, self.na)
        self.R1, m1a, m1b = splus_sparse(aa, bb, ncas)
        self.R2, _, _ = splus_sparse(m1a, m1b, ncas)
        self.R1t = self.R1.T.tocsr()
        self.R2t = self.R2.T.tocsr()
        self.m1 = int(self.R1.shape[0])
        self.m2 = int(self.R2.shape[0])
        self.nnz = int(self.R1.nnz + self.R2.nnz)

    def s4(self, c):
        """P (S^2)^2 P c (the true squared-penalty operator at ss = 0)."""
        c = np.asarray(c).ravel()
        W = self.R1 @ c
        W2 = self.R2 @ W
        return self.R1t @ (self.R2t @ W2 + 2.0 * W)


def _spin_moments_diag(a, b, c, ncas):
    """<S^2>, <S^4> of a single M_s=0 vector via the shipped moment identity
    (W = S+ c, G = W.W; W2 = S+ W, G4 = W2.W2 + 2 G). Independent of TrueS4."""
    R1, m1a, m1b = splus_sparse(a, b, ncas)
    W = R1 @ np.asarray(c).ravel()
    R2, _, _ = splus_sparse(m1a, m1b, ncas)
    W2 = R2 @ W
    return float(W @ W), float(W2 @ W2 + 2.0 * (W @ W))


def gauge():
    """G0: validate P(S^2)^2P against pyscf's independent S^2, three ways."""
    from pyscf import gto, scf
    from pyscf.fci import cistring, spin_op
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = int(mf.mo_coeff.shape[1])
    nelec = (2, 2)
    sa = np.asarray(cistring.make_strings(range(norb), nelec[0]), np.uint64)
    nb = len(sa)
    ndim = nb * nb
    fa = np.repeat(sa, nb)
    fb = np.tile(sa, nb)

    # (a) ladder S^2 (= S- S+ on M_s=0) vs pyscf spin_op.contract_ss, full space
    R1f, _, _ = splus_sparse(fa, fb, norb)
    S2_ladder = np.asarray((R1f.T @ R1f).todense())
    S2_ps = np.zeros((ndim, ndim))
    for i in range(ndim):
        e = np.zeros((nb, nb))
        e.ravel()[i] = 1.0
        S2_ps[:, i] = spin_op.contract_ss(e, norb, nelec).ravel()
    err_a = float(np.abs(S2_ladder - S2_ps).max())

    # (b) s4 matvec vs dense projected S^4 on a TRUNCATED product subspace
    ia = [0, 1, 3, 5]
    ib = [0, 2, 4, 5]
    sub_a, sub_b = sa[ia], sa[ib]
    ix = [ai * nb + bi for ai in ia for bi in ib]
    S4_ps = S2_ps @ S2_ps
    ref_true = S4_ps[np.ix_(ix, ix)]
    ref_projsq = S2_ps[np.ix_(ix, ix)] @ S2_ps[np.ix_(ix, ix)]
    t4 = TrueS4(sub_a, sub_b, norb)
    eye = np.eye(t4.n)
    M_true = np.column_stack([t4.s4(eye[:, j]) for j in range(t4.n)])
    err_b = float(np.abs(M_true - ref_true).max())
    leak = float(np.abs(ref_true - ref_projsq).max())
    sym = float(np.abs(M_true - M_true.T).max())

    # (c) diagonal <c|s4|c> vs the independent moment kernel <S^4>
    rng = np.random.default_rng(0)
    c = rng.standard_normal(t4.n)
    c /= np.linalg.norm(c)
    q_s4 = float(c @ t4.s4(c))
    _, q_mom = _spin_moments_diag(np.repeat(sub_a, len(sub_b)),
                                  np.tile(sub_b, len(sub_a)), c, norb)
    err_c = abs(q_s4 - q_mom)

    ok = (err_a < 1e-9 and err_b < 1e-9 and leak > 1e-6
          and sym < 1e-9 and err_c < 1e-9)
    log(f"G0 gauge: (a) ladder-vs-pyscf S^2 max|dif| {err_a:.1e}; "
        f"(b) s4-vs-dense P(S^2)^2P {err_b:.1e}, leakage max "
        f"{leak:.3e}, sym {sym:.1e}; (c) diag-vs-moment {err_c:.1e}  "
        f"[{'PASS' if ok else 'FAIL'}]")
    assert ok, "G0 GAUGE FAIL: true-form operator not validated"
    return leak


def make_mv(ops, t4, form, lam):
    if form == "true":
        pen = t4.s4
    elif form == "projsq":
        pen = lambda c: ops.ssop(ops.ssop(c))     # noqa: E731  (P S^2 P)^2
    else:
        raise ValueError(form)

    def mv(c):
        c = np.asarray(c).ravel()
        return ops.hop(c) + lam * pen(c)
    return mv


def solve(ops, t4, form, lam, v0, tol=TOL):
    mv = make_mv(ops, t4, form, lam)
    A = LinearOperator((ops.n, ops.n), matvec=mv, dtype=float)
    ncv = min(ops.n - 1, NCV)
    conv = True
    try:
        th, vec = eigsh(A, k=1, which="SA", v0=np.asarray(v0).ravel(),
                        tol=tol, maxiter=MAXIT, ncv=ncv)
        th, vec = float(th[0]), vec[:, 0]
    except ArpackNoConvergence as e:
        conv = False
        th = float(e.eigenvalues[0]) if len(e.eigenvalues) else float("nan")
        vec = e.eigenvectors[:, 0] if e.eigenvectors.shape[1] else v0
    vec = np.asarray(vec).ravel()
    vec = vec / np.linalg.norm(vec)
    resid = float(np.linalg.norm(mv(vec) - th * vec))
    e, s2, var_proj = ops.point(vec)
    q_true = float(vec @ t4.s4(vec))     # exact <S^4> of the eigenvector
    var_true = q_true - s2 * s2
    info = {"form": form, "lam": lam, "e_h": e,
            "err_mHa": (e - base.E_EXACT) * 1e3, "s2": s2,
            "var_true": var_true, "var_proj": var_proj, "s4": q_true,
            "resid": resid, "conv": conv, "theta": th + base.ECORE}
    return vec, info


def load_frozen_projsq(tag):
    """The frozen, audit-certified projected-square (P S^2 P)^2 point at
    lam = 0.2 from _sqdpen_sq.py (sqdpen_sq_{tag}.npz) -- the number the paper
    already reports; we compare the TRUE form against it rather than re-running
    that ~25 min/point solve."""
    f = f"sqdpen_sq_{tag}.npz"
    if not os.path.exists(f):
        return None
    blk = np.load(f, allow_pickle=True)["blk"].item()
    p = next((q for q in blk["pts"] if abs(q["lam"] - 0.2) < 1e-9), None)
    return p


def run_tag(tag):
    part = f"sqdpen_true_{tag}.npz"
    if os.path.exists(part):
        blk = np.load(part, allow_pickle=True)["blk"].item()
        log(f"{tag}: checkpoint hit")
        return blk
    r = base.load_spaces()[tag]
    ops = base.SpaceOps(r["strs_a"], r["strs_b"])
    tb = time.time()
    t4 = TrueS4(r["strs_a"], r["strs_b"], base.NC)
    log(f"{tag}: P0 {ops.na}x{ops.nb}={ops.n:,}  M1={t4.m1:,}  M2={t4.m2:,}  "
        f"nnz(R1,R2)={t4.R1.nnz:,},{t4.R2.nnz:,}  build {time.time()-tb:.1f}s")

    # unpenalized Davidson ground -> warm start + shipped-anchor cert
    myci0 = fci.selected_ci.SelectedCI()
    _, sv0 = kernel_fixed_space(myci0, base.H1, base.ERI, base.NC,
                                base.NELEC, ops.strs)
    sv0 = np.asarray(sv0).ravel()
    sv0 = sv0 / np.linalg.norm(sv0)
    e0, s20, _ = ops.point(sv0)
    assert abs(e0 - ANCH[tag][0]) < 5e-6, \
        f"{tag} lam=0 E vs shipped {e0} vs {ANCH[tag][0]}"
    assert abs(s20 - ANCH[tag][1]) < 5e-3
    q0 = float(sv0 @ t4.s4(sv0))     # unpenalized <S^4> (validated in G0)
    log(f"{tag}: lam=0 matches shipped record (E {e0:.6f}, S2 {s20:.4f}); "
        f"unpenalized <S^4> {q0:.4f}")

    # TRUE-form frontier: ramp lambda with continuation warm starts
    sweep = []
    vec = sv0
    for lam in LAMS:
        vec, info = solve(ops, t4, "true", lam, v0=vec)
        log(f"{tag} lam={lam:g} TRUE  P(S^2)^2P: err {info['err_mHa']:+.3f} "
            f"mHa  S2 {info['s2']:.4f}  VarTrue {info['var_true']:.4f}  "
            f"resid {info['resid']:.1e}  conv={info['conv']}")
        sweep.append(info)
    i_true = next(p for p in sweep if abs(p["lam"] - LAM) < 1e-9)

    # frozen, audit-certified projected-square point at the same lambda
    i_proj = load_frozen_projsq(tag)
    if i_proj is not None:
        log(f"{tag} lam={LAM:g} PROJSQ (P S^2 P)^2 [frozen sqdpen_sq]: "
            f"err {i_proj['err_mHa']:+.3f} mHa  S2 {i_proj['s2']:.4f}  "
            f"Var(proj) {i_proj['var_s2']:.4f}")
        log(f"{tag} lam={LAM:g} LEAKAGE (true - projsq): "
            f"dErr {i_true['err_mHa']-i_proj['err_mHa']:+.1f} mHa, "
            f"dS2 {i_true['s2']-i_proj['s2']:+.4f}  "
            f"(true reaches the singlet floor {LIN_FLOOR[tag]:+.1f}; "
            f"projsq stays contaminated)")

    blk = {"tag": tag, "na": ops.na, "nb": ops.nb, "n": ops.n,
           "m1": t4.m1, "m2": t4.m2, "lam": LAM, "sweep": sweep,
           "true": i_true, "projsq": i_proj,
           "lin_floor_mHa": LIN_FLOOR[tag], "unpen_s4": q0}
    np.savez(part, blk=np.array(blk, dtype=object))
    return blk


leak0 = gauge()
if os.environ.get("GAUGE_ONLY") == "1":
    log("GAUGE_ONLY: G0 passed, stopping before the space runs")
    print("SQDPEN_TRUE_DONE", flush=True)
    raise SystemExit(0)
allres = {}
for tag in TAGS:
    allres[tag] = run_tag(tag)
np.savez("sqdpen_true.npz", res=np.array(allres, dtype=object), gauge_leak=leak0)

log("==== TRUE  P(S^2)^2P   vs   PROJECTED-SQUARE  (P S^2 P)^2   (lam=0.2) ====")
for tag in TAGS:
    b = allres[tag]
    t, p = b["true"], b["projsq"]
    log(f"{tag} ({b['na']}x{b['nb']}; unpenalized <S^4> {b['unpen_s4']:.2f}):")
    log(f"   TRUE   P(S^2)^2P : err {t['err_mHa']:+9.3f} mHa  S2 {t['s2']:.4f}"
        f"  VarTrue {t['var_true']:.4f}  resid {t['resid']:.1e}")
    if p is not None:
        log(f"   PROJSQ (PS^2P)^2 : err {p['err_mHa']:+9.3f} mHa  "
            f"S2 {p['s2']:.4f}  Var(proj) {p['var_s2']:.4f}  [frozen]")
    log(f"   singlet floor ref {b['lin_floor_mHa']:+.1f} mHa  "
        f"(true form reaches it at lam=0.2; projsq does not)")

print("SQDPEN_TRUE_DONE", flush=True)
