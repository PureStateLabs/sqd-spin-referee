"""Penalty-ON audit, the flagship's ACTUAL squared form: H + lam*(S^2 - s(s+1))^2.

CORRECTION (2026-07-19, source-verified from raw arXiv bytes v1/v2/v3 + ar5iv +
published Sci. Adv.): Robledo-Moreno et al. (arXiv:2405.05068, all versions incl.
v1) DO use an active spin penalty, stated verbatim as

    (H + lam*[S^2 - s(s+1)]^2)|psi> = E|psi>,   "we employed ... lam = 0.2"

i.e. the SQUARED penalty, present since the first preprint. Our companion run
_sqdpen.py used pyscf's LINEAR shift*S^2 (which is what fix_spin_ implements);
pyscf does not even implement the squared form, so reproducing the flagship's
own operator requires applying the projected S^2 (contract_ss) TWICE per matvec.

This script does exactly that on the SAME captured subspaces _sqdpen.py used
(sqdpen_spaces.npz: it1 = 245^2, it4 = 441^2 = the +45.5 mHa benchmark-energy
soup row), so the squared frontier is directly comparable to the linear one in
sqdpen_frontier_{it1,it4}.npz.

For the singlet target ss=0, s(s+1)=0, so the penalty operator is (S^2)^2 and
<(S^2)^2> = 0 iff the state is an exact singlet -- the SAME zero-set as <S^2>=0.
Hence the lam->inf endpoint (the lowest exact singlet the subspace can express,
and its energy price above the contaminated ground) is IDENTICAL to the linear
run's, independent of penalty form and strength. Any lam >= 0 penalty is PSD, so
the penalized ground energy can only rise above the subspace's own lowest exact
singlet: a penalty cannot manufacture singlet support the subspace lacks.

Operators, spaces, ECORE, E_EXACT, anchors, and SpaceOps are imported verbatim
from _sqdpen.py (STAGES="" so importing runs no stage). Idempotent: checkpoints
per tag to sqdpen_sq_{tag}.npz (rewritten after every lam point for early
harvest), final roll-up to sqdpen_sq.npz. Sentinel: SQDPEN_SQ_DONE.

Env: NT(6) TAGS("it1,it4") LAMS("0.1,0.2,0.5,1,2") FLOORMAX(64) TOL(1e-9)
"""
import os

os.environ["STAGES"] = ""            # _sqdpen runs no stage on import
os.environ.setdefault("NT", "6")

import time                          # noqa: E402
import numpy as np                   # noqa: E402
import _sqdpen as base               # loads integrals, defines SpaceOps  # noqa: E402
from pyscf import fci                # noqa: E402
from pyscf.fci.selected_ci import kernel_fixed_space  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence  # noqa: E402

T0 = time.time()
TAGS = os.environ.get("TAGS", "it1,it4").split(",")
LAMS = [float(x) for x in os.environ.get("LAMS", "0.1,0.2,0.5,1,2").split(",")]
FLOORMAX = float(os.environ.get("FLOORMAX", "64"))
TOL = float(os.environ.get("TOL", "1e-6"))
TOL2 = float(os.environ.get("TOL2", "0"))   # >0: warm-started polish re-solve
NCV = int(os.environ.get("NCV", "48"))
ANCH = {"it1": (base.ANCHOR_E_IT1, base.ANCHOR_S2_IT1),
        "it4": (base.ANCHOR_E, base.ANCHOR_S2)}


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def solve_pen_sq(ops, lam, k=0.0, v0=None, tol=TOL, maxiter=2000):
    """Ground of H + lam*(S^2 - k)^2 via eigsh; k = s(s+1) (0 for singlet).

    (S^2 - k)^2 c = ssop(ssop(c)) - 2k ssop(c) + k^2 c.  ssop = contract_ss,
    the exact projected S^2 on this fixed determinant space (PS^2P)."""
    def mv(c):
        c = np.asarray(c).ravel()
        out = ops.hop(c)
        if lam != 0.0:
            s = ops.ssop(c)
            p = ops.ssop(s)
            if k != 0.0:
                p = p - 2.0 * k * s + (k * k) * c
            out = out + lam * p
        return out
    A = LinearOperator((ops.n, ops.n), matvec=mv, dtype=float)
    if v0 is None:
        v0 = np.zeros(ops.n)
        v0[int(np.argmin(ops.hdiag))] = 1.0
    ncv = min(ops.n - 1, NCV)
    conv = True
    try:
        th, vec = eigsh(A, k=1, which="SA", v0=np.asarray(v0).ravel(),
                        tol=tol, maxiter=maxiter, ncv=ncv)
        th, vec = th[0], vec[:, 0]
    except ArpackNoConvergence as e:          # record best partial, flag it
        conv = False
        th = float(e.eigenvalues[0]) if len(e.eigenvalues) else np.nan
        vec = e.eigenvectors[:, 0] if e.eigenvectors.shape[1] else v0
    vec = np.asarray(vec).ravel()
    vec = vec / np.linalg.norm(vec)
    resid = float(np.linalg.norm(mv(vec) - th * vec))
    e, s2, var = ops.point(vec)
    return vec, {"lam": lam, "theta": float(th) + base.ECORE, "e_h": e,
                 "err_mHa": (e - base.E_EXACT) * 1e3, "s2": s2,
                 "var_s2": var, "resid": resid, "eigsh_conv": conv}


spaces = base.load_spaces()
allres = {}
for tag in TAGS:
    part = f"sqdpen_sq_{tag}.npz"
    if os.path.exists(part):
        allres[tag] = np.load(part, allow_pickle=True)["blk"].item()
        log(f"{tag}: checkpoint hit ({len(allres[tag]['pts'])} pts)")
        continue
    r = spaces[tag]
    ops = base.SpaceOps(r["strs_a"], r["strs_b"])
    log(f"{tag}: space {ops.na}x{ops.nb} = {ops.n:,}")
    # warm start from the certified unpenalized kernel ground (fast, converges
    # unpenalized), then eigsh re-converges to its own tolerance.
    myci0 = fci.selected_ci.SelectedCI()
    _, sv0 = kernel_fixed_space(myci0, base.H1, base.ERI, base.NC, base.NELEC,
                                ops.strs)
    # cert directly from the Davidson (kernel_fixed_space) unpenalized ground --
    # it IS pyscf's own solve, so no separate lam=0 eigsh is needed; it also
    # serves as the warm start for the first penalized point.
    sv0 = np.asarray(sv0).ravel()
    sv0 = sv0 / np.linalg.norm(sv0)
    e0, s20, var0 = ops.point(sv0)
    info0 = {"lam": 0.0, "theta": float("nan"), "e_h": e0,
             "err_mHa": (e0 - base.E_EXACT) * 1e3, "s2": s20, "var_s2": var0,
             "resid": 0.0, "eigsh_conv": True}
    log(f"{tag} lam=0 (Davidson, unpenalized): E {e0:.6f} "
        f"(err {info0['err_mHa']:+.3f}) S2 {s20:.4f} Var {var0:.3f}")
    assert abs(e0 - ANCH[tag][0]) < 5e-6, \
        f"{tag} lam=0 E vs shipped: {e0} vs {ANCH[tag][0]}"
    assert abs(s20 - ANCH[tag][1]) < 5e-3
    log(f"{tag}: lam=0 matches the shipped record (operator kit certified)")

    pts = [info0]
    vec = sv0

    def _flush():
        blk = {"pts": pts, "na": ops.na, "nb": ops.nb,
               "nmv": tuple(ops.nmv), "form": "squared (S^2)^2, ss=0",
               "vec_last": np.asarray(vec, dtype=np.float64),
               "vec_lam": pts[-1]["lam"] if len(pts) > 1 else 0.0}
        np.savez(part, blk=np.array(blk, dtype=object))
        return blk

    for lam in LAMS:
        vec, info = solve_pen_sq(ops, lam, v0=vec)
        log(f"{tag} lam={lam:g} SQ: E_H {info['e_h']:.6f} "
            f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
            f"Var {info['var_s2']:.4f} resid {info['resid']:.1e} "
            f"conv={info['eigsh_conv']}")
        if TOL2 > 0:   # warm-started polish to audit-grade residual
            vec, info = solve_pen_sq(ops, lam, v0=vec, tol=TOL2)
            log(f"{tag} lam={lam:g} SQ POLISH: E_H {info['e_h']:.6f} "
                f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
                f"Var {info['var_s2']:.4f} resid {info['resid']:.1e} "
                f"conv={info['eigsh_conv']}")
        pts.append(info)
        _flush()
    # extend toward the singlet floor if not there yet
    lam = LAMS[-1]
    while pts[-1]["s2"] > 0.02 and lam < FLOORMAX:
        lam *= 2
        vec, info = solve_pen_sq(ops, lam, v0=vec)
        log(f"{tag} lam={lam:g} SQ: E_H {info['e_h']:.6f} "
            f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
            f"Var {info['var_s2']:.4f} resid {info['resid']:.1e} "
            f"conv={info['eigsh_conv']}")
        pts.append(info)
        _flush()
    allres[tag] = _flush()
    log(f"{tag}: singlet-jaw endpoint err {pts[-1]['err_mHa']:+.3f} mHa "
        f"at S2 {pts[-1]['s2']:.4f}  (linear floor for ref: "
        f"{'+2894.7' if tag == 'it4' else '+3034.5'} mHa)")

np.savez("sqdpen_sq.npz", res=np.array(allres, dtype=object))

log("==== SQUARED vs LINEAR (lam=0.2 headline + floor) ====")
for tag in TAGS:
    blk = allres[tag]
    p02 = next((p for p in blk["pts"] if abs(p["lam"] - 0.2) < 1e-9), None)
    flo = blk["pts"][-1]
    if p02:
        log(f"{tag} SQUARED lam=0.2: err {p02['err_mHa']:+.3f} mHa "
            f"S2 {p02['s2']:.4f} Var {p02['var_s2']:.4f}")
    log(f"{tag} SQUARED floor:     err {flo['err_mHa']:+.3f} mHa "
        f"S2 {flo['s2']:.4f}")

print("SQDPEN_SQ_DONE", flush=True)
