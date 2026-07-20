"""Penalty-ON audit on the shipped pipeline's own subspaces ([2Fe-2S], soup jaw).

CONTEXT (corrected 2026-07-19). The flagship IBM SQD paper (arXiv:2405.05068,
all versions from v1) states in its Methods: "we achieve the conservation of
total spin by a soft constraint in the eigenstate solver, i.e. by adding a
penalty term to mitigate spin contamination," (H + lambda*[S^2 - s(s+1)]^2),
"In this work, we employed a soft constraint with lambda = 0.2." (An earlier
revision of this header claimed the flagship contains no penalty term; that
was wrong -- a web-extraction false negative, corrected by grepping the raw
arXiv HTML source. The audit trail documents the correction.) Notes on forms:
the flagship's stated form is SQUARED; pyscf implements only the LINEAR
H + shift*S^2 via fix_spin_ (at M_s=0, ss=0), which is what the SHIPPED
qiskit-addon-sqd path (solve_sci -> fix_spin_(ss=spin_sq)) would apply, at
default shift=0.1, if its spin_sq argument were set -- the shipped default
leaves it None. This script runs the shipped LINEAR form across strengths
(the companion _sqdpen_sq.py runs the flagship's stated SQUARED form at its
stated lambda = 0.2) and answers the only question that matters:

    In the subspaces the shipped protocol actually converges on
    benchmark-grade input, what does a spin penalty buy, at any strength?

Design (all stages idempotent, checkpointed npz per stage):
  S1  Capture: rerun the as-shipped diagonalize_fermionic_hamiltonian off-arm
      (symmetrize_spin=True, SPB=500, NBATCH=3, DMAX=500, MAXIT=4, seed 17 --
      the paper's Sec 2.6 soup row) saving every (iter,batch) subspace's CI
      strings. Certify the best against the published row:
      E = -116.560107 (+45.5 mHa), their S^2 = 4.834.
  S2  Frontier: on the fixed it1-best (245^2) and it4-best (441^2) spaces,
      solve ground of H + lam*S^2 for lam in LAMS by scipy eigsh
      (matvecs: pyscf selected_ci.contract_2e for H, selected_ci.contract_ss
      for the projected S^2 -- the exact operators fix_spin penalizes and
      spin_square measures), continuation warm starts, explicit residuals.
      Report E_H = <v|H|v>, <S^2>, Var(S^2) per point. The lam->inf endpoint
      = the best TRUE singlet expressible in that subspace (S^2 PSD: <S^2>=0
      iff singlet) and its energy price. Independent xcheck: pyci solves the
      same det space at lam=0 (mu-Ha gate, the paper's certification pattern).
  S3  Their code, fixed spaces: pyscf kernel_fixed_space + fix_spin_(ss=0)
      exactly as shipped solve_sci wires it, at shift in {0.1, 0.2}:
      (a) shipped default knobs -- measures the silent under-convergence;
      (b) max_cycle=600, tight tol -- must land on the S2 frontier
      (their solver, properly driven, agrees with ours: double cross-cert).
  S4  Summary table.
  S5  The penalized PROTOCOL: rerun the full shipped recovery loop
      (identical knobs/seed) with the penalty active, ss=0, shift in
      {0.1, 0.2}, solves done by the certified eigsh solver (S3 shows the
      shipped path does not converge at these dimensions at default knobs).
      This is the "penalty-matched rerun" a rebuttal would demand.

Env: NT(6) SMOKE(0) STAGES("12345") LAMS("0.1,0.2,0.5,1,2,5,20")
     MAXCYC(600) SEED2(17) SPB(500) NBATCH(3) DMAX(500) MAXIT(4)
Out: sqdpen_spaces.npz sqdpen_frontier.npz sqdpen_theirs.npz sqdpen_loop.npz
Sentinel: SQDPEN_DONE
"""
import os
import time

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

from pyscf import ao2mo  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402
from pyscf.fci import direct_spin1, selected_ci  # noqa: E402
from pyscf.fci.selected_ci import (  # noqa: E402
    _all_linkstr_index, _as_SCIvector, contract_ss, kernel_fixed_space,
)
from pyscf import fci  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

T0 = time.time()
FCID = ("_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-"
        "Clusters/Fe2S2_and_Fe4S4/Fe2S2/fe2s2")
E_EXACT = -116.6056091
NC = 20
NELEC = (15, 15)
SPB = int(os.environ.get("SPB", "500"))
NBATCH = int(os.environ.get("NBATCH", "3"))
DMAX = int(os.environ.get("DMAX", "500"))
MAXIT = int(os.environ.get("MAXIT", "4"))
SEED2 = int(os.environ.get("SEED2", "17"))
MAXCYC = int(os.environ.get("MAXCYC", "600"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
STAGES = os.environ.get("STAGES", "12345")
LAMS = [float(x) for x in os.environ.get(
    "LAMS", "0.1,0.2,0.5,1,2,5,20").split(",")]
if SMOKE:
    MAXIT, LAMS, STAGES = 1, [0.2], "1234"

# Published anchors (paper Sec 2.6 as-shipped soup row, seed 17)
ANCHOR_E, ANCHOR_S2 = -116.560107, 4.834
ANCHOR_E_IT1, ANCHOR_S2_IT1 = -116.515503, 4.903


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


fd = fcidump.read(FCID)
ECORE = float(fd["ECORE"])  # 0.0 for this dump
H1 = fd["H1"]
ERI = ao2mo.restore(1, fd["H2"], NC)
log(f"integrals loaded (ecore {ECORE:.6f})")

# ---- S1: capture the shipped off-arm subspaces -------------------------------
if "1" in STAGES and not os.path.exists("sqdpen_spaces.npz"):
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    d = np.load("sqdship_counts.npz")
    counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
              for aa, bb, nn in zip(d["a"], d["b"], d["n"])}
    bit_array = BitArray.from_counts(counts, num_bits=2 * NC)
    log(f"S1: BitArray {len(counts)} unique keys, {sum(counts.values())} "
        f"shots; running shipped off-arm MAXIT={MAXIT}")
    cap = []
    it_box = [0]

    def cb(results):
        it_box[0] += 1
        for j, r in enumerate(results):
            s2 = float(r.sci_state.spin_square())
            e_tot = r.energy + ECORE
            cap.append({"iter": it_box[0], "batch": j,
                        "strs_a": np.asarray(r.sci_state.ci_strs_a),
                        "strs_b": np.asarray(r.sci_state.ci_strs_b),
                        "e_tot": e_tot, "s2": s2})
            log(f"S1 it{it_box[0]} b{j}: dim {len(r.sci_state.ci_strs_a)}x"
                f"{len(r.sci_state.ci_strs_b)} E {e_tot:.6f} "
                f"(err {(e_tot-E_EXACT)*1e3:+8.3f}) S2 {s2:.4f}")

    res = diagonalize_fermionic_hamiltonian(
        H1, ERI, bit_array, samples_per_batch=SPB, norb=NC, nelec=NELEC,
        num_batches=NBATCH, max_iterations=MAXIT, symmetrize_spin=True,
        max_dim=DMAX, callback=cb, seed=SEED2)
    ef = res.energy + ECORE
    s2f = float(res.sci_state.spin_square())
    log(f"S1 BEST: E {ef:.6f} (err {(ef-E_EXACT)*1e3:+.3f} mHa) S2 {s2f:.4f}")
    CAP = cap
    if not SMOKE:
        assert abs(ef - ANCHOR_E) < 2e-6, f"anchor E mismatch: {ef}"
        assert abs(s2f - ANCHOR_S2) < 5e-3, f"anchor S2 mismatch: {s2f}"
        log("S1 anchors MATCH the published soup row")
        np.savez("sqdpen_spaces.npz", cap=np.array(cap, dtype=object),
                 best_e=ef, best_s2=s2f,
                 best_a=np.asarray(res.sci_state.ci_strs_a),
                 best_b=np.asarray(res.sci_state.ci_strs_b))
else:
    CAP = None

# ---- operator kit on a fixed space ------------------------------------------
H2E = ao2mo.restore(1, direct_spin1.absorb_h1e(H1, ERI, NC, NELEC, .5), NC)


class SpaceOps:
    """H and projected-S^2 matvecs on a fixed (strs_a, strs_b) product space,
    the exact operators kernel_fixed_space/fix_spin_/spin_square use."""

    def __init__(self, strs_a, strs_b):
        self.strs = (np.asarray(strs_a, dtype=np.int64),
                     np.asarray(strs_b, dtype=np.int64))
        self.na, self.nb = len(self.strs[0]), len(self.strs[1])
        self.n = self.na * self.nb
        self.link = _all_linkstr_index(self.strs, NC, NELEC)
        self.hdiag = fci.selected_ci.SelectedCI().make_hdiag(
            H1, ERI, self.strs, NC, NELEC, compress=True)
        self.nmv = [0, 0]
        # one-time cost probe (hop vs ssop)
        v = np.random.default_rng(0).standard_normal(self.n)
        t = time.time(); self.hop(v); th = time.time() - t
        t = time.time(); self.ssop(v); ts = time.time() - t
        log(f"  ops {self.na}x{self.nb}: hop {th*1e3:.0f} ms, "
            f"ssop {ts*1e3:.0f} ms")

    def hop(self, c):
        self.nmv[0] += 1
        v = _as_SCIvector(np.asarray(c).reshape(self.na, self.nb), self.strs)
        return selected_ci.contract_2e(H2E, v, NC, NELEC,
                                       self.link).ravel()

    def ssop(self, c):
        self.nmv[1] += 1
        v = _as_SCIvector(np.asarray(c).reshape(self.na, self.nb), self.strs)
        return contract_ss(v, NC, NELEC).ravel()

    def point(self, v):
        """(E_H, <S^2>, Var(S^2)) of a normalized vector."""
        v = np.asarray(v).ravel()
        v = v / np.linalg.norm(v)
        hv, sv = self.hop(v), self.ssop(v)
        e = float(v @ hv) + ECORE
        s2 = float(v @ sv)
        var = float(sv @ sv) - s2 ** 2
        return e, s2, var

    def solve_pen(self, lam, v0=None, tol=1e-9):
        """Ground state of H + lam*S^2 by eigsh; returns (vec, info dict)."""
        def mv(c):
            out = self.hop(c)
            if lam != 0.0:
                out = out + lam * self.ssop(c)
            return out
        A = LinearOperator((self.n, self.n), matvec=mv, dtype=float)
        if v0 is None:
            v0 = np.zeros(self.n)
            v0[int(np.argmin(self.hdiag))] = 1.0
        ncv = min(self.n - 1, 64)
        th, vec = eigsh(A, k=1, which="SA", v0=v0, tol=tol, maxiter=5000,
                        ncv=ncv)
        vec = vec[:, 0]
        resid = float(np.linalg.norm(mv(vec) - th[0] * vec))
        e, s2, var = self.point(vec)
        return vec, {"lam": lam, "theta": float(th[0]) + ECORE, "e_h": e,
                     "err_mHa": (e - E_EXACT) * 1e3, "s2": s2, "var_s2": var,
                     "resid": resid}


def load_spaces():
    if os.path.exists("sqdpen_spaces.npz"):
        cap = list(np.load("sqdpen_spaces.npz", allow_pickle=True)["cap"])
    else:
        cap = CAP  # SMOKE: in-memory capture from S1
    it1 = min((r for r in cap if r["iter"] == 1), key=lambda r: r["e_tot"])
    best = min(cap, key=lambda r: r["e_tot"])
    return {"it1": it1, "it4": best}


def pyci_xcheck(strs_a, strs_b, tag):
    """Independent lam=0 ground energy on the same product det space."""
    import pyci
    ham = pyci.hamiltonian(FCID)
    wfn = pyci.fullci_wfn(ham.nbasis, *NELEC)
    aa = np.asarray(strs_a, dtype=np.uint64)
    bb = np.asarray(strs_b, dtype=np.uint64)
    for a in aa:
        for b in bb:
            wfn.add_det(np.array([a, b], dtype=np.uint64))
    op = pyci.sparse_op(ham, wfn)
    ev, _ = op.solve(n=1, tol=1e-9)
    log(f"S2 {tag}: pyci xcheck E {ev[0]:.8f}")
    return float(ev[0])


# ---- S2: the H-vs-S^2 frontier ----------------------------------------------
if "2" in STAGES and not os.path.exists("sqdpen_frontier.npz") \
        and (os.path.exists("sqdpen_spaces.npz") or CAP is not None):
    spaces = load_spaces()
    frontier = {}
    for tag in (["it1"] if SMOKE else ["it1", "it4"]):
        part = f"sqdpen_frontier_{tag}.npz"
        if os.path.exists(part):
            frontier[tag] = np.load(part, allow_pickle=True)["blk"].item()
            log(f"S2 {tag}: checkpoint hit")
            continue
        r = spaces[tag]
        ops = SpaceOps(r["strs_a"], r["strs_b"])
        aufbau = int("1" * NELEC[0], 2)
        log(f"S2 {tag}: space {ops.na}x{ops.nb}={ops.n:,} "
            f"(aufbau in space: {aufbau in set(int(x) for x in r['strs_a'])})")
        pts = []
        # lam=0 warm start from THEIR unpenalized kernel (which does converge
        # unpenalized) -- eigsh then re-converges independently to its own
        # tolerance: a free their-Davidson-vs-eigsh agreement check.
        myci0 = fci.selected_ci.SelectedCI()
        _, sv0 = kernel_fixed_space(myci0, H1, ERI, NC, NELEC, ops.strs)
        vec, info = ops.solve_pen(0.0, v0=np.asarray(sv0).ravel())
        log(f"S2 {tag} lam=0: E {info['e_h']:.6f} "
            f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
            f"Var {info['var_s2']:.3f} resid {info['resid']:.1e}")
        # certification against the shipped pipeline's own record
        if not SMOKE:
            anch = (ANCHOR_E_IT1, ANCHOR_S2_IT1) if tag == "it1" \
                else (ANCHOR_E, ANCHOR_S2)
            assert abs(info["e_h"] - anch[0]) < 5e-6, \
                f"{tag} lam=0 E vs shipped: {info['e_h']} vs {anch[0]}"
            assert abs(info["s2"] - anch[1]) < 5e-3
            log(f"S2 {tag}: lam=0 matches the shipped record")
        e_pyci = None if SMOKE else pyci_xcheck(r["strs_a"], r["strs_b"], tag)
        if e_pyci is not None:
            d_uHa = abs(e_pyci - info["e_h"]) * 1e6
            log(f"S2 {tag}: |pyci - eigsh| = {d_uHa:.2f} uHa")
            assert d_uHa < 50.0, "pyci xcheck gate"
        pts.append(info)
        lams = list(LAMS)
        for lam in lams:
            vec, info = ops.solve_pen(lam, v0=vec)
            log(f"S2 {tag} lam={lam:g}: E_H {info['e_h']:.6f} "
                f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
                f"Var {info['var_s2']:.4f} resid {info['resid']:.1e}")
            pts.append(info)
        # extend toward the singlet floor if not there yet
        lam = lams[-1]
        while not SMOKE and pts[-1]["s2"] > 0.02 and lam < 500:
            lam *= 4
            vec, info = ops.solve_pen(lam, v0=vec)
            log(f"S2 {tag} lam={lam:g}: E_H {info['e_h']:.6f} "
                f"(err {info['err_mHa']:+.3f}) S2 {info['s2']:.4f} "
                f"Var {info['var_s2']:.4f} resid {info['resid']:.1e}")
            pts.append(info)
        frontier[tag] = {"pts": pts, "e_pyci": e_pyci,
                         "na": ops.na, "nb": ops.nb,
                         "nmv": tuple(ops.nmv)}
        log(f"S2 {tag}: singlet-jaw endpoint err "
            f"{pts[-1]['err_mHa']:+.3f} mHa at S2 {pts[-1]['s2']:.4f}")
        if not SMOKE:
            np.savez(part, blk=np.array(frontier[tag], dtype=object))
    if not SMOKE:
        np.savez("sqdpen_frontier.npz",
                 frontier=np.array(frontier, dtype=object))

# ---- S3: their kernel with the penalty, fixed spaces ------------------------
if "3" in STAGES and not os.path.exists("sqdpen_theirs.npz") \
        and (os.path.exists("sqdpen_spaces.npz") or CAP is not None):
    spaces = load_spaces()
    rows = []
    # it1 pairs (default vs tuned) give the solver-vs-frontier cross-cert;
    # it4 default (shift 0.1, the shipped default) measures the silent
    # under-convergence at the operating point. it4 tuned omitted (hours;
    # the it4 frontier is certified by the lam=0 anchor + pyci instead).
    arms = [("it1", 0.1, None), ("it1", 0.2, None),
            ("it1", 0.1, MAXCYC), ("it1", 0.2, MAXCYC),
            ("it4", 0.1, None)]
    if SMOKE:
        arms = [("it1", 0.2, 5)]
    done_keys = set()
    if os.path.exists("sqdpen_theirs_part.npz"):
        rows = list(np.load("sqdpen_theirs_part.npz",
                            allow_pickle=True)["rows"])
        done_keys = {(r["tag"], r["shift"], r["max_cycle"]) for r in rows}
        log(f"S3: partial checkpoint hit ({len(rows)} arms)")
    for tag, shift, mc in arms:
        if (tag, shift, mc if mc is not None else "default") in done_keys:
            continue
        r = spaces[tag]
        strs = (np.asarray(r["strs_a"], dtype=np.int64),
                np.asarray(r["strs_b"], dtype=np.int64))
        myci = fci.selected_ci.SelectedCI()
        myci = fci.addons.fix_spin_(myci, ss=0.0, shift=shift)
        kw = {}
        if mc is not None:
            kw = {"max_cycle": mc, "tol": 1e-10}
        t1 = time.time()
        _, sv = kernel_fixed_space(myci, H1, ERI, NC, NELEC, strs, **kw)
        wall = time.time() - t1
        conv = bool(np.all(myci.converged)) \
            if hasattr(myci, "converged") else None
        ops = SpaceOps(*strs)
        v = np.asarray(sv).ravel()
        e, s2, var = ops.point(v)
        pen_rq = e + shift * s2
        rows.append({"tag": tag, "shift": shift,
                     "max_cycle": mc if mc is not None else "default",
                     "converged": conv, "e_h": e,
                     "err_mHa": (e - E_EXACT) * 1e3, "s2": s2,
                     "var_s2": var, "pen_rq": pen_rq, "wall_s": wall})
        log(f"S3 {tag} shift={shift} mc={mc or 'def'}: E_H {e:.6f} "
            f"(err {(e-E_EXACT)*1e3:+.3f}) S2 {s2:.4f} conv={conv} "
            f"penRQ {pen_rq:.6f} [{wall:.0f}s]")
        if not SMOKE:
            np.savez("sqdpen_theirs_part.npz",
                     rows=np.array(rows, dtype=object))
    if not SMOKE:
        np.savez("sqdpen_theirs.npz", rows=np.array(rows, dtype=object))

# ---- S5: the penalized protocol, end to end ---------------------------------
if "5" in STAGES and os.path.exists("sqdpen_spaces.npz"):
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import (
        diagonalize_fermionic_hamiltonian, SCIState, SCIResult,
    )

    d = np.load("sqdship_counts.npz")
    counts = {f"{int(bb):020b}{int(aa):020b}": int(nn)
              for aa, bb, nn in zip(d["a"], d["b"], d["n"])}
    bit_array = BitArray.from_counts(counts, num_bits=2 * NC)

    def pen_loop_solver(lam):
        def _solver(ci_strings, one_body, two_body, norb, nelec):
            out = []
            for sa, sb in ci_strings:
                ops = SpaceOps(sa, sb)
                # warm start: their unpenalized kernel (fast), then eigsh
                myci0 = fci.selected_ci.SelectedCI()
                _, sv0 = kernel_fixed_space(
                    myci0, H1, ERI, NC, NELEC,
                    (np.asarray(sa, dtype=np.int64),
                     np.asarray(sb, dtype=np.int64)))
                vec, info = ops.solve_pen(lam, v0=np.asarray(sv0).ravel())
                v2 = vec.reshape(ops.na, ops.nb)
                svv = _as_SCIvector(v2, ops.strs)
                mm = fci.selected_ci.SelectedCI()
                dm1s = mm.make_rdm1s(svv, NC, NELEC)
                occ = (np.diagonal(dm1s[0]).copy(),
                       np.diagonal(dm1s[1]).copy())
                st = SCIState(v2, ops.strs[0], ops.strs[1], NC, NELEC)
                out.append(SCIResult(info["e_h"] - ECORE, st,
                                     orbital_occupancies=occ))
            return out
        return _solver

    # 0.2 (the review's number) first so the headline arm lands earliest
    for lam in ([0.2, 0.1] if not SMOKE else []):
        outf = f"sqdpen_loop_l{int(round(lam*100)):02d}.npz"
        if os.path.exists(outf):
            log(f"S5 lam={lam}: {outf} exists, skip")
            continue
        log(f"=== S5: penalized protocol, ss=0 shift={lam}, converged "
            f"solves, MAXIT={MAXIT} seed {SEED2} ===")
        hist = []
        it_box = [0]

        def cb(results):
            it_box[0] += 1
            for j, r in enumerate(results):
                s2 = float(r.sci_state.spin_square())
                e_tot = r.energy + ECORE
                hist.append({"iter": it_box[0], "batch": j,
                             "dim_a": len(r.sci_state.ci_strs_a),
                             "dim_b": len(r.sci_state.ci_strs_b),
                             "e_tot": e_tot,
                             "err_mHa": (e_tot - E_EXACT) * 1e3, "s2": s2})
                log(f"S5 l{lam} it{it_box[0]} b{j}: dim "
                    f"{len(r.sci_state.ci_strs_a)}x"
                    f"{len(r.sci_state.ci_strs_b)} E {e_tot:.6f} "
                    f"(err {(e_tot-E_EXACT)*1e3:+8.3f}) S2 {s2:.4f}")

        res = diagonalize_fermionic_hamiltonian(
            H1, ERI, bit_array, samples_per_batch=SPB, norb=NC, nelec=NELEC,
            num_batches=NBATCH, max_iterations=MAXIT, symmetrize_spin=True,
            max_dim=DMAX, sci_solver=pen_loop_solver(lam), callback=cb,
            seed=SEED2)
        ef = res.energy + ECORE
        s2f = float(res.sci_state.spin_square())
        log(f"S5 lam={lam} BEST: E {ef:.6f} (err {(ef-E_EXACT)*1e3:+.3f} "
            f"mHa) S2 {s2f:.4f}")
        np.savez(outf, hist=np.array(hist, dtype=object), best_e=ef,
                 best_s2=s2f, lam=lam)

# ---- S4: summary -------------------------------------------------------------
if "4" in STAGES:
    log("==== SUMMARY ====")
    if os.path.exists("sqdpen_frontier.npz"):
        fr = np.load("sqdpen_frontier.npz", allow_pickle=True)[
            "frontier"].item()
        for tag, blk in fr.items():
            log(f"frontier {tag} ({blk['na']}x{blk['nb']}):")
            log(f"  {'lam':>8s} {'err_mHa':>10s} {'S2':>8s} {'VarS2':>8s} "
                f"{'resid':>9s}")
            for p in blk["pts"]:
                log(f"  {p['lam']:>8g} {p['err_mHa']:>+10.3f} "
                    f"{p['s2']:>8.4f} {p['var_s2']:>8.4f} "
                    f"{p['resid']:>9.1e}")
    if os.path.exists("sqdpen_theirs.npz"):
        rows = list(np.load("sqdpen_theirs.npz", allow_pickle=True)["rows"])
        log("their kernel_fixed_space + fix_spin_(ss=0):")
        for r in rows:
            log(f"  {r['tag']} shift={r['shift']} mc={r['max_cycle']}: "
                f"err {r['err_mHa']:+.3f} S2 {r['s2']:.4f} "
                f"conv={r['converged']} penRQ {r['pen_rq']:.6f}")
    for lam in [0.1, 0.2]:
        f = f"sqdpen_loop_l{int(round(lam*100)):02d}.npz"
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            log(f"penalized protocol lam={lam}: best err "
                f"{(float(z['best_e'])-E_EXACT)*1e3:+.3f} mHa "
                f"S2 {float(z['best_s2']):.4f}")

print("SQDPEN_DONE", flush=True)
