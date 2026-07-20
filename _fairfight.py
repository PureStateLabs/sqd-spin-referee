"""FAIRFIGHT: do quantum-circuit samples beat classical selection at matched cost?

The exact-regime adjudication of the SQD/QSCI controversy (Reinholdt et al.
arXiv:2501.07231 "no winning regime" vs the flagship SQD narrative), run on
systems where exact FCI is available so every claim is checkable.

Six arms, ONE pipeline: every arm produces a determinant set of matched size D;
the SAME sparse solve gives E_var, the SAME Epstein-Nesbet PT2 gives E_var+PT2;
scored against exact FCI (three-way cross-checked: our solve on the full sector
== pyscf FCI from (h1,h2) == the stored target e_fci).

  cipsi_hf   classical: CIPSI selection (|r|^2/dE ranking) from an HF seed
  hci_hf     classical: heat-bath selection (max_i |H_ai c_i|) from an HF seed
  hci_cisd   classical: heat-bath from the standard CISD starting space
             (HF + all singles+doubles) -- the STRONG classical baseline
  exact      quantum-idealized: sample B shots from the exact FCI |c|^2
             (Reinholdt's optimistic model), keep top-D dets by frequency
  lucj       quantum-realistic: sample the ffsim LUCJ(t2-CCSD) state (the
             flagship ansatz), top-D by frequency
  lucj_noise lucj under the depol+readout noise model with PN post-selection
             (REALCAL Cepheus files if present, else flat LAM/READOUT)

Per system: table over D in DGRID x arms of E_var+PT2-FCI (mHa), plus E_var,
and the diradical-configuration weight in the subspace ground state (wrong-
state-trapping diagnostic at near-degeneracies).

Env: GLOB (default "dz_r090.00.npz dz_i180.00.npz" space-separated or glob),
DGRID ("64,128,256,512,1024,2048,4096"), BSHOTS ("200000" shots for sampled
arms), NREPS (1, LUCJ layers), LAM (0.33), READOUT (0.03), REALCAL (0/1),
SEED (7), NT (4), TAG (out). Results: fairfight_{TAG}.npz. Sentinel FAIRFIGHT_DONE.
"""
import glob as globmod
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from pyscf import lib

lib.num_threads(int(os.environ.get("NT", "4")))
from pyscf import ao2mo, cc, fci, gto, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from indrajala.chem.hamiltonian import qubit_hamiltonian  # noqa: E402

T0 = time.time()
RNG = np.random.default_rng(int(os.environ.get("SEED", "7")))
M1 = 0x5555555555555555; M2 = 0x3333333333333333
M4 = 0x0f0f0f0f0f0f0f0f; H01 = 0x0101010101010101


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def popcount64(a):
    a = a - ((a >> 1) & M1)
    a = (a & M2) + ((a >> 2) & M2)
    a = (a + (a >> 4)) & M4
    return (a * H01) >> 56


def _fix_phase(v):
    """Complex eigensolvers return vectors with an arbitrary global phase;
    rotate by the largest component's phase so .real is the actual vector
    (signed amplitudes matter for S^2 / overlap consumers)."""
    v = np.asarray(v)
    if not np.iscomplexobj(v):
        return v
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
    return v.real


class Engine:
    """Pauli-action determinant-space engine (validated in _noise_det/_qsci):
    sparse H in any det subset, EN-PT2 of the external space, selection growth
    under either the CIPSI or the heat-bath criterion."""

    def __init__(self, h1, h2, ecore, ncas, na, nb):
        H = qubit_hamiltonian(h1, h2, ecore, ncas).simplify()
        self.ncas, self.na, self.nb = ncas, na, nb
        zmask = np.empty(len(H.paulis), np.int64)
        xmask = np.empty(len(H.paulis), np.int64)
        coeff = np.empty(len(H.paulis), complex)
        for i, (p, c) in enumerate(zip(H.paulis, H.coeffs)):
            z = p.z[::-1]; x = p.x[::-1]
            zmask[i] = int("".join("1" if b else "0" for b in z), 2)
            xmask[i] = int("".join("1" if b else "0" for b in x), 2)
            coeff[i] = complex(c) * (-1j) ** int(np.sum(p.z & p.x))
        diag = xmask == 0
        self.zd, self.cd = zmask[diag], coeff[diag].real
        self.zo, self.xo, self.co = zmask[~diag], xmask[~diag], coeff[~diag]
        self.am = np.int64((1 << ncas) - 1)
        self.bm = np.int64(((1 << ncas) - 1) << ncas)
        self.hf = np.int64(((1 << na) - 1) | (((1 << nb) - 1) << ncas))
        self.chunk = int(os.environ.get("CHUNK", "4000"))

    def Hdiag(self, dets):
        out = np.zeros(len(dets))
        for zmk, c in zip(self.zd, self.cd):
            out += c * (1 - 2 * (popcount64(dets & zmk) & 1))
        return out

    def solve(self, dets):
        dets = np.unique(np.asarray(dets, np.int64))
        Dn = len(dets)
        rows = [np.arange(Dn)]; cols = [np.arange(Dn)]
        vals = [self.Hdiag(dets).astype(complex)]
        for zmk, xmk, c in zip(self.zo, self.xo, self.co):
            conn = dets ^ xmk
            j = np.searchsorted(dets, conn)
            ok = (j < Dn) & (dets[np.clip(j, 0, Dn - 1)] == conn)
            if not ok.any():
                continue
            src = np.nonzero(ok)[0]; dst = j[ok]
            sign = 1 - 2 * (popcount64(dets[src] & zmk) & 1)
            rows.append(dst); cols.append(src); vals.append(c * sign)
        Hs = csr_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(Dn, Dn))
        Hs = (Hs + Hs.getH()) * 0.5
        if Dn < 50:
            w, V = np.linalg.eigh(Hs.toarray())
            return float(w[0]), _fix_phase(V[:, 0]), dets
        w, V = eigsh(Hs, k=1, which="SA", maxiter=4000, tol=1e-9)
        return float(w[0]), _fix_phase(V[:, 0]), dets

    def solve_k(self, dets, k=2):
        """Lowest-k eigenpairs in the det subspace (for state-averaged arms)."""
        dets = np.unique(np.asarray(dets, np.int64))
        Dn = len(dets)
        rows = [np.arange(Dn)]; cols = [np.arange(Dn)]
        vals = [self.Hdiag(dets).astype(complex)]
        for zmk, xmk, c in zip(self.zo, self.xo, self.co):
            conn = dets ^ xmk
            j = np.searchsorted(dets, conn)
            ok = (j < Dn) & (dets[np.clip(j, 0, Dn - 1)] == conn)
            if not ok.any():
                continue
            src = np.nonzero(ok)[0]; dst = j[ok]
            sign = 1 - 2 * (popcount64(dets[src] & zmk) & 1)
            rows.append(dst); cols.append(src); vals.append(c * sign)
        Hs = csr_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(Dn, Dn))
        Hs = (Hs + Hs.getH()) * 0.5
        kk = min(k, Dn - 1) if Dn > 1 else 1
        if Dn < 60 or kk < 1:
            w, V = np.linalg.eigh(Hs.toarray())
            return w[:k], np.column_stack(
                [_fix_phase(V[:, i]) for i in range(min(k, V.shape[1]))]), dets
        w, V = eigsh(Hs, k=kk, which="SA", maxiter=4000, tol=1e-9)
        o = np.argsort(w)
        return w[o], np.column_stack(
            [_fix_phase(V[:, i]) for i in o]), dets

    def grow_to_sa(self, dets, V, ws, target):
        """State-averaged CIPSI growth: external importance = max over the
        tracked roots of |r^(root)_a|^2 / |dE_root| — the standard classical
        fix for wrong-state trapping at near-degeneracies."""
        score = None
        u_ref = None
        for r_i in range(V.shape[1]):
            u, r = self._external(dets, V[:, r_i], "sum")
            if len(u) == 0:
                continue
            s = np.abs(r) ** 2 / np.abs(self.Hdiag(u) - ws[r_i] + 1e-9)
            if score is None:
                score, u_ref = s, u
            else:
                score = np.maximum(score, s)   # same u for every root
        if score is None:
            return dets
        k = max(0, target - len(dets))
        if k == 0:
            return dets
        top = u_ref[np.argsort(score)[::-1][:k]]
        return np.union1d(dets, top)

    def _external(self, dets, c, agg):
        """Accumulate per-external-det statistic over all H-connections from the
        current space. agg='sum' -> r_a = sum_i H_ai c_i (CIPSI/PT2);
        agg='max' -> max_i |H_ai c_i| (heat-bath)."""
        Dn = len(dets); cc_ = []; vv = []
        for s0 in range(0, Dn, self.chunk):
            ds = dets[s0:s0 + self.chunk]; cs = c[s0:s0 + self.chunk]
            for zmk, xmk, cf in zip(self.zo, self.xo, self.co):
                amp = cf * (1 - 2 * (popcount64(ds & zmk) & 1)) * cs
                cn = ds ^ xmk
                keep = ((popcount64(cn & self.am) == self.na)
                        & (popcount64(cn & self.bm) == self.nb))
                if not keep.any():
                    continue
                cc_.append(cn[keep]); vv.append(amp[keep])
        if not cc_:
            return np.empty(0, np.int64), np.empty(0)
        conn = np.concatenate(cc_); amp = np.concatenate(vv)
        j = np.searchsorted(dets, conn)
        inset = (j < Dn) & (dets[np.clip(j, 0, Dn - 1)] == conn)
        conn = conn[~inset]; amp = amp[~inset]
        if len(conn) == 0:
            return np.empty(0, np.int64), np.empty(0)
        u, inv = np.unique(conn, return_inverse=True)
        if agg == "sum":
            r = np.zeros(len(u), complex)
            np.add.at(r, inv, amp)
            return u, r
        r = np.zeros(len(u))
        np.maximum.at(r, inv, np.abs(amp))
        return u, r

    def pt2(self, dets, c, e0):
        u, r = self._external(dets, c, "sum")
        if len(u) == 0:
            return 0.0
        val = np.abs(r) ** 2 / np.abs(self.Hdiag(u) - e0 + 1e-9)
        return -float(val.sum())

    def grow_to(self, dets, c, e0, target, mode):
        """Add externals up to `target` total dets under `mode` criterion."""
        u, r = self._external(dets, c, "sum" if mode == "cipsi" else "max")
        if len(u) == 0:
            return dets
        if mode == "cipsi":
            score = np.abs(r) ** 2 / np.abs(self.Hdiag(u) - e0 + 1e-9)
        else:
            score = r
        k = max(0, target - len(dets))
        if k == 0:
            return dets
        top = u[np.argsort(score)[::-1][:k]]
        return np.union1d(dets, top)


def exact_reference(d, nroots=4):
    """pyscf FCI from (h1,h2), lowest-nroots WITH spin labels: [(E, mult,
    |c|^2)] + det encoding. The adjudication TARGET is selected by TARGET_MULT
    (env; default = root0 regardless of spin; 1 = lowest singlet, 3 = lowest
    triplet) — the dz_r090 lesson: silent sector-root0 targeting compares
    different spin states across geometries."""
    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    h1, h2, ecore = d["h1"], d["h2"], float(d["ecore"])
    es, vecs = fci.direct_spin1.kernel(h1, h2, ncas, (na, nb), ecore=ecore,
                                       nroots=nroots, max_cycle=500,
                                       conv_tol=1e-12)
    stra = cistring.make_strings(range(ncas), na).astype(np.int64)
    strb = cistring.make_strings(range(ncas), nb).astype(np.int64)
    dets = (stra[:, None] | (strb[None, :] << ncas)).ravel()
    roots = []
    for e, v in zip(es, vecs):
        _, mult = fci.spin_op.spin_square(v, ncas, (na, nb))
        roots.append((float(e), float(mult), (np.asarray(v) ** 2).ravel()))
    return roots, dets


def _lucj_ansatz(d, n_reps):
    """CCSD-t2-initialized UCJ op in the SAME orbital basis as (h1,h2), via a
    hand-built converged-RHF shell (mo_coeff = identity). Returns (ucj, note)."""
    import ffsim

    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    h1, h2 = d["h1"], d["h2"]
    assert na == nb, "LUCJ arm assumes closed-shell sector"
    mol = gto.M(verbose=0)
    mol.nelectron = na + nb
    mol.spin = 0
    mol.incore_anyway = True
    mol.max_memory = 8000
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: h1
    mf.get_ovlp = lambda *a: np.eye(ncas)
    mf._eri = ao2mo.restore(8, np.asarray(h2), ncas)
    dm = np.zeros((ncas, ncas))
    for i in range(na):
        dm[i, i] = 2.0
    vj, vk = mf.get_jk(mol, dm)
    fock = h1 + vj - 0.5 * vk
    mf.mo_coeff = np.eye(ncas)
    mf.mo_energy = np.diag(fock).copy()
    mf.mo_occ = np.array([2.0] * na + [0.0] * (ncas - na))
    mf.converged = True
    note = "ccsd"
    mycc = cc.CCSD(mf)
    mycc.max_cycle = 200
    try:
        mycc.kernel()
        t2 = mycc.t2
        if not mycc.converged:
            note = "ccsd-UNCONVERGED"
    except Exception as ex:
        note = f"ccsd-FAILED({type(ex).__name__})->mp2"
        t2 = mycc.init_amps()[2]     # MP2 amplitudes fallback
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps), note


def _lucj_vec_to_dets(vec, ncas, na, nb):
    """ffsim sector vector -> (dets in OUR encoding, probs), HF self-tested
    when HF is the max-weight config (skip the assert otherwise)."""
    import ffsim

    prob = np.abs(vec) ** 2
    stra, strb = ffsim.addresses_to_strings(
        np.arange(len(vec)), norb=ncas, nelec=(na, nb), concatenate=False)
    dets = np.asarray(stra, np.int64) | (np.asarray(strb, np.int64) << ncas)
    return dets, prob


def lucj_distribution(d, n_reps):
    """LUCJ(t2) distribution over dets in OUR encoding, HF-mapping self-tested.
    Returns (dets, probs, note)."""
    import ffsim

    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    ucj, note = _lucj_ansatz(d, n_reps)
    ref = ffsim.hartree_fock_state(ncas, (na, nb))
    vec = ffsim.apply_unitary(ref, ucj, norb=ncas, nelec=(na, nb))
    dets, prob = _lucj_vec_to_dets(vec, ncas, na, nb)
    hf = np.int64(((1 << na) - 1) | (((1 << nb) - 1) << ncas))
    imax = int(np.argmax(prob))
    assert dets[imax] == hf, (
        f"ffsim det-mapping self-test FAILED: argmax {dets[imax]:b} != HF {hf:b}")
    return dets, prob, note


def lucj_optimized(d, n_reps):
    """The quantum side's rebuttal: variationally optimize the LUCJ parameters
    (VQE-refined, classically simulable at this scale) starting from CCSD-t2,
    then sample the OPTIMIZED state. Returns (dets, probs, note)."""
    import ffsim
    from scipy.optimize import minimize

    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    h1, h2, ecore = d["h1"], np.asarray(d["h2"]), float(d["ecore"])
    ucj0, note0 = _lucj_ansatz(d, n_reps)
    mol_ham = ffsim.MolecularHamiltonian(
        one_body_tensor=np.asarray(h1, float),
        two_body_tensor=ao2mo.restore(1, h2, ncas), constant=ecore)
    linop = ffsim.linear_operator(mol_ham, norb=ncas, nelec=(na, nb))
    ref = ffsim.hartree_fock_state(ncas, (na, nb))

    def state(x):
        op = ffsim.UCJOpSpinBalanced.from_parameters(
            x, norb=ncas, n_reps=n_reps)
        return ffsim.apply_unitary(ref, op, norb=ncas, nelec=(na, nb))

    def energy(x):
        v = state(x)
        return float(np.real(np.vdot(v, linop @ v)))

    x0 = ucj0.to_parameters()
    e0 = energy(x0)
    maxit = int(os.environ.get("LUCJ_MAXITER", "40"))
    res = minimize(energy, x0, method="L-BFGS-B",
                   options={"maxiter": maxit, "ftol": 1e-12, "gtol": 1e-8})
    vec = state(res.x)
    dets, prob = _lucj_vec_to_dets(vec, ncas, na, nb)
    note = (f"{note0}; opt E {e0:.6f}->{res.fun:.6f} "
            f"({res.nit} it, {res.nfev} evals)")
    return dets, prob, note


def sample_topD(dets, prob, B, targets, noise=None):
    """B shots from (dets,prob); optional noise dict corrupts shots (depol ->
    uniform over 2^NQ + readout flips) then PN post-selects. Returns
    {D: det-array of the top-D most-frequent sampled dets} + survival."""
    p = prob / prob.sum()
    draws = RNG.choice(dets, size=B, p=p)
    surv = 1.0
    if noise is not None:
        NQ, na, nb = noise["NQ"], noise["na"], noise["nb"]
        f = 1.0 - np.exp(-noise["lam"])
        n_noise = RNG.binomial(B, f)
        if n_noise:
            draws[:n_noise] = RNG.integers(0, 1 << NQ, size=n_noise,
                                           dtype=np.int64)
        ro = noise["ro"]
        bits = ((draws[:, None] >> np.arange(NQ)) & 1).astype(np.int8)
        bits ^= (RNG.random((len(draws), NQ)) < ro)
        draws = (bits.astype(np.int64) << np.arange(NQ)).sum(axis=1)
        am = np.int64((1 << (NQ // 2)) - 1)
        bm = am << (NQ // 2)
        keep = ((popcount64(draws & am) == na)
                & (popcount64(draws & bm) == nb))
        draws = draws[keep]
        surv = len(draws) / B
    vals, counts = np.unique(draws, return_counts=True)
    order = np.argsort(counts)[::-1]
    vals = vals[order]
    out = {}
    for Dt in targets:
        out[Dt] = vals[:Dt].copy()
    return out, surv, len(vals)


def cisd_space(eng):
    """HF + all singles and doubles (the standard classical starting space)."""
    hf = eng.hf
    ncas, na, nb = eng.ncas, eng.na, eng.nb
    occ_a = list(range(na)); vir_a = list(range(na, ncas))
    occ_b = [ncas + i for i in range(nb)]; vir_b = [ncas + i for i in range(nb, ncas)]
    dets = {int(hf)}

    def exc(det, i, a):
        return int(det) ^ (1 << i) ^ (1 << a)

    singles = []
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        for i in occ:
            for a in vir:
                s = exc(hf, i, a)
                singles.append((i, a, s))
                dets.add(s)
    for k1 in range(len(singles)):
        i1, a1, s1 = singles[k1]
        for k2 in range(k1 + 1, len(singles)):
            i2, a2, _ = singles[k2]
            if i1 == i2 or a1 == a2:
                continue
            dets.add(exc(s1, i2, a2))
    return np.array(sorted(dets), np.int64)


def run_system(fname, DGRID, BSHOTS, NREPS):
    d = np.load(fname)
    ncas, na, nb = int(d["ncas"]), int(d["na"]), int(d["nb"])
    NQ = 2 * ncas
    e_fci_t = float(d["e_fci"]); e_hf = float(d["e_hf"])
    eng = Engine(d["h1"], d["h2"], float(d["ecore"]), ncas, na, nb)

    # --- exactness cross-checks + spin-resolved target selection ---
    roots, all_dets = exact_reference(d)
    assert abs(roots[0][0] - e_fci_t) < 5e-6, \
        f"pyscf root0 {roots[0][0]} vs stored {e_fci_t}"
    tm = os.environ.get("TARGET_MULT")
    if tm is None:
        ti = 0
    else:
        ti = next(i for i, r in enumerate(roots)
                  if abs(r[1] - float(tm)) < 0.2)
    e_ref, mult_ref, prob = roots[ti]
    oi = next(i for i, r in enumerate(roots) if i != ti)
    prob_oth = roots[oi][2]
    spins = "  ".join(f"r{i}:E{r[0]:.6f}/m{r[1]:.1f}" for i, r in
                      enumerate(roots[:3]))
    if len(all_dets) <= 60000:
        e_our, _, _ = eng.solve(all_dets)
        assert abs(e_our - roots[0][0]) < 5e-6, \
            f"engine {e_our} vs root0 {roots[0][0]}"
        chk = "3-way FCI cross-check OK"
    else:
        chk = "pyscf-vs-stored OK (engine full-solve skipped)"
    log(f"{fname}: {chk}; sector {len(all_dets)}; {spins}")
    log(f"  TARGET root{ti} (mult {mult_ref:.1f}) E {e_ref:.6f}; "
        f"HF-target {(e_hf-e_ref)*1e3:+.1f} mHa; other root{oi} "
        f"{(roots[oi][0]-e_ref)*1e3:+.1f} mHa away")
    e_fci_t = e_ref                     # every arm scored vs the TARGET state
    # det -> probability lookup for the phase-free state-character diagnostic
    order_all = np.argsort(all_dets)
    dets_sorted = all_dets[order_all]
    p0_sorted = prob[order_all]          # target-state mass (pT)
    p1_sorted = prob_oth[order_all]      # nearest-other-state mass (pX)

    def mass(dets_D, p_sorted):
        j = np.searchsorted(dets_sorted, dets_D)
        ok = (j < len(dets_sorted)) & (dets_sorted[np.clip(j, 0, None)] == dets_D)
        return float(p_sorted[j[ok]].sum())

    def score(dets_D):
        """(E_var-FCI, E_var+PT2-FCI, pT, pX): pT/pX = probability mass of the
        exact ground/first-excited state captured by this det set. A trapped
        space shows pX >> pT."""
        e0, c, dd = eng.solve(dets_D)
        ept2 = eng.pt2(dd, c, e0)
        return ((e0 - e_fci_t) * 1e3, (e0 + ept2 - e_fci_t) * 1e3,
                mass(dd, p0_sorted), mass(dd, p1_sorted))

    results = {}

    # --- classical arms ---
    # Incremental growth (x1.5 per step, re-diagonalizing between steps) up to
    # each checkpoint -- standard SCI practice for BOTH criteria; a checkpoint
    # smaller than the current space is scored on the top-D dets by |c|
    # (budget discipline: every reported D is the size actually diagonalized).
    for arm, seed, mode in (("cipsi_hf", [eng.hf], "cipsi"),
                            ("hci_hf", [eng.hf], "hci"),
                            ("hci_cisd", cisd_space(eng), "hci")):
        rows = []
        dets = np.unique(np.asarray(seed, np.int64))
        e0, c, dets = eng.solve(dets)
        for Dt in DGRID:
            while len(dets) < Dt:
                step = min(Dt, max(len(dets) + 32, int(len(dets) * 1.5)))
                grown = eng.grow_to(dets, c, e0, step, mode)
                if len(grown) == len(dets):
                    break
                dets = grown
                e0, c, dets = eng.solve(dets)
            if len(dets) > Dt:
                top = dets[np.argsort(np.abs(c))[::-1][:Dt]]
                rows.append((Dt, *score(top)))
            else:
                rows.append((Dt, *score(dets[:])))
            log(f"  [{arm:9}] D={Dt:>5}: E_var {rows[-1][1]:+8.3f}  "
                f"+PT2 {rows[-1][2]:+8.3f} mHa  pT {rows[-1][3]:.3f} "
                f"pX {rows[-1][4]:.3f}")
        results[arm] = rows

    # --- classical rebuttal: STATE-AVERAGED CIPSI (root-trap fix) ---
    rows = []
    dets = np.array([eng.hf], np.int64)
    ws, V, dets = eng.solve_k(dets, k=2)
    for Dt in DGRID:
        while len(dets) < Dt:
            step = min(Dt, max(len(dets) + 32, int(len(dets) * 1.5)))
            grown = eng.grow_to_sa(dets, V, ws, step)
            if len(grown) == len(dets):
                break
            dets = grown
            ws, V, dets = eng.solve_k(dets, k=2)
        if len(dets) > Dt:
            top = dets[np.argsort(np.abs(V[:, 0]))[::-1][:Dt]]
            rows.append((Dt, *score(top)))
        else:
            rows.append((Dt, *score(dets[:])))
        log(f"  [cipsi_sa ] D={Dt:>5}: E_var {rows[-1][1]:+8.3f}  "
            f"+PT2 {rows[-1][2]:+8.3f} mHa  pT {rows[-1][3]:.3f} "
            f"pX {rows[-1][4]:.3f}")
    results["cipsi_sa"] = rows

    # --- sampled arms ---
    B = BSHOTS
    lucj_dets, lucj_prob, note = lucj_distribution(d, NREPS)
    log(f"  LUCJ({note}, reps={NREPS}): support {np.sum(lucj_prob>1e-12)} dets, "
        f"HF weight {lucj_prob.max():.3f}")
    if os.environ.get("REALCAL") and os.path.exists("cepheus_ro.json"):
        import json
        RO = json.load(open("cepheus_ro.json"))
        P2E = json.load(open("cepheus_p2.json"))
        ro_med = float(np.median(list(RO.values())))
        lam = float(np.median(list(P2E.values()))) * float(
            os.environ.get("CZN", "30"))
        ro = np.full(NQ, ro_med)
        log(f"  REALCAL noise: lam {lam:.3f}, readout {ro_med:.3f}")
    else:
        lam = float(os.environ.get("LAM", "0.33"))
        ro = np.full(NQ, float(os.environ.get("READOUT", "0.03")))
    noise = {"NQ": NQ, "na": na, "nb": nb, "lam": lam, "ro": ro}

    sampled = [("exact", (all_dets, prob, None)),
               ("lucj", (lucj_dets, lucj_prob, None)),
               ("lucj_noise", (lucj_dets, lucj_prob, noise))]
    if os.environ.get("OPT", "1") == "1":
        od, op_, onote = lucj_optimized(d, NREPS)
        log(f"  LUCJ-OPT({onote}): support {np.sum(op_>1e-12)} dets, "
            f"max weight {op_.max():.3f}")
        sampled.append(("lucj_opt", (od, op_, None)))
    for arm, (dd, pp, nz) in sampled:
        byD, surv, nuniq = sample_topD(dd, pp, B, DGRID, noise=nz)
        rows = []
        for Dt in DGRID:
            got = byD[Dt]
            if len(got) < Dt:
                rows.append((Dt, *score(got), len(got)))
            else:
                rows.append((Dt, *score(got), Dt))
            log(f"  [{arm:9}] D={Dt:>5}: E_var {rows[-1][1]:+8.3f}  "
                f"+PT2 {rows[-1][2]:+8.3f} mHa  pT {rows[-1][3]:.3f} "
                f"pX {rows[-1][4]:.3f}"
                + (f"  (only {len(got)} uniq)" if len(got) < Dt else ""))
        log(f"  [{arm:9}] B={B} shots -> {nuniq} unique dets"
            + (f", PN survival {surv:.2f}" if nz else ""))
        results[arm] = rows

    return {"file": fname, "e_hf_mha": (e_hf - e_fci_t) * 1e3,
            "sector": len(all_dets), "lucj_note": note, "results": results}


def main():
    pats = os.environ.get("GLOB", "dz_r090.00.npz dz_i180.00.npz").split()
    files = []
    for p in pats:
        files.extend(sorted(globmod.glob(p)))
    assert files, f"no targets match {pats}"
    DGRID = [int(x) for x in os.environ.get(
        "DGRID", "64,128,256,512,1024,2048,4096").split(",")]
    BSHOTS = int(os.environ.get("BSHOTS", "2000000"))
    NREPS = int(os.environ.get("NREPS", "1"))
    out = [run_system(f, DGRID, BSHOTS, NREPS) for f in files]
    tag = os.environ.get("TAG", "out")
    np.savez(f"fairfight_{tag}.npz", results=np.array(out, dtype=object))
    log(f"saved fairfight_{tag}.npz")
    print("FAIRFIGHT_DONE", flush=True)


if __name__ == "__main__":
    main()
