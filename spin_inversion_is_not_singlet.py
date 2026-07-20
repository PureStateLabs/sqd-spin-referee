"""MINIMAL REPRODUCER (self-contained, ~35 s): spin-inversion closure -- the
'spin completion' mitigation used in flagship SQD work -- makes a determinant
subspace invariant under alpha<->beta exchange, but does NOT make its lowest
eigenvector a spin singlet. An <S^2> diagnostic is required to know which spin
state a reported energy belongs to.

Runs on IBM's OWN public [2Fe-2S] data (redistributed in their Zenodo archive
15324153): their MO FCIDUMP, and the exact-marginal ranking of their optimized
LUCJ circuit. Requires only numpy + pyscf.

  python spin_inversion_is_not_singlet.py

The key structural fact this exploits (printed and asserted below): for this
M_s=0 system the top-N alpha half-strings and top-N beta half-strings ranked by
the circuit marginals are IDENTICAL sets. An outer-product determinant space
built from one shared string set is therefore *already* invariant under the
alpha<->beta spin inversion -- it satisfies the symmetrize_spin closure
structurally, with nothing to add. So this is the closure's best case, not a
strawman. We then diagonalize IBM's Hamiltonian in it and measure spin.

Observed (a singlet would be <S^2> = 0):
  * the ground state's <S^2> is far from 0 (high-spin contaminated);
  * over the low-energy roots, the rotation-invariant <S^2> Gram matrix has a
    minimum eigenvalue also far from 0 -- so NO linear combination within that
    manifold is a singlet either.

Spin-inversion symmetry is present by construction; a singlet is not produced.
This N=60 space is a fast local witness (its energy is far from converged --
that is not the point); the contamination persists to millions of determinants
(<S^2> ~ 3.4-3.8 on the certified ladder) and to the 5.625e7 flagship dimension
(<S^2> ~ 1.3), documented in the paper. Reported to reproduce in one command.
"""
import numpy as np
from pyscf import ao2mo, fci
from pyscf.tools import fcidump

FCID = ("_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/"
        "fcidump_Fe2S2_MO.txt")
MARG = "lucjopt_marginals.npz"
NORB, NELEC = 20, (15, 15)          # [2Fe-2S] active space, M_s = 0
N = 60                              # strings/spin -> 3,600 dets, ~seconds
NROOTS = 1                          # ground root only (the dense near-degenerate
#                                     low-energy spectrum makes multi-root slow;
#                                     the ground <S^2> alone makes the point --
#                                     the full low-manifold Gram floor is the
#                                     gram2000/gram7500 result in the paper)

M1 = np.uint64(0x5555555555555555)
M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def popcount64(a):
    a = np.asarray(a, np.uint64)
    with np.errstate(over="ignore"):  # SWAR multiply overflow is intentional
        a = a - ((a >> np.uint64(1)) & M1)
        a = (a & M2) + ((a >> np.uint64(2)) & M2)
        a = (a + (a >> np.uint64(4))) & M4
        return (a * H01) >> np.uint64(56)


def s2_matrix(sa, sb, C, norb):
    """<v_i|S^2|v_j> = <S+ v_i|S+ v_j> for M_s=0 vectors over the outer-product
    space of alpha strings sa and beta strings sb; C columns are eigenvectors
    in (alpha-outer, beta-inner) row-major order."""
    a = np.repeat(np.asarray(sa, np.uint64), len(sb))
    b = np.tile(np.asarray(sb, np.uint64), len(sa))
    oa, ob, ov = [], [], []
    for p in range(norb):
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
    A = np.concatenate(oa)
    B = np.concatenate(ob)
    V = np.concatenate(ov, 0)
    _, inv = np.unique(np.stack([A, B], 1), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    W = np.column_stack([np.bincount(inv, weights=V[:, j],
                                     minlength=int(inv.max()) + 1)
                         for j in range(V.shape[1])])
    return W.T @ W


# --- IBM's own optimized-circuit marginal ranking ---
z = np.load(MARG)
ra, rb = z["ra"], z["rb"]
strs = z["strs"]
sa_all = strs[ra] if int(ra.max()) < 32767 else ra
sb_all = strs[rb] if int(rb.max()) < 32767 else rb
top_a = set(int(x) for x in sa_all[:N])
top_b = set(int(x) for x in sb_all[:N])
overlap = len(top_a & top_b)
print(f"IBM public [2Fe-2S]; singlet target has <S^2> = 0.\n")
print(f"top-{N} alpha strings vs top-{N} beta strings (circuit marginals): "
      f"{overlap}/{N} identical")
assert overlap == N, "expected identical alpha/beta sets for this M_s=0 state"
print("=> the outer-product space is ALREADY invariant under alpha<->beta "
      "spin inversion")
print("   (symmetrize_spin closure is structural here -- nothing to add; "
      "this is its best case)\n")

S = np.sort(sa_all[:N].astype(np.int64))     # one shared string set

fd = fcidump.read(FCID)
h1 = fd["H1"]
eri = ao2mo.restore(1, fd["H2"], NORB)

ci = fci.selected_ci.SelectedCI()
e, v = fci.selected_ci.kernel_fixed_space(
    ci, h1, eri, NORB, NELEC, (S, S), nroots=NROOTS, verbose=0)
e = np.atleast_1d(e)
vs = v if isinstance(v, (list, tuple)) else [v]
C = np.column_stack([np.asarray(x).ravel() for x in vs])
G = s2_matrix(S, S, C, NORB)

print(f"diagonalizing IBM's Hamiltonian in the {len(S)*len(S):,}-determinant "
      f"spin-inversion-symmetric space:")
print(f"  ground energy (Ha)      : {e[0]:.6f}  "
      f"(err {(e[0]+116.6056091)*1e3:+.1f} mHa vs exact singlet)")
print(f"  ground <S^2>            : {G[0,0]:.3f}     (singlet = 0)\n")
print(f"RESULT: the determinant space is exactly invariant under alpha<->beta "
      f"spin inversion, yet its ground state has <S^2> = {G[0,0]:.2f}, not 0. "
      f"Spin-inversion\nclosure permits spin symmetry but does not produce a "
      f"singlet -- an <S^2> readout is required to know which spin state the "
      f"energy belongs to.\n(The rotation-invariant floor over the full "
      f"low-energy manifold is the gram2000/gram7500 result in the paper.)")
