"""Gauge gate for the S^4 double-application kernel (Tier 3a validation).

Identity under test (M_s = 0 root vectors v_i):
  W  = S+ applied to {v_i}            (M_s = +1 image, deduped)
  G  = W^T W          = <v_i|S^2|v_j>          (the shipped Gram instrument)
  W2 = S+ applied to W                 (M_s = +2 image)
  G4 = W2^T W2 + 2 G  = <S^2 v_i|S^2 v_j>      (spin second moment)
  Var_i = G4_ii - G_ii^2

Known answers on H4/STO-3G FCI (M_s = 0):
  exact spin eigenstates  -> Var = 0 exactly, mean = S(S+1)
  mixture c|S=0> + s|S=1'> -> mean = 2 s^2, <S^4> = 4 s^2, Var = 4 s^2 c^2
"""
import numpy as np
from pyscf import gto, scf, fci

M1 = np.uint64(0x5555555555555555)
M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def popcount64(a):
    a = a - ((a >> np.uint64(1)) & M1)
    a = (a & M2) + ((a >> np.uint64(2)) & M2)
    a = (a + (a >> np.uint64(4))) & M4
    return (a * H01) >> np.uint64(56)


def splus_image(a, b, C, ncas):
    """Apply S+ = sum_p a^dag_{p,alpha} a_{p,beta} to vectors C living on
    dets (a, b). Returns deduped (A, B, W). Verbatim phase logic from
    _gram7500.py s2_gram_ab; only change: returns the image instead of W^T W.
    """
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
    _, inv = np.unique(np.stack([A, B], 1), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    nu = int(inv.max()) + 1
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return A[np.unique(inv, return_index=True)[1]], None, W


def splus_image_full(a, b, C, ncas):
    """Same but returns the deduped det words too (A, B, W)."""
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
    uniq, inv = np.unique(pair, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    nu = uniq.shape[0]
    W = np.column_stack([np.bincount(inv, weights=V[:, j], minlength=nu)
                         for j in range(k)])
    return uniq[:, 0], uniq[:, 1], W


def spin_moments(a, b, C, ncas):
    """Returns (G, G4): first and second S^2 moment matrices at M_s=0."""
    A, B, W = splus_image_full(a, b, C, ncas)
    G = W.T @ W
    _, _, W2 = splus_image_full(A, B, W, ncas)
    G4 = W2.T @ W2 + 2.0 * G
    return G, G4


mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
            basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
norb = mf.mo_coeff.shape[1]
nelec = (2, 2)
cis = fci.FCI(mol, mf.mo_coeff)
cis.nroots = 6
es, vs = cis.kernel()

# full-FCI det lists (pyscf string order)
from pyscf.fci import cistring
sa = np.asarray(cistring.make_strings(range(norb), nelec[0]), np.uint64)
sb = sa.copy()
na, nb = len(sa), len(sb)
aw = np.repeat(sa, nb)
bw = np.tile(sb, na)

print("root  E          pyscf_S2   gram_S2    S4         Var")
vmat = np.column_stack([np.asarray(v).ravel() for v in vs])
G, G4 = spin_moments(aw, bw, vmat, norb)
ok = True
for i, v in enumerate(vs):
    s2_ref = fci.spin_op.spin_square0(v, norb, nelec)[0]
    m, q = G[i, i], G4[i, i]
    var = q - m * m
    print(f"{i}  {es[i]:.8f}  {s2_ref:9.6f}  {m:9.6f}  {q:9.6f}  {var:.3e}")
    ok &= abs(m - s2_ref) < 1e-9
    # eigenstates: variance must vanish, mean must be S(S+1)-integral
    ok &= var < 1e-9

# mixture gate: c|root_singlet> + s|root_triplet>, exact Var = 4 s^2 c^2
i0 = min(range(len(vs)),
         key=lambda i: fci.spin_op.spin_square0(vs[i], norb, nelec)[0])
i1 = min(range(len(vs)), key=lambda i: abs(
    fci.spin_op.spin_square0(vs[i], norb, nelec)[0] - 2.0))
c, s = np.sqrt(0.7), np.sqrt(0.3)
mix = c * np.asarray(vs[i0]).ravel() + s * np.asarray(vs[i1]).ravel()
mix /= np.linalg.norm(mix)
Gm, G4m = spin_moments(aw, bw, mix, norb)
m, q = Gm[0, 0], G4m[0, 0]
var = q - m * m
print(f"mix 0.7/0.3 (roots {i0}/{i1}): mean {m:.6f} (want {2*0.3:.6f})  "
      f"S4 {q:.6f} (want {4*0.3:.6f})  Var {var:.6f} (want {4*0.3*0.7:.6f})")
ok &= abs(m - 0.6) < 1e-9 and abs(q - 1.2) < 1e-9 and abs(var - 0.84) < 1e-9
print("S4GATE", "PASS" if ok else "FAIL")
