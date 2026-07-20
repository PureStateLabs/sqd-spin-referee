"""TB7500BOX: independent-implementation (PyCI-built H + our own block
Davidson) three-root eigensolve of IBM's flagship [2Fe-2S] outer-product
subspace at D = 5.625e7 (7500^2), on their own MO FCIDUMP -- the second-kernel
run that closes paper Limitations 1(ii).

Architecture (every element locally pre-flight-validated, 2026-07-16;
round-4 correction: pyci's rect build row-enumerates ALL nrow rows and ncol
only filters STORAGE, so column-chunking costs NCH x full enumeration --
measured live, $8 -- the scheme below is ROW-chunked and costs ONE
enumeration total):
  * row-chunk pyci builds: chunk c = rows [r0, r1); permuted wfn
    [chunk, rest] (insertion order preserved -- proven); rect
    sparse_op(ham, w, CH, D, symmetric=False) = FULL rows for the chunk
    (cost ~ CH); closed-form column map g(j) = r0+j | j-CH | j; keep
    col_g <= row_g (global lower triangle; each physical pair kept EXACTLY
    once -- proven zero-duplicate, exact-nnz, matvec 1e-13)
  * chunk outputs are row-contiguous and disjoint -> assembly is pure
    concatenation in CANONICAL a-major global order (g = ia*N + ib), so the
    warm vecs (a-major ranked-string layout, overlap 0.9999 with delivered)
    map by IDENTITY
  * on-disk triangle (data f8 + idx i4, ~1.6 TB) on RAID-0 NVMe, np.memmap
    streamed through the numba tri_symm_mv kernel (proven 6e-16 vs pyci)
  * multiprocessing (fork) build workers -- pyci holds the GIL (measured), so
    processes, not threads
Gates (fail -> ABORT, everything syncs to S3):
  G0 payload hashes (strings sha256, FCIDUMP md5, first-8 probes)
  G1 chunk-0 determinism (parent build vs worker build: identical sha256)
  G2 assembled nnz == exact combinatorial prediction (f = 1.0000 identity)
  G3 warm Rayleigh quotients vs certified energies (< 2e-5 Ha per root)
  G4 final |E_k - e_cert_k| < 5e-6 Ha, residuals < 1e-7
  G5 |S2_k - s2_cert_k| < 5e-3 (pyscf full-sector embed)

Env: N (7500 | 600 smoke), NCH, NW, NBAND, NT, TOL, WALL_BUDGET_S, SCRATCH,
     FCID, STRINGS (payload npz), WARM (restart npz; smoke uses bank amps).
Smoke:  N=600 NCH=6 NW=2 NBAND=4 SCRATCH=tb7500_smoke  (desktop, ~10 min)
Box:    N=7500 NCH=512 NW=20 NBAND=64 NT=64 SCRATCH=/scratch
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NT = int(os.environ.get("NT", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("NUMBA_NUM_THREADS", str(NT))
import pyci  # noqa: E402
from numba import njit, prange  # noqa: E402

T0 = time.time()
N = int(os.environ.get("N", "7500"))
NCH = int(os.environ.get("NCH", "512"))
NW = int(os.environ.get("NW", "20"))
NBAND = int(os.environ.get("NBAND", "64"))
TOL = float(os.environ.get("TOL", "1e-8"))
BUDGET = float(os.environ.get("WALL_BUDGET_S", "1e9"))
SCRATCH = os.environ.get("SCRATCH", "/scratch")
FCID = os.environ.get(
    "FCID",
    "_ibm_data/sqd_data_repository-main/integrals/2Fe-2S/fcidump_Fe2S2_MO.txt")
STRINGS = os.environ.get("STRINGS", "tb7500_strings.npz")
WARM = os.environ.get("WARM", "gram7500_s4_restart.npz")
# warm-block jitter: 1e-5 guards fresh runs against exact deflation; a resume
# from a converged checkpoint (res ~1e-6) must set JITTER=0 or it re-diverges
JITTER = float(os.environ.get("JITTER", "1e-5"))
NC, NA, NB = 20, 15, 15
NROOTS = 3
D = N * N
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.time()-T0:9.1f}s] {m}", flush=True)


def fail(m):
    log(f"*** ABORT: {m}")
    sys.exit(3)


def sentinel(name, make=False):
    p = os.path.join(SCRATCH, name)
    if make:
        open(p, "w").write("1")
    return os.path.exists(p)


# ---------------- numba kernels (pre-flight proven) ----------------
@njit(parallel=True, fastmath=False, cache=True)
def tri_symm_mv(indptr, indices, data, diag, x, ybuf):
    nt = ybuf.shape[0]
    Dl = indptr.shape[0] - 1
    for t in prange(nt):
        for i in range(Dl):
            ybuf[t, i] = 0.0
        lo = (t * Dl) // nt
        hi = ((t + 1) * Dl) // nt
        for i in range(lo, hi):
            xi = x[i]
            acc = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                v = data[k]
                acc += v * x[j]
                ybuf[t, j] += v * xi
            ybuf[t, i] += acc - diag[i] * xi


@njit(parallel=True, cache=True)
def reduce_cols(ybuf, y):
    nt = ybuf.shape[0]
    for i in prange(y.shape[0]):
        s = 0.0
        for t in range(nt):
            s += ybuf[t, i]
        y[i] = s


@njit(cache=True)
def scatter_sorted(rows, cols, vals, cursors, out_idx, out_dat):
    for k in range(rows.shape[0]):
        r = rows[k]
        p = cursors[r]
        out_idx[p] = cols[k]
        out_dat[p] = vals[k]
        cursors[r] = p + 1


# ---------------- payload ----------------
pl = np.load(STRINGS, allow_pickle=True)
if N == 7500:
    A = np.asarray(pl["ci_strs_a"], np.int64)[:N]
    B = np.asarray(pl["ci_strs_b"], np.int64)[:N]
else:
    # smoke: strings MUST come from the same bank as the warm amps -- the
    # bank's own ranking is NOT a prefix of the 7500 payload ranking
    # (verified: first-600 sets differ), so slicing pl here mismatches the
    # amp ordering and G3 correctly aborts
    _bk = np.load(f"flagsolve_n{N}.npz", allow_pickle=True)
    A = np.asarray(_bk["ci_strs_a"], np.int64)[:N]
    B = np.asarray(_bk["ci_strs_b"], np.int64)[:N]
E_CERT = np.asarray(pl["e_cert"], np.float64)[:NROOTS] if "e_cert" in pl.files \
    else None
S2_CERT = np.asarray(pl["s2_cert"], np.float64)[:NROOTS] if "s2_cert" in \
    pl.files else None
if N == 7500:
    sha_a = hashlib.sha256(np.asarray(pl["ci_strs_a"],
                                      np.int64).tobytes()).hexdigest()
    sha_b = hashlib.sha256(np.asarray(pl["ci_strs_b"],
                                      np.int64).tobytes()).hexdigest()
    if sha_a != str(pl["sha_a"]) or sha_b != str(pl["sha_b"]):
        fail("G0 string sha256 mismatch")
    md5 = hashlib.md5(open(FCID, "rb").read()).hexdigest()
    if md5 != str(pl["fcid_md5"]):
        fail(f"G0 FCIDUMP md5 mismatch {md5}")
    if not (A[:8] == np.asarray(pl["sa_first8"], np.int64)).all():
        fail("G0 sa_first8 mismatch")
    log("G0 PASS payload hashes + probes")
pairs_a = np.repeat(A.astype(np.uint64), N)
pairs_b = np.tile(B.astype(np.uint64), N)

# ---------------- exact nnz prediction (structural gate) ----------------
_OI, _OJ = np.triu_indices(NA, 1)
_VI, _VJ = np.triu_indices(NC - NA, 1)


def partner_ranks(ranked):
    n = len(ranked)
    sidx = np.argsort(ranked, kind="stable")
    sval, rnk = ranked[sidx], sidx.astype(np.int32)
    big = np.int32(2 ** 31 - 1)
    R2 = np.empty((n, NA * (NC - NA)), np.int32)
    R4 = np.empty((n, len(_OI) * len(_VI)), np.int32)
    for r in range(n):
        s = int(ranked[r])
        bits = (s >> np.arange(NC)) & 1
        occ, vir = np.nonzero(bits)[0], np.nonzero(1 - bits)[0]
        for R, vals in ((R2, (s - (1 << occ)[:, None]
                              + (1 << vir)[None, :]).ravel()),
                        (R4, (s - ((1 << occ[_OI]) + (1 << occ[_OJ]))[:, None]
                              + ((1 << vir[_VI]) + (1 << vir[_VJ]))[None, :]
                              ).ravel())):
            p = np.clip(np.searchsorted(sval, vals), 0, n - 1)
            R[r] = np.where(sval[p] == vals, rnk[p], big)
    return R2, R4


def nnz_exact():
    R2a, R4a = partner_ranks(A)
    R2b, R4b = partner_ranks(B)
    s2a = float((R2a < N).sum(1).mean())
    s4a = float((R4a < N).sum(1).mean())
    s2b = float((R2b < N).sum(1).mean())
    s4b = float((R4b < N).sum(1).mean())
    kf = s2a + s4a + s2b + s4b + s2a * s2b
    return int(round(D * (1 + kf / 2.0)))


# ---------------- S1: chunked build (multiprocessing) ----------------
bounds = [(c * D) // NCH for c in range(NCH + 1)]


def build_chunk(c, return_sha=False):
    r0, r1 = bounds[c], bounds[c + 1]
    CH = r1 - r0
    order = np.concatenate([np.arange(r0, r1), np.arange(0, r0),
                            np.arange(r1, D)])
    ham_l = pyci.hamiltonian(FCID)
    w = pyci.fullci_wfn(ham_l.nbasis, NA, NB)
    pairs_perm = np.stack([pairs_a, pairs_b], 1)[order]
    for r_ in pairs_perm:
        w.add_det(r_)
    del pairs_perm
    if len(w) != D:
        raise RuntimeError(f"chunk {c}: wfn {len(w)} != {D}")
    # FULL rows for the chunk only: cost ~ CH (round-4 scaling assert)
    op = pyci.sparse_op(ham_l, w, CH, D, False)
    dat = np.asarray(op.data() if callable(op.data) else op.data, np.float64)
    idx = np.asarray(op.indices() if callable(op.indices) else op.indices
                     ).astype(np.int64)
    ipt = np.asarray(op.indptr() if callable(op.indptr) else op.indptr
                     ).astype(np.int64)
    del op, w
    # closed-form local-col -> global-col map for order [chunk, 0..r0, r1..D]
    cols_g = np.where(idx < CH, r0 + idx,
                      np.where(idx < CH + r0, idx - CH, idx)).astype(np.int32)
    del idx
    rows_g = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(ipt))
    keep = cols_g <= rows_g          # global lower triangle: each pair once
    kcols = cols_g[keep]
    kvals = dat[keep]
    cnt = np.bincount((rows_g[keep] - r0), minlength=CH).astype(np.int64)
    # per-row diagonal (every row's diag lives in its own chunk build)
    dsel = cols_g == rows_g
    diag_c = np.zeros(CH, np.float64)
    diag_c[(rows_g[dsel] - r0)] = dat[dsel]
    del rows_g, cols_g, dat
    sha = hashlib.sha256(kvals.tobytes()).hexdigest() if return_sha else ""
    base = os.path.join(SCRATCH, f"chunk_{c:04d}")
    kcols.tofile(base + ".cols.i4")
    kvals.tofile(base + ".vals.f8")
    cnt.tofile(base + ".cnt.i8")
    diag_c.tofile(base + ".diag.f8")
    return c, int(len(kvals)), sha


def _worker(c):
    try:
        return build_chunk(c, return_sha=(c == 0))
    except Exception as e:  # noqa: BLE001
        return c, -1, f"{type(e).__name__}: {e}"


# fingerprint of everything the triangle depends on: a stale SCRATCH from a
# different string set / integrals must never satisfy the stage sentinels
# (a leftover smoke dir once served a foreign triangle that passed RQ checks)
FPRINT = hashlib.sha256(
    A.tobytes() + B.tobytes()
    + hashlib.md5(open(FCID, "rb").read()).digest()).hexdigest()

nnz_pred = None
if not sentinel("S1_DONE"):
    log(f"=== S1 chunked build: N={N} D={D:,} NCH={NCH} NW={NW} ===")
    t = time.time()
    nnz_pred = nnz_exact()
    log(f"exact nnz_tri prediction: {nnz_pred:,} "
        f"(~{nnz_pred*12/1e12:.2f} TB on disk; {time.time()-t:.0f}s)")
    json.dump({"nnz_pred": nnz_pred, "fprint": FPRINT},
              open(os.path.join(SCRATCH, "nnz_pred.json"), "w"))
    # G1 determinism: chunk 0 in-parent
    t = time.time()
    _, k0, sha_parent = build_chunk(0, return_sha=True)
    log(f"chunk 0 parent build: kept {k0:,} ({time.time()-t:.0f}s)")
    kept_tot = 0
    done = 0
    t = time.time()
    with ProcessPoolExecutor(NW) as ex:
        for c, kept, sha in ex.map(_worker, range(NCH)):
            if kept < 0:
                fail(f"S1 chunk {c} failed: {sha}")
            kept_tot += kept
            done += 1
            if c == 0:
                if sha != sha_parent or kept != k0:
                    fail(f"G1 chunk-0 determinism: worker sha/kept differ "
                         f"({kept} vs {k0})")
                log("G1 PASS chunk-0 parent/worker builds identical")
            if done % max(1, NCH // 20) == 0:
                el = time.time() - t
                log(f"  S1 {done}/{NCH} chunks, kept {kept_tot:,} "
                    f"({el:.0f}s, ETA {el/done*(NCH-done):.0f}s)")
            if time.time() - T0 > BUDGET:
                fail("S1 wall budget exceeded")
    log(f"S1 kept total {kept_tot:,} vs predicted {nnz_pred:,}")
    if kept_tot != nnz_pred:
        fail(f"G2 nnz mismatch: kept {kept_tot:,} != predicted {nnz_pred:,}")
    log("G2 PASS assembled nnz == exact combinatorial prediction")
    sentinel("S1_DONE", True)
else:
    _meta = json.load(open(os.path.join(SCRATCH, "nnz_pred.json")))
    if _meta.get("fprint") != FPRINT:
        fail("S1_DONE sentinel from a DIFFERENT string set / integrals "
             "(fingerprint mismatch) -- wipe SCRATCH and rebuild")
    nnz_pred = _meta["nnz_pred"]
    log(f"S1 already done (nnz_pred {nnz_pred:,}, fingerprint verified)")

# ---------------- S2: band merge -> canonical CSR on disk ----------------
F_DAT = os.path.join(SCRATCH, "tri.data.f8")
F_IDX = os.path.join(SCRATCH, "tri.idx.i4")
F_IPT = os.path.join(SCRATCH, "tri.indptr.i8")
F_DIA = os.path.join(SCRATCH, "diag.f8")
if not sentinel("S2_DONE"):
    log("=== S2 concatenation assembly (row-contiguous chunks) ===")
    t = time.time()
    counts = np.concatenate([
        np.fromfile(os.path.join(SCRATCH, f"chunk_{c:04d}.cnt.i8"), np.int64)
        for c in range(NCH)])
    if counts.shape[0] != D:
        fail(f"S2 counts length {counts.shape[0]} != {D}")
    nnz_tot = int(counts.sum())
    if nnz_tot != nnz_pred:
        fail(f"S2 nnz {nnz_tot:,} != predicted {nnz_pred:,}")
    indptr = np.zeros(D + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    indptr.tofile(F_IPT)
    diag = np.concatenate([
        np.fromfile(os.path.join(SCRATCH, f"chunk_{c:04d}.diag.f8"),
                    np.float64) for c in range(NCH)])
    diag.tofile(F_DIA)
    out_dat = np.memmap(F_DAT, np.float64, "w+", shape=(nnz_tot,))
    out_idx = np.memmap(F_IDX, np.int32, "w+", shape=(nnz_tot,))
    pos = 0
    for c in range(NCH):
        kc = np.fromfile(os.path.join(SCRATCH, f"chunk_{c:04d}.cols.i4"),
                         np.int32)
        kv = np.fromfile(os.path.join(SCRATCH, f"chunk_{c:04d}.vals.f8"),
                         np.float64)
        out_idx[pos:pos + len(kc)] = kc
        out_dat[pos:pos + len(kv)] = kv
        pos += len(kv)
        del kc, kv
        if (c + 1) % max(1, NCH // 10) == 0:
            log(f"  S2 concat {c+1}/{NCH} ({time.time()-t:.0f}s)")
    if pos != nnz_tot:
        fail(f"S2 concat position {pos:,} != nnz {nnz_tot:,}")
    out_dat.flush()
    out_idx.flush()
    for c in range(NCH):
        for suf in (".cols.i4", ".vals.f8", ".cnt.i8", ".diag.f8"):
            try:
                os.remove(os.path.join(SCRATCH, f"chunk_{c:04d}{suf}"))
            except OSError:
                pass
    log(f"S2 assembled {nnz_tot:,} nnz ({time.time()-t:.0f}s); "
        "chunk files removed")
    sentinel("S2_DONE", True)
else:
    log("S2 already done")

# ---------------- S3: memmap + warm + RQ gates ----------------
indptr = np.fromfile(F_IPT, np.int64)
nnz_tot = int(indptr[-1])
diag = np.fromfile(F_DIA, np.float64)
mm_d = np.memmap(F_DAT, np.float64, "r", shape=(nnz_tot,))
mm_i = np.memmap(F_IDX, np.int32, "r", shape=(nnz_tot,))
ybuf = np.empty((NT, D))


def mv(x):
    y = np.empty(D)
    tri_symm_mv(indptr, mm_i, mm_d, diag, x, ybuf)
    reduce_cols(ybuf, y)
    return y


log("=== S3 warm + Rayleigh-quotient gates ===")
if N == 7500:
    wz = np.load(WARM, allow_pickle=True)
    vecs = np.asarray(wz["vecs"], np.float64)[:NROOTS]     # (3, D) a-major
    e_warm = np.asarray(wz["e_roots"], np.float64)[:NROOTS]
else:  # smoke: warm from the bank's ground amplitudes only
    bank = np.load(f"flagsolve_n{N}.npz", allow_pickle=True)
    vecs = np.asarray(bank["amps"], np.float64).reshape(1, D)
    e_warm = np.array([float(bank["e"])])
    E_CERT = np.array([float(bank["e"])])
    S2_CERT = np.array([float(bank["s2"])])
t = time.time()
rqs = []
for k in range(vecs.shape[0]):
    v = vecs[k] / np.linalg.norm(vecs[k])
    rq = float(v @ mv(v))
    rqs.append(rq)
    dg = abs(rq - e_warm[k])
    log(f"  warm root{k}: RQ {rq:.9f} vs cert {e_warm[k]:.9f} "
        f"|d|={dg:.2e} ({time.time()-t:.0f}s)")
    if dg > 2e-5:
        fail(f"G3 RQ gate root{k}: |d|={dg:.2e} > 2e-5")
log("G3 PASS warm Rayleigh quotients")

# ---------------- S4: block Davidson ----------------
rng = np.random.default_rng(17)


def davidson(nroots, x0block, tol, maxiter=200, maxm=None):
    nb = max(nroots + 3, x0block.shape[0] + 3)
    # restarts (m -> maxm/2) cost convergence rate near the 0.285 mHa
    # root0/root1 pair; MAXM=96 on the box trades ~86 GB RAM for fewer
    # restarts (512 GB available)
    maxm = maxm or int(os.environ.get("MAXM", str(8 * nb)))
    order = np.argsort(diag)[:nb]
    V = np.zeros((D, nb))
    V[order, np.arange(nb)] = 1.0
    V += 1e-6 * rng.standard_normal((D, nb))
    for j in range(x0block.shape[0]):
        V[:, j] = x0block[j]
        if JITTER > 0:
            V[:, j] += JITTER * rng.standard_normal(D)
    V, _ = np.linalg.qr(V)
    HV = np.column_stack([mv(V[:, j]) for j in range(V.shape[1])])
    it = 0
    while True:
        G = V.T @ HV
        G = 0.5 * (G + G.T)
        theta, S = np.linalg.eigh(G)
        theta, S = theta[:nroots], S[:, :nroots]
        X = V @ S
        HX = HV @ S
        R = HX - X * theta
        rn = np.linalg.norm(R, axis=0)
        log(f"    dav it {it:3d} m={V.shape[1]:3d} "
            f"E={np.array2string(theta, precision=9)} res_max={rn.max():.2e}")
        np.savez(os.path.join(SCRATCH, "tb7500_ck.npz"), it=it, theta=theta,
                 rn=rn, X=X.astype(np.float64))
        if rn.max() < tol or it >= maxiter:
            return theta, X, rn, it
        if time.time() - T0 > BUDGET:
            log("S4 wall budget hit -- returning current Ritz block")
            return theta, X, rn, it
        newd = []
        for j in range(nroots):
            if rn[j] < tol:
                continue
            den = diag - theta[j]
            den[np.abs(den) < 1e-8] = 1e-8
            tvec = R[:, j] / den
            tvec -= V @ (V.T @ tvec)
            nv = np.linalg.norm(tvec)
            if nv > 1e-10:
                newd.append(tvec / nv)
        if not newd:
            return theta, X, rn, it
        Tm = np.column_stack(newd)
        Tm -= V @ (V.T @ Tm)
        Tm, _ = np.linalg.qr(Tm)
        V = np.column_stack([V, Tm])
        HV = np.column_stack([HV] + [mv(Tm[:, j])
                                     for j in range(Tm.shape[1])])
        if V.shape[1] > maxm:
            keep = maxm // 2
            G = V.T @ HV
            G = 0.5 * (G + G.T)
            th, S = np.linalg.eigh(G)
            V = V @ S[:, :keep]
            HV = HV @ S[:, :keep]
            V, rr = np.linalg.qr(V)
            HV = HV @ np.linalg.inv(rr)
        it += 1


log(f"=== S4 block Davidson: nroots={NROOTS if N==7500 else 1} tol={TOL} ===")
t = time.time()
theta, X, rn, its = davidson(vecs.shape[0], vecs, TOL)
log(f"S4 done: {its} iters, {time.time()-t:.0f}s, "
    f"E={np.array2string(theta, precision=9)} res={rn}")

# ---------------- S5: S^2 (pyscf full-sector embed) + gates -------------
from pyscf import fci  # noqa: E402

aa = fci.cistring.strs2addr(NC, NA, A)
ab = fci.cistring.strs2addr(NC, NB, B)
dima = int(fci.cistring.num_strings(NC, NA))
dimb = int(fci.cistring.num_strings(NC, NB))
ridx = np.repeat(aa, N)
cidx = np.tile(ab, N)
s2s = []
for k in range(theta.shape[0]):
    civ = np.zeros((dima, dimb))
    civ[ridx, cidx] = X[:, k]
    ss, _ = fci.spin_op.spin_square(civ, NC, (NA, NB))
    del civ
    s2s.append(float(ss))
    log(f"  root{k}: E={theta[k]:.9f} res={rn[k]:.2e} S2={ss:.6f}")

okE = all(abs(theta[k] - E_CERT[k]) < 5e-6 for k in range(theta.shape[0]))
okR = bool(rn.max() < 1e-7)
okS = all(abs(s2s[k] - S2_CERT[k]) < 5e-3 for k in range(theta.shape[0]))
for k in range(theta.shape[0]):
    log(f"  G4/G5 root{k}: |dE|={abs(theta[k]-E_CERT[k]):.2e} "
        f"|dS2|={abs(s2s[k]-S2_CERT[k]):.2e}")
res_name = f"tb7500_result_n{N}.npz"
np.savez_compressed(res_name, n=N, e=theta, resid=rn, s2=np.array(s2s),
                    e_cert=E_CERT, s2_cert=S2_CERT, rq_warm=np.array(rqs),
                    nnz_tri=nnz_tot, iters=its, wall_s=time.time() - T0,
                    gates_ok=bool(okE and okR and okS))
log(f"result -> {res_name}")
if okE and okR and okS:
    log("TB7500_DONE -- ALL GATES PASS (independent kernel confirms the "
        "certified flagship manifold)")
    sys.exit(0)
log(f"TB7500_PARTIAL -- gates E:{okE} res:{okR} S2:{okS}")
sys.exit(2)
