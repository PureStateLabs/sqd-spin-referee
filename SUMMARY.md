# Which state are you converging? The short version

*The results without the machinery. The full paper is `paper_sqd_spin_audit.pdf`, where every
number below is reproduced with its harness and its raw data.*

---

## Background

IBM published SQD/QSCI results on the [2Fe-2S] and [4Fe-4S] iron-sulfur clusters (Sci. Adv. 2025)
as utility-scale evidence for quantum chemistry; a critique (arXiv:2501.07231) argues the quantum
samples never beat classical selected-CI at matched subspace dimension. Both sides report
energies. Neither reports which electronic state the energy belongs to.

On these particular systems that omission is expensive. The low-lying spectrum is a Heisenberg
exchange ladder: two high-spin ferric centres at S = 5/2 apiece, coupling to a total S of anywhere
from 0 to 5, so S² takes the values 0, 2, 6, 12, 20, 30. The whole ladder is only about 15 to
30 mHa wide, the same size as the accuracy differences under dispute, so landing on the wrong rung
costs you roughly the size of the effect being argued about with nothing in the output to say so.
⟨S²⟩ is the standard way to check, and it should come out at 0 for the singlet everyone is
targeting. So I measured it.

## 1. [2Fe-2S]: where the growth ladders land

Public Li-Chan active space (30e, 20o); M_s = 0 sector dimension 2.4×10⁸; near-exact singlet
reference E = −116.6056091 Ha. Spaces grown by HCI ladders from the two seeds the controversy
itself uses, a broken-symmetry two-determinant seed (the "good guess") and a closed-shell aufbau
seed (the "bad guess").

| Seed | D | E₀ (Ha) | err vs exact singlet (mHa) | ⟨S²⟩ root 0 | ⟨S²⟩ root 1 |
|---|---|---|---|---|---|
| BS | 1,011 | −116.456202 | +149.4 | 4.978 | 4.977 |
| BS | 25,734 | −116.569067 | +36.5 | 4.893 | 4.921 |
| BS | 65,124 | −116.584165 | +21.4 | 4.779 | 4.863 |
| BS | 155,030 | −116.592307 | **+13.3** | **4.659** | 4.809 |
| aufbau | 1,003 | −116.060455 | +545.2 | 3.033 | 3.015 |
| aufbau | 5,053 | −116.155646 | +450.0 | 2.050 | 2.073 |
| aufbau | 24,570 | −116.430693 | +174.9 | 4.672 | 4.461 |
| aufbau | 68,825 | −116.579203 | +26.4 | 4.824 | 4.876 |
| aufbau | 170,448 | −116.591986 | **+13.6** | **4.669** | 4.810 |

⟨S²⟩ ≈ 4.7 falls between the S = 1 rung (S² = 2) and the S = 2 rung (S² = 6), so what is being
followed is a mixture rather than any clean spin state, and root 1 is no better anywhere.

The two seeds are worth dwelling on, because the controversy frames convergence quality as a
function of the initial guess. By D ≈ 1.6×10⁵ the arms agree to 0.3 mHa in energy and 0.01 in
⟨S²⟩, so at scale the seed stops mattering and selection dynamics decide what gets converged.
⟨S²⟩ does come down, 0.085 to 0.133 per doubling of D, but with no sign of acceleration; at that
rate ⟨S²⟩ < 1 needs some 28 more doublings, five orders past the whole 2.4×10⁸ sector.

Both seeds also stall about +13 mHa above the exact singlet, against that 15 to 30 mHa ladder. A
state you cannot identify, carrying an error the width of the ladder, will not resolve the ladder.
The mechanism is not new: high-spin components need fewer determinants, so incremental CI
selection over-rewards them, and that bias is textbook. What I have not found measured anywhere is
its size and persistence out to 1.7×10⁵ determinants on these systems.

## 2. [4Fe-4S]: it gets worse with size

Public active space (54e, 36o); sector dimension ~8.8×10¹⁵; approximate DMRG reference
−327.2396369 Ha.

| D | E₀ (Ha) | err (mHa) | ⟨S²⟩ root 0 | ⟨S²⟩ root 1 |
|---|---|---|---|---|
| 1,106 | −326.184317 | +1055.3 | 5.293 | 5.270 |
| 5,905 | −326.385018 | +854.6 | 6.053 | 5.914 |
| 17,993 | −326.445889 | +793.7 | 6.464 | 6.302 |
| 40,435 | −326.484822 | +754.8 | **7.004** | 9.284 |

From a closed-shell seed at S² = 0, ⟨S²⟩ climbs with subspace size instead of falling, reaching
7.0 by 4×10⁴ determinants, past pure S = 2. Enlarging the calculation walks it away from the
target. This is a probe regime energetically, so the direction is the point rather than the
endpoint; flagship SQD runs on this system at sector fractions around 10⁻⁸.

## 3. What the shipped spin mitigation actually does

IBM ships spin-inversion completion (`symmetrize_spin`) for exactly this problem. Run as shipped
on a million samples from a converged benchmark state, using their driver, their recovery loop and
their own `spin_square()`:

| | best energy (Ha) | ⟨S²⟩ | determinants |
|---|---|---|---|
| completion off | −116.56010715 | 4.83432 | 49,042 |
| completion on | −116.56010715 | 4.83432 | 196,249 |
| delta | −0.001 nHa | 2.6×10⁻¹¹ | **4.00×** |

The reason it does nothing here turns out to be structural rather than a tuning failure. Under
completion the ground state becomes an exactly degenerate pair {v, v̄}, and the 2×2 S² block
across that pair comes out proportional to the identity, eigenvalues [x, x] at the original
mixture value. Since every rotation inside the mitigated ground manifold carries that same mixed
⟨S²⟩, no choice of vector out of it gives you a singlet.

The solver underneath does take a `spin_sq` argument, wired to PySCF's `fix_spin_`, which targets
a spin sector directly. The top-level driver has no such parameter, and with no custom solver
supplied it calls the batch solver positionally, so `spin_sq` stays `None` on every default run.
You can reach it by passing your own solver, and I have. But a benchmark run through the default
path and reported without a spin measurement cannot have been targeting the singlet, since nothing
along that path ever asks. A Qiskit maintainer has since agreed the docstring should say the flag
"does not impose or improve the total spin of the returned state" (qiskit-addon-sqd #337, #338).

If you only check one thing here, check that table. It runs in a browser with nothing to install:
[the Colab verifier](https://colab.research.google.com/github/PureStateLabs/sqd-spin-referee/blob/main/verify_sqd_spin.ipynb).

## 4. The flagship dimension

Reconstructing the flagship operating point (5.625×10⁷ determinants) from IBM's released
half-configuration support, then solving four roots to convergence:

| root | ΔE from ground (mHa) | ⟨S²⟩ | Var(S²) | sector support |
|---|---|---|---|---|
| 0 | 0 | **1.3711** | 6.92 | infeasible for S ≤ 2, requires S ≥ 3 |
| 1 | +0.285 | 2.363 | 3.54 | admits S ≤ 2 |
| 2 | +6.52 | 4.738 | 7.41 | requires S ≥ 3 |
| 3 | +12.54 | 11.590 | 4.41 | predominantly S = 3 |

This construction lands 170.3 mHa below IBM's published energy at that dimension while sitting
+13.6 mHa above the exact singlet reference, so the difficulty is not that the energy came out bad.

Spin identity turns out to be far more convergence-sensitive than energy, which is why I did not
stop at the first answer. At residual 3.5×10⁻⁴ the energy was stable to under 0.1 µHa per
iteration across the closing fifty iterations while ⟨S²⟩ was still drifting. Tightening to 6×10⁻⁵
moved the energy by 1.9 µHa and moved ⟨S²⟩ from 1.287 to 1.371; a further push to 3.2×10⁻⁶,
roughly 1% of the 0.285 mHa ground gap, left it at 1.3711 unchanged in the fourth decimal.

The variances are probably the more informative column. An eigenstate of S² gives Var(S²) = 0 and
none of these four is close, so the ⟨S²⟩ figures are means over broad mixtures, not labels for
slightly perturbed eigenstates. Feeding ⟨S²⟩ and ⟨S⁴⟩ for the ground root into a two-moment sector
solve returns weights (0.82, −0.07, 0.25) on S ∈ {0, 1, 2}, which is infeasible, so there is
provably S ≥ 3 character in it.

Near-singlet character does exist inside the three-root span, whose invariant S² spectrum is
{0.038, 2.357, 6.066}, but that direction is a superposition of two eigenstates sitting 1.44 mHa
above the variational minimum, and it is not stationary. Determinant selection does not commute
with S² ([PHP, PS²P] ≠ 0), so the eigenbasis has no reason to align with spin sectors.

## 5. Scope, limitations, and how to falsify this

None of this means quantum computing is fake, or that anybody involved was dishonest. The
variational upper bound is granted throughout, so these energies are true statements about
energies, and spin contamination in selected-CI has been understood in principle for a long time.
What had not happened, as far as I can find, is anyone measuring it on these systems at these
scales, where it is big enough to swallow the result. Limitations: comparison dimensions are
fixed, the [4Fe-4S] reference is approximate, and IBM's large deduplicated sample file does not
preserve shot-level statistics, so part of their numerics is not auditable from public data.

The change I would like to see is small: report ⟨S²⟩ with its variance next to the energy, the way
you would report an error bar. A solver flag alone will not fix it, since the deficit is upstream
in the subspace and turning the penalty on traces a spin-energy frontier whose singlet end sits
2.9 Ha above the reference. That points at spin-aware selection instead.

Three ways to show I am wrong, with harnesses for all of them in the archive:

1. Exhibit a state in a spin-completed ground manifold here with ⟨S²⟩ < 1, using the Gram
   instrument provided.
2. Produce a quantum-sampled subspace at matched determinant count whose spin-identified ground
   energy beats HCI or CIPSI.
3. Show any run of the shipped pipeline, on any sample source in IBM's archive, reaching
   ⟨S²⟩ < 1 and error < 50 mHa at once without an explicit spin penalty.

I will publish whatever comes back, including if it turns out I am the one who is wrong.

---

**Paper:** doi:10.26434/chemrxiv.15006382/v1 · **Data, code, audit trail:**
doi:10.5281/zenodo.21359922 · **Repository:** github.com/PureStateLabs/sqd-spin-referee

`_paperaudit.py` re-derives every quantitative claim in the full paper from the raw archives (370
checks, zero failures), and `REPRO_MAP.md` maps each claim to the command that regenerates it. The
computations were carried out by an AI research agent (Claude, Anthropic) under my direction; its
self-audit record, including five self-caught defects, one self-retraction, two reviewer-prompted
corrections and one post-publication correction, ships as `AUDIT_TRAIL.md`.

Tyler Vitale, Pure State Labs · founder@purestatelabs.com · ORCID 0009-0003-6156-0212
