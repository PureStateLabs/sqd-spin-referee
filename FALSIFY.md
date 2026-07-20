# FALSIFY — the Singlet Challenge

The paper makes one falsifiable claim:

> On these public iron–sulfur SQD/QSCI benchmarks, energy-based convergence does
> not establish state identity. Every directly audited competitive-energy
> realization fails to return the named singlet, and a post-hoc spin penalty
> cannot repair the audited subspaces, because their competitive singlet
> representation is absent — not merely unselected by the solver.

It fails — and we retract it — if anyone exhibits any **one** of the following,
using the instruments in this repository.

## The three escape conditions

1. **A competitive spin-pure singlet in an audited subspace.**
   A quasi-degenerate ground manifold (ΔE < 2 µHa) of a spin-completed subspace
   of these benchmark systems whose S²-Gram floor is < 1 (target 0), using the
   Gram instrument provided here.
   *Does not count:* a low-⟨S²⟩ state formed by rotating across a gap far larger
   than the degeneracy window — that is a superposition, not an eigenstate. This
   is the exact trap the residual-to-gap analysis exists to catch.

2. **A quantum-sampled subspace that wins on a spin-identified energy.**
   A quantum-sampled subspace at matched determinant count whose
   *spin-identified* ground-state energy beats HCI/CIPSI on any system here. The
   match must be at equal determinant count, and the winning state's spin must
   be **measured**, not inferred from its energy.

3. **A run that escapes the spin–energy vise.**
   A run of the shipped pipeline, on any sample source in IBM's own archive, that
   reaches ⟨S²⟩ < 1 **and** error < 50 mHa on [2Fe-2S] simultaneously, without
   explicit spin control (no spin-adapted CSFs, no S²-projected selection, no
   penalty tuned to the answer).

## Why a spin penalty does not already settle it

A solver-side penalty λ·PS²P is a fair thing to try, and we tried it (the
penalty frontier, §2.6). It does not rescue the audited subspaces, for a
structural reason: adding λ·PS²P to the projected Hamiltonian PHP leaves the
commutator [PHP, PS²P] unchanged, so the penalty only biases the variational
minimum toward whatever low-spin directions the selected space already
represents — it cannot manufacture a competitive singlet that selection never
placed in the subspace. Only an S²-invariant construction (spin-adapted CSFs, or
S²-projected selection) removes the noncommutation. So escape condition 3 asks
for spin-aware *selection*, not a penalty knob.

## The arena

The harnesses in this repository are the arena — the Gram instrument, the
moment/LP machinery, the shipped-pipeline drivers, and IBM's own archive.
[VALIDATE.md](VALIDATE.md) shows how to drive them; [REPRO_MAP.md](REPRO_MAP.md)
says where each lives and which claim it backs.

## Reporting a counterexample (or any discrepancy)

Once this repository is on GitHub, open an issue using the **Reproduction report**
template (`.github/ISSUE_TEMPLATE/reproduction_report.md`), or start a
Discussion. Include:

- the system ([2Fe-2S] / [4Fe-4S] / other) and the exact command run;
- your package versions (`python reproduce_level1.py` prints them);
- the determinant count and the **spin-identified** ground-state energy;
- the S² / S²-Gram readout for the state you are claiming.

A counterexample that meets any of the three conditions overturns the paper, and
we will say so in the record.
