# VALIDATE — three levels of verification

The paper's claims are meant to be checkable without trusting the authors or any
single implementation. Pick the level that matches how far you want to go; levels
1 and 2 are self-contained, so you do not need to read the paper to run them.

Set up once: `./setup.sh --min && . .venv/bin/activate` (Level 3's deeper
harnesses want the full `./setup.sh`). Expected numbers for every run below are
pinned in [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md).

---

## Level 1 — Five-minute reproduction

Run one script and confirm four things.

```bash
python reproduce_level1.py
```

Confirm:

1. **Energy before and after spin completion** — identical to < 1 nHa
   (−116.56010715 Ha both ways). Completion does not change the energy.
2. **⟨S²⟩ of the returned state** — 4.83 before and after: a high-spin mixture,
   not the singlet target (⟨S²⟩ = 0). Completion does not move it (Δ ~ 1e-11).
3. **Completed subspace size** — the determinant count grows ~4× (49,042 →
   196,249) while (1) and (2) hold. Symmetry is permitted; a singlet is not
   produced.
4. **Package versions** — printed at the top, so your stack is on record.

That is the whole thesis in one run: an energy does not tell you which spin
state it belongs to, and the shipped mitigation is a no-op on realistic input.

---

## Level 2 — Independent observable check

Do not take our ⟨S²⟩ operator on faith. Confirm the spin observable a different
way.

**(a) The moment / LP machinery, from scratch (analytic, no repo data):**

```bash
python higher_moment_example.py
```

Reproduces the identity ⟨S^2k⟩ = Σ_S w_S · [S(S+1)]^k and the sector linear
program on two hand-built states. Two states with identical ⟨S²⟩ = 2.00 are
separated by ⟨S⁶⟩ (8 vs 72), and the LP pins the mixture's triplet weight to 0.
If this LP is sound, the paper's field-identification machinery is sound, and
it is ~40 lines of numpy/scipy you can read in full.

**(b) An independent ⟨S²⟩ on a returned state:**

The total-spin operator is standard, S² = S_z² + ½(S₊S₋ + S₋S₊), so an
independent expectation is straightforward to write. Compute ⟨S²⟩ on a returned
state with spin code that is **not** our instrument (your own determinant
expectation, or a package's spin routine) and confirm it lands on the same
high-spin value. Our own cross-instrument certification runs this across eight
independent arms:

```bash
python _gramxcheck.py     # 8 arms, agreement gate < 1e-6 (observed ≤ 2.5e-14)
```

**(c) On IBM's own data and solver** (needs the IBM archive — see README):

```bash
python spin_inversion_is_not_singlet.py    # ~35 s
```

IBM's own `solve_fermion` on IBM's own samples agrees with our instrument to
1×10⁻⁴, and the returned ground state is ⟨S²⟩ = 3.50, not a singlet. That is a
genuinely independent implementation (theirs) confirming ours.

If any independent spin operator disagrees with ours by more than the stated
gate, that is a finding — report it (see [FALSIFY.md](FALSIFY.md)).

---

## Level 3 — Mathematical / code review

The instrument is small and meant to be read. Inspect:

1. **Ladder-operator signs & the S² construction.** Check that the total-spin
   operator (S² = S_z² + ½(S₊S₋ + S₋S₊)) is assembled with correct S₊/S₋
   determinant phases and S_z eigenvalues — in the audit instrument
   (`_s2audit.py`) and the as-shipped test (`_sqdship.py`).
2. **Determinant conventions.** Occupation/ordering and the α/β spin-orbital
   mapping used when the operators act on the basis; confirm they match the
   FCIDUMP convention of the consumed Hamiltonians.
3. **Moment identities.** ⟨S⁴⟩/⟨S⁶⟩/⟨S⁸⟩ via the sector-resolved kernel;
   `_s4gate.py` and `_s6stress.py` gauge it against analytic S = 1…5 references
   in seconds. `higher_moment_example.py` is the minimal readable version.
4. **LP sector bounds.** The full-simplex linear program over spin sectors
   S = 0…S_max constrained by the measured moments (the bound-the-triplet-weight
   argument). Confirm the reported numbers are the min/max over the feasible
   set, not a point estimate.
5. **Residual-to-gap analysis.** The certified-eigenstate argument: an eigen-
   residual must be small relative to the spectral gap for a state to be a
   genuine eigenstate rather than a superposition across a gap. See the flagship
   certification (`_gram7500.py`, `gram7500*.npz`) and the degeneracy
   adjudication (`_degen*.py`). The one-member-per-degenerate-pair Davidson
   artifact is dense-adjudicated there.

[REPRO_MAP.md](REPRO_MAP.md) ties each headline claim to the artifact that
proves it and the command that regenerates it. `_paperaudit.py` re-derives all
**370** quantitative claims from the raw archives and exits nonzero on any
mismatch (a reduced clone without the 413 MB flagship companion skips 19
companion-gated checks and runs the remaining 351).

---

Found a discrepancy at any level? See [FALSIFY.md](FALSIFY.md) for how to report it.
