# Start here

This repository is the complete, runnable evidence base for the paper
**"Which state are you converging? A spin audit of sample-based quantum
diagonalization (SQD/QSCI) benchmarks on iron–sulfur clusters."**

The one-line claim: on the public [2Fe-2S] and [4Fe-4S] SQD/QSCI benchmarks,
energy-based convergence does **not** establish which spin state was returned.
Every directly audited competitive-energy realization returns a high-spin
mixture, not the singlet target, and the shipped spin mitigation is a measured
no-op. State identity is load-bearing at the claimed accuracy and is currently
unmeasured field-wide.

You do not need to read the 33-page paper to check the core of this. Three short
scripts reproduce it from data committed in this repository.

## 1. Set up (one command)

```bash
./setup.sh --min      # numpy + scipy + pyscf: enough for the three reproducers
# or ./setup.sh       # the full stack, to reproduce every number in the paper
```

Linux or WSL, Python 3.11. Override the interpreter with
`PYTHON=/path/to/python3.11 ./setup.sh`. Then `. .venv/bin/activate`.

## 2. Run the three reproducers (each runs in seconds)

| Script | What it shows | Reads |
|---|---|---|
| `python reproduce_level1.py` | The shipped "spin-inversion completion" quadruples the determinant count but moves the energy < 1 nHa and ⟨S²⟩ by ~1e-11 — it permits spin symmetry without producing a singlet (returned ⟨S²⟩ = 4.83, not 0). | `sqdship_symm{0,1}.npz` |
| `python higher_moment_example.py` | Why ⟨S²⟩ alone cannot identify a state: two states share ⟨S²⟩ = 2.00 but ⟨S⁶⟩ = 8 vs 72, and a sector linear program pins them apart. Self-contained (analytic). | — |
| `python sampling_wall_calculator.py` | The coupon-collector sampling wall: a concentrated eigenvector distribution cannot deliver its own determinant support at any realistic shot budget (γ < 1 everywhere). Add `--measured` for the paper's [2Fe-2S] vector. | (`--measured`: `sqdship_vec.npz`) |

The exact numbers to match — and the input-file hashes — are in
[EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md).

## 3. Go deeper

- **[VALIDATE.md](VALIDATE.md)** — three levels of verification, from a
  five-minute run, to an independent-observable check, to a mathematics/code
  review checklist. Start here to contribute a check.
- **[FALSIFY.md](FALSIFY.md)** — the Singlet Challenge: the exact, falsifiable
  conditions that would overturn the paper's claim, and the instruments
  provided to attempt them.
- **[README.md](README.md)** — the full repository: every harness, every
  archive, external-input fetch instructions, and the claim-by-claim
  reproduction map ([REPRO_MAP.md](REPRO_MAP.md)).
- The paper itself: [pdf](paper_sqd_spin_audit.pdf) /
  [md](paper_sqd_spin_audit.md) / [tex](paper_sqd_spin_audit.tex).

Every file is hash-frozen in [`SHA256SUMS`](SHA256SUMS) — verify a clone with
`sha256sum -c SHA256SUMS`. Reproduction reports and counterexamples are welcome;
see [FALSIFY.md](FALSIFY.md).
