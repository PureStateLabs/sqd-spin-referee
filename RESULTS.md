# Independent results

This file exists so that a number you measure yourself has somewhere to live.

The claims in this repository are only worth what independent reproduction makes
them worth. If you ran anything here — the Colab verifier, one of the ten-minute
scripts, the full `_paperaudit.py`, or your own audit of your own states — please
add a row. **Disagreements are as welcome as confirmations**, and a null result
("I ran it, I got something else") is more useful to us than silence.

**How to add a row:** open an issue with the
[reproduction report template](.github/ISSUE_TEMPLATE/reproduction_report.md),
or send a pull request editing the table below. Either is fine.

If you believe you have overturned the result, [FALSIFY.md](FALSIFY.md) states
the three escape conditions precisely; that is the Singlet Challenge, and it is
meant to be winnable.

---

## Confirmations and discrepancies

| Date | Who | What was run | Observed | Matches expected? |
|---|---|---|---|---|
| — | — | *No independent reports yet.* | — | — |

<!-- Template row — copy, fill, delete this comment:
| 2026-09-04 | A. Researcher (Institution) | `_gramxcheck.py`, Level 1 | max deviation 2.5e-14 | yes |
-->

---

## What is most useful to report

Roughly in order of value to the field, not to us:

1. **A ⟨S²⟩ number from states you generated yourself.** This is the single most
   valuable thing anyone can contribute. If you run SQD/QSCI on any system with a
   spin-degenerate low-energy manifold, measure ⟨S²⟩ on the state you report and
   publish it beside the energy. It costs one sparse matrix–vector product. You do
   not need this repository to do it, and you do not need to cite us.
2. **A Singlet-Challenge counterexample** meeting any of the three conditions in
   [FALSIFY.md](FALSIFY.md) — most sharply, a spin-identified quantum-sampled
   subspace that beats HCI/CIPSI at matched determinant count.
3. **A reproduction that disagrees with [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md)**,
   at any level. Include your environment; version and platform differences are
   exactly what we want to find out about.
4. **A confirmation.** Less interesting than the above, still worth recording.

## What to include

The [issue template](.github/ISSUE_TEMPLATE/reproduction_report.md) asks for all
of this, but briefly: the exact command, your package versions and OS, your
`sha256sum -c SHA256SUMS` result, and the observed numbers beside the expected
ones. For anything involving a state, include the ⟨S²⟩ *and* its variance — a
scalar mean does not identify a state, which is most of the point of this work
(§2.11 of the paper: across eight audited states, exactly one was a spin
eigenstate, and it was in the wrong sector).

## A note on conditioning

If you reproduce one of the near-degenerate manifolds and get a ⟨S²⟩ that differs
from ours in the second decimal, that may not be a discrepancy. Section 2.10
measures this directly: at N = 2000, three converged runs of two independent
implementations span 22 µHa in energy but **0.126 in ⟨S²⟩**, with floating-point
summation order the only difference between them. The energy is a well-conditioned
output of this protocol; the spin identity is not. Report the residual you
converged to alongside the value, and we can compare like with like.
