---
name: Reproduction report
about: Report a reproduction result, a discrepancy, or a Singlet-Challenge counterexample
title: "[repro] "
labels: [reproduction]
---

<!-- Thanks for checking this work. VALIDATE.md explains how to verify (three
     levels); FALSIFY.md is the Singlet Challenge. Fill in what applies and
     delete the rest. -->

## What did you run?

- **Level** (see VALIDATE.md): Level 1 / Level 2 / Level 3 / other
- **System:** [2Fe-2S] / [4Fe-4S] / N2 / other
- **Exact command(s):**

```
$ python ...
```

## Environment

<!-- `python reproduce_level1.py` prints all of these -->

- python / numpy / scipy / pyscf / pyci / qiskit-addon-sqd versions:
- OS (Linux / WSL / other):
- `sha256sum -c SHA256SUMS` result (or the SHA-256 of the input files you used):

## Result

- **Observed** (energies, ⟨S²⟩ / S²-Gram, determinant counts):
- **Expected** (from EXPECTED_OUTPUTS.md or a specific paper claim — cite it):
- **Match or discrepancy:**

## If this is a Singlet-Challenge counterexample (FALSIFY.md)

- **Which escape condition** (1 / 2 / 3):
- The **spin-identified** energy and the S² / S²-Gram readout of the state:
- Determinant count (and, for condition 2, the matched-count comparison):
- Link or attach the state/subspace so it can be re-audited with the Gram
  instrument in this repository.
