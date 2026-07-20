# Expected outputs

Canonical outputs for the three quick reproducers, captured on the frozen
**v5.9.5** archives (2026-07-20). Your run should match the **numbers** below.

_(v5.9.5 = v5.9.4 with the AI-use disclosure rewritten as a single plain paragraph (the competing-interests/funding sentences moved to the submission form); v5.9.4 was a presentation pass — disclosure section retitled, four wide tables scaled to text width, Data-availability line-breaking fixed; no new results anywhere in this chain, the reproducer numbers below are unchanged. v5.9.3 added two review-prompted internal-consistency fixes: a projector-spectrum prose/table clarification and a flag on the unconverged fourth flagship root.)_
Cosmetic differences are expected and fine: package-version strings, float
formatting, and column widths depend on your environment. The demo/measured
sampling numbers are closed-form (no randomness) and analytic moments are exact,
so those match to the digit.

The reproducers read three committed data files. Confirm you have the same
inputs:

```bash
sha256sum sqdship_symm0.npz sqdship_symm1.npz sqdship_vec.npz
```

| File | SHA-256 |
|---|---|
| `sqdship_symm0.npz` | `3b4225b562db9e3381afbe9812ba7c375e74eb8ed4cfc6f07d6afdae044730e4` |
| `sqdship_symm1.npz` | `7cd9206b850e64b7cdac0b02e3113fbf67946ed7a08dcc01d02db88a7dbb8383` |
| `sqdship_vec.npz`   | `3c19ef5513d8cf188b810d5ec2f831e502e407731379590b6914cfc6a35ecfaa` |

All repository files are frozen in `SHA256SUMS`; `sha256sum -c SHA256SUMS`
verifies the whole clone at once.

---

## `python reproduce_level1.py`

The version block at the top reflects *your* stack; the table and verdict are
the load-bearing part:

```
                                best energy (Ha)       <S^2>    determinants
BEFORE completion (off)            -116.56010715     4.83432          49,042
AFTER  completion (on)             -116.56010715     4.83432         196,249
delta                                  -0.001 nHa     2.6e-11           4.00x

VERDICT:
  completion changed the energy by < 1 nHa .......... PASS
  completion left <S^2> essentially unchanged ....... PASS
  completion multiplied the determinant count ~4x ... PASS
  the returned state is a high-spin mixture, not 0 .. PASS  (<S^2> = 4.83)
```

The energy (−116.56010715 Ha), the determinant counts (49,042 → 196,249, 4.00×),
and ⟨S²⟩ = 4.83432 are the numbers to match; all four verdict lines must read
PASS.

## `python higher_moment_example.py`

Self-contained (no repo data) — this matches exactly:

```
spin sectors S           : [0, 1, 2, 3, 4, 5]
Casimir  S(S+1)          : [0, 2, 6, 12, 20, 30]

A: pure triplet          w = {S1: 1}
  <S^2> = 2.0000   <S^4> = 4.0000   <S^6> = 8.0000   <S^8> = 16.0000
  LP triplet-weight bound from (<S^2>,<S^4>,<S^6>): [1.00000, 1.00000]

B: 2:1 singlet:quintet   w = {S0: 2/3, S2: 1/3}
  <S^2> = 2.0000   <S^4> = 12.0000   <S^6> = 72.0000   <S^8> = 432.0000
  LP triplet-weight bound from (<S^2>,<S^4>,<S^6>): [0.00000, 0.00000]
```

Both states share ⟨S²⟩ = 2.0000; ⟨S⁶⟩ (8 vs 72) and the LP (triplet weight 1 vs
0) separate them.

## `python sampling_wall_calculator.py`

Demo — Zipf(s=1.3) over 170,448 determinants (a concentrated, eigenvector-like
distribution):

```
support (distinct determinants, p > 0): 170,448
  M =     1,000,000 shots -> E[U] =       34,500 distinct ( 20.2% of support) | gamma = 0.616 (10x shots -> 4.1x determinants)
  M =    10,000,000 shots -> E[U] =      114,281 distinct ( 67.0% of support) | gamma = 0.383 (10x shots -> 2.4x determinants)
  M =   100,000,000 shots -> E[U] =      170,068 distinct ( 99.8% of support) | gamma = 0.011 (10x shots -> 1.0x determinants)
```

`--measured` — the paper's [2Fe-2S] aufbau-HCI soup vector (`sqdship_vec.npz`):

```
support (distinct determinants, p > 0): 170,448
  M =     1,000,000 shots -> E[U] =       38,129 distinct ( 22.4% of support) | gamma = 0.442 (10x shots -> 2.8x determinants)
  M =    10,000,000 shots -> E[U] =       88,712 distinct ( 52.0% of support) | gamma = 0.277 (10x shots -> 1.9x determinants)
  M =   100,000,000 shots -> E[U] =      131,667 distinct ( 77.2% of support) | gamma = 0.082 (10x shots -> 1.2x determinants)

  paper cross-check: E[U] @ 1e6 shots = 38,129 (closed form 38,129; measured 38,118; 0.03%)
```

γ < 1 at every shot count — shots buy determinants sub-linearly, so sampling
cannot out-collect selection at matched determinant count.

---

*Captured under Python 3.11.15, numpy 2.3.0, scipy 1.17.1, pyscf 2.13.1,
pyci 0.6.1 (the module exposed by the `qc-pyci` 1.0.3 distribution). `qiskit_addon_sqd` prints its installed version but is not exercised
by these three scripts.*
