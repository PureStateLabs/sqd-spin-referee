# Which state are you converging?

**A spin audit of sample-based quantum diagonalization (SQD/QSCI) benchmarks on iron–sulfur clusters — supplementary code and data.**

This repository contains the complete evidence base for the paper: the paper itself ([md](paper_sqd_spin_audit.md) / [tex](paper_sqd_spin_audit.tex) / [pdf](paper_sqd_spin_audit.pdf)), the spin-audit instrument, every harness that produced a number in the paper, the result archives those numbers live in, and the machine-verification scripts that tie the two together. The computations were carried out by an AI research agent (Claude, Anthropic) under the author's direction; the agent's self-audit record — five self-caught defects, one self-retraction, two reviewer-prompted corrections, and one post-publication correction — is documented in [AUDIT_TRAIL.md](AUDIT_TRAIL.md).

**Headline:** on the public [2Fe-2S] and [4Fe-4S] benchmark systems, neither classically-selected nor quantum-sampled selected-CI subspaces converge the singlet target — the followed states are high-spin mixtures (⟨S²⟩ ≈ 4.7–5.0 on [2Fe-2S], *rising* to 7.0 on [4Fe-4S]) with errors the size of the entire Heisenberg exchange ladder, the shipped spin mitigation is measurably a no-op on realistic input, and IBM's own published archive shows its uniform-random control matching its hardware samples. State identity is load-bearing at the claimed accuracy scales, and it is currently unmeasured field-wide.

> **Did the quantum hardware beat random guessing? Check it in ten seconds, using only
> IBM's own published files — [open it in Colab](https://colab.research.google.com/github/PureStateLabs/sqd-spin-referee/blob/main/verify_hardware_vs_random.ipynb).**
> Nothing to download or install, and it reads nothing from this repository: it fetches six
> small files live from [IBM's own data archive](https://github.com/jrm874/sqd_data_repository)
> and re-derives the comparison. Their uniform-random null control reaches a lower mean energy
> than their quantum-hardware samples at **9 of 9** matched subspace dimensions (7 of 9 at
> p < 0.05, two-sided), by 4.8 to 26.7 mHa against a per-batch scatter of 3.7 to 16.3 mHa.
> Notebook: [`verify_hardware_vs_random.ipynb`](verify_hardware_vs_random.ipynb); method
> [`_ibmuniform2.py`](_ibmuniform2.py); paper §2.7.

> **Check the core result in your browser, in two seconds — [open the verifier in Colab](https://colab.research.google.com/github/PureStateLabs/sqd-spin-referee/blob/main/verify_sqd_spin.ipynb).** Nothing to download, nothing to install. It shows that the shipped spin mitigation quadruples the determinant count, moves the ground-state energy by under a nanohartree, and leaves the returned state at ⟨S²⟩ = 4.83 when the singlet target is 0. The two archives it reads are embedded in the notebook as base64 with their SHA-256 hashes, and those hashes are the ones for [`sqdship_symm0.npz`](sqdship_symm0.npz) and [`sqdship_symm1.npz`](sqdship_symm1.npz) in this repository, so you can confirm it is reading the real thing rather than a prepared demo.

> **New here?** [START_HERE.md](START_HERE.md) has one-command setup and three short reproducers that show the core result in minutes. [VALIDATE.md](VALIDATE.md) lays out three levels of verification, [FALSIFY.md](FALSIFY.md) is the Singlet Challenge, and [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md) pins the numbers to match.

> **Want the plain-English version first?** The writeup is at
> [purestatelabs.github.io/sqd-spin-referee](https://purestatelabs.github.io/sqd-spin-referee/).

> **Want the results without the machinery?** [SUMMARY.md](SUMMARY.md) is the
> short version (~3 pages): the tables and the verdicts, no derivations.

> **Ran it yourself?** Please add your number to [RESULTS.md](RESULTS.md), via the
> [reproduction report template](.github/ISSUE_TEMPLATE/reproduction_report.md) or a
> pull request. Disagreements are as welcome as confirmations.

## Integrity

Every file in this repository is frozen in [`SHA256SUMS`](SHA256SUMS) — verify a clone with `sha256sum -c SHA256SUMS`. The Zenodo release artifacts (supplementary zip, the four companion archives, paper PDF) are hashed in the record's own `SHA256SUMS`.

## Verify in ten minutes (no external downloads)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python _gramxcheck.py      # cross-instrument S² certification, 8 arms, gate <1e-6 (observed ≤2.5e-14)
python _sqdship_delta.py   # the mitigation no-op, batch by batch (<1 nHa, 4.00x the determinants)
python _costmodel.py       # price-independence of the matched-cost verdict
python _s2fig.py           # rebuild the paper's figures + master CSV from the raw archives
```

These run entirely from files in this repository. The stored outputs they must reproduce are committed alongside them (`_gramxcheck.out`, `s2audit_master.csv`, the figures).

## Full claim verification

`_paperaudit.py` re-derives **every quantitative claim in the paper** (370 checks — including the cross-review language and overstatement guards, two-sided statistics, the package-version matrix, the high-moment ⟨S⁴⟩/⟨S⁶⟩/⟨S⁸⟩ sector-LP certification, the penalty-frontier section with both compressions of the flagship's stated squared form at its stated strength (§2.6), the independent-implementation flagship cross-check, the threshold scan, and a self-referential check-count gate) from the raw archives and exits nonzero on any mismatch. Its section J audits IBM's own published data files, so it additionally needs the IBM archive on disk (see below):

```bash
python _paperaudit.py      # version-of-record output: _paperaudit.out (370 checks, 0 failures)
```

The claim-by-claim map — every headline claim, the artifact that proves it, the command that regenerates it — is [REPRO_MAP.md](REPRO_MAP.md).

## External inputs (for full verification and regeneration)

Two public datasets are consumed as released and are **not** redistributed here. The harnesses expect them at these exact relative paths:

| Input | Fetch | Expected path |
|---|---|---|
| Li–Chan iron–sulfur active-space FCIDUMPs (Hamiltonians for §2.4–2.6) | `git clone https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters.git _qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-Clusters` | `_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-Clusters/Fe2S2_and_Fe4S4/{Fe2S2/fe2s2, Fe4S4/fe4s4}` |
| IBM's data archive for arXiv:2405.05068 (raw hardware shots, MO FCIDUMPs, circuit template + parameters, 58.6M-configuration release, uniform controls, HCI references, recovery ablation) | [DOI 10.5281/zenodo.15324153](https://doi.org/10.5281/zenodo.15324153) (also github.com/jrm874/sqd_data_repository) — extract the zip | `_ibm_data/sqd_data_repository-main/` |
| The flagship-dimension result archive `flagsolve_n7500_box2.npz` (413 MB — full 7500×7500 amplitude matrix at 5.625×10⁷; primary evidence for REPRO_MAP #18, too large for git) | This paper's Zenodo record: [10.5281/zenodo.21359922](https://doi.org/10.5281/zenodo.21359922) (companion file next to the supplementary zip) | repo root (enables audit section L4 and the warm-provenance checks of section N; without it `_paperaudit.py` runs the remaining 351 checks) |
| The three-root certification vector block `gram7500_restart.npz` (1.3 GB — converged flagship-manifold eigenvectors; REPRO_MAP #26, too large for git) | Same Zenodo record (second companion file) | repo root (optional: recompute the certified Gram matrix from the vectors, or warm-resume `_gram7500.py`; audit section N runs from the in-repo `gram7500*.npz` summaries without it) |
| The stage-4 four-root vector block `gram7500_s4_restart.npz` (1.8 GB — converged four-root flagship-manifold eigenvectors at residual ≤3.3×10⁻⁵; REPRO_MAP #28, too large for git) | Same Zenodo record (third companion file) | repo root (optional: recompute the S²/S⁴ moments and sector solves from the vectors, or warm-resume `_gram7500s4.py`; audit section P runs from the in-repo `gram7500_s4ck*.npz` checkpoints without it) |
| The independent-implementation cross-check block `tb7500_ck.npz` (1.35 GB — in-house block-Davidson three-root vectors over an independently built 1.6 TB CSR Hamiltonian at the flagship dimension; §2.10) | Same Zenodo record (fourth companion file) | repo root (optional: re-derive the independent-kernel energies/⟨S²⟩ from the vectors; the in-repo `tb7500_result_n7500.npz` summary carries the reported numbers without it) |

What needs what:

- **Repo alone:** `_gramxcheck.py`, `_sqdship_delta.py`, `_costmodel.py`, `_couponlaw.py`, `_figcoupon.py`, `_s2fig.py`, `_fairfight.py`, `_hcicert.py`, `_n2xval.py` (builds its own N₂ integrals via PySCF), `_sqdraw_synth.py` (prints the §2.8 ladder matrix from the shipped `sqdraw_*.npz`), and the partA gauge gate of `_s2audit.py` (the diazene targets `dz_*.npz` ship in this repo; `_diazene_targets.py` regenerates the full set).
- **+ Li–Chan dumps:** the iron–sulfur audit ladders (`_s2audit.py` part B) and the as-shipped soup-sample test (`_sqdship.py`).
- **+ IBM archive (fastest first look, ~35 s):** `python spin_inversion_is_not_singlet.py` — IBM's own construction is spin-inversion symmetric by construction, its ground state is still ⟨S²⟩ = 3.50, and their own `solve_fermion` agrees with our instrument to 1×10⁻⁴.
- **+ IBM archive:** `_paperaudit.py` section J, the two-stream uniform-control statistics `_ibmuniform2.py`, and the archive-audit harnesses `_ibmsamples.py`, `_ibmsamples4.py`, `_ibmuniform.py`, `_lucjfe2.py`, `_lucjopt.py`, `_sqdraw.py` (the §2.8 published-construction ladder; N=2000 costs ~35 h on 32 cores).
- **+ flagship companion (`flagsolve_n7500_box2.npz`):** `_gram7500.py`, the three-root certification re-run (warm-start gates, tolerance ladder, per-stage S²-Gram; ~11 h / 42 GB RSS on 64 cores).
- **+ certification companions (`gram7500_restart.npz`, `gram7500.npz` in-repo):** `_gram7500s4.py`, the stage-4 re-run (residual push, fourth root, S⁴ moments/sector solves; ~14.8 h / 44 GB RSS on 64 cores; `_s4gate.py` gauges the moment kernel standalone in seconds).

One primary-result archive is distributed via the paper's Zenodo record instead of git (GitHub's 100 MB file limit): `flagsolve_n7500_box2.npz`, above. Four large regenerable intermediates are deliberately not committed (and are `.gitignore`d): `ibmraw_counts.npz` (413 MB), `ibm4raw_counts.npz` (936 MB), `lucjopt_counts.npz` (1.4 GB), `sqdraw_bits.npz` (12 MB packed BitArray cache). Each is rebuilt deterministically by the stage-1 block of its harness from the IBM archive.

## Environment

Python 3.11 on Linux (or WSL — the campaign ran under WSL2). Exact versions in [requirements.txt](requirements.txt); the load-bearing pins are `pyscf 2.13.1`, `qc-pyci 1.0.3` (imports as `pyci` — do **not** `pip install pyci`, that is an unrelated package), `qiskit-addon-sqd 0.12.1` (audited as shipped; `_verm.py` re-runs the semantics at 0.3.0, the earliest publicly auditable release — the flagship-stated 0.1.0 is retrievable from neither PyPI nor GitHub), `ffsim 0.0.80`, `numpy 2.3.0`. Regenerating the deep [2Fe-2S] ladders needs ~10+ GB RAM; verification scripts are light. Rebuilding the PDF needs a LaTeX toolchain (pdflatex / TeX Live; optional).

## Layout note

The layout is deliberately **flat**: every harness loads its inputs by bare relative path (`np.load("sqdship_vec.npz")`, `_ibm_data/...`) and is run from the repository root. Do not reorganize files into subdirectories. The two subdirectories that exist are load-bearing: `n2_r200/` (stretched-N₂ leg) and `lucj_run1_reference_choice_observation/` (the deliberately retained defective run — self-audit event #5 in AUDIT_TRAIL.md; clearly labeled, do not mistake it for a production leg).

## The Singlet Challenge

The paper's claims are falsifiable. They fail if anyone can exhibit:

1. a quasi-degenerate ground manifold (ΔE < 2 µHa) of a spin-completed subspace of these benchmark systems whose S²-Gram floor is < 1 (target 0), using the Gram instrument provided here — rotations across gaps far larger than the degeneracy window are superpositions, not eigenstates, and do not qualify;
2. a quantum-sampled subspace at matched determinant count whose *spin-identified* ground-state energy beats HCI/CIPSI on any system here; or
3. a run of the shipped pipeline, on any sample source in IBM's own archive, that escapes the spin–energy vise — ⟨S²⟩ < 1 *and* error < 50 mHa on [2Fe-2S] — without explicit spin control.

The harnesses in this repository are the arena; [FALSIFY.md](FALSIFY.md) states the challenge in full and explains how to report a counterexample. Issues and counterexamples welcome.

## License and citation

Code is released under the [MIT License](LICENSE). The paper text, figures, and result data (`*.npz`, `*.csv`, `*.png`, the PDF) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse freely with attribution. Citation metadata is in [CITATION.cff](CITATION.cff); the DOI-archived supplementary package (zip + flagship companion archive) is at [10.5281/zenodo.21359922](https://doi.org/10.5281/zenodo.21359922).

A full provenance inventory of every file — which harness wrote it, which paper claim it backs — is in [FILE_MANIFEST.md](FILE_MANIFEST.md).
