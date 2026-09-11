# Audit-pin changes in the v6.0 prose revision

v6.0 rewrites the prose of both masters for readability. No number, claim,
scope limit or citation changes. `_paperaudit.py` pins some claims by requiring
an exact phrase to appear in the manuscript, so where the prose around a pinned
claim was rewritten, the pin has to follow the new wording. Each such change is
logged here with the claim it guards, so a reviewer can confirm that each new
phrase still requires the same claim.

A pin that could be kept by setting the old phrase inside a plain sentence was
kept. Only phrases that were themselves the problem (verdict-style headings,
repeated "their own") were changed.

| # | Section | Old pinned phrase | New pinned phrase | Claim it guards |
|---|---|---|---|---|
| 1 | 2.6 | `The penalty, switched on` (md + tex, and a `_pdfcheck.py` probe) | `run with the penalty on` | The section runs the spin penalty on the released pipeline's subspaces |
| 2 | 2.6 | `the certification of the frontier is their own code's` (md + tex) | `the package's own kernel certifies the frontier` | `fix_spin_`, seeded with our minimum, converges to it at overlap 1.000000, so the frontier is confirmed by the package's own kernel |
| 3 | 2.6 | `The mechanism, measured per checkpoint.` (md + tex, and a `_pdfcheck.py` probe) | `The completion mechanism at each checkpoint.` | The per-checkpoint completion-mechanism table and its explanatory paragraph are present |
| 4 | 2.9 | `but this is *not* a triplet` (md) / `this is \emph{not} a triplet` (tex) | `but this is not a triplet` / `this is not a triplet` | The ⟨S²⟩ ≈ 2.00 set-draw arm is not a triplet. Only the emphasis italics were removed; the words are identical |
| 5 | 2.8 | `_pdfcheck.py` probe `vise` | `trade-off` | The PDF contains the spin–energy trade-off discussion. "Vise" was a metaphor for the trade-off and is replaced by the plain term |

`_pdfcheck.py` probe `kernel closed out` (the 2.10 heading) became
`Independent implementation at the flagship dimension`. It was a PDF probe
only, not an audit pin.

## One claim reworded for precision, not only for style

**2.6, leakage operator.** v5.9.7 said the non-vanishing of the leakage
operator PS²(I − P)S²P "*is* the noncommutation [PHP, PS²P] ≠ 0". That
overstates the link: the leakage is nonzero exactly when the subspace P is not
S²-invariant, and non-invariance is what makes [PHP, PS²P] ≠ 0 possible. The
new text says the leakage "is nonzero here for the same reason that
[PHP, PS²P] ≠ 0: the subspace is not S²-invariant." Both pinned phrases
(`leakage operator`, `[PHP, PS²P] ≠ 0`) are unchanged.

## Negative gates

The audit forbids the phrase "wrong state" (a reviewer-flagged overstatement).
A first draft of the Discussion reintroduced it ("escape from a wrong state");
the gate caught it and the sentence now reads "escape from selection
hysteresis". Hyphenated "wrong-state seed/bias", present since v5.4, is not the
forbidden phrase.

## JCTC generator anchors

The JCTC generators (`_jctcsplit.py`, `_jctcsplit_tex.py`) find spans in the
masters by the text they start and end with. These anchors changed with the
rewrite and have been updated in both generators. Every splice was checked to
cover the same paragraphs as before: the paragraph before and the paragraph
after each span are the same in the v5.9.7 and v6.0 masters.

| Generator entry | Old anchor | New anchor |
|---|---|---|
| s26_a start (md + tex) | `Flagship SQD work addresses spin using configuration recovery` | `Flagship SQD work addresses spin through configuration recovery` |
| s26_a end (md) | `**The penalty, switched on.**` | `**The spin penalty.**` |
| s26_a end (tex) | `\textbf{The penalty, switched on.}` | `\textbf{The spin penalty.}` |
| s26_b start (md + tex) | `The trade-off is a frontier with no competitive interior.` | `The resulting trade-off has no competitive middle ground.` |
| s26_b end (md + tex) | `Two corollaries complete the rebuttal.` | `The released penalty path does not reach this curve from a cold start.` |
| mech carry (tex) | `\textbf{The mechanism, measured per checkpoint.}` | `\textbf{The completion mechanism at each checkpoint.}` |
| as-shipped carry (tex) | `\textbf{As-shipped audit.}` | `\textbf{The released pipeline.}` |
| s28 start (md + tex) | `This section closes the two limitations a rebuttal would otherwise` | `The results above used our samples and our subspace construction.` |
| s28 end (md) | `**The spin–energy vise.** The first three input classes` | `**The spin–energy trade-off.** The first three input classes` |
| s28 end (tex) | `\textbf{The spin--energy vise.}` | `\textbf{The spin--energy trade-off.}` |
| s29 start (md) | `Three observations. **First, the reproduction is generous.**` | `Our reproduction is, if anything, favorable to the published results.` |
| s29 start (tex) | `Three observations. \textbf{First, the reproduction is generous.}` | `Our reproduction is, if anything, favorable to the published results.` |
| s29 end (md + tex) | `**Even their curated configuration set converges to a spin mixture` | `The mitigation again acts on this input class` (the curated-set paragraph was split in two; the span must end at the second half or that half would appear twice in the JCTC text) |
| raw-ladder carry (tex) | `\textbf{The hardware leg at their published construction:` | `\textbf{The hardware leg at the published construction.}` |
| SI title, Table S4 (md + tex) | `Cross-input audit: the spin–energy vise` | `Cross-input audit: the spin–energy trade-off` |
| s210_a start = end (md) | `**The flagship dimension itself: converged, and still not the singlet.**` | `**The largest published dimension.**` |
| s210_a start = end (tex) | `\textbf{The flagship dimension itself: converged, and still not the singlet.}` | `\textbf{The largest published dimension.}` |
| s210_b end (md) | `**The caveat retired, the fourth root, and the second moments.**` | `The same run measures ⟨S⁴⟩ for each root.` (stage-4 paragraph split in three; the span must end at the third) |
| s210_b end (tex) | `\textbf{The caveat retired, the fourth root, and the second moments.}` | `The same run measures $\langle S^4\rangle$ for each root.` |
| s210_c start (md) | `The inversion is reproducible from the two moments per root` | `The inversion can be reproduced from the two moments of each root` |
| s210_c start (tex) | `\textbf{The kernel closed out: an independent implementation` | `\textbf{Independent implementation at the flagship dimension.}` |
| s210_c end (md) | `**The kernel closed out: an independent implementation at the flagship` | `The certified stage-4 vectors are eigenvectors of the independently built operator` (kernel paragraph split in two) |
| s210_c end (tex) | same as its start (one paragraph) | `The certified stage-4 vectors are eigenvectors of the independently built operator` |
| s211_a start (md / tex) | `**The instrument turned on its own output states.**` / `\textbf{...}` | `**Measurement check on the pipeline states.**` / `\textbf{Measurement check on the pipeline states.}` |
| s211_a end (md) | same as its start | `A nonnegative reading from ⟨S²⟩ and ⟨S⁴⟩ alone is feasible but not unique.` |
| s211_a end (tex) | same as its start | `A nonnegative reading from $\ssq$ and $\langle S^4\rangle$ alone is feasible but not unique.` |
| s211_b end (md + tex) | same as its start | `We also stress-tested the bounds against their tolerance boxes.` |
| disc_a start (md) | `**What the controversy is actually about.**` | `Both positions in the dispute treat` |
| disc_a start (tex) | `\textbf{What the controversy is actually about.}` | `Both positions in the dispute treat` |
| disc_mech start (md) | `**The structural mechanism.** The pattern has a one-line root` | `**The structural mechanism.** The pattern has a single cause.` (tex start `\textbf{The structural mechanism.}` unchanged) |
| disc_mech end (md + tex) | same as its start | `The noncommutation also separates three conditions` (paragraph split in two) |
| disc_b start (md) | `**The full-scale successor inherits the gap.**` | `**The full-scale successor.**` |
| disc_b start (tex) | `\textbf{The full-scale successor inherits the gap.}` | `\textbf{The full-scale successor.}` |
| disc_b end | unchanged (`**Recommendations.** (1) Every selected-CI-family benchmark` / `\textbf{Recommendations.}`) | |
| limits start (md + tex) | `We flag, adversarially against ourselves, the gaps a rebuttal` | `We list the limitations a critical reader would raise` (tex end = start; md end `**Package version.**` unchanged) |
| abstract start = end (md) | `Sample-based quantum diagonalization (SQD/QSCI) selects determinant` | `Sample-based quantum diagonalization (SQD/QSCI) diagonalizes the Hamiltonian` |

## JCTC package: changes beyond the pins

The JCTC rewrite files (`jctc_rewrites/`, `jctc_rewrites_tex/`) were rewritten
from the v6.0 masters, and the markdown pair, the LaTeX pair and the preprint
each audit at 370/0. Five things changed beyond wording:

1. **A misstatement in the journal compression, corrected.** The compressed
   Section 2.6 said the released pipeline was run "on their own [2Fe-2S]
   samples". It was run on 10⁶ samples drawn from our converged [2Fe-2S] vector
   (D = 170,448, ⟨S²⟩ = 4.669), as the preprint has said in every version. The
   error was confined to the unsubmitted JCTC files.
2. **Abstract.** The JCTC author guidelines for Articles state that "An abstract
   of 3-4 sentences is required". The JCTC abstract is now four sentences
   (121 words). The preprint abstract is unchanged. The pinned phrase "A scalar
   mean does not identify these states", which had appeared only in the
   abstract, now opens the eight-state conclusion of Section 2.11.
3. **Table and figure citations.** The JCTC LaTeX text now cites the penalty,
   flagship-ladder and four-root tables, the flagship figure and Table S5,
   which it previously did not. Section 2.9 gains one sentence stating the
   ladder construction, because the paragraph that describes it moves to the
   SI with Table S5. The LaTeX master now cites the four-root table and the
   flagship figure in its text as well.
4. **A section number that was wrong in the SI.** The completion-mechanism
   table caption hard-coded "(Section 2.6)", which printed 2.6 in the JCTC SI,
   where the section is 3.6. It is now a `\ref`.
5. **Layout.** The completion-mechanism table ran 15.9 pt past the text block
   in the preprint and in the SI; its column padding is now 2.5 pt (was 4 pt).
   The JCTC eight-state table ran 17.7 pt over; its padding is now 2.2 pt (was
   3.5 pt). Type size is unchanged (tables stay uniform `\small`, no
   `\resizebox`). All three PDFs now build with zero overfull lines.
