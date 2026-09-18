# Challenge record — vit

Challenge: ViT, tier 3 (§30), 2026-09-18.
Source/paper: Dosovitskiy et al., An Image Is Worth 16x16 Words (2020); tiny 4x6 RGB image, 2x2 patches, one encoder block.
TENSA features stressed: rank-six reshape, three transposes, class token broadcast/concat, positional table, head split/merge.
Expected semantics: [B,C,Hp,P,Wp,P] → [B,Hp,Wp,C,P,P], six patches plus one learned class token; return its final feature.
TENSA implementation: `research.ts`, `vit`; RESEARCH_METAMORPHIC has manual vs catalog attention with matching parameter sequence.
Check result: accepted without warnings; all constraints proved. Extra patch twin → AXS0401.
Inspect result: 16 parameter tables including class token, positional table, attention, norms and MLP.
IR result: reshape/transpose/concat/positional/attention. No -1 shape placeholders or custom kernels.
Runtime result: [2,8]; manual and catalog versions agree within 1e-4.
Reference result: GPU emitted forward and gradients match CPU oracle with aligned weights. Reference patchify is also preserved in research.reference.py.
Gradient result: parameter gradients compared, including class and position parameters.
Diagnostics: absent transpose is NOT diagnosable by current shape rules: equal element count, wrong layout. Numeric mutant fails equivalence.
Finding classification: E-007 → G-cand-004, second distinct layout witness after MHA.
Severity: major design boundary, not a compiler shape bug.
Workaround: explicit transposes and numerical layout oracle.
Proposed action: proposals/layout-provenance.md; no named-axis syntax implemented.
Language change required?: not approved.
Regression test added?: challenge:vit; vit-manual-vs-catalog; E-007/G-cand-004 mutant; GPU gate.
