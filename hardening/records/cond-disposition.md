# §29 dynamic-control-flow disposition — M4

2026-09-18. Each row has one primary destination for the stated semantics; alternatives below are
explicitly different algorithms. `where/select`, `topk` and `cond` are NOT implemented tensor ops.
The data-plane `select` is unrelated.

| Algorithm / required semantics | Primary destination | Why; what is retained |
|---|---|---|
| Adaptive computation with fixed maximum depth, all branches evaluated and per-example halting weights | where/select | Eager masked updates suffice at fixed shapes; selection is a catalog gap. This saves no branch compute. Halting with actual compute savings would be future cond. |
| Early-exit network that must skip later expensive blocks | future cond | Equal-shaped branch results do not imply equal work/effects. Need branch-local state/randomness and gradient joins; eager selection is not equivalent. |
| Expert routing with top-k, packed variable token groups and capacity/overflow policy | custom region | Ranking is library vocabulary, but sparse dispatch/capacity creates dynamic cardinality. A scalar cond alone cannot represent it. Fixed-capacity padded routing could later be a separate static design. |
| Iterate until convergence, bounded budget and a masked converged carry | scan | Existing scan supplies shared iteration parameters; selection is still a library gap. The finite cap is a meaningful choice, not inferred. Unbounded termination belongs to host. |
| Beam search with variable hypotheses and pruning | host | Search bookkeeping, stopping and output cardinality are an inference algorithm; TENSA evaluates the scoring model. |
| Autoregressive decoding with a fixed token budget and fixed-size cache/carry | scan | One bounded recurrent scoring step; EOS masking needs selection. Dynamic-length streaming and cache allocation remain host concerns. Current single-carry representation is awkward for a realistic KV cache. |
| Rejection sampling until acceptance, with unbounded attempts | host | Unknown termination/count and random effects. A bounded retry sampler is a different, explicitly truncated algorithm; custom region is an optional boundary. |

## Executed evidence vs analysis
`challenge:moe-topk` fails with exact AXS0204 messages for `topk` and `where`; no invalid program is
run. M2's norms-and-gates Route evaluates every expert and remains a valid **dense** static graph,
not evidence of sparse routing support. Existing recurrence and the new nested for/scan properties
validate bounded recurrence/identity, not dynamic halting. The remaining rows are disposition analyses,
not invented successful source programs or measured performance results.

## G-cand-001 decision
Do **not** promote yet. Early exit and compute-saving adaptive depth look like equal-contract lazy
regions, but there are not three executed independent witnesses with settled state/effect/gradient
joins. MoE also needs dispatch, not merely cond. Beam search and unbounded rejection do not count as
language-cond witnesses. No ast.ts change, no fake where/topk stub, no general-purpose loop.

## §4 summary
Challenge: seven dynamic patterns, with MoE executable rejection.
Source/paper: protocol §29 / sparse MoE.
TENSA features stressed: static topology versus dynamic work/cardinality.
Expected semantics: preserve actual compute/effect behavior, not just output shape.
TENSA implementation: research.ts/moe-topk; dispositions above.
Check result: MoE rejected; conceptual rows not compiled.
Inspect result: no cond or dispatch node exists.
IR result: soft routing/scan available; dynamic regions absent.
Runtime result: no run of invalid source.
Reference result: host/custom boundary only, no numerical equivalence claim.
Gradient result: conditional join/routing derivative policy still open.
Diagnostics: AXS0204 topk/where, AXS0303 cascade.
Finding classification: H library gaps; G-cand-001 remains candidate; B host boundary.
Severity: research blocker for sparse computation.
Workaround: explicit dense/static algorithm or host/custom region.
Proposed action: collect lazy-effect witnesses before §43 language proposal.
Language change required?: not approved.
Regression test added?: challenge:moe-topk; parser mutants; bounded recurrence properties.
