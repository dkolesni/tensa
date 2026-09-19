# Challenge record — moe-topk

Challenge: top-k MoE routing, tier 6 (§29), 2026-09-18.
Source/paper: Shazeer et al., Sparsely-Gated Mixture-of-Experts (2017).
TENSA features stressed: top-k routing, tensor selection and sparse conditional expert execution.
Expected semantics: route tokens to selected experts without evaluating all experts; capacity/overflow and load-balancing choices remain explicit.
TENSA implementation: `research.ts`, `moe-topk`, minimal router probe. The proposed where call is a capability probe, not claimed to be a sufficient sparse-dispatch implementation.
Check result: expected rejection.
Inspect result: partial router IR only; cannot claim sparse execution from accepted softmax/linear nodes.
IR result: no topk, where or cond nodes. M2 soft routing computes EVERY expert and is not an equivalent sparse fallback.
Runtime result: NOT RUN (invalid program).
Reference result: host sparse dispatch is the appropriate experimental escape; no identity custom substitute.
Gradient result: routing/index gradients and capacity policy unresolved, not ordinary MLP gradient coverage.
Diagnostics: exact compiler messages `AXS0204: unknown operation 'topk'`, `AXS0204: unknown operation 'where'`, followed by missing-result cascade AXS0303.
Finding classification: H for missing selection/ranking library vocabulary; G-cand-001 for genuine conditional regions, not yet promoted.
Severity: blocker for sparse MoE.
Workaround: host/custom region with declared contracts, or an explicitly different dense soft mixture.
Proposed action: records/cond-disposition.md before considering cond.
Language change required?: not established by this witness alone.
Regression test added?: challenge:moe-topk, unknown-op mutant and parser mutation sweep.
