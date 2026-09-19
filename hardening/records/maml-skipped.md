# Challenge record — maml-skipped

Challenge: MAML-style inner/outer differentiation, tier 6 (§13/§30), 2026-09-18.
Source/paper: Finn et al., Model-Agnostic Meta-Learning (2017).
TENSA features stressed: differentiable parameter update theta' = theta - alpha*dL_inner/dtheta; outer evaluation at theta'.
Expected semantics: full MAML retains the inner derivative graph; first-order MAML deliberately drops it and must not be substituted silently.
TENSA implementation: skipped-with-reason; no fake executable Challenge entry or acceptance claim.
Check result: not attempted as a complete source program because the §13 precondition did not converge.
Inspect result: N/A.
IR result: N/A; neither gradient-as-value nor evaluable derived parameters exist.
Runtime result: NOT RUN.
Reference result: conceptual parameter-gradient witness in proposals/differentiation.md; no equivalence claim.
Gradient result: unresolved higher-order differentiation and parameter-state ownership.
Diagnostics: WGAN-GP already pins the exact unsupported grad diagnostic, AXS0204. MAML is not counted as an independent executed rejection.
Finding classification: G-cand-002 + G-cand-003.
Severity: research blocker.
Workaround: host torch functional parameter evaluation, with an explicit full/first-order choice.
Proposed action: settle differentiation retention/regions and derived-set BN policy before attempting syntax.
Language change required?: proposal stage only.
Regression test added?: no; intentionally skipped under the user's §13 condition, not a silently missing milestone item.
