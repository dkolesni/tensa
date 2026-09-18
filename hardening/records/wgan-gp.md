# Challenge record — wgan-gp

Challenge: WGAN-GP differentiation pressure, tier 5 (§13), 2026-09-18.
Source/paper: Gulrajani et al., Improved Training of Wasserstein GANs (2017).
TENSA features stressed: interpolated real/fake inputs, stopped generator contribution, critic, input-gradient norm penalty differentiable with respect to critic parameters.
Expected semantics: xhat = mix*real + (1-mix)*stop_grad(fake); penalty = mean((||d Critic(xhat)/d xhat||₂-1)²). The Wasserstein real/fake terms already fit ordinary objectives; this fixture isolates the first unrepresentable term, not a full training plan.
TENSA implementation: `research.ts`, `wgan-gp`.
Check result: rejected by design at grad(score,xhat).
Inspect result: partial error IR only, not a validated learning system.
IR result: interpolation and critic elaborate, but there is no differentiation node.
Runtime result: NOT RUN (invalid program).
Reference result: mathematical torch reference preserved in research.reference.py; not used as a custom-op substitute for the rejected TENSA program.
Gradient result: unavailable; must retain the input-gradient graph for critic differentiation, unlike a detached adversarial perturbation.
Diagnostics: exact primary compiler message `AXS0204: unknown operation 'grad'`; downstream AXS0408 errors are cascades. Unknown-op spelling mutant also pins AXS0204.
Finding classification: G-cand-002, differentiation analysis only.
Severity: blocker for WGAN-GP, not a bug in ordinary parameter backpropagation.
Workaround: host PyTorch/autograd region with explicit contracts; no fake successful reference run.
Proposed action: proposals/differentiation.md separates parameter gradients, penalties and input gradients.
Language change required?: undecided; no grad syntax added.
Regression test added?: challenge:wgan-gp checks rejection and exact primary message.
