# Challenge record — densenet

Challenge: DenseNet connectivity, tier 2 (§30), 2026-09-18.
Source/paper: Huang et al., Densely Connected Convolutional Networks (2017); tiny four-growth block, not a full classifier.
TENSA features stressed: long-lived feature names, four concatenations, free C/G/H/W and C + 4G algebra.
Expected semantics: each growth convolution reads all preceding features; concat retains their order.
TENSA implementation: `research.ts`, `densenet`; all four growth paths are explicit.
Check result: accepted without warnings; channel-sum constraints proved. Missing final growth twin → AXS0401.
Inspect result: eight parameter tables; input channel counts increase at each convolution.
IR result: concat/conv2d/relu; no foreign operations. Algebraic rewrite C+4G ↔ G+(C+3G) has identical IR/output.
Runtime result: CPU oracle, B=2,C=3,G=2,H=2,W=3; output [2,11,2,3].
Reference result: emitted PyTorch on RTX 4090, forward and parameter gradients match aligned fixtures; symbolic allocations use dims expressions.
Gradient result: numerical comparison of mean-output gradients passes, not a claim of classifier accuracy.
Diagnostics: AXS0401 for missing growth; a shape-correct aux-binding cursor mutant is accepted and pinned separately.
Finding classification: B (connectivity/order are mathematical choices); E-003/G-cand-005 (side let redirects cursor).
Severity: minor notation friction.
Workaround: explicitly pipe c3 into the fourth growth convolution.
Proposed action: cursor visibility review, not changed semantics.
Language change required?: no; candidate analysis only.
Regression test added?: challenge:densenet; dense-channel-algebra; flow E-003/G-cand-005; GPU gate.
