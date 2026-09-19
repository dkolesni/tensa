# Challenge record — vae

Challenge: VAE, tier 4 (§17), 2026-09-18.
Source/paper: Kingma & Welling, Auto-Encoding Variational Bayes (2013).
TENSA features stressed: tuple encoder outputs, reparameterization, graph stochasticity, weighted reconstruction + KL, tuple plan bindings.
Expected semantics: z = mu + exp(logvar/2)*epsilon; epsilon ~ N(0,I). Loss is mean reconstruction MSE + .01 times batch-mean summed latent KL.
TENSA implementation: `research.ts`, `vae`; ten parameter tables.
Check result: accepted without warnings; double-width sampling twin → AXS0401.
Inspect result: stochastic effect, encoder/decoder ownership and checkpoint parameters visible.
IR result: randn_like/exp/mean; randomness is library vocabulary (H-012), not AST syntax.
Runtime result: training reaches every parameter; eval still samples independently. Removing the explicit mu operand after tuple destructuring produces AXS0408.
Reference result: aligned-noise forward/gradient comparisons and unconstrained stochastic eval tests pass on CUDA. Tuple projections share one forward per loss (F-023).
Gradient result: epsilon has no derivative to the shape template; mu/logvar remain differentiable through reparameterization.
Diagnostics: AXS0401 twin, AXS0408 cursor probe.
Finding classification: H-012 and F-023 fixed; E-003/G-cand-005 cursor visibility.
Severity: major before fixes.
Workaround: none for sampling; use explicit operands after destructuring.
Proposed action: retain standard-normal/eval/gradient tests; no grad construct.
Language change required?: no.
Regression test added?: challenge:vae H-012 check; backend F-023; flow G-cand-005; GPU gate.
