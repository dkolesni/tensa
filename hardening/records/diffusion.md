# Challenge record — diffusion

Challenge: tiny diffusion noise-prediction training loop, tier 5, 2026-09-18.
Source/paper: Ho et al., Denoising Diffusion Probabilistic Models (2020), single sampled timestep training objective.
TENSA features stressed: stochastic tensor draws, timestep conditioning, tuple prediction/noise targets, stopped target and repeated optimizer updates.
Expected semantics: host supplies logit(alpha_bar); sigmoid maps it into (0,1). Noisy input is sqrt(alpha_bar)*x + sqrt(1-alpha_bar)*epsilon; target is THE SAME epsilon returned by the model. Predictor conditions on alpha_bar.
TENSA implementation: `research.ts`, `diffusion`; four parameter tables, ordinary epoch loop. This is a training kernel, not a full reverse-time sampler or a production scheduler.
Check result: accepted; nonbroadcastable alpha width → AXS0401.
Inspect result: stochastic effect and stopped target are visible; checkpoint contains parameters/optimizer/lifecycle state.
IR result: randn_like, sigmoid, sqrt, concat, MLP and stop_grad.
Runtime result: training reaches all parameters; F-023 fixes the formerly different target draw.
Reference result: GPU aligned-noise forward/gradients, independent eval draws and training plan pass. Host timestep sampling remains explicit in the reference boundary.
Gradient result: target has no gradient; reparameterized model path works normally.
Diagnostics: AXS0401 twin; no warnings.
Finding classification: H-012 reused, F-023 fixed; B for schedule selection. EMA remains the existing G-cand-003 witness, not a duplicate challenge.
Severity: major before tuple fix.
Workaround: no sampling escape needed; real data/timestep scheduling host-side.
Proposed action: reverse-time scan and schedule-data integration in a later round.
Language change required?: no for this training kernel.
Regression test added?: challenge:diffusion; backend F-023; GPU gate.
