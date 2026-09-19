# Challenge record — rl-policy-value

Challenge: actor/value update over fixed trajectories, tier 6 (§32), 2026-09-18.
Source/paper: standard advantage actor–critic objective; tiny shared trunk with categorical policy and scalar value head.
TENSA features stressed: tuple heads, selected log-probability via one_hot, stopped advantage, value regression, explicit objective ports.
Expected semantics: loss = -mean(log pi(action|obs)*stop_grad(advantage)) + .5*MSE(value,return). This is the learning side, not an environment implementation.
TENSA implementation: `research.ts`, `rl-policy-value`.
Check result: accepted without warnings; extra action axis → AXS0603.
Inspect result: six parameter tables and explicit policy/value outputs.
IR result: log_softmax/one_hot/stop_grad; shared tuple forward within the objective (F-023).
Runtime result: fixed synthetic trajectories train every parameter. No live rollout occurs.
Reference result: CUDA forward/gradient comparison and emitted plan pass. GPU one_hot initially rejected synthetic actions outside [0,3), exposing F-026's hidden reference clamp.
Gradient result: advantage targets stopped, shared trunk receives policy/value gradients.
Diagnostics: AXS0603 twin; invalid runtime classes now refuse instead of silently changing action identity.
Finding classification: E-013 (B boundary), F-026 fixed; dynamic rollout refresh remains unrepresented.
Severity: major limitation for on-policy learning, not a demand for shell/filesystem/network primitives.
Workaround: host owns environment step/reset, action sampling, rewards/dones, trajectory construction, return/advantage estimation, replay and rollout/update alternation. It supplies tensors to emitted learning code. Generic data declarations alone do not promise online freshness.
Proposed action: M5 learning-system driver contract; keep external application effects outside TENSA.
Language change required?: no for fixed-trajectory update; rollout integration is an open boundary study.
Regression test added?: challenge:rl-policy-value; backend F-026; GPU gate.
