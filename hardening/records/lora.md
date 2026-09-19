# Challenge record — lora

Challenge: LoRA, tier 4, 2026-09-18.
Source/paper: Hu et al., LoRA (2021); tiny linear adaptation, D=8,R=2.
TENSA features stressed: explicit down/up matrices, zero-initialized up matrix, base/adaptor structural regions, phase freeze.
Expected semantics: base(x) + .5*x*down*up; only adapter parameters are optimized. Down initially receives zero derivative because up starts at zero, but is on the tape.
TENSA implementation: `research.ts`, `lora`; source intentionally keeps the frozen base in a named stage.
Check result: accepted; optimizer over frozen base → AXS0704; unapplied base optimizer → AXS0708.
Inspect result: four tables, owners LoRA/base/linear#1 and LoRA/adapter.
IR result: explicit matmul and region-resolved optimizer membership.
Runtime result: three-step check reports both adapter tables updated and no base update.
Reference result: GPU forward/gradient fixture comparison and emitted training plan pass.
Gradient result: zero initial down derivative is correct, not a missing backend gradient; subsequent optimizer steps can move it.
Diagnostics: AXS0704/AXS0708 twins.
Finding classification: B (rank, scale, zero init and frozen region are choices).
Severity: none new.
Workaround: none.
Proposed action: keep structural region checks.
Language change required?: no.
Regression test added?: challenge:lora; GPU gate.
