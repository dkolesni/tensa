# Challenge record — distillation (M3 extended in place)

Challenge: temperature distillation, tier 4, 2026-09-18.
Source/paper: Hinton et al., Distilling the Knowledge in a Neural Network (2015).
TENSA features stressed: real residual teacher (32→64→32), separate smaller student, teacher stop-gradient, weighted soft/hard objectives.
Expected semantics: soft targets softmax(stop_grad(teacher)/2); student log_softmax(student/2); soft loss multiplied by T²=4, combined 1:1 with hard CE.
TENSA implementation: existing `learning.ts` distillation entry extended, not duplicated in research.ts. Teacher weights are randomly initialized fixture weights, NOT a claim to a trained teacher.
Check result: no warnings; contradictory stop conditions twin retains AXS0707.
Inspect result: 16 parameter tables; teacher and student roots distinct.
IR result: residual, norms, stop_grad, softmax/log_softmax and cross_entropy.
Runtime result: teacher not claimed by any optimizer; all student tables updated.
Reference result: CUDA forward/gradient fixtures for both architectures and emitted plan execute.
Gradient result: target-side teacher gradient cut by objective; independent forward-gradient fixture checks the teacher implementation too.
Diagnostics: AXS0707 twin. Teacher refresh is still unwritable (G-cand-003).
Finding classification: B for temperature/weights; G-cand-003 for refreshed teacher, unchanged.
Severity: major only for the unsupported refresh variant.
Workaround: fixed teacher weights supplied externally; no refresh represented.
Proposed action: derived-parameter-set analysis, not a second distillation challenge.
Language change required?: no for fixed teacher.
Regression test added?: challenge:distillation expanded layers; GPU gate.
