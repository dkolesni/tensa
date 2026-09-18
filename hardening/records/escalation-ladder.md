# Challenge record — the §26 lifecycle escalation ladder (Milestone 3)

§26 says: attempt the nine lifecycles in order and "record exactly where the current
lifecycle vocabulary stops fitting". This record is that account. Items 1–3 were
covered by `gan-alternating` (two optimizers, two objectives, `times 2`); 4–6 each got a
challenge; 7–9 were attempted on paper and are recorded as gaps with the reason the
program could not be written honestly.

```text
 #  lifecycle                    challenge           written?  where the vocabulary stops
 1  multiple optimizers          gan-alternating     yes       fits (F-017, F-022 found)
 2  multiple objectives          gan-alternating     yes       fits (H-002 earlier, H-007 now)
 3  alternating updates          gan-alternating     yes       fits — `update … times N` is plan data
 4  EMA teacher/student          ema-teacher         partly    the shadow is tracked; no model can
                                                                CONSUME it (G-cand-003)
 5  knowledge distillation       distillation        yes       fits for a FROZEN teacher; a teacher
                                                                refreshed from the student is G-cand-003
 6  self-training/pseudo-label   self-training       yes       fits; pseudo-labels are
                                                                `argmax(stop_grad(t))` — effects
                                                                {grad-stopped, nondiff} say exactly that
 7  adversarial training         —                   no        needs the gradient of the loss w.r.t.
                                                                the INPUT as a value (G-cand-002)
 8  RL rollout/update            —                   no        needs data produced by the model
                                                                during training (the data plane is
                                                                declared once, ahead of the plan)
 9  meta-learning inner/outer    —                   no        needs a temporary parameter set derived
                                                                from the current one by an inner update
                                                                (G-cand-003 + G-cand-002)
```

## Items 4–6 in detail

**4 — EMA teacher** (`ema-teacher`, record `ema-teacher.md`). Everything TENSA *does* say
is checked: two declarations are two parameter sets (F-015 warns when they are not),
`track ema_t = ema(student, rate: 0.99)` produces a shadow that the runtime and the
emitted plan both update with the same decay (E-009), the teacher is untouched by the
optimizer, the shadow appears in the checkpoint. What cannot be written: `teacher(x)`
evaluated *with the shadow weights*. The teacher in the program is a second copy that
never receives the average — the program is a faithful mean-teacher *except for the
teacher*.

**5 — Knowledge distillation** (`distillation`). A frozen teacher works completely:
`stop_grad(t)` in the objective carries `grad-stopped`, the teacher's tables report
`not claimed by any optimizer`, the student's are all updated. The moment the recipe
says "refresh the teacher from the student every K steps" (born-again networks,
self-distillation) the program needs to *assign one parameter set from another*, which
is the same missing thing as item 4.

**6 — Self-training** (`self-training`). Fits as written: `Pseudo(s: M(x), t: M(x))`
with `argmax(stop_grad(t))` inside the objective. The IR effect set `{grad-stopped,
nondiff}` documents precisely why the target side has no gradient; `augment train {x:
noise(0.1)}` puts the augmentation in the data plane. gradAll holds. The one thing the
program does not express is that the pseudo-label pass should run in eval mode (no
dropout, frozen state) while the prediction pass trains — a per-*call* mode, not a
per-phase one. Recorded, not escalated: one program.

## Items 7–9: why no program was written

**7 — Adversarial training (FGSM/PGD).** The perturbation is `x + ε·sign(∇ₓ L(f(x), y))`.
TENSA objectives are functions from tensors to a scalar; the gradient is something the
*plan* takes of the whole loss, never a value inside a graph. There is no honest way to
write the inner step. This is the third program to ask for gradient-as-value after
WGAN-GP and MAML → **G-cand-002 now has three witnesses**; a §43 analysis is the next
step, not syntax (it is not written in this milestone because the data-plane and
lifecycle findings above took precedence, and because the three witnesses want
different things — an input gradient, a gradient penalty, a parameter gradient — that
should be separated before a concept is named).

**8 — RL rollout/update.** The training data (trajectories) is produced by the model
being trained, inside the plan. TENSA's `data` block is a fixed pipeline declared before
`train`, sampled by the plan; there is no construct for "a source that the plan fills".
Not escalated: one witness, and it is arguably an *environment* concept rather than a
lifecycle one.

**9 — Meta-learning (MAML).** The inner loop computes `θ' = θ − α∇θ L_task(θ)` and the
outer loss is evaluated at `θ'`. Two missing things at once: a parameter set derived
from the current one (item 4's gap) and a gradient as a value (item 7's). It is the
strongest argument that G-cand-003 and G-cand-002 are separate concepts that compose.

## The temporal concept that recurs

Items 4, 5 (refresh variant) and 9 fail on the same sentence: *"a parameter set whose
values are a function of another parameter set's values over time, that a model can be
evaluated with."* `track` gives half of it (the derivation) and hides the other half
(the consumption). That is the "same missing temporal concept" §26 asks for before any
expansion → `proposals/derived-parameter-sets.md` (G-cand-003), analysis only.
