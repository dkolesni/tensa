# Language and implementation revalidation — September 2026

Performed 2026-09-18 after M4 architecture attempts. Read docs.ts Language/Compare/Report and all
bundled examples against the actual programs. No G-class changes implemented. Priority order below
separates defects fixed now from design/boundary work. Evidence is bounded: this is not an exhaustive
proof of compiler correctness or a production training benchmark.

## Priority findings and destinations

| Priority | Class / ID | Witness | Disposition / destination |
|---|---|---|---|
| P0 | F-023 | diffusion/VAE tuple loss ports | Fixed exec + emitter: one stochastic/stateful tuple forward per loss, fresh per update/loss. |
| P0 | F-027 | signed floor eqDim/leDim | Fixed dims/analyzer: sound conservative sign bounds; seeded rewrite and evaluation properties. |
| P0 | F-030 | unknown custom result sliced [:,0:T] | Fixed analyzer/runtime/emitter: unknown is not rank zero; retain runtime indexing, no false bounds. |
| P0 | F-028 | identical nonfinite programs; [2,2] vs [4] | Fixed metamorphic oracle: finite values, actual shapes and runtime errors checked. |
| P1 | F-024/F-025 | encoder-decoder Mask[B,1,S] | Fixed boolean synthetic masks and fully blocked catalog-attention output/gradient. |
| P1 | F-026 | RL actions outside three classes | Fixed silent one_hot/CE clamping; valid conservative fixture domain. CUDA exposed what CPU had concealed. |
| P1 | F-029 | MoCo query/key running_var | Fixed reference unbiased running estimate; population variance still normalizes training outputs. |
| P1 | H-010 | VAE/diffusion sampling | Added randn_like vocabulary/effect/runtime/lowering/coverage. Train/eval samples; no template gradient. |
| P1 | H-011 | all M4 emitted plans | GPU default required; CPU explicit only. Device placement and no-fallback gate implemented. |
| P1 | G-cand-003 | MoCo BN encoder + queue | Proposal updated: source/key statistics, mode and refresh order unresolved; no derived-set syntax. |
| P1 | G-cand-002 | WGAN-GP + input/parameter gradient witnesses | §43 analysis separates retention/targets/ownership. MAML skipped, not approximated. |
| P2 | E-007 / G-cand-004 | MHA merge + ViT patchify | Proposal analysis only; numeric layout mutants remain indispensable. |
| P2 | E-003 / G-cand-005 | M1 side let + DenseNet aux + VAE tuple | Existing semantics pinned. Tooling visibility first; bar 4 for new syntax not met. |
| P2 | E-004 | softmax model result bound to logits objective | Warning retained deliberately, 2026-09-18; no unilateral severity change. Plain Tensor remains permissive refinement. |
| P2 | E-011/E-012 (C) | stale Language/Report/Compare/example claims | Corrected docs and renamed fake NTXent example to Alignment; distinguish actual vs promised semantics. |
| P2 | E-013 (B) | toy policy/value update | Fixed-trajectory learning fits; host owns environment, reward/return calculation, action sampling, rollout refresh. |
| P2 | G-cand-001 | top-k MoE and seven dynamic dispositions | Remains candidate; missing topk/where vocabulary does not by itself justify cond. |

## Language design: what fought the notation

**Cursor.** DenseNet needs long-lived c1/c2/c3 names. Inserting a side aux binding before a bare
conv can change its input while preserving the final output shape. VAE's destructured tuple cannot
seed randn_like(): the explicit mu operand is essential. These are correct documented semantics,
not more F-007 bugs. Repeated witnesses justify G-cand-005 review, but explicit pipelines already
express intent; changing let would break straightforward sequential programs.

**Kinds.** Logits/Probs/Mask are shallow refinements, not nominal contracts. The objective binding
warning in E-004 remains a warning; direct Probs into cross_entropy has AXS0409. M4 did not establish
that every plain Tensor feeding logits is invalid. Mask values must nevertheless be boolean in the
fixture engine; that implementation bug was fixed without changing the type system.

**Axes/layout.** ViT's three transposes expose the same cardinality-versus-layout boundary as MHA.
A bad patch permutation still compiles and fails the numerical metamorphic comparison. DenseNet proves
channel SUMS, not feature order. Optional layout provenance is a candidate, not evidence that mandatory
named axes would solve arbitrary concat-order errors.

**track / stop_grad / region over.** LoRA and fixed-teacher distillation compose cleanly: over chooses
optimizer ownership, freeze excludes base parameters, stop_grad severs targets. None chooses forward
weights or prevents BN writes. MoCo therefore has an independent key encoder, not a true EMA encoder.
The derived-set proposal's original blanket gradient-stop invariant also conflicted with full MAML;
the analysis now distinguishes detached EMA from differentiably derived parameters. No code implements
that proposed concept.

**Public claims.** docs.ts incorrectly said select existed and the six temporal concepts covered EMA
teachers; observe's text implied returning the old value while both backends return the updated value
in training; mixed precision was described as placed despite being an annotation. The GAN example's
nested-call limitation was stale after H-007. A bundled NTXent-named function had no negative pairs.
All corrected (E-011/E-012), without pretending full SimCLR was validated.

## Compiler correctness attacks

- property.ts: 25 deterministic instances per generator. Equality and bound proved/refuted/carried
  outcomes are exclusive for each obligation and stable under k*T - T*k rewriting. Unknown output
  and unknown slice tests must create neither guessed equality nor bound obligations.
- Direct dims.ts distributivity, signed-floor evaluation over T=1…16 and conservative lower-bound
  checks exposed F-027. This is deliberately conservative, not a complete Presburger solver.
- Nested for/scan: varying static counts and runtime length; shared stages own exactly two tables
  at M/cell/linear#1, independent instances own 2*a*b unique tables. Renaming a shared stage preserves
  canonical IR and numerical output. No new identity collision found.
- Recursive effects: fn → residual → static_repeat → scan preserves the union of stochastic,
  training-sensitive, parameterized, state-read/write and grad-stopped effects (F-016 regression).
- Seeded parser evil twins hit every new research program AND extended distillation: removed final
  brace, mutated arrow, malformed model header. Each terminates with located AXS01xx syntax diagnostics.
  Semantic twins remain per-challenge. This is bounded mutation fuzzing, not exhaustive grammar fuzzing.
- Existing adversarial and property corpus still passes; no diagnostic codes renumbered, no AST changes.

## Backend fidelity

### Actual GPU execution
Default Python had no torch, but an existing CUDA-enabled interpreter was available. No dependency
install or hardware reconfiguration was needed. Reproduce via hardening/validate-m4.{ts,py}; GPU is
required by default. CPU comparison is explicitly the reference oracle, not a fallback execution mode.

Torch 2.11.0+cu128, RTX 4090, CUDA: all nine compiling M4 challenges, thirteen graphs, seven emitted
training plans pass. Weights and inputs align with the CPU oracle; stochastic draws are injected for
numeric comparisons, then fresh eval randomness is checked separately. Comparisons cover forward and
parameter gradients, train/eval state, buffer initialization and checkpoint load. Model/plan allocation
and host batch movement are on GPU. Mocked no-GPU default raises; CPU must be explicitly selected.
MPS path exists but was not hardware-tested.

### Specific audit points
- ViT and DenseNet derived extents use pyDim/dr; no executable show()-formatted algebra. ViT has
  multiple runtime dimensions and rank-six reshape; DenseNet free channel counts size actual weights.
- Both source and cross-attention padding sites negate true=blocked for SDPA and lift rank-three masks.
  Fully blocked catalog rows are zero; handwritten finite-sentinel softmax is not magically changed.
- randn_like remains stochastic in eval, returns floating noise even for integer-shaped templates,
  and has no template-value derivative. No RNG-state checkpoint/replay guarantee is claimed.
- MoCo queue initialization, FIFO updates and eval immutability are checked; query/key BN buffers
  and tracked parameter shadows are separate persistent-state slots. CUDA train-state comparison
  found F-029 even though forward values matched.
- Tuple projection caching lives for one loss only. It is not global common-subexpression elimination
  and never reuses an optimizer-step tape.
- The GPU plan smoke tests use actual PyTorch optimizers. They are not a claim that simulated
  reference optimizer trajectories numerically equal Adam/AdamW or that host data pipelines execute.
- Unknown custom slices are tested using a supplied actual foreign result. AXS0901 placeholders are
  not counted as a validated custom backend. Broader custom/dynamic shape parity remains a next target.

## Gate and remaining risk

578/578 headless tests; typecheck/build green; default-GPU fidelity gate green. The protocol was not
edited, develop remains uncommitted. Remaining high-priority design decisions are effect-safe
differentiation, evaluable parameter sets with buffer/mode policy, and online data freshness. Remaining
verification gaps include full SimCLR, true MoCo momentum evaluation, full diffusion sampling, real
RL rollout, MAML, RNG restoration and hardware outside CUDA. See report-m4.md for milestone scope.
