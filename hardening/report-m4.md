# TENSA hardening — Milestone 4: research pressure

2026-09-18, `develop`, no commit. Protocol `tensa_hardening_plan.md` unchanged; no AST/grammar changes.
Baseline independently verified after `npm ci`: 467 tests, typecheck and build green. Tier is defined
in challenges/types.ts (consumed by driver.ts), and M0–M3 corpus wiring was verified.

## What TENSA expresses exceptionally well

- DenseNet's four symbolic growth concatenations prove C+4G without handwritten input-channel counts.
- ViT patchify/class-token/position/attention/MLP stays in one language; manual and catalog attention agree numerically.
- Separate encoder/decoder graphs compose through a masked nested loss binding, with every parameter on the tape.
- VAE reparameterization and weighted KL/reconstruction need only a sampling catalog addition (H-012).
- LoRA's frozen base and explicit rank-two adapter use existing structural optimizer regions.
- Temperature-2 distillation extends the existing challenge with a residual teacher, stop_grad and T² loss scaling.

## What remains inherently complex

Patch/channel order, loss weights, LoRA rank/scale/initialization, KL reduction, diffusion timestep
schedule, MoCo positive targets/queue order/temperature, BN policy, gradient retention and RL return
estimation are choices. They remain explicit. Nine compiling fixtures are not nine production trainers:
MoCo is an independent-key subset, diffusion is a noise-prediction training kernel, RL is a fixed-trajectory update.

## Accidental friction still present

1. Cursor provenance recurs in DenseNet/VAE after M1 side bindings (G-cand-005). Explicit operands solve it; prefer tooling review.
2. Layout bookkeeping recurs in ViT after MHA (G-cand-004). Static extents cannot prove permutation intent.
3. Model tuple projections repeat source text. F-023 repairs coherent evaluation within a loss; separate losses remain separate forwards.
4. Host data fixtures/schedules remain necessary because the data plane is checked, not executed.

## Hidden information discovered

- F-023: apparently one diffusion tuple was actually multiple random forwards.
- F-024/F-025: Gaussian mask fixtures and empty attention rows disagreed with the stated mask contract.
- F-026: class-index clamping hid invalid RL actions from reference tests while CUDA rejected them.
- F-027/F-030: signed floor expressions and unknown-rank slices produced false shape certainty.
- F-028: the numerical oracle could silently pass nonfinite values or different shapes.
- F-029: equal training outputs hid divergent BN checkpoint variance.
- E-011/E-012: docs promised tensor select/EMA coverage and an example named alignment NT-Xent. Corrected, not papered over.

## Backend leakage discovered

GPU placement belongs in execution policy, not model mathematics. H-011 implements existing device
metadata: GPU (CUDA/MPS) by default, fail when unavailable, CPU only by explicit selection. Parameters,
state, constants, masks and train/validation batch transfer follow that policy. CLI/Run explicitly
selects the CPU reference oracle and is labeled accordingly. Mixed precision is still an annotation.

## Missing abstractions

- G-cand-002: differentiation analysis separates parameter gradient, differentiable input-gradient penalty,
  and adversarial input gradient. Scope, cotangents, graph retention and effects are not settled.
- G-cand-003: evaluable derived parameter sets. MoCo concretely hits the open BN-buffer/mode choice;
  stop_grad and freeze do not specify it. Detached EMA and differentiable MAML derivation cannot share an unconditional gradient-stop rule.
- G-cand-004: optional layout provenance, analysis only; two numerical witnesses.
- G-cand-005: cursor visibility candidate, but existing composition works; no semantic change justified yet.
- G-cand-001 is NOT promoted: the seven-case disposition splits selection, scan, true lazy execution,
  sparse dispatch and host search rather than calling every dynamic algorithm a cond witness.

## Backend capability gaps

H-012 randn_like and H-011 GPU policy are fixed. Ordinary parameter backpropagation works; derivative-as-value
is not conflated with a missing first-order gradient implementation. RNG checkpoint/replay, executable
source adapters, autocasting, distributed policy and a versioned restoration driver remain outside this milestone.
Dynamic custom slices retain unknown shape; a custom operation without a reference implementation remains unvalidated.

## Escape-hatch usage

No compiling M4 tensor math uses a custom op. WGAN-GP and MoE are rejected, not disguised by identity
custom substitutes. Host code owns real data/timestep schedules, RL environments/actions/trajectories and
actual MoCo momentum-parameter/buffer management. The compiled MoCo subset deliberately does NOT perform
the last item. MAML is skipped-with-reason because the differentiation precondition did not converge.

## Rejected language changes

No grad keyword, no cond AST, no architecture keywords, no global change to let/cursor semantics,
no silently copied model aliases, no forced named axes, no replacement of sparse routing by a dense
soft mixture, and no first-order-MAML substitute. Sampling is catalog vocabulary. GPU default is an
execution-policy requirement using the existing device setting, not a tensor-language extension.

## PyTorch comparison

**Executed**, not merely inspected: torch **2.11.0+cu128**, **NVIDIA GeForce RTX 4090**, default CUDA.
Nine compiling challenges, thirteen model graphs, seven training plans pass.
Hardware-label correction: the saved CUDA result names the RTX 4090; the earlier RTX 5090 label
incorrectly assumed CUDA enumeration matched nvidia-smi indices. The subsequent performance benchmark
explicitly selects the RTX 5090 (see records/cpu-gpu-benchmark-2026-09.md). Two expected rejections
are recorded; MAML is not counted as an executed rejection.

The reproducible gate is `hardening/validate-m4.ts` + `hardening/validate-m4.py`. It exports identical
weights, inputs and stochastic draws from the explicitly CPU reference oracle, compares GPU forward
values (3e-4), parameter gradients (5e-4), train forward/state (5e-4), read-only eval, checkpoint
state_dict load, unconstrained eval randomness, plan execution and host-to-GPU batches. Both generated
model and plan defaults require GPU; a mocked no-GPU environment must raise. The controlled draws test
math, not RNG-stream equivalence. Paper-level formulas are preserved in `records/research.reference.py`.

This is not numerical optimizer-trajectory equivalence: the reference training loop is a simulation,
while generated PyTorch uses actual Adam/AdamW. Likewise synthetic trajectories are not on-policy RL,
and independent key weights are not a momentum encoder. Python remains more natural for these host
orchestration experiments; TENSA supplies inspectable contracts and parameter ownership for the learning kernel.

Reproduce:

```sh
npm test
npm run typecheck
npm run build
npx tsx hardening/validate-m4.ts
<CUDA-enabled-python> hardening/validate-m4.py
# Explicit exception, never automatic fallback:
<torch-python> hardening/validate-m4.py --device cpu
```

Default `python` in this checkout had no torch. The tested installation was
`K:/Anima-TrainFlow/python_embeded/python.exe`. No Python packages or GPU settings were modified.
Generated fixtures/results are ignored under hardening/.m4-validation; the harness regenerates them.

## Next hardening targets

1. Functional parameter evaluation plus explicit buffer/mode/refresh policy, before language syntax.
2. Batch-coupled critic/VJP and higher-order gradient numerical witnesses, then reconsider MAML.
3. Unknown propagation through remaining structural nodes and stronger signed/floor dimension generators.
4. Full data execution and per-port categorical domains; real online rollout freshness contract.
5. M5 frontier decoder/KV cache and distribution thought experiments; lazy-effect joins before cond.
6. Expand GPU parity beyond the M4 set; M3's historical torch 2.12 claim is not a new run of that entire corpus here.

### Milestone coverage and gate

New research entries: densenet, vit, encoder-decoder, vae, lora, moco, wgan-gp, diffusion, moe-topk,
rl-policy-value. Distillation extended in learning.ts. U-Net, Transformer, GAN and recurrence retain
M1–M3 coverage. Existing contrastive examples are **alignment-only**, not full SimCLR; MoCo is the
explicit richer-objective/state attempt. Required failures are classified rather than counted as successes.

Final local gate: **578/578 tests**, `npm run typecheck`, `npm run build`, plus the default-GPU fidelity
gate. Revalidation is recorded separately at `records/revalidation-2026-09.md`. Added fixes:
F-023…F-030, H-012…H-011; design/documentation entries E-011…E-013, G-cand-004…005.
