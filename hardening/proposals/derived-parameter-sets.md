# Proposal analysis — derived parameter sets (G-cand-003)

**Status:** analysis only (§43). No syntax is proposed in this document; it exists to
decide whether the concept clears the seven evidence bars. Written 2026-09-17 during
Milestone 3.

## §43 evidence checklist

| # | Requirement | Verdict |
|---|-------------|---------|
| 1 | primarily G — missing abstraction | yes — nothing is *wrong*; the sentence cannot be said |
| 2 | ≥2, preferably 3 substantially different algorithms | 3: EMA teacher (mean-teacher / BYOL / DINO), teacher refresh in self-distillation (born-again networks), MAML inner loop |
| 3 | recurring ML concept | yes — Polyak/EMA weights, target networks (DQN), stop-gradient teachers, lookahead optimizers, SWA |
| 4 | cannot be handled through existing composition | see "Why existing constructs are insufficient" — `track` + a second `model` gets the derivation but not the use |
| 5 | improves semantic knowledge, not character count | yes — the checker can prove the teacher is not optimised, is checkpointed, and is evaluated with the derived values; today it can prove none of the third |
| 6 | does not hide a meaningful choice | the decay/refresh rule stays explicit; the risk is hiding *when* the derived set is refreshed — that must remain plan data |
| 7 | no general-purpose machinery | it is a parameter-set-level `state`, not a variable or a pointer |

Bars 1–3, 5, 7 are met. Bar 4 is met for the *consumption* half. Bar 6 is the open
design question, so this stays a proposal.

## §43 template

```text
Observed failures:
    · ema-teacher: `track ema_t = ema(student, rate: 0.99)` tracks the average, but
      `teacher(x)` in the loss is a second independent copy; the actual mean-teacher
      forward pass (student evaluated with averaged weights) is unwritable.
    · distillation with teacher refresh ("every 200 steps teacher ← student"): there
      is no plan statement that assigns one model's parameters from another's.
    · MAML: the outer loss must be evaluated at θ' = θ − α∇θ L_inner; the "evaluate at
      a derived θ" half is this concept (the gradient half is G-cand-002).
    (recorded in records/escalation-ladder.md items 4, 5, 9 and records/ema-teacher.md)

Common underlying concept:
    A DERIVED PARAMETER SET: a second set of values, shaped exactly like a model's
    parameters, whose contents are a declared function of that model's parameters over
    time (EMA with a decay, periodic copy, one inner gradient step), which is NOT
    optimised, IS checkpointed, and can be the parameter set a model is evaluated with.
    `track` already declares the first three properties; the missing one is the last.

Why existing constructs are insufficient:
    · A second `model` declaration is an independent parameter set (F-015 makes sure
      of it). Writing `model teacher = Teacher` gives the right shape and the wrong
      values forever.
    · `state` lives inside a model body and is per-slot; a parameter-set-shaped state
      would have to be spelled once per table and could not be the thing `linear`
      reads from.
    · `track` is write-only from the program's point of view: its value reaches the
      checkpoint and nothing else.
    · Composition (`stop_grad(teacher(x))`) fixes gradient flow, not which values the
      teacher holds.

Proposed semantics (concept, not syntax):
    A plan-level binding that names a model alias AND a derived parameter set, so that
    applying that alias in a loss binding reads the derived values. Invariants the
    checker would own:
      1. a derived set has exactly the tables of its source (same ids, same shapes);
      2. it is never claimed by an optimizer (AXS0703-style error if it is);
      3. it is refreshed by a plan rule (`ema(rate)`, `copy every N steps`) that is
         plan data — visible in inspect and simulated by the runtime;
      4. EMA/copy teachers carry `grad-stopped` toward the source. This MUST NOT be
         imposed on differentiably derived MAML parameters: full MAML retains the
         theta' → theta path. Derivation kind and differentiation policy are separate
         unresolved requirements (M4 revalidation; see differentiation.md);
      5. it is in the checkpoint under its own kind ("derived parameters").

Static knowledge gained:
    · the teacher is provably not trained and provably evaluated with the average;
    · gradient coverage can say "reached only through a derived set" instead of the
      current "no gradient path";
    · the emitted plan can lower the forward pass with `torch.func.functional_call`
      (or a parameter swap) instead of a second module that is silently wrong.

New ambiguity introduced:
    · when in a step the refresh happens (before or after the optimizer step) — must
      be defined once, like H-008's counters;
    · whether a derived set of a derived set is allowed (probably not, initially);
    · effects of `state` inside the source model when evaluated via the derived set
      (BN running stats: the teacher's or the student's?) — this is the bar-6 risk.

Interaction with existing constructs:
    `track` becomes the declaration of a derived set (today's form stays valid: a
    derived set nobody evaluates is just a checkpointed average). `freeze` is
    unrelated (a frozen set is still the source's values). Regions (`Net.encoder`)
    would need to resolve inside a derived set the same way. F-022's `*` region must
    exclude derived sets.

Programs simplified:
    mean teacher, BYOL/DINO, DQN target networks, self-distillation with refresh, SWA
    evaluation, lookahead — each drops a hand-maintained second model whose
    correctness the checker cannot see today.

Programs made harder:
    none of the existing corpus; a program that WANTS two independent teachers keeps
    writing two model declarations.

Alternative rejected:
    · "make `model teacher = Student` mean a copy" — it would silently change F-015's
      semantics and still not give the average;
    · "allow `state` at model-parameter granularity" — pushes a lifecycle concept into
      the model body and loses the plan-level visibility that is the point;
    · "do it in the emitter only" (`functional_call` with the tracked dict) — the IR
      would not know, so inspect/runtime/checkpoint would disagree with the Python.
```

## M4 witness: MoCo hits the BN-statistics bar (2026-09-18)

`research.ts/moco` has two BN encoders, a persistent FIFO bank and a query-parameter EMA
track. Inspect shows distinct query/key running mean/variance buffers; the track holds only
parameter shadows. There is still no connection from the shadow to Key's forward.

Thus bar 6 is **hit, not resolved**: substituting only weights says nothing about whether Key
uses independent running statistics, copied source buffers, averaged buffers, batch statistics
(as in training/shuffled-BN variants), or evaluation-only statistics. stop_grad prevents gradient
flow, NOT BN updates; freeze controls parameters, NOT model mode. Parameter regions compose
with those controls, but none specifies the key encoder's per-call state/mode policy.

The executable subset verifies FIFO updates/read-only eval and separate buffer ownership.
CUDA/reference buffer comparison also found and fixed F-029 (unbiased running variance), an
implementation error independent of the missing derived-set semantics. No extra EMA challenge
was added for diffusion; the existing teacher witness remains authoritative.

Decisions still required: refresh order relative to optimizer and queue update; buffer ownership
and checkpoint categories; per-call mode/effect policy; detached EMA versus differentiable inner
updates. Do not claim MoCo is implemented merely because the independent-key subset runs.

## What would move this to a proposal with syntax

One more witness that is *not* a teacher (a DQN target network is the obvious one), and
a decision on the BN-statistics question under "New ambiguity introduced". Until then
the language stays as it is; the corpus records the gap (`ema-teacher` title,
`escalation-ladder.md`).
