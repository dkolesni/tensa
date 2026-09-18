# G-cand-005 — cursor provenance / visibility (§43)

2026-09-18. Recurring-friction candidate as requested, not approval to change the cursor rule.

Observed failures: E-003's M1 side let redirects the next bare operation; a DenseNet aux feature redirects the fourth growth convolution while its output shape still passes; VAE destructuring leaves a tuple cursor that randn_like() cannot consume (AXS0408). All follow the documented rule.

Common underlying concept: make the implicit input edge apparent when named side computations coexist with an implicit stream.

Why existing constructs are insufficient: existing explicit operands DO express the intended programs cleanly. The problem is readability/visibility, not representability. This weakens §43 bar 4 and is why the candidate is not promoted.

Proposed semantics: preserve “most recent statement”; investigate inspect/editor display of the consuming edge and an optional lint at a bare operation following a side/tuple binding. No new cursor syntax or silently non-advancing let.

Static knowledge gained: no new type facts; users see already-known dataflow provenance and intentional versus surprising consumption.

New ambiguity introduced: a linter cannot reliably infer that a named value was intended as a side branch. Avoid mandatory warnings that punish deliberate sequential lets.

Interaction with existing constructs: tuple destructuring, multi-input entry, branch-local scope, scan and shared stage bindings. Explicit pipelines continue to remove doubt.

Programs simplified: reading DenseNet/VAE and future multitask objectives, without rewriting their graph.

Programs made harder: sequential naming if the language acquires two different let meanings or extra reseed syntax.

Alternative rejected: change let not to advance the cursor; guess a tuple element; auto-seed the first input; add a cursor keyword without a usability study.

Evidence bars: recurring friction yes, missing representational abstraction no. Prefer tooling review; no G-class implementation. Existing M1 and M4 regressions pin current semantics.
