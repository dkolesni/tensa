# TENSA Hardening — Execution Plan

Companion to [tensa_hardening_plan.md](tensa_hardening_plan.md) (the protocol). That document says
*what* a hardening campaign must do and *how* findings are classified. This document says how to
run it against the TENSA codebase as it exists today: what infrastructure is missing, what is
already covered, what each milestone requires concretely, and where the work lands in the tree.

Section numbers in the form §N refer to the protocol.

---

## 0. Current state vs. protocol requirements

### What exists

| Protocol area | Present in repo | Location |
|---|---|---|
| Example programs | 22 examples, one per concept (MLP, CNN, ResNet, AE, Inception, Siamese, Transformer, GPT, tensor-level attention, RMSNorm/gating, cross-attention, multi-objective, explicit params, state, vision data, contrastive data, transfer, GAN, scan, custom op, diagnostics tour) | `src/lang/examples.ts` |
| Test suite | ~50 tests in 17 groups (parsing, examples, shapes, topology, flow, parameters, effects, objectives, data, lifecycle, recurrence, custom, ir, backend, execution, checkpoint) | `src/lang/tests.ts` |
| Diagnostic catalogue | 43 stable codes AXS0101–AXS1002 | `src/lang/types.ts` |
| Taxonomy A–D | Applied to the 12 closed findings shown on the Hardening page | `HARDENING` in `src/lang/docs.ts` |
| Tensa branch regression | Present as a permanent test | `tests.ts` "REGRESSION: parallel branches…" + execution twin |
| Inspect-vs-IR check | One test | `tests.ts` "inspection agrees with lowering" |

### What the protocol requires that does not exist

1. **A headless test runner.** `runTests()` only executes in the browser (Tests page / CLI tab).
   Every milestone gate in §44–§48 says "all microtests and evil twins pass" — that needs a
   command, not a page.
2. **Challenge records (§4).** No durable record format or storage. Findings currently live only
   as prose in `docs.ts`.
3. **Evil-twin discipline (§2.3).** Negative tests exist but are ad hoc (~10). There is no
   pairing structure that guarantees every positive construct has a nearby failing sibling.
4. **Taxonomy codes E–H (§3).** `HARDENING` entries carry only A–D. Compiler bugs (F) and
   capability gaps (H) have no home, so they get conflated with design findings.
5. **Multi-layer verification (§2.2).** Most tests check one layer (compile ok / diag code). The
   11-point checklist (source → types → params → sharing → effects → IR → inspect → emit →
   runtime → gradients → checkpoint) is applied end-to-end nowhere.
6. **Metamorphic (§37) and property-based (§38) tests.** None.
7. **Diagnostic-quality suite (§36).** No test asserts location, architectural `where`, expected-vs-
   inferred text, or absence of backend jargon in a message.
8. **Backend-leakage audit (§33).** Not automated.
9. **Architecture ladder coverage (§30).** Tier 1 complete; Tier 2 missing U-Net and DenseNet;
   Tier 3 missing ViT and encoder-decoder; Tiers 4–6 essentially absent (VAE, SimCLR-full, MoCo,
   LoRA, distillation, WGAN-GP, EMA teacher, diffusion, MoE, RL toy, meta-learning).
10. **Tabular pipeline (§19).** Referenced by a closed finding but no example or test exists.
11. **`cond`** (§29) — not implemented; documented as a designed extension in `LIMITATIONS`.

Items 1–8 are infrastructure and must land before Milestone 1 can be honestly gated. Items 9–11
are the campaign itself.

---

## 1. Phase 0 — Infrastructure (prerequisite for every milestone)

### 1.1 Headless runner

Add a Node entry point that imports `runTests` and exits non-zero on failure. All of
`src/lang/*.ts` is DOM-free (only `App.tsx`/components touch the browser), so this needs no
refactor — only a script and a TS executor.

- Add `tsx` as a devDependency; add `"test": "tsx src/lang/run_tests.ts"` to `package.json`.
- `run_tests.ts`: call `runTests()`, print `testSummary`, print failures in the same format the
  CLI `test` command uses, `process.exit(failed ? 1 : 0)`.
- Optional: `"test:filter"` accepting a group name so a single challenge group can be iterated.

Update `CLAUDE.md` once this exists.

### 1.2 Challenge corpus module

Create `src/lang/challenges/` holding challenge programs *as data*, separate from the curated
`EXAMPLES` shown in the Playground picker (which should stay small and pedagogical).

```ts
export interface Challenge {
  id: string;             // "unet", "attention-explicit", ...
  section: string;        // protocol section, e.g. "§10", "§12"
  tier?: 1|2|3|4|5|6;     // architecture ladder tier when applicable
  code: string;
  expect: ChallengeExpectation;
  twins: EvilTwin[];      // §2.3 — at least one per positive construct exercised
  record?: string;        // path to the durable record under hardening/records/
}
export interface EvilTwin {
  id: string;
  code: string;
  expectCodes: string[];  // diagnostic codes that MUST fire
  forbidCodes?: string[]; // codes that must NOT fire (false-certainty guard)
  mutates: string;        // one line: what was changed relative to the parent
}
```

`ChallengeExpectation` carries the §2.2 layers that can be asserted mechanically:
`paramCount`, `sharedGroups`, `stateSlots`, `effects`, `irNodeOps`, `inspectContains`,
`emitContains`, `runtimeShape`, `gradReaches`, `checkpointSlots`. Absent fields are not checked;
a challenge whose record says a layer was verified should have that field populated.

A generic driver in `tests.ts` iterates every `Challenge` and produces one test per populated
layer plus one per twin. This is what turns the 11-point checklist from prose into a gate.

### 1.3 Durable records

`hardening/records/<challenge-id>.md`, one per nontrivial challenge, using the §4 template
verbatim. Reference implementations (PyTorch, for §39) go beside them as
`hardening/records/<challenge-id>.reference.py`. These are documentation, not build inputs.

`hardening/findings.md` is the running ledger: one row per finding with primary classification
A–H, severity, challenge id, status (open / fixed-F / backend-H / rejected / promoted-G). The
"Rejected language changes" section of the final report (§49) is generated from this ledger so
future agents stop rediscovering the same ideas — the protocol calls this out explicitly.

### 1.4 Taxonomy extension

Widen the `HARDENING` entry type in `docs.ts` from `"A"|"B"|"C"|"D"` to `"A"…"H"` and give the
Hardening page tones for E–H. This is a UI change only; existing entries are unaffected.

### 1.5 Diagnostic-quality harness (§36)

Add a helper in the challenge driver that, for every twin, asserts on the fired diagnostic:

- `loc.line/col` fall inside the mutated region (twin declares the expected line);
- `where` is non-empty for any code in the 03xx/04xx/08xx families (architectural location);
- message does not match a jargon denylist (`cuda`, `state_dict`, `requires_grad`, `autocast`,
  `backward`, `zero_grad`, `nn.Module`, `im2col` outside AXS0901);
- for shape codes (AXS0401/0402/0406), message contains both the expected and the inferred
  shape text.

This is the "what did I say / what did TENSA infer / why / where" test made mechanical.

### 1.6 Metamorphic and property helpers (§37, §38)

- `metamorphic.ts`: `assertEquivalent(a: string, b: string, opts)` compiling both programs and
  comparing `printIR` modulo value-id renaming, `mod.params` totals, and (when `opts.run`) the
  reference-backend output under a fixed seed.
- `property.ts`: a tiny deterministic generator (seeded PRNG, no external library) producing
  random rank/dim/op chains for the §38 property list. Start with the five cheapest properties:
  reshape preserves element count; concat axis equals symbolic sum; residual preserves shape;
  shared-stage count is invariant under application count; inspect totals equal IR totals.

### 1.7 Backend-leakage audit (§33)

A test that greps every `Challenge.code` and `EXAMPLES[].code` for the §33 token list and fails
if any appear outside a `custom op … backend torch { … }` block. Cheap, permanent.

---

## 2. Milestone 1 — Semantic core (§44)

Gate: every §5–§10 and §16 microtest plus its evil twin passes under `npm test`.

> **Status (2026-09-17): DONE.** Delivered as `src/lang/challenges/semantic.ts` (15 challenges,
> 30 twins, 4 metamorphic pairs) + 4 regression tests in `tests.ts`. Findings F-002…F-010,
> H-002, H-003, H-004, E-003…E-006 are in `hardening/findings.md`; tier-2 record at
> `hardening/records/unet-skip.md`. E-001 resolved (warn → AXS0304). 284/284 tests.

### 2.1 Symbolic dimensions (§5)

Existing: 6 shape tests. Required additions, each with a twin:

- Algebra table (§5.1): `D*4` vs `4*D`, `(D/H)*H`, `T+1-1`, nested products, flatten products,
  conv output expressions, floor division, constants mixed with template vars — one test each
  that both `dims.ts` normalises to the same `DimExpr` **and** that `analyze.ts` treats them as
  provably equal (not merely carried).
- Constraint outcome quartet (§5.2): four programs producing *proved*, *carried* (AXS0403 warning,
  visible in `inspect` constraints section), *impossible* (AXS0401 error), *unknown* (AXS0903 via
  custom op). Assert IR `constraints` and inspect text agree; run the carried case at runtime with
  a violating dim binding and assert the reference backend refuses rather than proceeding.
- Divisibility ladder (§5.4): `D / Heads` proved / impossible (AXS0407) / unresolved-until-
  instantiation / unresolved-at-execution. Assert no silent truncation at any level.
- Op coverage (§5.3): a symbolic-shape test per catalog op that has a `shape` rule. Generate this
  from `CATALOG` so a new op cannot be added without a shape test.

### 2.2 Implicit flow (§6)

Existing: 3 tests. Add the remaining seven bullets as individual tests, each with a twin that
must produce AXS0301 or AXS0302, and a "looks ambiguous, must fail" set (tuple output followed by
a bare transformation; `let` shadowing the cursor then a bare op; branch-local cursor escaping).

### 2.3 Topology (§7)

Existing: Tensa regression, concat compat, multi-statement branches, projected residual.
Add: many-branch split, nested split, split inside residual, residual inside split, branch
referencing an outer named value, `mean` merge, stochastic residual body, stateful residual body,
spatial (not channel) residual mismatch. For every split test, assert the §7.1 invariant directly
on IR regions (all branch entry nodes consume the split input id). Add that assertion to the
existing Tensa regression too — today it checks structure indirectly.

### 2.4 Parameter identity (§8)

Existing: 6 tests. Add stage reuse across residuals, inside nested blocks, across multiple model
outputs, inside `scan`, referenced by lifecycle regions; plus shadowing/rebinding twins. Each
asserts exact `mod.params` ownership strings, not just counts.

### 2.5 Static repetition (§9)

Existing: 3 tests. Add zero/one repetition, nested, index-dependent dims, shape-changing body,
parameterless body, stochastic body, stateful body, repetition inside and around branch/residual.
The stateful-body case must assert *N independent state slots* for `for N: Stage()` and *one* for
a bound stage.

### 2.6 Multi-stream (§10)

Existing: cross-attention example, tuple destructuring. Add two-independent-streams-then-join,
branch result reused later, several objective operands. **Write U-Net as the tier-2 challenge
here** with a full §4 record — the protocol names it as the ceremony test. Classify every line of
ceremony before any syntax proposal.

### 2.7 Objectives (§16)

Existing: 3 tests. Add reconstruction, weighted multitask, auxiliary classifier, regularisation
term, contrastive, triplet, masked sequence loss, loss on intermediate representation, two models
in one objective. Twin for each: wrong kind (AXS0409), wrong shape (AXS0603), and the
"shapes happen to match but no binding was written" case — must be AXS0301/0604, never inferred.

---

## 3. Milestone 2 — Tensor completeness (§45)

Gate: catalog ops in the list below have a tensor-level reference implementation in TENSA, and
metamorphic tests show equal shapes and (under controlled weights) equal outputs.

> **Status (2026-09-17): DONE.** Delivered as `src/lang/challenges/tensor.ts` (6 challenges,
> 18 twins, 8 metamorphic pairs — manual MHA, cross attention with padding mask, RMSNorm,
> LayerNorm, cosine, causal-flag-vs-mask, slicing, head split via fn vs inline — compared
> numerically against the catalog ops under the seeded parameter init) + 5 regression tests in
> `tests.ts`. Findings F-011 (slice/index bounds → AXS0410 + `≤` constraints), F-012 (mask
> contract), F-013 (optional ports slid), H-005 (emitter `-1` / dropped mask port), E-007
> (merge-without-transpose is statically invisible) in `hardening/findings.md`; tier-2 record
> `hardening/records/mha-explicit.md`. **No tensor-level code needed a `custom op`** (`noCustomOps`
> check on every tensor challenge — the §11 claim holds). 358/358 tests.

- Explicit multi-head attention (§12) — extend `attention-math` example to include head
  split/merge with symbolic `DH = D / Heads`, causal and padding masks, cross-attention inputs.
  Metamorphic pair against catalog `attention` using `explicit-param` weights set to known values.
- RMSNorm, LayerNorm, SwiGLU/GeGLU, cosine similarity, custom masking, small routing fn (§11).
  Metamorphic pair per op against the catalog entry where one exists.
- Reshape/transpose/slicing torture with symbolic dims (§5.3 subset).
- Record any place where tensor-level code was forced into `custom op` — that is a finding
  (likely G or H), and §11 says it must not happen "merely because it is mathematically detailed."

---

## 4. Milestone 3 — Learning-program completeness (§46)

> **Status (2026-09-17): DONE.** Delivered as `src/lang/challenges/learning.ts` (16 challenges,
> 22 twins: tabular / vision / padded-LM / contrastive data contracts; state zoo, shared stage,
> state in `scan`, EMA teacher; `fn` effects and the three-way gradient diagnosis; baseline
> ceremony, transfer, GAN, curriculum, distillation, self-training lifecycles) + 4 regression
> tests in `tests.ts`. Findings F-014…F-022 (frozen-only update, one declaration two aliases →
> AXS0706, `fn` effect union, unapplied optimizer/loss → AXS0708, unknown `until` metric →
> AXS0709, unread model input → AXS0305, leakage location, earlier-field references, region-less
> optimizer = `*`), H-001 **closed** (the reference loop now simulates schedules, `until`,
> events, region lr, clipping and tracked shadows; `RunReport.phases` / `.events`), H-006–H-009
> (emitted objective ports, nested bindings `D(G(z))`, phase-vs-plan event counters, data-op
> catalog gaps), E-008 (contract codes for data↔model refutations), E-009 (`ema(r)`: `r` is the
> decay), E-010 (AXS0901 is info) in `hardening/findings.md`; records in `hardening/records/`
> (per finding + `tabular-pipeline`, `ema-teacher`, `grad-three-way`, `transfer-lifecycle`,
> `curriculum-events`, `escalation-ladder`). Escalation ladder §26: items 1–6 written, 7–9
> recorded as gaps; items 4, 5 and 9 share one missing temporal concept → **G-cand-003 derived
> parameter sets**, §43 analysis in `hardening/proposals/derived-parameter-sets.md`, no syntax.
> G-cand-002 (gradient as value) now has three witnesses. All 16 learning programs and 20
> examples execute as emitted PyTorch under torch 2.12 (one expected failure: the user-supplied
> backend fn in `grad-three-way`). 467/467 tests.

### 4.1 Data (§18–§21)

- Vision pipeline: exists; add the five §18 twins (fit on val → AXS0620; stochastic eval →
  AXS0621; wrong layout / wrong channels → AXS0602; target kind → AXS0603).
- **Tabular pipeline (§19): new challenge.** Uses `impute`, `standardize`, `vocab`, `encode`,
  `select` — all present in the catalog but unexercised. Twins: imputation fitted on full set,
  unknown category unhandled.
- LM pipeline (§20): `gpt` example covers the shift. Add variable-length + padding mask variant.
- Contrastive (§21): existing example asserts two independent stochastic applications? Verify;
  if the IR does not distinguish `augment(x); augment(x)` from `let v = augment(x); v; v`, that is
  an F-class bug against the effects model.

### 4.2 State (§14)

Existing: batchnorm, explicit state, checkpoint slots. Add EMA value, counter, memory bank,
prototype vector, several states in one model, shared stage owning state, state inside `scan`.
**EMA teacher challenge (§14)**: student + EMA copy. This is the first test of whether model-sized
state is expressible; expect a G-candidate finding and record it — do not add syntax.

### 4.3 Effects (§15)

Add compositional propagation tests: a pure `fn` calling each effectful thing, asserting the
union appears on the caller. Assert the three-way distinction `nondiff` / `grad-stopped` /
AXS0901 on argmax, `stop_grad`, and a backend-unsupported op respectively.

### 4.4 Lifecycle (§22–§26)

- Baseline ceremony (§22): assert the derived default phase covers zero-grad/forward/loss/
  backward/step/eval-context/clipping/checkpoint by inspecting `IRPlan` — and that emitted
  PyTorch contains each.
- Transfer (§23): exists; add the four twins (AXS0701, 0703, 0704, conflicting phase config —
  the last may need a new code; record before adding).
- GAN (§24): exists; assert `IRUpdate.times` is data, and add "two objectives sharing a model".
- Curriculum/events (§25): `until`, `every`, schedule transitions. Note `LIMITATIONS` says the
  reference runtime records but does not honour schedules/`until` — that is an **H** item; log it
  in the ledger and fix in `exec.ts` (runtime-only change, no IR/grammar change).
- Escalation ladder (§26): attempt items 4–9 in order (EMA teacher, distillation, pseudo-
  labelling, adversarial training, RL rollout/update, meta-learning). Each gets a record; stop
  and write the §43 analysis only when ≥2 failures share a missing temporal concept.

---

## 5. Milestone 4 — Research pressure (§47)

> **Status (2026-09-18): required attempts complete; boundaries recorded.** `research.ts`
> adds ten challenges; the existing distillation challenge is extended in place. Nine compiling
> programs (including distillation), WGAN-GP/MoE expected rejections, MAML skipped under the §13
> condition. MoCo is explicitly an independent-key/queue subset, not a working EMA encoder;
> RL is fixed-trajectory learning, diffusion a noise-prediction training kernel. Historical
> contrastive examples are alignment-only, not full SimCLR. F-023…F-030 and H-010…H-011 fixed;
> E-011…E-013 and G-cand-004…005 recorded. §43 differentiation/layout/cursor analyses, MoCo BN
> update to derived sets, seven-case cond disposition, report-m4.md and revalidation record delivered.
> Gate: **578/578 tests**, typecheck/build, and emitted-code forward/gradient/state/plan validation
> on **torch 2.11.0+cu128 / RTX 4090** (13 graphs, 7 plans). No grammar changes, no commits.
>
> **Execution requirement (user, 2026-09-18): GPU is the default; no silent CPU fallback.**
> Generated models/plans select CUDA/MPS or fail. Explicit `device cpu` / Python `device="cpu"`
> is required for CPU. The Node/browser reference interpreter is a labeled CPU validation oracle.
> Reproduce with `npx tsx hardening/validate-m4.ts` then `<CUDA-python> hardening/validate-m4.py`;
> `--device cpu` is an explicit harness exception, not the default gate.

Architecture ladder tiers 2–6 as `Challenge` entries with records. Suggested order, cheapest
first within the constraint that each stresses something new:

1. DenseNet (tier 2) — long-lived concat chains, symbolic channel sums.
2. ViT (tier 3) — patchify via reshape/transpose torture, class token concat, positional.
3. Encoder–decoder Transformer (tier 3) — cross-attention across two graphs.
4. VAE (§17, tier 4) — multiple encoder outputs, stochastic sampling, weighted KL+recon.
5. LoRA (tier 4) — frozen base + explicit low-rank params + lifecycle region over adapters only.
6. Distillation (tier 4) — two models in one objective, `stop_grad` on the teacher.
7. SimCLR full / MoCo (tier 4) — MoCo needs a memory bank + EMA encoder: state torture.
8. WGAN-GP (tier 5) — **first hard differentiation pressure (§13)**; expect failure; record
   whether `grad` belongs to tensor, lifecycle, or escape hatch. Do not add `grad` yet.
9. EMA teacher/student, tiny diffusion loop (tier 5).
10. MoE with top-k routing (tier 6) — first real `cond`/dynamic pressure (§29).
11. Toy RL policy/value (tier 6, §32) — the learning-systems boundary test.
12. MAML-style (tier 6) — only if §13 findings converge on a coherent differentiation model.

Dynamic-control-flow disposition (§29): for each of adaptive computation, early-exit, expert
routing, iterate-until-convergence, beam search, autoregressive decoding, rejection sampling —
record which of {static graph, `where/select`, `scan`, future `cond`, custom region, host}
it lands in. This table is the evidence base for whether `cond` is a G finding.

---

## 6. Milestone 5 — Boundary validation (§48)

- Frontier-style decoder (§31): assemble from existing pieces (embedding, RMSNorm, GQA via
  explicit attention with `KVHeads < Heads`, gated MLP, `for N: block`, tied LM head via a shared
  stage). Then push: rotary at tensor level, KV-cache as persistent inference state, MoE, aux
  routing loss, precision policy. Each push is a record.
- Agent-learning toy (§32): observation → policy → action → *external* env → reward → trajectory
  → update. Only the learning side is TENSA; the record classifies what had to be host-side.
- Distribution thought experiments (§34): for three programs (MLP, Transformer, MoE), write the
  hypothetical DP/TP/PP/EP/sharding/activation-checkpoint plans as prose over the existing IR and
  list exactly which IR fields are missing. No code.

---

## 7. Change-control application (§41–§43)

The taxonomy decides the destination of every finding:

| Class | Destination | Repo action |
|---|---|---|
| F compiler bug | fix now | `analyze.ts`/`parser.ts`/`exec.ts`/`emit_torch.ts` + regression test |
| H capability gap | backend work | `exec.ts`/`tensor.ts`/`emit_torch.ts`; `capabilityOf` table; never grammar |
| A/C/D/E | ledger + review batch | may become F (if a rule is clear and violated) or G (if repeated) |
| B | record as inherent | appears in final report "remains inherently complex" |
| G | §43 analysis document, then language change | only after ≥2–3 distinct challenges; write the §43 template into `hardening/proposals/<name>.md` before touching `ast.ts` |

Known items already classifiable from the current docs:

- Schedules and `until` not simulated → **H**, fix in `exec.ts`.
- `cond` missing → **G-candidate**, evidence to be gathered in Milestones 4–5; do not implement
  before the §29 disposition table exists.
- `grad(...)` missing → **G-candidate**, evidence from WGAN-GP / MAML / gradient-reversal; §13
  says look for a coherent model, not a keyword.
- Distributed policy → deferred (§34 thought experiments only).
- Catalog default init hidden → **C**, already visible in inspect; add the §40 "what
  disappeared" review row, no code.

---

## 8. Reporting cadence (§49)

After each milestone, regenerate `hardening/report-<milestone>.md` with the §49 headings, filled
from `hardening/findings.md` and the records. Update `LIMITATIONS`/`REPORT` in `docs.ts` in the
same change so the Report page never lags the ledger.

---

## 9. Suggested order of work

1. Phase 0 items 1.1–1.4 (runner, corpus module, records dir, taxonomy widen) — half a day of
   plumbing that everything else depends on.
2. Phase 0 items 1.5–1.7 (diagnostic harness, metamorphic/property helpers, leakage audit).
3. Milestone 1: migrate existing tests into `Challenge` form where they have a natural twin;
   write the missing microtests; U-Net record.
4. Milestone 2: explicit attention metamorphic pair first (highest-leverage claim in the language).
5. Milestone 3: tabular pipeline, EMA teacher, `exec.ts` schedule/`until` fix, escalation ladder.
6. Milestone 4 in the listed order; stop to write §43 proposals only when the ledger forces it.
7. Milestone 5 thought experiments; final report.
