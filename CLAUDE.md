# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TENSA is an experimental ML-native programming language: a compiler (lexer → parser → analyzer →
IR), a reference interpreter, a PyTorch emitter, and a browser-based "portal" UI to explore all of
it. There is no server and no external compiler binary — everything (lexer, parser, type/shape
checker, IR, reference backend, PyTorch code generator, test suite) runs client-side in TypeScript,
bundled into a single static HTML file by Vite.

## Commands

```
npm run dev         # start Vite dev server
npm run build       # type-checked production build -> single-file dist/index.html (vite-plugin-singlefile)
npm run preview     # preview the production build
npm test            # headless test suite (tsx src/lang/run_tests.ts); exit code 1 on failure
npm test -- shapes  # filter by test group name or by substring of a test name
npm run typecheck   # tsc --noEmit (strict, noUnusedLocals/noUnusedParameters)
```

There is no ESLint config and no Jest/Vitest. The test suite is hand-rolled:

- **[src/lang/tests.ts](src/lang/tests.ts)** exports `runTests()` returning `TestResult[]`. It is
  driven three ways: `npm test` (node, via `tsx`), the app's **Tests** page, and the CLI tab's
  `test` command. When changing compiler/backend behavior, add cases to the relevant group there.
- **[src/lang/challenges/](src/lang/challenges/)** is the adversarial hardening corpus wired into
  the same suite: `corpus.ts` (base challenge programs + "evil twin" mutants with expected `AXS`
  codes, plus `METAMORPHIC` pairs; `CHALLENGES = [...BASE, ...SEMANTIC, ...TENSOR_CHALLENGES,
  ...LEARNING_CHALLENGES]`), `semantic.ts` (Milestone 1 semantic-core challenges: algebra, cursor
  rules, topology, identity paths, repetition, U-Net, objectives), `tensor.ts` (Milestone 2
  tensor completeness: explicit MHA vs catalog `attention`, cross attention with padding masks,
  norms/gates, slicing torture, plus manual-vs-catalog metamorphic pairs compared numerically —
  the seeded parameter init makes two programs with the same parameter sequence comparable via
  `assertEquivalent(..., {ir: false, run})`), `learning.ts` (Milestone 3 learning-program
  completeness: data contracts — tabular, vision, padded LM, contrastive; state zoo, shared
  stage, state in `scan`, EMA teacher; `fn` effect union and the three-way gradient diagnosis;
  baseline / transfer / GAN / curriculum / distillation / self-training lifecycles, checked
  through `RunReport.phases` and `.events`), `driver.ts` (expands each challenge into per-layer
  test cases: compile, leakage, parameter identity, IR, inspect, emit, run, checkpoint; the
  `Expectation` type lists every checkable field — `outputShape`, `paramOwners`, `irOps`,
  `emitContains`, `run.refuses`, `warnCodes` (warnings only; info-level codes such as AXS0901
  need a `custom` check), `checkpointKinds` (`"parameters"`, `"persistent state"`, …), `custom`,
  …), `metamorphic.ts`, `property.ts`, `ops.ts`
  (per-op shape table; every catalog op must appear or `uncoveredOps` fails). Adding a challenge
  = adding an entry to one of the arrays; test group names are `challenge:<id>`.
- **[hardening/](hardening/)** holds the human-readable side: `findings.md` (ledger, IDs like
  `F-001`), `records/` (per-challenge records, §4 template), `proposals/` (language-change
  proposals only). The protocol is `tensa_hardening_plan.md` (do not edit — it is the spec); the
  execution plan against it is `tensa_hardening_execution.md`.

## Architecture

### Pipeline

Everything funnels through one compile function and one command dispatcher, both backend-neutral:

```
source text
  → lexer.ts     tokens
  → parser.ts    ast.ts (Program/Decl/Stmt/Expr)
  → analyze.ts   compile(src) -> { mod: IRModule, diags, errors, warnings, ok }
  → ir.ts        IRModule (backend-neutral IR: graphs, params, state, objectives, data, training plan)
  → exec.ts      runProgram(mod)   — reference interpreter (forward pass + simulated training loop:
                  schedules, `until`, events, region lr, clipping, tracked shadows; the RunReport
                  carries per-phase `stoppedBy` and an event log; one epoch = STEPS_PER_EPOCH steps)
  → emit_torch.ts emitTorch(mod)   — generates PyTorch source as text
```

`analyze.ts` (~2100 lines) is the core: it does name resolution, symbolic shape/dim inference
(via `dims.ts`'s `DimExpr` algebra), semantic-kind checking (`Tensor`/`Tokens`/`Logits`/`Probs`/...
in `types.ts`), stage/parameter identity and sharing, effect tracking, and data/objective/lifecycle
contract checking, producing the diagnostics catalogued in `types.ts` (`DIAGNOSTICS`, stable
`AXS####` codes).

`cli.ts` defines `executeCommand(cmd, src)` — the single entry point for every user-facing
operation (`check`, `inspect`, `ir`, `emit`, `run`, `test`, `examples`, `catalog`, `codes`,
`help`). Both the in-app CLI tab and (by design) any future standalone CLI/node driver are meant
to call exactly this function, so command behavior should be added/changed here, not duplicated in
the UI.

### Key modules

- `catalog.ts` — the standard library of tensor operations (`linear`, `conv`, etc.): shapes,
  required/optional args, effects. This is library vocabulary, not language semantics — adding an
  op means extending this catalog, not the grammar.
- `dims.ts` — symbolic dimension expressions and the algebra used to prove (or fail to prove)
  shape equalities at compile time.
- `tensor.ts` — the actual runtime tensor implementation used by the reference backend (`exec.ts`).
- `inspect.ts` — turns a compiled `IRModule` into a human-readable structural report (graph,
  shapes, parameter/sharing table, state, effects, lifecycle, checkpoint coverage) — used by both
  `cli.ts`'s `inspect` command and `InspectView` in the UI.
- `examples.ts` — bundled TENSA programs shown in the Playground's example picker; each can carry
  `friction` annotations (`FindingKind` A–H from `docs.ts`) used by the Hardening page to argue where the language design
  succeeded or leaked.
- `docs.ts` — long-form Markdown content backing the Learn/Compare/Hardening/Report pages (not
  code documentation — it's rendered UI content).

### UI

`App.tsx` is a single-file React app with an in-memory `Page` switch (Playground, Language,
vs PyTorch, Hardening, Diagnostics, Tests, Report) — there is no router. The Playground page is
the primary surface: an editor bound to `compile(source)`, with tabs (`check`/`inspect`/`ir`/
`emit`/`run`/`cli`) that all read from the same memoized `compile` result. `components/panels.tsx`
holds the larger view components (`Editor`, `DiagnosticList`, `InspectView`, `RunView`,
`TestsView`); `components/ui.tsx` holds small presentational primitives (`Panel`, `Tag`,
`CodeBlock`, `Markdown`).

### Conventions worth knowing

- **GPU execution is required by default (2026-09-18).** Generated PyTorch models/plans select
  CUDA (or MPS) and fail clearly if no GPU exists; never silently fall back to CPU. CPU requires
  explicit `device cpu` in a plan or `device="cpu"` at the generated Python entry point.
  The browser/Node reference interpreter is explicitly a CPU-only validation oracle, not a GPU run.
  M4 fidelity gate: `npx tsx hardening/validate-m4.ts`, then run `hardening/validate-m4.py` with a
  CUDA-enabled Python. The harness defaults to GPU; `--device cpu` is an explicit exception.
- M4 corpus: `research.ts` exports `RESEARCH_CHALLENGES`, appended in `corpus.ts`; records and
  `hardening/report-m4.md` distinguish compiling subsets from unsupported complete algorithms.
  `randn_like` is catalog vocabulary: independent standard-normal draws in train AND eval,
  no gradient to the template. Loss bindings projecting the same tuple call share one forward
  within that loss evaluation (F-023), never across updates or separate losses.

- Diagnostic codes (`AXS####`) in `types.ts` are stable identifiers — never renumber or reuse one;
  add new codes at the end of their range.
- `TensorKind` (`Tensor`, `Tokens`, `Class`, `Mask`, `Logits`, `Probs`, `Image`) is a refinement
  system, not a nominal type hierarchy — every kind is assignable to `Tensor` and plain `Tensor`
  is assignable to every kind; kinds exist only to power specific diagnostics (e.g.
  softmax-of-probabilities).
- Shape equality (`eqDim` in `analyze.ts`) has four outcomes: proved, refuted (constant or
  same-sign polynomial difference → AXS0401), carried (AXS0403, stored as an `assumed` constraint
  that `exec.ts` verifies against the dim bindings before any forward pass), and unknown
  (`TensorType.unknown` / `Tensor[?]` — nothing downstream is sized or contradicted).
  Bounds (`leDim`, used for slice/index checks) follow the same discipline; a carried bound is a
  `Constraint` with `rel: "<="`, printed `≤`, refuted as AXS0410.
- Mask contract: TENSA `Mask` true = *blocked*. `attention(mask:)` accepts `Mask[Tq, Tk]` or
  `Mask[B|1, Tq|1, Tk]` (padding masks are written `Mask[B, 1, S]`); the runtime lifts rank 3 to
  `[B, 1, Tq, Tk]` and the emitter negates (`attn_mask=~m`) because SDPA's bool mask means
  *allowed*. Layer-style optional ports keep their positional slot — a skipped one defaults to the
  port before it (key ← query, value ← key).
- The emitter must never print a `DimExpr` with `show()` into executable Python; use `pyDim`
  (symbolic extents become expressions over the dims dict, derived dims are computed by
  `resolve_dims`). The generated `forward` first binds plain runtime dims from the input shapes
  (`dims = resolve_dims({**self.dims, "B": x.shape[0], …})`); every reshape/slice/mask extent
  goes through `dr()` — never `-1`.
- Lifecycle conventions shared by `exec.ts` and `emit_torch.ts` (keep them in step): an optimizer
  without `over` has region `*` = every parameter of every model bound in the plan (emitted
  parameter lists are grouped by owning alias); phase-level `every N steps` counts steps within
  the phase (`phase_step`), plan-level counts steps across the run (`step`); `ema(r)` keeps
  fraction `r` of the previous value (`emaDecay` helper); loss bindings such as `D(G(z))` lower
  recursively so every model on the path is on the tape; two plan aliases of one `model`
  declaration are one parameter set (AXS0706).
- Data plane: a field pipeline may start from a field declared earlier in the same `example`
  (lowered to op `field.<name>`); the field's declared type is the contract, `DataOpSpec`
  shape/kind metadata is documentary. Fitted statistics carry the op's location so AXS0620
  points at the operation.
- Hardening ledger IDs (`F-###`, `H-###`, `E-###`, `G-cand-###`) are allocated in
  `hardening/findings.md`; every fixed F/H gets a `hardening/records/<id>.md` and a regression
  test that cites the ID in its name.
- Path alias `@/*` maps to `src/*` (see `tsconfig.json` / `vite.config.ts`).
