# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
  the same suite: `corpus.ts` (challenge programs + "evil twin" mutants with expected `AXS` codes),
  `driver.ts` (expands each challenge into per-layer test cases: compile, leakage, parameter
  identity, IR, inspect, emit, run, checkpoint), `metamorphic.ts` (equivalent-program pairs),
  `property.ts` (seeded generators). Adding a challenge = adding an entry to `CHALLENGES`.
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
  → exec.ts      runProgram(mod)   — reference interpreter (forward pass + training loop)
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

- Diagnostic codes (`AXS####`) in `types.ts` are stable identifiers — never renumber or reuse one;
  add new codes at the end of their range.
- `TensorKind` (`Tensor`, `Tokens`, `Class`, `Mask`, `Logits`, `Probs`, `Image`) is a refinement
  system, not a nominal type hierarchy — every kind is assignable to `Tensor`; kinds exist only to
  power specific diagnostics (e.g. softmax-of-probabilities).
- Path alias `@/*` maps to `src/*` (see `tsconfig.json` / `vite.config.ts`).
