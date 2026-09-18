# Tensa

Tensa is an experimental ML-native programming language: a compiler
(lexer → parser → analyzer → IR), a reference interpreter, a PyTorch emitter,
and a browser-based "portal" UI to explore all of it.

There is no server and no external compiler binary — everything (lexer,
parser, type/shape checker, IR, reference backend, PyTorch code generator,
test suite) runs client-side in TypeScript, bundled into a single static
HTML file by Vite.

## Why

Most of the complexity in machine-learning code is not mathematics. It is the
cost of expressing mathematics through a general-purpose host language and a
framework. Tensa keeps the mathematics and deletes the machinery: symbolic
dimensions, transformations, graph topology, parameter identity, persistent
state, objectives, data semantics, differentiation, optimisation, training
lifecycle and execution policy are all first-class, checked citizens of the
language — not framework conventions layered on top of Python.

## Pipeline

```
source text
  → lexer      tokens
  → parser     AST (Program/Decl/Stmt/Expr)
  → analyze    compile(src) -> { mod: IRModule, diags, errors, warnings, ok }
  → ir         IRModule (backend-neutral IR: graphs, params, state, objectives, data, training plan)
  → exec       reference interpreter (forward pass + simulated training loop)
  → emit_torch PyTorch source (text)
```

## Getting started

```
npm install
npm run dev         # start the Vite dev server (the portal UI)
npm run build        # type-checked production build -> single-file dist/index.html
npm test              # headless test suite (467 cases; exit code 1 on failure)
npm run typecheck    # tsc --noEmit
```

Open the dev server and use the **Playground** page to write Tensa source and
inspect its compiled IR, run it against the reference backend, or emit
PyTorch. The **Language**, **vs PyTorch**, **Hardening**, **Diagnostics**,
**Tests** and **Report** pages document the design, compare it against
hand-written PyTorch, and surface the adversarial hardening corpus described
below.

## Hardening

`tensa_hardening_plan.md` is the adversarial validation protocol;
`tensa_hardening_execution.md` is the execution plan against it.
`hardening/findings.md` is the running ledger of what broke and how it was
fixed, with one record per finding under `hardening/records/`. The corpus
itself lives in `src/lang/challenges/` and runs as part of the normal test
suite (`npm test`).

## License

MIT — see [LICENSE](LICENSE).
