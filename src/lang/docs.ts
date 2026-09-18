/**
 * TENSA — documentation content (rendered by the portal).
 * Markdown-lite: `## heading`, `- bullet`, ```code fences```, plain paragraphs.
 */
export interface DocSection {
  id: string;
  title: string;
  blurb: string;
  body: string;
}

export const DOCS: DocSection[] = [
  {
    id: "intro",
    title: "What TENSA is",
    blurb: "One language for architecture, tensor mathematics, data, objectives, state and lifecycle.",
    body: `TENSA is an ML-native programming language. It consolidates three earlier prototypes — Vect
(implicit sequential flow, first-class graph topology), Axon (explicit layer contracts) and Lumen
(compile-time vs runtime separation, stage identity, effects) — into one coherent design.

The hypothesis is simple:

> Most of the complexity in machine-learning code is not mathematics. It is the cost of expressing
> mathematics through a general-purpose host language and a framework.

TENSA keeps the mathematics and deletes the machinery. The recurring semantic concepts of ML are
first-class: symbolic dimensions, transformations, graph topology, parameter identity, persistent
state, objectives, data semantics, differentiation, optimisation, training lifecycle and execution
policy.

## The stack

\`\`\`
data semantics → example construction → architecture → blocks
    → tensor functions → tensor primitives → backend-neutral IR → backend
\`\`\`

Training and lifecycle semantics operate *across* that stack rather than sitting outside it in a
host-language loop.

## The two rules that decide everything

- **Remove accidental complexity, not mathematical complexity.**
- **Require the programmer to specify information when it represents a choice. Infer it when it is a
  consequence.**

\`linear(256)\` states a choice. The input width 784 is a consequence, so it is never written. A
projection on a residual skip path is a choice, so the compiler will never invent one.`,
  },
  {
    id: "principles",
    title: "Design principles",
    blurb: "Nine commitments that the implementation is held to.",
    body: `## 1. Choices are written; consequences are inferred
Output widths, channel counts, kernel sizes, head counts, loss weights, update ratios and
augmentation policy are choices. Input widths, flattened sizes, spatial extents after convolution,
parameter shapes and gradient wiring are consequences.

## 2. Static where possible, dynamic where necessary
Shapes, parameter tables, topology, effects, objective graphs and the training plan are compile-time
knowledge. Tensor values, batch size, device, precision, stochastic outcomes and optimizer evolution
are runtime.

## 3. Two levels, one language
Architecture level (\`attention(heads: 8)\`) and tensor level (\`matmul\`, \`transpose\`, \`softmax\`)
lower into the same IR. You can descend without leaving the language.

## 4. Parameter identity follows value identity
Reusing a bound stage shares parameters. Recreating it does not. No registration, no ModuleList, no
string names.

## 5. Topology is explicit in the IR
\`residual\` and \`split … merge\` are not sugar: they produce composite IR nodes with regions plus an
explicit merge node. Inspection is generated from the same IR that executes.

## 6. Effects are part of the type of a program
Pure, stochastic, stateful, parameterised, training-sensitive, non-differentiable and
gradient-stopped are distinct and tracked.

## 7. Backend limitations are never language semantics
Convolution is differentiable. If a backend lacks a gradient rule, that is a *capability* report
(AXS0901), not a claim about mathematics.

## 8. Data transformations that change what is learned are part of the program
Normalisation statistics, vocabularies, augmentation, windowing and target shifting live in the
language, with leakage checks. Credentials, drivers, file formats and worker pools do not.

## 9. Diagnostics are a feature
Stable codes, source locations, architectural location, the axis at fault, the expected and inferred
expressions, and a suggested repair.`,
  },
  {
    id: "types",
    title: "Dimensions and types",
    blurb: "Symbolic shapes, dimension algebra, semantic tensor kinds.",
    body: `## Declaring dimensions

\`\`\`
dim B                 # free at compile time — bound at runtime (batch size)
dim D = 512           # static
dim Heads = 8
dim DH = D / Heads    # symbolic algebra, proved exact: 64
\`\`\`

A dimension expression is a canonical polynomial over dimension atoms, so \`D*4\` and \`4*D\` are the
same expression and \`(D/Heads)*Heads\` simplifies to \`D\` when the division is exact. Non-exact
division produces a floor-division atom that survives symbolically (\`⌊(H + 2 - 3)/2⌋ + 1\`).

## Tensor types

\`\`\`
Tensor[B, T, D]      Scalar         Tokens[B, T]      Class[B]
Image[B, 3, H, W]    Mask[T, T]     Logits[B, K]      Probs[B, K]
\`\`\`

Semantic kinds are *refinements*, not a nominal hierarchy: everything is a tensor, and a plain
\`Tensor\` is accepted anywhere. They exist because they buy real diagnostics:

- \`cross_entropy(Probs, …)\` → AXS0409 (softmax applied twice — a classic silent bug)
- \`embedding(Tensor)\` → AXS0409 (index semantics not proved)
- \`softmax\` maps \`Logits → Probs\`

## Proving, unifying and carrying constraints

\`\`\`
model M(a: Tensor[B, S, D], b: Tensor[B, T, D]) -> Tensor[B, S, D] {
  return a + b     # AXS0403: cannot prove S = T
}
\`\`\`

The checker never assumes two differently named dimensions are equal. It either

1. **proves** the equality symbolically,
2. **unifies** it, when one side is a declaration-local template variable, or
3. **carries** it as an assumption recorded in the IR and reported as AXS0403.

Declaration-local dimensions are template variables bound at the call site:

\`\`\`
block ResBlock(x: Tensor[B, C, H, W], out: Dim = 64, stride: Dim = 1) { … }
\`\`\`

\`C\`, \`H\`, \`W\` are bound by the incoming tensor; \`out\` and \`stride\` are compile-time configuration
usable inside shape expressions.`,
  },
  {
    id: "flow",
    title: "Flow: implicit, piped, named, multi-stream",
    blurb: "One rule for the implicit value, and an explicit discipline for several streams.",
    body: `## The cursor rule

> **The implicit value is the value produced by the most recent statement. A bare transformation
> consumes it and produces the next one.**

At the start of a declaration the cursor is seeded **only if there is exactly one tensor input**.

\`\`\`
model MLP(x: Tensor[B, 784]) -> Logits[B, 10] {
  linear(256)      # consumes x
  gelu             # consumes the previous value
  dropout(0.1)
  linear(10)
}
\`\`\`

Pipelines express the same thing inside an expression: \`linear(256) |> gelu |> dropout(0.1)\`.

## Naming

Use \`let\` when identity matters — when a value is reused, branched, returned or inspected:

\`\`\`
let features = linear(128) |> gelu
let a = features |> linear(10)
let b = features |> linear(1)
return (a, b)
\`\`\`

Ordinary graph edges stay anonymous. Meaningful ones get names.

## Several inputs

With more than one input there is **no** implicit stream, and a bare transformation is an error
(AXS0301) instead of a guess:

\`\`\`
model Fusion(text: Tensor[B, T, D], image: Tensor[B, S, D]) -> Tensor[B, D] {
  let t = text  |> layernorm
  let i = image |> layernorm
  let fused = attention(query: t, key: i, value: i, heads: 8)
  return mean(fused, axis: 1)
}
\`\`\`

Every multi-input operation names its ports (\`query:\`, \`key:\`, \`value:\`, \`mask:\`). Any explicit
join (\`concat\`, \`add\`, a stage application) re-establishes a single stream, and implicit flow
resumes from there.

## Several outputs

Tuple results and destructuring:

\`\`\`
model MultiTask(x: Tensor[B, 128]) -> (Logits[B, 10], Tensor[B, 1]) {
  let h = linear(256) |> gelu
  return (h |> linear(10), h |> linear(1))
}

let (logits, value) = MultiTask(x)
\`\`\`

Objectives consume them by position or by port name: \`Value(pred: Net(x)[1], target: value)\`.`,
  },
  {
    id: "topology",
    title: "Topology: residual, split/merge, repetition",
    blurb: "Branching is a language construct with explicit IR, not a sugar for sequential code.",
    body: `## Residual

\`\`\`
residual {
  layernorm
  attention(heads: Heads, causal: true)
  dropout(0.1)
}
\`\`\`

If the body changes the shape, the compiler reports AXS0404 rather than inserting a learned
projection. Projections are explicit:

\`\`\`
residual via conv2d(128, kernel: 1, stride: 2) {
  conv2d(128, kernel: 3, stride: 2, pad: 1)
  batchnorm
  relu
  conv2d(128, kernel: 3, pad: 1)
  batchnorm
}
\`\`\`

Lowering is a composite \`residual\` node with a \`body\` region (and a \`projection\` region when
present) followed by an **explicit \`add\` node**.

## Parallel split and merge

\`\`\`
split merge concat(1) {
  conv2d(32, kernel: 1)
  conv2d(32, kernel: 3, pad: 1)
  conv2d(32, kernel: 5, pad: 2)
  branch pooled {                    # multi-statement branch
    maxpool2d(3, stride: 1, pad: 1)
    conv2d(32, kernel: 1)
  }
}
\`\`\`

Semantics:

- every branch receives **the same** incoming value,
- every branch has its **own** implicit cursor and its own scope,
- the merge (\`concat(axis)\`, \`add\`, \`mean\`) is an explicit IR node consuming every branch result.

A branch is either a single statement (typically a pipeline) or a labelled \`branch name { … }\` block.
The Vect prototype once lowered these sequentially; a dedicated regression test asserts that all
branch entry nodes consume the split input and that the merge node has one input per branch.

## Static repetition

\`\`\`
for 12: TransformerBlock()      # 12 independent parameter sets
\`\`\`

\`\`\`
let block = TransformerBlock()
for 12: block                   # ONE parameter set applied 12 times
\`\`\`

\`\`\`
for i in 0..3 { linear(16 * (i + 1)) }   # index available when it matters
\`\`\`

The IR records \`static_repeat[count, mode]\` where mode is \`independent-parameters\` or
\`shared-stage\`; \`inspect\` prints it. Static repetition is *not* recurrence — see \`scan\`.`,
  },
  {
    id: "params",
    title: "Parameters, sharing and state",
    blurb: "Value identity decides sharing; state is a first-class kind, not hidden backend memory.",
    body: `## Implicit parameters

Catalog operations own their parameters. \`linear(256)\` owns \`w\` and \`b\`; \`attention\` owns
\`wq/wk/wv/wo\`; \`layernorm\` owns \`gamma/beta\`.

## Explicit parameters

\`\`\`
param W: Tensor[D, D] init: xavier
param temperature: Scalar init: ones
frozen param prior: Tensor[K] init: zeros
\`\`\`

The IR records owner, role, shape, initialisation, trainability, sharing and optimizer membership.
There is no registration step because registration is not a mathematical concept.

## Identity and sharing

> Reusing the same configured stage value shares its parameters. Recreating the stage creates
> independent parameters.

\`\`\`
let encoder = Encoder()          # one stage value
return (encoder(a), encoder(b))  # one parameter set, two applications
\`\`\`

\`\`\`
return (Encoder(a), Encoder(b))  # two stage values, two parameter sets
\`\`\`

\`inspect\` prints applications per table and the values that were *not* duplicated. There is no
\`share\` keyword because bindings already carry the meaning.

## Persistent state

Four kinds of state are distinguished:

- **parameter** — trainable, owned by a stage
- **persistent state** — non-parameter algorithmic memory (running statistics, EMA, counters, banks)
- **optimizer state** — moments, step counters, schedule position
- **temporary tensor** — graph values, never checkpointed

\`\`\`
model Centered(x: Tensor[B, D]) -> Tensor[B, D] {
  state center: Tensor[D] init: zeros update: ema(0.9)
  let feats = linear(D) |> gelu
  let c = observe(center, mean(feats, axis: 0))   # read + scheduled update
  return feats - c
}
\`\`\`

\`observe\` reads the current value and schedules the declared update; it updates in training context
and is read-only in evaluation context. \`batchnorm\` uses exactly the same mechanism internally.
Everything with a state kind appears in the checkpoint coverage report.

## Effects

\`pure\`, \`stochastic\`, \`reads-state\`, \`writes-state\`, \`parameterized\`, \`training-sensitive\`,
\`nondiff\` (no gradient exists), \`grad-stopped\` (the program cut it). Backend gradient gaps are a
separate capability table and are reported as AXS0901.`,
  },
  {
    id: "objectives",
    title: "Objectives",
    blurb: "Typed computation graphs from predictions and targets to a scalar.",
    body: `An objective is an ordinary typed graph:

\`\`\`
objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}

objective Reconstruction(pred: Tensor[B, D], original: Tensor[B, D]) -> Scalar {
  return mse(pred, original)
}
\`\`\`

There is no \`objective clf = cross_entropy(Model)\` form: the relationship between predictions and
targets is information, and it is written down.

Objectives support several model outputs, several targets, auxiliary terms, regularisation and
arbitrary tensor mathematics:

\`\`\`
objective Total(logits: Logits[B, K], labels: Class[B], code: Tensor[B, Z]) -> Scalar {
  return cross_entropy(logits, labels) + 0.001 * l2(code)
}
\`\`\`

The plan binds ports to model applications and data fields, and the compiler checks both contracts
(AXS0602 for model inputs, AXS0603 for objective targets):

\`\`\`
loss cls = Classification(logits: Net(x)[0], labels: label) weight: 1.0
loss val = Value(pred: Net(x)[1], target: value) weight: 0.3
\`\`\`

Language modelling makes the shift explicit in the data declaration, never in a dataset name.`,
  },
  {
    id: "data",
    title: "Data semantics",
    blurb: "Source adapters stay outside; everything that changes what is learned stays inside.",
    body: `## The boundary

**Outside the language** (source adapters): HTTP, credentials, database clients, file formats,
decoding libraries, worker processes, sharding.

**Inside the language**: example construction, preprocessing, normalisation statistics, vocabularies,
augmentation policy, split policy, batching, and the mapping to model inputs and objective targets.

> If a data transformation can change what the model learns, it is part of the ML program.

\`\`\`
source Cifar = image_folder(path: "cifar10/", classes: 10)

data VisionData from Cifar {
  example {
    field image: Image[3, 32, 32] = decode(file)
    field label: Class = as_class(folder)
  }
  preprocess { image: resize(32, 32) |> to_float |> normalize(fit: train) }
  augment train { image: random_crop(32, pad: 4) |> random_flip(p: 0.5) }
  augment eval  { image: center_crop(32) }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 64
  shuffle true
}
\`\`\`

- \`fit: train\` is required for any fitted statistic; \`fit: val\` is a leakage error (AXS0620).
- A stochastic transform in an eval pipeline is a warning (AXS0621).
- Tabular work uses the same shape: \`select\`, \`impute(strategy: median, fit: train)\`,
  \`standardize(fit: train)\`, \`vocab(fit: train, max: N)\`, \`encode\`.

## Language modelling

\`\`\`
example {
  field window:  Tokens[T + 1] = tokenize(text, vocab: V) |> window(T + 1, stride: T)
  field inputs:  Tokens[T] = window[0 : T]
  field targets: Tokens[T] = window[1 : T + 1]
}
\`\`\`

## Contrastive / metric learning

Example construction may emit several views of one record; each view is an ordinary field with its
own augmentation pipeline.

## What the compiler checks

\`Data → Example → Batch → Model → Prediction → Objective\` is one chain of contracts. Field shapes
and semantic kinds are checked against model inputs and objective ports before anything runs.`,
  },
  {
    id: "lifecycle",
    title: "Training lifecycle",
    blurb: "Six temporal concepts: step, phase, update, schedule, event, training state.",
    body: `Ordinary supervised training stays short — the mechanics are derived:

\`\`\`
train DigitRun {
  data DigitData
  model Net = MLP
  loss main = Classification(logits: Net(x), labels: label)
  optimizer opt = adamw(lr: 3e-4, wd: 0.01)
  epochs 3
  every 1 epochs { validate ; checkpoint }
}
\`\`\`

Derived automatically: gradient zeroing, forward, loss evaluation, backward, optimizer step,
train/eval context switching, device movement, precision placement, validation, checkpointing and
gradient clipping.

## The temporal vocabulary

The research question is *which* temporal concepts deserve first-class semantics. TENSA promotes six:

- **step** — one optimisation update of one objective by one optimizer set
- **phase** — a named interval with its own freeze set, learning rates, schedule and updates
- **update** — \`update <loss> with <optimizers> times <n>\` (the update ratio is data, not control flow)
- **schedule** — a declarative learning-rate programme (\`cosine(warmup: 20)\`)
- **event** — \`every n steps|epochs { … }\` and \`until <metric> <cmp> <value>\`
- **training state** — optimizer moments, EMA/teacher copies, counters, schedule position

Everything else stays out. There is no general imperative loop in the language.

\`\`\`
train Finetune {
  data VisionData
  model Net = Transfer
  loss main = Classification(logits: Net(image), labels: label)

  optimizer head_opt = adamw(lr: 1e-3) over Net.head
  optimizer enc_opt  = adamw(lr: 1e-5) over Net.encoder

  phase warmup {
    epochs 2
    freeze Net.encoder
    update main with head_opt
  }
  phase finetune {
    epochs 3
    unfreeze Net.encoder
    lr Net.encoder = 0.00001
    update main with head_opt, enc_opt
    every 1 epochs { validate ; checkpoint }
  }
  device auto
  precision mixed
}
\`\`\`

## Parameter regions

\`Net.encoder\` is not a string filter. It resolves structurally against the elaborated stage tree
(the model alias, then let-bound stage names). A region that matches no parameters is an error
(AXS0701/AXS0702), and two optimizers claiming the same parameter inside one phase is AXS0703.

## Alternating optimisation

\`\`\`
phase adversarial {
  epochs 2
  update d_loss with d_opt times 2
  update g_loss with g_opt times 1
}
\`\`\`

## Curriculum

\`phase easy { epochs 2 until val_acc > 0.6 }\` — an event condition on a metric, evaluated between
steps.

## Escape hatch

When an algorithm does not fit the vocabulary, the IR and reference runtime are the documented
escape: a plan may be executed step-by-step by a host driver that keeps the compiler's parameter,
state and objective model. The language does not grow an imperative loop to accommodate it.`,
  },
  {
    id: "dynamic",
    title: "Recurrence and dynamic computation",
    blurb: "scan for temporal computation; a precise statement of what is restricted and why.",
    body: `## scan

\`\`\`
let (states, final) = scan over x axis: 1 carry h: Tensor[B, Hid] init: zeros {
  let combined = concat(step, h, axis: -1)
  yield combined |> linear(Hid) |> tanh
}
\`\`\`

- \`step\` is the slice of the scanned tensor with the scan axis removed.
- \`carry\` declares the recurrent state, its shape contract and its initialisation.
- \`yield\` produces the next carry; the scan result is \`(stacked outputs, final carry)\`.
- The body is elaborated **once**, so parameters inside it are *shared across time steps* — that is
  what recurrence means. The IR node records the axis, the length and the carried contract.

This covers RNNs, iterative refinement, diffusion-style sampling loops and recurrent attention.
\`for 12:\` is a different construct and is never confused with it.

## Dynamic control flow: what is restricted and why

Currently restricted:

- **data-dependent branching on tensor values** inside a graph (\`if tensor > 0 then A else B\`)
- **data-dependent shapes** (a length that depends on runtime values)

Why: the first is an *implementation* restriction — the IR has no \`cond\` node yet — and the second is
*semantic*: static shape contracts are the core value proposition, and a shape that no longer has a
symbolic expression breaks every downstream contract.

What is **not** restricted: branching whose arms have identical static contracts is a legitimate
design and the intended next extension (\`select(cond, a, b)\` exists today as an eager elementwise
form). Nothing in the type system requires banning it.

Escape today: a \`custom op\` may implement the dynamic region, declare its contracts and effects, and
keep static reasoning downstream. If it cannot state its output shape, AXS0903 reports the loss of
knowledge rather than pretending certainty.`,
  },
  {
    id: "custom",
    title: "Custom operations and the catalog",
    blurb: "A missing layer is a library problem, not a compiler problem.",
    body: `The catalog is **library vocabulary**: linear, conv2d, embedding, positional, attention,
layernorm, rmsnorm, batchnorm, dropout, pooling, activations, tensor primitives and losses. Flow,
topology, tensor mathematics, parameter identity, state, objectives, data and lifecycle are
**language semantics**.

Three ways to add an operation without touching the compiler:

1. **A block** — architecture-level composition with its own parameters.
2. **A tensor function** (\`fn\`) — real mathematics, statically shape-checked.
3. **A custom op** — a foreign or experimental kernel:

\`\`\`
custom op flash_attention(q: Tensor[B, H, T, DH],
                          k: Tensor[B, H, T, DH],
                          v: Tensor[B, H, T, DH]) -> Tensor[B, H, T, DH] {
  effects: pure
  backend torch: "flash_attn_func(q, k, v, causal=True)"
}
\`\`\`

A custom op declares input/output contracts, shape semantics, effects, parameter/state ownership,
backend implementations and an optional reference implementation. Loss of static guarantees is
explicit: \`shape: unknown\` produces AXS0903 and downstream shapes become unknown instead of
silently wrong. \`differentiable: false\` is recorded as a *semantic* property, distinct from a
backend that merely lacks a gradient rule.`,
  },
  {
    id: "tooling",
    title: "Tooling",
    blurb: "check, run, inspect, emit, ir, test — and diagnostics as a first-class feature.",
    body: `\`\`\`
axis check     parse + static semantic analysis, no execution
axis inspect   what did the compiler actually build?
axis ir        backend-neutral IR
axis emit      generated PyTorch
axis run       execute with the reference backend
axis test      semantic + regression suite
axis codes     diagnostic catalogue
\`\`\`

\`inspect\` answers one question — *what did the compiler actually build?* — and answers it from the
same IR the backend executes: graph with every intermediate shape, branching and merging made
visually obvious, parameter counts and ownership, sharing, trainable/frozen state, persistent state,
effects, dimension constraints, data/model/objective compatibility, lifecycle plan, checkpoint
coverage and backend capability gaps.

Diagnostics carry a stable code, a source location, the architectural location, the axis at fault,
the expected and inferred expressions and a suggested repair. Warnings (such as a carried dimension
constraint) are not failures: they are the compiler telling you exactly what it could not prove.`,
  },
];

export const COMPARISONS: { id: string; title: string; axis: string; torch: string; note: string }[] = [
  {
    id: "siamese",
    title: "Tied encoder (parameter identity)",
    axis: `block Encoder(x: Image[B, 1, 28, 28]) -> Tensor[B, 64] {
  conv2d(16, kernel: 3, stride: 2, pad: 1)
  relu
  conv2d(32, kernel: 3, stride: 2, pad: 1)
  relu
  global_avgpool
  linear(64)
}

model Siamese(a: Image[B, 1, 28, 28],
              b: Image[B, 1, 28, 28]) -> (Tensor[B, 64], Tensor[B, 64]) {
  let encoder = Encoder()
  return (encoder(a), encoder(b))
}`,
    torch: `class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()   # sharing is a side effect of attribute reuse

    def forward(self, a, b):
        return self.encoder(a), self.encoder(b)`,
    note: `PyTorch expresses sharing correctly here, but only by convention: the guarantee comes from how
you happened to write the attribute. TENSA makes it a checked property — inspect reports "6 tables,
2 applications, 6,912 values not duplicated", and a regression test asserts that recreating the
stage doubles the count. The input channel counts (1, 16, 32) disappear because they are
consequences.`,
  },
  {
    id: "inception",
    title: "Multi-scale branching",
    axis: `split merge concat(1) {
  conv2d(32, kernel: 1)
  conv2d(32, kernel: 3, pad: 1)
  conv2d(32, kernel: 5, pad: 2)
  branch pooled {
    maxpool2d(3, stride: 1, pad: 1)
    conv2d(32, kernel: 1)
  }
}`,
    torch: `self.b1 = nn.Conv2d(16, 32, 1)
self.b2 = nn.Conv2d(16, 32, 3, padding=1)
self.b3 = nn.Conv2d(16, 32, 5, padding=2)
self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(16, 32, 1))
...
def forward(self, x):
    return torch.cat([self.b1(x), self.b2(x),
                      self.b3(x), self.b4(x)], dim=1)`,
    note: `The PyTorch version is not much longer, but the *topology is implicit*: nothing prevents
accidentally writing \`self.b2(y)\`. In TENSA the branch structure is a compiler object — every branch
provably consumes the split input, the merge is an explicit IR node, and inspect draws it. The
in-channel count 16 is written four times in PyTorch and zero times in TENSA.`,
  },
  {
    id: "lifecycle",
    title: "Transfer-learning lifecycle",
    axis: `optimizer head_opt = adamw(lr: 1e-3) over Net.head
optimizer enc_opt  = adamw(lr: 1e-5) over Net.encoder

phase warmup {
  epochs 2
  freeze Net.encoder
  update main with head_opt
}
phase finetune {
  epochs 3
  unfreeze Net.encoder
  lr Net.encoder = 0.00001
  update main with head_opt, enc_opt
  every 1 epochs { validate ; checkpoint }
}`,
    torch: `for p in model.encoder.parameters():
    p.requires_grad_(False)
head_opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
for epoch in range(2):
    for batch in loader:
        head_opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch["image"]), batch["label"])
        loss.backward(); head_opt.step()

for p in model.encoder.parameters():
    p.requires_grad_(True)
opt = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(),    "lr": 1e-3}])
for epoch in range(3):
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch["image"]), batch["label"])
        loss.backward(); opt.step()
        if ...: validate(); torch.save(...)`,
    note: `The PyTorch code is a *transcript* of the algorithm mixed with plumbing. Phase boundaries,
freezing and per-region learning rates are real algorithmic content and stay visible in TENSA; zeroing,
backward, stepping, mode switching and checkpoint writing disappear. The compiler can also check
that \`Net.encoder\` exists, that the two optimizers are disjoint and that frozen regions are not
silently updated — none of which the Python version can do.`,
  },
  {
    id: "attention",
    title: "Descending to tensor mathematics",
    axis: `fn attention_core(q: Tensor[B, H, T, DH],
                  k: Tensor[B, H, T, DH],
                  v: Tensor[B, H, T, DH]) -> Tensor[B, H, T, DH] {
  let scores  = matmul(q, transpose(k, 2, 3)) / sqrt(DH)
  let masked  = masked_fill(scores, causal_mask(T), value: -1000000.0)
  let weights = softmax(masked, axis: -1)
  return matmul(weights, v)
}`,
    torch: `def attention_core(q, k, v):
    # shapes are a comment, not a contract
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1),
                                 dtype=torch.bool, device=q.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v`,
    note: `Nearly the same length — correctly so, because this is the mathematics. The difference is that
every intermediate shape is a verified contract, \`DH\` is a symbolic dimension rather than a runtime
\`size(-1)\`, and the mask length is checked against T instead of being recomputed from whatever
arrived. This is category-2 code: TENSA should make it *clearer*, not shorter.`,
  },
];

export const COMPRESSION = {
  categories: [
    { id: 1, name: "ML / architectural intent", axis: "kept, and made checkable", torch: "present, mixed with 6 and 7" },
    { id: 2, name: "Mathematical algorithm", axis: "kept at the same size, with contracts", torch: "present" },
    { id: 3, name: "Data / learning-problem definition", axis: "promoted into the program with leakage checks", torch: "usually scattered in Python" },
    { id: 4, name: "Training / lifecycle policy", axis: "declarative phases, updates, events", torch: "hand-written loops" },
    { id: 5, name: "Execution policy", axis: "annotations (device, precision, clipping)", torch: "interleaved with the algorithm" },
    { id: 6, name: "Framework / runtime plumbing", axis: "≈ eliminated", torch: "substantial" },
    { id: 7, name: "Host-language plumbing", axis: "≈ eliminated", torch: "substantial" },
  ],
  note: `The claim is not character count. A successful abstraction removes accidental machinery,
preserves meaningful choices, improves static reasoning, improves diagnostics, composes, stays usable
for research and gives the compiler semantic knowledge it can act on. Categories 1–5 should become
*clearer*; only 6 and 7 should shrink toward zero.`,
};

/** Finding taxonomy of the hardening protocol (§3): one primary class per finding. */
export type FindingKind = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H";

export const FINDING_KINDS: { k: FindingKind; label: string; tone: string; desc: string }[] = [
  { k: "A", label: "accidental friction", tone: "emerald", desc: "the language forced machinery that should be inferred" },
  { k: "B", label: "essential complexity", tone: "cyan", desc: "the programmer is expressing real information" },
  { k: "C", label: "hidden information", tone: "amber", desc: "conciseness came from silently guessing" },
  { k: "D", label: "backend leakage", tone: "rose", desc: "implementation details contaminated semantics" },
  { k: "E", label: "semantic ambiguity", tone: "violet", desc: "valid source has more than one reasonable reading" },
  { k: "F", label: "compiler bug", tone: "orange", desc: "semantics are clear; the implementation gets them wrong" },
  { k: "G", label: "missing abstraction", tone: "fuchsia", desc: "several programs expose the same inexpressible concept" },
  { k: "H", label: "capability gap", tone: "sky", desc: "the program is right; a backend capability is missing" },
];

export const toneOfKind = (k: string): string => FINDING_KINDS.find((x) => x.k === k)?.tone ?? "zinc";

export const HARDENING: { model: string; finding: string; kind: FindingKind; action: string }[] = [
  {
    model: "Inception-style multi-scale block",
    finding: "Branch bodies needed more than one statement.",
    kind: "A",
    action: "Added labelled `branch name { … }` bodies instead of forcing every branch to be a pipeline.",
  },
  {
    model: "ResNet stage with stride",
    finding: "The skip path must change shape; earlier prototypes silently inserted a 1×1 projection.",
    kind: "C",
    action: "`residual via <stage>` is mandatory when shapes differ; AXS0404 explains the repair.",
  },
  {
    model: "Siamese / two-tower",
    finding: "Sharing was previously expressed with registration machinery or string labels.",
    kind: "A",
    action: "Sharing follows value identity; inspect proves the parameter count.",
  },
  {
    model: "Cross-attention fusion",
    finding: "Implicit flow has no obvious meaning with two inputs.",
    kind: "C",
    action: "The cursor is unseeded for multi-input entry points; AXS0301 instead of a guess.",
  },
  {
    model: "Decoder-only LM",
    finding: "Target shifting was previously derived from a dataset name.",
    kind: "C",
    action: "Shifting is explicit example construction (`window[0:T]`, `window[1:T+1]`).",
  },
  {
    model: "RNN tagger",
    finding: "`for 12:` cannot express recurrence; unrolling would duplicate parameters.",
    kind: "B",
    action: "Added `scan` with a declared carry contract; the body is elaborated once, so parameters are shared across steps.",
  },
  {
    model: "GAN",
    finding: "Two objectives, two optimizers and an update ratio do not fit one supervised loop.",
    kind: "B",
    action: "`update <loss> with <opt> times <n>` inside a phase; disjointness is checked.",
  },
  {
    model: "BatchNorm in a frozen backbone",
    finding: "Running statistics keep moving even when parameters are frozen.",
    kind: "B",
    action: "State is separate from parameters; freeze affects parameters, and train/eval context controls state updates.",
  },
  {
    model: "Flash attention kernel",
    finding: "A backend-specific kernel must not leak into language semantics.",
    kind: "D",
    action: "`custom op` declares contracts and per-backend implementations; missing shape semantics is AXS0903.",
  },
  {
    model: "Convolution gradients in the reference backend",
    finding: "An earlier prototype reported convolution as non-differentiable.",
    kind: "D",
    action: "Capability tables are per backend; `nondiff` is reserved for mathematics (argmax) and `grad-stopped` for intent.",
  },
  {
    model: "Tabular pipeline",
    finding: "Statistics fitted on the whole dataset leak validation information.",
    kind: "C",
    action: "`fit:` is required on fitted statistics and must name the training split (AXS0620).",
  },
  {
    model: "Multi-task head",
    finding: "Multiple outputs made objective binding ambiguous.",
    kind: "A",
    action: "Tuple results plus port-named objective bindings (`Net(x)[1]`).",
  },
  {
    model: "Property test: shared vs. independent stage counts (F-001)",
    finding:
      "Two `let y = Enc(x)` bindings with the same name produced one parameter set: instance paths keyed on the binding name collided and the second application was recorded as *shared*.",
    kind: "F",
    action: "Instance paths are uniquified; regression test + property generator fixed. E-001 resolved: same-scope rebinding now warns (AXS0304) and says the rebound stage owns fresh parameters.",
  },
  {
    model: "Carried constraints at runtime (F-002, F-004)",
    finding:
      "Equalities the checker could only *assume* (`S = T`, `heads ∣ D` over a free width) were never verified when the program ran.",
    kind: "F",
    action: "The runtime checks every assumed constraint against the dim bindings before the first forward pass and refuses with the equation (`S = T but 5 ≠ 3`).",
  },
  {
    model: "U-Net skip connections (§10, tier 2)",
    finding:
      "`2K` vs `4K` at the wrong-level skip was *carried* rather than refuted, and the emitted module header read `STATIC_DIMS = {\"H\": 4*K}` — a NameError at import (H-003).",
    kind: "F",
    action: "Same-sign polynomial differences are refuted at compile time; derived dims are resolved after runtime bindings and symbolic extents lower to Python expressions. `upsample` joined the catalog.",
  },
  {
    model: "Custom op with `shape: unknown` (F-009)",
    finding: "Unknown was modelled as a rank-1 tensor `[?]`, so a legal program was rejected with a false result-shape error.",
    kind: "F",
    action: "`Tensor[?]` is its own outcome (§5.2): nothing downstream is sized or contradicted; the result contract is reported as uncheckable, not violated.",
  },
  {
    model: "Multi-objective plan (§16)",
    finding:
      "Gradient coverage reflected only the last update's gradients (H-002); a misspelled objective port and a tuple bound without an index were accepted silently (F-010, F-008).",
    kind: "F",
    action: "Coverage accumulates over the run; unknown ports and un-indexed tuple bindings are errors at the binding.",
  },
  {
    model: "Cursor rules (§6)",
    finding: "Destructuring `let` did not move the implicit cursor (F-007); a side `let` *does* redirect it, which is the documented rule but the most surprising reading (E-003).",
    kind: "E",
    action: "F-007 fixed; E-003 pinned by a twin as a design note rather than changed.",
  },
  {
    model: "Explicit multi-head attention (§12, tier 2)",
    finding:
      "MHA written from primitives (`reshape` → `transpose` → `matmul` → `causal_mask` → `softmax`) needed no custom op and matches the catalog `attention` numerically — but the emitter printed `-1` for every symbolic reshape extent and dropped the `mask` port entirely (H-005).",
    kind: "F",
    action: "The generated `forward` binds runtime dims from the input shapes; masks lower to `attn_mask=~m`. Merging heads *without* the transpose is shape-correct and only the metamorphic pair catches it (E-007, recorded as the boundary of static checking).",
  },
  {
    model: "Slice and index bounds (§5.3, F-011)",
    finding: "`x[:, 0:T+1, :]` was typed `Tensor[B, T + 1, D]` and ran past the operand; indices were never bounds-checked.",
    kind: "F",
    action: "Bounds follow the four-outcome discipline: proved, refuted (AXS0410), or carried as a `≤` constraint (`W + 2 ≤ T`) that the runtime verifies before the first forward pass.",
  },
  {
    model: "Attention masks (§12, F-012 / F-013)",
    finding:
      "`attention(mask:)` had no shape rule, `masked_fill` checked only the kind, a rank-3 mask broadcast its batch axis onto the heads axis at runtime — and a skipped optional port let the mask slide into the *key* slot.",
    kind: "F",
    action: "Mask contract `Mask[Tq, Tk]` or `Mask[B|1, Tq|1, Tk]`; padding masks are `Mask[B, 1, S]`; a skipped optional port defaults to the port before it (key ← query, value ← key).",
  },
  {
    model: "Padded language model (§20, Milestone 3)",
    finding:
      "A field could not be built from an earlier field by name (`field inputs: Tokens[T] = ids[0 : T]` was an *unknown data operation*; the gpt example only worked because `window` is also an op) — F-021. A pad `Mask` declared as a model input and never passed to `attention` was silent — F-019.",
    kind: "F",
    action: "Earlier fields lower to `field.<name>`; a model input the body never reads warns (AXS0305) and, for a Mask, says where it belongs. `pad_to` / `pad_mask` joined the data catalog (H-009).",
  },
  {
    model: "Tabular and vision data contracts (§19, Milestone 3)",
    finding:
      "Leakage (AXS0620) pointed at the `data` block rather than the fitted op (F-020); a `vocab(fit: train)` with no policy for unseen categories was silent; HWC-vs-CHW got a generic shape code while a channel mismatch got the contract code (E-008).",
    kind: "F",
    action: "AXS0620 sits on the operation; AXS0622 asks for `unknown:` on a fitted vocabulary; every refuted data↔model axis is AXS0602 (model↔objective AXS0603) with the axis named.",
  },
  {
    model: "`fn` effects and the three ways to lose a gradient (§15, Milestone 3)",
    finding:
      "A `fn` calling `dropout` was `pure` at every call site (F-016). Non-differentiable (`argmax`), gradient-stopped (`stop_grad`) and backend-only (custom op without reference) first layers all read as \"no gradient path\".",
    kind: "F",
    action: "Region-owning nodes carry the union of their body's effects. The IR effect on the breaking node (`nondiff` vs `grad-stopped`) and AXS0901 (info) for the custom op tell the three apart.",
  },
  {
    model: "EMA teacher / distillation / self-training (§26 items 4–6, Milestone 3)",
    finding:
      "`model teacher = Student` beside `model student = Student` is one parameter set (correct) and nothing said so (F-015). `ema(r)` never said whether `r` is the fraction kept or the fraction of the new value (E-009). And the tracked shadow is write-only: no model can be *evaluated with* it.",
    kind: "G",
    action: "AXS0706 warns at the second alias; `r` is the decay (`next = r·cur + (1−r)·new`) in both backends. The consumption half is G-cand-003 *derived parameter sets* — three witnesses (EMA teacher, teacher refresh, MAML inner loop), §43 analysis in `hardening/proposals/`, no syntax.",
  },
  {
    model: "Transfer, GAN and curriculum lifecycles (§22, §25, §26 items 1–3, Milestone 3)",
    finding:
      "A phase could update with an optimizer it had frozen entirely (F-014); an optimizer or loss no phase applied compiled silently (F-017); `until` could name a metric nothing produces (F-018); an optimizer without `over` covered only the *first* model (F-022); the runtime did not simulate schedules, `until` or events at all (H-001), the emitted objectives raised NameError on their own ports (H-006), `D(G(z))` gave the generator no gradient (H-007) and the two backends counted `every N steps` differently (H-008).",
    kind: "F",
    action: "AXS0704 / AXS0707 / AXS0708 / AXS0709 at the plan; region `*` = every bound model; the reference loop simulates the whole plan and the run report carries per-phase `stoppedBy` and an event log; phase events count phase steps, plan events run steps; all 16 learning programs execute under torch 2.12.",
  },
];

export const LIMITATIONS = `## Known limitations

- **No data-dependent control flow in the static core.** There is no \`cond\` IR node yet. Branching
  whose arms share a static contract is a designed extension, not a semantic prohibition.
- **The reference backend is an interpreter.** Convolution is a naive loop; large models are slow in
  the browser. Runs use small runtime dimensions and a small step budget.
- **Data execution is simulated.** Source adapters are declared and checked but not executed: the
  reference backend feeds deterministic synthetic tensors that match the declared example contracts.
  Pipelines are analysed (effects, leakage, shape effects), not run.
- **The reference training loop is a simulation.** Schedules, \`until\`, events, region learning rates
  and tracked shadows are honoured (one epoch = 6 synthetic steps), but the metrics an \`until\` reads
  are losses over synthetic batches — the run report tells you *whether* a rule fires, not *when* it
  would on real data.
- **A tracked shadow is write-only.** \`track t = ema(region, rate: r)\` reaches the checkpoint and the
  emitted plan, but no model can be evaluated with it (G-cand-003, proposal stage).
- **Semantic kinds are shallow.** Six kinds with warnings, no user-defined refinements, no unit or
  layout tracking.
- **No distributed or sharding semantics.** Execution policy currently covers device, precision,
  gradient clipping and seeds only.
- **IR is stable but not versioned.** Textual IR is deterministic for a given compiler build; there is
  no on-disk format or schema guarantee yet.
- **Effects are declared per catalog entry**, not inferred for user blocks beyond the union of their
  contents.`;

export const JOURNAL = `## Design journal — the synthesis decisions

**1. Which prototype won which argument.**
Vect won flow and topology: implicit sequential flow is genuinely better for the 80% case, and
\`residual\` / \`split … merge\` are real concepts. Lumen won the semantic core: compile-time vs runtime,
stage identity as the basis of parameter sharing, and effects. Axon won almost nothing syntactically —
mandatory \`Linear[784 -> 256]\` contracts restate consequences — but it was right that contracts must
exist; TENSA keeps them as *verified return types*, not as mandatory plumbing.

**2. The cursor needed one sentence, not a rulebook.**
Earlier drafts had separate rules for statements, pipelines, branches and blocks. The final rule is:
*the implicit value is the value produced by the most recent statement*, seeded only when a
declaration has exactly one tensor input. Everything else (branch-local cursors, unseeded multi-input
entry points) follows from it.

**3. Rejected: auto-seeding multi-input flow with the first parameter.**
It reads well in a demo and hides a modelling decision. AXS0301 was preferred.

**4. Rejected: a \`share\` keyword.**
Bindings already express identity. A keyword would create two ways to say the same thing and a third
way to get it wrong. \`inspect\` reports sharing instead.

**5. Rejected: making every architecture a keyword.**
No \`transformer\`, \`resnet\` or \`unet\` syntax. The catalog is a table of shape rules; language
semantics are flow, topology, identity, state, objectives, data and lifecycle.

**6. Rejected: string-based parameter selection.**
\`freeze "encoder.*"\` is fragile. Regions resolve against the elaborated stage tree, so a typo is
AXS0701 instead of a silent no-op — the single most dangerous bug class in fine-tuning code.

**7. Promoted: \`observe\`.**
Persistent state needed a read-and-update primitive that is honest about training context. Making
BatchNorm use exactly the same mechanism as user state proved the model was general.

**8. Promoted: \`scan\`.**
Unrolling recurrence with \`for\` would have duplicated parameters and lied about the algorithm. A
carry contract plus a single elaboration of the body gives shared parameters and static shapes.

**9. Promoted: composite IR nodes with regions.**
The Vect branching bug (parallel branches lowered sequentially) was possible because branches became
plain sequential calls. Regions make the topology a first-class IR object, and inspection is generated
from it, so the report cannot drift from the lowering.

**10. Data was the hardest boundary.**
The chosen line — *anything that can change what is learned is in the language; anything about
getting bytes off a disk is not* — is defensible and checkable. Leakage checking is the proof that it
was worth doing: it is a correctness property no framework-level API can offer.

**11. Lifecycle: six concepts, no loop.**
Phase, step, update, schedule, event and training state covered every algorithm in the hardening set
(fine-tuning, GAN, curriculum, EMA teacher). The moment a seventh concept looked necessary, it was
always an escape-hatch case — so the escape hatch is documented instead.`;

export const REPORT = `## Final report

### What became simpler than PyTorch, and why
- **Shape plumbing disappeared.** Input widths, channel counts, flattened sizes and head dimensions are
  consequences. They are computed, not written, and every intermediate shape is a checked contract.
- **Parameter identity became a property instead of a convention.** Sharing is value identity; inspect
  proves the count; a regression test enforces it.
- **Topology became inspectable.** Branch/merge/residual are IR objects; the compiler cannot claim one
  structure and execute another.
- **The training loop disappeared** without becoming a black box: phases, updates and events are data
  in the IR, so they can be checked (disjoint optimizers, non-existent regions, frozen-only updates).
- **Data leakage became a compile error.** No framework can do this, because no framework knows which
  split a statistic was fitted on.
- **The lifecycle became checkable.** A warmup that trains nothing, an optimizer no phase applies, a
  stopping rule on a metric nothing produces, a teacher that is secretly the student — each is a
  diagnostic at the plan, not a discovery after the run (Milestone 3).

### What remains equally complex, because the complexity is inherent
Attention mathematics, normalisation formulas, loss definitions, augmentation policy, update ratios,
learning-rate programmes and curriculum conditions. These are the algorithm. TENSA makes them clearer
(named ports, verified shapes, explicit ratios) but not shorter.

### Where the language still hides information
- The catalog hides *parameter initialisation choices* behind defaults (\`xavier\`, \`kaiming\`). This is
  a real hidden choice; it is visible in \`inspect\` and overridable only for explicit parameters today.
- \`attention(heads: 8)\` hides the internal projection structure. Descending to the tensor level is
  supported, but the catalog version is not literally the same program.
- Batching hides host-side collation policy.
- The derived default phase hides the update/optimizer pairing when only one of each exists.

### Where backend leakage remains
- \`precision mixed\` and \`device auto\` are execution-policy annotations that only some backends honour.
- The reference backend's capability table (conv2d "naive im2col; small inputs only") reaches the run
  report — intentionally, but it is still backend information in a user-facing surface.
- Custom ops embed literal backend source text.

### Which abstractions provide the most semantic leverage
1. Symbolic dimensions with carried constraints.
2. Stage identity as the definition of parameter sharing.
3. Composite IR regions for topology.
4. The effect/state model (it powers train/eval semantics, checkpointing and diagnostics at once).
5. Data contracts from field to model input to objective port.

### Which proposed abstractions were rejected
Auto-seeded multi-stream flow; a \`share\` keyword; architecture keywords; string parameter filters;
a general imperative training loop; mandatory input dimensions; dataset-name-derived targets.

### Which programs still require an escape hatch
Data-dependent control flow inside a graph; anything that needs a gradient as a value (WGAN-GP,
adversarial training's input gradient, MAML — G-cand-002, three witnesses); a model evaluated with a
*derived* parameter set (EMA teacher, target networks — G-cand-003); reinforcement-learning loops whose
environment interaction is not a data pipeline; kernels with no shape semantics.

### What would need to change before serious research use
An executable data plane (real adapters), a \`cond\` IR node, gradient-level programming
(\`grad(loss, wrt: region)\` as a first-class value), distributed/sharding policy, a compiled backend
rather than an interpreter, and a versioned IR format with a checkpoint schema.

### What should **not** be added
An encyclopedia of architecture keywords; general-purpose control flow; classes and inheritance;
implicit projections or reshapes that make shapes "just work"; automatic dataset-derived target
semantics; and any feature whose only justification is saving characters.`;
