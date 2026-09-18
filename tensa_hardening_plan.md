# TENSA Challenge & Hardening Plan

**Status:** Pre-hardening language specification\
**Purpose:** Adversarial validation of TENSA semantics, compiler
correctness, diagnostics, IR, runtime behavior, and abstraction quality\
**Primary question:** *Where does TENSA stop being a better way to
express machine learning, and why?*

------------------------------------------------------------------------

## 1. Purpose

TENSA has reached the point where adding features speculatively is more
likely to make the language worse than better. The next phase is
therefore not feature development but **adversarial hardening**.

The objective is not to demonstrate that TENSA can express familiar toy
networks. It is to deliberately find programs that stress its claims:

-   choices are written and consequences inferred;
-   symbolic shapes are useful and trustworthy;
-   implicit flow remains unambiguous;
-   topology lowers correctly;
-   parameter identity follows value identity;
-   architecture and tensor mathematics compose naturally;
-   objectives explicitly define learning relationships;
-   state has coherent semantics;
-   data transformations that affect learning are represented;
-   lifecycle concepts cover meaningful temporal behavior without
    recreating Python;
-   recurrence is distinct from static repetition;
-   diagnostics expose uncertainty rather than hiding it;
-   backend limitations do not contaminate language semantics.

A successful hardening campaign should produce both a stronger compiler
and a **failure map of the language design**.

The most valuable result is not "all tests pass." It is knowing
precisely which complexity TENSA removes, which complexity is inherent,
and where its abstractions fail.

------------------------------------------------------------------------

## 2. Hardening rules

### 2.1 Freeze before testing

Freeze the current language reference for each hardening round.

Do not modify syntax or semantics merely because a challenge is awkward.
Record the failure first. Language changes should occur in deliberate
review batches after patterns emerge.

### 2.2 Test semantics, not parser acceptance

A program parsing or passing `axis check` is not sufficient.

For relevant challenges verify:

1.  source acceptance/rejection;
2.  inferred types and shapes;
3.  parameter count and ownership;
4.  sharing identity;
5.  state/effects;
6.  generated IR;
7.  `inspect` representation;
8.  backend lowering;
9.  runtime result;
10. gradients where applicable;
11. checkpoint contents where applicable.

The compiler, inspector, emitted backend and runtime must describe the
**same program**.

### 2.3 Every positive test gets an evil twin

For every valid construct, create a nearby invalid or ambiguous version.

Examples:

-   valid residual → mismatched residual;
-   valid concat → incompatible branch;
-   logits into cross entropy → probabilities into cross entropy;
-   train-fitted normalization → validation-fitted normalization;
-   shared stage → accidentally recreated stage;
-   valid attention split → non-divisible head dimension.

Good diagnostics are part of passing the test.

### 2.4 Do not reward brevity by itself

For each challenge ask:

> Did TENSA remove accidental machinery, or merely hide meaningful
> information?

Conciseness caused by inference is desirable only when the compiler can
justify the inference.

### 2.5 New syntax requires repeated evidence

One difficult program does not justify a new construct.

Prefer adding a language abstraction only when multiple substantially
different programs expose the same missing concept.

------------------------------------------------------------------------

## 3. Finding taxonomy

Every discovered issue must receive one primary classification.

  ---------------------------------------------------------------------------------
  Code                    Category                Meaning
  ----------------------- ----------------------- ---------------------------------
  **A**                   Accidental friction     TENSA requires information or
                                                  machinery that should be
                                                  inferable.

  **B**                   Essential complexity    The complexity belongs to the
                                                  algorithm and should remain
                                                  visible.

  **C**                   Hidden information      TENSA appears simple because it
                                                  guessed or concealed a meaningful
                                                  choice.

  **D**                   Backend leakage         Backend/runtime implementation
                                                  details have entered
                                                  language-level code or semantics.

  **E**                   Semantic ambiguity      Valid source has more than one
                                                  reasonable interpretation.

  **F**                   Compiler bug            Language semantics are clear, but
                                                  parser/checker/lowering/runtime
                                                  implements them incorrectly.

  **G**                   Missing abstraction     Multiple programs expose the same
                                                  recurring ML concept that TENSA
                                                  cannot express naturally.

  **H**                   Capability gap          TENSA represents the program
                                                  correctly, but a backend/runtime
                                                  capability is missing.
  ---------------------------------------------------------------------------------

Only **G** should normally trigger consideration of a new language-level
abstraction.

An issue may have secondary classifications, but one primary cause must
be chosen.

------------------------------------------------------------------------

## 4. Challenge record

Every nontrivial challenge should produce a durable record.

``` text
Challenge:
Source/paper:
TENSA features stressed:
Expected semantics:
TENSA implementation:
Check result:
Inspect result:
IR result:
Runtime result:
Reference result:
Gradient result:
Diagnostics:
Finding classification:
Severity:
Workaround:
Proposed action:
Language change required?:
Regression test added?:
```

When comparing against an existing implementation, preserve both the
TENSA program and the reference implementation.

------------------------------------------------------------------------

# Part I --- Semantic Microtests

## 5. Symbolic dimension and shape torture

Symbolic shape reasoning is a core TENSA claim and should receive
unusually aggressive testing.

### 5.1 Algebra

Test equivalence and simplification involving:

-   `D * 4` vs `4 * D`;
-   `(D / H) * H`;
-   `T + 1 - 1`;
-   nested products;
-   flatten/unflatten products;
-   convolution output expressions;
-   floor division;
-   constants mixed with template dimensions.

### 5.2 Constraint outcomes

Ensure the compiler distinguishes at least:

**Proved**

``` text
D = D
```

**Runtime-constrained / carried**

``` text
S = T
```

where both are independently free dimensions but equality is required.

**Impossible**

``` text
64 = 128
```

**Unknown**

A custom operation or insufficient contract prevents proof.

The IR and `inspect` output must expose carried constraints. Runtime
enforcement must agree with static reporting.

### 5.3 Operations to torture

Exercise symbolic shapes through:

-   broadcasting;
-   matmul;
-   concat;
-   split;
-   reshape;
-   flatten;
-   transpose/permute;
-   slicing;
-   convolution;
-   pooling;
-   attention;
-   head splitting/merging;
-   sequence windowing;
-   recurrence.

### 5.4 Attention divisibility

Test `D / Heads` where divisibility is:

1.  statically proved;
2.  statically impossible;
3.  unresolved until instantiation;
4.  still unresolved at execution.

No case should silently truncate a dimension unless floor semantics were
explicitly requested.

------------------------------------------------------------------------

## 6. Implicit-flow torture

Verify the cursor rule independently of architecture complexity.

Test:

-   one-input declaration seeds the cursor;
-   multi-input declaration does not;
-   bare transformation with no cursor fails;
-   `let` behavior does not unexpectedly destroy or replace the cursor;
-   explicit join re-establishes a single stream;
-   tuple output does not accidentally become an implicit stream;
-   nested block cursor scope;
-   branch-local cursor isolation;
-   objective cursor behavior if objectives support it;
-   tensor-function behavior remains explicit where intended.

Construct deliberately ambiguous-looking programs and verify they fail
instead of being guessed.

------------------------------------------------------------------------

## 7. Topology torture

### 7.1 Split/merge invariant

For every `split`, verify directly in IR:

> Every branch entry consumes the same split input.

Test:

-   two branches;
-   many branches;
-   single-expression branches;
-   multi-statement branches;
-   pipelines inside branches;
-   named branches;
-   nested split;
-   split inside residual;
-   residual inside split;
-   branch referencing a named outer value;
-   concat;
-   add;
-   mean merge.

The historical Tensa sequential-branch bug must remain a permanent
regression test.

### 7.2 Residual invariant

Verify:

> The body and projection consume the same incoming value, and an
> explicit add consumes their results.

Test:

-   identity residual;
-   projected residual;
-   channel mismatch;
-   spatial mismatch;
-   nested residuals;
-   stochastic body;
-   stateful body;
-   explicit named values inside body.

The compiler must never invent a learned projection.

------------------------------------------------------------------------

## 8. Parameter identity torture

TENSA claims that parameter identity follows stage-value identity. Test
this until it becomes boring.

Compare:

``` text
let encoder = Encoder()
encoder(a)
encoder(b)
```

with:

``` text
Encoder(a)
Encoder(b)
```

Verify exact parameter counts and ownership.

Then test stage reuse:

-   across branches;
-   across residuals;
-   inside nested blocks;
-   across multiple model outputs;
-   inside `for`;
-   inside `scan`;
-   passed into another construct if stage values are first-class there;
-   referenced by lifecycle parameter regions.

Test accidental shadowing and rebinding.

The invariant is:

> A programmer should be able to determine sharing from lexical/value
> identity without understanding compiler internals.

------------------------------------------------------------------------

## 9. Static repetition torture

Verify separately:

``` text
for 12: TransformerBlock()
```

creates twelve independent parameter sets, while:

``` text
let block = TransformerBlock()
for 12: block
```

applies one shared stage twelve times.

Test:

-   zero repetitions if legal;
-   one repetition;
-   nested repetition;
-   indexed repetition;
-   index-dependent dimensions;
-   repetition whose shape changes each iteration;
-   repeated parameterless operation;
-   repeated stochastic operation;
-   repeated stateful stage;
-   repetition inside branch/residual;
-   branch/residual inside repetition.

Do not impose shape-preserving restrictions unless mathematically
required.

------------------------------------------------------------------------

## 10. Multi-stream torture

Test the boundary where implicit flow gives way to explicit dataflow.

Required patterns:

-   two inputs → explicit join → implicit stream resumes;
-   cross-attention;
-   Siamese/twin input;
-   tuple destructuring;
-   multiple model outputs;
-   several objective operands;
-   two independent streams processed for several stages before joining;
-   branch result reused later;
-   nested tuple structures if supported.

### Key real-world challenge: U-Net

U-Net is an important test because encoder activations are saved at
several resolutions and consumed substantially later by decoder stages.

It stresses:

-   naming discipline;
-   long-lived graph values;
-   spatial shape inference;
-   concat;
-   multi-resolution topology;
-   ergonomics of non-local skip connections.

If U-Net requires excessive ceremony, classify every source of ceremony
before proposing syntax.

------------------------------------------------------------------------

# Part II --- Tensor Mathematics and Differentiation

## 11. Architecture → tensor-level descent

Implement important operations both through catalog primitives and
explicit tensor mathematics where practical.

Required:

-   scaled dot-product attention;
-   RMSNorm;
-   gated MLP / SwiGLU or GeGLU;
-   normalization;
-   custom masking;
-   cosine similarity;
-   small routing function.

Compare shape inference and runtime output.

The tensor-level implementation must not become an opaque "custom
operation" merely because it is mathematically detailed.

------------------------------------------------------------------------

## 12. Explicit attention challenge

Implement multi-head attention from tensor primitives:

``` text
projection
reshape/split heads
transpose
QKᵀ
scale
mask
softmax
attention × V
merge heads
output projection
```

Stress:

-   symbolic `D`;
-   `Heads`;
-   `DH = D / Heads`;
-   causal masks;
-   padding masks;
-   broadcasting;
-   semantic kinds;
-   multiple inputs for cross-attention.

Compare with catalog `attention` numerically under controlled weights
where feasible.

This is a primary test of TENSA's two-level language claim.

------------------------------------------------------------------------

## 13. Differentiation hardening

Explicit differentiation remains a likely semantic gap.

Challenge TENSA with:

-   stop-gradient;
-   gradient penalty;
-   input gradients;
-   Jacobian-vector or vector-Jacobian products;
-   higher-order gradient;
-   gradient reversal;
-   multiple losses sharing computation;
-   gradient accumulation;
-   custom backward rule;
-   gradient clipping at different scopes.

Candidate algorithms:

-   WGAN-GP;
-   MAML-style inner/outer differentiation;
-   adversarial input perturbation;
-   gradient-reversal domain adaptation.

For each failure determine whether explicit differentiation belongs:

1.  in tensor semantics;
2.  in lifecycle semantics;
3.  in the runtime escape hatch.

Do not automatically add `grad` merely because one example needs it.
Look for a coherent differentiation model.

------------------------------------------------------------------------

# Part III --- State, Effects, and Context

## 14. Persistent-state torture

Test all four state categories:

-   parameter;
-   persistent non-parameter state;
-   optimizer state;
-   temporary tensor.

Required cases:

-   BatchNorm running statistics;
-   explicit EMA value;
-   counter;
-   memory bank;
-   prototype vector;
-   model containing several independent state values;
-   shared stage owning state;
-   state inside `scan`.

Verify:

-   training update behavior;
-   evaluation read-only behavior;
-   checkpoint inclusion;
-   restoration;
-   stage ownership;
-   sharing behavior;
-   inspector reporting.

### EMA teacher challenge

Implement a student model plus EMA teacher.

This stresses whether persistent model-like state can be represented
naturally or whether TENSA's state model is too scalar/tensor-centric.

------------------------------------------------------------------------

## 15. Effects torture

Verify that effects propagate compositionally.

Test pure functions containing or calling:

-   stochastic operation;
-   state read;
-   state write;
-   parameterized stage;
-   gradient stop;
-   non-differentiable operation.

Ensure the compiler distinguishes:

-   semantic non-differentiability;
-   deliberate gradient stop;
-   missing backend gradient capability.

Test stochastic transforms in evaluation and data pipelines.

------------------------------------------------------------------------

# Part IV --- Objectives and Learning Problems

## 16. Objective torture

Go beyond one-output supervised classification.

Required:

-   classification;
-   reconstruction;
-   weighted multitask loss;
-   auxiliary classifier;
-   regularization term;
-   contrastive loss;
-   triplet loss;
-   sequence loss with mask;
-   language-model loss;
-   losses involving intermediate representations;
-   two models contributing to one objective.

Test wrong semantic kinds and shape mismatches.

The compiler must never infer a target relationship merely because
input/output shapes happen to match.

------------------------------------------------------------------------

## 17. Autoencoder and VAE

A plain autoencoder verifies explicit input-as-target binding.

A VAE adds:

-   multiple encoder outputs;
-   stochastic latent sampling;
-   reparameterization;
-   KL term;
-   reconstruction term;
-   weighted objective composition.

This is a strong test of the boundary among architecture, stochastic
tensor math and objective semantics.

------------------------------------------------------------------------

# Part V --- Data Semantics

## 18. Vision pipeline

Implement a realistic image classification pipeline with:

-   decoding boundary;
-   resize/crop;
-   normalization fitted on train;
-   stochastic train augmentation;
-   deterministic evaluation transform;
-   split;
-   batching;
-   model mapping;
-   objective mapping.

Evil twins:

-   fit normalization on validation;
-   stochastic eval augmentation;
-   wrong image layout;
-   model expects different channels;
-   target semantic kind mismatch.

------------------------------------------------------------------------

## 19. Tabular pipeline

Use:

-   numeric fields;
-   missing values;
-   categorical values;
-   train-fitted imputation;
-   standardization;
-   train-fitted vocabulary;
-   unknown-category handling;
-   split;
-   multiple model inputs if appropriate.

This should test whether TENSA can reason about **fitted
transformations**, not merely tensor shapes.

Verify leakage detection.

------------------------------------------------------------------------

## 20. Language-model data pipeline

Implement:

``` text
raw text
→ tokenization
→ T+1 window
→ input [0:T]
→ target [1:T+1]
→ packing/batching
→ model
→ objective
```

Stress variable document lengths, boundary handling and masks.

The target shift must remain explicit and inspectable.

------------------------------------------------------------------------

## 21. Contrastive-learning data pipeline

Implement two stochastic views of the same source example.

This tests whether TENSA understands that:

``` text
view1 = augment(example)
view2 = augment(example)
```

are two independent stochastic applications rather than one computed
value reused twice.

SimCLR is an excellent full challenge here.

------------------------------------------------------------------------

# Part VI --- Lifecycle

## 22. Ordinary supervised training baseline

Establish the minimum ceremony for conventional training.

Verify derived:

-   zeroing gradients;
-   forward;
-   objective;
-   backward;
-   optimizer step;
-   train/eval context;
-   validation;
-   clipping;
-   checkpointing;
-   device;
-   precision.

The programmer should not need to mention these mechanics unless
changing their semantics.

------------------------------------------------------------------------

## 23. Transfer-learning phases

Implement:

``` text
warmup:
    encoder frozen
    head optimizer only

finetune:
    encoder unfrozen
    separate encoder/head learning rates
```

Verify structural parameter regions, optimizer ownership and checkpoint
behavior.

Evil twins:

-   nonexistent region;
-   overlapping optimizers;
-   frozen parameters claimed by optimizer;
-   conflicting phase configuration.

------------------------------------------------------------------------

## 24. Alternating optimization

Implement a small GAN.

Required lifecycle:

``` text
update discriminator loss with discriminator optimizer times 2
update generator loss with generator optimizer times 1
```

Verify that the IR represents an update ratio rather than lowering
source syntax into arbitrary host control flow.

Then test objectives sharing models and parameters.

------------------------------------------------------------------------

## 25. Curriculum and metric events

Implement phases controlled by:

-   fixed epochs;
-   fixed steps;
-   metric threshold;
-   validation cadence;
-   early stopping;
-   schedule transition.

Test conflicting termination conditions and metric availability.

The important question is whether TENSA's six temporal concepts remain
sufficient without general imperative control flow.

------------------------------------------------------------------------

## 26. Lifecycle escalation challenges

Attempt, in order:

1.  multiple optimizers;
2.  multiple objectives;
3.  alternating updates;
4.  EMA teacher/student;
5.  knowledge distillation;
6.  self-training/pseudo-labeling;
7.  adversarial training;
8.  RL-style rollout/update separation;
9.  meta-learning inner/outer loop.

Record exactly where the current lifecycle vocabulary stops fitting.

Do not expand the language until several failures can be described by
the same missing temporal concept.

------------------------------------------------------------------------

# Part VII --- Recurrence and Dynamic Computation

## 27. `scan` baseline

Implement from primitives:

-   vanilla RNN;
-   GRU if practical;
-   simple iterative refinement.

Verify:

-   body elaborated once;
-   parameters shared across time;
-   carry shape;
-   output stacking;
-   final state;
-   gradients through recurrence;
-   state versus carry distinction.

Compare parameter counts against an equivalent statically repeated
construction to ensure they differ as intended.

------------------------------------------------------------------------

## 28. Scan stress

Test:

-   multiple carries;
-   tuple carry;
-   multiple outputs per step;
-   scan over different axes;
-   symbolic sequence length;
-   nested scan;
-   scan containing stochastic operations;
-   scan containing persistent state;
-   scan calling shared external stages.

Determine whether the current single-carry model is sufficient.

------------------------------------------------------------------------

## 29. Dynamic-control-flow pressure

Use algorithms that naturally want runtime decisions:

-   adaptive computation;
-   early-exit network;
-   conditional expert routing;
-   variable iteration until convergence;
-   beam search;
-   autoregressive decoding;
-   rejection/resampling patterns.

For each, determine whether it can be represented as:

-   static graph;
-   elementwise `where/select`;
-   `scan`;
-   future `cond`;
-   custom dynamic region;
-   host/runtime algorithm.

Do not treat every inability as a language failure.

------------------------------------------------------------------------

# Part VIII --- Real Architecture Suite

## 30. Architecture ladder

Implement increasingly difficult real architectures, preferably from
their papers or canonical descriptions rather than copying framework
code.

### Tier 1 --- structural sanity

-   MLP;
-   LeNet-like CNN;
-   autoencoder;
-   small ResNet.

### Tier 2 --- graph topology

-   Inception-style multi-scale block;
-   U-Net;
-   Siamese network;
-   DenseNet-style connectivity.

### Tier 3 --- modern sequence models

-   original Transformer;
-   GPT-style decoder;
-   ViT;
-   encoder-decoder Transformer with cross-attention.

### Tier 4 --- richer objectives/state

-   VAE;
-   SimCLR;
-   MoCo;
-   LoRA fine-tuning;
-   knowledge distillation.

### Tier 5 --- lifecycle/differentiation

-   GAN;
-   WGAN-GP;
-   EMA teacher/student;
-   tiny diffusion training loop.

### Tier 6 --- dynamic/sparse/research pressure

-   small Mixture-of-Experts;
-   recurrent/iterative model;
-   toy RL policy/value setup;
-   meta-learning example if differentiation permits.

The goal is not a huge model. Use tiny dimensions while preserving the
algorithm's structural semantics.

------------------------------------------------------------------------

# Part IX --- Frontier-Model Thought Experiment

## 31. Tiny frontier-style decoder

Construct a tiny but structurally realistic modern decoder containing,
where supported:

-   token embedding;
-   positional strategy;
-   RMSNorm;
-   grouped-query or multi-query attention;
-   gated MLP;
-   residuals;
-   repeated blocks;
-   tied or untied LM head;
-   causal masking;
-   long-context shape variables.

Then progressively add:

-   rotary position encoding implemented at tensor level;
-   KV cache semantics for inference;
-   MoE layer;
-   top-k routing;
-   auxiliary routing loss;
-   quantized or low-precision execution policy where relevant.

This challenge is intended to expose the boundary between:

``` text
architecture
tensor mathematics
persistent inference state
dynamic routing
execution policy
```

Do not distort language semantics merely to make a frontier-model
feature parse.

------------------------------------------------------------------------

## 32. Agent-training thought experiment

Do **not** attempt to implement a coding-agent application in TENSA.

Instead test the ML boundary by representing a toy learned agent:

``` text
observation
→ policy/model
→ action
→ external environment
→ next observation
→ reward
→ trajectory
→ update
```

The filesystem, shell, Git, network and process execution belong to an
external environment.

Ask whether TENSA can naturally represent:

-   trajectory construction;
-   policy outputs;
-   rewards;
-   value estimates;
-   rollout state;
-   objective;
-   update relationship.

This is a test of whether TENSA is a language for **learning systems**,
not a demand that it become a general-purpose application language.

------------------------------------------------------------------------

# Part X --- Execution and Distribution Boundary

## 33. Backend-leakage audit

Search all TENSA examples for concepts such as:

``` text
cuda
device copies
rank
world size
all_reduce
framework module registration
autocast blocks
state_dict
requires_grad
zero_grad
backward
optimizer.step
```

Determine why each appears.

If it represents execution machinery rather than an ML choice, it should
normally not be present in model/tensor code.

------------------------------------------------------------------------

## 34. Distribution thought experiments

Without necessarily implementing distributed execution yet, take several
programs and design hypothetical execution plans for:

-   data parallelism;
-   tensor parallelism;
-   pipeline parallelism;
-   expert parallelism;
-   parameter/optimizer sharding;
-   activation checkpointing.

Ask:

> Can these be expressed as execution/distribution policy over the
> existing IR without changing the mathematical TENSA program?

If not, identify exactly which semantic information the IR lacks.

Do not prematurely introduce communication primitives into
architecture-level code.

------------------------------------------------------------------------

# Part XI --- Compiler and Tooling Integrity

## 35. Inspect must be authoritative

For every major test, verify:

``` text
source
  ↓
typed/elaborated program
  ↓
IR
  ├── inspect
  ├── emit
  └── runtime
```

`inspect` must derive from the executable IR, not independently
reconstruct the graph from source.

Specifically compare:

-   node connectivity;
-   shapes;
-   branch structure;
-   parameter ownership;
-   sharing;
-   effects;
-   state;
-   objective graph;
-   lifecycle plan.

A disagreement is a correctness bug even if runtime output happens to
look plausible.

------------------------------------------------------------------------

## 36. Diagnostic hardening

For every diagnostic family, deliberately trigger it.

Evaluate diagnostics on:

-   correctness;
-   source location;
-   architectural location;
-   relevant axis;
-   expected versus inferred expression;
-   suggested repair;
-   absence of backend jargon.

A useful diagnostic should answer:

> What did I say, what did TENSA infer, why are they incompatible, and
> where should I look?

Keep a diagnostic-quality regression suite.

------------------------------------------------------------------------

## 37. Metamorphic tests

Create pairs of programs that should be semantically equivalent.

Examples:

``` text
implicit sequence
```

versus:

``` text
pipeline
```

or a catalog operation versus its tensor-level reference implementation.

Other useful transformations:

-   rename irrelevant bindings;
-   reorder independent declarations;
-   replace repeated explicit stage applications with a bound shared
    stage where sharing is intended;
-   algebraically equivalent dimension expressions;
-   explicit versus inferred incoming dimension assertion if supported.

Compare IR semantics and numerical results where appropriate.

------------------------------------------------------------------------

## 38. Property-based testing

Generate valid and invalid combinations of:

-   tensor ranks;
-   symbolic dimensions;
-   broadcasting;
-   reshape;
-   concat;
-   matmul;
-   residual;
-   split/merge;
-   repetition.

Useful properties include:

-   reshape preserves element count;
-   concat output dimension equals symbolic sum;
-   residual result shape equals input shape;
-   parameter count is invariant under irrelevant renaming;
-   shared stage parameter count does not depend on application count;
-   independent stage parameter count does;
-   `inspect` parameter totals equal IR parameter-table totals.

------------------------------------------------------------------------

# Part XII --- PyTorch Comparison

## 39. Semantic-compression audit

For representative challenges, maintain an idiomatic PyTorch
implementation.

Classify both implementations into:

1.  architecture intent;
2.  mathematical algorithm;
3.  data/learning-problem definition;
4.  training/lifecycle policy;
5.  execution policy;
6.  framework/runtime plumbing;
7.  general-purpose host-language plumbing.

TENSA should primarily reduce 6 and 7.

It should make 1--5 clearer, not hide them.

Track LOC/token count only as secondary evidence.

------------------------------------------------------------------------

## 40. "What information disappeared?" review

For every major TENSA abstraction ask:

> What information present in the PyTorch implementation disappeared?

For each removed item classify it:

-   inferred consequence --- desirable;
-   framework machinery --- desirable;
-   backend machinery --- desirable;
-   meaningful mathematical choice --- dangerous;
-   learning-problem definition --- dangerous;
-   temporal/lifecycle choice --- dangerous.

This is the primary defense against semantic compression becoming
semantic concealment.

------------------------------------------------------------------------

# Part XIII --- Change-Control Protocol

## 41. When to fix the compiler

Fix immediately when:

-   documented syntax has incorrect semantics;
-   IR contradicts language rules;
-   `inspect` disagrees with IR/runtime;
-   parameter identity is wrong;
-   shape reasoning is unsound;
-   a diagnostic reports false certainty;
-   state/checkpoint behavior violates specification.

These are **F** issues, not language-design opportunities.

------------------------------------------------------------------------

## 42. When to extend a backend

Prefer backend work when:

-   the IR already represents the program correctly;
-   semantics are clear;
-   static checking succeeds;
-   only execution or gradient support is missing.

These are **H** issues.

Do not change the language to work around a weak backend.

------------------------------------------------------------------------

## 43. When to change the language

Require evidence that:

1.  the issue is primarily **G --- missing abstraction**;
2.  at least two or preferably three substantially different algorithms
    expose it;
3.  the abstraction represents a recurring ML concept;
4.  it cannot be handled naturally through existing composition;
5.  it improves semantic knowledge, not just character count;
6.  it does not hide a meaningful choice;
7.  it does not introduce general-purpose machinery unnecessarily.

For every proposed language addition write:

``` text
Observed failures:
Common underlying concept:
Why existing constructs are insufficient:
Proposed semantics:
Static knowledge gained:
New ambiguity introduced:
Interaction with existing constructs:
Programs simplified:
Programs made harder:
Alternative rejected:
```

Syntax comes **after** this analysis.

------------------------------------------------------------------------

# Part XIV --- Hardening Milestones

## 44. Milestone 1 --- Semantic core

Required before moving on:

-   symbolic dimensions;
-   cursor semantics;
-   topology;
-   parameter identity;
-   repetition;
-   multi-stream behavior;
-   objectives.

All microtests and evil twins pass.

------------------------------------------------------------------------

## 45. Milestone 2 --- Tensor completeness

Required:

-   explicit attention;
-   normalization;
-   masking;
-   reshaping;
-   tensor-level custom mathematics;
-   architecture/tensor equivalence tests.

TENSA must demonstrate that researchers can descend below catalog
abstractions without leaving the language.

------------------------------------------------------------------------

## 46. Milestone 3 --- Learning-program completeness

Required:

-   data contracts;
-   fitted transforms/leakage checks;
-   multi-objective learning;
-   persistent state;
-   lifecycle phases;
-   alternating optimization;
-   checkpoint semantics.

At this point TENSA should describe more than model structure.

------------------------------------------------------------------------

## 47. Milestone 4 --- Research pressure

Required attempts:

-   U-Net;
-   Transformer;
-   VAE;
-   SimCLR;
-   GAN;
-   WGAN-GP;
-   LoRA;
-   diffusion;
-   MoE;
-   recurrence.

Not every program must work without an escape hatch. Every failure must
be understood and classified.

------------------------------------------------------------------------

## 48. Milestone 5 --- Boundary validation

Attempt:

-   frontier-style decoder;
-   agent-learning toy;
-   distributed execution thought experiments.

The objective is to establish what TENSA deliberately **does not** own.

A mature language has a clear boundary, not infinite scope.

------------------------------------------------------------------------

# Part XV --- Hardening Report

## 49. Final report structure

At the end of each major hardening cycle produce:

### What TENSA expresses exceptionally well

Concrete examples only.

### What remains inherently complex

Do not treat these as failures.

### Accidental friction still present

Rank by recurrence across examples.

### Hidden information discovered

These are high priority even if examples currently "work."

### Backend leakage discovered

Identify the correct abstraction boundary.

### Missing abstractions

Only include findings supported by multiple challenges.

### Backend capability gaps

Keep separate from language shortcomings.

### Escape-hatch usage

For every use, explain why it was necessary.

### Rejected language changes

Record these. Prevent future agents from repeatedly rediscovering and
proposing the same bad ideas.

### PyTorch comparison

Where TENSA provides semantic leverage and where Python remains more
natural.

### Next hardening targets

Choose them based on uncovered weaknesses, not novelty.

------------------------------------------------------------------------

# 50. Success criteria

TENSA is not hardened because every famous architecture can be written in
it.

It is hardened when the following are true:

-   valid programs have predictable semantics;
-   invalid programs fail early and intelligibly;
-   symbolic reasoning never silently invents equality;
-   inspection faithfully represents executable IR;
-   parameter identity is unsurprising;
-   topology cannot silently lower into another topology;
-   state ownership and checkpoint behavior are explicit;
-   learning targets are never guessed;
-   data leakage is detectable where the language has enough
    information;
-   routine training mechanics disappear without hiding meaningful
    update behavior;
-   recurrence, repetition and lifecycle are clearly distinct;
-   tensor mathematics remains available when architecture abstractions
    run out;
-   backend gaps remain backend gaps;
-   escape hatches have explicit boundaries;
-   repeated hard cases produce coherent abstractions rather than piles
    of special syntax.

The ultimate test is not:

> **Can TENSA implement this model?**

It is:

> **When TENSA implements this learning system, does the source primarily
> describe the learning system itself?**

Whenever the answer is no, determine precisely what the remaining code
is describing.

That is the next hardening challenge.
