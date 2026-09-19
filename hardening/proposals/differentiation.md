# G-cand-002 — differentiation analysis (§43)

2026-09-18. Analysis only. No syntax, AST changes or grad catalog stub.

## Separate the witnesses before naming a concept

| Witness | Differentiated value | Derivative destination | Higher order? | Other missing ownership |
|---|---|---|---|---|
| Ordinary parameter optimization | scalar objective, w.r.t. optimizer region | optimizer step, not a graph value | no | already supported |
| MAML inner parameter gradient | inner loss, w.r.t. theta | temporary theta' = theta - alpha*g | yes for full MAML; deliberately no for first-order MAML | evaluation with theta', G-cand-003 |
| WGAN-GP penalty | critic scores at interpolated input, w.r.t. that input | norm penalty inside outer critic objective | must differentiate through the input derivative into critic parameters | retain/consume derivative graph; interpolation and batch semantics |
| FGSM/PGD input gradient | task loss, w.r.t. input | adversarial input construction | usually detach attack step; not universally | repeated forwards, clipping/projection, randomness/mode |

Only WGAN-GP has an executed M4 rejection: `AXS0204: unknown operation 'grad'`.
MAML is a mathematical witness, explicitly skipped as a complete source program; adversarial input
perturbation was the M3 boundary witness. Do not count those as three successfully compiled experiments.

## Observed failures
WGAN-GP cannot express its penalty, M3 adversarial training cannot construct its input perturbation,
and MAML cannot compute an inner parameter update or evaluate the outer loss at that update.
Ordinary parameter gradients already work and are NOT evidence that ordinary lifecycle is missing.

## Common underlying concept
A scoped differentiation transformation whose selected inputs/parameters, seed/cotangent, graph
retention, effects and result ownership are known. This is broader than a magic `grad` function,
and separate from the ability to evaluate a model using derived parameter values.

## Why existing constructs are insufficient
Objectives return scalars; lifecycle updates consume their gradients internally. stop_grad can cut
a path but cannot expose a derivative. scan shares recurrence parameters; it does not create an
inner optimizer tape. track stores averages but cannot supply functional parameters to a forward.
A custom backend region can do the math, but the current CPU identity substitute cannot validate it.

## Proposed semantics (requirements, not an approved design)
Distinguish tensor-input derivatives from parameter-region derivatives. Require explicit scalar
reduction or VJP cotangent; require an explicit differentiability/retention choice for derivative
results. Pure graph differentiation should not silently replay state writes or stochastic draws.
Lifecycle still owns ordinary optimization; derivative values belong in graph/region semantics,
not an arbitrary imperative training loop. Functional parameter evaluation composes with, but is
not subsumed by, differentiation. First-order MAML must never silently replace full MAML.

## Static knowledge gained
Which values/regions a derivative ranges over; derivative shapes; whether higher-order paths exist;
which state/random effects need capture/replay; which optimizer can consume resulting parameter sets.

## New ambiguity introduced
Vector score sum vs per-example Jacobian (batch coupling/BN makes these different); materialized
Jacobian vs VJP; graph lifetime; zero derivative vs unused input; repeated random draws; BN writes;
stop_grad across derivative boundaries; source vs derived parameters in region selectors.
These are unresolved. There is no coherent implementable differentiation model approved yet.

## Interaction with existing constructs
stop_grad must keep its meaning under nested differentiation. Effects must survive derivative
regions. Region over selects parameter identity, not copied tensor storage. scan needs a differentiable
carry and explicit higher-order policy. Derived parameter sets need functional evaluation and buffer
ownership, not mutation of a live optimized model. Checkpoints must not accidentally persist tapes.

## Programs simplified
WGAN-GP, adversarial perturbation, Jacobian regularization; MAML only when G-cand-003 also lands.
Gradient reversal and custom backward are related pressure tests, not evidence that one operator
already covers them.

## Programs made harder
Simple training if graph-retention policy leaks into every objective; stateful/stochastic models if
replay restrictions are overbroad. Preserve the existing implicit ordinary backward path.

## Alternative rejected
Add `grad(loss,wrt)` immediately; put all differentiation into lifecycle; call host autograd from
the emitter without IR representation; substitute first-order MAML; treat a custom identity reference
as successful validation. None resolves the witnesses' distinct retention/ownership needs.

## Evidence bars / next experiment
Recurring need and inadequacy of current composition are established. Meaningful choices and effect
semantics (§43 bars 6–7) are not settled. Next: tiny batch-coupled critic and stopped PGD examples with
explicit numerical VJP references, then a functional single-step parameter experiment. MAML remains
skipped-with-reason until those converge. No language change authorized.
