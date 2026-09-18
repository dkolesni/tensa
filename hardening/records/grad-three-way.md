# Challenge record — grad-three-way (Milestone 3, §15 effects, §16 gradient)

```text
Challenge:                  grad-three-way
Source/paper:               —  the three ways a parameter ends up without a gradient
TENSA features stressed:     `argmax |> one_hot` (non-differentiable), `stop_grad`,
                            `custom op` with a torch backend string and no reference
                            implementation; gradient coverage report
TENSA implementation:
    custom op weird(x: Tensor[B, K]) -> Tensor[B, K] { effects: pure  backend torch: "weird(x)" }
    model NonDiff(x)  { x |> linear(K) |> argmax(axis: -1) |> one_hot(classes: K) |> linear(K) }
    model Stopped(x)  { x |> linear(K) |> stop_grad |> linear(K) }
    model Foreign(x)  { x |> linear(K) |> weird |> linear(K) }
    (three losses, one region-less optimizer — region `*`, F-022)
Check result:               ok; no warnings. Module effects {nondiff, grad-stopped}.
                            AXS0901 (INFO, not a warning) names `weird`: "custom op has
                            no reference backend; the reference runtime substitutes the
                            identity". The severity is deliberate — a custom op with a
                            torch backend is a legitimate program, the note just says
                            what `run` will do with it.
Runtime result:             gradient coverage after 2 steps:
                              NonDiff/linear#1  no gradient path (argmax breaks it)
                              Stopped/linear#1  no gradient path (stop_grad)
                              Foreign/linear#1  updated (identity substitute keeps the
                                                path; AXS0901 said so at compile time)
                            The three first layers are told apart by (a) the effect the
                            IR carries on the breaking node — `nondiff` vs `grad-stopped`
                            — and (b) the compile-time info for the third.
Reference result:           emitted module calls `weird(x)`; the torch harness fails
                            with NameError, as it must — the backend function is the
                            user's to supply. Recorded as the one expected harness
                            failure (19/19 + 1 by design).
Finding classification:     none new; pins AXS0901's severity (E — info is the right
                            level; a `warnCodes: []` expectation plus a custom check
                            asserts that it is NOT a warning)
Severity:                   —
Proposed action:            none
Language change required?:  no
Regression test added?:     yes — `challenge:grad-three-way` (effects, AXS0901 info, run
                            coverage split)
```
