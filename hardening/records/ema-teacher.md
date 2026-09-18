# Challenge record — ema-teacher (Milestone 3, §14 state, §26 item 4)

```text
Challenge:                  ema-teacher (+ twin one-decl-two-aliases)
Source/paper:               Mean Teacher (Tarvainen & Valpola 2017), BYOL, DINO — a
                            teacher whose weights are an exponential moving average of
                            the student's
TENSA features stressed:     two model declarations of one `block`, `track … = ema(…)`,
                            `stop_grad` inside an objective, optimizer `over` one model,
                            checkpoint kinds
TENSA implementation:
    block Body(x: Tensor[B, D]) -> Tensor[B, E] { linear(8)  gelu  linear(E) }
    model Student(x: Tensor[B, D]) -> Tensor[B, E] { Body() }
    model Teacher(x: Tensor[B, D]) -> Tensor[B, E] { Body() }
    objective Distill(s, t) -> Scalar { let target = stop_grad(t)  return mean((s-target)^2) }
    train T {
      model student = Student
      model teacher = Teacher
      track ema_t = ema(student, rate: 0.99)
      loss d = Distill(s: student(x), t: teacher(x))
      optimizer opt = adamw(lr: 1e-3) over student
      epochs 1
    }
Check result:               ok, no warnings. 8 parameter tables (Student and Teacher each
                            own a Body), 4 state slots (the tracked shadow, one per
                            student table). `inspect` lists `ema_t`.
Runtime result:             3 steps; teacher tables never optimised; the shadow moves
                            with the student (decay 0.99, E-009).
Reference result:           emitted `ema_t = {k: student.p[k].detach().clone() …}` and
                            per step `v.mul_(0.99).add_(student.p[k].detach(),
                            alpha=0.01)`; runs under torch.
Twin:                       `model teacher = Student` → AXS0706 (F-015): one declaration
                            bound twice is one parameter set.
What the program CANNOT say: the point of the paper. The teacher's forward pass should
                            use the shadow weights `ema_t`, and the program has no way
                            to say "run Teacher with ema_t's values". `track` produces a
                            derived parameter set that only checkpoints can see; the
                            model that consumes it is written as a second, independent
                            copy that never receives the average. The challenge is
                            honest about this in its title — it pins everything the
                            language does say, and records the gap as G-cand-003.
Finding classification:     G-cand-003 (derived parameter sets) — see
                            proposals/derived-parameter-sets.md; F-015 fixed en route
Severity:                   major as a research gap; not a bug in what exists
Proposed action:            evidence collected (this, distillation teacher refresh,
                            meta-learning inner loop); §43 analysis written, no syntax
Language change required?:  candidate — proposal stage only
Regression test added?:     yes — `challenge:ema-teacher` (paramTables, stateSlots,
                            inspect, emit, run, twin)
```
