# Challenge record — curriculum-events (Milestone 3, §25 schedules and events)

```text
Challenge:                  curriculum-events (+ twin until-unknown-metric)
Source/paper:               curriculum learning: an "easy" phase that ends on a loss
                            threshold, a "hard" phase with its own warmup
TENSA features stressed:     `until <metric> < <value>`, per-phase `linear(warmup:)`
                            schedule, phase-level and plan-level `every N steps`
TENSA implementation:
    phase easy { epochs 10  until main < 0.9  update main with opt }
    phase hard { steps 6  linear(warmup: 2)  update main with opt
                 every 3 steps { validate } }
    every 5 steps { checkpoint }
Check result:               ok, no warnings.
Runtime result:             `easy` stopped by `until` well before its 60-step budget;
                            `hard` ran exactly 6 steps, lr multiplier 0.5 at step 0
                            (warmup 2) and decaying; 2 `validate` events, both in
                            `hard`; checkpoints at global steps with (step+1) % 5 == 0.
Reference result:           emitted plan: `if metrics.get("main", float("inf")) < 0.9:
                            # until main < 0.9` → `stop = True; break`; `schedule =
                            {"kind": "linear", "args": {"warmup": 2}}`; `if phase_step
                            % 3 == 0:` and `if step % 5 == 0:` (H-008).
Twin:                       `until val_acc > 0.9` → AXS0709 (F-018).
What was learnt:            H-001 had to be closed for this challenge to be checkable
                            at all; H-008 fell out of comparing the two backends on the
                            same program.
Finding classification:     F-018, H-001, H-008 (all fixed)
Severity:                   major (F-018), minor (H-008)
Proposed action:            DONE
Language change required?:  no
Regression test added?:     yes — `challenge:curriculum-events`; tests.ts lifecycle
                            (H-001, H-008)
```
