# Challenge record — <id>

Copy this file to `hardening/records/<id>.md` for every nontrivial challenge
(protocol §4). Keep the TENSA program **and** the reference implementation when
one exists; link the corpus entry in `src/lang/challenges/corpus.ts` by id.

```text
Challenge:
Source/paper:
TENSA features stressed:
Expected semantics:
TENSA implementation:            (corpus id, or inline program)
Check result:
Inspect result:
IR result:
Runtime result:
Reference result:
Gradient result:
Diagnostics:
Finding classification:         (A–H, see protocol §3)
Severity:                       (blocker / major / minor / cosmetic)
Workaround:
Proposed action:
Language change required?:      (no / yes → hardening/proposals/<id>.md, §43)
Regression test added?:         (test group + name)
```
