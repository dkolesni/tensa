# Challenge record — moco (supported subset, not a complete MoCo)

Challenge: MoCo memory bank + momentum encoder pressure, tier 4, 2026-09-18.
Source/paper: He et al., Momentum Contrast (2020).
TENSA features stressed: FIFO persistent state [4,4], two BN encoders, stopped keys, positive/negative logits, parameter EMA track, checkpoint ownership.
Expected semantics: a query encoder learns; keys use momentum-derived parameters; old bank supplies negatives before enqueue; query/key BN policy must be explicit.
TENSA implementation: `research.ts`, `moco`. Queue and contrastive objective execute. Key is an INDEPENDENT encoder; momentum track cannot be consumed. This is deliberately NOT reported as faithful MoCo. Positive target column zero is supplied by the host GPU fixture; the generic synthetic reference driver is only a structural simulation.
Check result: accepted without warnings; queue overrun → AXS0410.
Inspect result: 12 parameter tables; 11 state slots = 4 encoder BN buffers + bank + 6 tracked parameter shadows. Checkpoint kinds: parameters, persistent state, optimizer state, lifecycle position.
IR result: state read/update, stop_grad, matmul/concat; no assignment connects the tracked shadow to Key.
Runtime result: custom test proves eval leaves queue unchanged; two train calls shift [ones,ones] → [ones,keys3] → [keys3,keys4]. Source and key own distinct BN buffers.
Reference result: CUDA eval/train forward, parameter gradients, state comparison, checkpoint load and emitted plan pass for this subset. Running-var comparison found F-029.
Gradient result: query-only optimizer; keys detached. No claim that gradient correctness repairs the wrong key weights.
Diagnostics: AXS0410 twin; missing derived-set consumption has no diagnostic because the independent-key source is legal.
Finding classification: G-cand-003 bar 6 concretely hit; F-029 fixed; B for FIFO capacity, temperature and BN policy.
Severity: major research limitation.
Workaround: host-driven momentum encoder and buffer policy; not hidden in the emitter.
Proposed action: update derived-parameter-sets.md; distinguish independent BN, copied BN, averaged buffers and evaluation mode. MoCo often uses batch/shuffled BN: parameter EMA does not specify any of these.
Language change required?: proposal pending, not implemented.
Regression test added?: challenge:moco state test; backend F-029; GPU gate.
