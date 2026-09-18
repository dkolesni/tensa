# Challenge record — encoder-decoder

Challenge: encoder–decoder Transformer, tier 3, 2026-09-18.
Source/paper: Vaswani et al. (2017), tiny attention/norm core with distinct source and target lengths; not a tokenization benchmark.
TENSA features stressed: Encoder and Decoder graphs, recursive loss binding dec(q, enc(x,pad),pad), causal self-attention, cross-attention with Mask[B,1,S].
Expected semantics: true blocks keys; rank-three padding masks lift over heads, then negate for SDPA. S=3, T=2 make swapped sequence axes observable.
TENSA implementation: `research.ts`, `encoder-decoder`.
Check result: accepted; wrong source mask length → AXS0401.
Inspect result: 18 parameter tables across the two model roots.
IR result: three attention sites; both source-padding masks are connected; decoder self-attention is causal.
Runtime result: both graphs execute and all parameters receive updates through the nested binding.
Reference result: CUDA forward, gradients and emitted plan pass. Gaussian synthetic Masks initially disagreed with bool conversion, exposing F-024. All-blocked rows exposed F-025.
Gradient result: aligned forward-graph parameter gradients pass; reference gradAll passes on training.
Diagnostics: AXS0401 twin; no warnings in parent.
Finding classification: F-024, F-025 fixed.
Severity: major backend fidelity.
Workaround: none needed after fixes.
Proposed action: retain boolean fixture and zero-context empty-row regressions.
Language change required?: no.
Regression test added?: challenge:encoder-decoder; backend F-024/F-025; GPU gate.
