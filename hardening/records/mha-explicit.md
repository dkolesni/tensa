# Challenge record — mha-explicit (tier 2, §12 explicit attention)

```text
Challenge:                  mha-explicit (tensor.ts) + cross-attention-padding,
                            twins split-swapped / key-not-transposed / mask-wrong-length /
                            projection-width; metamorphic mha-manual-vs-catalog,
                            cross-attention-manual-vs-catalog, causal-flag-vs-explicit-mask
Source/paper:               Vaswani et al. 2017, "Attention Is All You Need" — scaled
                            dot-product attention with head split/merge; §12 asks that the
                            catalog `attention` op be *reproducible from primitives*
TENSA features stressed:     §5.3 reshape/transpose with symbolic dims, §5.4 derived dim
                            `DH = D / Heads`, §11 pure tensor `fn`s, §8 block-owned params,
                            §12 catalog-vs-explicit equivalence, masks (causal + padding)
Expected semantics:         [B, T, D] → project → [B, T, Heads, DH] → transpose → scores
                            [B, Heads, T, T] / √DH → causal mask → softmax → context →
                            transpose back → [B, T, D] → output projection. Numerically
                            identical to `attention(heads: Heads, causal: true)` when both
                            programs draw the same four D×D parameter tables.
TENSA implementation:
    dim B
    dim T
    dim D = 16
    dim Heads = 2
    dim DH = D / Heads

    fn split_heads(x: Tensor[B, T, D]) -> Tensor[B, Heads, T, DH] {
      return transpose(reshape(x, [B, T, Heads, DH]), 1, 2)
    }
    fn merge_heads(x: Tensor[B, Heads, T, DH]) -> Tensor[B, T, D] {
      return reshape(transpose(x, 1, 2), [B, T, D])
    }
    fn sdpa(q: Tensor[B, Heads, T, DH], k: Tensor[B, Heads, T, DH], v: Tensor[B, Heads, T, DH]) -> Tensor[B, Heads, T, DH] {
      let scores = matmul(q, transpose(k, 2, 3)) / sqrt(DH)
      let masked = masked_fill(scores, causal_mask(T), value: -1000000000.0)
      let weights = softmax(masked, axis: -1)
      return matmul(weights, v)
    }
    block MHA(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
      param wq: Tensor[D, D] init: xavier
      param wk: Tensor[D, D] init: xavier
      param wv: Tensor[D, D] init: xavier
      param wo: Tensor[D, D] init: xavier
      let q = split_heads(matmul(x, wq))
      let k = split_heads(matmul(x, wk))
      let v = split_heads(matmul(x, wv))
      return matmul(merge_heads(sdpa(q, k, v)), wo)
    }
    model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { MHA() }
Reference (PyTorch, hand-written):
    class MHA(nn.Module):
        def __init__(s, D=16, H=2):
            super().__init__(); s.H, s.DH = H, D // H
            s.wq = nn.Parameter(torch.empty(D, D)); s.wk = nn.Parameter(torch.empty(D, D))
            s.wv = nn.Parameter(torch.empty(D, D)); s.wo = nn.Parameter(torch.empty(D, D))
        def split(s, x): B, T, _ = x.shape; return x.view(B, T, s.H, s.DH).transpose(1, 2)
        def forward(s, x):
            B, T, _ = x.shape
            q, k, v = s.split(x @ s.wq), s.split(x @ s.wk), s.split(x @ s.wv)
            att = (q @ k.transpose(2, 3)) / math.sqrt(s.DH)
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), -1e9)
            ctx = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
            return ctx @ s.wo
Check result:               ok, no diagnostics. Output Tensor[B, T, 16]. Every constraint
                            [proved]: `DH = D / Heads` normalises to 8 with D, Heads
                            constant; reshape element counts B·T·16 = B·T·2·8 are proved
                            polynomially; the fn result contracts bind per application.
Inspect result:             parameters: 4 tables, 1,024 values, single owner `M/MHA#1`
                            (4 · 16 · 16 = 1,024 — matches the reference; the catalog
                            program `attention(heads: 2)` also reports 4 tables / 1,024).
IR result:                  ops reshape, transpose, matmul, softmax, masked_fill,
                            causal_mask, const_dim — no `attention`, no `linear`, and no
                            custom op anywhere (noCustomOps check, §11 claim holds).
Runtime result:             B=2, T=5 → forward M -> [2, 5, 16].
Metamorphic result:         mha-manual-vs-catalog: identical parameter sequence, outputs
                            equal to 1e-4 at B=2, T=5. cross-attention-manual-vs-catalog
                            (padding mask Mask[B, 1, S], S=3) equal to 1e-4.
                            causal-flag-vs-explicit-mask (`causal: true` vs
                            `mask: causal_mask(T)`) equal to 1e-5.
Backend result:             emitted forward binds `dims = resolve_dims({**self.dims,
                            "B": x.shape[0], "T": x.shape[1]})` and lowers the split to
                            `.reshape(dims["B"], dims["T"], 2, 8)`; the causal mask to
                            `torch.triu(torch.ones(dims["T"], dims["T"], …), 1)`.
                            Before this challenge every symbolic reshape extent printed
                            as `-1` (two of them in one view → invalid), `causal_mask`
                            lowered to `?? 1`, and the attention `mask` port was dropped
                            → H-005, fixed.
Diagnostics (twins):
    split-swapped           reshape [B, T, DH, Heads] → the fn's declared result
                            Tensor[B, Heads, T, DH] refutes 8 ≠ 2 at the first
                            application (AXS0401, not carried).
    key-not-transposed      matmul(q, k): contraction DH = 8 against T — unprovable for
                            symbolic T, so carried (AXS0403) and refused at runtime when
                            T ≠ 8 is bound. The checker is honest here: T = 8 really
                            would run (and be wrong) — see E-007 for the class.
    mask-wrong-length       causal_mask(DH) against scores [.., T, T] → carried 8 = T
                            (AXS0403), refused for T = 5.
    projection-width        wo: Tensor[D, DH] → output 8 vs declared 16 → AXS0401.
    (not a twin)            merge_heads WITHOUT the transpose — reshape [B, Heads, T, DH]
                            straight to [B, T, D] — has the right element count; no shape
                            rule can see it. Recorded as E-007; the metamorphic pair
                            catches it numerically (tests.ts backend/E-007).
Ceremony classification (§10 / §36):
    essential   — `fn split_heads/merge_heads` (layout is the whole content of MHA; the
                  declared result types are what turn the swapped split into a static
                  error), `dim DH = D / Heads` (names the derived width once),
                  `/ sqrt(DH)` (the scale is a semantic choice).
    incidental  — `value: -1000000000.0` (a literal for −∞; a `-inf` literal would read
                  better — noted, not proposed), `matmul(x, wq)` instead of `linear` (only
                  because the challenge forbids `linear` to stay parameter-count aligned
                  with the catalog op; ordinary programs would use `linear(D)`).
    accidental  — none. Nothing here needed a custom op, an escape hatch, or a language
                  change; the catalog `attention` is fully reproducible from primitives.
Finding classification:     H (H-005 emitter); F (F-012 mask contract, F-013 optional ports —
                            found by the cross-attention companion challenge); E (E-007)
Severity:                   H-005 major (generated module did not run for any head split
                            with a symbolic axis); F-012/F-013 major
Workaround:                 before the fixes: none for the emitter; for the checker, write
                            the mask as `Mask[T, T]` and pass every optional port
Proposed action:            DONE — see F-012, F-013, H-005 records
Language change required?:  no
Regression test added?:     yes — challenge:mha-explicit (compile/IR/inspect/emit/run +
                            4 twins), challenge:cross-attention-padding (+ 4 twins),
                            metamorphic mha-manual-vs-catalog /
                            cross-attention-manual-vs-catalog / causal-flag-vs-explicit-mask /
                            head-split-via-fn-vs-inline, backend tests H-005 and E-007
```
