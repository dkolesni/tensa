# Challenge record — unet-skip (tier 2, §10 multi-stream)

```text
Challenge:                  unet-skip (semantic.ts) + unet-odd-input, twins
                            free-height-carried / skip-from-wrong-level / concat-on-spatial-axis
Source/paper:               Ronneberger et al. 2015, "U-Net" — encoder/decoder with
                            skip connections concatenated on the channel axis
TENSA features stressed:     §5 symbolic dims (floordiv from pooling, derived dims),
                            §6 implicit flow vs. explicit `let` for multi-use values,
                            §7 topology (two skip paths + one trunk, explicit concat),
                            §8 parameter identity (a block applied twice, no sharing),
                            §10 multi-stream merge, §12 backend consistency
Expected semantics:         e1 and e2 are used twice (trunk + skip); the decoder must
                            match the encoder spatially at every level; the checker
                            should prove H/2·2 = H when H is a declared multiple of 4,
                            and carry it (then enforce it at runtime) when H is free
TENSA implementation:
    dim B
    dim K
    dim H = 4 * K
    dim W = 4 * K

    block Down(x: Tensor[B, C, H0, W0]) -> Tensor[B, C, H0 / 2, W0 / 2] { maxpool2d(2) }

    model UNet(x: Image[B, 3, H, W]) -> Tensor[B, 2, H, W] {
      let e1 = x |> conv2d(8, kernel: 3, pad: 1) |> relu
      let e2 = e1 |> Down() |> conv2d(16, kernel: 3, pad: 1) |> relu
      let e3 = e2 |> Down() |> conv2d(32, kernel: 3, pad: 1) |> relu
      let d2 = concat(e3 |> upsample(2), e2, axis: 1) |> conv2d(16, kernel: 3, pad: 1) |> relu
      let d1 = concat(d2 |> upsample(2), e1, axis: 1) |> conv2d(8, kernel: 3, pad: 1) |> relu
      return d1 |> conv2d(2, kernel: 1)
    }
Reference (PyTorch, hand-written):
    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv2d(3, 8, 3, padding=1);  s.c2 = nn.Conv2d(8, 16, 3, padding=1)
            s.c3 = nn.Conv2d(16, 32, 3, padding=1)
            s.u2 = nn.Conv2d(32 + 16, 16, 3, padding=1); s.u1 = nn.Conv2d(16 + 8, 8, 3, padding=1)
            s.out = nn.Conv2d(8, 2, 1)
        def forward(s, x):
            e1 = F.relu(s.c1(x)); e2 = F.relu(s.c2(F.max_pool2d(e1, 2))); e3 = F.relu(s.c3(F.max_pool2d(e2, 2)))
            d2 = F.relu(s.u2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1)))
            d1 = F.relu(s.u1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1)))
            return s.out(d1)
Check result:               ok, no diagnostics. Output Tensor[B, 2, 4*K, 4*K].
                            6 constraints, all [proved]: the block's template dims
                            C/H0/W0 bind per application (Down#3: C=8, H0=W0=4K;
                            Down#6: C=16, H0=W0=2K), and ⌊4K/2⌋ normalises to 2K so
                            the concat extents are proved without carrying.
Inspect result:             graph shows two `apply[stage=UNet/Down#3 | #6]` nodes,
                            two upsample and two concat nodes with channel sums
                            32+16 and 16+8; parameters: 12 tables, 14,714 values
                            (6 conv weight/bias pairs — matches the reference count
                            3·8·9+8 + 8·16·9+16 + 16·32·9+32 + 48·16·9+16 + 24·8·9+8 + 8·2+2 = 14,714).
                            No sharing table entries: the two Down applications own
                            no parameters, so nothing is (or could wrongly be) tied.
IR result:                  ops maxpool2d ×2, upsample ×2, concat ×2, conv2d ×6, relu ×5.
                            Skips are ordinary value reuse (e1/e2 consumed twice); no
                            special construct.
Runtime result:             K=2, B=1 → forward UNet -> [1, 2, 8, 8]. Matches reference shape.
Gradient result:            n/a (no training plan in this challenge; the objective-graph
                            challenge covers gradient coverage).
Backend result:             emitted PyTorch uses F.interpolate(scale_factor=2, mode="nearest")
                            and torch.cat([...], dim=1). Before this challenge the module
                            header emitted `STATIC_DIMS = {"H": 4*K}` — `K` is a bare
                            Python name that does not exist at import time → H-003, fixed:
                            constants stay in STATIC_DIMS, derived dims are computed by
                            `resolve_dims()` after the runtime bindings are supplied, and
                            symbolic extents lower to `4 * dims["K"]` / `dims["H"] // 2`
                            rather than the non-existent key `dims["4*K"]`.
Diagnostics (twins):
    free-height-carried     dim H free → AXS0403 "2*⌊H/2⌋ = H" carried (no false error);
                            unet-odd-input then refuses H=5 at runtime:
                            "2*⌊H/2⌋ = H but 4 ≠ 5" — before any forward pass.
    skip-from-wrong-level   concat(e3 |> upsample(2), e1): 2K vs 4K → AXS0401 (refuted,
                            not carried) — required the same-sign refutation added to
                            `eqDim` (§5.2: "2K − 4K is never zero for K ≥ 1").
    concat-on-spatial-axis  axis: 2 → channel extents 8 vs 8 fine, height 4K vs 4K fine,
                            but the *other* spatial axis / declared output no longer
                            matches → AXS0401 at the concat.
Ceremony classification (§10 / §36):
    essential   — `let e1/e2` (a value used twice must be named), `concat(..., axis: 1)`
                  (the merge axis is a semantic choice), `dim H = 4 * K` (the program
                  really does require divisibility by 4; making that explicit is the
                  point — with a free H the checker carries and the runtime enforces).
    incidental  — the `Down` block needs template dims (C, H0, W0) to stay generic; a
                  bare `maxpool2d(2)` inline would do the same with less text. Kept in
                  the challenge to exercise block-level template binding.
    accidental  — none found. `upsample` was missing from the catalog and was added as
                  library vocabulary (no grammar change).
Finding classification:     H (backend: H-003; catalog gap H-004 `upsample`); language: none
Severity:                   H-003 major for the emitter (generated file would not import
                            whenever a derived dim referenced a runtime dim)
Workaround:                 before the fix: declare every dim as a constant
Proposed action:            DONE (emit_torch.ts `pyDim`, `resolve_dims`); catalog `upsample`
Language change required?:  no
Regression test added?:     yes — challenge:unet-skip (compile/IR/inspect/emit/run + 3 twins),
                            challenge:unet-odd-input, backend / "symbolic extents lower
                            to Python expressions, not dictionary keys (H-003)",
                            metamorphic carried-dim-vs-declared-multiple
```
