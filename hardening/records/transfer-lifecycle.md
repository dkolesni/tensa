# Challenge record — transfer-lifecycle (Milestone 3, §22 lifecycle, §26 item 4)

```text
Challenge:                  transfer-lifecycle (+ 7 twins)
Source/paper:               transfer learning: freeze the encoder, warm the head, then
                            unfreeze with a small encoder learning rate
TENSA features stressed:     structural regions (`Net.encoder`, `Net.head` resolve to the
                            `let` names inside the model), per-region optimizers,
                            persistent `freeze` across phases, `unfreeze`, per-phase
                            `lr <region> = …`, `every 1 epochs { validate ; checkpoint }`
TENSA implementation:
    model Transfer(image) -> Logits[B, K] {
      let encoder = Backbone()   let head = Head()   image |> encoder |> head
    }
    train Finetune {
      optimizer head_opt = adamw(lr: 1e-3) over Net.head
      optimizer enc_opt  = adamw(lr: 1e-5) over Net.encoder
      phase warmup   { epochs 2  freeze Net.encoder  update main with head_opt }
      phase finetune { epochs 3  unfreeze Net.encoder  lr Net.encoder = 0.00001
                       update main with head_opt, enc_opt
                       every 1 epochs { validate ; checkpoint } }
    }
Check result:               ok, no warnings.
Runtime result:             two phases, 12 + 18 steps (6 synthetic steps per epoch);
                            encoder tables frozen throughout warmup, updated in
                            finetune; three epoch-end `validate` events, all in
                            `finetune`.
Reference result:           emitted plan: `head_opt_params = [Net.p[k] for k in
                            ["Transfer_head_linear…`, `p.requires_grad_(False)  # freeze
                            Net.encoder`, `set_lr(enc_opt, [Net.p[k] for k in
                            ["Transfer_encoder_conv2d…`, `if (epoch + 1) % 1 == 0:`.
                            Forbidden: `region_params(model` (the H-006 bug shape).
Twins:                      unknown-region → AXS0701; overlapping-optimizers → AXS0703;
                            frozen-but-updated → AXS0704 (F-014, new);
                            freeze-and-unfreeze, epochs-and-steps, two-until,
                            lr-on-frozen → AXS0707 "conflicting phase configuration"
                            (new code: one phase, two incompatible instructions).
Finding classification:     F-014 fixed; AXS0707 added as a contract (no separate F —
                            the checks did not exist)
Severity:                   major (F-014)
Proposed action:            DONE
Language change required?:  no
Regression test added?:     yes — `challenge:transfer-lifecycle`
```
