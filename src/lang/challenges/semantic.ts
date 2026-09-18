/**
 * TENSA hardening — Milestone 1 "semantic core" challenges (§5–§10, §16).
 *
 * Same shape as corpus.ts: programs are test data.  Each challenge targets one
 * section of tensa_hardening_plan.md and carries the evil twins that show the
 * checker does not merely accept — it discriminates.
 */
import { IRModule } from "../ir";
import { allNodes, splitInvariant } from "./driver";
import { Challenge, CustomCheck } from "./types";

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

/** Every recorded constraint was proved (no `assumed`, no `failed`) and at least one exists. */
const allProved: CustomCheck = {
  name: "every equality is proved, none merely assumed (§5.2)",
  check: (mod: IRModule) => {
    const bad = mod.constraints.filter((c) => c.status !== "proved");
    assert(bad.length === 0, `not proved: ${bad.map((c) => `${c.origin} (${c.status})`).join("; ")}`);
    return `${mod.constraints.length} constraints, all proved`;
  },
};

/** The carried (`assumed`) constraints are exactly the given origins — nothing extra is guessed. */
function carriedExactly(origins: string[]): CustomCheck {
  return {
    name: "carried constraints are exactly the unprovable ones",
    check: (mod: IRModule) => {
      const got = mod.constraints.filter((c) => c.status === "assumed").map((c) => c.origin);
      assert(got.length === origins.length && origins.every((o) => got.includes(o)), `carried: ${got.join("; ") || "(none)"}; expected: ${origins.join("; ")}`);
      return `${got.length} carried: ${got.join("; ")}`;
    },
  };
}

function nodeCount(op: string, n: number): CustomCheck {
  return {
    name: `IR contains exactly ${n} '${op}' node(s)`,
    check: (mod: IRModule) => {
      const got = allNodes(mod).filter((x) => x.op === op).length;
      assert(got === n, `found ${got} '${op}' nodes`);
      return `${got} × ${op}`;
    },
  };
}

export const SEMANTIC_CHALLENGES: Challenge[] = [
  // ================================================================ §5 symbolic dimensions
  {
    id: "algebra-proved",
    title: "algebraic identities the checker must prove without help",
    section: "§5.1",
    code: `dim B
dim T
dim C
dim H
dim D
dim K
dim W = 4 * K

model Div(x: Tensor[B, W]) -> Tensor[B, W] { return reshape(reshape(x, [B, 4, W / 4]), [B, W]) }
model Sum(x: Tensor[B, T + 1 - 1]) -> Tensor[B, T] { return x }
model Flat(x: Tensor[B, C, H, H]) -> Tensor[B, C * H * H] { flatten }
model Mixed(x: Tensor[B, D]) -> Tensor[B, 2 * D + 6] { return concat(x, x, axis: -1) |> linear(2 * D + 6) }
model Pool(x: Tensor[B, C, H, H]) -> Tensor[B, C, H / 2, H / 2] { maxpool2d(2) }
model Stride(x: Tensor[B, C, H, H]) -> Tensor[B, 8, (H + 1) / 2, (H + 1) / 2] { conv2d(8, kernel: 3, stride: 2, pad: 1) }`,
    expect: {
      warnCodes: [],
      outputShape: {
        Div: "Tensor[B, 4*K]",
        Flat: "Tensor[B, C*H^2]",
        Mixed: "Tensor[B, 2*D + 6]",
        Pool: "Tensor[B, C, ⌊H/2⌋, ⌊H/2⌋]",
        Stride: "Tensor[B, 8, ⌊(H + 1)/2⌋, ⌊(H + 1)/2⌋]",
      },
      custom: [allProved],
      run: { dims: { B: 2, T: 3, C: 2, H: 5, D: 6, K: 3 }, outputs: { Div: "[2, 12]", Sum: "[2, 3]", Flat: "[2, 50]", Mixed: "[2, 18]", Pool: "[2, 2, 2, 2]", Stride: "[2, 8, 3, 3]" } },
    },
    twins: [
      {
        id: "off-by-one",
        mutates: "declared width is 2*D+5 while the body produces 2*D+6",
        code: `dim B
dim D
model Mixed(x: Tensor[B, D]) -> Tensor[B, 2 * D + 5] { return concat(x, x, axis: -1) |> linear(2 * D + 6) }`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 3,
      },
      {
        id: "flatten-wrong-power",
        mutates: "flatten of [C, H, H] declared as C*H, not C*H*H",
        code: `dim B
dim C
dim H
model Flat(x: Tensor[B, C, H, H]) -> Tensor[B, C * H] { flatten }`,
        expectCodes: ["AXS0403"],
        forbidCodes: ["AXS0401"],
      },
      {
        id: "floor-is-not-exact",
        mutates: "stride-2/pad-1 conv declared as H/2 instead of (H+1)/2 — differs for odd H",
        code: `dim B
dim H
model Stride(x: Tensor[B, 3, H, H]) -> Tensor[B, 8, H / 2, H / 2] { conv2d(8, kernel: 3, stride: 2, pad: 1) }`,
        expectCodes: ["AXS0403"],
        forbidCodes: ["AXS0401"],
      },
    ],
  },

  {
    id: "carried-holds",
    title: "an unprovable equality is carried, reported once, and verified when dimensions bind",
    section: "§5.2",
    code: `dim B
dim S
dim T
dim D = 8
model M(a: Tensor[B, S, D], b: Tensor[B, T, D]) -> Tensor[B, S, D] { return a + b }`,
    expect: {
      warnCodes: ["AXS0403"],
      constraints: ["assumed"],
      custom: [carriedExactly(["add broadcast axis 1"])],
      run: { dims: { B: 2, S: 4, T: 4 }, outputs: { M: "[2, 4, 8]" } },
    },
    twins: [],
  },
  {
    id: "carried-violated",
    title: "the same program refuses to execute when the bound dimensions break the carried equality",
    section: "§5.2",
    code: `dim B
dim S
dim T
dim D = 8
model M(a: Tensor[B, S, D], b: Tensor[B, T, D]) -> Tensor[B, S, D] { return a + b }`,
    expect: {
      warnCodes: ["AXS0403"],
      run: { dims: { B: 2, S: 5, T: 3 }, refuses: true, errorContains: "S = T but 5 ≠ 3" },
    },
    twins: [],
  },
  {
    id: "impossible-constant",
    title: "a constant contradiction is an error, not a carried assumption",
    section: "§5.2",
    code: `dim B
model M(a: Tensor[B, 64], b: Tensor[B, 128]) -> Tensor[B, 64] { return a + b }`,
    expect: { ok: false, constraints: ["failed"] },
    twins: [
      {
        id: "is-an-error",
        mutates: "(same program) — must be AXS0401 with no AXS0403 hedge",
        code: `dim B
model M(a: Tensor[B, 64], b: Tensor[B, 128]) -> Tensor[B, 64] { return a + b }`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 2,
      },
    ],
  },
  {
    id: "unknown-shape",
    title: "lost shape knowledge stays lost: no rank is invented downstream of a shapeless custom op",
    section: "§5.2",
    code: `dim B
dim T
dim D = 8
custom op mystery(x: Tensor[B, T, D]) {
  effects: pure
  shape: unknown
}
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return mystery(x) |> linear(D) |> gelu }`,
    expect: {
      warnCodes: ["AXS0903"],
      outputShape: { M: "Tensor[?]" },
      custom: [
        {
          name: "nothing after the custom op is sized or constrained",
          check: (mod) => {
            assert(mod.params.length === 0, `parameters were sized from an unknown shape: ${mod.params.map((p) => p.id).join(", ")}`);
            assert(mod.constraints.length === 0, `constraints were derived from an unknown shape: ${mod.constraints.map((c) => c.origin).join("; ")}`);
            return "0 parameters, 0 constraints";
          },
        },
      ],
    },
    twins: [
      {
        id: "declared-shape-restores-checking",
        mutates: "custom op declares `-> Tensor[B, T, 4]`, so the following linear(D) result is checked and the declared result fails",
        code: `dim B
dim T
dim D = 8
custom op mystery(x: Tensor[B, T, D]) -> Tensor[B, T, 4] {
  effects: pure
}
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return mystery(x) |> linear(D) |> reshape([B, T * D]) }`,
        expectCodes: ["AXS0406"],
        forbidCodes: ["AXS0903"],
      },
    ],
  },

  {
    id: "divisibility-ladder",
    title: "attention head split: exact when D is constant or a multiple, carried when D is free",
    section: "§5.4",
    code: `dim B
dim T
dim D
dim Heads = 4
dim W = 16 * Heads
model ConstExact(x: Tensor[B, T, 64]) -> Tensor[B, T, 64] { attention(heads: Heads) }
model SymbolicExact(x: Tensor[B, T, W]) -> Tensor[B, T, W] { attention(heads: Heads) }
model Carried(x: Tensor[B, T, D]) -> Tensor[B, T, D] { attention(heads: Heads) }
model ReshapeCarried(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let h = reshape(x, [B, T, Heads, D / Heads])
  return reshape(h, [B, T, D])
}`,
    expect: {
      warnCodes: ["AXS0403"],
      constraints: ["assumed"],
      custom: [
        carriedExactly([
          "attention: attention head split (heads must divide D)",
          "reshape: reshape element count",
          "reshape: reshape element count",
        ]),
      ],
      run: { dims: { B: 2, T: 3, D: 8 }, outputs: { ConstExact: "[2, 3, 64]", SymbolicExact: "[2, 3, 64]", Carried: "[2, 3, 8]", ReshapeCarried: "[2, 3, 8]" } },
    },
    twins: [
      {
        id: "constant-indivisible",
        mutates: "D = 10 with 4 heads: divisibility is refutable at compile time",
        code: `dim B
dim T
dim D = 10
dim Heads = 4
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { attention(heads: Heads) }`,
        expectCodes: ["AXS0407"],
        forbidCodes: ["AXS0403"],
        line: 5,
      },
    ],
  },
  {
    id: "divisibility-violated",
    title: "a free D that is not a multiple of the head count is refused when bound, before any forward pass",
    section: "§5.4",
    code: `dim B
dim T
dim D
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { attention(heads: 4) }`,
    expect: {
      warnCodes: ["AXS0403"],
      run: { dims: { B: 2, T: 3, D: 10 }, refuses: true, errorContains: "heads must divide D" },
    },
    twins: [],
  },

  // ================================================================ §6 implicit flow
  {
    id: "flow-cursor-rules",
    title: "where the implicit value comes from, and where it does not exist",
    section: "§6",
    code: `dim B
fn act(x: Tensor[B, 8]) -> Tensor[B, 8] { gelu }
block Two(a: Tensor[B, 8], b: Tensor[B, 8]) -> Tensor[B, 8] { return a + b |> linear(8) }
model Seeded(x: Tensor[B, 8]) -> Tensor[B, 8] { return act(x) }
model Explicit(x: Tensor[B, 8]) -> Tensor[B, 8] { return Two(x, x) }
model Chain(x: Tensor[B, 8]) -> Tensor[B, 8] {
  linear(8)
  gelu
  let side = mean(x, axis: 0)
  return side + x
}`,
    expect: {
      warnCodes: [],
      outputShape: { Seeded: "Tensor[B, 8]", Explicit: "Tensor[B, 8]", Chain: "Tensor[B, 8]" },
      run: { dims: { B: 2 }, outputs: { Seeded: "[2, 8]", Explicit: "[2, 8]", Chain: "[2, 8]" } },
    },
    twins: [
      {
        id: "two-inputs-bare",
        mutates: "a two-input block uses a bare op: no single implicit value exists",
        code: `dim B
block Two(a: Tensor[B, 8], b: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { return Two(x, x) }`,
        expectCodes: ["AXS0301"],
        line: 2,
      },
      {
        id: "zero-inputs-bare",
        mutates: "a model with no inputs starts with a bare op",
        code: `dim B
model M() -> Tensor[B, 8] { linear(8) }`,
        expectCodes: ["AXS0301"],
        line: 2,
      },
      {
        id: "tuple-then-bare",
        mutates: "a tuple expression is the most recent value; the next bare op has no tensor to consume",
        code: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 8] {
  let h = linear(16)
  (h |> linear(8), h |> linear(4))
  linear(8)
}`,
        expectCodes: ["AXS0408"],
        line: 5,
      },
      {
        id: "destructure-then-bare",
        mutates: "`let (a, b) = Two(x)` is followed by a bare op — the tuple, not a stale cursor, is current (F-007)",
        code: `dim B
model Two(x: Tensor[B, 16]) -> (Tensor[B, 8], Tensor[B, 4]) {
  let h = linear(16)
  return (h |> linear(8), h |> linear(4))
}
model M(x: Tensor[B, 16]) -> Tensor[B, 8] {
  let (a, b) = Two(x)
  linear(8)
}`,
        expectCodes: ["AXS0408"],
        line: 8,
      },
      {
        id: "branch-local-escape",
        mutates: "a name bound inside a branch is used after the split",
        code: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 8] {
  split merge add {
    branch a { let t = linear(16) ; gelu }
    branch b { linear(16) }
  }
  return t |> linear(8)
}`,
        expectCodes: ["AXS0201"],
        line: 7,
      },
      {
        id: "let-redirects-cursor",
        mutates: "a side `let` of a [16] value redirects the implicit flow; the following bare op continues from it (E-003)",
        code: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 16] {
  linear(16)
  let s = mean(x, axis: 0)
  gelu
}`,
        expectCodes: ["AXS0406"],
        line: 2,
      },
    ],
  },

  // ================================================================ §7 topology
  {
    id: "topology-nesting",
    title: "split/residual nest in every order; every merge kind lowers to its own op",
    section: "§7.1 / §7.2",
    code: `dim B
model NestedSplit(x: Tensor[B, 8, 8, 8]) -> Tensor[B, 24, 8, 8] {
  split merge concat(1) {
    conv2d(8, kernel: 1)
    branch inner {
      split merge concat(1) { conv2d(8, kernel: 1) ; conv2d(8, kernel: 3, pad: 1) }
    }
  }
}
model SplitInResidual(x: Tensor[B, 8, 8, 8]) -> Tensor[B, 8, 8, 8] {
  residual {
    split merge add { conv2d(8, kernel: 1) ; conv2d(8, kernel: 3, pad: 1) }
  }
}
model ResidualInSplit(x: Tensor[B, 8, 8, 8]) -> Tensor[B, 16, 8, 8] {
  split merge concat(1) {
    branch a { residual { conv2d(8, kernel: 3, pad: 1) } }
    branch b { conv2d(8, kernel: 1) }
  }
}
model NestedResidual(x: Tensor[B, 8]) -> Tensor[B, 8] {
  residual {
    linear(8)
    residual { linear(8) ; gelu }
  }
}
model OuterNamed(x: Tensor[B, 8]) -> Tensor[B, 8] {
  let skip = linear(8)
  gelu
  split merge mean {
    linear(8)
    branch b { linear(8) + skip }
  }
}
model Add3(x: Tensor[B, 8]) -> Tensor[B, 8] {
  split merge add { linear(8) ; gelu ; linear(8) }
}`,
    expect: {
      warnCodes: [],
      irOps: ["parallel", "concat", "residual", "add", "merge_mean", "merge_add"],
      custom: [
        { name: "split invariant: every branch reads the split input, one merge consumes all (§7.1)", check: splitInvariant },
        nodeCount("residual", 4),
        nodeCount("parallel", 6),
      ],
      emitContains: [") / 2", "torch.cat"],
      emitForbids: [".mean("],
      run: {
        dims: { B: 2 },
        outputs: {
          NestedSplit: "[2, 24, 8, 8]",
          SplitInResidual: "[2, 8, 8, 8]",
          ResidualInSplit: "[2, 16, 8, 8]",
          NestedResidual: "[2, 8]",
          OuterNamed: "[2, 8]",
          Add3: "[2, 8]",
        },
      },
    },
    twins: [
      {
        id: "residual-spatial-mismatch",
        mutates: "residual body halves the spatial extent without a projection",
        code: `dim B
model M(x: Tensor[B, 8, 8, 8]) -> Tensor[B, 8, 4, 4] {
  residual { maxpool2d(2) }
}`,
        expectCodes: ["AXS0404"],
        forbidCodes: ["AXS0401"],
        line: 3,
      },
      {
        id: "add-merge-width-mismatch",
        mutates: "`merge add` over branches of different widths",
        code: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] {
  split merge add { linear(8) ; linear(4) }
}`,
        expectCodes: ["AXS0405"],
      },
      {
        id: "unknown-merge",
        mutates: "merge kind `max` does not exist",
        code: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] {
  split merge max { linear(8) ; linear(8) }
}`,
        expectCodes: ["AXS0204"],
      },
    ],
  },

  // ================================================================ §8 parameter identity
  {
    id: "identity-paths",
    title: "a bound stage is one parameter set wherever it is applied; a rebound name is a different one",
    section: "§8",
    code: `dim B
dim T
dim F = 4
block E(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) }
block Cell(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) ; tanh }
model Branches(x: Tensor[B, 8]) -> Tensor[B, 16] {
  let e = E()
  split merge concat(-1) { e ; branch b { gelu ; e } }
}
model InScan(x: Tensor[B, T, F]) -> Tensor[B, T, 8] {
  let cell = Cell()
  let (outs, last) = scan over x axis: 1 carry h: Tensor[B, 8] init: zeros {
    let hn = cell(concat(step, h, axis: -1) |> linear(8))
    yield hn
  }
  return outs
}
model Rebound(x: Tensor[B, 8]) -> Tensor[B, 8] {
  let e = E()
  let y = e(x)
  let e = E()
  return e(y)
}
model Unbound(x: Tensor[B, 8]) -> Tensor[B, 8] {
  E()
  E()
}`,
    expect: {
      // Rebound deliberately rebinds `e` in one scope: independent parameters (F-001) and a warning (E-001 → AXS0304)
      warnCodes: ["AXS0304"],
      paramOwners: ["Branches/e/linear#1", "InScan/cell/linear#1", "InScan/linear#3", "Rebound/e/linear#1", "Rebound/e#2/linear#1", "Unbound/E#1/linear#1", "Unbound/E#2/linear#1"],
      paramTables: 14,
      sharedGroups: 1,
      inspectContains: ["shared"],
      custom: [
        {
          name: "the shared stage inside the scan is applied once per scan, not per step",
          check: (mod) => {
            const p = mod.params.find((x) => x.owner === "InScan/cell/linear#1")!;
            assert(!!p, "no parameter for InScan/cell");
            assert(p.applications.length === 1, `cell parameters record ${p.applications.length} applications`);
            return `applications: ${p.applications.length}`;
          },
        },
      ],
      run: { dims: { B: 2, T: 3 }, outputs: { Branches: "[2, 16]", InScan: "[2, 3, 8]", Rebound: "[2, 8]", Unbound: "[2, 8]" } },
    },
    twins: [
      {
        id: "shared-shape-conflict",
        mutates: "one bound stage is applied to two different widths",
        code: `dim B
dim E = 8
block Enc(x: Tensor[B, N]) -> Tensor[B, E] { linear(E) }
model M(a: Tensor[B, 4], b: Tensor[B, 6]) -> (Tensor[B, E], Tensor[B, E]) {
  let enc = Enc()
  return (enc(a), enc(b))
}`,
        expectCodes: ["AXS0801"],
      },
    ],
  },

  // ================================================================ §9 static repetition
  {
    id: "repetition-forms",
    title: "every repetition form unrolls to the parameter and state count it implies",
    section: "§9",
    code: `dim B
block L(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) }
block S(x: Tensor[B, 8]) -> Tensor[B, 8] { batchnorm }
model Nested(x: Tensor[B, 8]) -> Tensor[B, 8] { for 2 { for 3: L() } }
model Shaped(x: Tensor[B, 8]) -> Tensor[B, 64] { for i in 0..3 { linear(16 * (i + 1)) ; gelu } ; linear(64) }
model StateIndep(x: Tensor[B, 8]) -> Tensor[B, 8] { for 3: S() }
model StateShared(x: Tensor[B, 8]) -> Tensor[B, 8] { let s = S() ; for 3: s }
model Zero(x: Tensor[B, 8]) -> Tensor[B, 8] { for 0: L() }
model InResidual(x: Tensor[B, 8]) -> Tensor[B, 8] { residual { for 2: L() } }`,
    expect: {
      warnCodes: [],
      paramOwners: ["Nested/L#6/linear#1", "Shaped/linear#7", "StateIndep/S#3/batchnorm#1", "StateShared/s/batchnorm#1", "InResidual/L#2/linear#1"],
      stateSlots: 8,
      custom: [
        {
          name: "unrolled owners: Nested has 6, StateShared has 1, Zero has none",
          check: (mod) => {
            const owners = (m: string) => new Set(mod.params.filter((p) => p.owner.startsWith(m + "/")).map((p) => p.owner));
            assert(owners("Nested").size === 6, `Nested owners: ${[...owners("Nested")].join(", ")}`);
            assert(owners("StateShared").size === 1, `StateShared owners: ${[...owners("StateShared")].join(", ")}`);
            assert(owners("Zero").size === 0, `Zero owners: ${[...owners("Zero")].join(", ")}`);
            assert(owners("Shaped").size === 4, `Shaped owners: ${[...owners("Shaped")].join(", ")}`);
            return "6 / 1 / 0 / 4";
          },
        },
      ],
      emitContains: ["zero repetitions: identity"],
      run: {
        dims: { B: 2 },
        outputs: { Nested: "[2, 8]", Shaped: "[2, 64]", StateIndep: "[2, 8]", StateShared: "[2, 8]", Zero: "[2, 8]", InResidual: "[2, 8]" },
      },
    },
    twins: [
      {
        id: "count-changes-shape",
        mutates: "`for 2` over a halving body while the declared result assumes one halving",
        code: `dim B
model M(x: Tensor[B, 8, 4, 4]) -> Tensor[B, 8, 2, 2] { for 2: maxpool2d(2) }`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 2,
      },
      {
        id: "symbolic-count",
        mutates: "repetition count is a free dimension: the grammar itself demands a literal (repetition is static)",
        code: `dim B
dim N
block L(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { for N: L() }`,
        expectCodes: ["AXS0102"],
      },
    ],
  },

  // ================================================================ §10 multi-stream (tier 2)
  {
    id: "unet-skip",
    title: "U-Net: encoder/decoder skip connections across a downsample/upsample pair",
    section: "§10",
    tier: 2,
    record: "records/unet-skip.md",
    code: `dim B
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
}`,
    expect: {
      warnCodes: [],
      outputShape: { UNet: "Tensor[B, 2, 4*K, 4*K]" },
      irOps: ["concat", "upsample", "maxpool2d"],
      paramTables: 12,
      custom: [allProved],
      emitContains: ["F.interpolate", "torch.cat"],
      run: { dims: { B: 1, K: 2 }, outputs: { UNet: "[1, 2, 8, 8]" } },
    },
    twins: [
      {
        id: "free-height-carried",
        mutates: "H is free (not declared a multiple of 4): 2*⌊H/2⌋ = H is carried, not proved",
        code: `dim B
dim H
model UNet(x: Image[B, 3, H, H]) -> Tensor[B, 2, H, H] {
  let e1 = x |> conv2d(8, kernel: 3, pad: 1) |> relu
  let e2 = e1 |> maxpool2d(2) |> conv2d(16, kernel: 3, pad: 1) |> relu
  return concat(e2 |> upsample(2), e1, axis: 1) |> conv2d(2, kernel: 1)
}`,
        expectCodes: ["AXS0403"],
        forbidCodes: ["AXS0401", "AXS0405"],
      },
      {
        id: "skip-from-wrong-level",
        mutates: "decoder concatenates the skip from the wrong encoder level (spatial 2*⌊H/4⌋ vs H)",
        code: `dim B
dim K
dim H = 4 * K
model UNet(x: Image[B, 3, H, H]) -> Tensor[B, 2, H, H] {
  let e1 = x |> conv2d(8, kernel: 3, pad: 1) |> relu
  let e2 = e1 |> maxpool2d(2) |> conv2d(16, kernel: 3, pad: 1) |> relu
  let e3 = e2 |> maxpool2d(2) |> conv2d(32, kernel: 3, pad: 1) |> relu
  return concat(e3 |> upsample(2), e1, axis: 1) |> conv2d(2, kernel: 1)
}`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
      },
      {
        id: "concat-on-spatial-axis",
        mutates: "skip is concatenated on axis 2 (height) instead of the channel axis",
        code: `dim B
dim K
dim H = 4 * K
model UNet(x: Image[B, 3, H, H]) -> Tensor[B, 2, H, H] {
  let e1 = x |> conv2d(8, kernel: 3, pad: 1) |> relu
  let e2 = e1 |> maxpool2d(2) |> conv2d(8, kernel: 3, pad: 1) |> relu
  return concat(e2 |> upsample(2), e1, axis: 2) |> conv2d(2, kernel: 1)
}`,
        expectCodes: ["AXS0401"],
      },
    ],
  },
  {
    id: "unet-odd-input",
    title: "U-Net with a free height refuses an odd input instead of producing a misaligned concat",
    section: "§10",
    tier: 2,
    code: `dim B
dim H
model UNet(x: Image[B, 3, H, H]) -> Tensor[B, 2, H, H] {
  let e1 = x |> conv2d(8, kernel: 3, pad: 1) |> relu
  let e2 = e1 |> maxpool2d(2) |> conv2d(16, kernel: 3, pad: 1) |> relu
  return concat(e2 |> upsample(2), e1, axis: 1) |> conv2d(2, kernel: 1)
}`,
    expect: {
      warnCodes: ["AXS0403"],
      run: { dims: { B: 1, H: 5 }, refuses: true, errorContains: "2*⌊H/2⌋ = H but 4 ≠ 5" },
    },
    twins: [],
  },

  // ================================================================ §16 objectives
  {
    id: "objective-graph",
    title: "objectives are typed graphs: multi-task weights, an auxiliary term, and one objective over two models",
    section: "§16",
    code: `dim B
dim K = 4
model MT(x: Tensor[B, 16]) -> (Logits[B, K], Tensor[B, 1]) {
  let h = linear(32) |> gelu
  return (h |> linear(K), h |> linear(1))
}
model EncA(x: Tensor[B, 16]) -> Tensor[B, 8] { linear(8) }
model EncB(x: Tensor[B, 16]) -> Tensor[B, 8] { linear(8) }
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
objective Val(pred: Tensor[B, 1], target: Tensor[B, 1]) -> Scalar { return mse(pred, target) }
objective Reg(h: Tensor[B, 8]) -> Scalar { return l2(h) }
objective Agree(a: Tensor[B, 8], b: Tensor[B, 8]) -> Scalar { return mse(a, b) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
    field value: Tensor[1] = decode(target) |> to_float
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = MT
  model A = EncA
  model Bm = EncB
  loss cls = Cls(logits: Net(x)[0], labels: label) weight: 1.0
  loss val = Val(pred: Net(x)[1], target: value) weight: 0.3
  loss reg = Reg(h: A(x)) weight: 0.01
  loss agree = Agree(a: A(x), b: Bm(x))
  optimizer opt = adamw(lr: 1e-3) over Net
  optimizer opt_a = adamw(lr: 1e-3) over A
  optimizer opt_b = adamw(lr: 1e-3) over Bm
  phase main { epochs 1 ; update cls with opt ; update val with opt ; update reg with opt_a ; update agree with opt_a ; update agree with opt_b }
}`,
    expect: {
      warnCodes: [],
      inspectContains: ["cls", "agree"],
      emitContains: ["* 0.3", "* 0.01"],
      custom: [
        {
          name: "every loss binds every port of its objective",
          check: (mod) => {
            const plan = mod.plans[0];
            assert(!!plan, "no training plan");
            for (const l of plan.losses) {
              const obj = mod.objectives.find((o) => o.name === l.objective)!;
              assert(l.bindings.length === obj.inputs.length, `loss ${l.name} binds ${l.bindings.length}/${obj.inputs.length} ports`);
            }
            return `${plan.losses.length} losses fully bound`;
          },
        },
      ],
      run: { dims: { B: 2 }, steps: 3, gradAll: true },
    },
    twins: [
      {
        id: "kind-mismatch-model",
        mutates: "a Probs result (already soft-maxed) is bound to a Logits port of a cross-entropy objective",
        code: `dim B
dim K = 4
model Plain(x: Tensor[B, 16]) -> Probs[B, K] { linear(K) ; softmax }
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = Plain
  loss cls = Cls(logits: Net(x), labels: label)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        expectCodes: ["AXS0602"],
        forbidCodes: ["AXS0401"],
      },
      {
        id: "missing-port",
        mutates: "loss binding omits `labels`",
        code: `dim B
dim K = 4
model Net0(x: Tensor[B, 16]) -> Logits[B, K] { linear(K) }
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = Net0
  loss cls = Cls(logits: Net(x))
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        expectCodes: ["AXS0503"],
        line: 17,
      },
      {
        id: "misspelled-port",
        mutates: "loss binding names a port `logit` that the objective does not have (F-010: previously vanished silently)",
        code: `dim B
dim K = 4
model Net0(x: Tensor[B, 16]) -> Logits[B, K] { linear(K) }
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = Net0
  loss cls = Cls(logit: Net(x), labels: label)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        expectCodes: ["AXS0503"],
        line: 17,
      },
      {
        id: "shape-mismatch",
        mutates: "an 8-wide encoder output is bound to a 1-wide regression port",
        code: `dim B
model Enc(x: Tensor[B, 16]) -> Tensor[B, 8] { linear(8) }
objective Val(pred: Tensor[B, 1], target: Tensor[B, 1]) -> Scalar { return mse(pred, target) }
source S = synthetic(features: 16)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field value: Tensor[1] = decode(target) |> to_float
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = Enc
  loss val = Val(pred: Net(x), target: value)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        // a refuted binding axis is a contract failure between the model and
        // the objective, so it carries the contract code (E-008)
        expectCodes: ["AXS0602"],
        forbidCodes: ["AXS0403", "AXS0401"],
      },
      {
        id: "tuple-without-index",
        mutates: "a two-output model is bound to a port without selecting an output (F-008)",
        code: `dim B
dim K = 4
model MT(x: Tensor[B, 16]) -> (Logits[B, K], Tensor[B, 1]) {
  let h = linear(32) |> gelu
  return (h |> linear(K), h |> linear(1))
}
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = MT
  loss cls = Cls(logits: Net(x), labels: label)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        expectCodes: ["AXS0408"],
        line: 20,
      },
      {
        id: "unknown-field",
        mutates: "loss binding refers to a data field that does not exist",
        code: `dim B
dim K = 4
model Net0(x: Tensor[B, 16]) -> Logits[B, K] { linear(K) }
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 16, classes: 4)
data D from S {
  example {
    field x: Tensor[16] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train J {
  data D
  model Net = Net0
  loss cls = Cls(logits: Net(x), labels: lbl)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`,
        expectCodes: ["AXS0604"],
        line: 17,
      },
      {
        id: "non-scalar-objective",
        mutates: "objective returns a per-example vector, not a Scalar",
        code: `dim B
objective Bad(a: Tensor[B, 8], b: Tensor[B, 8]) -> Scalar { return a - b }`,
        expectCodes: ["AXS0406"],
      },
    ],
  },
];
