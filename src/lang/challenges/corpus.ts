/**
 * TENSA hardening — challenge corpus.
 *
 * Programs here are test data, not pedagogy: keep the Playground's EXAMPLES
 * small and curated; put every torture case, evil twin and tier challenge here.
 * Section tags refer to tensa_hardening_plan.md.
 */
import { LEARNING_CHALLENGES } from "./learning";
import { RESEARCH_CHALLENGES, RESEARCH_METAMORPHIC } from "./research";
import { SEMANTIC_CHALLENGES } from "./semantic";
import { TENSOR_CHALLENGES, TENSOR_METAMORPHIC } from "./tensor";
import { Challenge } from "./types";

const BASE_CHALLENGES: Challenge[] = [
  // ---------------------------------------------------------------- §7.1 split/merge
  {
    id: "split-merge-3",
    title: "three-way split with explicit concat merge",
    section: "§7.1",
    code: `dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 96, 32, 32] {
  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3, pad: 1)
    conv2d(32, kernel: 5, pad: 2)
  }
}`,
    expect: {
      irOps: ["parallel", "concat"],
      paramTables: 6,
      inspectContains: ["concat"],
      emitContains: ["torch.cat"],
      run: { dims: { B: 1 }, outputs: { M: "96, 32, 32" } },
    },
    twins: [
      {
        id: "spatial-mismatch",
        mutates: "second branch drops pad so spatial extent is 30, not 32",
        code: `dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 64, 32, 32] {
  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3)
  }
}`,
        expectCodes: ["AXS0405"],
      },
      {
        id: "no-merge",
        mutates: "split without a merge clause",
        code: `dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 64, 32, 32] {
  split {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3, pad: 1)
  }
}`,
        expectCodes: ["AXS0102"],
      },
    ],
  },

  // ---------------------------------------------------------------- §7.2 residual
  {
    id: "residual-projected",
    title: "projected residual lowers to body + projection regions and an explicit add",
    section: "§7.2",
    code: `dim B
model M(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual via conv2d(64, kernel: 1, stride: 2) {
    conv2d(64, kernel: 3, stride: 2, pad: 1)
    batchnorm
  }
}`,
    expect: {
      irOps: ["residual", "add"],
      stateSlots: 2,
      effects: ["reads-state", "writes-state"],
      emitContains: ["+"],
      run: { dims: { B: 1 }, outputs: { M: "64, 16, 16" } },
    },
    twins: [
      {
        id: "missing-projection",
        mutates: "body changes shape but no `via` projection is given",
        code: `dim B
model M(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual { conv2d(64, kernel: 3, stride: 2, pad: 1) }
}`,
        expectCodes: ["AXS0404"],
        forbidCodes: ["AXS0401"],
        line: 3,
      },
      {
        id: "projection-wrong-stride",
        mutates: "projection keeps stride 1 so skip and body disagree spatially",
        code: `dim B
model M(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual via conv2d(64, kernel: 1) {
    conv2d(64, kernel: 3, stride: 2, pad: 1)
  }
}`,
        expectCodes: ["AXS0404"],
      },
    ],
  },

  // ---------------------------------------------------------------- §8 parameter identity
  {
    id: "siamese-identity",
    title: "bound stage applied twice shares one parameter set; recreated stage does not",
    section: "§8",
    code: `dim B
dim E = 8

block Encoder(x: Tensor[B, 4]) -> Tensor[B, E] {
  linear(E)
  gelu
}

model Tied(a: Tensor[B, 4], b: Tensor[B, 4]) -> (Tensor[B, E], Tensor[B, E]) {
  let enc = Encoder()
  return (enc(a), enc(b))
}

model Untied(a: Tensor[B, 4], b: Tensor[B, 4]) -> (Tensor[B, E], Tensor[B, E]) {
  return (Encoder(a), Encoder(b))
}`,
    expect: {
      paramTables: 6,
      paramCount: 3 * (4 * 8 + 8),
      sharedGroups: 1,
      inspectContains: ["shared"],
      emitContains: ["shared by 2 applications"],
      run: { dims: { B: 2 }, outputs: { Tied: "[2, 8]", Untied: "[2, 8]" } },
    },
    twins: [
      {
        id: "shape-conflict",
        mutates: "the shared stage is applied to two incompatible input widths",
        code: `dim B
dim E = 8

block Encoder(x: Tensor[B, N]) -> Tensor[B, E] {
  linear(E)
}

model Tied(a: Tensor[B, 4], b: Tensor[B, 6]) -> (Tensor[B, E], Tensor[B, E]) {
  let enc = Encoder()
  return (enc(a), enc(b))
}`,
        expectCodes: ["AXS0801"],
      },
    ],
  },
];

/** Milestone 1 (semantic core) lives in semantic.ts, Milestone 2 (tensor completeness) in tensor.ts. */
export const CHALLENGES: Challenge[] = [...BASE_CHALLENGES, ...SEMANTIC_CHALLENGES, ...TENSOR_CHALLENGES, ...LEARNING_CHALLENGES, ...RESEARCH_CHALLENGES];

// ---------------------------------------------------------------- §37 metamorphic pairs

export interface MetamorphicPair {
  id: string;
  section: string;
  a: string;
  b: string;
  run?: { dims?: Record<string, number>; tol?: number };
  ignore?: string[];
  ir?: boolean;
}

export const METAMORPHIC: MetamorphicPair[] = [
  {
    id: "sequence-vs-pipeline",
    section: "§37",
    a: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 4] {
  linear(8)
  gelu
  linear(4)
}`,
    b: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 4] {
  return x |> linear(8) |> gelu |> linear(4)
}`,
    run: { dims: { B: 2 } },
  },
  {
    id: "irrelevant-rename",
    section: "§37",
    a: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 4] {
  let h = linear(8)
  let g = h |> gelu
  return g |> linear(4)
}`,
    b: `dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 4] {
  let hidden = linear(8)
  let act = hidden |> gelu
  return act |> linear(4)
}`,
    run: { dims: { B: 2 } },
  },
  {
    id: "algebraic-dims",
    section: "§5.1 / §37",
    a: `dim B
dim D = 16
model M(x: Tensor[B, D]) -> Tensor[B, D * 4] { linear(D * 4) }`,
    b: `dim B
dim D = 16
model M(x: Tensor[B, D]) -> Tensor[B, 4 * D] { linear(4 * D) }`,
    run: { dims: { B: 2 } },
  },
  // ---------------------------------------------------------------- Milestone 1 pairs
  {
    id: "named-vs-anonymous-branches",
    section: "§7.1 / §37",
    a: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 16] {
  split merge concat(-1) { linear(8) ; branch b { linear(8) ; gelu } }
}`,
    b: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 16] {
  split merge concat(-1) { branch a { linear(8) } ; branch b { linear(8) ; gelu } }
}`,
    run: { dims: { B: 2 } },
    ignore: ["region branch0", "region a"],
  },
  {
    id: "repeat-vs-unrolled",
    section: "§9 / §37",
    a: `dim B
block L(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { for 2: L() }`,
    b: `dim B
block L(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { L() ; L() }`,
    run: { dims: { B: 2 } },
    ir: false,
  },
  {
    id: "residual-vs-explicit-add",
    section: "§7.2 / §37",
    a: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { residual { linear(8) ; gelu } }`,
    b: `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { return x + (x |> linear(8) |> gelu) }`,
    run: { dims: { B: 2 } },
    ir: false,
  },
  {
    id: "carried-dim-vs-declared-multiple",
    section: "§5.2 / §37",
    a: `dim B
dim H
model M(x: Tensor[B, 3, H, H]) -> Tensor[B, 3, 2 * (H / 2), 2 * (H / 2)] { maxpool2d(2) ; upsample(2) }`,
    b: `dim B
dim K
dim H = 2 * K
model M(x: Tensor[B, 3, H, H]) -> Tensor[B, 3, H, H] { maxpool2d(2) ; upsample(2) }`,
    run: { dims: { B: 1, H: 6, K: 3 } },
    ir: false,
  },
  // ---------------------------------------------------------------- Milestone 2 pairs (tensor.ts)
  ...TENSOR_METAMORPHIC,
  ...RESEARCH_METAMORPHIC,
];
