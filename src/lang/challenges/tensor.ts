/**
 * TENSA hardening — Milestone 2 "tensor completeness" challenges (§5.3, §11, §12, §45).
 *
 * The claim under test: every catalog operation has a tensor-level reference
 * implementation *in TENSA* (reshape / transpose / matmul / slicing / masking),
 * the checker sizes and discriminates those implementations as well as it does
 * the catalog entry, and nothing tensor-level was forced into a `custom op`.
 * The metamorphic pairs (TENSOR_METAMORPHIC) close the loop numerically:
 * manual and catalog versions produce equal outputs under controlled weights.
 */
import { CATALOG_MAP } from "../catalog";
import { IRModule } from "../ir";
import type { MetamorphicPair } from "./corpus";
import { allNodes } from "./driver";
import { Challenge, CustomCheck } from "./types";

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

/** structural IR nodes that are language, not catalog vocabulary */
const STRUCTURAL = new Set([
  "const", "const_dim", "param_read", "state_read", "state_update", "slice", "apply", "residual", "parallel",
  "static_repeat", "scan", "merge_add", "merge_mean", "add", "sub", "mul", "div",
]);

/** §11: the program used only catalog primitives and language structure — no `custom op` escape hatch. */
const noCustomOps: CustomCheck = {
  name: "tensor-level code needed no custom op (§11)",
  check: (mod: IRModule) => {
    assert(mod.customOps.length === 0, `custom ops declared: ${mod.customOps.map((c) => c.name).join(", ")}`);
    const foreign = allNodes(mod).filter((n) => !STRUCTURAL.has(n.op) && !CATALOG_MAP.has(n.op));
    assert(foreign.length === 0, `non-catalog ops: ${[...new Set(foreign.map((n) => n.op))].join(", ")}`);
    return `${allNodes(mod).length} nodes, all catalog or structural`;
  },
};

const allProved: CustomCheck = {
  name: "every shape equality is proved, none merely assumed (§5.2)",
  check: (mod: IRModule) => {
    const bad = mod.constraints.filter((c) => c.status !== "proved");
    assert(bad.length === 0, `not proved: ${bad.map((c) => `${c.origin} (${c.status})`).join("; ")}`);
    return `${mod.constraints.length} constraints, all proved`;
  },
};

function carriedExactly(origins: string[]): CustomCheck {
  return {
    name: "carried constraints are exactly the unprovable bounds",
    check: (mod: IRModule) => {
      const got = mod.constraints.filter((c) => c.status === "assumed").map((c) => `${c.origin}${c.rel ? " ≤" : ""}`);
      assert(got.length === origins.length && origins.every((o) => got.includes(o)), `carried: ${got.join("; ") || "(none)"}; expected: ${origins.join("; ")}`);
      return `${got.length} carried: ${got.join("; ")}`;
    },
  };
}

// ------------------------------------------------------------------ shared program text

/** D = 16, Heads = 2, DH = 8: the head axis and the head width differ, so a swapped split is visible */
const HEAD_FNS = `dim B
dim T
dim D = 16
dim Heads = 2
dim DH = D / Heads

fn split_heads(x: Tensor[B, T, D]) -> Tensor[B, Heads, T, DH] {
  return transpose(reshape(x, [B, T, Heads, DH]), 1, 2)
}
fn merge_heads(x: Tensor[B, Heads, T, DH]) -> Tensor[B, T, D] {
  return reshape(transpose(x, 1, 2), [B, T, D])
}`;

const MHA_MANUAL = `${HEAD_FNS}
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
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { MHA() }`;

const MHA_CATALOG = `dim B
dim T
dim D = 16
dim Heads = 2
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { attention(heads: Heads, causal: true) }`;

/** cross attention over a key sequence S with a key-padding mask Mask[B, 1, S] */
const CROSS_HEAD = `dim B
dim T
dim S
dim D = 16
dim Heads = 2
dim DH = D / Heads`;

const CROSS_MANUAL = `${CROSS_HEAD}
model X(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, T, D] {
  param wq: Tensor[D, D] init: xavier
  param wk: Tensor[D, D] init: xavier
  param wv: Tensor[D, D] init: xavier
  param wo: Tensor[D, D] init: xavier
  let qh = transpose(reshape(matmul(q, wq), [B, T, Heads, DH]), 1, 2)
  let kh = transpose(reshape(matmul(kv, wk), [B, S, Heads, DH]), 1, 2)
  let vh = transpose(reshape(matmul(kv, wv), [B, S, Heads, DH]), 1, 2)
  let scores = matmul(qh, transpose(kh, 2, 3)) / sqrt(DH)
  let masked = masked_fill(scores, reshape(pad, [B, 1, 1, S]), value: -1000000000.0)
  let w = softmax(masked, axis: -1)
  let ctx = reshape(transpose(matmul(w, vh), 1, 2), [B, T, D])
  return matmul(ctx, wo)
}`;

const CROSS_CATALOG = `${CROSS_HEAD}
model X(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, T, D] {
  return attention(query: q, key: kv, value: kv, mask: pad, heads: Heads)
}`;

const NORM_HEAD = `dim B
dim T
dim D = 16`;

const RMSNORM_MANUAL = `${NORM_HEAD}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param g: Tensor[D] init: ones
  let ms = mean(x * x, axis: -1, keep: true)
  return x * rsqrt(ms + 0.000001) * g
}`;
const RMSNORM_CATALOG = `${NORM_HEAD}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] { rmsnorm }`;

const LAYERNORM_MANUAL = `${NORM_HEAD}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param g: Tensor[D] init: ones
  param b: Tensor[D] init: zeros
  let mu = mean(x, axis: -1, keep: true)
  let xc = x - mu
  let var = mean(xc * xc, axis: -1, keep: true)
  return xc * rsqrt(var + 0.00001) * g + b
}`;
const LAYERNORM_CATALOG = `${NORM_HEAD}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] { layernorm }`;

const COSINE_MANUAL = `dim B
dim D = 16
model C(a: Tensor[B, D], b: Tensor[B, D]) -> Tensor[B] {
  let dot = sum(a * b, axis: -1)
  let na = sqrt(sum(a * a, axis: -1))
  let nb = sqrt(sum(b * b, axis: -1))
  return dot / (na * nb)
}`;
const COSINE_CATALOG = `dim B
dim D = 16
model C(a: Tensor[B, D], b: Tensor[B, D]) -> Tensor[B] { return cosine_similarity(a, b, axis: -1) }`;

const GATES = `dim B
dim T
dim D = 8
dim E = 3
block SwiGLU(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let gate = x |> linear(D * 2) |> silu
  let up = x |> linear(D * 2)
  return gate * up |> linear(D)
}
block GeGLU(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let h = x |> linear(D * 4)
  let a = h[:, :, 0:D*2]
  let g = h[:, :, D*2:D*4]
  return a * gelu(g) |> linear(D)
}
block Route(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let gates = x |> linear(E) |> softmax(axis: -1)
  let e0 = x |> linear(D) |> gelu
  let e1 = x |> linear(D) |> gelu
  let e2 = x |> linear(D) |> gelu
  let experts = stack(e0, e1, e2, axis: -1)
  return sum(experts * reshape(gates, [B, T, 1, E]), axis: -1)
}
model G(x: Tensor[B, T, D]) -> Tensor[B, T, D] { SwiGLU() ; GeGLU() ; Route() }`;

const SLICING = `dim B
dim T
dim D = 8
dim W = 2
model Shift(x: Tensor[B, T, D]) -> Tensor[B, T - 1, D] {
  let prev = x[:, 0:T-1, :]
  let next = x[:, 1:T, :]
  return next - prev
}
model Last(x: Tensor[B, T, D]) -> Tensor[B, 1, D] {
  return x[:, T-1:T, :]
}
model First(x: Tensor[B, T, D]) -> Tensor[B, D] {
  return x[:, 0, :]
}
model Flat(x: Tensor[B, T, D]) -> Tensor[B, T * D] {
  return reshape(x, [B, T * D])
}
model Fold(x: Tensor[B, T, D]) -> Tensor[B * T, D] {
  return reshape(x, [B * T, D])
}
model Tr(x: Tensor[B, T, D]) -> Tensor[B, D, T] {
  return transpose(x, -1, -2)
}
model RoundTrip(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  return reshape(transpose(reshape(x, [B, T, 2, D / 2]), 1, 2), [B, T, D])
}
model Windows(x: Tensor[B, T, D]) -> Tensor[B, 3, W, D] {
  let w0 = x[:, 0:W, :]
  let w1 = x[:, 1:W+1, :]
  let w2 = x[:, 2:W+2, :]
  return stack(w0, w1, w2, axis: 1)
}`;

// ------------------------------------------------------------------ challenges

export const TENSOR_CHALLENGES: Challenge[] = [
  // ================================================================ §12 explicit attention
  {
    id: "mha-explicit",
    title: "multi-head attention written from tensor primitives (head split / merge, causal mask)",
    section: "§12",
    tier: 2,
    record: "hardening/records/mha-explicit.md",
    code: MHA_MANUAL,
    expect: {
      ok: true,
      warnCodes: [],
      paramTables: 4,
      paramCount: 4 * 16 * 16,
      paramOwners: ["M/MHA#1"],
      irOps: ["reshape", "transpose", "matmul", "softmax", "masked_fill", "causal_mask", "const_dim"],
      irForbidOps: ["attention", "linear"],
      outputShape: { M: "Tensor[B, T, 16]" },
      emitContains: ["torch.triu(torch.ones(dims[\"T\"], dims[\"T\"]", ".reshape(dims[\"B\"], dims[\"T\"], 2, 8)"],
      emitForbids: ["reshape(-1", "dims[\"DH\"]"],
      run: { dims: { B: 2, T: 5 }, outputs: { M: "[2, 5, 16]" } },
      custom: [allProved, noCustomOps],
    },
    twins: [
      {
        id: "split-swapped",
        mutates: "reshape splits into [B, T, DH, Heads] — head axis and head width swapped",
        code: MHA_MANUAL.replace("reshape(x, [B, T, Heads, DH])", "reshape(x, [B, T, DH, Heads])"),
        // the fn's declared result is the contract: the swapped layout is refuted (2 ≠ 8) at the first application
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 24,
      },
      {
        id: "key-not-transposed",
        mutates: "scores = q @ k without transposing the key",
        code: MHA_MANUAL.replace("matmul(q, transpose(k, 2, 3))", "matmul(q, k)"),
        // contraction DH = 8 against T: unprovable for symbolic T, so carried — and refused when T ≠ 8 is bound
        expectCodes: ["AXS0403"],
        line: 14,
      },
      {
        id: "mask-wrong-length",
        mutates: "causal mask built over DH instead of T",
        code: MHA_MANUAL.replace("causal_mask(T)", "causal_mask(DH)"),
        expectCodes: ["AXS0403"],
        line: 15,
      },
      // NOT a twin: merge_heads without the transpose (reshape [B, Heads, T, DH]
      // straight to [B, T, D]) has the right element count, so no shape rule can
      // see the wrong layout.  tests.ts shows the metamorphic pair catches it
      // numerically (E-007).
      {
        id: "projection-width",
        mutates: "wo declared as Tensor[D, DH]",
        code: MHA_MANUAL.replace("param wo: Tensor[D, D]", "param wo: Tensor[D, DH]"),
        expectCodes: ["AXS0401"],
        line: 29,
      },
    ],
  },
  {
    id: "cross-attention-padding",
    title: "cross attention with a key-padding mask, manual and catalog, in one program",
    section: "§12",
    tier: 2,
    code: `${CROSS_MANUAL.replace("model X(", "model Manual(")}
model Cat(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, T, D] {
  return attention(query: q, key: kv, value: kv, mask: pad, heads: Heads)
}`,
    expect: {
      ok: true,
      warnCodes: [],
      paramTables: 8,
      outputShape: { Manual: "Tensor[B, T, 16]", Cat: "Tensor[B, T, 16]" },
      emitContains: ["attn_mask=m", "m = ~vv", "[:, None]"],
      emitForbids: ["is_causal=False)"],
      run: { dims: { B: 2, T: 5, S: 3 }, outputs: { Manual: "[2, 5, 16]", Cat: "[2, 5, 16]" } },
      custom: [allProved, noCustomOps],
    },
    twins: [
      {
        id: "catalog-mask-rank",
        mutates: "catalog attention receives Mask[B, 1, 1, S]",
        code: `${CROSS_HEAD}
model Cat(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, 1, S]) -> Tensor[B, T, D] {
  return attention(query: q, key: kv, value: kv, mask: pad, heads: Heads)
}`,
        expectCodes: ["AXS0402"],
        line: 8,
      },
      {
        id: "catalog-mask-query-length",
        mutates: "padding mask indexed by the query length (T = 5) instead of the key length (S = 3)",
        code: `dim B
dim T = 5
dim S = 3
dim D = 16
dim Heads = 2
model Cat(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, T]) -> Tensor[B, T, D] {
  return attention(query: q, key: kv, value: kv, mask: pad, heads: Heads)
}`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 7,
      },
      {
        id: "manual-mask-not-lifted",
        mutates: "masked_fill applied with the raw Mask[B, 1, S] (batch axis would broadcast onto heads)",
        code: CROSS_MANUAL.replace("reshape(pad, [B, 1, 1, S])", "pad"),
        expectCodes: ["AXS0403"],
        line: 16,
      },
      {
        id: "manual-mask-cannot-broadcast",
        mutates: "masked_fill with a mask longer than the scores on the key axis (S + 1)",
        code: CROSS_MANUAL.replace("pad: Mask[B, 1, S]", "pad: Mask[B, 1, S + 1]").replace("reshape(pad, [B, 1, 1, S])", "reshape(pad, [B, 1, 1, S + 1])"),
        expectCodes: ["AXS0401"],
        line: 16,
      },
    ],
  },
  // ================================================================ §11 norms, gates, routing
  {
    id: "norms-and-gates",
    title: "RMSNorm, LayerNorm, SwiGLU, GeGLU (by slicing) and a softmax router, all from primitives",
    section: "§11",
    code: `${GATES}
fn my_rmsnorm(x: Tensor[B, T, D], g: Tensor[D]) -> Tensor[B, T, D] {
  let ms = mean(x * x, axis: -1, keep: true)
  return x * rsqrt(ms + 0.000001) * g
}
fn my_layernorm(x: Tensor[B, T, D], g: Tensor[D], b: Tensor[D]) -> Tensor[B, T, D] {
  let mu = mean(x, axis: -1, keep: true)
  let xc = x - mu
  let var = mean(xc * xc, axis: -1, keep: true)
  return xc * rsqrt(var + 0.00001) * g + b
}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param g1: Tensor[D] init: ones
  param g2: Tensor[D] init: ones
  param b2: Tensor[D] init: zeros
  return my_layernorm(my_rmsnorm(x, g1), g2, b2)
}`,
    expect: {
      ok: true,
      warnCodes: [],
      irOps: ["slice", "stack", "silu", "gelu", "softmax", "rsqrt", "mean"],
      irForbidOps: ["rmsnorm", "layernorm", "attention"],
      outputShape: { G: "Tensor[B, T, 8]", N: "Tensor[B, T, 8]" },
      run: { dims: { B: 2, T: 5 }, outputs: { G: "[2, 5, 8]", N: "[2, 5, 8]" } },
      custom: [allProved, noCustomOps],
    },
    twins: [
      {
        id: "geglu-slice-overshoots",
        mutates: "GeGLU gate slice ends at D*4 + 1",
        code: GATES.replace("h[:, :, D*2:D*4]", "h[:, :, D*2:D*4+1]"),
        expectCodes: ["AXS0410"],
        line: 13,
      },
      {
        id: "geglu-halves-unequal",
        mutates: "GeGLU value slice takes D*3 columns, gate takes D",
        code: GATES.replace("h[:, :, 0:D*2]", "h[:, :, 0:D*3]"),
        expectCodes: ["AXS0401"],
        line: 14,
      },
      {
        id: "router-gates-not-lifted",
        mutates: "router multiplies experts [B, T, D, E] by gates [B, T, E] without the reshape",
        code: GATES.replace("reshape(gates, [B, T, 1, E])", "gates"),
        // [B, T, D, E] * [B, T, E] trailing-aligns T↔D and B↔T: unprovable for symbolic B, T, so carried
        expectCodes: ["AXS0403"],
        line: 22,
      },
      {
        id: "gain-width",
        mutates: "rmsnorm gain declared Tensor[D * 2]",
        code: `${NORM_HEAD}
model N(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param g: Tensor[D * 2] init: ones
  let ms = mean(x * x, axis: -1, keep: true)
  return x * rsqrt(ms + 0.000001) * g
}`,
        expectCodes: ["AXS0401"],
        line: 7,
      },
    ],
  },
  // ================================================================ §5.3 reshape / transpose / slicing torture
  {
    id: "slicing-torture",
    title: "shift, last, first, flatten, fold, transpose, split/merge round trip and symbolic windows",
    section: "§5.3",
    code: SLICING,
    expect: {
      ok: true,
      warnCodes: ["AXS0403"],
      constraints: ["assumed"],
      outputShape: {
        Shift: "Tensor[B, T - 1, 8]",
        Last: "Tensor[B, 1, 8]",
        First: "Tensor[B, 8]",
        Flat: "Tensor[B, 8*T]",
        Fold: "Tensor[B*T, 8]",
        Tr: "Tensor[B, 8, T]",
        RoundTrip: "Tensor[B, T, 8]",
        Windows: "Tensor[B, 3, 2, 8]",
      },
      inspectContains: ["2 ≤ T", "3 ≤ T", "4 ≤ T"],
      emitContains: ["[0:dims[\"B\"], 0:(dims[\"T\"] - 1), 0:8]", "[0:dims[\"B\"], 1:dims[\"T\"], 0:8]", ".reshape(dims[\"B\"], 8 * dims[\"T\"])", ".reshape(dims[\"B\"] * dims[\"T\"], 8)"],
      emitForbids: ["-1 +", "reshape(-1"],
      run: {
        dims: { B: 2, T: 5 },
        outputs: { Shift: "[2, 4, 8]", Last: "[2, 1, 8]", First: "[2, 8]", Flat: "[2, 40]", Fold: "[10, 8]", Tr: "[2, 8, 5]", RoundTrip: "[2, 5, 8]", Windows: "[2, 3, 2, 8]" },
      },
      custom: [
        // only the window ends are unprovable: 0:T-1, 1:T, T-1:T, and the index 0 are all decided statically
        carriedExactly(["slice end on axis 1 (must be ≤ extent) ≤", "slice end on axis 1 (must be ≤ extent) ≤", "slice end on axis 1 (must be ≤ extent) ≤"]),
        noCustomOps,
      ],
    },
    twins: [
      {
        id: "slice-past-end",
        mutates: "x[:, 0:T+1, :] — end one past the extent",
        code: `dim B
dim T
dim D = 8
model Over(x: Tensor[B, T, D]) -> Tensor[B, T + 1, D] {
  return x[:, 0:T+1, :]
}`,
        expectCodes: ["AXS0410"],
        forbidCodes: ["AXS0403", "AXS0406"],
        line: 5,
      },
      {
        id: "index-at-extent",
        mutates: "x[:, T, :] — index equal to the extent",
        code: `dim B
dim T
dim D = 8
model Idx(x: Tensor[B, T, D]) -> Tensor[B, D] {
  return x[:, T, :]
}`,
        expectCodes: ["AXS0410"],
        line: 5,
      },
      {
        id: "start-after-end",
        mutates: "x[:, 3:2, :]",
        code: `dim B
dim T
dim D = 8
model Bad(x: Tensor[B, T, D]) -> Tensor[B, 1, D] {
  return x[:, 3:2, :]
}`,
        expectCodes: ["AXS0410"],
        line: 5,
      },
      {
        id: "constant-axis-overrun",
        mutates: "x[:, :, 4:9] on a constant axis of extent 8",
        code: `dim B
dim T
dim D = 8
model Bad(x: Tensor[B, T, D]) -> Tensor[B, T, 5] {
  return x[:, :, 4:9]
}`,
        expectCodes: ["AXS0410"],
        line: 5,
      },
      {
        id: "reshape-count",
        mutates: "flatten to [B, T * D + 1]",
        code: `dim B
dim T
dim D = 8
model Flat(x: Tensor[B, T, D]) -> Tensor[B, T * D + 1] {
  return reshape(x, [B, T * D + 1])
}`,
        expectCodes: ["AXS0401"],
        forbidCodes: ["AXS0403"],
        line: 5,
      },
      {
        id: "roundtrip-wrong-group",
        mutates: "split into [B, T, 3, D / 2] — 3 groups of D/2 do not tile D",
        code: `dim B
dim T
dim D = 8
model RoundTrip(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  return reshape(transpose(reshape(x, [B, T, 3, D / 2]), 1, 2), [B, T, D])
}`,
        expectCodes: ["AXS0401"],
        line: 5,
      },
    ],
  },
  {
    id: "window-bound-runtime",
    title: "a carried slice bound is verified when T is bound too small",
    section: "§5.3",
    code: `dim B
dim T
dim D = 4
dim W = 2
model Win(x: Tensor[B, T, D]) -> Tensor[B, W, D] {
  return x[:, 2:W+2, :]
}`,
    expect: {
      ok: true,
      warnCodes: ["AXS0403"],
      constraints: ["assumed"],
      inspectContains: ["4 ≤ T [assumed]"],
      run: { dims: { B: 1, T: 3 }, refuses: true, errorContains: "4 ≤ T but 4 > 3" },
    },
    twins: [],
  },
  {
    id: "head-split-free-width",
    title: "head split with a runtime model width: divisibility is carried, not assumed",
    section: "§5.3 / §12",
    code: `dim B
dim T
dim D
dim Heads = 4
model H(x: Tensor[B, T, D]) -> Tensor[B, Heads, T, D / Heads] {
  return transpose(reshape(x, [B, T, Heads, D / Heads]), 1, 2)
}`,
    expect: {
      ok: true,
      warnCodes: ["AXS0403"],
      constraints: ["assumed"],
      outputShape: { H: "Tensor[B, 4, T, ⌊D/4⌋]" },
      run: { dims: { B: 1, T: 2, D: 10 }, refuses: true, errorContains: "20 ≠ 16" },
    },
    twins: [],
  },
];

// ------------------------------------------------------------------ §37 / §45 metamorphic pairs

export const TENSOR_METAMORPHIC: MetamorphicPair[] = [
  { id: "mha-manual-vs-catalog", section: "§12 / §45", a: MHA_MANUAL, b: MHA_CATALOG, run: { dims: { B: 2, T: 5 }, tol: 1e-4 }, ir: false },
  { id: "cross-attention-manual-vs-catalog", section: "§12 / §45", a: CROSS_MANUAL, b: CROSS_CATALOG, run: { dims: { B: 2, T: 5, S: 3 }, tol: 1e-4 }, ir: false },
  { id: "rmsnorm-manual-vs-catalog", section: "§11 / §45", a: RMSNORM_MANUAL, b: RMSNORM_CATALOG, run: { dims: { B: 2, T: 5 }, tol: 1e-4 }, ir: false },
  { id: "layernorm-manual-vs-catalog", section: "§11 / §45", a: LAYERNORM_MANUAL, b: LAYERNORM_CATALOG, run: { dims: { B: 2, T: 5 }, tol: 1e-4 }, ir: false },
  { id: "cosine-manual-vs-catalog", section: "§11 / §45", a: COSINE_MANUAL, b: COSINE_CATALOG, run: { dims: { B: 3 }, tol: 1e-4 }, ir: false },
  {
    // the mask port on its own (no key:/value:) must reach the mask slot, not the key slot (F-013)
    id: "causal-flag-vs-explicit-mask",
    section: "§12 / §37",
    a: MHA_CATALOG,
    b: `dim B
dim T
dim D = 16
dim Heads = 2
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return attention(query: x, mask: causal_mask(T), heads: Heads) }`,
    run: { dims: { B: 2, T: 5 }, tol: 1e-5 },
    ir: false,
  },
  {
    id: "open-vs-explicit-slice",
    section: "§5.3 / §37",
    a: `dim B
dim T
model S(x: Tensor[B, T, 8]) -> Tensor[B, T - 1, 8] { return x[:, 1:T, :] }`,
    b: `dim B
dim T
model S(x: Tensor[B, T, 8]) -> Tensor[B, T - 1, 8] { return x[:, 1:, :] }`,
    run: { dims: { B: 2, T: 5 } },
  },
  {
    id: "head-split-via-fn-vs-inline",
    section: "§12 / §37",
    a: `${HEAD_FNS}
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return merge_heads(split_heads(x)) }`,
    b: `dim B
dim T
dim D = 16
dim Heads = 2
dim DH = D / Heads
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return reshape(transpose(transpose(reshape(x, [B, T, Heads, DH]), 1, 2), 1, 2), [B, T, D]) }`,
    run: { dims: { B: 2, T: 5 } },
    ir: false,
  },
];
