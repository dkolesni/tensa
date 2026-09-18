/**
 * TENSA — standard catalog.
 *
 * The catalog is *library vocabulary*, not language semantics.  Every entry is
 * a shape rule + parameter/state ownership + effect signature.  Adding a layer
 * means adding a table entry (or writing a `block`, `fn` or `custom op` in
 * source) — never changing the compiler.
 */
import { DimExpr, asConst, dAdd, dConst, dDiv, dMul, dSub, show } from "./dims";
import { IRAttr } from "./ir";
import { Effect, Loc, TensorKind, TensorType, ValueType, scalar, tensor } from "./types";

export interface ParamRequest {
  role: string;
  shape: DimExpr[];
  init: string;
  kind: "weight" | "bias" | "norm" | "embedding" | "explicit";
  trainable?: boolean;
}
export interface StateRequest {
  role: string;
  shape: DimExpr[];
  init: string;
  update: string;
  category: "running-stat" | "ema" | "counter" | "algorithmic";
}

export interface InferCtx {
  ins: TensorType[];
  raw: ValueType[];
  num(name: string, def?: number): number;
  dim(name: string, def?: DimExpr): DimExpr;
  bool(name: string, def: boolean): boolean;
  str(name: string, def: string): string;
  dims(name: string): DimExpr[] | null;
  has(name: string): boolean;
  err(code: string, msg: string, notes?: string[]): void;
  warn(code: string, msg: string, notes?: string[]): void;
  /** record/prove a symbolic equality; returns true when provable */
  eq(a: DimExpr, b: DimExpr, origin: string): boolean;
  loc: Loc;
}

export interface OpResult {
  out: ValueType;
  params?: ParamRequest[];
  states?: StateRequest[];
  effects?: Effect[];
  attrs?: Record<string, IRAttr>;
}

export interface ConfigSpec {
  name: string;
  pos?: number;
  kind: "dim" | "num" | "bool" | "axis" | "str" | "tensor";
  required?: boolean;
  doc?: string;
}

export interface OpSpec {
  name: string;
  style: "layer" | "primitive";
  category: string;
  /** tensor ports; the first is filled by the implicit cursor / pipeline value */
  ports: string[];
  optionalPorts?: string[];
  config: ConfigSpec[];
  effects: Effect[];
  doc: string;
  infer(c: InferCtx): OpResult;
}

const last = (t: TensorType) => t.shape[t.shape.length - 1];

function checkRank(c: InferCtx, t: TensorType, r: number, op: string): boolean {
  if (t.shape.length !== r) {
    c.err("AXS0402", `${op} expects a rank-${r} tensor, received rank ${t.shape.length} ${`[${t.shape.map(show).join(", ")}]`}`);
    return false;
  }
  return true;
}

function convOut(inp: DimExpr, k: number, s: number, p: number, d: number): DimExpr {
  const eff = dSub(dAdd(inp, dConst(2 * p)), dConst(d * (k - 1) + 1));
  return dAdd(dDiv(eff, dConst(s)), dConst(1));
}

function ew(name: string, effects: Effect[] = [], kindOut?: TensorKind): OpSpec {
  return {
    name,
    style: "primitive",
    category: "elementwise",
    ports: ["x"],
    config: [],
    effects,
    doc: `elementwise ${name}`,
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      return { out: tensor(t.shape, kindOut ?? (t.kind === "Probs" || t.kind === "Logits" ? "Tensor" : t.kind), t.dtype) };
    },
  };
}

function reduceSpec(name: string): OpSpec {
  return {
    name,
    style: "primitive",
    category: "reduction",
    ports: ["x"],
    config: [
      { name: "axis", kind: "axis", pos: 1, doc: "axis to reduce (omit to reduce everything)" },
      { name: "keep", kind: "bool", doc: "keep the reduced axis with extent 1" },
    ],
    effects: name === "argmax" ? ["nondiff"] : ["pure"],
    doc: `${name} reduction`,
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      if (!c.has("axis")) return { out: scalar, attrs: { axis: null } };
      let ax = c.num("axis", -1);
      if (ax < 0) ax += t.shape.length;
      if (ax < 0 || ax >= t.shape.length) {
        c.err("AXS0401", `axis ${c.num("axis", -1)} out of range for rank ${t.shape.length}`);
        return { out: t };
      }
      const keep = c.bool("keep", false);
      const shape = t.shape.slice();
      if (keep) shape[ax] = dConst(1);
      else shape.splice(ax, 1);
      const kind: TensorKind = name === "argmax" ? "Class" : "Tensor";
      return { out: tensor(shape, kind, name === "argmax" ? "i32" : "f32"), attrs: { axis: ax, keep } };
    },
  };
}

export const CATALOG: OpSpec[] = [
  // ------------------------------------------------------------ architecture
  {
    name: "linear",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [
      { name: "out", pos: 0, kind: "dim", required: true, doc: "output width — an architectural choice" },
      { name: "bias", kind: "bool" },
    ],
    effects: ["parameterized"],
    doc: "Affine map on the last axis. The input width is a consequence of the incoming tensor and is never written.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || t.shape.length < 1) {
        c.err("AXS0402", "linear expects a tensor with at least one axis");
        return { out: scalar };
      }
      const inW = last(t);
      const out = c.dim("out");
      const bias = c.bool("bias", true);
      const shape = [...t.shape.slice(0, -1), out];
      return {
        out: tensor(shape, t.kind === "Image" ? "Tensor" : t.kind === "Tokens" ? "Tensor" : t.kind, "f32"),
        params: [
          { role: "w", shape: [inW, out], init: "xavier", kind: "weight" },
          ...(bias ? [{ role: "b", shape: [out], init: "zeros", kind: "bias" as const }] : []),
        ],
        attrs: { out, bias },
      };
    },
  },
  {
    name: "conv2d",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [
      { name: "out", pos: 0, kind: "dim", required: true, doc: "output channels" },
      { name: "kernel", kind: "num", required: true },
      { name: "stride", kind: "num" },
      { name: "pad", kind: "num" },
      { name: "dilation", kind: "num" },
      { name: "bias", kind: "bool" },
    ],
    effects: ["parameterized"],
    doc: "2-D convolution over [B, C, H, W]. Input channels are inferred.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 4, "conv2d")) return { out: c.ins[0] ?? scalar };
      const [B, C, H, W] = t.shape;
      const k = c.num("kernel", 3);
      const s = c.num("stride", 1);
      const p = c.num("pad", 0);
      const d = c.num("dilation", 1);
      const out = c.dim("out");
      const bias = c.bool("bias", true);
      return {
        out: tensor([B, out, convOut(H, k, s, p, d), convOut(W, k, s, p, d)], "Tensor"),
        params: [
          { role: "w", shape: [out, C, dConst(k), dConst(k)], init: "kaiming", kind: "weight" },
          ...(bias ? [{ role: "b", shape: [out], init: "zeros", kind: "bias" as const }] : []),
        ],
        attrs: { out, kernel: k, stride: s, pad: p, dilation: d, bias },
      };
    },
  },
  {
    name: "embedding",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [
      { name: "vocab", pos: 0, kind: "dim", required: true },
      { name: "width", pos: 1, kind: "dim", required: true },
    ],
    effects: ["parameterized"],
    doc: "Token table lookup: Tokens[...] -> Tensor[..., width].",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      if (t.kind !== "Tokens" && t.kind !== "Class")
        c.warn("AXS0409", `embedding expects Tokens[...] but received ${t.kind}[${t.shape.map(show).join(", ")}]`, [
          "declare the input as Tokens[...] so the compiler can prove index semantics",
        ]);
      const vocab = c.dim("vocab");
      const width = c.dim("width");
      return {
        out: tensor([...t.shape, width], "Tensor"),
        params: [{ role: "table", shape: [vocab, width], init: "normal", kind: "embedding" }],
        attrs: { vocab, width },
      };
    },
  },
  {
    name: "positional",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [{ name: "max", kind: "dim", required: true, doc: "maximum sequence length" }],
    effects: ["parameterized"],
    doc: "Learned absolute positional embedding added to [B, T, D].",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 3, "positional")) return { out: c.ins[0] ?? scalar };
      const maxLen = c.dim("max");
      return {
        out: t,
        params: [{ role: "pos", shape: [maxLen, t.shape[2]], init: "normal", kind: "embedding" }],
        attrs: { max: maxLen },
      };
    },
  },
  {
    name: "attention",
    style: "layer",
    category: "architecture",
    ports: ["query"],
    optionalPorts: ["key", "value", "mask"],
    config: [
      { name: "heads", kind: "dim", required: true },
      { name: "causal", kind: "bool" },
    ],
    effects: ["parameterized"],
    doc: "Multi-head scaled dot-product attention. Self-attention by default; supply key:/value: for cross attention.",
    infer: (c) => {
      const q = c.ins[0];
      if (!q || !checkRank(c, q, 3, "attention")) return { out: c.ins[0] ?? scalar };
      const k = c.ins[1] ?? q;
      const v = c.ins[2] ?? k;
      const D = q.shape[2];
      c.eq(k.shape[2], D, "attention key width");
      c.eq(v.shape[1], k.shape[1], "attention key/value length");
      c.eq(k.shape[0], q.shape[0], "attention batch");
      const heads = c.dim("heads");
      const dh = dDiv(D, heads);
      const hc = asConst(heads);
      const dc = asConst(D);
      if (hc !== null && dc !== null && dc % hc !== 0)
        c.err("AXS0407", `attention head count ${hc} does not divide model width ${dc}`);
      // symbolic divisibility: `heads * ⌊D/heads⌋ = D` is proved when the division
      // is exact and otherwise carried to runtime instead of silently truncating (F-004)
      else if (hc === null || dc === null) c.eq(dMul(dh, heads), D, "attention head split (heads must divide D)");
      // the mask port has a shape contract too (F-012): Mask[Tq, Tk] or
      // Mask[B|1, Tq|1, Tk], broadcast over heads.  Anything else silently
      // misaligned at runtime before this check existed.
      const m = c.ins[3];
      if (m) {
        if (m.kind !== "Mask") c.warn("AXS0409", `attention mask expects Mask[...], received ${m.kind}`);
        const Tq = q.shape[1];
        const Tk = k.shape[1];
        if (m.shape.length === 2) {
          c.eq(m.shape[0], Tq, "attention mask query length");
          c.eq(m.shape[1], Tk, "attention mask key length");
        } else if (m.shape.length === 3) {
          if (asConst(m.shape[0]) !== 1) c.eq(m.shape[0], q.shape[0], "attention mask batch");
          if (asConst(m.shape[1]) !== 1) c.eq(m.shape[1], Tq, "attention mask query length");
          c.eq(m.shape[2], Tk, "attention mask key length");
        } else
          c.err("AXS0402", `attention mask must be Mask[Tq, Tk] or Mask[B, Tq, Tk] (1 broadcasts), received rank ${m.shape.length} [${m.shape.map(show).join(", ")}]`);
      }
      return {
        out: tensor([q.shape[0], q.shape[1], D], "Tensor"),
        params: [
          { role: "wq", shape: [D, D], init: "xavier", kind: "weight" },
          { role: "wk", shape: [k.shape[2], D], init: "xavier", kind: "weight" },
          { role: "wv", shape: [v.shape[2], D], init: "xavier", kind: "weight" },
          { role: "wo", shape: [D, D], init: "xavier", kind: "weight" },
        ],
        attrs: { heads, head_dim: dh, causal: c.bool("causal", false) },
      };
    },
  },
  {
    name: "layernorm",
    style: "layer",
    category: "normalization",
    ports: ["x"],
    config: [{ name: "eps", kind: "num" }],
    effects: ["parameterized"],
    doc: "Normalise over the last axis with learned scale and shift.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      const D = last(t);
      return {
        out: tensor(t.shape, "Tensor"),
        params: [
          { role: "gamma", shape: [D], init: "ones", kind: "norm" },
          { role: "beta", shape: [D], init: "zeros", kind: "norm" },
        ],
        attrs: { eps: c.num("eps", 1e-5) },
      };
    },
  },
  {
    name: "rmsnorm",
    style: "layer",
    category: "normalization",
    ports: ["x"],
    config: [{ name: "eps", kind: "num" }],
    effects: ["parameterized"],
    doc: "Root-mean-square normalisation with a learned scale.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      return {
        out: tensor(t.shape, "Tensor"),
        params: [{ role: "gamma", shape: [last(t)], init: "ones", kind: "norm" }],
        attrs: { eps: c.num("eps", 1e-6) },
      };
    },
  },
  {
    name: "batchnorm",
    style: "layer",
    category: "normalization",
    ports: ["x"],
    config: [{ name: "momentum", kind: "num" }],
    effects: ["parameterized", "reads-state", "writes-state", "training-sensitive"],
    doc: "Batch normalisation. Owns persistent running statistics and behaves differently in train and eval context.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      if (t.shape.length !== 4 && t.shape.length !== 2)
        c.err("AXS0402", `batchnorm expects [B, C, H, W] or [B, C], received rank ${t.shape.length}`);
      const C = t.shape.length === 4 ? t.shape[1] : last(t);
      return {
        out: tensor(t.shape, "Tensor"),
        params: [
          { role: "gamma", shape: [C], init: "ones", kind: "norm" },
          { role: "beta", shape: [C], init: "zeros", kind: "norm" },
        ],
        states: [
          { role: "running_mean", shape: [C], init: "zeros", update: "ema(momentum)", category: "running-stat" },
          { role: "running_var", shape: [C], init: "ones", update: "ema(momentum)", category: "running-stat" },
        ],
        attrs: { momentum: c.num("momentum", 0.1) },
      };
    },
  },
  {
    name: "randn_like",
    style: "primitive",
    category: "tensor",
    ports: ["x"],
    config: [],
    effects: ["stochastic"],
    doc: "Independent standard normal noise with the input shape, in train AND eval. No gradient to the shape template (H-010).",
    infer: (c) => ({ out: tensor(c.ins[0]?.shape ?? []) }),
  },
  {
    name: "dropout",
    style: "layer",
    category: "regularisation",
    ports: ["x"],
    config: [{ name: "p", pos: 0, kind: "num", required: true }],
    effects: ["stochastic", "training-sensitive"],
    doc: "Stochastic zeroing during training; identity during evaluation.",
    infer: (c) => ({ out: c.ins[0] ?? scalar, attrs: { p: c.num("p", 0.5) } }),
  },
  {
    name: "maxpool2d",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [
      { name: "kernel", pos: 0, kind: "num", required: true },
      { name: "stride", kind: "num" },
      { name: "pad", kind: "num" },
    ],
    effects: ["pure"],
    doc: "Max pooling over spatial axes.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 4, "maxpool2d")) return { out: c.ins[0] ?? scalar };
      const k = c.num("kernel", 2);
      const s = c.num("stride", k);
      const p = c.num("pad", 0);
      return {
        out: tensor([t.shape[0], t.shape[1], convOut(t.shape[2], k, s, p, 1), convOut(t.shape[3], k, s, p, 1)]),
        attrs: { kernel: k, stride: s, pad: p },
      };
    },
  },
  {
    name: "avgpool2d",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [
      { name: "kernel", pos: 0, kind: "num", required: true },
      { name: "stride", kind: "num" },
    ],
    effects: ["pure"],
    doc: "Average pooling over spatial axes.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 4, "avgpool2d")) return { out: c.ins[0] ?? scalar };
      const k = c.num("kernel", 2);
      const s = c.num("stride", k);
      return {
        out: tensor([t.shape[0], t.shape[1], convOut(t.shape[2], k, s, 0, 1), convOut(t.shape[3], k, s, 0, 1)]),
        attrs: { kernel: k, stride: s },
      };
    },
  },
  {
    name: "upsample",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [{ name: "scale", pos: 0, kind: "num", required: true }],
    effects: ["pure"],
    doc: "Nearest-neighbour spatial upsampling: [B, C, H, W] -> [B, C, H*s, W*s].",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 4, "upsample")) return { out: c.ins[0] ?? scalar };
      const s = c.num("scale", 2);
      if (!Number.isInteger(s) || s < 1) c.err("AXS0407", `upsample scale must be a positive integer, got ${s}`);
      return { out: tensor([t.shape[0], t.shape[1], dMul(t.shape[2], dConst(s)), dMul(t.shape[3], dConst(s))]), attrs: { scale: s } };
    },
  },
  {
    name: "global_avgpool",
    style: "layer",
    category: "architecture",
    ports: ["x"],
    config: [],
    effects: ["pure"],
    doc: "Average over spatial axes: [B, C, H, W] -> [B, C].",
    infer: (c) => {
      const t = c.ins[0];
      if (!t || !checkRank(c, t, 4, "global_avgpool")) return { out: c.ins[0] ?? scalar };
      return { out: tensor([t.shape[0], t.shape[1]]) };
    },
  },
  {
    name: "flatten",
    style: "layer",
    category: "shape",
    ports: ["x"],
    config: [],
    effects: ["pure"],
    doc: "Flatten every axis after the batch axis.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      let prod = dConst(1);
      for (const d of t.shape.slice(1)) prod = dMul(prod, d);
      return { out: tensor([t.shape[0], prod]) };
    },
  },
  // ------------------------------------------------------------ activations
  ew("relu"),
  ew("gelu"),
  ew("silu"),
  ew("sigmoid"),
  ew("tanh"),
  ew("exp"),
  ew("log"),
  ew("sqrt"),
  ew("rsqrt"),
  ew("abs"),
  ew("neg"),
  {
    name: "softmax",
    style: "primitive",
    category: "activation",
    ports: ["x"],
    config: [{ name: "axis", kind: "axis" }],
    effects: ["pure"],
    doc: "Softmax along an axis. Maps Logits to Probs.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      if (t.kind === "Probs")
        c.warn("AXS0409", "softmax applied to a tensor already carrying semantic kind Probs");
      let ax = c.num("axis", -1);
      if (ax < 0) ax += t.shape.length;
      return { out: tensor(t.shape, "Probs"), attrs: { axis: ax } };
    },
  },
  {
    name: "log_softmax",
    style: "primitive",
    category: "activation",
    ports: ["x"],
    config: [{ name: "axis", kind: "axis" }],
    effects: ["pure"],
    doc: "Numerically stable log-softmax.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      let ax = c.num("axis", -1);
      if (ax < 0) ax += t.shape.length;
      return { out: tensor(t.shape, "Tensor"), attrs: { axis: ax } };
    },
  },
  // ------------------------------------------------------------ tensor math
  {
    name: "matmul",
    style: "primitive",
    category: "tensor",
    ports: ["a", "b"],
    config: [],
    effects: ["pure"],
    doc: "Batched matrix multiplication over the last two axes.",
    infer: (c) => {
      const [a, b] = c.ins;
      if (!a || !b) return { out: scalar };
      if (a.shape.length < 2 || b.shape.length < 2) {
        c.err("AXS0402", "matmul requires rank >= 2 on both operands");
        return { out: a };
      }
      const K1 = a.shape[a.shape.length - 1];
      const K2 = b.shape[b.shape.length - 2];
      c.eq(K1, K2, "matmul contraction axis");
      const batch = a.shape.slice(0, -2);
      const bBatch = b.shape.slice(0, -2);
      for (let i = 0; i < Math.min(batch.length, bBatch.length); i++)
        c.eq(batch[batch.length - 1 - i], bBatch[bBatch.length - 1 - i], "matmul batch axis");
      const lead = batch.length >= bBatch.length ? batch : bBatch;
      return { out: tensor([...lead, a.shape[a.shape.length - 2], b.shape[b.shape.length - 1]]) };
    },
  },
  {
    name: "transpose",
    style: "primitive",
    category: "tensor",
    ports: ["x"],
    config: [
      { name: "i", pos: 1, kind: "axis", required: true },
      { name: "j", pos: 2, kind: "axis", required: true },
    ],
    effects: ["pure"],
    doc: "Swap two axes.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      let i = c.num("i", 0);
      let j = c.num("j", 1);
      if (i < 0) i += t.shape.length;
      if (j < 0) j += t.shape.length;
      if (i >= t.shape.length || j >= t.shape.length) {
        c.err("AXS0401", `transpose axes (${i}, ${j}) out of range for rank ${t.shape.length}`);
        return { out: t };
      }
      const shape = t.shape.slice();
      [shape[i], shape[j]] = [shape[j], shape[i]];
      return { out: tensor(shape, t.kind), attrs: { i, j } };
    },
  },
  {
    name: "reshape",
    style: "primitive",
    category: "tensor",
    ports: ["x"],
    config: [{ name: "shape", pos: 1, kind: "dim", required: true }],
    effects: ["pure"],
    doc: "Reshape to an explicit symbolic shape. The element count must be provably preserved.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      const target = c.dims("shape");
      if (!target) {
        c.err("AXS0501", "reshape requires a shape list, e.g. reshape(x, [B, T, Heads, DH])");
        return { out: t };
      }
      let src = dConst(1);
      for (const d of t.shape) src = dMul(src, d);
      let dst = dConst(1);
      for (const d of target) dst = dMul(dst, d);
      c.eq(src, dst, "reshape element count");
      return { out: tensor(target, t.kind), attrs: { shape: target } };
    },
  },
  {
    name: "concat",
    style: "primitive",
    category: "tensor",
    ports: ["a", "b"],
    optionalPorts: ["c", "d", "e", "f"],
    config: [{ name: "axis", kind: "axis", required: true }],
    effects: ["pure"],
    doc: "Concatenate tensors along an axis. Every other axis must agree.",
    infer: (c) => {
      const ts = c.ins;
      if (ts.length === 0) return { out: scalar };
      let ax = c.num("axis", -1);
      if (ax < 0) ax += ts[0].shape.length;
      const shape = ts[0].shape.slice();
      let sum = ts[0].shape[ax];
      for (let n = 1; n < ts.length; n++) {
        const t = ts[n];
        if (t.shape.length !== shape.length) {
          c.err("AXS0405", `concat operands differ in rank (${shape.length} vs ${t.shape.length})`);
          continue;
        }
        for (let d = 0; d < shape.length; d++)
          if (d !== ax) c.eq(shape[d], t.shape[d], `concat axis ${d}`);
        sum = dAdd(sum, t.shape[ax]);
      }
      shape[ax] = sum;
      return { out: tensor(shape, ts[0].kind), attrs: { axis: ax } };
    },
  },
  {
    name: "stack",
    style: "primitive",
    category: "tensor",
    ports: ["a", "b"],
    optionalPorts: ["c", "d"],
    config: [{ name: "axis", kind: "axis", required: true }],
    effects: ["pure"],
    doc: "Stack tensors along a new axis.",
    infer: (c) => {
      const ts = c.ins;
      if (!ts.length) return { out: scalar };
      let ax = c.num("axis", 0);
      if (ax < 0) ax += ts[0].shape.length + 1;
      for (let n = 1; n < ts.length; n++)
        for (let d = 0; d < ts[0].shape.length; d++) c.eq(ts[0].shape[d], ts[n].shape[d], "stack axis");
      const shape = ts[0].shape.slice();
      shape.splice(ax, 0, dConst(ts.length));
      return { out: tensor(shape, ts[0].kind), attrs: { axis: ax } };
    },
  },
  reduceSpec("sum"),
  reduceSpec("mean"),
  reduceSpec("max"),
  reduceSpec("min"),
  reduceSpec("argmax"),
  {
    name: "masked_fill",
    style: "primitive",
    category: "tensor",
    ports: ["x", "mask"],
    config: [{ name: "value", kind: "num", required: true }],
    effects: ["pure"],
    doc: "Fill positions where the mask is true with a constant.",
    infer: (c) => {
      const [x, m] = c.ins;
      if (!x) return { out: scalar };
      if (m && m.kind !== "Mask")
        c.warn("AXS0409", `masked_fill expects Mask[...] as the second operand, received ${m.kind}`);
      // the mask broadcasts against x from the trailing axis (F-011): equal
      // extents or a constant 1 on the mask side; a longer mask never fits
      if (m) {
        if (m.shape.length > x.shape.length)
          c.err("AXS0402", `masked_fill mask rank ${m.shape.length} exceeds operand rank ${x.shape.length}`);
        else
          m.shape.forEach((d, i) => {
            if (asConst(d) === 1) return;
            c.eq(d, x.shape[x.shape.length - m.shape.length + i], `masked_fill mask axis ${i} (broadcast against operand)`);
          });
      }
      return { out: tensor(x.shape, x.kind), attrs: { value: c.num("value", 0) } };
    },
  },
  {
    name: "causal_mask",
    style: "primitive",
    category: "tensor",
    ports: [],
    config: [{ name: "n", pos: 0, kind: "dim", required: true }],
    effects: ["pure"],
    doc: "Upper-triangular boolean mask of shape Mask[n, n] (true above the diagonal).",
    infer: (c) => {
      const n = c.dim("n");
      return { out: tensor([n, n], "Mask", "bool"), attrs: { n } };
    },
  },
  {
    name: "one_hot",
    style: "primitive",
    category: "tensor",
    ports: ["x"],
    config: [{ name: "classes", kind: "dim", required: true }],
    effects: ["pure"],
    doc: "Expand class indices into a one-hot axis.",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      const k = c.dim("classes");
      return { out: tensor([...t.shape, k]), attrs: { classes: k } };
    },
  },
  {
    name: "stop_grad",
    style: "primitive",
    category: "differentiation",
    ports: ["x"],
    config: [],
    effects: ["grad-stopped"],
    doc: "Value passes through unchanged; gradient is intentionally cut here.",
    infer: (c) => ({ out: c.ins[0] ?? scalar }),
  },
  {
    name: "detach_kind",
    style: "primitive",
    category: "types",
    ports: ["x"],
    config: [{ name: "as", kind: "str", required: true }],
    effects: ["pure"],
    doc: "Reinterpret the semantic kind of a tensor (storage unchanged).",
    infer: (c) => {
      const t = c.ins[0];
      if (!t) return { out: scalar };
      return { out: tensor(t.shape, c.str("as", "Tensor") as TensorKind, t.dtype) };
    },
  },
  // ------------------------------------------------------------ losses
  {
    name: "cross_entropy",
    style: "primitive",
    category: "objective",
    ports: ["logits", "labels"],
    config: [{ name: "ignore", kind: "num" }],
    effects: ["pure"],
    doc: "Mean categorical cross entropy of logits against integer class targets.",
    infer: (c) => {
      const [lg, lb] = c.ins;
      if (!lg || !lb) return { out: scalar };
      if (lg.kind === "Probs")
        c.warn("AXS0409", "cross_entropy expects Logits, received Probs — applying softmax twice is a common bug");
      if (lb.kind !== "Class" && lb.kind !== "Tokens")
        c.warn("AXS0409", `cross_entropy targets should be Class[...] or Tokens[...], received ${lb.kind}[...]`);
      if (lb.shape.length !== lg.shape.length - 1)
        c.err(
          "AXS0402",
          `cross_entropy expects targets of rank ${lg.shape.length - 1} for logits of rank ${lg.shape.length}`
        );
      else for (let i = 0; i < lb.shape.length; i++) c.eq(lg.shape[i], lb.shape[i], `cross_entropy axis ${i}`);
      return { out: scalar };
    },
  },
  {
    name: "mse",
    style: "primitive",
    category: "objective",
    ports: ["pred", "target"],
    config: [],
    effects: ["pure"],
    doc: "Mean squared error.",
    infer: (c) => {
      const [a, b] = c.ins;
      if (!a || !b) return { out: scalar };
      if (a.shape.length !== b.shape.length)
        c.err("AXS0402", `mse operands differ in rank (${a.shape.length} vs ${b.shape.length})`);
      else for (let i = 0; i < a.shape.length; i++) c.eq(a.shape[i], b.shape[i], `mse axis ${i}`);
      return { out: scalar };
    },
  },
  {
    name: "bce",
    style: "primitive",
    category: "objective",
    ports: ["pred", "target"],
    config: [],
    effects: ["pure"],
    doc: "Binary cross entropy on probabilities.",
    infer: (c) => {
      const [a, b] = c.ins;
      if (a && b) for (let i = 0; i < Math.min(a.shape.length, b.shape.length); i++) c.eq(a.shape[i], b.shape[i], `bce axis ${i}`);
      return { out: scalar };
    },
  },
  {
    name: "cosine_similarity",
    style: "primitive",
    category: "objective",
    ports: ["a", "b"],
    config: [{ name: "axis", kind: "axis" }],
    effects: ["pure"],
    doc: "Cosine similarity along an axis.",
    infer: (c) => {
      const [a, b] = c.ins;
      if (!a || !b) return { out: scalar };
      let ax = c.num("axis", -1);
      if (ax < 0) ax += a.shape.length;
      for (let i = 0; i < a.shape.length; i++) c.eq(a.shape[i], b.shape[i], `cosine axis ${i}`);
      const shape = a.shape.slice();
      shape.splice(ax, 1);
      return { out: tensor(shape), attrs: { axis: ax } };
    },
  },
  {
    name: "l2",
    style: "primitive",
    category: "objective",
    ports: ["x"],
    config: [],
    effects: ["pure"],
    doc: "Sum of squares — useful as an explicit regularisation term.",
    infer: () => ({ out: scalar }),
  },
];

export const CATALOG_MAP = new Map(CATALOG.map((o) => [o.name, o]));

// ------------------------------------------------------------------ data ops

export interface DataOpSpec {
  name: string;
  doc: string;
  stochastic?: boolean;
  fits?: string; // statistic fitted from data
  needs?: string;
  reshape?: (s: DimExpr[], args: number[]) => DimExpr[];
  kindOut?: TensorKind;
}

export const DATA_OPS: DataOpSpec[] = [
  { name: "decode", doc: "decode raw bytes supplied by the source adapter into a tensor" },
  { name: "resize", doc: "resize spatial axes", reshape: (s, a) => (s.length === 3 ? [s[0], dConst(a[0]), dConst(a[1] ?? a[0])] : s) },
  { name: "center_crop", doc: "deterministic centre crop", reshape: (s, a) => (s.length === 3 ? [s[0], dConst(a[0]), dConst(a[0])] : s) },
  { name: "random_crop", doc: "random crop (train-time augmentation)", stochastic: true, reshape: (s, a) => (s.length === 3 ? [s[0], dConst(a[0]), dConst(a[0])] : s) },
  { name: "random_flip", doc: "random horizontal flip", stochastic: true },
  { name: "color_jitter", doc: "random brightness/contrast/saturation jitter", stochastic: true },
  { name: "noise", doc: "add gaussian noise of the given scale (train-time augmentation for feature vectors)", stochastic: true },
  { name: "to_float", doc: "convert to float in [0, 1]" },
  { name: "normalize", doc: "subtract mean / divide by std", fits: "mean+std" },
  { name: "standardize", doc: "zero-mean unit-variance per feature", fits: "mean+std" },
  { name: "impute", doc: "replace missing values", fits: "median|mean" },
  { name: "vocab", doc: "build a categorical vocabulary; `unknown: \"<unk>\"` maps unseen categories, `unknown: error` refuses them (AXS0622 when neither is given)", fits: "vocabulary" },
  { name: "encode", doc: "map categories to indices using a fitted vocabulary", kindOut: "Tokens" },
  { name: "tokenize", doc: "text -> token ids", kindOut: "Tokens" },
  { name: "window", doc: "cut a token stream into fixed windows" },
  { name: "pad_to", doc: "right-pad (or truncate) a variable-length token sequence to a fixed length", reshape: (s, a) => (s.length === 1 ? [dConst(a[0])] : s) },
  { name: "pad_mask", doc: "true at padded positions of a sequence padded to the given length — the attention mask contract (true = blocked), written Mask[1, T]", kindOut: "Mask", reshape: (s, a) => (s.length === 1 ? [dConst(1), dConst(a[0])] : s) },
  { name: "as_class", doc: "interpret as an integer class label", kindOut: "Class" },
  { name: "select", doc: "select a subset of tabular features" },
  { name: "mask_tokens", doc: "randomly mask token positions", stochastic: true },
];

export const DATA_OP_MAP = new Map(DATA_OPS.map((o) => [o.name, o]));

// ------------------------------------------------------------------ sources

export const SOURCE_ADAPTERS = [
  { name: "image_folder", doc: "directory of labelled images (external adapter)" },
  { name: "csv", doc: "tabular rows (external adapter)" },
  { name: "text_file", doc: "raw text shards (external adapter)" },
  { name: "synthetic", doc: "deterministic generated data — always executable in the reference backend" },
  { name: "tensor_store", doc: "pre-materialised tensors from an external store" },
];

// ------------------------------------------------------------------ backends

export interface Capability {
  forward: boolean;
  grad: boolean;
  note?: string;
}

/** Reference backend (in-browser tensor engine). */
export const REF_CAPS: Record<string, Capability> = {
  conv2d: { forward: true, grad: true, note: "naive im2col; small inputs only" },
  maxpool2d: { forward: true, grad: true },
  avgpool2d: { forward: true, grad: true },
  attention: { forward: true, grad: true },
  batchnorm: { forward: true, grad: true },
  argmax: { forward: true, grad: false, note: "mathematically non-differentiable" },
  stop_grad: { forward: true, grad: false, note: "gradient intentionally stopped" },
  scan: { forward: true, grad: true, note: "unrolled at runtime" },
};

export const TORCH_CAPS: Record<string, Capability> = {
  argmax: { forward: true, grad: false, note: "mathematically non-differentiable" },
  stop_grad: { forward: true, grad: false, note: "gradient intentionally stopped" },
};

export function capabilityOf(op: string, backend: "reference" | "torch"): Capability {
  const table = backend === "reference" ? REF_CAPS : TORCH_CAPS;
  return table[op] ?? { forward: true, grad: true };
}
