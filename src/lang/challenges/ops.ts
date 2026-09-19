/**
 * TENSA hardening — per-operator symbolic shape table (§5.3).
 *
 * Every CATALOG op is applied once to a fully symbolic input (no dimension has
 * a value) and its inferred output type is compared verbatim.  The test that
 * consumes this table also asserts that no catalog op is missing from it, so
 * adding an op without a symbolic shape case fails the suite.
 */
import { compile } from "../analyze";
import { CATALOG } from "../catalog";
import { showType } from "../types";

export interface OpShapeCase {
  op: string;
  /** declared type of the single input `x` (a second input `y` when `y` is set) */
  x: string;
  y?: string;
  /** expression over x (and y) */
  call: string;
  /** exact inferred output type (as `inspect` renders it) */
  out: string;
  /** source-level spelling of the declared result when `out` uses ⌊⌋ (not writable in source) */
  declared?: string;
  /** warnings this symbolic application is allowed (and required) to raise */
  warn?: string[];
}

const DIMS = "dim B\ndim T\ndim D\ndim C\ndim H\ndim W\ndim V\ndim K\n";

export const OP_SHAPE_CASES: OpShapeCase[] = [
  { op: "linear", x: "Tensor[B, T, D]", call: "linear(K)", out: "Tensor[B, T, K]" },
  { op: "conv2d", x: "Tensor[B, C, H, W]", call: "conv2d(K, kernel: 3, pad: 1)", out: "Tensor[B, K, H, W]" },
  {
    op: "conv2d",
    x: "Tensor[B, C, H, W]",
    call: "conv2d(K, kernel: 3, stride: 2, pad: 1)",
    out: "Tensor[B, K, ⌊(H + 1)/2⌋, ⌊(W + 1)/2⌋]",
    declared: "Tensor[B, K, (H + 1) / 2, (W + 1) / 2]",
  },
  {
    op: "conv2d",
    x: "Tensor[B, C, H, W]",
    call: "conv2d(K, kernel: 4, stride: 2, pad: 1)",
    out: "Tensor[B, K, ⌊H/2⌋, ⌊W/2⌋]",
    declared: "Tensor[B, K, H / 2, W / 2]",
  },
  { op: "embedding", x: "Tokens[B, T]", call: "embedding(V, D)", out: "Tensor[B, T, D]" },
  { op: "positional", x: "Tensor[B, T, D]", call: "positional(max: 512)", out: "Tensor[B, T, D]" },
  { op: "attention", x: "Tensor[B, T, 4 * K]", call: "attention(heads: 4)", out: "Tensor[B, T, 4*K]" },
  { op: "attention", x: "Tensor[B, T, D]", call: "attention(heads: 4)", out: "Tensor[B, T, D]", warn: ["AXS0403"] },
  { op: "layernorm", x: "Tensor[B, T, D]", call: "layernorm", out: "Tensor[B, T, D]" },
  { op: "rmsnorm", x: "Tensor[B, T, D]", call: "rmsnorm", out: "Tensor[B, T, D]" },
  { op: "batchnorm", x: "Tensor[B, C, H, W]", call: "batchnorm", out: "Tensor[B, C, H, W]" },
  { op: "randn_like", x: "Tensor[B, T, D]", call: "randn_like", out: "Tensor[B, T, D]" },
  { op: "dropout", x: "Tensor[B, T, D]", call: "dropout(0.1)", out: "Tensor[B, T, D]" },
  { op: "maxpool2d", x: "Tensor[B, C, H, W]", call: "maxpool2d(2)", out: "Tensor[B, C, ⌊H/2⌋, ⌊W/2⌋]", declared: "Tensor[B, C, H / 2, W / 2]" },
  { op: "avgpool2d", x: "Tensor[B, C, H, W]", call: "avgpool2d(2)", out: "Tensor[B, C, ⌊H/2⌋, ⌊W/2⌋]", declared: "Tensor[B, C, H / 2, W / 2]" },
  { op: "upsample", x: "Tensor[B, C, H, W]", call: "upsample(2)", out: "Tensor[B, C, 2*H, 2*W]" },
  { op: "global_avgpool", x: "Tensor[B, C, H, W]", call: "global_avgpool", out: "Tensor[B, C]" },
  { op: "flatten", x: "Tensor[B, C, H, W]", call: "flatten", out: "Tensor[B, C*H*W]" },
  { op: "relu", x: "Tensor[B, T, D]", call: "relu", out: "Tensor[B, T, D]" },
  { op: "gelu", x: "Tensor[B, T, D]", call: "gelu", out: "Tensor[B, T, D]" },
  { op: "silu", x: "Tensor[B, T, D]", call: "silu", out: "Tensor[B, T, D]" },
  { op: "sigmoid", x: "Tensor[B, T, D]", call: "sigmoid", out: "Tensor[B, T, D]" },
  { op: "tanh", x: "Tensor[B, T, D]", call: "tanh", out: "Tensor[B, T, D]" },
  { op: "exp", x: "Tensor[B, T, D]", call: "exp", out: "Tensor[B, T, D]" },
  { op: "log", x: "Tensor[B, T, D]", call: "log", out: "Tensor[B, T, D]" },
  { op: "sqrt", x: "Tensor[B, T, D]", call: "sqrt", out: "Tensor[B, T, D]" },
  { op: "rsqrt", x: "Tensor[B, T, D]", call: "rsqrt", out: "Tensor[B, T, D]" },
  { op: "abs", x: "Tensor[B, T, D]", call: "abs", out: "Tensor[B, T, D]" },
  { op: "neg", x: "Tensor[B, T, D]", call: "neg", out: "Tensor[B, T, D]" },
  { op: "softmax", x: "Logits[B, T, V]", call: "softmax", out: "Probs[B, T, V]" },
  { op: "log_softmax", x: "Logits[B, T, V]", call: "log_softmax", out: "Tensor[B, T, V]" },
  { op: "matmul", x: "Tensor[B, T, D]", y: "Tensor[B, D, K]", call: "matmul(x, y)", out: "Tensor[B, T, K]" },
  { op: "transpose", x: "Tensor[B, T, D]", call: "transpose(x, 1, 2)", out: "Tensor[B, D, T]" },
  { op: "reshape", x: "Tensor[B, T, D]", call: "reshape(x, [B, T * D])", out: "Tensor[B, D*T]" },
  { op: "reshape", x: "Tensor[B, T, 4 * K]", call: "reshape(x, [B, T, 4, K])", out: "Tensor[B, T, 4, K]" },
  { op: "concat", x: "Tensor[B, T, D]", y: "Tensor[B, T, K]", call: "concat(x, y, axis: -1)", out: "Tensor[B, T, D + K]" },
  { op: "stack", x: "Tensor[B, D]", y: "Tensor[B, D]", call: "stack(x, y, axis: 1)", out: "Tensor[B, 2, D]" },
  { op: "sum", x: "Tensor[B, T, D]", call: "sum(x, axis: 1)", out: "Tensor[B, D]" },
  { op: "mean", x: "Tensor[B, T, D]", call: "mean(x, axis: -1)", out: "Tensor[B, T]" },
  { op: "mean", x: "Tensor[B, T, D]", call: "mean(x, axis: 1, keep: true)", out: "Tensor[B, 1, D]" },
  { op: "max", x: "Tensor[B, T, D]", call: "max(x, axis: 2)", out: "Tensor[B, T]" },
  { op: "min", x: "Tensor[B, T, D]", call: "min(x, axis: 0)", out: "Tensor[T, D]" },
  { op: "argmax", x: "Logits[B, V]", call: "argmax(x, axis: -1)", out: "Class[B]" },
  { op: "masked_fill", x: "Tensor[B, T, T]", y: "Mask[T, T]", call: "masked_fill(x, y, value: -1e9)", out: "Tensor[B, T, T]" },
  { op: "causal_mask", x: "Tensor[B, T, D]", call: "causal_mask(T)", out: "Mask[T, T]" },
  { op: "one_hot", x: "Class[B, T]", call: "one_hot(x, classes: V)", out: "Tensor[B, T, V]" },
  { op: "stop_grad", x: "Tensor[B, T, D]", call: "stop_grad", out: "Tensor[B, T, D]" },
  { op: "detach_kind", x: "Tensor[B, T, V]", call: `detach_kind(x, as: "Logits")`, out: "Logits[B, T, V]" },
  { op: "cross_entropy", x: "Logits[B, T, V]", y: "Tokens[B, T]", call: "cross_entropy(x, y)", out: "Scalar" },
  { op: "mse", x: "Tensor[B, D]", y: "Tensor[B, D]", call: "mse(x, y)", out: "Scalar" },
  { op: "bce", x: "Probs[B, 1]", y: "Tensor[B, 1]", call: "bce(x, y)", out: "Scalar" },
  { op: "cosine_similarity", x: "Tensor[B, D]", y: "Tensor[B, D]", call: "cosine_similarity(x, y)", out: "Tensor[B]" },
  { op: "l2", x: "Tensor[B, D]", call: "l2", out: "Scalar" },
];

export function opCaseSource(c: OpShapeCase): string {
  const params = c.y ? `x: ${c.x}, y: ${c.y}` : `x: ${c.x}`;
  // the declared result is the inferred type itself: the contract check then
  // doubles as a second, independent equality proof
  const explicit = /\bx\b/.test(c.call) || c.op === "causal_mask";
  const body = explicit ? `return ${c.call}` : `return x |> ${c.call}`;
  return `${DIMS}model M(${params}) -> ${c.declared ?? c.out} { ${body} }`;
}

/** Compile one case; returns problems (empty when the op behaves). */
export function checkOpCase(c: OpShapeCase): string[] {
  const problems: string[] = [];
  const r = compile(opCaseSource(c), `op:${c.op}`);
  for (const e of r.errors) problems.push(`error ${e.code}: ${e.message}`);
  const warnAllowed = new Set(c.warn ?? []);
  for (const w of r.warnings) if (!warnAllowed.has(w.code)) problems.push(`unexpected warning ${w.code}: ${w.message}`);
  for (const w of warnAllowed) if (!r.warnings.some((x) => x.code === w)) problems.push(`expected warning ${w} did not fire`);
  const g = r.mod.graphs.find((x) => x.name === "M");
  const got = g ? g.outputs.map((o) => showType(r.mod.values.get(o)!.type)).join(", ") : "(no graph)";
  if (got !== c.out) problems.push(`inferred ${got}, expected ${c.out}`);
  return problems;
}

export function uncoveredOps(): string[] {
  const covered = new Set(OP_SHAPE_CASES.map((c) => c.op));
  return CATALOG.map((o) => o.name).filter((n) => !covered.has(n));
}
