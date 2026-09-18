/**
 * TENSA — semantic and regression test suite.
 * Runs in the browser; every example in the catalogue is compiled here.
 */
import { compile } from "./analyze";
import { capabilityOf } from "./catalog";
import { CHALLENGES, METAMORPHIC } from "./challenges/corpus";
import { expand, leakageHits } from "./challenges/driver";
import { assertEquivalent } from "./challenges/metamorphic";
import { OP_SHAPE_CASES, checkOpCase, uncoveredOps } from "./challenges/ops";
import { PROPERTIES, checkProperty } from "./challenges/property";
import { EXAMPLES } from "./examples";
import { emitTorch } from "./emit_torch";
import { Runtime, runProgram } from "./exec";
import { inspectModule } from "./inspect";
import { IRModule, IRNode, printIR } from "./ir";
import * as X from "./tensor";
import { TensorType, isTensor } from "./types";

export interface TestResult {
  name: string;
  group: string;
  ok: boolean;
  detail: string;
}

const results: TestResult[] = [];

function t(group: string, name: string, fn: () => string) {
  try {
    const detail = fn();
    results.push({ group, name, ok: true, detail });
  } catch (e) {
    results.push({ group, name, ok: false, detail: (e as Error).message });
  }
}

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

function allNodes(mod: IRModule): IRNode[] {
  const out: IRNode[] = [];
  const walk = (ns: IRNode[]) => {
    for (const n of ns) {
      out.push(n);
      if (n.regions) for (const r of n.regions) walk(r.nodes);
    }
  };
  for (const g of mod.graphs) walk(g.nodes);
  for (const o of mod.objectives) walk(o.graph.nodes);
  return out;
}

function codes(mod: IRModule): string[] {
  return mod.diags.map((d) => d.code);
}

function paramsOf(mod: IRModule, prefix: string) {
  return mod.params.filter((p) => p.owner === prefix || p.owner.startsWith(prefix + "/"));
}

function countValues(mod: IRModule, prefix: string): number {
  let n = 0;
  const env = new Map<string, number>();
  for (const d of mod.dims) if (d.value) env.set(d.name, Number(d.value.terms[0]?.coef ?? 0));
  for (const p of paramsOf(mod, prefix)) {
    let prod = 1;
    for (const s of p.shape) {
      const c = s.terms.length === 1 && s.terms[0].vars.length === 0 ? s.terms[0].coef : null;
      if (c === null) return -1;
      prod *= c;
    }
    n += prod;
  }
  return n;
}

export function runTests(): TestResult[] {
  results.length = 0;

  // ---------------------------------------------------------------- parsing
  for (const ex of EXAMPLES) {
    t("parsing", `example '${ex.id}' parses`, () => {
      const r = compile(ex.code, ex.id);
      const parseErrors = r.mod.diags.filter((d) => d.code.startsWith("AXS010"));
      assert(parseErrors.length === 0, `parse errors: ${parseErrors.map((d) => `${d.code} ${d.message}`).join("; ")}`);
      return `${r.mod.graphs.length} graphs, ${r.mod.params.length} parameter tables`;
    });
  }
  for (const ex of EXAMPLES.filter((e) => e.id !== "diagnostics")) {
    t("examples", `example '${ex.id}' has no errors`, () => {
      const r = compile(ex.code, ex.id);
      assert(
        r.errors.length === 0,
        r.errors.map((d) => `${d.code} @${d.loc.line}:${d.loc.col} ${d.message}`).join(" | ")
      );
      return `${r.warnings.length} warning(s)`;
    });
  }

  // ---------------------------------------------------------------- shapes
  t("shapes", "symbolic dimension propagation through an MLP", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 784]) -> Tensor[B, 10] {
  linear(256)
  gelu
  linear(10)
}`);
    assert(r.ok, "should compile");
    const lin = allNodes(r.mod).filter((n) => n.op === "linear");
    const ty = r.mod.values.get(lin[0].outputs[0])!.type as TensorType;
    assert(ty.shape.length === 2 && ty.shape[1].terms[0].coef === 256, "first linear should be [B, 256]");
    return "shapes propagate symbolically with B free";
  });

  t("shapes", "symbolic expressions: D*4 and D/Heads", () => {
    const r = compile(`dim B
dim T
dim D = 512
dim Heads = 8
dim DH = D / Heads
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  linear(D * 4)
  gelu
  linear(D)
}
fn f(x: Tensor[B, Heads, T, DH]) -> Tensor[B, Heads, T, DH] { return x }`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const lin = allNodes(r.mod).filter((n) => n.op === "linear");
    const ty = r.mod.values.get(lin[0].outputs[0])!.type as TensorType;
    assert(ty.shape[2].terms[0].coef === 2048, "D*4 should fold to 2048");
    return "D*4 = 2048, DH = 64 (exact division proved)";
  });

  t("shapes", "different symbolic names are NOT assumed compatible", () => {
    const r = compile(`dim B
dim S
dim T
dim D = 8
model M(a: Tensor[B, S, D], b: Tensor[B, T, D]) -> Tensor[B, S, D] { return a + b }`);
    assert(codes(r.mod).includes("AXS0403"), `expected AXS0403, got ${codes(r.mod).join(",")}`);
    const carried = r.mod.constraints.filter((c) => c.status === "assumed");
    assert(carried.length > 0, "constraint should be carried into the IR");
    return `carried constraint: ${carried[0].origin}`;
  });

  t("shapes", "invalid rank is rejected", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 16] { conv2d(4, kernel: 3) }`);
    assert(codes(r.mod).includes("AXS0402"), `expected AXS0402, got ${codes(r.mod).join(",")}`);
    return "conv2d on a rank-2 tensor reports AXS0402";
  });

  t("shapes", "return contract is verified", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 10] { linear(12) }`);
    assert(codes(r.mod).includes("AXS0401") || codes(r.mod).includes("AXS0406"), "expected a contract violation");
    return "declared [B, 10] vs produced [B, 12] is reported";
  });

  t("shapes", "reshape element count must be provable", () => {
    const r = compile(`dim B
dim T
dim D = 64
model M(x: Tensor[B, T, D]) -> Tensor[B, T, 32] { return reshape(x, [B, T, 32]) }`);
    assert(r.mod.constraints.some((c) => c.status !== "proved"), "element-count constraint should not silently pass");
    return "reshape mismatch is reported instead of assumed";
  });

  t("shapes", "same-sign polynomial differences are refuted, mixed-sign ones are carried (§5.2)", () => {
    const r = compile(`dim B
dim K
dim S
dim T
model Refuted(x: Tensor[B, 2 * K]) -> Tensor[B, 4 * K] { return x }
model Carried(x: Tensor[B, S]) -> Tensor[B, T] { return x }`);
    const byOrigin = (s: string) => r.mod.constraints.find((c) => c.origin.includes(s))!;
    assert(byOrigin("Refuted").status === "failed", "2*K = 4*K should be refuted: no dimension ≥ 1 satisfies it");
    assert(byOrigin("Carried").status === "assumed", "S = T must stay carried");
    assert(codes(r.mod).includes("AXS0401") && codes(r.mod).includes("AXS0403"), `got ${codes(r.mod).join(",")}`);
    return "2*K ≠ 4*K is an error; S = T is carried";
  });

  t("shapes", "carried constraints are enforced before execution (F-002)", () => {
    const r = compile(`dim B
dim S
dim T
model M(a: Tensor[B, S], b: Tensor[B, T]) -> Tensor[B, S] { return a + b }`);
    assert(r.ok, "should compile with a carried S = T");
    const bad = runProgram(r.mod, { maxSteps: 0, dims: { B: 1, S: 3, T: 2 } });
    assert(bad.errors.some((e) => e.includes("carried constraint violated")), `expected refusal, got ${bad.errors.join("; ")}`);
    assert(bad.forward.length === 0, "nothing may execute after a refusal");
    const good = runProgram(r.mod, { maxSteps: 0, dims: { B: 1, S: 3, T: 3 } });
    assert(good.errors.length === 0 && good.forward.length === 1, `S = T = 3 should run: ${good.errors.join("; ")}`);
    return "S≠T refused before any forward pass; S=T runs";
  });

  t("shapes", "slice bounds are proved, refuted or carried as ≤ constraints (F-011)", () => {
    const head = `dim B
dim T
dim W
`;
    // provable without help: 1 ≤ T, T-1 ≥ 0, T ≤ T — no diagnostics at all
    const fine = compile(`${head}model S(x: Tensor[B, T, 8]) -> Tensor[B, 1, 8] { let a = x[:, 1:T, :] ; return x[:, T-1:T, :] }`);
    assert(fine.ok && fine.mod.diags.length === 0, `provable bounds must be silent: ${fine.mod.diags.map((d) => d.message).join("; ")}`);
    // provably violated: constant overshoot, index at the extent, start after end
    for (const s of ["x[:, 0:T+1, :]", "x[:, T, :]", "x[:, 3:2, :]", "x[:, :, 4:9]"]) {
      const r = compile(`${head}model S(x: Tensor[B, T, 8]) -> Tensor[B, 8] { return ${s} }`);
      assert(r.errors.some((e) => e.code === "AXS0410"), `${s} must be refuted; got ${r.mod.diags.map((d) => d.code).join(", ") || "nothing"}`);
    }
    // unprovable: carried with rel ≤, verified against the bindings before any forward pass
    const win = compile(`${head}model S(x: Tensor[B, T, 8]) -> Tensor[B, W, 8] { return x[:, 2:W+2, :] }`);
    assert(win.ok && win.warnings.some((w) => w.code === "AXS0403" && w.message.includes("W + 2 ≤ T")), "W + 2 ≤ T must be carried");
    const c = win.mod.constraints.find((k) => k.rel === "<=" && k.status === "assumed");
    assert(!!c, "carried bound must be a ≤ constraint in the IR");
    const bad = runProgram(win.mod, { maxSteps: 0, dims: { B: 1, T: 3, W: 2 } });
    assert(bad.errors.some((e) => e.includes("W + 2 ≤ T but 4 > 3")) && bad.forward.length === 0, `T = 3 must be refused: ${bad.errors.join("; ")}`);
    const good = runProgram(win.mod, { maxSteps: 0, dims: { B: 1, T: 4, W: 2 } });
    assert(good.errors.length === 0 && good.forward[0].output === "[1, 2, 8]", `T = 4 should run: ${good.errors.join("; ")}`);
    return "1:T and T-1:T silent; 4 overruns refuted; W+2 ≤ T carried and enforced";
  });

  t("shapes", "attention and masked_fill check the mask shape (F-012)", () => {
    const head = `dim B
dim T
dim S
dim D = 16
`;
    const ok = compile(`${head}model X(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, T, D] { return attention(query: q, key: kv, value: kv, mask: pad, heads: 4) }`);
    assert(ok.ok && ok.mod.diags.length === 0, `Mask[B, 1, S] is the padding-mask contract: ${ok.mod.diags.map((d) => d.message).join("; ")}`);
    const run = runProgram(ok.mod, { maxSteps: 0, dims: { B: 2, T: 5, S: 3 } });
    assert(run.errors.length === 0 && run.forward[0].output === "[2, 5, 16]", `rank-3 mask must align against [B, H, Tq, Tk]: ${run.errors.join("; ")}`);
    // a rank-2 mask is [Tq, Tk]: Mask[B, S] is a carried B = T, refused when B ≠ T is bound
    const two = compile(`${head}model X(q: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, S]) -> Tensor[B, T, D] { return attention(query: q, key: kv, value: kv, mask: pad, heads: 4) }`);
    assert(two.warnings.some((w) => w.code === "AXS0403" && w.message.includes("mask query length")), "Mask[B, S] must not pass silently as a padding mask");
    assert(runProgram(two.mod, { maxSteps: 0, dims: { B: 2, T: 5, S: 3 } }).errors.length > 0, "B ≠ T must be refused at runtime");
    // constants are refuted outright; wrong rank is a rank error
    const wrong = compile(`dim B
dim D = 16
model X(q: Tensor[B, 5, D], kv: Tensor[B, 3, D], pad: Mask[B, 1, 5]) -> Tensor[B, 5, D] { return attention(query: q, key: kv, value: kv, mask: pad, heads: 4) }`);
    assert(wrong.errors.some((e) => e.code === "AXS0401" && e.message.includes("mask key length")), "mask over the query length must be refuted");
    const rank = compile(`${head}model X(q: Tensor[B, T, D], pad: Mask[B, 1, 1, T]) -> Tensor[B, T, D] { return attention(query: q, mask: pad, heads: 4) }`);
    assert(rank.errors.some((e) => e.code === "AXS0402"), "rank-4 mask must be a rank error");
    const mf = compile(`${head}model X(s: Tensor[B, T, S], pad: Mask[T + 1, S]) -> Tensor[B, T, S] { return masked_fill(s, pad, value: -1.0) }`);
    assert(mf.errors.some((e) => e.code === "AXS0401" && e.message.includes("broadcast")), "masked_fill must check trailing broadcast");
    return "Mask[B, 1, S] accepted and aligned; Mask[B, S] carried and refused; constants refuted; rank and broadcast errors";
  });

  t("shapes", "optional ports keep their position: attention(mask:) does not slide into the key slot (F-013)", () => {
    const r = compile(`dim B
dim T
dim D = 16
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return attention(query: x, mask: causal_mask(T), heads: 4) }`);
    assert(r.ok && r.mod.diags.length === 0, `mask-only attention must check cleanly: ${r.mod.diags.map((d) => d.message).join("; ")}`);
    const att = r.mod.graphs[0].nodes.find((n) => n.op === "attention")!;
    assert(att.inputs.length === 4, `expected 4 tensor inputs (query, key, value, mask); got ${att.inputs.length}`);
    assert(att.inputs[1] === att.inputs[0] && att.inputs[2] === att.inputs[0], "skipped key/value default to the query");
    const mt = r.mod.values.get(att.inputs[3])!.type;
    assert(isTensor(mt) && mt.kind === "Mask", "the fourth input is the mask");
    return "inputs = [query, query, query, mask]";
  });

  t("shapes", "unknown shape does not become a rank (F-009)", () => {
    const r = compile(`dim B
custom op mystery(x: Tensor[B, 8]) { effects: pure  shape: unknown }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { return mystery(x) |> linear(8) + x }`);
    assert(r.ok, `spurious errors: ${r.errors.map((e) => e.message).join("; ")}`);
    assert(r.mod.params.length === 0, "linear must not be sized from an unknown input");
    const out = r.mod.values.get(r.mod.graphs[0].outputs[0])!.type;
    assert(isTensor(out) && out.unknown === true, "result must stay unknown");
    return "unknown stays unknown through linear and +";
  });

  // ---------------------------------------------------------------- topology
  t("topology", "residual shape mismatch demands an explicit projection", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual { conv2d(64, kernel: 3, stride: 2, pad: 1) }
}`);
    assert(codes(r.mod).includes("AXS0404"), `expected AXS0404, got ${codes(r.mod).join(",")}`);
    return "AXS0404 suggests `residual via ...`";
  });

  t("topology", "projected residual type-checks and lowers to an explicit add", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual via conv2d(64, kernel: 1, stride: 2) {
    conv2d(64, kernel: 3, stride: 2, pad: 1)
    batchnorm
  }
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const res = allNodes(r.mod).find((n) => n.op === "residual")!;
    assert(res.regions!.length === 2, "residual should carry body + projection regions");
    const add = allNodes(r.mod).find((n) => n.op === "add" && n.note === "residual merge");
    assert(Boolean(add), "the residual merge must be an explicit IR node");
    return "residual = 2 regions + explicit add";
  });

  t("topology", "REGRESSION: parallel branches are not compiled sequentially", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 96, 32, 32] {
  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3, pad: 1)
    conv2d(32, kernel: 5, pad: 2)
  }
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const par = allNodes(r.mod).find((n) => n.op === "parallel");
    assert(Boolean(par), "a parallel node must exist");
    const input = par!.inputs[0];
    const firstOps = par!.regions!.map((rg) => rg.nodes[0]);
    for (const [i, nd] of firstOps.entries())
      assert(nd.inputs[0] === input, `branch ${i} consumes %${nd.inputs[0]} instead of the split input %${input}`);
    const concat = allNodes(r.mod).find((n) => n.op === "concat" && n.note === "explicit branch merge");
    assert(Boolean(concat), "the merge must appear explicitly in the IR");
    assert(concat!.inputs.length === 3, "concat must consume all three branch results");
    const outTy = r.mod.values.get(concat!.outputs[0])!.type as TensorType;
    assert(outTy.shape[1].terms[0].coef === 96, "concat axis 1 must sum to 96 channels");
    return "3 branches share one input; explicit concat -> 96 channels";
  });

  t("topology", "concat compatibility is enforced", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 64, 32, 32] {
  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3)
  }
}`);
    assert(codes(r.mod).includes("AXS0405"), `expected AXS0405, got ${codes(r.mod).join(",")}`);
    return "30x30 vs 32x32 on a non-concat axis is an error";
  });

  t("topology", "multi-statement branch bodies keep branch boundaries", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16, 32, 32]) -> Tensor[B, 64, 32, 32] {
  split merge concat(1) {
    branch a { conv2d(32, kernel: 1) ; relu }
    branch b { maxpool2d(3, stride: 1, pad: 1) ; conv2d(32, kernel: 1) }
  }
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const par = allNodes(r.mod).find((n) => n.op === "parallel")!;
    assert(par.regions!.every((rg) => rg.nodes.length === 2), "each branch should contain two nodes");
    assert(par.regions![0].label === "a" && par.regions![1].label === "b", "branch labels are preserved");
    return "labelled multi-statement branches";
  });

  // ---------------------------------------------------------------- flow
  t("flow", "implicit cursor is refused when there are multiple inputs", () => {
    const r = compile(`dim B
model M(a: Tensor[B, 16], b: Tensor[B, 16]) -> Tensor[B, 32] { linear(32) }`);
    assert(codes(r.mod).includes("AXS0301"), `expected AXS0301, got ${codes(r.mod).join(",")}`);
    return "no hidden guess about which stream is meant";
  });

  t("flow", "multi-input models work once a stream is seeded", () => {
    const r = compile(`dim B
dim D = 32
model M(a: Tensor[B, D], b: Tensor[B, D]) -> Tensor[B, D] {
  let joined = concat(a, b, axis: -1)
  joined |> linear(D) |> gelu
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    return "explicit join re-establishes a single implicit stream";
  });

  t("flow", "tuple results and destructuring", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> (Tensor[B, 8], Tensor[B, 4]) {
  let h = linear(16) |> gelu
  return (h |> linear(8), h |> linear(4))
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const g = r.mod.graphs[0];
    assert(g.resultType!.t === "tuple", "result should be a tuple type");
    assert(g.outputs.length === 2, "two graph outputs");
    return "(Tensor[B, 8], Tensor[B, 4])";
  });

  // ---------------------------------------------------------------- parameters
  t("parameters", "MLP parameter count", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 784]) -> Tensor[B, 10] {
  linear(256)
  gelu
  linear(128)
  gelu
  linear(10)
}`);
    const n = countValues(r.mod, "M");
    assert(n === 235146, `expected 235146 values, got ${n}`);
    return "235,146 values in 6 tables";
  });

  t("parameters", "Siamese: reusing a bound stage shares one parameter set", () => {
    const ex = EXAMPLES.find((e) => e.id === "siamese")!;
    const r = compile(ex.code, "siamese");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const siam = countValues(r.mod, "Siamese");
    const two = countValues(r.mod, "TwoTowers");
    assert(two === 2 * siam, `expected TwoTowers (${two}) to be exactly twice Siamese (${siam})`);
    const shared = paramsOf(r.mod, "Siamese").every((p) => p.applications.length === 2);
    assert(shared, "every Siamese parameter must record two applications");
    return `tied encoder: ${siam} values; recreated: ${two} values`;
  });

  t("parameters", "rebinding a let name does not tie parameters (F-001)", () => {
    const r = compile(`dim B
block Enc(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] {
  let y = Enc(x)
  let y = Enc(x)
  return y
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const n = countValues(r.mod, "M");
    assert(n === 144, `two independent applications should own 144 values, got ${n}`);
    const owners = new Set(r.mod.params.map((p) => p.owner));
    assert(owners.size === 2, `expected 2 distinct owners, got ${[...owners].join(", ")}`);
    assert(r.mod.params.every((p) => p.applications.length === 1), "no parameter may record two applications");
    return "144 values in 4 tables, two owners";
  });

  t("parameters", "rebinding a let name in the same scope warns; nested regions do not (E-001)", () => {
    const r = compile(`dim B
block Enc(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, 8] {
  let y = Enc(x)
  let y = Enc(x)
  return y
}`);
    const w = r.warnings.filter((d) => d.code === "AXS0304");
    assert(w.length === 1 && w[0].loc.line === 5, `expected one AXS0304 at line 5, got ${w.map((d) => d.loc.line).join(",")}`);
    assert(w[0].where === "M", `where should be the model, got ${w[0].where}`);
    const ok = compile(`dim B
model N(x: Tensor[B, 8]) -> Tensor[B, 8] {
  let y = x |> linear(8)
  residual { let y = linear(8) ; gelu }
  return y
}`);
    assert(ok.ok && !ok.warnings.some((d) => d.code === "AXS0304"), "a branch-local let is a fresh scope");
    return "same-scope rebinding warns once, at the second let";
  });

  t("parameters", "static repetition creates independent parameters", () => {
    const r = compile(`dim B
dim D = 16
block Layer(x: Tensor[B, D]) -> Tensor[B, D] { linear(D) }
model M(x: Tensor[B, D]) -> Tensor[B, D] { for 12: Layer() }`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const owners = new Set(paramsOf(r.mod, "M").map((p) => p.owner));
    assert(owners.size === 12, `expected 12 distinct owners, got ${owners.size}`);
    const rep = allNodes(r.mod).find((n) => n.op === "static_repeat")!;
    assert(rep.attrs.mode === "independent-parameters", `mode was ${rep.attrs.mode}`);
    return "12 independent stages, mode reported in the IR";
  });

  t("parameters", "repeating a bound stage shares one parameter set", () => {
    const r = compile(`dim B
dim D = 16
block Layer(x: Tensor[B, D]) -> Tensor[B, D] { linear(D) }
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  let step = Layer()
  for 12: step
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const owners = new Set(paramsOf(r.mod, "M").map((p) => p.owner));
    assert(owners.size === 1, `expected 1 owner, got ${owners.size}`);
    const rep = allNodes(r.mod).find((n) => n.op === "static_repeat")!;
    assert(String(rep.attrs.mode).startsWith("shared-stage"), `mode was ${rep.attrs.mode}`);
    return "one parameter table applied 12 times";
  });

  t("parameters", "indexed repetition can vary the architecture", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 48] {
  for i in 0..3 { linear(16 * (i + 1)) }
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const widths = paramsOf(r.mod, "M")
      .filter((p) => p.role === "b")
      .map((p) => p.shape[0].terms[0].coef);
    assert(widths.join(",") === "16,32,48", `widths were ${widths.join(",")}`);
    return "widths 16, 32, 48";
  });

  t("parameters", "explicit and frozen parameters are tracked", () => {
    const r = compile(`dim B
dim D = 8
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  param w: Tensor[D, D] init: xavier
  frozen param bias: Tensor[D] init: zeros
  return matmul(x, w) + bias
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const w = r.mod.params.find((p) => p.role === "w")!;
    const b = r.mod.params.find((p) => p.role === "bias")!;
    assert(w.trainable && w.init === "xavier" && w.kind === "explicit", "w should be trainable/xavier/explicit");
    assert(!b.trainable, "frozen param must not be trainable");
    return "owner, shape, init, trainability recorded";
  });

  // ---------------------------------------------------------------- effects & state
  t("effects", "dropout is stochastic and training-sensitive", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { dropout(0.5) }`);
    const d = allNodes(r.mod).find((n) => n.op === "dropout")!;
    assert(d.effects.includes("stochastic") && d.effects.includes("training-sensitive"), "effects missing");
    return d.effects.join(", ");
  });

  t("effects", "batchnorm owns persistent running statistics", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 4, 8, 8]) -> Tensor[B, 4, 8, 8] { batchnorm }`);
    assert(r.mod.states.length === 2, `expected 2 state slots, got ${r.mod.states.length}`);
    assert(r.mod.states.every((s) => s.category === "running-stat" && s.checkpointed), "state metadata wrong");
    return "running_mean + running_var, checkpointed";
  });

  t("effects", "explicit persistent state updates only in training context", () => {
    const r = compile(`dim B
dim D = 8
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  state center: Tensor[D] init: zeros update: ema(0.9)
  let c = observe(center, mean(x, axis: 0))
  return x - c
}`);
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const upd = allNodes(r.mod).find((n) => n.op === "state_update")!;
    assert(upd.effects.includes("writes-state") && upd.effects.includes("training-sensitive"), "effects missing");
    assert(r.mod.states[0].category === "ema", "category should be ema");
    return "observe() reads state and schedules an EMA update";
  });

  t("effects", "gradient capability is a backend property, not a semantic one", () => {
    assert(capabilityOf("conv2d", "reference").grad, "conv2d must be differentiable");
    assert(!capabilityOf("argmax", "reference").grad, "argmax is mathematically non-differentiable");
    assert(!capabilityOf("stop_grad", "reference").grad, "stop_grad intentionally cuts the gradient");
    const r = compile(`dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { return stop_grad(x) + x }`);
    const sg = allNodes(r.mod).find((n) => n.op === "stop_grad")!;
    assert(sg.effects.includes("grad-stopped"), "stop_grad should carry the grad-stopped effect");
    return "three distinct notions kept apart";
  });

  // ---------------------------------------------------------------- objectives & data
  t("objectives", "objective must return a scalar", () => {
    const r = compile(`dim B
dim K = 4
objective O(logits: Logits[B, K], labels: Class[B]) -> Scalar { return logits }`);
    assert(codes(r.mod).includes("AXS0601") || codes(r.mod).includes("AXS0406"), "expected a scalar-result error");
    return "non-scalar objective rejected";
  });

  t("objectives", "objective port shapes are checked against model outputs", () => {
    const r = compile(`dim B
dim K = 4
model M(x: Tensor[B, 8]) -> Logits[B, 3] { linear(3) }
objective O(logits: Logits[B, K], labels: Class[B]) -> Scalar { return cross_entropy(logits, labels) }
source S = synthetic(features: 8)
data D from S {
  example {
    field x: Tensor[8] = decode(row)
    field label: Class = as_class(t)
  }
  batch 4
}
train R {
  data D
  model Net = M
  loss main = O(logits: Net(x), labels: label)
  optimizer opt = adamw(lr: 0.001)
  epochs 1
}`);
    assert(codes(r.mod).includes("AXS0401") || codes(r.mod).includes("AXS0602"), `got ${codes(r.mod).join(",")}`);
    return "3 != K=4 is caught before execution";
  });

  t("objectives", "semantic kinds catch softmax-then-cross-entropy", () => {
    const r = compile(`dim B
dim K = 4
objective O(p: Probs[B, K], labels: Class[B]) -> Scalar { return cross_entropy(p, labels) }`);
    assert(codes(r.mod).includes("AXS0409"), `expected AXS0409, got ${codes(r.mod).join(",")}`);
    return "Probs where Logits are required is reported";
  });

  t("data", "data field / model input contract mismatch", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 2] { linear(2) }
objective O(p: Tensor[B, 2], y: Class[B]) -> Scalar { return cross_entropy(p, y) }
source S = csv(path: "a.csv")
data D from S {
  example {
    field x: Tensor[8] = decode(row)
    field y: Class = as_class(t)
  }
  batch 4
}
train R {
  data D
  model Net = M
  loss main = O(p: Net(x), y: y)
  optimizer opt = adamw(lr: 0.001)
  epochs 1
}`);
    assert(codes(r.mod).includes("AXS0401") || codes(r.mod).includes("AXS0602"), `got ${codes(r.mod).join(",")}`);
    return "field [8] cannot feed model input [B, 16]";
  });

  t("data", "leakage: statistics fitted outside the train split", () => {
    const r = compile(`source S = csv(path: "a.csv")
data D from S {
  example { field x: Tensor[8] = decode(row) |> standardize(fit: val) }
  split { train: 0.8, val: 0.2 }
  batch 4
}`);
    assert(codes(r.mod).includes("AXS0620"), `expected AXS0620, got ${codes(r.mod).join(",")}`);
    return "fitting on val is rejected";
  });

  t("data", "stochastic augmentation in the eval pipeline is flagged", () => {
    const r = compile(`source S = image_folder(path: "x")
data D from S {
  example { field image: Image[3, 8, 8] = decode(file) }
  augment eval { image: random_flip(p: 0.5) }
  batch 4
}`);
    assert(codes(r.mod).includes("AXS0621"), `expected AXS0621, got ${codes(r.mod).join(",")}`);
    return "eval transforms should be deterministic";
  });

  t("data", "language-model shifting is explicit and type-checked", () => {
    const ex = EXAMPLES.find((e) => e.id === "gpt")!;
    const r = compile(ex.code, "gpt");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const data = r.mod.data[0];
    const inputs = data.fields.find((f) => f.name === "inputs")!;
    const targets = data.fields.find((f) => f.name === "targets")!;
    assert(inputs.pipeline.some((p) => p.op === "slice"), "inputs must be an explicit slice");
    assert(targets.pipeline.some((p) => p.op === "slice"), "targets must be an explicit slice");
    return "inputs = window[0:T], targets = window[1:T+1]";
  });

  // ---------------------------------------------------------------- lifecycle
  t("lifecycle", "phases, freeze/unfreeze and per-region learning rates", () => {
    const ex = EXAMPLES.find((e) => e.id === "transfer")!;
    const r = compile(ex.code, "transfer");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const plan = r.mod.plans[0];
    assert(plan.phases.length === 2, "two phases");
    assert(plan.phases[0].frozen[0] === "Net.encoder", "warmup freezes the encoder");
    const enc = plan.optimizers.find((o) => o.name === "enc_opt")!;
    const head = plan.optimizers.find((o) => o.name === "head_opt")!;
    assert(enc.params.length > 0 && head.params.length > 0, "both regions resolve to parameters");
    assert(!enc.params.some((p) => head.params.includes(p)), "regions must be disjoint");
    return `${enc.params.length} encoder tables, ${head.params.length} head tables`;
  });

  t("lifecycle", "unknown parameter region is an error, not a silent no-op", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 8]) -> Tensor[B, 8] { linear(8) }
objective O(p: Tensor[B, 8], y: Tensor[B, 8]) -> Scalar { return mse(p, y) }
source S = synthetic(features: 8)
data D from S { example { field x: Tensor[8] = decode(row) } batch 4 }
train R {
  data D
  model Net = M
  loss main = O(p: Net(x), y: x)
  optimizer opt = adamw(lr: 0.001) over Net.encoder
  epochs 1
}`);
    assert(codes(r.mod).includes("AXS0702") || codes(r.mod).includes("AXS0701"), `got ${codes(r.mod).join(",")}`);
    return "region 'Net.encoder' does not exist";
  });

  t("lifecycle", "alternating updates with two optimizers", () => {
    const ex = EXAMPLES.find((e) => e.id === "gan")!;
    const r = compile(ex.code, "gan");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const ph = r.mod.plans[0].phases[0];
    assert(ph.updates.length === 2, "two updates");
    assert(ph.updates[0].times === 2 && ph.updates[1].times === 1, "2:1 ratio recorded");
    return "discriminator ×2, generator ×1";
  });

  t("lifecycle", "simple training derives a single default phase", () => {
    const ex = EXAMPLES.find((e) => e.id === "mlp")!;
    const r = compile(ex.code, "mlp");
    const plan = r.mod.plans[0];
    assert(plan.phases.length === 1 && plan.phases[0].epochs === 3, "one derived phase of 3 epochs");
    assert(plan.phases[0].updates.length === 1, "the update is derived from the declared loss");
    return "no boilerplate for the common case";
  });

  const TWO_MODEL_PLAN = `dim B
model A(x: Tensor[B, 4]) -> Tensor[B, 2] { linear(2) }
model Bm(x: Tensor[B, 4]) -> Tensor[B, 2] { linear(2) }
objective L(y: Tensor[B, 2]) -> Scalar { return mean(y * y) }
source S = synthetic(features: 4)
data Dd from S { example { field x: Tensor[4] = decode(row) |> to_float } batch 4 }
train T {
  data Dd
  model a = A
  model b = Bm
  loss la = L(y: a(x))
  loss lb = L(y: b(x))
  optimizer opt = adamw(lr: 1e-2)
  epochs 1
}`;

  t("lifecycle", "a region-less optimizer covers every model in the plan, not just the first (F-022)", () => {
    const r = compile(TWO_MODEL_PLAN, "two-models");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const opt = r.mod.plans[0].optimizers[0];
    assert(opt.region === "*" && opt.params.length === 4, `region '${opt.region}' covers ${opt.params.length} tables`);
    const rep = runProgram(r.mod, { maxSteps: 2 });
    assert(rep.gradCoverage.every((g) => g.updated), rep.gradCoverage.map((g) => `${g.param}: ${g.reason ?? "updated"}`).join("; "));
    const py = emitTorch(r.mod);
    const line = py.split("\n").find((l) => l.includes("opt_params = ")) ?? "";
    assert(/\[a\.p\[k\] for k in \[[^\]]*\]\] \+ \[b\.p\[k\] for k in \[/.test(line), `emitted optimizer must list parameters grouped by owning model alias: ${line}`);
    return "region '*': 4 tables, all updated; emitted list spans both models";
  });

  t("lifecycle", "plan-level events count run steps, phase events count phase steps; `until` ends a phase (H-001, H-008)", () => {
    const src = TWO_MODEL_PLAN.replace(
      "  epochs 1\n}",
      "  phase p1 { steps 3 ; update la with opt ; every 2 steps { validate } }\n  phase p2 { epochs 5 ; until la < 100 ; update lb with opt }\n  every 4 steps { checkpoint }\n}"
    );
    const r = compile(src, "events");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const rep = runProgram(r.mod, { maxSteps: 40 });
    const [p1, p2] = rep.phases;
    assert(p1.steps === 3 && p1.stoppedBy === "steps", JSON.stringify(p1));
    assert(p2.steps === 1 && p2.stoppedBy === "until", JSON.stringify(p2));
    const ev = rep.events.map((e) => `${e.action}@${e.step}`).join(" ");
    assert(ev === "validate@1 checkpoint@3", ev);
    const py = emitTorch(r.mod);
    assert(py.includes("if step % 4 == 0:") && py.includes("if phase_step % 2 == 0:"), "emitter uses the same two counters");
    return `phases ${rep.phases.map((p) => `${p.name}:${p.steps}/${p.stoppedBy}`).join(" ")}; ${ev}`;
  });

  t("data", "a field may be built from the fields declared before it, whatever its name (F-021)", () => {
    const src = `dim T = 4
source C = text_file(path: "x.txt")
data D from C {
  example {
    field ids: Tokens[T + 1] = tokenize(text, vocab: 16) |> pad_to(T + 1)
    field inputs: Tokens[T] = ids[0 : T]
    field later: Tokens[T] = missing[0 : T]
  }
  batch 2
}`;
    const r = compile(src, "fields");
    const unknown = r.mod.diags.filter((d) => d.code === "AXS0204");
    assert(unknown.length === 1 && unknown[0].message.includes("'missing'"), `AXS0204: ${unknown.map((d) => d.message).join("; ")}`);
    const inputs = r.mod.data[0].fields.find((f) => f.name === "inputs")!;
    assert(inputs.pipeline[0].op === "field.ids", `pipeline: ${inputs.pipeline.map((p) => p.op).join(" |> ")}`);
    return "`ids` resolves to field.ids; only the genuinely unknown name warns";
  });

  t("data", "leakage is reported at the fitting operation, and unused model inputs are named (F-019, F-020)", () => {
    const src = `dim B
source C = csv(path: "a.csv")
data D from C {
  example {
    field a: Tensor[2] = select(column: "a") |> to_float
    field b: Tensor[2] = select(column: "b") |> to_float |> standardize(fit: all)
  }
  batch 2
}
model M(a: Tensor[B, 2], b: Tensor[B, 2]) -> Tensor[B, 1] { return a |> linear(1) }`;
    const r = compile(src, "leak");
    const leak = r.mod.diags.find((d) => d.code === "AXS0620")!;
    assert(leak && leak.loc.line === 6, `AXS0620 at line ${leak?.loc.line}`);
    const unused = r.mod.diags.find((d) => d.code === "AXS0305")!;
    assert(unused && unused.severity === "warning" && unused.message.includes("'b'"), `AXS0305: ${unused?.message}`);
    return `AXS0620 @${leak.loc.line}; ${unused.message}`;
  });

  // ---------------------------------------------------------------- recurrence
  t("recurrence", "scan carries state and shares parameters across steps", () => {
    const ex = EXAMPLES.find((e) => e.id === "recurrence")!;
    const r = compile(ex.code, "recurrence");
    assert(r.ok, r.errors.map((e) => e.message).join("; "));
    const scan = allNodes(r.mod).find((n) => n.op === "scan")!;
    assert(scan.regions!.length === 1, "one step region");
    const stepParams = r.mod.params.filter((p) => p.owner.includes("RNNTagger"));
    assert(stepParams.length > 0, "the step body owns parameters");
    return "one step region, one parameter set, T iterations at runtime";
  });

  t("recurrence", "carried state shape is a contract", () => {
    const r = compile(`dim B
dim T
dim F = 4
dim H = 8
model M(x: Tensor[B, T, F]) -> Tensor[B, T, H] {
  let (outs, last) = scan over x axis: 1 carry h: Tensor[B, H] init: zeros {
    yield concat(step, h, axis: -1) |> linear(4)
  }
  return outs
}`);
    assert(codes(r.mod).includes("AXS0401") || codes(r.mod).includes("AXS0402"), `got ${codes(r.mod).join(",")}`);
    return "yielding [B, 4] for a [B, H] carry is rejected";
  });

  // ---------------------------------------------------------------- custom ops
  t("custom", "custom op without shape semantics degrades knowledge explicitly", () => {
    const r = compile(`dim B
dim T
dim D = 8
custom op mystery(x: Tensor[B, T, D]) {
  effects: pure
  shape: unknown
}
model M(x: Tensor[B, T, D]) -> Tensor[B, T, D] { return mystery(x) }`);
    assert(codes(r.mod).includes("AXS0903"), `expected AXS0903, got ${codes(r.mod).join(",")}`);
    return "downstream shapes are reported as unknown, not guessed";
  });

  t("custom", "custom op contracts reach the backend", () => {
    const ex = EXAMPLES.find((e) => e.id === "custom-op")!;
    const r = compile(ex.code, "custom");
    const code = emitTorch(r.mod);
    assert(code.includes("flash_attn_func"), "torch implementation should be emitted");
    return "declared torch implementation is used verbatim";
  });

  // ---------------------------------------------------------------- IR & backend
  t("ir", "IR printing is stable across compilations", () => {
    const ex = EXAMPLES.find((e) => e.id === "inception")!;
    const a = printIR(compile(ex.code, "x").mod);
    const b = printIR(compile(ex.code, "x").mod);
    assert(a === b, "two compilations produced different IR");
    return `${a.split("\n").length} IR lines, byte-identical`;
  });

  t("ir", "inspection agrees with lowering", () => {
    const ex = EXAMPLES.find((e) => e.id === "inception")!;
    const r = compile(ex.code, "x");
    const rep = inspectModule(r.mod);
    const irMerge = allNodes(r.mod).filter((n) => n.op === "concat").length;
    const inspectMerge = rep.graphs.flatMap((g) => g.lines).filter((l) => l.kind === "merge").length;
    assert(irMerge === inspectMerge, `IR has ${irMerge} concat nodes, inspect shows ${inspectMerge}`);
    return "inspect is generated from the executed IR";
  });

  t("backend", "PyTorch lowering contains the expected structure", () => {
    const ex = EXAMPLES.find((e) => e.id === "inception")!;
    const code = emitTorch(compile(ex.code, "x").mod);
    assert(code.includes("class MultiScale(nn.Module)"), "module class");
    assert(code.includes("torch.cat"), "explicit merge");
    assert(code.includes("F.conv2d"), "conv lowering");
    assert(code.includes("nn.ParameterDict"), "parameter identity is explicit");
    return `${code.split("\n").length} lines of PyTorch`;
  });

  t("backend", "the emitted plan accepts host-owned model instances (H-010)", () => {
    const src = `dim B
model Policy(obs: Tensor[B, 3]) -> Tensor[B, 2] {
  return obs |> linear(2)
}
objective Fit(mu: Tensor[B, 2], action: Tensor[B, 2]) -> Scalar {
  return mse(mu, action)
}
source Rollouts = tensor_store(path: "rollouts")
data Experience from Rollouts {
  example {
    field obs: Tensor[3] = decode(obs) |> to_float
    field action: Tensor[2] = decode(action) |> to_float
  }
  batch 32
}
train Agent {
  data Experience
  model net = Policy
  loss fit = Fit(mu: net(obs), action: action)
  optimizer opt = adam(lr: 1e-3)
  phase learn {
    steps 10
    update fit with opt
  }
}`;
    const r = compile(src, "h010");
    assert(r.ok, r.errors.map((d) => d.message).join("; "));
    const code = emitTorch(r.mod);
    assert(code.includes("def train_Agent(loader, val_loader=None, models=None, **dims):"), "models parameter");
    assert(code.includes('net = models.get("net") or Policy(**dims)'), "host instance is used when supplied");
    return "the acting network and the trained network can be one object";
  });

  t("backend", "symbolic extents lower to Python expressions, not dictionary keys (H-003)", () => {
    const src = `dim B
dim K
dim H = 4 * K
dim D = 8
model M(x: Tensor[B, H]) -> Tensor[B, D] {
  return x |> linear(D)
}
model N(x: Tensor[B, H]) -> Tensor[B, H] {
  return x / D
}`;
    const r = compile(src, "h003");
    assert(r.ok, r.errors.map((d) => d.message).join("; "));
    const code = emitTorch(r.mod);
    const block = code.slice(code.indexOf("STATIC_DIMS = {"), code.indexOf("}", code.indexOf("STATIC_DIMS = {")));
    assert(!block.includes("K"), "derived dim must not be emitted as bare Python names in STATIC_DIMS");
    assert(!/dims\["[^"\]]*[*+/ ][^"\]]*"\]/.test(code), "no compound dictionary keys");
    assert(code.includes('d["H"] = 4 * d["K"]'), "derived dims resolve after runtime bindings");
    assert(/dims\["[HK]"\]/.test(code), "runtime-bound extents are looked up by name");
    return "STATIC_DIMS holds constants; resolve_dims computes derived dims";
  });

  t("backend", "runtime dims are bound from the inputs; reshape, slice and masks never lower to -1 (H-005)", () => {
    const src = `dim B
dim T
dim S
dim D = 16
dim Heads = 4
model M(x: Tensor[B, T, D], kv: Tensor[B, S, D], pad: Mask[B, 1, S], cls: Tokens[B]) -> Tensor[B, T, D] {
  let h = transpose(reshape(x, [B, T, Heads, D / Heads]), 1, 2)
  let prev = x[:, 0:T-1, :]
  let cm = causal_mask(T)
  let oh = one_hot(cls, classes: T)
  return attention(query: x, key: kv, value: kv, mask: pad, heads: Heads, causal: true)
}`;
    const r = compile(src, "h005");
    assert(r.ok, r.errors.map((d) => d.message).join("; "));
    const code = emitTorch(r.mod);
    assert(code.includes('dims = resolve_dims({**self.dims, "B": x.shape[0], "T": x.shape[1], "S": kv.shape[1]})'), "forward must bind B, T, S from the input shapes");
    assert(code.includes('.reshape(dims["B"], dims["T"], 4, 4)'), "symbolic reshape uses the bound dims");
    assert(code.includes('[0:dims["B"], 0:(dims["T"] - 1), 0:16]'), "symbolic slice uses the bound dims");
    assert(code.includes('torch.ones(dims["T"], dims["T"], dtype=torch.bool)'), "causal_mask sized from T, not 1");
    assert(code.includes('F.one_hot(vv') && code.includes('.long(), dims["T"])'), "one_hot sized from T, not 2");
    assert(code.includes("m = ~vv") && code.includes("[:, None]") && code.includes("attn_mask=m"), "the mask port reaches SDPA as an inverted, head-broadcast attn_mask");
    assert(code.includes("torch.tril(") && !code.includes("is_causal=True"), "causal + mask combine into one attn_mask (SDPA forbids both)");
    assert(!/reshape\(-1|:-1\]|\(-1,/.test(code), "no -1 placeholders remain");
    return "dims bound in forward; reshape/slice/causal_mask/one_hot/attention lowered symbolically";
  });

  t("backend", "a layout error the shape rules cannot see is caught by the metamorphic pair (E-007)", () => {
    const pair = METAMORPHIC.find((m) => m.id === "mha-manual-vs-catalog")!;
    const broken = pair.a.replace("return reshape(transpose(x, 1, 2), [B, T, D])", "return reshape(x, [B, T, D])");
    assert(compile(broken, "e007").ok, "merge_heads without the transpose has the right element count and must type-check");
    let threw = "";
    try {
      assertEquivalent(broken, pair.b, { ir: false, run: pair.run });
    } catch (e) {
      threw = (e as Error).message;
    }
    assert(threw.includes("differ") || threw.includes("output"), `the numeric comparison must reject the wrong head merge; got '${threw || "equal"}'`);
    return `type-checks, but outputs differ from the catalog attention: ${threw.slice(0, 80)}`;
  });

  t("backend", "shared parameters lower to one tensor", () => {
    const ex = EXAMPLES.find((e) => e.id === "siamese")!;
    const code = emitTorch(compile(ex.code, "siamese").mod);
    const decls = code.split("\n").filter((l) => l.includes("Siamese_encoder") && l.includes("nn.Parameter"));
    assert(decls.length === 6, `expected 6 parameter declarations for the tied encoder, got ${decls.length}`);
    assert(decls.every((d) => d.includes("shared by 2 applications")), "sharing should be annotated");
    return "one tensor per table, used twice";
  });

  // ---------------------------------------------------------------- execution
  t("execution", "forward execution produces the declared shapes", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 4] { linear(8) ; gelu ; linear(4) }`);
    const rep = runProgram(r.mod, { maxSteps: 0 });
    assert(rep.forward[0].output.includes("4"), `output was ${rep.forward[0].output}`);
    assert(rep.errors.length === 0, rep.errors.join("; "));
    return `${rep.forward[0].inputs[0]} -> ${rep.forward[0].output}`;
  });

  t("execution", "train/eval semantics: dropout is stochastic only in training", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 16]) -> Tensor[B, 16] { dropout(0.5) }`);
    const rt = new Runtime(r.mod, {});
    rt.allocate();
    const g = r.mod.graphs[0];
    const input = X.full([2, 16], 1);
    X.beginTape();
    rt.training = false;
    rt.env = new Map();
    const a = rt.evalGraph(g, [input]).map((o) => [...o.data]);
    rt.env = new Map();
    const b = rt.evalGraph(g, [input]).map((o) => [...o.data]);
    assert(JSON.stringify(a) === JSON.stringify(b), "eval must be deterministic");
    rt.training = true;
    rt.env = new Map();
    const c = rt.evalGraph(g, [input]).map((o) => [...o.data]);
    assert(JSON.stringify(a) !== JSON.stringify(c), "training must sample the dropout mask");
    return "identity in eval, sampled in train";
  });

  t("execution", "gradients reach every trainable parameter and the loss moves", () => {
    const ex = EXAMPLES.find((e) => e.id === "mlp")!;
    const r = compile(ex.code, "mlp");
    const rep = runProgram(r.mod, { maxSteps: 4, dims: { B: 2 } });
    assert(rep.losses.length > 0, "training steps should be recorded");
    const finite = rep.losses.every((l) => Object.values(l.values).every((v) => Number.isFinite(v)));
    assert(finite, "losses must be finite");
    const missing = rep.gradCoverage.filter((g) => !g.updated);
    assert(missing.length === 0, `parameters without updates: ${missing.map((m) => `${m.param} (${m.reason})`).join(", ")}`);
    return `${rep.losses.length} steps, final ${JSON.stringify(rep.finalMetrics)}`;
  });

  t("execution", "parallel branches execute from the same input at runtime", () => {
    const r = compile(`dim B
model M(x: Tensor[B, 2, 4, 4]) -> Tensor[B, 4, 4, 4] {
  split merge concat(1) {
    conv2d(2, kernel: 1)
    conv2d(2, kernel: 3, pad: 1)
  }
}`);
    const rep = runProgram(r.mod, { maxSteps: 0, dims: { B: 1 } });
    assert(rep.errors.length === 0, rep.errors.join("; "));
    assert(rep.forward[0].output.includes("4, 4, 4"), `got ${rep.forward[0].output}`);
    return `runtime output ${rep.forward[0].output}`;
  });

  t("execution", "runtime recurrence unrolls over the sequence axis", () => {
    const ex = EXAMPLES.find((e) => e.id === "recurrence")!;
    const r = compile(ex.code, "recurrence");
    const rep = runProgram(r.mod, { maxSteps: 0, dims: { B: 2, T: 5 } });
    assert(rep.errors.length === 0, rep.errors.join("; "));
    const tagger = rep.forward.find((f) => f.model === "RNNTagger")!;
    assert(tagger.output.includes("5"), `expected a length-5 sequence, got ${tagger.output}`);
    return `scan produced ${tagger.output}`;
  });

  // ---------------------------------------------------------------- checkpoint
  t("checkpoint", "checkpoint coverage includes state and optimizer slots", () => {
    const ex = EXAMPLES.find((e) => e.id === "transfer")!;
    const rep = inspectModule(compile(ex.code, "transfer").mod);
    const kinds = Object.fromEntries(rep.checkpoint.map((c) => [c.kind, c.items]));
    assert(kinds["parameters"] > 0, "parameters must be checkpointed");
    assert(kinds["persistent state"] > 0, "batchnorm statistics must be checkpointed");
    assert(kinds["optimizer state"] > 0, "optimizer moments must be checkpointed");
    return JSON.stringify(kinds);
  });

  // ---------------------------------------------------------------- hardening corpus
  for (const ch of CHALLENGES) for (const c of expand(ch)) t(`challenge:${ch.id}`, c.name, c.fn);

  for (const m of METAMORPHIC)
    t("metamorphic", `${m.id} (${m.section})`, () => assertEquivalent(m.a, m.b, { run: m.run, ignore: m.ignore, ir: m.ir }));

  for (const p of PROPERTIES) t("property", p.name, () => checkProperty(p));

  // §5.3: every catalog op on a fully symbolic input
  t("op-shapes", "every catalog op has a symbolic shape case", () => {
    const missing = uncoveredOps();
    assert(missing.length === 0, `ops without a symbolic case: ${missing.join(", ")}`);
    return `${OP_SHAPE_CASES.length} cases cover ${new Set(OP_SHAPE_CASES.map((c) => c.op)).size} ops`;
  });
  for (const c of OP_SHAPE_CASES)
    t("op-shapes", `${c.op}: ${c.call} on ${c.x}${c.y ? ` × ${c.y}` : ""} → ${c.out}`, () => {
      const problems = checkOpCase(c);
      assert(problems.length === 0, problems.join("; "));
      return c.out;
    });

  t("leakage", "no execution machinery in any bundled example (§33)", () => {
    const hits = EXAMPLES.flatMap((e) => leakageHits(e.code).map((h) => `${e.id}: ${h}`));
    assert(hits.length === 0, hits.join(", "));
    return `${EXAMPLES.length} examples clean`;
  });

  return results;
}

export function testSummary(rs: TestResult[]) {
  const passed = rs.filter((r) => r.ok).length;
  return { total: rs.length, passed, failed: rs.length - passed };
}

export function isTensorT(t: unknown): boolean {
  return Boolean(t && typeof t === "object" && isTensor(t as never));
}
