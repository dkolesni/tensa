/**
 * TENSA — backend-neutral intermediate representation.
 *
 * Everything the compiler knows is in here: values and symbolic shapes, graph
 * edges, composite topology (residual / parallel / scan regions), parameters and
 * their identity, persistent state, effects, objectives, data contracts,
 * the training plan and execution policy.  Backends consume only this.
 */
import { DimExpr, show } from "./dims";
import { Diagnostic, Effect, Loc, ValueType, showType } from "./types";

export type IRAttr =
  | number
  | string
  | boolean
  | number[]
  | string[]
  | DimExpr
  | DimExpr[]
  | null
  | undefined;

export function isDimExpr(a: unknown): a is DimExpr {
  return typeof a === "object" && a !== null && Array.isArray((a as DimExpr).terms);
}

export interface IRValue {
  id: string;
  type: ValueType;
  name?: string;
}

export interface IRRegion {
  label: string;
  nodes: IRNode[];
  results: string[];
}

export interface IRNode {
  id: string;
  op: string;
  inputs: string[];
  attrs: Record<string, IRAttr>;
  outputs: string[];
  params: string[];
  states: string[];
  effects: Effect[];
  regions?: IRRegion[];
  stage: string;
  loc: Loc;
  note?: string;
  /** true when the node applies a stage instance whose params are shared */
  sharedStage?: string;
}

export interface IRGraph {
  name: string;
  kind: "model" | "block" | "fn" | "objective" | "data";
  inputs: IRValue[];
  outputs: string[];
  nodes: IRNode[];
  standalone: boolean;
  resultType?: ValueType;
  loc: Loc;
}

export interface IRParam {
  id: string;
  owner: string; // stage instance path that owns it
  role: string; // w, b, gamma, embed, explicit name...
  shape: DimExpr[];
  init: string;
  trainable: boolean;
  kind: "weight" | "bias" | "norm" | "embedding" | "explicit";
  applications: string[]; // stage paths where the parameter is used
}

export interface IRState {
  id: string;
  owner: string;
  role: string;
  shape: DimExpr[];
  init: string;
  update: string;
  checkpointed: boolean;
  category: "running-stat" | "ema" | "counter" | "algorithmic";
}

export interface IRObjective {
  name: string;
  graph: IRGraph;
  inputs: { name: string; type: ValueType }[];
}

export interface IRDataField {
  name: string;
  type: ValueType;
  pipeline: { op: string; args: string; effects: Effect[]; split: string }[];
  stochasticInTrain: boolean;
}

export interface IRData {
  name: string;
  source: { adapter: string; args: string };
  fields: IRDataField[];
  splits: { name: string; frac: number }[];
  batch: number;
  shuffle: boolean;
  fitted: { stat: string; field: string; split: string }[];
}

export interface IRUpdate {
  loss: string;
  optimizers: string[];
  times: number;
}

export interface IRPhase {
  name: string;
  epochs: number;
  steps: number | null;
  until: { metric: string; cmp: string; value: number } | null;
  frozen: string[];
  unfrozen: string[];
  lrOverrides: { region: string; lr: number }[];
  updates: IRUpdate[];
  events: IREvent[];
  schedule: string | null;
}

export interface IREvent {
  every: number;
  unit: "steps" | "epochs";
  actions: { kind: string; args: string }[];
}

export interface IROptimizer {
  name: string;
  kind: string;
  lr: number;
  args: Record<string, number>;
  region: string;
  params: string[];
  stateSlots: string[];
}

export interface IRLoss {
  name: string;
  objective: string;
  bindings: { port: string; kind: "model" | "field" | "expr"; text: string; type?: ValueType }[];
  weight: number;
}

export interface IRPlan {
  name: string;
  data: string | null;
  models: { alias: string; model: string }[];
  losses: IRLoss[];
  optimizers: IROptimizer[];
  tracks: { name: string; kind: string; region: string; args: string }[];
  phases: IRPhase[];
  events: IREvent[];
  settings: Record<string, string>;
}

export interface IRCustomOp {
  name: string;
  inputs: { name: string; type: ValueType }[];
  result: ValueType | null;
  effects: Effect[];
  backends: { target: string; code: string }[];
  shapeUnknown: boolean;
  differentiable: boolean;
}

export interface Constraint {
  lhs: DimExpr;
  rhs: DimExpr;
  /** equality by default; `<=` carries a slice/window bound (F-011) */
  rel?: "<=";
  status: "proved" | "assumed" | "failed";
  origin: string;
  loc: Loc;
}

export interface IRModule {
  name: string;
  dims: { name: string; value: DimExpr | null; runtime: boolean }[];
  graphs: IRGraph[];
  params: IRParam[];
  states: IRState[];
  objectives: IRObjective[];
  data: IRData[];
  plans: IRPlan[];
  customOps: IRCustomOp[];
  constraints: Constraint[];
  values: Map<string, IRValue>;
  diags: Diagnostic[];
}

// ------------------------------------------------------------------ printing

function attrStr(a: IRAttr): string {
  if (a === null || a === undefined) return "null";
  if (Array.isArray(a)) {
    const arr = a as unknown[];
    if (arr.length > 0 && typeof arr[0] === "object") return `[${(a as DimExpr[]).map(show).join(", ")}]`;
    return `[${arr.join(", ")}]`;
  }
  if (isDimExpr(a)) return show(a);
  if (typeof a === "string") return JSON.stringify(a);
  return String(a);
}

function nodeLine(n: IRNode, mod: IRModule, indent: string): string[] {
  const out: string[] = [];
  const res = n.outputs.map((o) => `%${o}`).join(", ") || "()";
  const attrs = Object.entries(n.attrs)
    .map(([k, v]) => `${k}=${attrStr(v)}`)
    .join(" ");
  const ins = n.inputs.map((i) => `%${i}`).join(", ");
  const ty = n.outputs
    .map((o) => {
      const v = mod.values.get(o);
      return v ? showType(v.type) : "?";
    })
    .join(", ");
  let line = `${indent}${res} = ${n.op}${attrs ? `[${attrs}]` : ""}(${ins}) : ${ty}`;
  const tags: string[] = [];
  if (n.stage) tags.push(`@${n.stage}`);
  if (n.params.length) tags.push(`params{${n.params.join(", ")}}`);
  if (n.states.length) tags.push(`state{${n.states.join(", ")}}`);
  const eff = n.effects.filter((e) => e !== "pure");
  if (eff.length) tags.push(`effects{${eff.join(",")}}`);
  if (n.sharedStage) tags.push(`shares:${n.sharedStage}`);
  if (tags.length) line += `   // ${tags.join(" ")}`;
  out.push(line);
  if (n.regions) {
    for (const r of n.regions) {
      out.push(`${indent}  region ${r.label} {`);
      for (const c of r.nodes) out.push(...nodeLine(c, mod, indent + "    "));
      out.push(`${indent}    yield ${r.results.map((x) => `%${x}`).join(", ")}`);
      out.push(`${indent}  }`);
    }
  }
  return out;
}

export function printGraph(g: IRGraph, mod: IRModule): string[] {
  const lines: string[] = [];
  const sig = g.inputs.map((i) => `%${i.id}: ${showType(i.type)}`).join(", ");
  lines.push(`graph ${g.kind} ${g.name}(${sig}) -> ${g.resultType ? showType(g.resultType) : "?"}`);
  for (const n of g.nodes) lines.push(...nodeLine(n, mod, "  "));
  lines.push(`  return ${g.outputs.map((o) => `%${o}`).join(", ")}`);
  return lines;
}

export function printIR(mod: IRModule): string {
  const L: string[] = [];
  L.push(`; TENSA backend-neutral IR — module ${mod.name}`);
  L.push("");
  if (mod.dims.length) {
    L.push("dims:");
    for (const d of mod.dims)
      L.push(`  dim ${d.name}${d.value ? ` = ${show(d.value)}` : ""}${d.runtime ? "   ; bound at runtime" : ""}`);
    L.push("");
  }
  for (const g of mod.graphs) {
    L.push(...printGraph(g, mod));
    L.push("");
  }
  for (const o of mod.objectives) {
    L.push(...printGraph(o.graph, mod));
    L.push("");
  }
  if (mod.params.length) {
    L.push(`parameters (${mod.params.length} tables):`);
    for (const p of mod.params) {
      const sh = `[${p.shape.map(show).join(", ")}]`;
      const shared = p.applications.length > 1 ? `  shared by ${p.applications.length} applications` : "";
      L.push(
        `  ${p.id} : ${sh} init=${p.init} ${p.trainable ? "trainable" : "frozen"} owner=${p.owner}${shared}`
      );
    }
    L.push("");
  }
  if (mod.states.length) {
    L.push(`persistent state (${mod.states.length}):`);
    for (const s of mod.states)
      L.push(
        `  ${s.id} : [${s.shape.map(show).join(", ")}] init=${s.init} update=${s.update} category=${s.category} checkpointed=${s.checkpointed}`
      );
    L.push("");
  }
  for (const d of mod.data) {
    L.push(`data ${d.name} from ${d.source.adapter}(${d.source.args})`);
    for (const f of d.fields) {
      L.push(`  field ${f.name} : ${showType(f.type)}`);
      for (const st of f.pipeline)
        L.push(`    ${st.op}(${st.args})${st.split !== "both" ? ` [${st.split}]` : ""}${st.effects.includes("stochastic") ? " ; stochastic" : ""}`);
    }
    L.push(`  splits: ${d.splits.map((s) => `${s.name}=${s.frac}`).join(", ")}   batch=${d.batch} shuffle=${d.shuffle}`);
    for (const f of d.fitted) L.push(`  fitted ${f.stat} on ${f.field} using split '${f.split}'`);
    L.push("");
  }
  for (const p of mod.plans) {
    L.push(`plan ${p.name}`);
    if (p.data) L.push(`  data ${p.data}`);
    for (const m of p.models) L.push(`  model ${m.alias} = ${m.model}`);
    for (const l of p.losses)
      L.push(
        `  loss ${l.name} = ${l.objective}(${l.bindings.map((b) => `${b.port}: ${b.text}`).join(", ")}) weight=${l.weight}`
      );
    for (const o of p.optimizers)
      L.push(`  optimizer ${o.name} = ${o.kind}(lr=${o.lr}) over ${o.region} covering ${o.params.length} tables`);
    for (const t of p.tracks) L.push(`  track ${t.name} = ${t.kind}(${t.region}${t.args ? ", " + t.args : ""})`);
    for (const ph of p.phases) {
      L.push(
        `  phase ${ph.name}: ${ph.steps !== null ? `${ph.steps} steps` : `${ph.epochs} epochs`}${
          ph.until ? ` until ${ph.until.metric} ${ph.until.cmp} ${ph.until.value}` : ""
        }`
      );
      if (ph.frozen.length) L.push(`    freeze ${ph.frozen.join(", ")}`);
      if (ph.unfrozen.length) L.push(`    unfreeze ${ph.unfrozen.join(", ")}`);
      for (const lo of ph.lrOverrides) L.push(`    lr ${lo.region} = ${lo.lr}`);
      for (const u of ph.updates)
        L.push(`    update ${u.loss} with [${u.optimizers.join(", ")}] x${u.times}`);
      for (const e of ph.events)
        L.push(`    every ${e.every} ${e.unit}: ${e.actions.map((a) => a.kind).join(", ")}`);
    }
    for (const e of p.events)
      L.push(`  every ${e.every} ${e.unit}: ${e.actions.map((a) => `${a.kind}(${a.args})`).join(", ")}`);
    const st = Object.entries(p.settings);
    if (st.length) L.push(`  execution policy: ${st.map(([k, v]) => `${k}=${v}`).join(", ")}`);
    L.push("");
  }
  if (mod.customOps.length) {
    L.push("custom operations:");
    for (const c of mod.customOps)
      L.push(
        `  ${c.name}(${c.inputs.map((i) => `${i.name}: ${showType(i.type)}`).join(", ")}) -> ${
          c.shapeUnknown ? "Tensor[?]" : c.result ? showType(c.result) : "?"
        }  effects{${c.effects.join(",")}} backends{${c.backends.map((b) => b.target).join(",")}} differentiable=${c.differentiable}`
      );
    L.push("");
  }
  if (mod.constraints.length) {
    L.push("dimension constraints:");
    for (const c of mod.constraints)
      L.push(`  ${show(c.lhs)} ${c.rel ? "≤" : "="} ${show(c.rhs)}   [${c.status}]  from ${c.origin}`);
    L.push("");
  }
  return L.join("\n");
}
