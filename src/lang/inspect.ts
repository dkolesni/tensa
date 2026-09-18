/**
 * TENSA — `inspect`: what did the compiler actually build?
 *
 * The report is generated from the same IR the backend executes, so inspection
 * can never disagree with lowering.
 */
import { capabilityOf } from "./catalog";
import { DimExpr, asConst, evalDim, show } from "./dims";
import { IRModule, IRNode } from "./ir";
import { Effect, isTensor, showType } from "./types";

export interface GraphLine {
  depth: number;
  kind: "node" | "region" | "merge" | "header";
  text: string;
  shape: string;
  detail: string;
  effects: Effect[];
  params: string[];
  shared?: string;
  op: string;
}

export interface InspectReport {
  module: string;
  graphs: { name: string; kind: string; signature: string; lines: GraphLine[] }[];
  params: {
    id: string;
    owner: string;
    shape: string;
    count: number | null;
    trainable: boolean;
    kind: string;
    applications: number;
  }[];
  totalParams: number | null;
  trainableParams: number | null;
  sharedGroups: { stage: string; applications: number; tables: number; savedParams: number | null }[];
  states: { id: string; shape: string; update: string; category: string; checkpointed: boolean }[];
  effects: { effect: Effect; sites: string[] }[];
  dims: { name: string; value: string; kind: string }[];
  constraints: { text: string; status: string; origin: string }[];
  compatibility: string[];
  capabilities: { op: string; forward: boolean; grad: boolean; note?: string }[];
  checkpoint: { kind: string; items: number; detail: string }[];
  lifecycle: string[];
  warnings: string[];
}

function dimVal(d: DimExpr, env: Map<string, number>): number | null {
  const c = asConst(d);
  if (c !== null) return c;
  return evalDim(d, env);
}

export function inspectModule(mod: IRModule): InspectReport {
  const env = new Map<string, number>();
  for (const d of mod.dims) if (d.value) env.set(d.name, evalDim(d.value, env) ?? 0);

  const graphs: InspectReport["graphs"] = [];
  const effectSites = new Map<Effect, string[]>();

  const emitNodes = (nodes: IRNode[], depth: number, lines: GraphLine[]) => {
    for (const n of nodes) {
      const ty = mod.values.get(n.outputs[0])?.type;
      const shape = ty ? showType(ty) : "";
      const attrs = Object.entries(n.attrs)
        .filter(([k]) => !k.startsWith("__") && k !== "step_value" && k !== "carry_values")
        .map(([k, v]) => `${k}=${typeof v === "object" && v && "terms" in v ? show(v as DimExpr) : Array.isArray(v) ? `[${(v as unknown[]).map((x) => (typeof x === "object" && x && "terms" in (x as object) ? show(x as DimExpr) : String(x))).join(",")}]` : String(v)}`)
        .join(" ");
      // container nodes carry the union of their body's effects (F-016); the
      // site list names the inner node that actually has the effect
      if (!n.regions)
        for (const e of n.effects)
          if (e !== "pure") effectSites.set(e, [...(effectSites.get(e) ?? []), `${n.stage} › ${n.op}`]);
      lines.push({
        depth,
        kind: n.op === "concat" && n.note === "explicit branch merge" ? "merge" : "node",
        text: n.op,
        shape,
        detail: attrs,
        effects: n.effects.filter((e) => e !== "pure"),
        params: n.params,
        shared: n.sharedStage,
        op: n.op,
      });
      if (n.regions)
        for (const r of n.regions) {
          lines.push({
            depth: depth + 1,
            kind: "region",
            text: r.label,
            shape: "",
            detail: n.op === "parallel" ? "receives the split input" : "",
            effects: [],
            params: [],
            op: n.op,
          });
          emitNodes(r.nodes, depth + 2, lines);
        }
    }
  };

  for (const g of [...mod.graphs, ...mod.objectives.map((o) => o.graph)]) {
    const lines: GraphLine[] = [];
    emitNodes(g.nodes, 0, lines);
    graphs.push({
      name: g.name,
      kind: g.kind,
      signature: `(${g.inputs.map((i) => `${i.name ?? i.id}: ${showType(i.type)}`).join(", ")}) -> ${
        g.resultType ? showType(g.resultType) : "?"
      }`,
      lines,
    });
  }

  let total = 0;
  let trainable = 0;
  let unknown = false;
  const params = mod.params.map((p) => {
    const nums = p.shape.map((d) => dimVal(d, env));
    const count = nums.every((x) => x !== null) ? nums.reduce((a, b) => a! * b!, 1)! : null;
    if (count === null) unknown = true;
    else {
      total += count;
      if (p.trainable) trainable += count;
    }
    return {
      id: p.id,
      owner: p.owner,
      shape: `[${p.shape.map(show).join(", ")}]`,
      count,
      trainable: p.trainable,
      kind: p.kind,
      applications: p.applications.length,
    };
  });

  const sharedMap = new Map<string, { applications: number; tables: number; saved: number }>();
  for (const p of mod.params) {
    if (p.applications.length < 2) continue;
    const stage = p.owner.split("/").slice(0, -1).join("/") || p.owner;
    const nums = p.shape.map((d) => dimVal(d, env));
    const count = nums.every((x) => x !== null) ? nums.reduce((a, b) => a! * b!, 1)! : 0;
    const cur = sharedMap.get(stage) ?? { applications: p.applications.length, tables: 0, saved: 0 };
    cur.tables++;
    cur.saved += count * (p.applications.length - 1);
    cur.applications = Math.max(cur.applications, p.applications.length);
    sharedMap.set(stage, cur);
  }

  const capOps = new Set<string>();
  const walkCaps = (nodes: IRNode[]) => {
    for (const n of nodes) {
      capOps.add(n.op);
      if (n.regions) for (const r of n.regions) walkCaps(r.nodes);
    }
  };
  for (const g of mod.graphs) walkCaps(g.nodes);
  for (const o of mod.objectives) walkCaps(o.graph.nodes);

  const compat: string[] = [];
  for (const plan of mod.plans) {
    const data = mod.data.find((d) => d.name === plan.data);
    if (data) {
      compat.push(`data ${data.name}: ${data.fields.length} fields, batch ${data.batch}, splits ${data.splits.map((s) => `${s.name} ${Math.round(s.frac * 100)}%`).join(" / ") || "none declared"}`);
      for (const f of data.fields)
        compat.push(
          `  field ${f.name}: ${showType(f.type)} — ${f.pipeline.length} pipeline step(s)${
            f.stochasticInTrain ? ", stochastic in train" : ""
          }`
        );
    }
    for (const l of plan.losses)
      compat.push(
        `loss ${l.name} = ${l.objective}(${l.bindings.map((b) => `${b.port} ← ${b.text}${b.type ? ` : ${showType(b.type)}` : ""}`).join(", ")})`
      );
    for (const o of plan.optimizers)
      compat.push(`optimizer ${o.name} (${o.kind}, lr ${o.lr}) covers ${o.params.length} parameter tables in region '${o.region}'`);
  }

  const lifecycle: string[] = [];
  for (const plan of mod.plans) {
    lifecycle.push(`plan ${plan.name}: ${plan.phases.length} phase(s)`);
    for (const ph of plan.phases) {
      const dur = ph.steps !== null ? `${ph.steps} steps` : `${ph.epochs} epochs`;
      lifecycle.push(
        `  phase ${ph.name} — ${dur}${ph.until ? ` until ${ph.until.metric} ${ph.until.cmp} ${ph.until.value}` : ""}${
          ph.schedule ? `, schedule ${ph.schedule}` : ""
        }`
      );
      if (ph.frozen.length) lifecycle.push(`    frozen: ${ph.frozen.join(", ")}`);
      if (ph.unfrozen.length) lifecycle.push(`    unfrozen: ${ph.unfrozen.join(", ")}`);
      for (const lo of ph.lrOverrides) lifecycle.push(`    lr override ${lo.region} = ${lo.lr}`);
      for (const u of ph.updates)
        lifecycle.push(`    update ${u.loss} ×${u.times} with ${u.optimizers.join(" + ") || "(no optimizer)"}`);
      for (const ev of ph.events)
        lifecycle.push(`    every ${ev.every} ${ev.unit}: ${ev.actions.map((a) => a.kind).join(", ")}`);
    }
    for (const ev of plan.events)
      lifecycle.push(`  every ${ev.every} ${ev.unit}: ${ev.actions.map((a) => `${a.kind}(${a.args})`).join(", ")}`);
    if (Object.keys(plan.settings).length)
      lifecycle.push(`  execution policy: ${Object.entries(plan.settings).map(([k, v]) => `${k}=${v}`).join(", ")}`);
  }

  const optStateSlots = mod.plans.flatMap((p) =>
    p.optimizers.flatMap((o) => o.stateSlots.map((s) => `${o.name}.${s}`))
  );

  return {
    module: mod.name,
    graphs,
    params,
    totalParams: unknown ? null : total,
    trainableParams: unknown ? null : trainable,
    sharedGroups: [...sharedMap.entries()].map(([stage, v]) => ({
      stage,
      applications: v.applications,
      tables: v.tables,
      savedParams: v.saved,
    })),
    states: mod.states.map((s) => ({
      id: s.id,
      shape: `[${s.shape.map(show).join(", ")}]`,
      update: s.update,
      category: s.category,
      checkpointed: s.checkpointed,
    })),
    effects: [...effectSites.entries()].map(([effect, sites]) => ({ effect, sites: [...new Set(sites)] })),
    dims: mod.dims.map((d) => ({
      name: d.name,
      value: d.value ? show(d.value) : "bound at runtime",
      kind: d.value ? "static" : "runtime",
    })),
    constraints: mod.constraints
      .filter((c, i, arr) => arr.findIndex((x) => show(x.lhs) === show(c.lhs) && show(x.rhs) === show(c.rhs) && x.origin === c.origin) === i)
      .map((c) => ({ text: `${show(c.lhs)} ${c.rel ? "≤" : "="} ${show(c.rhs)}`, status: c.status, origin: c.origin })),
    compatibility: compat,
    capabilities: [...capOps]
      .map((op) => ({ op, ...capabilityOf(op, "reference") }))
      .filter((c) => !c.forward || !c.grad || c.note),
    checkpoint: [
      { kind: "parameters", items: mod.params.length, detail: `${mod.params.filter((p) => p.trainable).length} trainable, ${mod.params.filter((p) => !p.trainable).length} frozen` },
      { kind: "persistent state", items: mod.states.length, detail: mod.states.map((s) => s.category).filter((v, i, a) => a.indexOf(v) === i).join(", ") || "none" },
      { kind: "optimizer state", items: optStateSlots.length, detail: optStateSlots.join(", ") || "none" },
      { kind: "lifecycle position", items: mod.plans.length, detail: "phase index, step, epoch, schedule state" },
    ],
    lifecycle,
    warnings: mod.diags.filter((d) => d.severity === "warning").map((d) => `${d.code}: ${d.message}`),
  };
}

export function renderInspect(r: InspectReport): string {
  const L: string[] = [];
  L.push(`inspect ${r.module}`);
  L.push("");
  for (const g of r.graphs) {
    L.push(`${g.kind} ${g.name}${g.signature}`);
    for (const l of g.lines) {
      const ind = "  ".repeat(l.depth + 1);
      if (l.kind === "region") L.push(`${ind}┌ ${l.text}${l.detail ? `  (${l.detail})` : ""}`);
      else
        L.push(
          `${ind}${l.kind === "merge" ? "▼ " : ""}${l.text}${l.detail ? `[${l.detail}]` : ""}${
            l.shape ? ` : ${l.shape}` : ""
          }${l.effects.length ? `  {${l.effects.join(",")}}` : ""}${l.shared ? `  (shared ${l.shared})` : ""}`
        );
    }
    L.push("");
  }
  L.push(`parameters: ${r.params.length} tables, ${r.totalParams ?? "?"} values (${r.trainableParams ?? "?"} trainable)`);
  for (const p of r.params)
    L.push(`  ${p.id} ${p.shape} = ${p.count ?? "?"}  ${p.trainable ? "trainable" : "frozen"}${p.applications > 1 ? `  shared×${p.applications}` : ""}`);
  if (r.sharedGroups.length) {
    L.push("");
    L.push("shared stages:");
    for (const s of r.sharedGroups)
      L.push(`  ${s.stage}: ${s.applications} applications over ${s.tables} tables (${s.savedParams} values not duplicated)`);
  }
  if (r.states.length) {
    L.push("");
    L.push("persistent state:");
    for (const s of r.states) L.push(`  ${s.id} ${s.shape} update=${s.update} (${s.category})`);
  }
  if (r.effects.length) {
    L.push("");
    L.push("effects:");
    for (const e of r.effects) L.push(`  ${e.effect}: ${e.sites.slice(0, 6).join(", ")}${e.sites.length > 6 ? " …" : ""}`);
  }
  if (r.constraints.length) {
    L.push("");
    L.push("dimension constraints:");
    for (const c of r.constraints) L.push(`  ${c.text} [${c.status}] — ${c.origin}`);
  }
  if (r.compatibility.length) {
    L.push("");
    L.push("data / model / objective:");
    for (const c of r.compatibility) L.push(`  ${c}`);
  }
  if (r.lifecycle.length) {
    L.push("");
    L.push("training lifecycle:");
    for (const c of r.lifecycle) L.push(`  ${c}`);
  }
  L.push("");
  L.push("checkpoint coverage:");
  for (const c of r.checkpoint) L.push(`  ${c.kind}: ${c.items} (${c.detail})`);
  if (r.capabilities.length) {
    L.push("");
    L.push("backend capabilities (reference):");
    for (const c of r.capabilities)
      L.push(`  ${c.op}: forward=${c.forward} grad=${c.grad}${c.note ? ` — ${c.note}` : ""}`);
  }
  if (r.warnings.length) {
    L.push("");
    L.push("warnings:");
    for (const w of r.warnings) L.push(`  ${w}`);
  }
  return L.join("\n");
}

export function isTensorType(t: unknown): boolean {
  return Boolean(t && typeof t === "object" && isTensor(t as never));
}
