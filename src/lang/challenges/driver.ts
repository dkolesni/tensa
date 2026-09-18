/**
 * TENSA hardening — challenge driver.
 *
 * Expands a Challenge into individual test cases: one per populated expectation
 * layer, one per evil twin, plus the diagnostic-quality checks (§36) and the
 * backend-leakage audit (§33) that apply to every program in the corpus.
 */
import { compile } from "../analyze";
import { emitTorch } from "../emit_torch";
import { runProgram } from "../exec";
import { inspectModule, renderInspect } from "../inspect";
import { IRModule, IRNode } from "../ir";
import { Diagnostic, showType } from "../types";
import { Challenge, EvilTwin } from "./types";

export interface Case {
  name: string;
  fn: () => string;
}

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

export function allNodes(mod: IRModule): IRNode[] {
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

function fmt(ds: Diagnostic[]): string {
  return ds.map((d) => `${d.code}@${d.loc.line}:${d.loc.col} ${d.message}`).join(" | ") || "(none)";
}

// ------------------------------------------------------------------ §33 leakage audit

/** Tokens that describe execution machinery, not ML choices. */
export const LEAKAGE_TOKENS = [
  "cuda",
  ".to(device",
  "world_size",
  "all_reduce",
  "register_module",
  "register_buffer",
  "autocast",
  "state_dict",
  "requires_grad",
  "zero_grad",
  ".backward(",
  "optimizer.step",
  "nn.Module",
];

/** Strip `backend <target> { ... }` bodies — literal foreign source is the one place leakage is allowed. */
export function stripBackendBlocks(src: string): string {
  return src.replace(/backend\s+\w+\s*\{[^}]*\}/g, "backend <elided>");
}

export function leakageHits(src: string): string[] {
  const clean = stripBackendBlocks(src);
  return LEAKAGE_TOKENS.filter((t) => clean.includes(t));
}

// ------------------------------------------------------------------ §36 diagnostic quality

/** Backend jargon that must never reach a user-facing message (AXS0901 is exempt: it is *about* the backend). */
const JARGON = ["cuda", "state_dict", "requires_grad", "autocast", "zero_grad", "nn.Module", "im2col", "torch."];

const NEEDS_WHERE = /^AXS0(3|4|8)\d\d$/;
const SHAPE_CODES = new Set(["AXS0401", "AXS0404", "AXS0406"]);

export function diagnosticQuality(d: Diagnostic): string[] {
  const problems: string[] = [];
  if (d.code !== "AXS0901") for (const j of JARGON) if (d.message.includes(j)) problems.push(`jargon '${j}' in message`);
  if (NEEDS_WHERE.test(d.code) && !d.where) problems.push("missing architectural location (where)");
  if (SHAPE_CODES.has(d.code)) {
    const m = d.message;
    const hasBoth = /expected .+, found .+/.test(m) || /declares .+ but produces .+/.test(m) || /produces .+ but .+ carries .+/.test(m);
    if (!hasBoth) problems.push("shape diagnostic does not state both expected and inferred");
  }
  if (d.loc.line < 1 || d.loc.col < 1) problems.push("missing source location");
  return problems;
}

// ------------------------------------------------------------------ expansion

function twinCases(ch: Challenge, tw: EvilTwin): Case[] {
  return [
    {
      name: `twin '${tw.id}' (${tw.mutates}) fires ${tw.expectCodes.join("+")}`,
      fn: () => {
        const r = compile(tw.code, `${ch.id}/${tw.id}`);
        const got = r.mod.diags.map((d) => d.code);
        for (const c of tw.expectCodes) assert(got.includes(c), `expected ${c}; got ${fmt(r.mod.diags)}`);
        for (const c of tw.forbidCodes ?? []) assert(!got.includes(c), `forbidden ${c} fired; got ${fmt(r.mod.diags)}`);
        const primary = r.mod.diags.find((d) => d.code === tw.expectCodes[0])!;
        if (tw.line !== undefined) assert(primary.loc.line === tw.line, `expected line ${tw.line}, diagnostic points at ${primary.loc.line}`);
        const q = diagnosticQuality(primary);
        assert(q.length === 0, `diagnostic quality: ${q.join("; ")} — "${primary.message}"`);
        return `${primary.code} @${primary.loc.line}:${primary.loc.col}${primary.where ? ` in ${primary.where}` : ""}`;
      },
    },
  ];
}

export function expand(ch: Challenge): Case[] {
  const cases: Case[] = [];
  const e = ch.expect;
  const okExpected = e.ok !== false;

  // compiled lazily once; every case re-reads the same result
  let cached: ReturnType<typeof compile> | null = null;
  const compiled = () => (cached ??= compile(ch.code, ch.id));

  cases.push({
    name: okExpected ? "compiles without errors" : "is rejected",
    fn: () => {
      const r = compiled();
      if (okExpected) assert(r.errors.length === 0, fmt(r.errors));
      else assert(r.errors.length > 0, "expected errors, got none");
      if (e.warnCodes) {
        const unexpected = r.warnings.filter((w) => !e.warnCodes!.includes(w.code));
        assert(unexpected.length === 0, `unexpected warnings: ${fmt(unexpected)}`);
        for (const c of e.warnCodes) assert(r.warnings.some((w) => w.code === c), `expected warning ${c}`);
      }
      return `${r.mod.graphs.length} graphs, ${r.errors.length} errors, ${r.warnings.length} warnings`;
    },
  });

  cases.push({
    name: "no backend leakage in source (§33)",
    fn: () => {
      const hits = leakageHits(ch.code);
      assert(hits.length === 0, `execution machinery in TENSA source: ${hits.join(", ")}`);
      return "clean";
    },
  });

  if (e.paramTables !== undefined || e.paramOwners || e.paramCount !== undefined || e.sharedGroups !== undefined)
    cases.push({
      name: "parameter identity",
      fn: () => {
        const mod = compiled().mod;
        const rep = inspectModule(mod);
        if (e.paramTables !== undefined) assert(mod.params.length === e.paramTables, `expected ${e.paramTables} tables, got ${mod.params.length}`);
        if (e.paramCount !== undefined) assert(rep.totalParams === e.paramCount, `expected ${e.paramCount} values, inspect says ${rep.totalParams}`);
        if (e.sharedGroups !== undefined) assert(rep.sharedGroups.length === e.sharedGroups, `expected ${e.sharedGroups} shared groups, got ${rep.sharedGroups.length}`);
        for (const o of e.paramOwners ?? []) assert(mod.params.some((p) => p.owner === o), `no parameter owned by '${o}'; owners: ${[...new Set(mod.params.map((p) => p.owner))].join(", ")}`);
        return `${mod.params.length} tables, ${rep.totalParams ?? "?"} values, ${rep.sharedGroups.length} shared`;
      },
    });

  if (e.stateSlots !== undefined || e.effects)
    cases.push({
      name: "state and effects",
      fn: () => {
        const mod = compiled().mod;
        if (e.stateSlots !== undefined) assert(mod.states.length === e.stateSlots, `expected ${e.stateSlots} state slots, got ${mod.states.length}`);
        const present = new Set(allNodes(mod).flatMap((n) => n.effects));
        for (const ef of e.effects ?? []) assert(present.has(ef), `effect '${ef}' never appears; present: ${[...present].join(", ")}`);
        return `${mod.states.length} states, effects {${[...present].filter((x) => x !== "pure").join(", ")}}`;
      },
    });

  if (e.irOps || e.irForbidOps || e.constraints)
    cases.push({
      name: "IR structure",
      fn: () => {
        const mod = compiled().mod;
        const ops = new Set(allNodes(mod).map((n) => n.op));
        for (const op of e.irOps ?? []) assert(ops.has(op), `IR lacks op '${op}'; has ${[...ops].join(", ")}`);
        for (const op of e.irForbidOps ?? []) assert(!ops.has(op), `IR must not contain '${op}'`);
        const statuses = new Set(mod.constraints.map((c) => c.status));
        for (const s of e.constraints ?? []) assert(statuses.has(s), `no '${s}' constraint recorded; have ${[...statuses].join(", ") || "none"}`);
        return `${ops.size} distinct ops, constraints {${[...statuses].join(", ")}}`;
      },
    });

  if (e.inspectContains)
    cases.push({
      name: "inspect is authoritative (§35)",
      fn: () => {
        const mod = compiled().mod;
        const rep = inspectModule(mod);
        const text = renderInspect(rep);
        for (const s of e.inspectContains!) assert(text.includes(s), `inspect output lacks '${s}'`);
        // §35: inspect totals equal IR totals
        assert(rep.params.length === mod.params.length, `inspect lists ${rep.params.length} params, IR has ${mod.params.length}`);
        assert(rep.states.length === mod.states.length, `inspect lists ${rep.states.length} states, IR has ${mod.states.length}`);
        return `${text.split("\n").length} lines`;
      },
    });

  if (e.emitContains || e.emitForbids)
    cases.push({
      name: "PyTorch lowering",
      fn: () => {
        const code = emitTorch(compiled().mod);
        for (const s of e.emitContains ?? []) assert(code.includes(s), `emitted code lacks '${s}'`);
        for (const s of e.emitForbids ?? []) assert(!code.includes(s), `emitted code must not contain '${s}'`);
        return `${code.split("\n").length} lines`;
      },
    });

  if (e.run)
    cases.push({
      name: "reference execution",
      fn: () => {
        const run = e.run!;
        const rep = runProgram(compiled().mod, { maxSteps: run.steps ?? 0, dims: run.dims });
        if (run.refuses) {
          assert(rep.errors.length > 0, "runtime should have refused");
          if (run.errorContains) assert(rep.errors.some((x) => x.includes(run.errorContains!)), `no runtime error mentions '${run.errorContains}': ${rep.errors.join("; ")}`);
          // §5.2: a refusal must be a refusal — nothing may have executed
          assert(rep.forward.length === 0, `runtime reported errors but still executed ${rep.forward.map((f) => f.model).join(", ")}`);
          return `refused: ${rep.errors[0]}`;
        }
        assert(rep.errors.length === 0, rep.errors.join("; "));
        for (const [model, sub] of Object.entries(run.outputs ?? {})) {
          const f = rep.forward.find((x) => x.model === model);
          assert(!!f, `model '${model}' did not run; ran ${rep.forward.map((x) => x.model).join(", ")}`);
          assert(f.output.includes(sub), `'${model}' produced ${f.output}, expected to contain ${sub}`);
        }
        if (run.gradAll) {
          const missing = rep.gradCoverage.filter((g) => !g.updated);
          assert(missing.length === 0, `no gradient: ${missing.map((m) => `${m.param} (${m.reason})`).join(", ")}`);
        }
        return `${rep.forward.map((f) => `${f.model}->${f.output}`).join("; ")}${rep.losses.length ? `; ${rep.losses.length} steps` : ""}`;
      },
    });

  if (e.checkpointKinds)
    cases.push({
      name: "checkpoint coverage",
      fn: () => {
        const rep = inspectModule(compiled().mod);
        const kinds = Object.fromEntries(rep.checkpoint.map((c) => [c.kind, c.items]));
        for (const k of e.checkpointKinds!) assert((kinds[k] ?? 0) > 0, `checkpoint lacks '${k}'; has ${JSON.stringify(kinds)}`);
        return JSON.stringify(kinds);
      },
    });

  if (e.outputShape)
    cases.push({
      name: "static output shape",
      fn: () => {
        const mod = compiled().mod;
        const got: string[] = [];
        for (const [model, want] of Object.entries(e.outputShape!)) {
          const g = mod.graphs.find((x) => x.name === model);
          assert(!!g, `no graph named '${model}'`);
          const shown = g.outputs.map((o) => showType(mod.values.get(o)!.type)).join(", ");
          assert(shown === want, `'${model}' infers ${shown}, expected ${want}`);
          got.push(`${model}: ${shown}`);
        }
        return got.join("; ");
      },
    });

  for (const c of e.custom ?? []) cases.push({ name: c.name, fn: () => c.check(compiled().mod) });

  for (const tw of ch.twins) cases.push(...twinCases(ch, tw));
  return cases;
}

// ------------------------------------------------------------------ reusable custom checks

/** Every branch of every `parallel` node must consume the split input as its first tensor (§7.1 invariant). */
export function splitInvariant(mod: IRModule): string {
  const splits = allNodes(mod).filter((n) => n.op === "parallel");
  assert(splits.length > 0, "no parallel node in IR");
  for (const s of splits) {
    const input = s.inputs[0];
    for (const r of s.regions ?? []) {
      // the first node of a branch that reads any value must read the split input
      const first = r.nodes.find((n) => n.inputs.length > 0);
      assert(!!first, `branch '${r.label}' has no consuming node`);
      assert(first.inputs.includes(input), `branch '${r.label}' entry node '${first.op}' reads %${first.inputs.join(",%")}, not the split input %${input}`);
    }
    const merge = allNodes(mod).find((n) => s.outputs.length && s.outputs.every((o) => n.inputs.includes(o)));
    assert(!!merge, "no merge node consumes every branch result");
  }
  return `${splits.length} split(s), every branch entry reads the split input`;
}
