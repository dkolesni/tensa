/**
 * TENSA hardening — metamorphic equivalence (§37).
 *
 * Two programs that should mean the same thing must produce the same IR
 * (modulo value naming), the same parameter tables, and — when asked — the
 * same forward output under a fixed seed and identical synthetic inputs.
 */
import { compile } from "../analyze";
import { Runtime, guessVocab, syntheticField } from "../exec";
import { inspectModule } from "../inspect";
import { IRModule, printIR } from "../ir";
import * as X from "../tensor";
import { TensorType } from "../types";

export interface EquivalenceOptions {
  /** also execute every model graph and compare outputs numerically */
  run?: { dims?: Record<string, number>; tol?: number };
  /** compare IR text (default true); turn off when only parameter identity should match */
  ir?: boolean;
  /** strings to strip from both IR texts before comparing (e.g. user-chosen binding names) */
  ignore?: string[];
}

/** Rename %values in order of first appearance so two lowerings can be compared textually. */
export function canonicalIR(mod: IRModule, ignore: string[] = []): string {
  let text = printIR(mod);
  for (const s of ignore) text = text.split(s).join("_");
  const seen = new Map<string, string>();
  return text.replace(/%[A-Za-z0-9_.]+/g, (m) => {
    if (!seen.has(m)) seen.set(m, `%v${seen.size}`);
    return seen.get(m)!;
  });
}

function forwardOutputs(mod: IRModule, dims?: Record<string, number>): number[][] {
  const rt = new Runtime(mod, { dims, seed: 11 });
  rt.allocate();
  const out: number[][] = [];
  for (const g of mod.graphs.filter((g) => g.kind === "model")) {
    X.beginTape();
    rt.env = new Map();
    rt.training = false;
    const inputs = g.inputs.map((iv, i) => {
      const t = iv.type as TensorType;
      return syntheticField(rt.dims(t.shape), t.kind, guessVocab(mod, t), i + 1);
    });
    for (const o of rt.evalGraph(g, inputs)) out.push([...o.data]);
  }
  return out;
}

export function assertEquivalent(a: string, b: string, opts: EquivalenceOptions = {}): string {
  // same module name on both sides: printIR embeds it in the header
  const ra = compile(a, "m");
  const rb = compile(b, "m");
  if (ra.errors.length) throw new Error(`A does not compile: ${ra.errors.map((d) => d.message).join("; ")}`);
  if (rb.errors.length) throw new Error(`B does not compile: ${rb.errors.map((d) => d.message).join("; ")}`);

  const pa = inspectModule(ra.mod);
  const pb = inspectModule(rb.mod);
  if (ra.mod.params.length !== rb.mod.params.length)
    throw new Error(`parameter tables differ: ${ra.mod.params.length} vs ${rb.mod.params.length}`);
  if (pa.totalParams !== pb.totalParams) throw new Error(`parameter totals differ: ${pa.totalParams} vs ${pb.totalParams}`);
  if (pa.sharedGroups.length !== pb.sharedGroups.length)
    throw new Error(`shared groups differ: ${pa.sharedGroups.length} vs ${pb.sharedGroups.length}`);

  if (opts.ir !== false) {
    const ia = canonicalIR(ra.mod, opts.ignore);
    const ib = canonicalIR(rb.mod, opts.ignore);
    if (ia !== ib) {
      const la = ia.split("\n");
      const lb = ib.split("\n");
      const i = la.findIndex((l, k) => l !== lb[k]);
      throw new Error(`IR differs at line ${i + 1}:\n  A: ${la[i]}\n  B: ${lb[i]}`);
    }
  }

  if (opts.run) {
    const oa = forwardOutputs(ra.mod, opts.run.dims);
    const ob = forwardOutputs(rb.mod, opts.run.dims);
    const tol = opts.run.tol ?? 1e-5;
    if (oa.length !== ob.length) throw new Error(`output count differs: ${oa.length} vs ${ob.length}`);
    for (let i = 0; i < oa.length; i++) {
      if (oa[i].length !== ob[i].length) throw new Error(`output ${i} size differs: ${oa[i].length} vs ${ob[i].length}`);
      for (let k = 0; k < oa[i].length; k++)
        if (Math.abs(oa[i][k] - ob[i][k]) > tol)
          throw new Error(`output ${i}[${k}] differs: ${oa[i][k]} vs ${ob[i][k]} (tol ${tol})`);
    }
  }
  return `${ra.mod.params.length} tables, ${pa.totalParams ?? "?"} values${opts.run ? ", outputs equal" : ""}`;
}
