/** Export identical CPU-reference fixtures for the GPU fidelity gate.
 * npx tsx hardening/validate-m4.ts [output-directory]
 * The reference interpreter is explicitly the CPU oracle, not the execution target.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { compile } from "../src/lang/analyze";
import { emitTorch } from "../src/lang/emit_torch";
import { Runtime, syntheticField, guessVocab } from "../src/lang/exec";
import { IRNode } from "../src/lang/ir";
import { TensorType } from "../src/lang/types";
import * as X from "../src/lang/tensor";
import { RESEARCH_CHALLENGES } from "../src/lang/challenges/research";
import { LEARNING_CHALLENGES } from "../src/lang/challenges/learning";
const dir = process.argv[2] ?? "hardening/.m4-validation";
mkdirSync(dir, { recursive: true });
const pack = (t: X.T) => ({ shape: t.shape, data: [...t.data] });
class Oracle extends Runtime {
  draws: ReturnType<typeof pack>[] = [];
  override evalNode(n: IRNode) {
    super.evalNode(n);
    if (n.op === "randn_like") this.draws.push(pack(this.get(n.outputs[0])));
  }
}
const summaries = [];
for (const ch of [...RESEARCH_CHALLENGES, LEARNING_CHALLENGES.find(c => c.id === "distillation")!]) {
  const r = compile(ch.code, ch.id);
  if (!r.ok) { summaries.push({ id: ch.id, rejected: r.errors.map(d => `${d.code}: ${d.message}`) }); continue; }
  const rt = new Oracle(r.mod, { dims: ch.expect.run?.dims ?? { B: 2 }, seed: 11 });
  rt.allocate(); rt.training = false;
  const initial = { params: Object.fromEntries([...rt.params].map(([id, t]) => [id, pack(t)])),
    states: Object.fromEntries([...rt.states].map(([id, t]) => [id, pack(t)])) };
  const graphs = [];
  for (const g of r.mod.graphs.filter(g => g.kind === "model")) {
    X.beginTape(); rt.env = new Map(); rt.draws = [];
    for (const p of rt.params.values()) p.g = null;
    const inputs = g.inputs.map((iv, i) => {
      const ty = iv.type as TensorType;
      const t = syntheticField(rt.dims(ty.shape), ty.kind, guessVocab(r.mod, ty), i + 1);
      // F-025: exercise an empty attention row on GPU, not only ordinary padding.
      if (ty.kind === "Mask") t.data.fill(1, 0, t.size / t.shape[0]);
      return { ...pack(t), kind: ty.kind };
    });
    const output = rt.evalGraph(g, inputs.map(t => X.fromArray(t.shape, t.data)));
    const loss = output.reduce((acc, t) => X.add(acc, X.meanAll(t)), X.scalarT(0));
    X.backward(loss);
    const gradients = Object.fromEntries([...rt.params].filter(([,t]) => t.g !== null).map(([id,t]) => [id, [...t.g!]]));
    const evalDraws = rt.draws;
    rt.training = true; rt.env = new Map(); rt.draws = []; X.beginTape();
    const trainingOutput = rt.evalGraph(g, inputs.map(t => X.fromArray(t.shape,t.data))).map(pack);
    const trainingStates = Object.fromEntries([...rt.states].filter(([id]) => id.startsWith(g.name + "/") || id.startsWith(g.name + ".")).map(([id,t]) => [id,pack(t)]));
    graphs.push({ name: g.name, inputs, output: output.map(pack), draws: evalDraws, gradients,
      trainingOutput, trainingStates, trainingDraws: rt.draws });
    rt.training = false;
  }
  if (rt.errors.length) throw new Error(`${ch.id}: ${rt.errors.join(";")}`);
  const batch: Record<string, unknown> = {};
  for (const f of r.mod.data[0]?.fields ?? []) {
    const ty = f.type as TensorType;
    const t = syntheticField([rt.dimEnv.get("B")!, ...rt.dims(ty.shape ?? [])], ty.kind ?? "Tensor", guessVocab(r.mod, ty), 2);
    if (ch.id === "moco" && f.name === "label") t.data.fill(0); // positive key is column zero
    batch[f.name] = { ...pack(t), kind: ty.kind ?? "Tensor" };
  }
  writeFileSync(join(dir, `${ch.id}.py`), emitTorch(r.mod));
  writeFileSync(join(dir, `${ch.id}.json`), JSON.stringify({ id: ch.id, dims: Object.fromEntries(rt.dimEnv), ...initial, graphs, batch, plans: r.mod.plans.map(p => p.name) }));
  summaries.push({ id: ch.id, graphs: graphs.length, plans: r.mod.plans.length });
}
writeFileSync(join(dir, "manifest.json"), JSON.stringify(summaries, null, 2));
console.log(`Exported ${summaries.length} challenges to ${dir}; reference oracle explicitly CPU-only.`);
