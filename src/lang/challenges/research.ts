/** Milestone 4: tiny architectures, preserving the research boundary rather than hiding it. */
import { Challenge, CustomCheck } from "./types";
import { Runtime, runProgram } from "../exec";
import * as X from "../tensor";
import type { MetamorphicPair } from "./corpus";

function assert(ok: boolean, message: string): asserts ok { if (!ok) throw new Error(message); }
const proved: CustomCheck = {
  name: "all shape constraints are proved",
  check(mod) { assert(mod.constraints.every(c => c.status === "proved"), "unproved shape constraint"); return "proved"; },
};
const dense = `dim B
dim C
dim G
dim H
dim W
model Dense(x: Tensor[B, C, H, W]) -> Tensor[B, C + 4*G, H, W] {
  let a = x |> conv2d(G, kernel: 1) |> relu
  let c1 = concat(x, a, axis: 1)
  let b = c1 |> conv2d(G, kernel: 1) |> relu
  let c2 = concat(c1, b, axis: 1)
  let c = c2 |> conv2d(G, kernel: 1) |> relu
  let c3 = concat(c2, c, axis: 1)
  let d = c3 |> conv2d(G, kernel: 1) |> relu
  return concat(c3, d, axis: 1)
}`;

export const VIT = `dim B
dim H = 4
dim W = 6
dim P = 2
dim N = (H/P)*(W/P)
dim D = 8
model ViT(x: Image[B, 3, H, W]) -> Tensor[B, D] {
  param cls: Tensor[1, 1, D] init: normal
  let grid = reshape(x, [B, 3, H/P, P, W/P, P])
  let order = transpose(transpose(transpose(grid, 1, 2), 2, 4), 3, 4)
  let patches = reshape(order, [B, N, 3*P*P]) |> linear(D)
  let token = mean(patches, axis: 1, keep: true) * 0.0 + cls
  let tokens = concat(token, patches, axis: 1) |> positional(max: 7)
  let norm = tokens |> layernorm
  let att = norm |> attention(heads: 2)
  let skip = tokens + att
  let mlp = skip |> layernorm |> linear(16) |> gelu |> linear(D)
  return (skip + mlp)[:, 0, :]
}`;
const manualAttention = `block Att(x: Tensor[B, N + 1, D]) -> Tensor[B, N + 1, D] {
  param wq: Tensor[D, D] init: xavier
  param wk: Tensor[D, D] init: xavier
  param wv: Tensor[D, D] init: xavier
  param wo: Tensor[D, D] init: xavier
  let q = transpose(reshape(matmul(x, wq), [B, N+1, 2, D/2]), 1, 2)
  let k = transpose(reshape(matmul(x, wk), [B, N+1, 2, D/2]), 1, 2)
  let v = transpose(reshape(matmul(x, wv), [B, N+1, 2, D/2]), 1, 2)
  let w = softmax(matmul(q, transpose(k, 2, 3)) / sqrt(D/2), axis: -1)
  return matmul(reshape(transpose(matmul(w, v), 1, 2), [B, N+1, D]), wo)
}
`;

const seq2seq = `dim B
dim S = 3
dim T = 2
dim D = 8
model Encoder(x: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, S, D] {
  let a = attention(query: x, mask: pad, heads: 2)
  return x + a |> layernorm
}
model Decoder(q: Tensor[B, T, D], memory: Tensor[B, S, D], pad: Mask[B, 1, S]) -> Tensor[B, T, D] {
  let a = q |> attention(heads: 2, causal: true)
  let h = q + a
  let cross = attention(query: h, key: memory, value: memory, mask: pad, heads: 2)
  return h + cross |> layernorm |> linear(D)
}
objective L(y: Tensor[B, T, D], target: Tensor[B, T, D]) -> Scalar { return mse(y, target) }
source S0 = synthetic(features: 8)
data Data from S0 { example {
  field x: Tensor[S, D] = decode(row)
  field q: Tensor[T, D] = decode(row)
  field pad: Mask[1, S] = pad_mask(S)
  field target: Tensor[T, D] = decode(row)
} batch 2 }
train Fit {
  data Data
  model enc = Encoder
  model dec = Decoder
  loss main = L(y: dec(q, enc(x, pad), pad), target: target)
  optimizer opt = adam(lr: 0.001)
  epochs 1
}`;

export const VAE = `dim B
dim D = 6
dim Z = 2
block Encode(x: Tensor[B, D]) -> (Tensor[B, Z], Tensor[B, Z]) {
  let h = x |> linear(8) |> gelu
  return (h |> linear(Z), h |> linear(Z))
}
model VAE(x: Tensor[B, D]) -> (Tensor[B, D], Tensor[B, Z], Tensor[B, Z]) {
  let (mu, logvar) = Encode(x)
  let eps = randn_like(mu)
  let z = mu + exp(0.5 * logvar) * eps
  let recon = z |> linear(8) |> gelu |> linear(D)
  return (recon, mu, logvar)
}
objective ELBO(recon: Tensor[B, D], mu: Tensor[B, Z], logvar: Tensor[B, Z], x: Tensor[B, D]) -> Scalar {
  let kl = 0.5 * mean(sum(mu * mu + exp(logvar) - 1.0 - logvar, axis: -1))
  return mse(recon, x) + 0.01 * kl
}
source S = synthetic(features: 6)
data Data from S { example { field x: Tensor[D] = decode(row) } batch 2 }
train Fit {
  data Data
  model m = VAE
  loss main = ELBO(recon: m(x)[0], mu: m(x)[1], logvar: m(x)[2], x: x)
  optimizer opt = adam(lr: 0.001)
  epochs 1
}`;

const lora = `dim B
dim D = 8
dim R = 2
block Base(x: Tensor[B, D]) -> Tensor[B, D] { linear(D) }
block Adapter(x: Tensor[B, D]) -> Tensor[B, D] {
  param down: Tensor[D, R] init: normal
  param up: Tensor[R, D] init: zeros
  return matmul(matmul(x, down), up) * 0.5
}
model LoRA(x: Tensor[B, D]) -> Tensor[B, D] {
  let base = Base()
  let adapter = Adapter()
  return base(x) + adapter(x)
}
objective L(y: Tensor[B, D], target: Tensor[B, D]) -> Scalar { return mse(y, target) }
source S = synthetic(features: 8)
data Data from S { example { field x: Tensor[D] = decode(row) field target: Tensor[D] = decode(row) } batch 2 }
train Tune {
  data Data
  model m = LoRA
  loss main = L(y: m(x), target: target)
  optimizer opt = adam(lr: 0.001) over m.adapter
  phase tune { steps 3 freeze m.base update main with opt }
}`;

const moco = `dim B = 2
dim D = 4
block Body(x: Tensor[B, D]) -> Tensor[B, D] { linear(D) ; batchnorm ; relu ; linear(D) }
model Query(x: Tensor[B, D]) -> Tensor[B, D] { Body() }
model Key(x: Tensor[B, D]) -> Tensor[B, D] { Body() }
model Bank(q: Tensor[B, D], k: Tensor[B, D]) -> Logits[B, 5] {
  state bank: Tensor[4, D] init: ones update: assign
  let keys = stop_grad(k)
  let old = bank * 1.0
  let positive = sum(q * keys, axis: -1, keep: true)
  let negatives = matmul(q, transpose(old, 0, 1))
  let next = concat(old[2:4, :], keys, axis: 0)
  let saved = observe(bank, next)
  return concat(positive, negatives, axis: -1) / 0.2
}
objective Contrast(y: Logits[B, 5], label: Class[B]) -> Scalar { return cross_entropy(y, label) }
source S = synthetic(features: 4)
data Data from S { example {
  field a: Tensor[D] = decode(row)
  field b: Tensor[D] = decode(row)
  field label: Class = as_class(target)
} batch 2 }
train Fit {
  data Data
  model query = Query
  model key = Key
  model bank = Bank
  track momentum = ema(query, rate: 0.99)
  loss main = Contrast(y: bank(query(a), key(b)), label: label)
  optimizer opt = adam(lr: 0.001) over query
  epochs 1
}`;

const wgan = `dim B
dim D = 4
model Critic(x: Tensor[B, D]) -> Tensor[B, 1] { linear(8) ; gelu ; linear(1) }
model Penalty(real: Tensor[B, D], fake: Tensor[B, D], mix: Tensor[B, 1]) -> Scalar {
  let xhat = mix * real + (1.0 - mix) * stop_grad(fake)
  let score = Critic(xhat)
  let gx = grad(score, xhat)
  return mean((sqrt(sum(gx * gx, axis: -1)) - 1.0) * (sqrt(sum(gx * gx, axis: -1)) - 1.0))
}`;
const diffusion = `dim B
dim D = 4
model Denoise(x: Tensor[B, D], alpha: Tensor[B, 1]) -> (Tensor[B, D], Tensor[B, D]) {
  let eps = randn_like(x)
  let strength = sigmoid(alpha)
  let noisy = sqrt(strength) * x + sqrt(1.0 - strength) * eps
  let pred = concat(noisy, strength, axis: -1) |> linear(8) |> gelu |> linear(D)
  return (pred, eps)
}
objective Noise(pred: Tensor[B, D], eps: Tensor[B, D]) -> Scalar { return mse(pred, stop_grad(eps)) }
source S = synthetic(features: 4)
data Data from S { example { field x: Tensor[D] = decode(row) field alpha: Tensor[1] = decode(row) } batch 2 }
train Fit {
  data Data
  model m = Denoise
  loss main = Noise(pred: m(x, alpha)[0], eps: m(x, alpha)[1])
  optimizer opt = adam(lr: 0.001)
  epochs 1
}`;
const moe = `dim B
dim D = 4
model MoE(x: Tensor[B, D]) -> Tensor[B, D] {
  let gates = x |> linear(3) |> softmax
  let routes = topk(gates, k: 2)
  return where(routes, linear(x: x, out: D), x)
}`;
const rl = `dim B
dim D = 4
model Agent(obs: Tensor[B, D]) -> (Logits[B, 3], Tensor[B, 1]) {
  let h = obs |> linear(8) |> tanh
  return (h |> linear(3), h |> linear(1))
}
objective AC(policy: Logits[B, 3], value: Tensor[B, 1], action: Class[B], advantage: Tensor[B], target: Tensor[B, 1]) -> Scalar {
  let chosen = sum(log_softmax(policy, axis: -1) * one_hot(action, classes: 3), axis: -1)
  return mean(0.0 - chosen * stop_grad(advantage)) + 0.5 * mse(value, target)
}
source Rollouts = synthetic(features: 4)
data Trajectories from Rollouts { example {
  field obs: Tensor[D] = decode(row)
  field action: Class = as_class(target)
  field advantage: Scalar = decode(row)
  field target: Tensor[1] = decode(row)
} batch 2 }
train Update {
  data Trajectories
  model m = Agent
  loss main = AC(policy: m(obs)[0], value: m(obs)[1], action: action, advantage: advantage, target: target)
  optimizer opt = adam(lr: 0.001)
  epochs 1
}`;

function positive(id: string, tier: Challenge["tier"], code: string, ops: string[], tables: number, dims: Record<string, number> = { B: 2 }): Challenge {
  return { id, title: id, section: "§30 / §47", tier, code, record: `${id}.md`,
    expect: { warnCodes: [], paramTables: tables, irOps: ops, inspectContains: ["parameters"], emitForbids: ["NotImplemented"], checkpointKinds: ["parameters"], run: { dims, steps: code.includes("train ") ? 2 : 0 } }, twins: [] };
}
const denseChallenge = positive("densenet", 2, dense, ["concat", "conv2d"], 8, { B: 2, C: 3, G: 2, H: 2, W: 3 });
denseChallenge.expect.outputShape = { Dense: "Tensor[B, C + 4*G, H, W]" };
denseChallenge.expect.custom = [proved];
denseChallenge.twins = [{ id: "missing-growth", mutates: "drop the last growth features", code: dense.replace("concat(c3, d, axis: 1)", "c3"), expectCodes: ["AXS0401"] }];
const vitChallenge = positive("vit", 3, VIT, ["reshape", "transpose", "concat", "positional", "attention"], 16);
vitChallenge.expect.custom = [proved];
vitChallenge.expect.emitContains = ['.reshape(dims["B"], 3, 2, 2, 3, 2)'];
vitChallenge.twins = [{ id: "patch-count", mutates: "one extra patch", code: VIT.replace("[B, N, 3*P*P]", "[B, N+1, 3*P*P]"), expectCodes: ["AXS0401"] }];
const seqChallenge = positive("encoder-decoder", 3, seq2seq, ["attention", "layernorm"], 18);
seqChallenge.expect.emitContains = ["attn_mask=m", "[:, None]", 'dec(batch["q"], enc(batch["x"], batch["pad"]), batch["pad"])'];
seqChallenge.expect.run!.gradAll = true;
seqChallenge.twins = [{ id: "wrong-mask-length", mutates: "query-length mask instead of source-length mask", code: seq2seq.split("Mask[B, 1, S]").join("Mask[B, 1, T]"), expectCodes: ["AXS0401"] }];
const vaeChallenge = positive("vae", 4, VAE, ["randn_like", "exp", "mean"], 10);
vaeChallenge.expect.effects = ["stochastic"];
vaeChallenge.expect.emitContains = ["torch.randn_like"];
vaeChallenge.expect.run!.gradAll = true;
vaeChallenge.twins = [{ id: "latent-width", mutates: "sample wider than the mean", code: VAE.replace("randn_like(mu)", "randn_like(concat(mu, mu, axis: -1))"), expectCodes: ["AXS0401"] }];
vaeChallenge.expect.custom = [{ name: "sampling stays stochastic in eval, with no gradient to its template (H-012)", check(mod) {
  const rt = new Runtime(mod, { dims: { B: 2 }, seed: 11 }); rt.allocate(); rt.training = false;
  const g = mod.graphs.find(g => g.name === "VAE")!;
  const x = X.full([2, 6], 0.2);
  const a = [...rt.evalGraph(g, [x])[0].data]; rt.env = new Map();
  const b = [...rt.evalGraph(g, [x])[0].data];
  assert(a.some((v, i) => v !== b[i]), "eval incorrectly disabled randomness");
  const noise = X.randnLike(X.full([4096], 1, true));
  assert(!noise.req && !noise.back, "noise must not differentiate its shape template");
  const mean = noise.data.reduce((s, v) => s + v, 0) / noise.size;
  const variance = noise.data.reduce((s, v) => s + v*v, 0) / noise.size;
  assert(Math.abs(mean) < 0.1 && variance > 0.85 && variance < 1.15, "not standard normal");
  return "independent eval draws, N(0,1), no template gradient";
} }];
const loraChallenge = positive("lora", 4, lora, ["matmul"], 4);
loraChallenge.expect.paramOwners = ["LoRA/base/linear#1", "LoRA/adapter"];
loraChallenge.expect.custom = [{ name: "only adapters update", check(mod) {
  const r = runProgram(mod, { maxSteps: 3 }); assert(!r.errors.length, r.errors.join(";"));
  const a = r.gradCoverage.filter(g => g.param.includes("/adapter."));
  assert(a.length === 2 && a.every(g => g.updated), JSON.stringify(r.gradCoverage));
  assert(r.gradCoverage.filter(g => g.param.includes("/base/")).every(g => !g.updated), "base moved"); return "adapters only";
} }];
loraChallenge.twins = [
  { id: "frozen-updated", mutates: "optimizer claims only frozen base", code: lora.replace("over m.adapter", "over m.base"), expectCodes: ["AXS0704"] },
  { id: "unused-optimizer", mutates: "declared adapter optimizer never applied", code: lora.replace("  phase tune", "  optimizer unused = adam(lr: 0.001) over m.base\n  phase tune"), expectCodes: ["AXS0708"] },
];
const mocoChallenge = positive("moco", 4, moco, ["state_update", "batchnorm", "stop_grad"], 12);
mocoChallenge.title = "MoCo queue + independent key encoder: EMA consumption deliberately unsupported";
mocoChallenge.expect.custom = [{ name: "queue shifts exactly, eval is read-only, BN ownership is separate", check(mod) {
  const rt = new Runtime(mod, { dims: { B: 2 }, seed: 11 }); rt.allocate();
  const bank = mod.graphs.find(g => g.name === "Bank")!;
  const slot = mod.states.find(s => s.owner === "Bank")!;
  assert(!!slot, "missing bank state");
  rt.training = false;
  rt.evalGraph(bank, [X.full([2,4],2), X.full([2,4],3)]);
  assert(rt.states.get(slot.id)!.data.every(v => v === 1), "eval enqueued keys");
  rt.training = true; rt.env = new Map();
  rt.evalGraph(bank, [X.full([2,4],2), X.full([2,4],3)]);
  const first = [...rt.states.get(slot.id)!.data];
  assert(first.slice(0,8).every(v => v === 1) && first.slice(8).every(v => v === 3), "wrong first queue shift");
  rt.env = new Map(); rt.evalGraph(bank, [X.full([2,4],2), X.full([2,4],4)]);
  const next = [...rt.states.get(slot.id)!.data];
  assert(next.slice(0,8).every(v => v === 3) && next.slice(8).every(v => v === 4), "queue did not persist");
  for (const owner of ["Query", "Key"]) assert(mod.states.filter(s => s.owner.startsWith(owner + "/")).length === 2, `missing ${owner} BN stats`);
  assert(mod.plans[0].tracks[0].region === "query", "EMA source changed");
  return "FIFO state persists; query/key have separate BN buffers; EMA still cannot feed Key";
} }];
mocoChallenge.expect.stateSlots = 11;
mocoChallenge.expect.effects = ["writes-state", "grad-stopped"];
mocoChallenge.expect.checkpointKinds = ["parameters", "persistent state"];
mocoChallenge.twins = [{ id: "queue-overrun", mutates: "slice past the bank capacity", code: moco.replace("old[2:4, :]", "old[2:5, :]"), expectCodes: ["AXS0410"] }];
const diffusionChallenge = positive("diffusion", 5, diffusion, ["randn_like", "sqrt", "stop_grad"], 4);
// Host supplies logit(alpha_bar); sigmoid guarantees a valid variance for synthetic fixtures too.
diffusionChallenge.expect.run!.gradAll = true;
diffusionChallenge.twins = [{ id: "alpha-width", mutates: "alpha no longer broadcasts", code: diffusion.split("alpha: Tensor[B, 1]").join("alpha: Tensor[B, 3]"), expectCodes: ["AXS0401"] }];
const rlChallenge = positive("rl-policy-value", 6, rl, ["log_softmax", "one_hot", "stop_grad"], 6);
rlChallenge.expect.run!.gradAll = true;
rlChallenge.twins = [{ id: "action-rank", mutates: "action has a spurious axis", code: rl.replace("field action: Class", "field action: Class[1]"), expectCodes: ["AXS0603"] }];
function rejected(id: string, tier: Challenge["tier"], code: string, op: string): Challenge {
  return { id, title: `${id}: expected research boundary`, section: "§13 / §29 / §47", tier, code, record: `${id}.md`, expect: { ok: false,
    custom: [{ name: `exact missing operation: ${op}`, check(mod) {
      const d = mod.diags.find(d => d.code === "AXS0204" && d.message.includes(op));
      assert(!!d, mod.diags.map(d => `${d.code}: ${d.message}`).join("; ")); return d.message;
    } }],
  }, twins: [{ id: "unknown-op-mutant", mutates: "misspell the unavailable primitive", code: code.replace(op + "(", "missing_research_op("), expectCodes: ["AXS0204"] }] };
}
export const RESEARCH_CHALLENGES: Challenge[] = [denseChallenge, vitChallenge, seqChallenge, vaeChallenge, loraChallenge, mocoChallenge,
  rejected("wgan-gp", 5, wgan, "grad"), diffusionChallenge, rejected("moe-topk", 6, moe, "topk"), rlChallenge];
export const RESEARCH_METAMORPHIC: MetamorphicPair[] = [
  { id: "vit-manual-vs-catalog", section: "§12 / §30", a: VIT, b: VIT.replace("model ViT", manualAttention + "model ViT").replace("attention(heads: 2)", "Att()"), ir: false, run: { dims: { B: 2 }, tol: 1e-4 } },
  { id: "dense-channel-algebra", section: "§5 / §30", a: dense, b: dense.replace("C + 4*G", "G + (C + 3*G)"), run: { dims: { B: 2, C: 3, G: 2, H: 2, W: 3 } } },
];
