/**
 * TENSA — reference backend: executes the backend-neutral IR.
 *
 * The executor is deliberately a straight interpreter over IR nodes so that
 * what `inspect` claims and what `run` does cannot drift apart.
 */
import { capabilityOf } from "./catalog";
import { DimExpr, evalDim, show } from "./dims";
import { IREvent, IRGraph, IRModule, IRNode, IRParam, IRPhase } from "./ir";
import * as X from "./tensor";
import { T } from "./tensor";
import { TensorType, ValueType, isTensor } from "./types";

export interface RunOptions {
  maxSteps?: number;
  seed?: number;
  dims?: Record<string, number>;
}

export interface RunReport {
  dims: { name: string; value: number; source: string }[];
  forward: { model: string; inputs: string[]; output: string; ms: number }[];
  losses: { step: number; phase: string; values: Record<string, number> }[];
  finalMetrics: Record<string, number>;
  paramCount: number;
  notes: string[];
  errors: string[];
  approximations: string[];
  gradCoverage: { param: string; updated: boolean; reason?: string }[];
  /** one row per executed phase: how long it ran and what ended it (H-001) */
  phases: { name: string; steps: number; stoppedBy: "epochs" | "steps" | "until" | "maxSteps"; lrStart: number; lrEnd: number }[];
  /** `every … { validate ; checkpoint }` firings, in order (H-001) */
  events: { step: number; phase: string; action: string; detail: string }[];
}

/** the reference runtime maps one epoch to this many synthetic steps */
export const STEPS_PER_EPOCH = 6;

/** `ema(r)` → r, the decay (weight kept from the previous value) */
export function emaDecay(rule: string): number {
  const m = /\(\s*(?:rate\s*:\s*)?([\d.eE+-]+)\s*\)/.exec(rule);
  return m ? parseFloat(m[1]) : 0.9;
}

/**
 * Learning-rate multiplier at `step` of a phase `total` steps long under a
 * lifecycle schedule such as `cosine(warmup: 20)`, `linear(warmup: 5)`,
 * `step(every: 10, gamma: 0.5)` or `constant`.
 */
export function scheduleMultiplier(schedule: string | null, step: number, total: number): number {
  if (!schedule) return 1;
  const kind = /^(\w+)/.exec(schedule)?.[1] ?? "constant";
  const arg = (name: string, d: number) => {
    const m = new RegExp(`${name}\\s*:\\s*([\\d.eE+-]+)`).exec(schedule);
    return m ? parseFloat(m[1]) : d;
  };
  const warmup = arg("warmup", 0);
  if (warmup > 0 && step < warmup) return (step + 1) / warmup;
  const span = Math.max(1, total - warmup);
  const t = Math.min(1, Math.max(0, (step - warmup) / span));
  switch (kind) {
    case "cosine":
      return 0.5 * (1 + Math.cos(Math.PI * t));
    case "linear":
      return 1 - t;
    case "step":
      return Math.pow(arg("gamma", 0.1), Math.floor((step - warmup) / Math.max(1, arg("every", 10))));
    default:
      return 1;
  }
}

type Env = Map<string, T>;

const DEFAULT_DIMS: Record<string, number> = { B: 4, N: 4, T: 8, S: 8, L: 8, H: 16, W: 16, C: 3 };

export class Runtime {
  env: Env = new Map();
  params = new Map<string, T>();
  states = new Map<string, T>();
  optState = new Map<string, { m: Float32Array; v: Float32Array; t: number }>();
  dimEnv = new Map<string, number>();
  training = true;
  notes: string[] = [];
  approx: string[] = [];
  errors: string[] = [];

  constructor(public mod: IRModule, public opts: RunOptions = {}) {
    X.setSeed(opts.seed ?? 7);
    for (const d of mod.dims) {
      if (d.value) {
        const v = evalDim(d.value, this.dimEnv);
        if (v !== null) this.dimEnv.set(d.name, v);
      }
    }
    const data = mod.data[0];
    const batch = data ? Math.min(data.batch, 4) : 4;
    for (const d of mod.dims)
      if (!this.dimEnv.has(d.name))
        this.dimEnv.set(
          d.name,
          opts.dims?.[d.name] ?? (d.name === "B" ? batch : DEFAULT_DIMS[d.name] ?? 8)
        );
    // template dims that never appear as declarations
    for (const g of mod.graphs)
      for (const i of g.inputs)
        if (isTensor(i.type))
          for (const s of i.type.shape)
            for (const t of s.terms)
              for (const v of t.vars) if (!this.dimEnv.has(v)) this.dimEnv.set(v, DEFAULT_DIMS[v] ?? 8);
  }

  dims(shape: DimExpr[]): number[] {
    return shape.map((s) => {
      const v = evalDim(s, this.dimEnv);
      if (v === null || v <= 0) {
        this.errors.push(`cannot bind symbolic dimension ${show(s)} to a runtime value`);
        return 1;
      }
      return v;
    });
  }

  initTensor(shape: number[], init: string, fanIn: number): T {
    const t = X.zeros(shape, true);
    const n = t.size;
    switch (init) {
      case "zeros":
        break;
      case "ones":
        t.data.fill(1);
        break;
      case "normal":
        for (let i = 0; i < n; i++) t.data[i] = X.randn() * 0.02;
        break;
      case "kaiming": {
        const s = Math.sqrt(2 / Math.max(fanIn, 1));
        for (let i = 0; i < n; i++) t.data[i] = X.randn() * s;
        break;
      }
      case "xavier":
      default: {
        const s = Math.sqrt(1 / Math.max(fanIn, 1));
        for (let i = 0; i < n; i++) t.data[i] = X.randn() * s;
      }
    }
    return t;
  }

  allocate() {
    for (const p of this.mod.params) {
      const shape = this.dims(p.shape);
      const fanIn = shape.length > 1 ? shape.slice(0, -1).reduce((a, b) => a * b, 1) : shape[0] ?? 1;
      this.params.set(p.id, this.initTensor(shape, p.init, fanIn));
    }
    for (const s of this.mod.states) {
      const shape = this.dims(s.shape);
      this.states.set(s.id, this.initTensor(shape, s.init === "copy" ? "zeros" : s.init, 1));
    }
  }

  /**
   * Constraints the checker could not prove statically are carried into the
   * IR as `assumed`; once dimensions are bound they must hold, otherwise the
   * runtime refuses to execute (finding F-002).  Returns the violations.
   */
  checkConstraints(): string[] {
    const out: string[] = [];
    for (const c of this.mod.constraints) {
      if (c.status !== "assumed") continue;
      const l = evalDim(c.lhs, this.dimEnv);
      const r = evalDim(c.rhs, this.dimEnv);
      if (l === null || r === null) continue;
      if (c.rel === "<=") {
        if (l <= r) continue;
        out.push(`carried constraint violated: ${c.origin}: ${show(c.lhs)} ≤ ${show(c.rhs)} but ${l} > ${r} under the bound dimensions`);
        continue;
      }
      if (l === r) continue;
      out.push(`carried constraint violated: ${c.origin}: ${show(c.lhs)} = ${show(c.rhs)} but ${l} ≠ ${r} under the bound dimensions`);
    }
    return out;
  }

  paramCount(): number {
    let n = 0;
    for (const p of this.mod.params) n += this.dims(p.shape).reduce((a, b) => a * b, 1);
    return n;
  }

  // ---------------------------------------------------------------- graph eval
  evalGraph(g: IRGraph, inputs: T[]): T[] {
    g.inputs.forEach((iv, i) => {
      if (inputs[i]) this.env.set(iv.id, inputs[i]);
    });
    this.evalNodes(g.nodes);
    return g.outputs.map((o) => this.env.get(o) ?? X.zeros([1]));
  }

  evalNodes(nodes: IRNode[]) {
    for (const n of nodes) this.evalNode(n);
  }

  get(id: string): T {
    const t = this.env.get(id);
    if (!t) {
      this.errors.push(`value %${id} was not produced before use`);
      return X.zeros([1]);
    }
    return t;
  }

  outType(n: IRNode, i = 0): ValueType | undefined {
    return this.mod.values.get(n.outputs[i])?.type;
  }

  evalNode(n: IRNode) {
    const ins = n.inputs.map((i) => this.get(i));
    const set = (i: number, t: T) => this.env.set(n.outputs[i], t);
    const P = (i: number) => this.params.get(n.params[i]) ?? X.zeros([1], true);
    const ot = this.outType(n);
    const oshape = ot && isTensor(ot) ? this.dims(ot.shape) : [];
    const cap = capabilityOf(n.op, "reference");
    if (!cap.forward) {
      this.errors.push(`reference backend cannot execute '${n.op}' (${cap.note ?? "not implemented"})`);
      set(0, X.zeros(oshape));
      return;
    }
    if (cap.note && !this.approx.includes(`${n.op}: ${cap.note}`)) this.approx.push(`${n.op}: ${cap.note}`);

    switch (n.op) {
      case "const":
        set(0, X.scalarT(Number(n.attrs.value ?? 0)));
        return;
      case "const_dim": {
        const v = evalDim(n.attrs.dim as DimExpr, this.dimEnv) ?? 0;
        set(0, X.scalarT(v));
        return;
      }
      case "param_read":
        set(0, P(0));
        return;
      case "state_read":
        set(0, this.states.get(n.states[0]) ?? X.zeros([1]));
        return;
      case "state_update": {
        const cur = this.states.get(n.states[0]) ?? X.zeros(ins[0].shape);
        const rule = String(n.attrs.rule ?? "assign");
        // `ema(r)`: r is the DECAY — the weight kept from the previous value
        // (next = r·cur + (1−r)·new), the reading `ema(0.99)` has everywhere
        // else in ML (E-009); the emitter lowers the same formula
        const decay = emaDecay(rule);
        if (this.training) {
          const next = X.zeros(cur.shape);
          for (let i = 0; i < next.size; i++)
            next.data[i] = rule.startsWith("ema") ? decay * cur.data[i] + (1 - decay) * ins[0].data[i] : ins[0].data[i];
          this.states.set(n.states[0], next);
          set(0, next);
        } else set(0, cur);
        return;
      }
      case "add":
        set(0, X.add(ins[0], ins[1]));
        return;
      case "merge_add":
      case "merge_mean": {
        // n-ary branch join: every branch result participates (F-005)
        let acc = ins[0];
        for (let i = 1; i < ins.length; i++) acc = X.add(acc, ins[i]);
        set(0, n.op === "merge_mean" ? X.scaleT(acc, 1 / ins.length) : acc);
        return;
      }
      case "sub":
        set(0, X.sub(ins[0], ins[1]));
        return;
      case "mul":
        set(0, X.mul(ins[0], ins[1]));
        return;
      case "div":
        set(0, X.div(ins[0], ins[1]));
        return;
      case "linear": {
        let y = X.matmul(ins[0], P(0));
        if (n.params.length > 1) y = X.add(y, P(1));
        set(0, y);
        return;
      }
      case "conv2d":
        set(0, X.conv2d(ins[0], P(0), n.params.length > 1 ? P(1) : null, Number(n.attrs.stride ?? 1), Number(n.attrs.pad ?? 0)));
        return;
      case "maxpool2d":
        set(0, X.pool2d(ins[0], Number(n.attrs.kernel ?? 2), Number(n.attrs.stride ?? 2), Number(n.attrs.pad ?? 0), "max"));
        return;
      case "avgpool2d":
        set(0, X.pool2d(ins[0], Number(n.attrs.kernel ?? 2), Number(n.attrs.stride ?? 2), 0, "avg"));
        return;
      case "upsample":
        set(0, X.upsample2d(ins[0], Number(n.attrs.scale ?? 2)));
        return;
      case "global_avgpool": {
        const [B, C, H, W] = ins[0].shape;
        set(0, X.reshape(X.reduce(X.reshape(ins[0], [B, C, H * W]), 2, false, "mean"), [B, C]));
        return;
      }
      case "flatten":
        set(0, X.reshape(ins[0], [ins[0].shape[0], ins[0].size / ins[0].shape[0]]));
        return;
      case "embedding":
        set(0, X.embedding(P(0), ins[0]));
        return;
      case "positional": {
        const [, Tn] = ins[0].shape;
        const table = P(0);
        const pos = X.sliceT(table, [0], [0], [Tn], [0]);
        set(0, X.add(ins[0], X.reshape(pos, [1, Tn, table.shape[1]])));
        return;
      }
      case "layernorm": {
        const x = ins[0];
        const D = x.shape[x.shape.length - 1];
        const mean = X.reduce(x, x.shape.length - 1, true, "mean");
        const c = X.sub(x, mean);
        const varr = X.reduce(X.mul(c, c), x.shape.length - 1, true, "mean");
        const norm = X.mul(c, X.rsqrtT(X.add(varr, X.scalarT(Number(n.attrs.eps ?? 1e-5)))));
        set(0, X.add(X.mul(norm, X.reshape(P(0), [D])), X.reshape(P(1), [D])));
        return;
      }
      case "rmsnorm": {
        const x = ins[0];
        const D = x.shape[x.shape.length - 1];
        const ms = X.reduce(X.mul(x, x), x.shape.length - 1, true, "mean");
        set(0, X.mul(X.mul(x, X.rsqrtT(X.add(ms, X.scalarT(Number(n.attrs.eps ?? 1e-6))))), X.reshape(P(0), [D])));
        return;
      }
      case "batchnorm": {
        const x = ins[0];
        const is4 = x.shape.length === 4;
        const C = is4 ? x.shape[1] : x.shape[1];
        const gamma = X.reshape(P(0), is4 ? [1, C, 1, 1] : [1, C]);
        const beta = X.reshape(P(1), is4 ? [1, C, 1, 1] : [1, C]);
        const rm = this.states.get(n.states[0])!;
        const rv = this.states.get(n.states[1])!;
        const mom = Number(n.attrs.momentum ?? 0.1);
        let mean: T;
        let varr: T;
        if (this.training) {
          const axes = is4 ? [0, 2, 3] : [0];
          let m = x;
          for (const a of axes.slice().reverse()) m = X.reduce(m, a, true, "mean");
          mean = m;
          const c = X.sub(x, mean);
          let v = X.mul(c, c);
          for (const a of axes.slice().reverse()) v = X.reduce(v, a, true, "mean");
          varr = v;
          const samples = x.size / C;
          if (samples <= 1) throw new Error("batchnorm training requires more than one sample per channel");
          const nm = X.zeros([C]);
          const nv = X.zeros([C]);
          for (let i = 0; i < C; i++) {
            nm.data[i] = (1 - mom) * rm.data[i] + mom * mean.data[i];
            // F-029: normalize with population variance, store the unbiased estimate,
            // matching PyTorch BatchNorm's running-statistics convention.
            nv.data[i] = (1 - mom) * rv.data[i] + mom * varr.data[i] * samples / (samples - 1);
          }
          this.states.set(n.states[0], nm);
          this.states.set(n.states[1], nv);
        } else {
          mean = X.reshape(rm, is4 ? [1, C, 1, 1] : [1, C]);
          varr = X.reshape(rv, is4 ? [1, C, 1, 1] : [1, C]);
        }
        const norm = X.mul(X.sub(x, mean), X.rsqrtT(X.add(varr, X.scalarT(1e-5))));
        set(0, X.add(X.mul(norm, gamma), beta));
        return;
      }
      case "randn_like":
        set(0, X.randnLike(ins[0]));
        break;
      case "dropout":
        set(0, X.dropout(ins[0], Number(n.attrs.p ?? 0.5), this.training));
        return;
      case "attention": {
        const q0 = ins[0];
        const k0 = ins[1] ?? ins[0];
        const v0 = ins[2] ?? k0;
        const [B, Tq, D] = q0.shape;
        const Tk = k0.shape[1];
        const heads = evalDim(n.attrs.heads as DimExpr, this.dimEnv) ?? 1;
        const dh = Math.floor(D / heads);
        const split = (t: T, len: number) => X.transpose(X.reshape(t, [B, len, heads, dh]), 1, 2);
        const q = split(X.matmul(q0, P(0)), Tq);
        const k = split(X.matmul(k0, P(1)), Tk);
        const v = split(X.matmul(v0, P(2)), Tk);
        let scores = X.scaleT(X.matmul(q, X.transpose(k, 2, 3)), 1 / Math.sqrt(dh));
        const masks: T[] = [];
        if (n.attrs.causal) {
          const mask = X.zeros([Tq, Tk]);
          for (let i = 0; i < Tq; i++) for (let j = 0; j < Tk; j++) mask.data[i * Tk + j] = j > i ? 1 : 0;
          const lifted = X.reshape(mask, [1, 1, Tq, Tk]);
          masks.push(lifted);
          scores = X.maskedFill(scores, lifted, -1e9);
        }
        if (ins[3]) {
          // Mask[Tq, Tk] or Mask[B|1, Tq|1, Tk] -> aligned against scores [B, H, Tq, Tk]
          // (a rank-3 mask used to trailing-broadcast its batch axis onto heads, F-012)
          const m = ins[3];
          const m4 = m.shape.length === 3 ? X.reshape(m, [m.shape[0], 1, m.shape[1], m.shape[2]]) : m;
          masks.push(m4);
          scores = X.maskedFill(scores, m4, -1e9);
        }
        let w = X.softmax(scores, 3);
        // F-025: an entirely blocked row has zero context/gradient, as in SDPA,
        // not the uniform distribution produced by softmax of a finite sentinel.
        for (const mask of masks) w = X.maskedFill(w, mask, 0);
        const o = X.reshape(X.transpose(X.matmul(w, v), 1, 2), [B, Tq, D]);
        set(0, X.matmul(o, P(3)));
        return;
      }
      case "relu":
        set(0, X.relu(ins[0]));
        return;
      case "gelu":
        set(0, X.gelu(ins[0]));
        return;
      case "silu":
        set(0, X.silu(ins[0]));
        return;
      case "sigmoid":
        set(0, X.sigmoid(ins[0]));
        return;
      case "tanh":
        set(0, X.tanhT(ins[0]));
        return;
      case "exp":
        set(0, X.expT(ins[0]));
        return;
      case "log":
        set(0, X.logT(ins[0]));
        return;
      case "sqrt":
        set(0, X.sqrtT(ins[0]));
        return;
      case "rsqrt":
        set(0, X.rsqrtT(ins[0]));
        return;
      case "abs":
        set(0, X.absT(ins[0]));
        return;
      case "neg":
        set(0, X.neg(ins[0]));
        return;
      case "softmax":
        set(0, X.softmax(ins[0], Number(n.attrs.axis ?? ins[0].shape.length - 1)));
        return;
      case "log_softmax":
        set(0, X.logSoftmax(ins[0], Number(n.attrs.axis ?? ins[0].shape.length - 1)));
        return;
      case "matmul":
        set(0, X.matmul(ins[0], ins[1]));
        return;
      case "transpose":
        set(0, X.transpose(ins[0], Number(n.attrs.i ?? 0), Number(n.attrs.j ?? 1)));
        return;
      case "reshape":
        set(0, X.reshape(ins[0], oshape));
        return;
      case "concat":
        set(0, X.concat(ins, Number(n.attrs.axis ?? 1)));
        return;
      case "stack":
        set(0, X.stackT(ins, Number(n.attrs.axis ?? 0)));
        return;
      case "mean":
      case "sum":
      case "max":
      case "min":
      case "argmax": {
        const ax = n.attrs.axis === null || n.attrs.axis === undefined ? null : Number(n.attrs.axis);
        set(0, X.reduce(ins[0], ax, Boolean(n.attrs.keep), n.op as "sum"));
        return;
      }
      case "masked_fill":
        set(0, X.maskedFill(ins[0], ins[1], Number(n.attrs.value ?? 0)));
        return;
      case "causal_mask": {
        const nn = evalDim(n.attrs.n as DimExpr, this.dimEnv) ?? 1;
        const m = X.zeros([nn, nn]);
        for (let i = 0; i < nn; i++) for (let j = 0; j < nn; j++) m.data[i * nn + j] = j > i ? 1 : 0;
        set(0, m);
        return;
      }
      case "one_hot": {
        const k = evalDim(n.attrs.classes as DimExpr, this.dimEnv) ?? 2;
        const src = ins[0];
        const out = X.zeros([...src.shape, k]);
        for (let i = 0; i < src.size; i++) {
          const c = X.classIndex(src.data[i], k);
          out.data[i * k + c] = 1;
        }
        set(0, out);
        return;
      }
      case "stop_grad":
        set(0, X.detach(ins[0]));
        return;
      case "detach_kind":
        set(0, ins[0]);
        return;
      case "slice": {
        if (n.attrs.dynamic) {
          const kinds = n.attrs.kinds as string[];
          const starts = n.attrs.from as DimExpr[], ends = n.attrs.to as DimExpr[];
          const openTo = n.attrs.openTo as number[];
          const axes: number[] = [], from: number[] = [], to: number[] = [], drop: number[] = [];
          let axis = 0;
          for (let i = 0; i < kinds.length; i++) {
            if (kinds[i] === "ellipsis") { axis += ins[0].shape.length - (kinds.length - 1); continue; }
            const extent = ins[0].shape[axis];
            if (extent === undefined || axis < 0) throw new Error("too many indices for runtime slice rank");
            const start = kinds[i] === "all" ? 0 : evalDim(starts[i], this.dimEnv);
            const end = kinds[i] === "index" && start !== null ? start + 1 : openTo[i] ? extent : evalDim(ends[i], this.dimEnv);
            if (start === null || end === null || start < 0 || end > extent || start > end)
              throw new Error("dynamic slice bounds are unresolved or out of range");
            axes.push(axis++); from.push(start); to.push(end); drop.push(kinds[i] === "index" ? 1 : 0);
          }
          set(0, X.sliceT(ins[0], axes, from, to, drop));
          return;
        }
        const axes = (n.attrs.axes as number[]) ?? [];
        const from = ((n.attrs.from as DimExpr[]) ?? []).map((d) => evalDim(d, this.dimEnv) ?? 0);
        const to = ((n.attrs.to as DimExpr[]) ?? []).map((d) => evalDim(d, this.dimEnv) ?? 1);
        // bounds are checked here too: the checker carries symbolic ranges as
        // constraints, but a slice must never read outside its operand (F-011)
        axes.forEach((ax, i) => {
          const extent = ins[0].shape[ax];
          if (from[i] < 0 || to[i] > extent || from[i] > to[i])
            throw new Error(`slice [${from[i]}:${to[i]}] is out of bounds for axis ${ax} of extent ${extent}`);
        });
        set(0, X.sliceT(ins[0], axes, from, to, (n.attrs.drop as number[]) ?? []));
        return;
      }
      case "cross_entropy":
        set(0, X.crossEntropy(ins[0], ins[1]));
        return;
      case "mse":
        set(0, X.mseLoss(ins[0], ins[1]));
        return;
      case "bce":
        set(0, X.bceLoss(ins[0], ins[1]));
        return;
      case "cosine_similarity": {
        const ax = Number(n.attrs.axis ?? ins[0].shape.length - 1);
        const dot = X.reduce(X.mul(ins[0], ins[1]), ax, false, "sum");
        const na = X.sqrtT(X.reduce(X.mul(ins[0], ins[0]), ax, false, "sum"));
        const nb = X.sqrtT(X.reduce(X.mul(ins[1], ins[1]), ax, false, "sum"));
        set(0, X.div(dot, X.add(X.mul(na, nb), X.scalarT(1e-8))));
        return;
      }
      case "l2":
        set(0, X.reduce(X.mul(ins[0], ins[0]), null, false, "sum"));
        return;
      case "residual": {
        const regions = n.regions ?? [];
        this.evalNodes(regions[0].nodes);
        set(0, this.get(regions[0].results[0]));
        if (regions[1]) {
          this.evalNodes(regions[1].nodes);
          set(1, this.get(regions[1].results[0]));
        }
        return;
      }
      case "parallel": {
        // every branch consumes the *same* incoming value
        (n.regions ?? []).forEach((r, i) => {
          this.evalNodes(r.nodes);
          set(i, this.get(r.results[0]));
        });
        return;
      }
      case "static_repeat": {
        const regions = n.regions ?? [];
        let lastId = "";
        for (const r of regions) {
          this.evalNodes(r.nodes);
          lastId = r.results[0] ?? lastId;
        }
        if (lastId && n.outputs.length) set(0, this.get(lastId));
        // `for 0:` is the identity: the incoming value passes through (F-006)
        else if (n.outputs.length && ins[0]) set(0, ins[0]);
        return;
      }
      case "apply": {
        const r = (n.regions ?? [])[0];
        if (!r) return;
        this.evalNodes(r.nodes);
        r.results.forEach((res, i) => set(i, this.get(res)));
        return;
      }
      case "scan": {
        const r = (n.regions ?? [])[0];
        const axis = Number(n.attrs.axis ?? 1);
        const stepId = String(n.attrs.step_value ?? "");
        const carryIds = (n.attrs.carry_values as string[]) ?? [];
        const x = ins[0];
        const len = x.shape[axis];
        let carries: T[] = carryIds.map((cid, i) => {
          const t = this.mod.values.get(cid)?.type;
          const shape = t && isTensor(t) ? this.dims(t.shape) : [1];
          const init = ((n.attrs.init as string[]) ?? [])[i] ?? "zeros";
          return init === "ones" ? X.full(shape, 1) : X.zeros(shape);
        });
        const outs: T[] = [];
        for (let t = 0; t < len; t++) {
          const from = new Array(x.shape.length).fill(0);
          const to = x.shape.slice();
          from[axis] = t;
          to[axis] = t + 1;
          const axesAll = x.shape.map((_, i) => i);
          const step = X.sliceT(
            x,
            axesAll,
            from,
            to,
            axesAll.map((i) => (i === axis ? 1 : 0))
          );
          this.env.set(stepId, step);
          carryIds.forEach((cid, i) => this.env.set(cid, carries[i]));
          this.evalNodes(r.nodes);
          const results = r.results.map((res) => this.get(res));
          outs.push(results[0]);
          carries = results;
        }
        set(0, X.stackT(outs, axis));
        carries.forEach((c, i) => {
          if (n.outputs[i + 1]) set(i + 1, c);
        });
        return;
      }
      default: {
        const custom = this.mod.customOps.find((c) => c.name === n.op);
        if (custom) {
          this.notes.push(
            `custom op '${n.op}' has no reference implementation; the executor substituted an identity/zero tensor of the declared shape`
          );
          set(0, ins[0] && ins[0].shape.join() === oshape.join() ? ins[0] : X.zeros(oshape));
          return;
        }
        this.errors.push(`reference backend has no implementation for '${n.op}'`);
        set(0, X.zeros(oshape));
      }
    }
  }
}

// ------------------------------------------------------------------ data

export function syntheticField(shape: number[], kind: string, vocab: number, idx: number): T {
  const t = X.zeros(shape);
  if (kind === "Tokens" || kind === "Class") {
    for (let i = 0; i < t.size; i++) t.data[i] = (idx * 7 + i * 3) % Math.max(1, vocab);
  } else if (kind === "Mask") {
    // F-024: a Mask is boolean, not a normally distributed feature vector.
    for (let i = 0; i < t.size; i++) t.data[i] = X.rand() < 0.25 ? 1 : 0;
  } else {
    for (let i = 0; i < t.size; i++) t.data[i] = X.randn() * 0.6 + (idx % 3) - 1;
  }
  return t;
}

// ------------------------------------------------------------------ driver

export function runProgram(mod: IRModule, opts: RunOptions = {}): RunReport {
  const rt = new Runtime(mod, opts);
  rt.allocate();
  const report: RunReport = {
    dims: [...rt.dimEnv.entries()].map(([name, value]) => ({
      name,
      value,
      source: mod.dims.find((d) => d.name === name && d.value) ? "static" : "runtime binding",
    })),
    forward: [],
    losses: [],
    finalMetrics: {},
    paramCount: rt.paramCount(),
    notes: [],
    errors: [],
    approximations: [],
    gradCoverage: [],
    phases: [],
    events: [],
  };

  const violations = rt.checkConstraints();
  if (violations.length) {
    report.errors.push(...violations, "refusing to execute: the bound dimensions violate constraints the checker carried");
    report.notes.push(...rt.notes);
    report.approximations.push(...rt.approx);
    return report;
  }

  const models = mod.graphs.filter((g) => g.kind === "model");
  // ---- forward pass on every model
  for (const g of models) {
    X.beginTape();
    rt.env = new Map();
    rt.training = false;
    const inputs = g.inputs.map((iv, i) => {
      const t = iv.type as TensorType;
      const shape = rt.dims(t.shape);
      return syntheticField(shape, t.kind, guessVocab(mod, t), i + 1);
    });
    const t0 = performance.now();
    let outs: T[] = [];
    try {
      outs = rt.evalGraph(g, inputs);
    } catch (e) {
      rt.errors.push(`${g.name}: ${(e as Error).message}`);
    }
    report.forward.push({
      model: g.name,
      inputs: g.inputs.map((iv) => `${iv.name ?? "in"}: [${rt.dims((iv.type as TensorType).shape).join(", ")}]`),
      output: outs.map((o) => `[${o.shape.join(", ")}]`).join(" , "),
      ms: Math.round((performance.now() - t0) * 10) / 10,
    });
  }

  // ---- training plan
  const plan = mod.plans[0];
  if (plan && plan.losses.length && models.length) {
    // one source of truth for the batch axis: the runtime binding of B
    const batch = rt.dimEnv.get("B") ?? Math.min(mod.data.find((d) => d.name === plan.data)?.batch ?? 4, 4);
    const maxSteps = opts.maxSteps ?? 24;
    let step = 0;
    const frozen = new Set<string>();
    const everGrad = new Set<string>();
    const lrOf = new Map<string, number>();
    for (const o of plan.optimizers) for (const p of o.params) lrOf.set(p, o.lr);

    const objectiveOf = (name: string) => mod.objectives.find((o) => o.name === name);
    const modelGraph = (alias: string) => {
      const m = plan.models.find((x) => x.alias === alias);
      return mod.graphs.find((g) => g.name === (m ? m.model : alias));
    };

    const clipNorm = plan.settings.clip_grad ? parseFloat(plan.settings.clip_grad) : 0;
    const trackStates = plan.tracks.map((t) => ({
      track: t,
      states: mod.states.filter((s) => s.owner === t.name),
    }));
    // `init: copy` tracks start as a copy of the parameter they shadow, not zeros
    for (const { states } of trackStates)
      for (const s of states) {
        const src = rt.params.get(s.id.slice(s.owner.length + 1));
        if (!src) continue;
        const copy = X.zeros(src.shape);
        copy.data.set(src.data);
        rt.states.set(s.id, copy);
      }

    /**
     * Evaluate one objective-port binding. Nested model calls (`D(G(z))`,
     * `Enc(x)[1]`) recurse so that every model on the path is on the tape and
     * receives gradient (H-007); one synthetic tensor per field name keeps
     * `augment(x)` and `x` consistent within a step.
     */
    const fieldCache = new Map<string, T>();
    // F-023: projections of one tuple-valued call share its stochastic/stateful forward.
    // Unindexed applications remain independent; the cache lives for ONE loss evaluation.
    const projectionCache = new Map<string, T[]>();
    const evalBinding = (text: string, want: TensorType | null): T => {
      const src = text.trim();
      const call = /^(\w+)\((.*)\)(?:\[(\d+)\])?$/.exec(src);
      if (call && modelGraph(call[1])) {
        const key = `${call[1]}(${call[2]})`;
        const cached = call[3] !== undefined ? projectionCache.get(key) : undefined;
        if (cached) return cached[Number(call[3])];
        const g = modelGraph(call[1])!;
        const argTexts = splitTopLevel(call[2]);
        const args = g.inputs.map((iv, i) => {
          const t = iv.type as TensorType;
          const argText = argTexts[i];
          if (argText !== undefined) return evalBinding(argText, t);
          return syntheticField([batch, ...rt.dims(t.shape.slice(1))], t.kind, guessVocab(mod, t), step + i + 1);
        });
        const outs = rt.evalGraph(g, args);
        if (call[3] !== undefined) projectionCache.set(key, outs);
        return outs[call[3] ? parseInt(call[3], 10) : 0] ?? outs[0];
      }
      // a data field (or an expression over one): one deterministic tensor per name per step
      const key = src.replace(/^\w+\((.*)\)$/, "$1");
      const cached = fieldCache.get(key);
      if (cached && want && cached.shape.length === want.shape.length) return cached;
      const shape = want ? [batch, ...rt.dims(want.shape.slice(1))] : [batch];
      const t = syntheticField(shape, want?.kind ?? "Tensor", want ? guessVocab(mod, want) : 0, step + 1 + key.length);
      fieldCache.set(key, t);
      return t;
    };

    const evalLoss = (loss: (typeof plan.losses)[number], training: boolean): T | null => {
      const obj = objectiveOf(loss.objective);
      if (!obj) return null;
      X.beginTape();
      rt.env = new Map();
      rt.training = training;
      fieldCache.clear();
      projectionCache.clear();
      const portTensors: T[] = obj.inputs.map((b) => {
        const binding = loss.bindings.find((x) => x.port === b.name);
        const want = b.type as TensorType;
        if (binding && (binding.kind === "model" || binding.kind === "expr")) return evalBinding(binding.text, want);
        return evalBinding(binding?.text ?? b.name ?? "field", want);
      });
      return rt.evalGraph(obj.graph, portTensors)[0];
    };

    const fireEvents = (events: IREvent[], phaseName: string, stepInPhase: number) => {
      for (const ev of events) {
        const period = ev.unit === "epochs" ? ev.every * STEPS_PER_EPOCH : ev.every;
        if (period <= 0 || (stepInPhase + 1) % period !== 0) continue;
        for (const a of ev.actions) {
          if (a.kind === "validate") {
            const vals: Record<string, number> = {};
            for (const l of plan.losses) {
              const t = evalLoss(l, false);
              if (t) vals[`val_${l.name}`] = t.data[0] * l.weight;
            }
            Object.assign(lastVal, vals);
            report.events.push({
              step,
              phase: phaseName,
              action: "validate",
              detail: Object.entries(vals).map(([k, v]) => `${k}=${(Math.round(v * 1e4) / 1e4).toFixed(4)}`).join(" "),
            });
          } else if (a.kind === "checkpoint") {
            const nState = mod.states.filter((s) => s.checkpointed).length;
            report.events.push({
              step,
              phase: phaseName,
              action: "checkpoint",
              detail: `${mod.params.length} params, ${nState} states, ${rt.optState.size} optimizer slots`,
            });
          } else {
            report.events.push({ step, phase: phaseName, action: a.kind, detail: a.args });
          }
        }
      }
    };

    const lastVal: Record<string, number> = {};
    const untilMet = (phase: IRPhase, train: Record<string, number>): boolean => {
      if (!phase.until) return false;
      const { metric, cmp, value } = phase.until;
      let cur: number | undefined;
      if (metric.startsWith("val_")) cur = lastVal[metric];
      else if (metric.startsWith("train_")) cur = train[metric.slice(6)];
      else cur = train[metric] ?? lastVal[`val_${metric}`];
      if (cur === undefined) {
        const note = `phase '${phase.name}': \`until ${metric}\` names a metric the reference runtime does not produce; the phase runs to its epoch/step budget`;
        if (!report.approximations.includes(note)) report.approximations.push(note);
        return false;
      }
      switch (cmp) {
        case "<":
          return cur < value;
        case "<=":
          return cur <= value;
        case ">":
          return cur > value;
        case ">=":
          return cur >= value;
        case "==":
          return cur === value;
        default:
          return false;
      }
    };

    for (const phase of plan.phases) {
      // freeze/unfreeze persist across phases (the emitter's `requires_grad_` does too)
      for (const region of phase.frozen)
        for (const p of paramsOfRegion(mod, plan, region)) frozen.add(p.id);
      for (const region of phase.unfrozen)
        for (const p of paramsOfRegion(mod, plan, region)) frozen.delete(p.id);
      const phaseLr = new Map(lrOf);
      for (const ov of phase.lrOverrides)
        for (const p of paramsOfRegion(mod, plan, ov.region)) phaseLr.set(p.id, ov.lr);
      const budget = phase.steps ?? phase.epochs * STEPS_PER_EPOCH;
      const totalSteps = Math.min(budget, maxSteps);
      let stoppedBy: RunReport["phases"][number]["stoppedBy"] = phase.steps !== null ? "steps" : "epochs";
      if (totalSteps < budget) stoppedBy = "maxSteps";
      let lrStart = 1;
      let lrEnd = 1;
      let ran = 0;
      for (let s = 0; s < totalSteps; s++) {
        const mult = scheduleMultiplier(phase.schedule, s, budget);
        if (s === 0) lrStart = mult;
        lrEnd = mult;
        const values: Record<string, number> = {};
        for (const upd of phase.updates) {
          for (let rep = 0; rep < upd.times; rep++) {
            const loss = plan.losses.find((l) => l.name === upd.loss);
            if (!loss) continue;
            for (const p of rt.params.values()) p.g = null;
            const lossT = evalLoss(loss, true);
            if (!lossT) continue;
            values[loss.name] = lossT.data[0] * loss.weight;
            X.backward(lossT);
            // coverage is "ever reached by any loss", not "reached by the last update" (H-002)
            for (const [pid, p] of rt.params) if (p.g) everGrad.add(pid);
            // optimizer step
            for (const oname of upd.optimizers) {
              const o = plan.optimizers.find((x) => x.name === oname);
              if (!o) continue;
              if (clipNorm > 0) clipGradNorm(rt, o.params, clipNorm);
              for (const pid of o.params) {
                if (frozen.has(pid)) continue;
                const p = rt.params.get(pid);
                const meta = mod.params.find((x) => x.id === pid);
                if (!p || !p.g || !meta?.trainable) continue;
                applyUpdate(rt, pid, p, o.kind, (phaseLr.get(pid) ?? o.lr) * mult, o.args);
              }
            }
            // tracked shadows (`track t = ema(region, rate: r)`) follow every optimizer step
            for (const { track, states } of trackStates) {
              if (track.kind !== "ema") continue;
              const decay = emaDecay(`ema(${track.args})`);
              for (const st of states) {
                const cur = rt.states.get(st.id);
                const src = rt.params.get(st.id.slice(track.name.length + 1));
                if (!cur || !src) continue;
                for (let i = 0; i < cur.size; i++) cur.data[i] = decay * cur.data[i] + (1 - decay) * src.data[i];
              }
            }
          }
        }
        report.losses.push({ step, phase: phase.name, values });
        // phase events count steps within the phase; plan-level events count
        // steps across the whole run (the emitter's `step % n` convention)
        fireEvents(phase.events, phase.name, s);
        fireEvents(plan.events, phase.name, step);
        step++;
        ran++;
        if (untilMet(phase, values)) {
          stoppedBy = "until";
          break;
        }
      }
      report.phases.push({ name: phase.name, steps: ran, stoppedBy, lrStart, lrEnd });
    }
    for (const [k, v] of Object.entries(lastVal)) report.finalMetrics[k] = Math.round(v * 1e4) / 1e4;
    // gradient coverage report
    for (const p of mod.params) {
      const reached = everGrad.has(p.id);
      const covered = plan.optimizers.some((o) => o.params.includes(p.id));
      report.gradCoverage.push({
        param: p.id,
        updated: Boolean(reached && covered && p.trainable),
        reason: !p.trainable
          ? "declared frozen"
          : !covered
          ? "not claimed by any optimizer"
          : !reached
          ? "no gradient path reached this parameter"
          : undefined,
      });
    }
    const lastLosses = report.losses[report.losses.length - 1]?.values ?? {};
    for (const [k, v] of Object.entries(lastLosses)) report.finalMetrics[`final_${k}`] = Math.round(v * 1e4) / 1e4;
    const first = report.losses[0]?.values ?? {};
    for (const [k, v] of Object.entries(first)) report.finalMetrics[`initial_${k}`] = Math.round(v * 1e4) / 1e4;
  } else if (!plan) {
    report.notes.push("no `train` declaration — only forward execution was performed");
  }

  report.notes.push(...rt.notes);
  report.errors.push(...rt.errors);
  report.approximations.push(...rt.approx);
  report.approximations.push(
    "the reference backend trains on deterministic synthetic tensors; it validates shapes, wiring, gradient flow and lifecycle, not accuracy"
  );
  return report;
}

export function guessVocab(mod: IRModule, t: TensorType): number {
  if (t.kind !== "Tokens" && t.kind !== "Class") return 0;
  // F-026: synthetic labels must be valid, not rely on silent index clamping.
  // This is a conservative fixture domain, NOT an inferred target relationship.
  const env = new Map<string, number>();
  for (const d of mod.dims) { const v = d.value && evalDim(d.value, env); if (v !== null && v !== undefined) env.set(d.name, v); }
  const bounds: number[] = [];
  const add = (d: DimExpr | undefined) => { const v = d && evalDim(d, env); bounds.push(v && v > 0 ? v : 1); };
  for (const p of mod.params) if (p.kind === "embedding") add(p.shape[0]);
  const walk = (nodes: IRNode[]) => { for (const n of nodes) {
    if (n.op === "one_hot") add(n.attrs.classes as DimExpr);
    if (n.op === "cross_entropy") {
      const ty = mod.values.get(n.inputs[0])?.type;
      if (ty && isTensor(ty)) add(ty.shape[ty.shape.length - 1]);
    }
    for (const r of n.regions ?? []) walk(r.nodes);
  } };
  for (const g of [...mod.graphs, ...mod.objectives.map(o => o.graph)]) walk(g.nodes);
  return bounds.length ? Math.min(...bounds) : 1;
}

/** `Net.encoder` → the parameters whose owner path lies under that region (shared with the emitter) */
export function paramsOfRegion(mod: IRModule, plan: { models: { alias: string; model: string }[] }, region: string): IRParam[] {
  if (region === "*") {
    const roots = new Set(plan.models.map((m) => m.model));
    return mod.params.filter((p) => roots.has(p.owner.split("/")[0]));
  }
  const parts = region.split(".");
  const m = plan.models.find((x) => x.alias === parts[0]);
  const prefix = [m ? m.model : parts[0], ...parts.slice(1)].join("/");
  return mod.params.filter((p) => p.owner === prefix || p.owner.startsWith(prefix + "/") || p.owner.startsWith(prefix + "."));
}

/** split `a, f(b, c), d` at top-level commas */
function splitTopLevel(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of s) {
    if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") depth--;
    if (ch === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** `clip_grad n` — global-norm clipping over the optimizer's parameter set */
function clipGradNorm(rt: Runtime, ids: string[], maxNorm: number) {
  let sq = 0;
  for (const id of ids) {
    const g = rt.params.get(id)?.g;
    if (g) for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
  }
  const norm = Math.sqrt(sq);
  if (norm <= maxNorm || norm === 0) return;
  const k = maxNorm / (norm + 1e-6);
  for (const id of ids) {
    const g = rt.params.get(id)?.g;
    if (g) for (let i = 0; i < g.length; i++) g[i] *= k;
  }
}

function applyUpdate(rt: Runtime, id: string, p: T, kind: string, lr: number, args: Record<string, number>) {
  const g = p.g!;
  if (kind === "sgd") {
    const mom = args.momentum ?? 0;
    let st = rt.optState.get(id);
    if (!st) {
      st = { m: new Float32Array(p.size), v: new Float32Array(0), t: 0 };
      rt.optState.set(id, st);
    }
    for (let i = 0; i < p.size; i++) {
      st.m[i] = mom * st.m[i] + g[i];
      p.data[i] -= lr * st.m[i];
    }
    return;
  }
  // adam / adamw
  let st = rt.optState.get(id);
  if (!st) {
    st = { m: new Float32Array(p.size), v: new Float32Array(p.size), t: 0 };
    rt.optState.set(id, st);
  }
  st.t++;
  const b1 = args.beta1 ?? 0.9;
  const b2 = args.beta2 ?? 0.999;
  const eps = args.eps ?? 1e-8;
  const wd = kind === "adamw" ? args.wd ?? 0.01 : 0;
  const c1 = 1 - Math.pow(b1, st.t);
  const c2 = 1 - Math.pow(b2, st.t);
  for (let i = 0; i < p.size; i++) {
    st.m[i] = b1 * st.m[i] + (1 - b1) * g[i];
    st.v[i] = b2 * st.v[i] + (1 - b2) * g[i] * g[i];
    const mh = st.m[i] / c1;
    const vh = st.v[i] / c2;
    p.data[i] -= lr * (mh / (Math.sqrt(vh) + eps) + wd * p.data[i]);
  }
}
