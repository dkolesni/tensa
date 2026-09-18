/**
 * TENSA — reference tensor engine with reverse-mode autodiff.
 * Deliberately small and readable: correctness and inspectability over speed.
 */

export class T {
  shape: number[];
  data: Float32Array;
  g: Float32Array | null = null;
  req: boolean;
  back: (() => void) | null = null;
  constructor(shape: number[], data?: Float32Array, req = false) {
    this.shape = shape;
    this.data = data ?? new Float32Array(shape.reduce((a, b) => a * b, 1));
    this.req = req;
  }
  get size() {
    return this.data.length;
  }
  ensureGrad() {
    if (!this.g) this.g = new Float32Array(this.size);
    return this.g;
  }
}

let tape: T[] = [];
let taping = true;
export function beginTape() {
  tape = [];
  taping = true;
}
export function setTaping(on: boolean) {
  taping = on;
}
export function backward(loss: T) {
  loss.ensureGrad()[0] = 1;
  for (let i = tape.length - 1; i >= 0; i--) {
    const n = tape[i];
    if (n.back && n.g) n.back();
  }
}
function record(t: T, back: () => void, parents: T[]) {
  if (taping && parents.some((p) => p.req)) {
    t.req = true;
    t.back = back;
    tape.push(t);
  }
  return t;
}

export function strides(shape: number[]): number[] {
  const s = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1];
  return s;
}
export function numel(shape: number[]) {
  return shape.reduce((a, b) => a * b, 1);
}

export function zeros(shape: number[], req = false) {
  return new T(shape, undefined, req);
}
export function full(shape: number[], v: number, req = false) {
  const t = new T(shape, undefined, req);
  t.data.fill(v);
  return t;
}
export function fromArray(shape: number[], arr: number[] | Float32Array, req = false) {
  return new T(shape, arr instanceof Float32Array ? arr : Float32Array.from(arr), req);
}
export function scalarT(v: number) {
  return fromArray([], [v]);
}

// deterministic RNG so runs are reproducible
let seed = 12345;
export function setSeed(s: number) {
  seed = s >>> 0;
}
export function rand(): number {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}
export function randn(): number {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** H-010: sampling is independent of the template values and of train/eval mode. */
export function randnLike(x: T): T {
  const out = new T([...x.shape]);
  for (let i = 0; i < out.size; i++) out.data[i] = randn();
  return out;
}

// ------------------------------------------------------------------ broadcasting

function bshape(a: number[], b: number[]): number[] {
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1;
    const y = b[b.length - n + i] ?? 1;
    out.push(Math.max(x, y));
  }
  return out;
}

function bIndex(outIdx: number[], shape: number[]): number {
  const st = strides(shape);
  let idx = 0;
  const off = outIdx.length - shape.length;
  for (let i = 0; i < shape.length; i++) {
    const dimIdx = shape[i] === 1 ? 0 : outIdx[off + i];
    idx += dimIdx * st[i];
  }
  return idx;
}

function unravel(i: number, shape: number[], out: number[]) {
  for (let d = shape.length - 1; d >= 0; d--) {
    out[d] = i % shape[d];
    i = Math.floor(i / shape[d]);
  }
}

function binop(a: T, b: T, f: (x: number, y: number) => number, df: (x: number, y: number, g: number) => [number, number]): T {
  const shape = bshape(a.shape, b.shape);
  const out = new T(shape);
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < out.size; i++) {
    unravel(i, shape, idx);
    out.data[i] = f(a.data[bIndex(idx, a.shape)], b.data[bIndex(idx, b.shape)]);
  }
  return record(
    out,
    () => {
      const g = out.g!;
      const ga = a.req ? a.ensureGrad() : null;
      const gb = b.req ? b.ensureGrad() : null;
      const id = new Array(shape.length).fill(0);
      for (let i = 0; i < out.size; i++) {
        unravel(i, shape, id);
        const ia = bIndex(id, a.shape);
        const ib = bIndex(id, b.shape);
        const [da, db] = df(a.data[ia], b.data[ib], g[i]);
        if (ga) ga[ia] += da;
        if (gb) gb[ib] += db;
      }
    },
    [a, b]
  );
}

export const add = (a: T, b: T) => binop(a, b, (x, y) => x + y, (_x, _y, g) => [g, g]);
export const sub = (a: T, b: T) => binop(a, b, (x, y) => x - y, (_x, _y, g) => [g, -g]);
export const mul = (a: T, b: T) => binop(a, b, (x, y) => x * y, (x, y, g) => [g * y, g * x]);
export const div = (a: T, b: T) => binop(a, b, (x, y) => x / y, (x, y, g) => [g / y, (-g * x) / (y * y)]);

function unop(a: T, f: (x: number) => number, df: (x: number, y: number, g: number) => number): T {
  const out = new T(a.shape);
  for (let i = 0; i < a.size; i++) out.data[i] = f(a.data[i]);
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      for (let i = 0; i < a.size; i++) ga[i] += df(a.data[i], out.data[i], out.g![i]);
    },
    [a]
  );
}

export const relu = (a: T) => unop(a, (x) => (x > 0 ? x : 0), (x, _y, g) => (x > 0 ? g : 0));
export const sigmoid = (a: T) => unop(a, (x) => 1 / (1 + Math.exp(-x)), (_x, y, g) => g * y * (1 - y));
export const tanhT = (a: T) => unop(a, Math.tanh, (_x, y, g) => g * (1 - y * y));
export const expT = (a: T) => unop(a, Math.exp, (_x, y, g) => g * y);
export const logT = (a: T) => unop(a, (x) => Math.log(Math.max(x, 1e-12)), (x, _y, g) => g / Math.max(x, 1e-12));
export const sqrtT = (a: T) => unop(a, (x) => Math.sqrt(Math.max(x, 0)), (_x, y, g) => (y > 0 ? g / (2 * y) : 0));
export const rsqrtT = (a: T) => unop(a, (x) => 1 / Math.sqrt(Math.max(x, 1e-12)), (x, y, g) => (-0.5 * g * y) / Math.max(x, 1e-12));
export const absT = (a: T) => unop(a, Math.abs, (x, _y, g) => (x >= 0 ? g : -g));
export const neg = (a: T) => unop(a, (x) => -x, (_x, _y, g) => -g);
export const gelu = (a: T) =>
  unop(
    a,
    (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
    (x, _y, g) => {
      const t = Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x));
      const d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * x * x);
      return g * d;
    }
  );
export const silu = (a: T) =>
  unop(
    a,
    (x) => x / (1 + Math.exp(-x)),
    (x, _y, g) => {
      const s = 1 / (1 + Math.exp(-x));
      return g * (s + x * s * (1 - s));
    }
  );
export const scaleT = (a: T, k: number) => unop(a, (x) => x * k, (_x, _y, g) => g * k);

// ------------------------------------------------------------------ shape ops

export function reshape(a: T, shape: number[]): T {
  const n = shape.reduce((x, y) => x * y, 1);
  if (n !== a.size) throw new Error(`reshape: cannot view ${a.size} elements [${a.shape.join(", ")}] as [${shape.join(", ")}] (${n} elements)`);
  const out = new T(shape, a.data.slice());
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      for (let i = 0; i < a.size; i++) ga[i] += out.g![i];
    },
    [a]
  );
}

export function transpose(a: T, i: number, j: number): T {
  const shape = a.shape.slice();
  [shape[i], shape[j]] = [shape[j], shape[i]];
  const out = new T(shape);
  const sa = strides(a.shape);
  const idx = new Array(shape.length).fill(0);
  for (let k = 0; k < out.size; k++) {
    unravel(k, shape, idx);
    const src = idx.slice();
    [src[i], src[j]] = [src[j], src[i]];
    let off = 0;
    for (let d = 0; d < src.length; d++) off += src[d] * sa[d];
    out.data[k] = a.data[off];
  }
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      const id = new Array(shape.length).fill(0);
      for (let k = 0; k < out.size; k++) {
        unravel(k, shape, id);
        const src = id.slice();
        [src[i], src[j]] = [src[j], src[i]];
        let off = 0;
        for (let d = 0; d < src.length; d++) off += src[d] * sa[d];
        ga[off] += out.g![k];
      }
    },
    [a]
  );
}

export function sliceT(a: T, axes: number[], from: number[], to: number[], drop: number[]): T {
  const shape = a.shape.slice();
  const starts = new Array(a.shape.length).fill(0);
  for (let i = 0; i < axes.length; i++) {
    starts[axes[i]] = from[i];
    shape[axes[i]] = to[i] - from[i];
  }
  const dropAxes = axes.filter((_, i) => drop[i] === 1).sort((x, y) => y - x);
  const outShape = shape.slice();
  for (const d of dropAxes) outShape.splice(d, 1);
  const out = new T(outShape);
  const sa = strides(a.shape);
  const idx = new Array(shape.length).fill(0);
  for (let k = 0; k < out.size; k++) {
    unravel(k, shape, idx);
    let off = 0;
    for (let d = 0; d < shape.length; d++) off += (idx[d] + starts[d]) * sa[d];
    out.data[k] = a.data[off];
  }
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      const id = new Array(shape.length).fill(0);
      for (let k = 0; k < out.size; k++) {
        unravel(k, shape, id);
        let off = 0;
        for (let d = 0; d < shape.length; d++) off += (id[d] + starts[d]) * sa[d];
        ga[off] += out.g![k];
      }
    },
    [a]
  );
}

export function concat(ts: T[], axis: number): T {
  const shape = ts[0].shape.slice();
  shape[axis] = ts.reduce((s, t) => s + t.shape[axis], 0);
  const out = new T(shape);
  const outer = shape.slice(0, axis).reduce((a, b) => a * b, 1);
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  let axOff = 0;
  const offsets: number[] = [];
  for (const t of ts) {
    offsets.push(axOff);
    for (let o = 0; o < outer; o++)
      for (let c = 0; c < t.shape[axis]; c++)
        for (let i = 0; i < inner; i++)
          out.data[(o * shape[axis] + axOff + c) * inner + i] = t.data[(o * t.shape[axis] + c) * inner + i];
    axOff += t.shape[axis];
  }
  return record(
    out,
    () => {
      ts.forEach((t, ti) => {
        if (!t.req) return;
        const gt = t.ensureGrad();
        for (let o = 0; o < outer; o++)
          for (let c = 0; c < t.shape[axis]; c++)
            for (let i = 0; i < inner; i++)
              gt[(o * t.shape[axis] + c) * inner + i] += out.g![(o * shape[axis] + offsets[ti] + c) * inner + i];
      });
    },
    ts
  );
}

export function stackT(ts: T[], axis: number): T {
  const expanded = ts.map((t) => {
    const s = t.shape.slice();
    s.splice(axis, 0, 1);
    return reshape(t, s);
  });
  return concat(expanded, axis);
}

// ------------------------------------------------------------------ reductions

export function reduce(a: T, axis: number | null, keep: boolean, kind: "sum" | "mean" | "max" | "min" | "argmax"): T {
  if (axis === null) {
    const out = new T([]);
    let acc = kind === "max" ? -Infinity : kind === "min" ? Infinity : 0;
    let arg = 0;
    for (let i = 0; i < a.size; i++) {
      const v = a.data[i];
      if (kind === "max" || kind === "argmax") {
        if (v > acc) {
          acc = v;
          arg = i;
        }
      } else if (kind === "min") acc = Math.min(acc, v);
      else acc += v;
    }
    if (kind === "mean") acc /= a.size;
    out.data[0] = kind === "argmax" ? arg : acc;
    if (kind === "argmax") return out;
    return record(
      out,
      () => {
        const ga = a.ensureGrad();
        const g = out.g![0];
        for (let i = 0; i < a.size; i++) {
          if (kind === "sum") ga[i] += g;
          else if (kind === "mean") ga[i] += g / a.size;
          else if (a.data[i] === acc) ga[i] += g;
        }
      },
      [a]
    );
  }
  const shape = a.shape.slice();
  const n = shape[axis];
  const outShape = shape.slice();
  if (keep) outShape[axis] = 1;
  else outShape.splice(axis, 1);
  const out = new T(outShape);
  const outer = shape.slice(0, axis).reduce((x, y) => x * y, 1);
  const inner = shape.slice(axis + 1).reduce((x, y) => x * y, 1);
  const argIdx = new Int32Array(out.size);
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      let acc = kind === "max" || kind === "argmax" ? -Infinity : kind === "min" ? Infinity : 0;
      let arg = 0;
      for (let c = 0; c < n; c++) {
        const v = a.data[(o * n + c) * inner + i];
        if (kind === "max" || kind === "argmax") {
          if (v > acc) {
            acc = v;
            arg = c;
          }
        } else if (kind === "min") {
          if (v < acc) {
            acc = v;
            arg = c;
          }
        } else acc += v;
      }
      if (kind === "mean") acc /= n;
      const oi = o * inner + i;
      out.data[oi] = kind === "argmax" ? arg : acc;
      argIdx[oi] = arg;
    }
  if (kind === "argmax") return out;
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          const oi = o * inner + i;
          const g = out.g![oi];
          if (kind === "sum") for (let c = 0; c < n; c++) ga[(o * n + c) * inner + i] += g;
          else if (kind === "mean") for (let c = 0; c < n; c++) ga[(o * n + c) * inner + i] += g / n;
          else ga[(o * n + argIdx[oi]) * inner + i] += g;
        }
    },
    [a]
  );
}

export function softmax(a: T, axis: number): T {
  const n = a.shape[axis];
  const outer = a.shape.slice(0, axis).reduce((x, y) => x * y, 1);
  const inner = a.shape.slice(axis + 1).reduce((x, y) => x * y, 1);
  const out = new T(a.shape);
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      let m = -Infinity;
      for (let c = 0; c < n; c++) m = Math.max(m, a.data[(o * n + c) * inner + i]);
      let s = 0;
      for (let c = 0; c < n; c++) {
        const e = Math.exp(a.data[(o * n + c) * inner + i] - m);
        out.data[(o * n + c) * inner + i] = e;
        s += e;
      }
      for (let c = 0; c < n; c++) out.data[(o * n + c) * inner + i] /= s;
    }
  return record(
    out,
    () => {
      const ga = a.ensureGrad();
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          let dot = 0;
          for (let c = 0; c < n; c++) {
            const k = (o * n + c) * inner + i;
            dot += out.g![k] * out.data[k];
          }
          for (let c = 0; c < n; c++) {
            const k = (o * n + c) * inner + i;
            ga[k] += out.data[k] * (out.g![k] - dot);
          }
        }
    },
    [a]
  );
}

export function logSoftmax(a: T, axis: number): T {
  const sm = softmax(a, axis);
  return logT(sm);
}

// ------------------------------------------------------------------ matmul

export function matmul(a: T, b: T): T {
  const ar = a.shape.length;
  const br = b.shape.length;
  const M = a.shape[ar - 2];
  const K = a.shape[ar - 1];
  const N = b.shape[br - 1];
  const aBatch = a.shape.slice(0, -2);
  const bBatch = b.shape.slice(0, -2);
  const batchShape = bshape(aBatch, bBatch);
  const batch = numel(batchShape);
  const out = new T([...batchShape, M, N]);
  const idx = new Array(batchShape.length).fill(0);
  for (let bi = 0; bi < batch; bi++) {
    unravel(bi, batchShape, idx);
    const ao = bIndex(idx, aBatch) * M * K;
    const bo = bIndex(idx, bBatch) * K * N;
    const oo = bi * M * N;
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += a.data[ao + m * K + k] * b.data[bo + k * N + n];
        out.data[oo + m * N + n] = s;
      }
  }
  return record(
    out,
    () => {
      const ga = a.req ? a.ensureGrad() : null;
      const gb = b.req ? b.ensureGrad() : null;
      const id = new Array(batchShape.length).fill(0);
      for (let bi = 0; bi < batch; bi++) {
        unravel(bi, batchShape, id);
        const ao = bIndex(id, aBatch) * M * K;
        const bo = bIndex(id, bBatch) * K * N;
        const oo = bi * M * N;
        for (let m = 0; m < M; m++)
          for (let n = 0; n < N; n++) {
            const g = out.g![oo + m * N + n];
            if (g === 0) continue;
            for (let k = 0; k < K; k++) {
              if (ga) ga[ao + m * K + k] += g * b.data[bo + k * N + n];
              if (gb) gb[bo + k * N + n] += g * a.data[ao + m * K + k];
            }
          }
      }
    },
    [a, b]
  );
}

// ------------------------------------------------------------------ nn ops

export function embedding(table: T, ids: T): T {
  const [_v, D] = table.shape;
  void _v;
  const out = new T([...ids.shape, D]);
  for (let i = 0; i < ids.size; i++) {
    const t = Math.max(0, Math.min(table.shape[0] - 1, Math.round(ids.data[i])));
    for (let d = 0; d < D; d++) out.data[i * D + d] = table.data[t * D + d];
  }
  return record(
    out,
    () => {
      const gt = table.ensureGrad();
      for (let i = 0; i < ids.size; i++) {
        const t = Math.max(0, Math.min(table.shape[0] - 1, Math.round(ids.data[i])));
        for (let d = 0; d < D; d++) gt[t * D + d] += out.g![i * D + d];
      }
    },
    [table]
  );
}

export function conv2d(x: T, w: T, b: T | null, stride: number, pad: number): T {
  const [B, C, H, W] = x.shape;
  const [O, , KH, KW] = w.shape;
  const OH = Math.floor((H + 2 * pad - KH) / stride) + 1;
  const OW = Math.floor((W + 2 * pad - KW) / stride) + 1;
  const out = new T([B, O, OH, OW]);
  for (let n = 0; n < B; n++)
    for (let o = 0; o < O; o++)
      for (let oh = 0; oh < OH; oh++)
        for (let ow = 0; ow < OW; ow++) {
          let s = b ? b.data[o] : 0;
          for (let c = 0; c < C; c++)
            for (let kh = 0; kh < KH; kh++) {
              const ih = oh * stride + kh - pad;
              if (ih < 0 || ih >= H) continue;
              for (let kw = 0; kw < KW; kw++) {
                const iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                s += x.data[((n * C + c) * H + ih) * W + iw] * w.data[((o * C + c) * KH + kh) * KW + kw];
              }
            }
          out.data[((n * O + o) * OH + oh) * OW + ow] = s;
        }
  return record(
    out,
    () => {
      const gx = x.req ? x.ensureGrad() : null;
      const gw = w.ensureGrad();
      const gb = b ? b.ensureGrad() : null;
      for (let n = 0; n < B; n++)
        for (let o = 0; o < O; o++)
          for (let oh = 0; oh < OH; oh++)
            for (let ow = 0; ow < OW; ow++) {
              const g = out.g![((n * O + o) * OH + oh) * OW + ow];
              if (g === 0) continue;
              if (gb) gb[o] += g;
              for (let c = 0; c < C; c++)
                for (let kh = 0; kh < KH; kh++) {
                  const ih = oh * stride + kh - pad;
                  if (ih < 0 || ih >= H) continue;
                  for (let kw = 0; kw < KW; kw++) {
                    const iw = ow * stride + kw - pad;
                    if (iw < 0 || iw >= W) continue;
                    const xi = ((n * C + c) * H + ih) * W + iw;
                    const wi = ((o * C + c) * KH + kh) * KW + kw;
                    gw[wi] += g * x.data[xi];
                    if (gx) gx[xi] += g * w.data[wi];
                  }
                }
            }
    },
    [x, w, ...(b ? [b] : [])]
  );
}

/** nearest-neighbour spatial upsampling of [B, C, H, W] by an integer factor */
export function upsample2d(x: T, scale: number): T {
  const [B, C, H, W] = x.shape;
  const OH = H * scale;
  const OW = W * scale;
  const out = new T([B, C, OH, OW]);
  const src = (n: number, c: number, oh: number, ow: number) =>
    ((n * C + c) * H + Math.floor(oh / scale)) * W + Math.floor(ow / scale);
  for (let n = 0; n < B; n++)
    for (let c = 0; c < C; c++)
      for (let oh = 0; oh < OH; oh++)
        for (let ow = 0; ow < OW; ow++) out.data[((n * C + c) * OH + oh) * OW + ow] = x.data[src(n, c, oh, ow)];
  return record(
    out,
    () => {
      const gx = x.ensureGrad();
      for (let n = 0; n < B; n++)
        for (let c = 0; c < C; c++)
          for (let oh = 0; oh < OH; oh++)
            for (let ow = 0; ow < OW; ow++) gx[src(n, c, oh, ow)] += out.g![((n * C + c) * OH + oh) * OW + ow];
    },
    [x]
  );
}

export function pool2d(x: T, k: number, stride: number, pad: number, kind: "max" | "avg"): T {
  const [B, C, H, W] = x.shape;
  const OH = Math.floor((H + 2 * pad - k) / stride) + 1;
  const OW = Math.floor((W + 2 * pad - k) / stride) + 1;
  const out = new T([B, C, OH, OW]);
  const arg = new Int32Array(out.size);
  for (let n = 0; n < B; n++)
    for (let c = 0; c < C; c++)
      for (let oh = 0; oh < OH; oh++)
        for (let ow = 0; ow < OW; ow++) {
          let acc = kind === "max" ? -Infinity : 0;
          let cnt = 0;
          let best = 0;
          for (let i = 0; i < k; i++)
            for (let j = 0; j < k; j++) {
              const ih = oh * stride + i - pad;
              const iw = ow * stride + j - pad;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              const idx = ((n * C + c) * H + ih) * W + iw;
              const v = x.data[idx];
              if (kind === "max") {
                if (v > acc) {
                  acc = v;
                  best = idx;
                }
              } else {
                acc += v;
                cnt++;
              }
            }
          const oi = ((n * C + c) * OH + oh) * OW + ow;
          out.data[oi] = kind === "max" ? acc : acc / Math.max(cnt, 1);
          arg[oi] = kind === "max" ? best : cnt;
        }
  return record(
    out,
    () => {
      const gx = x.ensureGrad();
      for (let n = 0; n < B; n++)
        for (let c = 0; c < C; c++)
          for (let oh = 0; oh < OH; oh++)
            for (let ow = 0; ow < OW; ow++) {
              const oi = ((n * C + c) * OH + oh) * OW + ow;
              const g = out.g![oi];
              if (kind === "max") gx[arg[oi]] += g;
              else {
                for (let i = 0; i < k; i++)
                  for (let j = 0; j < k; j++) {
                    const ih = oh * stride + i - pad;
                    const iw = ow * stride + j - pad;
                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                    gx[((n * C + c) * H + ih) * W + iw] += g / Math.max(arg[oi], 1);
                  }
              }
            }
    },
    [x]
  );
}

export function dropout(x: T, p: number, training: boolean): T {
  if (!training || p <= 0) return x;
  const mask = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) mask[i] = rand() < p ? 0 : 1 / (1 - p);
  const out = new T(x.shape);
  for (let i = 0; i < x.size; i++) out.data[i] = x.data[i] * mask[i];
  return record(
    out,
    () => {
      const gx = x.ensureGrad();
      for (let i = 0; i < x.size; i++) gx[i] += out.g![i] * mask[i];
    },
    [x]
  );
}

export function maskedFill(x: T, mask: T, value: number): T {
  const out = new T(x.shape);
  const idx = new Array(x.shape.length).fill(0);
  for (let i = 0; i < x.size; i++) {
    unravel(i, x.shape, idx);
    const m = mask.data[bIndex(idx, mask.shape)];
    out.data[i] = m > 0.5 ? value : x.data[i];
  }
  return record(
    out,
    () => {
      const gx = x.ensureGrad();
      const id = new Array(x.shape.length).fill(0);
      for (let i = 0; i < x.size; i++) {
        unravel(i, x.shape, id);
        if (mask.data[bIndex(id, mask.shape)] <= 0.5) gx[i] += out.g![i];
      }
    },
    [x]
  );
}

/** F-026: never silently turn an invalid label into a different class. */
export function classIndex(value: number, classes: number): number {
  if (!Number.isInteger(value) || value < 0 || value >= classes)
    throw new Error(`class index ${value} is outside [0, ${classes})`);
  return value;
}

export function crossEntropy(logits: T, labels: T): T {
  const D = logits.shape[logits.shape.length - 1];
  const n = logits.size / D;
  const out = new T([]);
  const probs = new Float32Array(logits.size);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let c = 0; c < D; c++) m = Math.max(m, logits.data[i * D + c]);
    let s = 0;
    for (let c = 0; c < D; c++) {
      const e = Math.exp(logits.data[i * D + c] - m);
      probs[i * D + c] = e;
      s += e;
    }
    for (let c = 0; c < D; c++) probs[i * D + c] /= s;
    const t = classIndex(labels.data[i], D);
    loss += -Math.log(Math.max(probs[i * D + t], 1e-12));
  }
  out.data[0] = loss / n;
  return record(
    out,
    () => {
      const gl = logits.ensureGrad();
      const g = out.g![0] / n;
      for (let i = 0; i < n; i++) {
        const t = classIndex(labels.data[i], D);
        for (let c = 0; c < D; c++) gl[i * D + c] += g * (probs[i * D + c] - (c === t ? 1 : 0));
      }
    },
    [logits]
  );
}

export function mseLoss(a: T, b: T): T {
  const out = new T([]);
  let s = 0;
  const n = Math.max(a.size, b.size);
  for (let i = 0; i < n; i++) {
    const d = a.data[i % a.size] - b.data[i % b.size];
    s += d * d;
  }
  out.data[0] = s / n;
  return record(
    out,
    () => {
      const ga = a.req ? a.ensureGrad() : null;
      const gb = b.req ? b.ensureGrad() : null;
      const g = out.g![0];
      for (let i = 0; i < n; i++) {
        const d = 2 * (a.data[i % a.size] - b.data[i % b.size]) * g / n;
        if (ga) ga[i % a.size] += d;
        if (gb) gb[i % b.size] -= d;
      }
    },
    [a, b]
  );
}

export function bceLoss(p: T, t: T): T {
  const out = new T([]);
  let s = 0;
  for (let i = 0; i < p.size; i++) {
    const x = Math.min(Math.max(p.data[i], 1e-6), 1 - 1e-6);
    const y = t.data[i % t.size];
    s += -(y * Math.log(x) + (1 - y) * Math.log(1 - x));
  }
  out.data[0] = s / p.size;
  return record(
    out,
    () => {
      const gp = p.ensureGrad();
      const g = out.g![0] / p.size;
      for (let i = 0; i < p.size; i++) {
        const x = Math.min(Math.max(p.data[i], 1e-6), 1 - 1e-6);
        const y = t.data[i % t.size];
        gp[i] += g * ((x - y) / (x * (1 - x)));
      }
    },
    [p]
  );
}

export function detach(x: T): T {
  return new T(x.shape, x.data.slice(), false);
}

export function clone(x: T): T {
  return new T(x.shape, x.data.slice(), x.req);
}

export function meanAll(x: T): T {
  return reduce(x, null, false, "mean");
}
