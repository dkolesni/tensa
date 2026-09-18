/**
 * TENSA hardening — property-based checks (§38).
 *
 * A tiny seeded generator (no external library) produces random but valid
 * programs; each property is checked over N instances.  Failures report the
 * generated program so the case can be promoted to a fixed regression test.
 */
import { compile } from "../analyze";
import { inspectModule } from "../inspect";
import { TensorType } from "../types";
import { allNodes } from "./driver";
import { dAdd, dSub, dMul, dDiv, dConst, dVar, dEquals, evalDim, isNonZero, lowerBound } from "../dims";
import { RESEARCH_CHALLENGES } from "./research";
import { LEARNING_CHALLENGES } from "./learning";
import { assertEquivalent } from "./metamorphic";

export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 1;
  }
  next(): number {
    // xorshift32
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x >>> 0;
    return this.s / 0x100000000;
  }
  int(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo + 1));
  }
  pick<T>(xs: readonly T[]): T {
    return xs[this.int(0, xs.length - 1)];
  }
}

export interface Property {
  name: string;
  /** produce one program + a checker for it */
  gen(r: Rng): { code: string; check(): void; label: string };
}

export function checkProperty(p: Property, n = 25, seed = 1): string {
  const r = new Rng(seed);
  for (let i = 0; i < n; i++) {
    const inst = p.gen(r);
    try {
      inst.check();
    } catch (e) {
      throw new Error(`${p.name} failed on instance ${i} (${inst.label}): ${(e as Error).message}\n--- program ---\n${inst.code}`);
    }
  }
  return `${n} instances`;
}

function assert(c: boolean, m: string): asserts c {
  if (!c) throw new Error(m);
}

function outShape(code: string): TensorType {
  const res = compile(code, "prop");
  assert(res.errors.length === 0, res.errors.map((d) => `${d.code} ${d.message}`).join("; "));
  const g = res.mod.graphs.find((g) => g.kind === "model")!;
  return res.mod.values.get(g.outputs[0])!.type as TensorType;
}

const SMALL = [2, 3, 4, 6, 8] as const;

// ------------------------------------------------------------------ properties

export const reshapePreservesCount: Property = {
  name: "reshape preserves element count",
  gen(r) {
    const a = r.pick(SMALL), b = r.pick(SMALL), c = r.pick(SMALL);
    const total = a * b * c;
    // pick a different factoring of the same total
    const p = r.pick(SMALL.filter((x) => total % x === 0));
    const q = total / p;
    const code = `dim B
model M(x: Tensor[B, ${a}, ${b}, ${c}]) -> Tensor[B, ${p}, ${q}] { return reshape(x, [B, ${p}, ${q}]) }`;
    const bad = `dim B
model M(x: Tensor[B, ${a}, ${b}, ${c}]) -> Tensor[B, ${p}, ${q + 1}] { return reshape(x, [B, ${p}, ${q + 1}]) }`;
    return {
      code,
      label: `[${a},${b},${c}] -> [${p},${q}]`,
      check() {
        const res = compile(code, "prop");
        assert(res.errors.length === 0, `valid reshape rejected: ${res.errors.map((d) => d.message).join("; ")}`);
        assert(res.mod.constraints.every((k) => k.status === "proved"), "valid constant reshape should be fully proved");
        const rb = compile(bad, "prop");
        assert(rb.errors.length > 0 || rb.mod.constraints.some((k) => k.status !== "proved"), "count-violating reshape silently accepted");
      },
    };
  },
};

export const concatSumsAxis: Property = {
  name: "concat output extent equals the sum of branch extents",
  gen(r) {
    const n = r.int(2, 4);
    const widths = Array.from({ length: n }, () => r.pick([8, 16, 32]));
    const code = `dim B
model M(x: Tensor[B, 4, 8, 8]) -> Tensor[B, ${widths.reduce((s, w) => s + w, 0)}, 8, 8] {
  split merge concat(1) {
${widths.map((w) => `    conv2d(${w}, kernel: 1)`).join("\n")}
  }
}`;
    return {
      code,
      label: `widths ${widths.join("+")}`,
      check() {
        const ty = outShape(code);
        assert(ty.shape[1].terms[0].coef === widths.reduce((s, w) => s + w, 0), `got ${JSON.stringify(ty.shape[1])}`);
      },
    };
  },
};

export const residualPreservesShape: Property = {
  name: "identity residual preserves input shape",
  gen(r) {
    const c = r.pick([4, 8, 16]), hw = r.pick([8, 16]);
    const k = r.pick([1, 3, 5]);
    const code = `dim B
model M(x: Tensor[B, ${c}, ${hw}, ${hw}]) -> Tensor[B, ${c}, ${hw}, ${hw}] {
  residual { conv2d(${c}, kernel: ${k}, pad: ${(k - 1) / 2}) ; relu }
}`;
    return {
      code,
      label: `C=${c} HW=${hw} k=${k}`,
      check() {
        const ty = outShape(code);
        assert(ty.shape.length === 4 && ty.shape[1].terms[0].coef === c && ty.shape[2].terms[0].coef === hw, "shape changed through identity residual");
      },
    };
  },
};

export const sharedStageCountInvariant: Property = {
  name: "shared stage parameter count is independent of application count; independent stages scale linearly",
  gen(r) {
    const apps = r.int(2, 5);
    const w = r.pick([8, 16]);
    const shared = `dim B
block Enc(x: Tensor[B, 8]) -> Tensor[B, ${w}] { linear(${w}) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, ${w}] {
  let e = Enc()
${Array.from({ length: apps }, (_, i) => `  let y${i} = e(x)`).join("\n")}
  return ${Array.from({ length: apps }, (_, i) => `y${i}`).join(" + ")}
}`;
    // distinct binding names on purpose: rebinding `y` collapses the
    // applications into one parameter table (finding F-001)
    const independent = `dim B
block Enc(x: Tensor[B, 8]) -> Tensor[B, ${w}] { linear(${w}) ; gelu }
model M(x: Tensor[B, 8]) -> Tensor[B, ${w}] {
${Array.from({ length: apps }, (_, i) => `  let y${i} = Enc(x)`).join("\n")}
  return ${Array.from({ length: apps }, (_, i) => `y${i}`).join(" + ")}
}`;
    return {
      code: `${shared}\n\n# --- independent ---\n${independent}`,
      label: `${apps} applications, width ${w}`,
      check() {
        const rs = compile(shared, "s");
        const ri = compile(independent, "i");
        assert(rs.errors.length === 0, `shared: ${rs.errors.map((d) => d.message).join("; ")}`);
        assert(ri.errors.length === 0, `independent: ${ri.errors.map((d) => d.message).join("; ")}`);
        const ps = inspectModule(rs.mod).totalParams!;
        const pi = inspectModule(ri.mod).totalParams!;
        const one = 8 * w + w;
        assert(ps === one, `shared stage should own ${one} values, has ${ps}`);
        assert(pi === one * apps, `${apps} independent stages should own ${one * apps} values, have ${pi}`);
      },
    };
  },
};

export const inspectTotalsMatchIR: Property = {
  name: "inspect parameter totals equal IR parameter-table totals",
  gen(r) {
    const depth = r.int(1, 4);
    const widths = Array.from({ length: depth }, () => r.pick([4, 8, 16, 32]));
    const code = `dim B
model M(x: Tensor[B, 8]) -> Tensor[B, ${widths[widths.length - 1]}] {
${widths.map((w) => `  linear(${w})\n  gelu`).join("\n")}
}`;
    return {
      code,
      label: `widths ${widths.join(",")}`,
      check() {
        const res = compile(code, "prop");
        assert(res.errors.length === 0, res.errors.map((d) => d.message).join("; "));
        const rep = inspectModule(res.mod);
        let ir = 0;
        for (const p of res.mod.params) ir += p.shape.reduce((s, d) => s * d.terms[0].coef, 1);
        assert(rep.totalParams === ir, `inspect ${rep.totalParams} vs IR ${ir}`);
        const nodeParams = new Set(allNodes(res.mod).flatMap((n) => n.params));
        for (const p of res.mod.params) assert(nodeParams.has(p.id), `param ${p.id} is not referenced by any IR node`);
      },
    };
  },
};

export const algebraOutcomes: Property = {
  name: "eqDim/leDim outcomes are exclusive and rewrite-stable; floor signs are sound (F-027)",
  gen(r) {
    const k = r.int(2, 8);
    const code = `dim B\ndim T\ndim S\nmodel M(x: Tensor[B, T]) -> Tensor[B, ${k}*T - ${k-1}*T] { return x }`;
    return { code, label: `coefficient ${k}`, check() {
      assert(compile(code).ok, "rewritten identity rejected");
      const states = (src: string, origin: string) => {
        const c = compile(src);
        const found = new Set(c.mod.constraints.filter(x => x.origin.includes(origin)).map(x => x.status));
        assert(found.size <= 1, `contradictory outcomes: ${[...found]}`);
        return found.size ? [...found][0] : "proved";
      };
      for (const [rhs, expected] of [["T", "proved"], ["T+1", "failed"], ["S", "assumed"]]) {
        for (const rewrite of [rhs, `(${rhs}) + ${k}*T - T*${k}`]) {
          const src = `dim B\ndim T\ndim S\nmodel M(x: Tensor[B,T]) -> Tensor[B,${rewrite}] { return x }`;
          assert(states(src, "result of 'M' axis 1") === expected, `eq ${rewrite} != ${expected}`);
        }
      }
      for (const [end, expected] of [["T", "proved"], ["T+1", "failed"], ["S", "assumed"]]) {
        for (const rewrite of [end, `(${end}) + ${k}*T - T*${k}`]) {
          const src = `dim B\ndim T\ndim S\nmodel M(x: Tensor[B,T]) -> Tensor[B,${end}] { return x[:,0:${rewrite}] }`;
          assert(states(src, "slice end on axis 1") === expected, `le ${rewrite} != ${expected}`);
        }
      }
      const unknown = compile(`dim B\ndim T\ncustom op opaque(x: Tensor[B,T]) -> Tensor[B,T] { shape: unknown }\nmodel M(x: Tensor[B,T]) -> Tensor[B,T] { return opaque(x) }`);
      assert(unknown.ok && unknown.mod.diags.some(d => d.code === "AXS0903"), "unknown not preserved");
      assert(!unknown.mod.constraints.some(c => c.origin.includes("result of 'M'")), "unknown invented a shape constraint");
      const sliced = compile(`dim B\ndim T\ncustom op opaque(x: Tensor[B,T]) -> Tensor[B,T] { shape: unknown }\nmodel M(x: Tensor[B,T]) -> Tensor[B,T] { return opaque(x)[:,0:T] }`);
      assert(sliced.ok && !sliced.mod.constraints.some(c => c.origin.includes("slice")), "unknown rank became false slice bounds (F-030)");
      const t = dVar("T"), n = dConst(k);
      assert(dEquals(dMul(n, dAdd(t, dConst(1))), dAdd(dMul(t,n),n)), "distributivity");
      const expr = dAdd(dConst(1), dDiv(dSub(dConst(1), t), dConst(2)));
      assert(!isNonZero(expr) && lowerBound(expr) < 0, "negative floor atom used as a positive proof");
      for (let value = 1; value <= 16; value++) {
        const v = evalDim(expr, new Map([["T", value]]))!;
        assert(v === 1 + Math.floor((1-value)/2), "floor rewrite changed value");
        assert(lowerBound(expr) <= v, "unsound bound");
      }
      const signed = compile(`dim B\ndim T\nmodel M(x: Tensor[B,2+(1-T)/2]) -> Tensor[B,1] { return x }`);
      assert(signed.ok && signed.warnings.some(d => d.code === "AXS0403"), "satisfiable signed-floor equality refuted");
    } };
  },
};

export const nestedIdentity: Property = {
  name: "nested for/scan preserves shared identity and fresh-stage owner paths",
  gen(r) {
    const a = r.int(1,3), b = r.int(1,3), time = r.int(2,5);
    const code = `dim B\ndim T\nblock Cell(x: Tensor[B,4]) -> Tensor[B,4] { linear(4) ; tanh }\nmodel M(x: Tensor[B,T,4]) -> Tensor[B,T,4] {
      let cell = Cell()
      let (ys, last) = scan over x axis: 1 carry h: Tensor[B,4] init: zeros {
        step + h
        for ${a} { for ${b}: cell }
        yield tanh()
      }
      return ys
    }`;
    return { code, label: `${a}x${b}, T=${time}`, check() {
      const shared = compile(code), fresh = compile(code.replace(`for ${b}: cell`, `for ${b}: Cell()`));
      assert(shared.ok && fresh.ok, [...shared.errors, ...fresh.errors].map(d => d.message).join(";"));
      assert(shared.mod.params.length === 2, "shared params duplicated");
      assert(fresh.mod.params.length === 2*a*b, "fresh params tied across iterations");
      assert(new Set(fresh.mod.params.map(p => p.id)).size === 2*a*b, "identity collision");
      assert(shared.mod.params.every(p => p.owner === "M/cell/linear#1"), "shared path escaped lexical owner");
      assertEquivalent(code, code.replace(/\bcell\b/g, "renamed"), { ignore: ["cell", "renamed"], run: { dims: { B:2, T:time } } });
    } };
  },
};

export const recursiveEffects: Property = {
  name: "effect unions survive nested functions, residuals, repeats and scan (F-016)",
  gen(r) {
    const depth = r.int(1,3);
    const code = `dim B\ndim T
fn f(x: Tensor[B,4]) -> Tensor[B,4] { return x |> batchnorm |> dropout(0.2) |> stop_grad }
model M(x: Tensor[B,T,4]) -> Tensor[B,T,4] {
  let (ys, last) = scan over x axis: 1 carry h: Tensor[B,4] init: zeros {
    step + h
    for ${depth} { residual { f() } }
    yield tanh()
  }
  return ys
}`;
    return { code, label: `depth ${depth}`, check() {
      const c = compile(code); assert(c.ok, c.errors.map(d => d.message).join(";"));
      for (const n of allNodes(c.mod).filter(n => n.regions?.length)) {
        const effects = new Set<string>();
        const walk = (ns: typeof n[]) => { for (const x of ns) { x.effects.filter(e => e !== "pure").forEach(e => effects.add(e)); for (const reg of x.regions ?? []) walk(reg.nodes); } };
        for (const reg of n.regions!) walk(reg.nodes);
        for (const e of effects) assert(n.effects.some(x => x === e), `${n.op} lost ${e}`);
      }
    } };
  },
};

export const researchParserMutants: Property = {
  name: "seeded parser evil twins against every M4 program terminate with syntax diagnostics",
  gen(r) {
    const mutation = r.int(0,2);
    return { code: "all RESEARCH_CHALLENGES", label: `mutation ${mutation}`, check() {
      for (const ch of [...RESEARCH_CHALLENGES, ...LEARNING_CHALLENGES.filter(c => c.id === "distillation")]) {
        const code = mutation === 0 ? ch.code.slice(0, ch.code.lastIndexOf("}")) : mutation === 1 ? ch.code.replace("->", "=>") : ch.code.replace("model ", "model : ");
        const c = compile(code);
        assert(c.errors.some(d => /^AXS01/.test(d.code)), `${ch.id}: mutation ${mutation} not rejected by parser: ${c.errors.map(d=>d.code)}`);
        assert(c.errors.every(d => d.loc.line >= 1 && d.loc.col >= 1), "diagnostic lost location");
      }
    } };
  },
};

export const PROPERTIES: Property[] = [
  algebraOutcomes,
  nestedIdentity,
  recursiveEffects,
  researchParserMutants,
  reshapePreservesCount,
  concatSumsAxis,
  residualPreservesShape,
  sharedStageCountInvariant,
  inspectTotalsMatchIR,
];
