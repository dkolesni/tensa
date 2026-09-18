/**
 * TENSA — symbolic dimension algebra.
 *
 * A dimension expression is a canonical polynomial over *atoms*.
 * An atom is either a dimension variable (`B`, `T`, `D`) or an opaque
 * structured atom such as an integer floor-division produced by a
 * convolution shape rule.  Canonicalisation lets the checker decide
 * `D*4 == 4*D`, `(D/Heads)*Heads == D` (when exact) and, crucially,
 * refuse to decide `S == T` unless it can prove it.
 */

export type AtomDef =
  | { kind: "var"; name: string }
  | { kind: "floordiv"; num: DimExpr; den: DimExpr };

export interface Monomial {
  coef: number;
  /** sorted atom ids, repeated for powers */
  vars: string[];
}

export interface DimExpr {
  terms: Monomial[];
}

// ---------------------------------------------------------------- atom table

const atomTable = new Map<string, AtomDef>();

export function atomKey(def: AtomDef): string {
  if (def.kind === "var") return def.name;
  // multi-term operands are parenthesised so `⌊(H - 1)/2⌋` cannot be misread as `⌊H - 1/2⌋` (F-003)
  const grp = (e: DimExpr) => (e.terms.length > 1 ? `(${show(e)})` : show(e));
  return `⌊${grp(def.num)}/${grp(def.den)}⌋`;
}

export function internAtom(def: AtomDef): string {
  const key = atomKey(def);
  if (!atomTable.has(key)) atomTable.set(key, def);
  return key;
}

export function atomDef(id: string): AtomDef {
  return atomTable.get(id) ?? { kind: "var", name: id };
}

// ---------------------------------------------------------------- constructors

export function dConst(n: number): DimExpr {
  return n === 0 ? { terms: [] } : { terms: [{ coef: n, vars: [] }] };
}

export function dVar(name: string): DimExpr {
  const id = internAtom({ kind: "var", name });
  return { terms: [{ coef: 1, vars: [id] }] };
}

function normalize(terms: Monomial[]): DimExpr {
  const map = new Map<string, Monomial>();
  for (const t of terms) {
    if (t.coef === 0) continue;
    const vars = [...t.vars].sort();
    const key = vars.join("*");
    const cur = map.get(key);
    if (cur) cur.coef += t.coef;
    else map.set(key, { coef: t.coef, vars });
  }
  const out = [...map.values()].filter((t) => t.coef !== 0);
  out.sort((a, b) => a.vars.length - b.vars.length || a.vars.join("*").localeCompare(b.vars.join("*")));
  return { terms: out };
}

export function dAdd(a: DimExpr, b: DimExpr): DimExpr {
  return normalize([...a.terms, ...b.terms]);
}

export function dNeg(a: DimExpr): DimExpr {
  return normalize(a.terms.map((t) => ({ coef: -t.coef, vars: t.vars })));
}

export function dSub(a: DimExpr, b: DimExpr): DimExpr {
  return dAdd(a, dNeg(b));
}

export function dMul(a: DimExpr, b: DimExpr): DimExpr {
  const terms: Monomial[] = [];
  for (const x of a.terms)
    for (const y of b.terms) terms.push({ coef: x.coef * y.coef, vars: [...x.vars, ...y.vars] });
  return normalize(terms);
}

/** Exact division when provable, otherwise a floor-division atom. */
export function dDiv(a: DimExpr, b: DimExpr): DimExpr {
  const bc = asConst(b);
  if (bc !== null) {
    if (bc === 0) return dConst(0);
    const ac = asConst(a);
    if (ac !== null && ac % bc === 0) return dConst(ac / bc);
    // exact symbolic division when every coefficient divides
    if (a.terms.length > 0 && a.terms.every((t) => t.coef % bc === 0)) {
      return normalize(a.terms.map((t) => ({ coef: t.coef / bc, vars: t.vars })));
    }
    // ⌊(a + c)/b⌋ = ⌊(a + r)/b⌋ + k with c = k·b + r, 0 ≤ r < b: the constant
    // term is reduced modulo the divisor so that a stride-2 pool yields ⌊H/2⌋
    // rather than 1 + ⌊(H - 2)/2⌋ and a padded conv yields ⌊(H + 1)/2⌋ (E-005)
    const k0 = a.terms.find((t) => t.vars.length === 0);
    if (k0 && bc > 0 && a.terms.length > 1) {
      const k = Math.floor(k0.coef / bc);
      if (k !== 0) {
        const rest = dAdd(normalize(a.terms.filter((t) => t !== k0)), dConst(k0.coef - k * bc));
        return dAdd(dDiv(rest, b), dConst(k));
      }
    }
  }
  // a == k*b  →  k
  const q = tryDivideByExpr(a, b);
  if (q) return q;
  const id = internAtom({ kind: "floordiv", num: a, den: b });
  return { terms: [{ coef: 1, vars: [id] }] };
}

function tryDivideByExpr(a: DimExpr, b: DimExpr): DimExpr | null {
  if (b.terms.length !== 1) return null;
  const bt = b.terms[0];
  const outTerms: Monomial[] = [];
  for (const t of a.terms) {
    if (t.coef % bt.coef !== 0) return null;
    const vars = [...t.vars];
    for (const v of bt.vars) {
      const i = vars.indexOf(v);
      if (i < 0) return null;
      vars.splice(i, 1);
    }
    outTerms.push({ coef: t.coef / bt.coef, vars });
  }
  return normalize(outTerms);
}

/** Conservative lower bound for positive dimension variables (F-027).
 * Floor atoms are NOT automatically nonnegative: floor((1-T)/2) can be negative.
 * Unknown bounds return -Infinity, never an invented sign proof.
 */
export function lowerBound(e: DimExpr): number {
  let sum = 0;
  for (const t of e.terms) {
    if (!t.vars.length) { sum += t.coef; continue; }
    if (t.coef < 0) return -Infinity;
    let product = t.coef;
    for (const v of t.vars) {
      const a = atomDef(v);
      if (a.kind === "var") continue; // each dimension variable is at least one
      const num = lowerBound(a.num), den = lowerBound(a.den);
      if (num < 0 || den <= 0) return -Infinity;
      const dc = asConst(a.den);
      product *= dc !== null ? Math.floor(num / dc) : 0;
    }
    sum += product;
  }
  return sum;
}

/** True only when zero is excluded by a sound sign bound. */
export function isNonZero(e: DimExpr): boolean {
  return lowerBound(e) > 0 || lowerBound(dNeg(e)) > 0;
}

export function asConst(e: DimExpr): number | null {
  if (e.terms.length === 0) return 0;
  if (e.terms.length === 1 && e.terms[0].vars.length === 0) return e.terms[0].coef;
  return null;
}

export function isConst(e: DimExpr): boolean {
  return asConst(e) !== null;
}

/** The single variable of a bare `X` expression, if any. */
export function asVar(e: DimExpr): string | null {
  if (e.terms.length === 1 && e.terms[0].coef === 1 && e.terms[0].vars.length === 1) {
    const d = atomDef(e.terms[0].vars[0]);
    return d.kind === "var" ? d.name : null;
  }
  return null;
}

export function dEquals(a: DimExpr, b: DimExpr): boolean {
  return show(dSub(a, b)) === "0";
}

export function freeVars(e: DimExpr, out = new Set<string>()): Set<string> {
  for (const t of e.terms)
    for (const v of t.vars) {
      const d = atomDef(v);
      if (d.kind === "var") out.add(d.name);
      else {
        freeVars(d.num, out);
        freeVars(d.den, out);
      }
    }
  return out;
}

export function substitute(e: DimExpr, env: Map<string, DimExpr>): DimExpr {
  let acc = dConst(0);
  for (const t of e.terms) {
    let term = dConst(t.coef);
    for (const v of t.vars) {
      const d = atomDef(v);
      if (d.kind === "var") {
        const rep = env.get(d.name);
        term = dMul(term, rep ? rep : dVar(d.name));
      } else {
        term = dMul(term, dDiv(substitute(d.num, env), substitute(d.den, env)));
      }
    }
    acc = dAdd(acc, term);
  }
  return acc;
}

export function evalDim(e: DimExpr, env: Map<string, number>): number | null {
  let acc = 0;
  for (const t of e.terms) {
    let term = t.coef;
    for (const v of t.vars) {
      const d = atomDef(v);
      if (d.kind === "var") {
        const val = env.get(d.name);
        if (val === undefined) return null;
        term *= val;
      } else {
        const n = evalDim(d.num, env);
        const dd = evalDim(d.den, env);
        if (n === null || dd === null || dd === 0) return null;
        term *= Math.floor(n / dd);
      }
    }
    acc += term;
  }
  return acc;
}

function showMono(t: Monomial): string {
  if (t.vars.length === 0) return String(t.coef);
  const counts = new Map<string, number>();
  for (const v of t.vars) counts.set(v, (counts.get(v) ?? 0) + 1);
  const body = [...counts.entries()]
    .map(([v, n]) => (n === 1 ? v : `${v}^${n}`))
    .join("*");
  if (t.coef === 1) return body;
  if (t.coef === -1) return `-${body}`;
  return `${t.coef}*${body}`;
}

export function show(e: DimExpr): string {
  if (e.terms.length === 0) return "0";
  let s = "";
  // print the constant term last so `T - 1` reads as written, not `-1 + T`
  const terms = [...e.terms.filter((t) => t.vars.length > 0), ...e.terms.filter((t) => t.vars.length === 0)];
  for (const t of terms) {
    const piece = showMono(t);
    if (s === "") s = piece;
    else if (piece.startsWith("-")) s += ` - ${piece.slice(1)}`;
    else s += ` + ${piece}`;
  }
  return s;
}

export const DimExprShow = show;
