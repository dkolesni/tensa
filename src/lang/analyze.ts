/**
 * TENSA — semantic analysis and elaboration to backend-neutral IR.
 *
 * Responsibilities:
 *   - symbolic dimension propagation, unification and constraint carrying
 *   - implicit-flow (cursor) semantics and multi-stream discipline
 *   - stage identity: which applications share one parameter table
 *   - effects and persistent state
 *   - objectives, data contracts and the training plan
 */
import {
  Arg, Branch, DataDecl, Decl, DimAst, Expr, FnLikeDecl, ParamDef, Program, SliceAst, Stmt,
  TrainDecl, TypeExpr,
} from "./ast";
import { CATALOG_MAP, DATA_OP_MAP, InferCtx, OpSpec } from "./catalog";
import {
  DimExpr, asConst, asVar, atomDef, dAdd, dConst, dDiv, dEquals, dMul, dSub, dVar, freeVars, isNonZero, show, substitute,
} from "./dims";
import {
  IRAttr, IRData, IRDataField, IRGraph, IRLoss, IRModule, IRNode, IROptimizer, IRParam,
  IRPhase, IRRegion, IRState, IRValue,
} from "./ir";
import { parse } from "./parser";
import {
  Diagnostic, Effect, Loc, NOLOC, TensorKind, TensorType, ValueType, diag, isTensor, scalar, showType, tensor, unknownTensor,
} from "./types";

// ------------------------------------------------------------------ compile values

type CVal =
  | { v: "val"; id: string; type: ValueType }
  | { v: "stage"; inst: Instance }
  | { v: "dim"; expr: DimExpr }
  | { v: "num"; value: number }
  | { v: "str"; value: string }
  | { v: "bool"; value: boolean }
  | { v: "tuple"; items: CVal[] }
  | { v: "op"; name: string }
  | { v: "decl"; decl: FnLikeDecl }
  | { v: "state"; id: string }
  | { v: "none" };

interface Instance {
  id: string;
  declName: string;
  path: string;
  decl: FnLikeDecl;
  config: Map<string, CVal>;
  counter: number;
  applications: string[];
  params: string[];
  states: string[];
  region: string;
}

interface Frame {
  scope: Map<string, CVal>;
  dimEnv: Map<string, DimExpr>;
  inst: Instance;
  cursor: CVal | null;
  returned: CVal | null;
  /** names bound by `let` in the current region — rebinding one warns (E-001) */
  bound?: Set<string>;
}

export interface AnalyzeResult {
  mod: IRModule;
  instances: Instance[];
}

let uid = 0;
const nextId = (p: string) => `${p}${++uid}`;

export class Analyzer {
  mod: IRModule;
  decls = new Map<string, Decl>();
  globalDims = new Map<string, DimExpr>();
  runtimeDims = new Set<string>();
  instances: Instance[] = [];
  frames: Frame[] = [];
  nodeStack: IRNode[][] = [];
  templateVars = new Set<string>();
  flexible = new Set<string>();
  emittedCodes = new Set<string>();

  constructor(public program: Program, name: string, diags: Diagnostic[]) {
    this.mod = {
      name,
      dims: [],
      graphs: [],
      params: [],
      states: [],
      objectives: [],
      data: [],
      plans: [],
      customOps: [],
      constraints: [],
      values: new Map(),
      diags: [...diags],
    };
  }

  // ---------------------------------------------------------------- diagnostics
  err(code: string, msg: string, loc: Loc, notes?: string[]) {
    this.mod.diags.push(diag(code, "error", msg, loc, this.where(), notes));
  }
  warn(code: string, msg: string, loc: Loc, notes?: string[]) {
    this.mod.diags.push(diag(code, "warning", msg, loc, this.where(), notes));
  }
  info(code: string, msg: string, loc: Loc, notes?: string[]) {
    this.mod.diags.push(diag(code, "info", msg, loc, this.where(), notes));
  }
  where(): string {
    const f = this.frame;
    return f ? f.inst.path : this.mod.name;
  }
  get frame(): Frame {
    return this.frames[this.frames.length - 1];
  }
  get nodes(): IRNode[] {
    return this.nodeStack[this.nodeStack.length - 1];
  }

  // ---------------------------------------------------------------- values
  value(type: ValueType, name?: string): IRValue {
    const v: IRValue = { id: nextId("v"), type, name };
    this.mod.values.set(v.id, v);
    return v;
  }

  emit(
    op: string,
    inputs: string[],
    attrs: Record<string, IRAttr>,
    outTypes: ValueType[],
    opts: {
      params?: string[];
      states?: string[];
      effects?: Effect[];
      regions?: IRRegion[];
      loc?: Loc;
      note?: string;
      instance?: string;
    } = {}
  ): IRValue[] {
    const outs = outTypes.map((t) => this.value(t));
    // a node that owns regions (apply, residual, parallel, scan, static_repeat)
    // carries the union of its body's effects: a pure `fn` that calls dropout
    // is stochastic at every call site (F-016, §15)
    let effects = opts.effects;
    if (!effects && opts.regions) {
      const union = new Set<Effect>();
      const walk = (ns: IRNode[]) => {
        for (const n of ns) {
          for (const e of n.effects) if (e !== "pure") union.add(e);
          if (n.regions) for (const r of n.regions) walk(r.nodes);
        }
      };
      for (const r of opts.regions) walk(r.nodes);
      effects = union.size ? [...union] : ["pure"];
    }
    const node: IRNode = {
      id: nextId("n"),
      op,
      inputs,
      attrs,
      outputs: outs.map((o) => o.id),
      params: opts.params ?? [],
      states: opts.states ?? [],
      effects: effects ?? ["pure"],
      regions: opts.regions,
      stage: this.where(),
      loc: opts.loc ?? NOLOC,
      note: opts.note,
      sharedStage: undefined,
    };
    (node as IRNode & { instance?: string }).instance = opts.instance;
    this.nodes.push(node);
    return outs;
  }

  // ---------------------------------------------------------------- dims
  resolveDimName(name: string, loc: Loc): DimExpr {
    const f = this.frame;
    if (f) {
      const local = f.dimEnv.get(name);
      if (local) return local;
    }
    const g = this.globalDims.get(name);
    if (g) return g;
    if (this.runtimeDims.has(name) || this.templateVars.has(name)) return dVar(name);
    this.err("AXS0203", `unknown dimension '${name}'`, loc, [
      `declare it with 'dim ${name}' or introduce it as a declaration-local template variable in a parameter type`,
    ]);
    this.runtimeDims.add(name);
    return dVar(name);
  }

  dimOf(d: DimAst): DimExpr {
    switch (d.d) {
      case "num":
        return dConst(d.value);
      case "name":
        return this.resolveDimName(d.name, d.loc);
      case "bin": {
        const l = this.dimOf(d.l);
        const r = this.dimOf(d.r);
        if (d.op === "+") return dAdd(l, r);
        if (d.op === "-") return dSub(l, r);
        if (d.op === "*") return dMul(l, r);
        return dDiv(l, r);
      }
    }
  }

  typeOf(t: TypeExpr): ValueType {
    switch (t.t) {
      case "scalarType":
        return scalar;
      case "intType":
        return { t: "int" };
      case "dimType":
        return { t: "int" };
      case "tupleType":
        return { t: "tuple", items: t.items.map((i) => this.typeOf(i)) };
      case "tensorType": {
        const dtype = t.kind === "Tokens" || t.kind === "Class" ? "i32" : t.kind === "Mask" ? "bool" : "f32";
        return tensor(t.dims.map((d) => this.dimOf(d)), t.kind as TensorKind, dtype);
      }
    }
  }

  /** prove / unify / carry a symbolic equality */
  eqDim(a: DimExpr, b: DimExpr, origin: string, loc: Loc, code = "AXS0401"): boolean {
    const f = this.frame;
    const env = f ? f.dimEnv : new Map<string, DimExpr>();
    const ae = substitute(a, env);
    const be = substitute(b, env);
    if (dEquals(ae, be)) return true;
    const av = asVar(ae);
    const bv = asVar(be);
    if (av && this.flexible.has(av) && !freeVars(be).has(av)) {
      f?.dimEnv.set(av, be);
      this.mod.constraints.push({ lhs: ae, rhs: be, status: "proved", origin: `${origin} (bound template dim ${av})`, loc });
      return true;
    }
    if (bv && this.flexible.has(bv) && !freeVars(ae).has(bv)) {
      f?.dimEnv.set(bv, ae);
      this.mod.constraints.push({ lhs: be, rhs: ae, status: "proved", origin: `${origin} (bound template dim ${bv})`, loc });
      return true;
    }
    const diff = dSub(ae, be);
    const c = asConst(diff);
    if (c !== null && c !== 0) {
      this.mod.constraints.push({ lhs: ae, rhs: be, status: "failed", origin, loc });
      this.err(code, `${origin}: expected ${show(be)}, found ${show(ae)}`, loc, [
        `the two extents differ by the constant ${Math.abs(c)}`,
      ]);
      return false;
    }
    if (isNonZero(diff)) {
      // e.g. 2*K vs 4*K: no positive binding of the dimensions can make them equal
      this.mod.constraints.push({ lhs: ae, rhs: be, status: "failed", origin, loc });
      this.err(code, `${origin}: expected ${show(be)}, found ${show(ae)}`, loc, [
        `the two extents differ by ${show(diff.terms[0].coef < 0 ? dSub(be, ae) : diff)}, which is never zero for dimensions ≥ 1`,
      ]);
      return false;
    }
    this.mod.constraints.push({ lhs: ae, rhs: be, status: "assumed", origin, loc });
    this.warn("AXS0403", `${origin}: cannot prove ${show(ae)} = ${show(be)}`, loc, [
      "the constraint is carried into the IR and verified when dimensions are bound at runtime",
      `add 'dim' definitions or an explicit reshape to make this provable`,
    ]);
    return false;
  }

  /**
   * prove / refute / carry a symbolic bound `a ≤ b` (slice and index ranges,
   * F-011).  Dimension atoms are ≥ 1 (variables) or ≥ 0 (floor divisions), so a
   * difference whose coefficients are all non-negative is proved and one whose
   * coefficients are all negative (and cannot vanish) is refuted; anything else
   * is carried as an `assumed` `<=` constraint that the runtime verifies.
   */
  leDim(a: DimExpr, b: DimExpr, origin: string, loc: Loc): boolean {
    const env = this.frame ? this.frame.dimEnv : new Map<string, DimExpr>();
    const ae = substitute(a, env);
    const be = substitute(b, env);
    const diff = dSub(be, ae);
    if (diff.terms.every((t) => t.coef >= 0)) return true;
    // `T - 1 ≥ 0` holds because every plain-variable monomial is ≥ 1: a negative
    // constant is absorbed when the positive variable coefficients cover it
    const negConst = diff.terms.find((t) => t.vars.length === 0 && t.coef < 0);
    if (negConst && diff.terms.every((t) => t === negConst || t.coef > 0)) {
      const cover = diff.terms
        .filter((t) => t !== negConst && t.vars.every((v) => atomDef(v).kind === "var"))
        .reduce((s, t) => s + t.coef, 0);
      if (cover + negConst.coef >= 0) return true;
    }
    if (diff.terms.every((t) => t.coef < 0) && isNonZero(diff)) {
      this.mod.constraints.push({ lhs: ae, rhs: be, rel: "<=", status: "failed", origin, loc });
      const c = asConst(diff);
      this.err("AXS0410", `${origin}: ${show(ae)} exceeds ${show(be)}`, loc, [
        c !== null ? `the range overshoots by ${-c}` : `${show(dSub(ae, be))} is positive for every binding of the dimensions`,
      ]);
      return false;
    }
    this.mod.constraints.push({ lhs: ae, rhs: be, rel: "<=", status: "assumed", origin, loc });
    this.warn("AXS0403", `${origin}: cannot prove ${show(ae)} ≤ ${show(be)}`, loc, [
      "the bound is carried into the IR and verified when dimensions are bound at runtime",
    ]);
    return false;
  }

  eqShape(a: DimExpr[], b: DimExpr[], origin: string, loc: Loc): boolean {
    if (a.length !== b.length) {
      this.err("AXS0402", `${origin}: rank ${a.length} vs ${b.length}`, loc);
      return false;
    }
    let ok = true;
    for (let i = 0; i < a.length; i++) ok = this.eqDim(a[i], b[i], `${origin} axis ${i}`, loc) && ok;
    return ok;
  }

  // ---------------------------------------------------------------- entry
  run(): IRModule {
    // pass 1 — collect declarations
    for (const d of this.program.decls) {
      const name = (d as { name: string }).name;
      if (d.d === "dim") {
        if (this.globalDims.has(name) || this.runtimeDims.has(name))
          this.err("AXS0202", `duplicate dim '${name}'`, d.loc);
        if (d.value) this.globalDims.set(name, this.dimOf(d.value));
        else this.runtimeDims.add(name);
        this.mod.dims.push({ name, value: d.value ? this.globalDims.get(name)! : null, runtime: !d.value });
        continue;
      }
      if (this.decls.has(name)) this.err("AXS0202", `duplicate declaration '${name}'`, d.loc);
      this.decls.set(name, d);
    }
    // custom ops
    for (const d of this.program.decls)
      if (d.d === "customop") {
        this.templateVars = new Set(this.collectTemplateVars(d.params));
        for (const v of this.templateVars) this.flexible.add(v);
        this.mod.customOps.push({
          name: d.name,
          inputs: d.params.map((p) => ({ name: p.name, type: this.typeOf(p.type) })),
          result: d.ret ? this.typeOf(d.ret) : null,
          effects: (d.effects.length ? d.effects : ["pure"]) as Effect[],
          backends: d.backend,
          shapeUnknown: d.shapeUnknown,
          differentiable: d.differentiable,
        });
      }

    const referenced = this.collectReferenced();

    // pass 2 — elaborate models
    for (const d of this.program.decls)
      if (d.d === "model") this.elabDeclaration(d, false);
    for (const d of this.program.decls)
      if (d.d === "objective") this.elabDeclaration(d, false);
    // unreferenced blocks / fns are still statically checked on their own
    for (const d of this.program.decls)
      if ((d.d === "block" || d.d === "fn") && !referenced.has(d.name)) this.elabDeclaration(d, true);
    for (const d of this.program.decls) if (d.d === "data") this.elabData(d);
    for (const d of this.program.decls) if (d.d === "train") this.elabTrain(d);

    // parameter sharing post-pass
    const shared = new Map<string, Instance>();
    for (const i of this.instances) if (i.applications.length > 1) shared.set(i.id, i);
    const walk = (ns: IRNode[]) => {
      for (const n of ns) {
        const inst = (n as IRNode & { instance?: string }).instance;
        if (inst && shared.has(inst)) n.sharedStage = shared.get(inst)!.path;
        if (n.regions) for (const r of n.regions) walk(r.nodes);
      }
    };
    for (const g of this.mod.graphs) walk(g.nodes);
    for (const o of this.mod.objectives) walk(o.graph.nodes);
    return this.mod;
  }

  collectTemplateVars(params: ParamDef[]): string[] {
    const out: string[] = [];
    const visit = (d: DimAst) => {
      if (d.d === "name") {
        if (!this.globalDims.has(d.name) && !this.runtimeDims.has(d.name) && !out.includes(d.name)) out.push(d.name);
      } else if (d.d === "bin") {
        visit(d.l);
        visit(d.r);
      }
    };
    for (const p of params) if (p.type.t === "tensorType") p.type.dims.forEach(visit);
    return out;
  }

  collectReferenced(): Set<string> {
    const used = new Set<string>();
    const visitExpr = (e: Expr) => {
      switch (e.e) {
        case "name":
          used.add(e.name);
          break;
        case "call":
          visitExpr(e.callee);
          e.args.forEach((a) => visitExpr(a.value));
          break;
        case "member":
          visitExpr(e.obj);
          break;
        case "index":
          visitExpr(e.obj);
          break;
        case "bin":
          visitExpr(e.l);
          visitExpr(e.r);
          break;
        case "unary":
          visitExpr(e.v);
          break;
        case "pipe":
          e.stages.forEach(visitExpr);
          break;
        case "tuple":
        case "list":
          e.items.forEach(visitExpr);
          break;
        case "scan":
          visitExpr(e.over);
          e.body.forEach(visitStmt);
          break;
        default:
          break;
      }
    };
    const visitStmt = (s: Stmt) => {
      switch (s.s) {
        case "let":
        case "expr":
          visitExpr(s.value);
          break;
        case "return":
          if (s.value) visitExpr(s.value);
          break;
        case "yield":
          s.values.forEach(visitExpr);
          break;
        case "residual":
          if (s.via) visitExpr(s.via);
          s.body.forEach(visitStmt);
          break;
        case "split":
          s.branches.forEach((b) => b.body.forEach(visitStmt));
          break;
        case "for":
          s.body.forEach(visitStmt);
          break;
        default:
          break;
      }
    };
    for (const d of this.program.decls) {
      if (d.d === "model" || d.d === "objective") d.body.forEach(visitStmt);
      if (d.d === "block" || d.d === "fn") d.body.forEach(visitStmt);
      if (d.d === "train")
        for (const st of d.body) {
          if (st.t === "loss") st.args.forEach((a) => visitExpr(a.value));
          if (st.t === "model") used.add(st.name);
        }
    }
    // a block referenced only from another unused block is still "referenced"
    return used;
  }

  // ---------------------------------------------------------------- declarations
  newInstance(decl: FnLikeDecl, path: string, config: Map<string, CVal>, region: string): Instance {
    const inst: Instance = {
      id: nextId("inst"),
      declName: decl.name,
      path,
      decl,
      config,
      counter: 0,
      applications: [],
      params: [],
      states: [],
      region,
    };
    this.instances.push(inst);
    return inst;
  }

  elabDeclaration(d: FnLikeDecl, standalone: boolean) {
    const tvars = this.collectTemplateVars(d.params);
    const prevTemplates = this.templateVars;
    this.templateVars = new Set(tvars);
    for (const v of tvars) if (standalone || d.d === "model" || d.d === "objective") this.runtimeDims.add(v);

    const inst = this.newInstance(d, d.name, new Map(), d.name);
    const frame: Frame = { scope: new Map(), dimEnv: new Map(), inst, cursor: null, returned: null };
    const graph: IRGraph = {
      name: d.name,
      kind: d.d,
      inputs: [],
      outputs: [],
      nodes: [],
      standalone,
      loc: d.loc,
    };
    this.frames.push(frame);
    this.nodeStack.push(graph.nodes);
    const tensorParams: ParamDef[] = [];
    for (const p of d.params) {
      const ty = this.typeOf(p.type);
      if (ty.t === "int") {
        frame.scope.set(p.name, p.def ? this.constArg(p.def) : { v: "num", value: 0 });
        continue;
      }
      const v = this.value(ty, p.name);
      graph.inputs.push(v);
      frame.scope.set(p.name, { v: "val", id: v.id, type: ty });
      tensorParams.push(p);
    }
    frame.cursor = graph.inputs.length === 1 ? { v: "val", id: graph.inputs[0].id, type: graph.inputs[0].type } : null;
    inst.applications.push(d.name);

    this.elabBody(d.body);

    const result = frame.returned ?? frame.cursor;
    // a model input that nothing reads is a contract the body silently ignores
    // (the classic case: a padding mask declared and bound, never given to attention)
    if (d.d === "model" && graph.inputs.length > 1 && result && result.v !== "none") {
      const read = new Set<string>();
      const walk = (ns: IRNode[]) => {
        for (const n of ns) {
          for (const i of n.inputs) read.add(i);
          if (n.regions) for (const r of n.regions) walk(r.nodes);
        }
      };
      walk(graph.nodes);
      for (const id of this.cvalIds(result)) read.add(id);
      for (const iv of graph.inputs)
        if (!read.has(iv.id))
          this.warn("AXS0305", `model '${d.name}' never reads its input '${iv.name ?? iv.id}'`, d.loc, [
            iv.type.t === "tensor" && iv.type.kind === "Mask" ? "pass it to the operation it guards, e.g. `attention(mask: ...)`" : "drop the input or use it in the body",
          ]);
    }
    if (!result || result.v === "none") {
      if (d.ret) this.err("AXS0303", `${d.d} '${d.name}' declares a result but produces no value`, d.loc);
    } else {
      const rt = this.cvalType(result);
      graph.outputs = this.cvalIds(result);
      graph.resultType = rt;
      if (d.ret) {
        const want = this.typeOf(d.ret);
        this.checkResult(want, rt, d.loc, d.name);
        // the declared type is the contract callers see (it carries semantic kinds)
        graph.resultType = want;
      }
      if (d.d === "objective" && rt.t !== "scalar")
        this.err("AXS0601", `objective '${d.name}' must return Scalar, found ${showType(rt)}`, d.loc, [
          "reduce the per-example loss, e.g. with mean(...)",
        ]);
    }
    this.nodeStack.pop();
    this.frames.pop();
    this.templateVars = prevTemplates;
    if (d.d === "objective")
      this.mod.objectives.push({
        name: d.name,
        graph,
        inputs: d.params.map((p) => ({ name: p.name, type: this.typeOf(p.type) })),
      });
    else this.mod.graphs.push(graph);
  }

  checkResult(want: ValueType, got: ValueType, loc: Loc, name: string) {
    if (want.t === "tuple" && got.t === "tuple") {
      if (want.items.length !== got.items.length) {
        this.err("AXS0406", `'${name}' returns ${got.items.length} values but declares ${want.items.length}`, loc);
        return;
      }
      want.items.forEach((w, i) => this.checkResult(w, (got as { items: ValueType[] }).items[i], loc, name));
      return;
    }
    if (isTensor(want) && isTensor(got)) {
      if (got.unknown) {
        // knowledge was lost upstream (AXS0903 already fired there); the declared
        // result is a claim the checker can neither prove nor refute — say so
        // instead of inventing a rank
        this.info("AXS0903", `result of '${name}' cannot be checked against ${showType(want)}: its shape is unknown`, loc);
        return;
      }
      if (want.shape.length !== got.shape.length) {
        this.err(
          "AXS0406",
          `'${name}' declares result ${showType(want)} but produces ${showType(got)} (rank differs)`,
          loc
        );
        return;
      }
      for (let i = 0; i < want.shape.length; i++)
        this.eqDim(got.shape[i], want.shape[i], `result of '${name}' axis ${i}`, loc);
      if (want.kind !== "Tensor" && got.kind !== want.kind && got.kind !== "Tensor")
        this.warn("AXS0409", `'${name}' declares ${want.kind}[...] but produces ${got.kind}[...]`, loc);
      return;
    }
    if (want.t !== got.t)
      this.err("AXS0406", `'${name}' declares ${showType(want)} but produces ${showType(got)}`, loc);
  }

  cvalType(c: CVal): ValueType {
    switch (c.v) {
      case "val":
        return c.type;
      case "tuple":
        return { t: "tuple", items: c.items.map((i) => this.cvalType(i)) };
      case "num":
      case "dim":
        return scalar;
      default:
        return scalar;
    }
  }
  cvalIds(c: CVal): string[] {
    if (c.v === "val") return [c.id];
    if (c.v === "tuple") return c.items.flatMap((i) => this.cvalIds(i));
    return [];
  }

  // ---------------------------------------------------------------- statements
  elabBody(body: Stmt[]) {
    for (const s of body) {
      if (this.frame.returned) break;
      this.elabStmt(s);
    }
  }

  elabStmt(s: Stmt) {
    const f = this.frame;
    switch (s.s) {
      case "let": {
        const val = this.elabExpr(s.value, null, s.names.length === 1 ? s.names[0] : undefined);
        f.bound ??= new Set();
        for (const n of s.names) {
          if (f.bound.has(n))
            this.warn("AXS0304", `'${n}' is rebound in the same scope; the earlier binding is now unreachable`, s.loc, [
              "a rebound stage application gets its own parameters (it is not shared with the earlier one)",
              "rename the binding if both values are needed",
            ]);
          f.bound.add(n);
        }
        if (s.names.length === 1) {
          f.scope.set(s.names[0], val);
          if (val.v === "val" || val.v === "tuple") f.cursor = val;
        } else {
          if (val.v !== "tuple" || val.items.length !== s.names.length) {
            this.err(
              "AXS0406",
              `cannot destructure ${s.names.length} names from ${showType(this.cvalType(val))}`,
              s.loc
            );
            s.names.forEach((n) => f.scope.set(n, { v: "none" }));
          } else {
            s.names.forEach((n, i) => f.scope.set(n, (val as { items: CVal[] }).items[i]));
            // the tuple is the most recent value: a following bare op must fail
            // with AXS0408 instead of silently consuming the stale cursor (F-007)
            f.cursor = val;
          }
        }
        return;
      }
      case "return": {
        f.returned = s.value ? this.elabExpr(s.value, null) : f.cursor ?? { v: "none" };
        return;
      }
      case "yield": {
        const vals = s.values.map((v) => this.elabExpr(v, null));
        f.scope.set("__yield", vals.length === 1 ? vals[0] : { v: "tuple", items: vals });
        return;
      }
      case "expr": {
        const val = this.elabExpr(s.value, null);
        if (val.v === "val" || val.v === "tuple") f.cursor = val;
        else if (val.v === "stage")
          this.warn("AXS0302", `stage value created but not bound; its parameters are unreachable`, s.loc);
        return;
      }
      case "param": {
        const ty = this.typeOf(s.type);
        const shape = isTensor(ty) ? ty.shape : [];
        const id = `${f.inst.path}.${s.name}`;
        const existing = this.mod.params.find((p) => p.id === id);
        if (existing) {
          existing.applications.push(this.where());
          const v = this.emit("param_read", [], { param: id }, [ty], {
            params: [id],
            effects: ["parameterized"],
            loc: s.loc,
            instance: f.inst.id,
          });
          f.scope.set(s.name, { v: "val", id: v[0].id, type: ty });
          return;
        }
        const p: IRParam = {
          id,
          owner: f.inst.path,
          role: s.name,
          shape,
          init: s.init,
          trainable: !s.frozen,
          kind: "explicit",
          applications: [this.where()],
        };
        this.mod.params.push(p);
        f.inst.params.push(id);
        const v = this.emit("param_read", [], { param: id }, [ty], {
          params: [id],
          effects: ["parameterized"],
          loc: s.loc,
          instance: f.inst.id,
        });
        f.scope.set(s.name, { v: "val", id: v[0].id, type: ty });
        return;
      }
      case "state": {
        const ty = this.typeOf(s.type);
        const id = `${f.inst.path}.${s.name}`;
        if (!this.mod.states.find((x) => x.id === id)) {
          const st: IRState = {
            id,
            owner: f.inst.path,
            role: s.name,
            shape: isTensor(ty) ? ty.shape : [],
            init: s.init,
            update: s.update ?? "assign",
            checkpointed: true,
            category: s.update && s.update.startsWith("ema") ? "ema" : "algorithmic",
          };
          this.mod.states.push(st);
          f.inst.states.push(id);
        }
        f.scope.set(s.name, { v: "state", id });
        return;
      }
      case "residual":
        this.elabResidual(s.via, s.body, s.loc);
        return;
      case "split":
        this.elabSplit(s.branches, s.merge, s.loc);
        return;
      case "for":
        this.elabFor(s);
        return;
    }
  }

  requireCursor(loc: Loc, what: string): CVal | null {
    const c = this.frame.cursor;
    if (!c || (c.v !== "val" && c.v !== "tuple")) {
      this.err("AXS0301", `${what} needs an implicit value but none is in scope`, loc, [
        "this entry point has multiple inputs, so there is no single implicit stream",
        "seed the stream explicitly, e.g. `let h = encoder(a)` or `a |> linear(64)`",
      ]);
      return null;
    }
    return c;
  }

  region(label: string, seed: CVal | null, fn: () => CVal | null): { region: IRRegion; result: CVal | null } {
    const f = this.frame;
    const saveScope = new Map(f.scope);
    const saveCursor = f.cursor;
    const saveBound = f.bound;
    const nodes: IRNode[] = [];
    this.nodeStack.push(nodes);
    f.cursor = seed;
    f.bound = new Set();
    const result = fn();
    this.nodeStack.pop();
    f.scope = saveScope;
    f.cursor = saveCursor;
    f.bound = saveBound;
    return { region: { label, nodes, results: result ? this.cvalIds(result) : [] }, result };
  }

  elabResidual(via: Expr | null, body: Stmt[], loc: Loc) {
    const f = this.frame;
    const input = this.requireCursor(loc, "residual");
    if (!input || input.v !== "val") return;
    const { region, result } = this.region("body", input, () => {
      this.elabBody(body);
      return this.frame.cursor;
    });
    if (!result || result.v !== "val") {
      this.err("AXS0303", "residual body produced no tensor", loc);
      return;
    }
    let skipType = input.type;
    const regions = [region];
    let projected: CVal | null = null;
    if (via) {
      const { region: pr, result: prRes } = this.region("projection", input, () => this.elabExpr(via, input));
      regions.push(pr);
      projected = prRes;
      if (prRes && prRes.v === "val") skipType = prRes.type;
    }
    const bt = result.type;
    if (isTensor(bt) && isTensor(skipType)) {
      if (bt.shape.length !== skipType.shape.length || !this.shapeMatches(bt.shape, skipType.shape)) {
        this.err(
          "AXS0404",
          `residual branch produces ${showType(bt)} but the skip path carries ${showType(skipType)}`,
          loc,
          via
            ? ["the explicit projection does not restore the branch shape"]
            : [
                "add an explicit projection: `residual via linear(<out>) { ... }` or `residual via conv2d(<ch>, kernel: 1, stride: <s>) { ... }`",
                "the compiler never invents a learned projection to make shapes fit",
              ]
        );
      }
    }
    const outs = this.emit(
      "residual",
      [input.id],
      { projected: via !== null },
      [bt, ...(projected && projected.v === "val" ? [skipType] : [])],
      { regions, loc, instance: f.inst.id, note: "topology: residual" }
    );
    const skipId = projected && projected.v === "val" ? outs[1].id : input.id;
    const add = this.emit("add", [skipId, outs[0].id], {}, [bt], { loc, note: "residual merge" });
    f.cursor = { v: "val", id: add[0].id, type: bt };
  }

  shapeMatches(a: DimExpr[], b: DimExpr[]): boolean {
    if (a.length !== b.length) return false;
    const env = this.frame.dimEnv;
    return a.every((x, i) => dEquals(substitute(x, env), substitute(b[i], env)));
  }

  elabSplit(branches: Branch[], merge: { kind: string; axis?: number; loc: Loc }, loc: Loc) {
    const f = this.frame;
    const input = this.requireCursor(loc, "split");
    if (!input || input.v !== "val") return;
    const regions: IRRegion[] = [];
    const results: CVal[] = [];
    branches.forEach((b, i) => {
      // every branch receives *the same* incoming value and has its own cursor
      const { region, result } = this.region(b.label ?? `branch${i}`, input, () => {
        this.elabBody(b.body);
        return this.frame.returned ?? this.frame.cursor;
      });
      regions.push(region);
      if (result && result.v === "val") results.push(result);
      else this.err("AXS0303", `branch ${i} of split produced no tensor`, b.loc);
    });
    if (results.length === 0) return;
    const outs = this.emit(
      "parallel",
      [input.id],
      { branches: branches.length, merge: merge.kind },
      results.map((r) => (r as { type: ValueType }).type),
      { regions, loc, instance: f.inst.id, note: "topology: parallel split" }
    );
    // the merge is always an explicit IR node consuming every branch result
    const ins = outs.map((o) => o.id);
    const types = outs.map((o) => o.type);
    let mergedType: ValueType = types[0];
    const attrs: Record<string, IRAttr> = {};
    if (merge.kind === "concat") {
      let ax = merge.axis ?? 1;
      const t0 = types[0];
      if (isTensor(t0)) {
        if (ax < 0) ax += t0.shape.length;
        const shape = t0.shape.slice();
        let sum = t0.shape[ax];
        for (let n = 1; n < types.length; n++) {
          const tn = types[n];
          if (!isTensor(tn)) continue;
          if (tn.shape.length !== shape.length) {
            this.err("AXS0405", `merge concat: branch ${n} has rank ${tn.shape.length}, branch 0 has rank ${shape.length}`, loc);
            continue;
          }
          for (let d = 0; d < shape.length; d++)
            if (d !== ax)
              this.eqDim(
                tn.shape[d],
                shape[d],
                `merge concat(${ax}) branch ${n} axis ${d}`,
                branches[n].loc,
                "AXS0405"
              );
          sum = dAdd(sum, tn.shape[ax]);
        }
        shape[ax] = sum;
        mergedType = tensor(shape, t0.kind);
      }
      attrs.axis = ax;
    } else if (merge.kind === "add" || merge.kind === "mean") {
      const t0 = types[0];
      if (isTensor(t0))
        for (let n = 1; n < types.length; n++) {
          const tn = types[n];
          if (!isTensor(tn)) continue;
          // same code as the concat merge: a branch disagreement is AXS0405 whatever the merge kind
          if (tn.shape.length !== t0.shape.length) {
            this.err("AXS0405", `merge ${merge.kind}: branch ${n} has rank ${tn.shape.length}, branch 0 has rank ${t0.shape.length}`, loc);
            continue;
          }
          for (let d = 0; d < t0.shape.length; d++)
            this.eqDim(tn.shape[d], t0.shape[d], `merge ${merge.kind} branch ${n} axis ${d}`, branches[n].loc, "AXS0405");
        }
    } else {
      this.err("AXS0204", `unknown merge '${merge.kind}'`, merge.loc, ["supported merges: concat(axis), add, mean"]);
    }
    // `merge add`/`merge mean` are n-ary joins; they must not reuse the binary
    // `add` or the reduction `mean` op names (finding F-005)
    const mergeOp = merge.kind === "concat" ? "concat" : merge.kind === "add" ? "merge_add" : "merge_mean";
    const merged = this.emit(mergeOp, ins, attrs, [mergedType], {
      loc,
      note: "explicit branch merge",
    });
    f.cursor = { v: "val", id: merged[0].id, type: mergedType };
  }

  elabFor(s: Extract<Stmt, { s: "for" }>) {
    const f = this.frame;
    const n = s.count !== null ? s.count : s.to - s.from;
    const regions: IRRegion[] = [];
    let cursor = f.cursor;
    let mode = "independent-parameters";
    for (let i = 0; i < n; i++) {
      const idx = s.index ? s.from + i : i;
      const { region, result } = this.region(s.index ? `${s.index}=${idx}` : `iter ${i}`, cursor, () => {
        if (s.index) this.frame.scope.set(s.index, { v: "num", value: idx });
        this.elabBody(s.body);
        return this.frame.cursor;
      });
      regions.push(region);
      if (result && (result.v === "val" || result.v === "tuple")) cursor = result;
    }
    // detect whether the body applies a previously bound stage (=> shared params)
    const bodyIsBoundStage =
      s.body.length === 1 &&
      s.body[0].s === "expr" &&
      s.body[0].value.e === "name" &&
      this.lookup(s.body[0].value.name)?.v === "stage";
    if (bodyIsBoundStage) mode = "shared-stage (one parameter set applied repeatedly)";
    const outTypes = cursor ? [this.cvalType(cursor)] : [];
    const inputs = f.cursor && f.cursor.v === "val" ? [f.cursor.id] : [];
    const outs = this.emit("static_repeat", inputs, { count: n, mode }, outTypes, {
      regions,
      loc: s.loc,
      instance: f.inst.id,
      note: `static architecture repetition ×${n}`,
    });
    if (outs.length) f.cursor = { v: "val", id: outs[0].id, type: outs[0].type };
  }

  lookup(name: string): CVal | undefined {
    return this.frame?.scope.get(name);
  }

  // ---------------------------------------------------------------- expressions
  elabExpr(e: Expr, piped: CVal | null, nameHint?: string): CVal {
    switch (e.e) {
      case "num":
        return { v: "num", value: e.value };
      case "str":
        return { v: "str", value: e.value };
      case "bool":
        return { v: "bool", value: e.value };
      case "tuple":
        return { v: "tuple", items: e.items.map((i) => this.elabExpr(i, null)) };
      case "list":
        return { v: "tuple", items: e.items.map((i) => this.elabExpr(i, null)) };
      case "name": {
        const local = this.lookup(e.name);
        if (local) {
          if (local.v === "stage") {
            // bare application of a bound stage in statement/pipe position
            const inp = piped ?? this.frame.cursor;
            if (inp && (inp.v === "val" || inp.v === "tuple")) return this.applyStage(local.inst, [inp], [], e.loc);
            return local;
          }
          return local;
        }
        if (this.globalDims.has(e.name) || this.runtimeDims.has(e.name) || this.templateVars.has(e.name))
          return { v: "dim", expr: this.resolveDimName(e.name, e.loc) };
        const d = this.decls.get(e.name);
        if (d && (d.d === "block" || d.d === "fn" || d.d === "model")) {
          const inp = piped ?? this.frame.cursor;
          const inst = this.instantiate(d, new Map(), nameHint, e.loc);
          if (inp && (inp.v === "val" || inp.v === "tuple")) return this.applyStage(inst, [inp], [], e.loc);
          return { v: "stage", inst };
        }
        if (CATALOG_MAP.has(e.name)) return this.applyCatalog(CATALOG_MAP.get(e.name)!, [], piped, e.loc);
        if (this.mod.customOps.find((c) => c.name === e.name))
          return this.applyCustom(e.name, [], piped, e.loc);
        this.err("AXS0201", `unknown name '${e.name}'`, e.loc, this.suggest(e.name));
        return { v: "none" };
      }
      case "pipe": {
        let cur: CVal | null = piped;
        for (let i = 0; i < e.stages.length; i++) {
          const st = e.stages[i];
          cur = this.elabExpr(st, i === 0 ? piped : cur);
        }
        return cur ?? { v: "none" };
      }
      case "bin":
        return this.elabBinary(e, piped);
      case "unary": {
        const v = this.elabExpr(e.v, null);
        if (v.v === "num") return { v: "num", value: -v.value };
        if (v.v === "dim") return { v: "dim", expr: dMul(v.expr, dConst(-1)) };
        return this.applyCatalog(CATALOG_MAP.get("neg")!, [], v, e.loc);
      }
      case "index":
        return this.elabIndex(e.obj, e.slices, e.loc);
      case "member": {
        const obj = this.elabExpr(e.obj, null);
        if (obj.v === "tuple") {
          const idx = parseInt(e.name.replace(/^_/, ""), 10);
          if (!isNaN(idx) && obj.items[idx]) return obj.items[idx];
        }
        this.err("AXS0201", `no member '${e.name}' on this value`, e.loc);
        return { v: "none" };
      }
      case "scan":
        return this.elabScan(e);
      case "call":
        return this.elabCall(e, piped, nameHint);
      case "dimref":
        return { v: "dim", expr: this.dimOf(e.dim) };
    }
  }

  suggest(name: string): string[] {
    const names = [...CATALOG_MAP.keys(), ...this.decls.keys()];
    const close = names.filter((n) => n.startsWith(name.slice(0, 3)) || n.includes(name)).slice(0, 4);
    return close.length ? [`did you mean: ${close.join(", ")}?`] : [];
  }

  constArg(e: Expr): CVal {
    if (e.e === "num") return { v: "num", value: e.value };
    if (e.e === "str") return { v: "str", value: e.value };
    if (e.e === "bool") return { v: "bool", value: e.value };
    return this.elabExpr(e, null);
  }

  elabBinary(e: Extract<Expr, { e: "bin" }>, piped: CVal | null): CVal {
    const l = this.elabExpr(e.l, piped);
    const r = this.elabExpr(e.r, null);
    const asDim = (c: CVal): DimExpr | null =>
      c.v === "dim" ? c.expr : c.v === "num" && Number.isInteger(c.value) ? dConst(c.value) : null;
    const dl = asDim(l);
    const dr = asDim(r);
    if (dl && dr && ["+", "-", "*", "/"].includes(e.op)) {
      const both = l.v === "num" && r.v === "num";
      const val =
        e.op === "+" ? dAdd(dl, dr) : e.op === "-" ? dSub(dl, dr) : e.op === "*" ? dMul(dl, dr) : dDiv(dl, dr);
      if (both) {
        const c = asConst(val);
        if (c !== null) return { v: "num", value: e.op === "/" ? (l.value as number) / (r.value as number) : c };
      }
      return { v: "dim", expr: val };
    }
    if (l.v === "num" && r.v === "num") {
      const a = l.value;
      const b = r.value;
      switch (e.op) {
        case "+":
          return { v: "num", value: a + b };
        case "-":
          return { v: "num", value: a - b };
        case "*":
          return { v: "num", value: a * b };
        case "/":
          return { v: "num", value: a / b };
        case "==":
          return { v: "bool", value: a === b };
        case "!=":
          return { v: "bool", value: a !== b };
        case "<":
          return { v: "bool", value: a < b };
        case ">":
          return { v: "bool", value: a > b };
        case "<=":
          return { v: "bool", value: a <= b };
        case ">=":
          return { v: "bool", value: a >= b };
        default:
          return { v: "num", value: a };
      }
    }
    const op = { "+": "add", "-": "sub", "*": "mul", "/": "div" }[e.op];
    if (!op) {
      this.err("AXS0408", `operator '${e.op}' is not defined on tensors`, e.loc);
      return { v: "none" };
    }
    const lv = this.toValue(l, e.loc);
    const rv = this.toValue(r, e.loc);
    if (!lv || !rv) return { v: "none" };
    const lt = lv.type;
    const rt = rv.type;
    let out: ValueType = scalar;
    if ((isTensor(lt) && lt.unknown) || (isTensor(rt) && rt.unknown)) out = unknownTensor();
    else if (isTensor(lt) && isTensor(rt)) out = tensor(this.broadcast(lt.shape, rt.shape, e.loc, op), "Tensor", lt.dtype);
    else if (isTensor(lt)) out = tensor(lt.shape, lt.kind === "Probs" || lt.kind === "Logits" ? "Tensor" : lt.kind, lt.dtype);
    else if (isTensor(rt)) out = tensor(rt.shape, "Tensor", rt.dtype);
    const outs = this.emit(op, [lv.id, rv.id], {}, [out], { loc: e.loc });
    return { v: "val", id: outs[0].id, type: out };
  }

  broadcast(a: DimExpr[], b: DimExpr[], loc: Loc, origin: string): DimExpr[] {
    const n = Math.max(a.length, b.length);
    const out: DimExpr[] = [];
    for (let i = 0; i < n; i++) {
      const x = a[a.length - n + i];
      const y = b[b.length - n + i];
      if (x === undefined) out.push(y);
      else if (y === undefined) out.push(x);
      else if (asConst(x) === 1) out.push(y);
      else if (asConst(y) === 1) out.push(x);
      else {
        this.eqDim(x, y, `${origin} broadcast axis ${i}`, loc);
        out.push(x);
      }
    }
    return out;
  }

  /** materialise a compile-time value as a graph value */
  toValue(c: CVal, loc: Loc): { id: string; type: ValueType } | null {
    if (c.v === "val") return { id: c.id, type: c.type };
    if (c.v === "num") {
      const o = this.emit("const", [], { value: c.value }, [scalar], { loc });
      return { id: o[0].id, type: scalar };
    }
    if (c.v === "dim") {
      const o = this.emit("const_dim", [], { dim: c.expr }, [scalar], { loc, note: `symbolic dim ${show(c.expr)}` });
      return { id: o[0].id, type: scalar };
    }
    if (c.v === "state") {
      const st = this.mod.states.find((s) => s.id === c.id)!;
      const ty = tensor(st.shape);
      const o = this.emit("state_read", [], { state: c.id }, [ty], {
        states: [c.id],
        effects: ["reads-state"],
        loc,
      });
      return { id: o[0].id, type: ty };
    }
    this.err("AXS0408", `expected a tensor value here, found ${c.v}`, loc);
    return null;
  }

  elabIndex(objE: Expr, slices: SliceAst[], loc: Loc): CVal {
    const obj = this.elabExpr(objE, null);
    if (obj.v === "tuple") {
      const s = slices[0];
      if (s && s.s === "index" && s.value.d === "num") return obj.items[s.value.value] ?? { v: "none" };
    }
    const v = this.toValue(obj, loc);
    if (!v || !isTensor(v.type)) {
      this.err("AXS0408", "indexing requires a tensor", loc);
      return { v: "none" };
    }
    const shape = v.type.shape;
    const specs: { axis: number; from: DimExpr; to: DimExpr; drop: boolean }[] = [];
    let ax = 0;
    const ell = slices.findIndex((s) => s.s === "ellipsis");
    const expand = ell >= 0 ? shape.length - (slices.length - 1) : slices.length;
    const outShape: DimExpr[] = [];
    for (let i = 0; i < slices.length; i++) {
      const s = slices[i];
      if (s.s === "ellipsis") {
        for (let k = 0; k < expand; k++) {
          outShape.push(shape[ax]);
          ax++;
        }
        continue;
      }
      if (ax >= shape.length) {
        this.err("AXS0401", `too many indices for rank ${shape.length}`, loc);
        break;
      }
      if (s.s === "all") {
        outShape.push(shape[ax]);
        specs.push({ axis: ax, from: dConst(0), to: shape[ax], drop: false });
      } else if (s.s === "index") {
        const at = this.dimOf(s.value);
        // an index must lie inside the axis: 0 ≤ i and i + 1 ≤ extent (F-011)
        this.leDim(dConst(0), at, `index on axis ${ax} (must be ≥ 0)`, loc);
        this.leDim(dAdd(at, dConst(1)), shape[ax], `index on axis ${ax} (must be < extent)`, loc);
        specs.push({ axis: ax, from: at, to: dAdd(at, dConst(1)), drop: true });
      } else {
        const from = s.from ? this.dimOf(s.from) : dConst(0);
        const to = s.to ? this.dimOf(s.to) : shape[ax];
        // a slice must lie inside the axis and be non-decreasing (F-011)
        if (s.from) this.leDim(dConst(0), from, `slice start on axis ${ax} (must be ≥ 0)`, loc);
        if (s.to) this.leDim(to, shape[ax], `slice end on axis ${ax} (must be ≤ extent)`, loc);
        if (s.from && s.to) this.leDim(from, to, `slice on axis ${ax} (start must not exceed end)`, loc);
        outShape.push(dSub(to, from));
        specs.push({ axis: ax, from, to, drop: false });
      }
      ax++;
    }
    for (; ax < shape.length; ax++) outShape.push(shape[ax]);
    const out = tensor(outShape, v.type.kind, v.type.dtype);
    const outs = this.emit(
      "slice",
      [v.id],
      {
        axes: specs.map((s) => s.axis),
        from: specs.map((s) => s.from),
        to: specs.map((s) => s.to),
        drop: specs.map((s) => (s.drop ? 1 : 0)),
      },
      [out],
      { loc }
    );
    return { v: "val", id: outs[0].id, type: out };
  }

  elabScan(e: Extract<Expr, { e: "scan" }>): CVal {
    const f = this.frame;
    const overV = this.toValue(this.elabExpr(e.over, null), e.loc);
    if (!overV || !isTensor(overV.type)) {
      this.err("AXS0408", "scan requires a tensor to iterate over", e.loc);
      return { v: "none" };
    }
    const axis = e.axis < 0 ? overV.type.shape.length + e.axis : e.axis;
    const stepShape = overV.type.shape.slice();
    const length = stepShape[axis];
    stepShape.splice(axis, 1);
    const stepType = tensor(stepShape, overV.type.kind, overV.type.dtype);
    const carryTypes = e.carry.map((c) => this.typeOf(c.type));
    const stepVal = this.value(stepType, "step");
    const carryVals = e.carry.map((c, i) => this.value(carryTypes[i], c.name));
    const saveScope = new Map(f.scope);
    const saveCursor = f.cursor;
    const saveBound = f.bound;
    const nodes: IRNode[] = [];
    this.nodeStack.push(nodes);
    f.bound = new Set();
    f.cursor = { v: "val", id: stepVal.id, type: stepType };
    f.scope.set("step", { v: "val", id: stepVal.id, type: stepType });
    e.carry.forEach((c, i) => f.scope.set(c.name, { v: "val", id: carryVals[i].id, type: carryTypes[i] }));
    this.elabBody(e.body);
    const yielded = f.scope.get("__yield");
    this.nodeStack.pop();
    const results: CVal[] = yielded
      ? yielded.v === "tuple"
        ? yielded.items
        : [yielded]
      : f.cursor
      ? [f.cursor]
      : [];
    f.scope = saveScope;
    f.cursor = saveCursor;
    f.bound = saveBound;
    if (results.length !== carryTypes.length)
      this.err("AXS1001", `scan yields ${results.length} values but declares ${carryTypes.length} carried state(s)`, e.loc);
    results.forEach((r, i) => {
      const rt = this.cvalType(r);
      const ct = carryTypes[i];
      if (isTensor(rt) && isTensor(ct)) this.eqShape(rt.shape, ct.shape, `scan carry '${e.carry[i]?.name}'`, e.loc);
    });
    const region: IRRegion = { label: "step", nodes, results: results.flatMap((r) => this.cvalIds(r)) };
    const stackedShape = results[0] && isTensor(this.cvalType(results[0])) ? (this.cvalType(results[0]) as TensorType).shape.slice() : [];
    stackedShape.splice(axis, 0, length);
    const stacked = tensor(stackedShape);
    const outs = this.emit(
      "scan",
      [overV.id],
      { axis, length, carries: e.carry.length, init: e.carry.map((c) => c.init) },
      [stacked, ...carryTypes],
      {
        regions: [region],
        loc: e.loc,
        instance: f.inst.id,
        note: "runtime recurrence — one parameter set applied at every step",
      }
    );
    // bind region formals: recorded as attrs so backends know the loop signature
    (outs[0] as IRValue).name = "scan_outputs";
    const node = this.nodes[this.nodes.length - 1];
    node.attrs.step_value = stepVal.id;
    node.attrs.carry_values = carryVals.map((c) => c.id);
    return { v: "tuple", items: outs.map((o) => ({ v: "val" as const, id: o.id, type: o.type })) };
  }

  // ---------------------------------------------------------------- calls
  elabCall(e: Extract<Expr, { e: "call" }>, piped: CVal | null, nameHint?: string): CVal {
    if (e.callee.e !== "name") {
      const target = this.elabExpr(e.callee, null);
      if (target.v === "stage") {
        const args = e.args.map((a) => this.elabExpr(a.value, null));
        return this.applyStage(target.inst, args, [], e.loc);
      }
      this.err("AXS0204", "this expression is not callable", e.loc);
      return { v: "none" };
    }
    const name = e.callee.name;
    const local = this.lookup(name);
    if (local && local.v === "stage") {
      const args = e.args.map((a) => this.elabExpr(a.value, null));
      return this.applyStage(local.inst, args, [], e.loc);
    }
    const spec = CATALOG_MAP.get(name);
    if (spec) return this.applyCatalog(spec, e.args, piped, e.loc);
    if (name === "observe") return this.elabObserve(e.args, e.loc);
    if (this.mod.customOps.find((c) => c.name === name)) return this.applyCustom(name, e.args, piped, e.loc);
    const d = this.decls.get(name);
    if (d && (d.d === "block" || d.d === "fn" || d.d === "model")) {
      const tensorParams = d.params.filter((p) => p.type.t !== "intType" && p.type.t !== "dimType");
      const positional = e.args.filter((a) => !a.name);
      const named = e.args.filter((a) => a.name);
      const config = new Map<string, CVal>();
      for (const a of named) {
        const isTensorPort = tensorParams.some((p) => p.name === a.name);
        if (!isTensorPort) config.set(a.name!, this.constArg(a.value));
      }
      const inst = this.instantiate(d, config, nameHint, e.loc);
      const portArgs = named
        .filter((a) => tensorParams.some((p) => p.name === a.name))
        .map((a) => ({ name: a.name!, val: this.elabExpr(a.value, null) }));
      if (positional.length === 0 && portArgs.length === 0) {
        // `let enc = Encoder(...)` *binds a configured stage*; the same call in
        // flow position applies it to the implicit value.
        if (nameHint !== undefined) return { v: "stage", inst };
        const seed = piped ?? this.frame.cursor;
        if (seed && (seed.v === "val" || seed.v === "tuple")) return this.applyStage(inst, [seed], [], e.loc);
        return { v: "stage", inst };
      }
      const pos = positional.map((a) => this.elabExpr(a.value, null));
      return this.applyStage(inst, pos, portArgs, e.loc);
    }
    this.err("AXS0204", `unknown operation '${name}'`, e.loc, this.suggest(name));
    return { v: "none" };
  }

  elabObserve(args: Arg[], loc: Loc): CVal {
    if (args.length !== 2 || args[0].value.e !== "name") {
      this.err("AXS0503", "observe(state, value) requires a state name and a tensor", loc);
      return { v: "none" };
    }
    const stateRef = this.lookup(args[0].value.name);
    if (!stateRef || stateRef.v !== "state") {
      this.err("AXS0201", `'${args[0].value.name}' is not a persistent state declared in this stage`, loc);
      return { v: "none" };
    }
    const st = this.mod.states.find((s) => s.id === stateRef.id)!;
    const obs = this.toValue(this.elabExpr(args[1].value, null), loc);
    if (!obs) return { v: "none" };
    if (isTensor(obs.type)) this.eqShape(obs.type.shape, st.shape, `observation for state '${st.role}'`, loc);
    const ty = tensor(st.shape);
    const outs = this.emit("state_update", [obs.id], { state: st.id, rule: st.update }, [ty], {
      states: [st.id],
      effects: ["reads-state", "writes-state", "training-sensitive"],
      loc,
      note: `persistent state '${st.role}' updated with ${st.update} during training; read-only during evaluation`,
    });
    return { v: "val", id: outs[0].id, type: ty };
  }

  instantiate(d: FnLikeDecl, config: Map<string, CVal>, nameHint: string | undefined, loc: Loc): Instance {
    const f = this.frame;
    const n = ++f.inst.counter;
    let path = nameHint ? `${f.inst.path}/${nameHint}` : `${f.inst.path}/${d.name}#${n}`;
    // instance paths key parameter identity: a rebound `let` name must not
    // silently tie its parameters to the earlier binding's (finding F-001)
    if (this.instances.some((i) => i.path === path)) path = `${path}#${n}`;
    void loc;
    return this.newInstance(d, path, config, nameHint ?? `${d.name}#${n}`);
  }

  applyStage(inst: Instance, positional: CVal[], ports: { name: string; val: CVal }[], loc: Loc): CVal {
    const d = inst.decl;
    const tensorParams = d.params.filter((p) => p.type.t !== "intType" && p.type.t !== "dimType");
    const args: CVal[] = [];
    let pi = 0;
    for (const p of tensorParams) {
      const port = ports.find((x) => x.name === p.name);
      if (port) args.push(port.val);
      else if (pi < positional.length) args.push(positional[pi++]);
      else {
        this.err("AXS0503", `'${d.name}' expects ${tensorParams.length} input(s), received ${positional.length + ports.length}`, loc);
        return { v: "none" };
      }
    }
    const firstApplication = inst.applications.length === 0;
    inst.applications.push(this.where());
    inst.counter = 0;

    const dimEnv = new Map<string, DimExpr>();
    const tvars = this.collectTemplateVars(d.params);
    const savedFlexible = new Set(this.flexible);
    for (const v of tvars) this.flexible.add(v);
    const savedTemplates = this.templateVars;
    this.templateVars = new Set(tvars);

    const frame: Frame = { scope: new Map(), dimEnv, inst, cursor: null, returned: null };
    // configuration values (Dim / Int parameters) are bound before the shapes are read
    for (const p of d.params)
      if (p.type.t === "intType" || p.type.t === "dimType") {
        const cv = inst.config.get(p.name) ?? (p.def ? this.constArg(p.def) : { v: "num" as const, value: 0 });
        frame.scope.set(p.name, cv);
        const dv = cv.v === "num" ? dConst(cv.value) : cv.v === "dim" ? cv.expr : dConst(0);
        dimEnv.set(p.name, dv);
      }
    this.frames.push(frame);
    // unify declared parameter shapes with the actual arguments
    const inputIds: string[] = [];
    tensorParams.forEach((p, i) => {
      const declared = this.typeOf(p.type);
      const actual = args[i];
      const av = actual ? this.toValue(actual, loc) : null;
      if (!av) return;
      inputIds.push(av.id);
      if (isTensor(declared) && isTensor(av.type)) {
        if (av.type.unknown)
          this.info("AXS0903", `'${d.name}' input '${p.name}' receives a value of unknown shape; its contract is not checked`, loc);
        else if (declared.shape.length !== av.type.shape.length)
          this.err(
            "AXS0402",
            `'${d.name}' input '${p.name}' declares rank ${declared.shape.length}, received ${showType(av.type)}`,
            loc
          );
        else
          declared.shape.forEach((ds, k) =>
            this.eqDim(ds, (av.type as TensorType).shape[k], `'${d.name}' input '${p.name}' axis ${k}`, loc)
          );
        if (declared.kind !== "Tensor" && av.type.kind !== declared.kind && av.type.kind !== "Tensor")
          this.warn(
            "AXS0409",
            `'${d.name}' input '${p.name}' declares ${declared.kind}[...] but received ${av.type.kind}[...]`,
            loc
          );
        frame.scope.set(p.name, { v: "val", id: av.id, type: av.type });
      } else frame.scope.set(p.name, { v: "val", id: av.id, type: av.type });
    });
    frame.cursor = tensorParams.length === 1 && args[0] ? frame.scope.get(tensorParams[0].name) ?? null : null;

    const nodes: IRNode[] = [];
    this.nodeStack.push(nodes);
    this.elabBody(d.body);
    const result = frame.returned ?? frame.cursor;
    this.nodeStack.pop();
    this.frames.pop();
    this.flexible = savedFlexible;
    this.templateVars = savedTemplates;

    if (!result || result.v === "none") {
      this.err("AXS0303", `'${d.name}' produced no value`, loc);
      return { v: "none" };
    }
    const rt = this.cvalType(result);
    if (d.ret) {
      this.frames.push(frame);
      this.checkResult(this.typeOf(d.ret), rt, loc, d.name);
      this.frames.pop();
    }
    const sigOk = this.instanceSignature(inst, args, loc);
    if (!firstApplication && !sigOk)
      this.err(
        "AXS0801",
        `stage '${inst.path}' is applied to incompatible input shapes, so its parameters cannot be shared`,
        loc,
        ["create a second stage value if the two call sites should have independent parameters"]
      );
    const outTypes = result.v === "tuple" ? result.items.map((i) => this.cvalType(i)) : [rt];
    const region: IRRegion = { label: `${d.d} ${d.name}`, nodes, results: this.cvalIds(result) };
    const outs = this.emit("apply", inputIds, { stage: inst.path, decl: d.name }, outTypes, {
      regions: [region],
      loc,
      instance: inst.id,
      note: `${d.d} ${d.name}${inst.applications.length > 1 ? ` (application #${inst.applications.length}, parameters shared)` : ""}`,
    });
    if (outs.length === 1) return { v: "val", id: outs[0].id, type: outs[0].type };
    return { v: "tuple", items: outs.map((o) => ({ v: "val" as const, id: o.id, type: o.type })) };
  }

  instanceSignature(inst: Instance, args: CVal[], loc: Loc): boolean {
    void loc;
    const key = args
      .map((a) => (a.v === "val" && isTensor(a.type) ? a.type.shape.map(show).join("x") : "?"))
      .join("|");
    const store = inst as Instance & { sig?: string };
    if (store.sig === undefined) {
      store.sig = key;
      return true;
    }
    return store.sig === key;
  }

  // ---------------------------------------------------------------- catalog application
  applyCatalog(spec: OpSpec, args: Arg[], piped: CVal | null, loc: Loc): CVal {
    const f = this.frame;
    const n = ++f.inst.counter;
    const positional: CVal[] = [];
    const named = new Map<string, CVal>();
    for (const a of args) {
      const val = a.value.e === "list" ? { v: "tuple" as const, items: a.value.items.map((i) => this.elabExpr(i, null)) } : this.elabExpr(a.value, null);
      if (a.name) named.set(a.name, val);
      else positional.push(val);
    }
    // tensor inputs
    const tensorIns: { id: string; type: ValueType }[] = [];
    const portNames = [...spec.ports, ...(spec.optionalPorts ?? [])];
    const usedPositional = new Set<number>();
    if (spec.style === "primitive") {
      let pi = 0;
      for (const p of portNames) {
        if (named.has(p)) {
          const v = this.toValue(named.get(p)!, loc);
          if (v) tensorIns.push(v);
          continue;
        }
        if (pi < positional.length) {
          const cand = positional[pi];
          // dims and numbers used as tensor arguments (e.g. `sqrt(DH)`) are
          // materialised as scalar constants instead of stealing the cursor
          if (cand.v === "val" || cand.v === "state" || cand.v === "dim" || cand.v === "num") {
            const v = this.toValue(cand, loc);
            if (v) tensorIns.push(v);
            usedPositional.add(pi);
            pi++;
          }
        }
      }
      if (tensorIns.length < spec.ports.length && piped) {
        const v = this.toValue(piped, loc);
        if (v) tensorIns.unshift(v);
      }
      if (tensorIns.length < spec.ports.length && !piped && f.cursor && spec.ports.length > 0) {
        const v = this.toValue(f.cursor, loc);
        if (v) tensorIns.unshift(v);
      }
    } else {
      const seed = piped ?? f.cursor;
      if (spec.ports.length > 0) {
        if (named.has(spec.ports[0])) {
          const v = this.toValue(named.get(spec.ports[0])!, loc);
          if (v) tensorIns.push(v);
        } else if (seed) {
          const v = this.toValue(seed, loc);
          if (v) tensorIns.push(v);
        } else {
          this.err("AXS0301", `'${spec.name}' needs an incoming tensor but no implicit value is in scope`, loc, [
            "pipe a value into it (`x |> " + spec.name + "(...)`) or pass it by port name",
          ]);
          return { v: "none" };
        }
      }
      // optional ports keep their position: `attention(mask: m)` must not slide
      // the mask into the key slot (F-013).  A skipped port defaults to the one
      // before it (key <- query, value <- key), which is the catalog's own rule.
      const opts = spec.optionalPorts ?? [];
      const lastNamed = opts.reduce((acc, p, i) => (named.has(p) ? i : acc), -1);
      opts.forEach((p, i) => {
        if (i > lastNamed) return;
        if (named.has(p)) {
          const v = this.toValue(named.get(p)!, loc);
          if (v) tensorIns.push(v);
        } else if (tensorIns.length) tensorIns.push(tensorIns[tensorIns.length - 1]);
      });
    }
    // config
    const cfg = new Map<string, CVal>();
    let posIdx = 0;
    for (const c of spec.config) {
      if (named.has(c.name)) {
        cfg.set(c.name, named.get(c.name)!);
        continue;
      }
      if (c.pos !== undefined) {
        while (posIdx < positional.length && usedPositional.has(posIdx)) posIdx++;
        if (posIdx < positional.length) {
          cfg.set(c.name, positional[posIdx]);
          usedPositional.add(posIdx);
          posIdx++;
        }
      }
    }
    for (const [k] of named)
      if (!spec.config.some((c) => c.name === k) && !portNames.includes(k))
        this.err("AXS0502", `'${spec.name}' has no argument '${k}'`, loc, [
          `accepted: ${[...spec.config.map((c) => c.name), ...portNames].join(", ") || "none"}`,
        ]);
    for (const c of spec.config)
      if (c.required && !cfg.has(c.name))
        this.err("AXS0501", `'${spec.name}' requires '${c.name}'`, loc, [c.doc ?? "this argument is an architectural choice, not a consequence"]);

    const ins = tensorIns.filter((t) => isTensor(t.type)).map((t) => t.type as TensorType);
    const ctx: InferCtx = {
      ins,
      raw: tensorIns.map((t) => t.type),
      num: (name, def = 0) => {
        const v = cfg.get(name);
        if (!v) return def;
        if (v.v === "num") return v.value;
        if (v.v === "dim") {
          const c = asConst(substitute(v.expr, f.dimEnv));
          if (c !== null) return c;
          this.err("AXS0407", `'${spec.name}' argument '${name}' must be a compile-time constant, found ${show(v.expr)}`, loc);
        }
        return def;
      },
      dim: (name, def = dConst(0)) => {
        const v = cfg.get(name);
        if (!v) return def;
        if (v.v === "dim") return substitute(v.expr, f.dimEnv);
        if (v.v === "num") return dConst(v.value);
        this.err("AXS0408", `'${spec.name}' argument '${name}' must be a dimension`, loc);
        return def;
      },
      bool: (name, def) => {
        const v = cfg.get(name);
        return v && v.v === "bool" ? v.value : def;
      },
      str: (name, def) => {
        const v = cfg.get(name);
        return v && v.v === "str" ? v.value : def;
      },
      dims: (name) => {
        const v = cfg.get(name);
        if (!v) return null;
        if (v.v === "tuple")
          return v.items.map((i) =>
            i.v === "dim" ? substitute(i.expr, f.dimEnv) : i.v === "num" ? dConst(i.value) : dConst(0)
          );
        if (v.v === "dim") return [substitute(v.expr, f.dimEnv)];
        return null;
      },
      has: (name) => cfg.has(name),
      err: (code, msg, notes) => this.err(code, `${spec.name}: ${msg}`, loc, notes),
      warn: (code, msg, notes) => this.warn(code, `${spec.name}: ${msg}`, loc, notes),
      eq: (a, b, origin) => this.eqDim(a, b, `${spec.name}: ${origin}`, loc),
      loc,
    };
    // an unknown shape stays unknown: no rule is applied, no parameter is sized,
    // nothing downstream is invented (§5.2 outcome "unknown")
    const res = ins.some((t) => t.unknown) ? { out: unknownTensor() } : spec.infer(ctx);
    // parameters & state
    const paramIds: string[] = [];
    const base = `${f.inst.path}/${spec.name}#${n}`;
    for (const p of res.params ?? []) {
      const id = `${base}.${p.role}`;
      const existing = this.mod.params.find((x) => x.id === id);
      if (existing) existing.applications.push(this.where());
      else {
        this.mod.params.push({
          id,
          owner: base,
          role: p.role,
          shape: p.shape.map((s) => substitute(s, f.dimEnv)),
          init: p.init,
          trainable: p.trainable ?? true,
          kind: p.kind,
          applications: [this.where()],
        });
        f.inst.params.push(id);
      }
      paramIds.push(id);
    }
    const stateIds: string[] = [];
    for (const s of res.states ?? []) {
      const id = `${base}.${s.role}`;
      if (!this.mod.states.find((x) => x.id === id)) {
        this.mod.states.push({
          id,
          owner: base,
          role: s.role,
          shape: s.shape.map((x) => substitute(x, f.dimEnv)),
          init: s.init,
          update: s.update,
          checkpointed: true,
          category: s.category,
        });
        f.inst.states.push(id);
      }
      stateIds.push(id);
    }
    const effects: Effect[] = [...(res.effects ?? spec.effects)];
    const attrs = { ...(res.attrs ?? {}) };
    const outs = this.emit(spec.name, tensorIns.map((t) => t.id), attrs, [res.out], {
      params: paramIds,
      states: stateIds,
      effects,
      loc,
      instance: f.inst.id,
    });
    return { v: "val", id: outs[0].id, type: res.out };
  }

  applyCustom(name: string, args: Arg[], piped: CVal | null, loc: Loc): CVal {
    const spec = this.mod.customOps.find((c) => c.name === name)!;
    const vals: { id: string; type: ValueType }[] = [];
    const seed = piped ?? this.frame.cursor;
    const positional = args.filter((a) => !a.name);
    if (positional.length < spec.inputs.length && seed) {
      const v = this.toValue(seed, loc);
      if (v) vals.push(v);
    }
    for (const a of positional) {
      const v = this.toValue(this.elabExpr(a.value, null), loc);
      if (v) vals.push(v);
    }
    let out: ValueType;
    if (spec.shapeUnknown || !spec.result) {
      out = unknownTensor();
      this.warn("AXS0903", `custom op '${name}' does not declare output shape semantics — downstream shapes are unknown`, loc, [
        "declare `-> Tensor[...]` on the custom op to keep static shape knowledge",
      ]);
    } else out = spec.result;
    const outs = this.emit(name, vals.map((v) => v.id), { custom: true }, [out], {
      effects: spec.effects,
      loc,
      note: `custom operation (backends: ${spec.backends.map((b) => b.target).join(", ") || "none"})`,
    });
    if (!spec.differentiable)
      this.info("AXS0902", `custom op '${name}' declares no gradient; parameters upstream of it receive no updates through this path`, loc);
    // three different things (§15): `nondiff` is mathematics, `grad-stopped`
    // is intent, and THIS is a backend gap — the reference backend has no
    // implementation of a custom op, whatever the torch backend can do
    if (!spec.backends.some((b) => b.target === "reference"))
      this.info("AXS0901", `custom op '${name}' has no reference implementation; the reference backend substitutes an identity/zero tensor for it (declared backends: ${spec.backends.map((b) => b.target).join(", ") || "none"})`, loc);
    return { v: "val", id: outs[0].id, type: out };
  }

  // ---------------------------------------------------------------- data
  elabData(d: DataDecl) {
    const srcDecl = this.decls.get(d.source);
    const source =
      srcDecl && srcDecl.d === "source"
        ? { adapter: srcDecl.adapter, args: srcDecl.args.map((a) => `${a.name ? a.name + ": " : ""}${exprText(a.value)}`).join(", ") }
        : { adapter: d.source || "<unbound>", args: "" };
    if (!srcDecl && d.source) this.err("AXS0201", `unknown source '${d.source}'`, d.loc);
    const fields: IRDataField[] = [];
    const fitted: { stat: string; field: string; split: string; loc: Loc }[] = [];
    const splits: { name: string; frac: number }[] = [];
    let batch = 32;
    let shuffle = true;
    const example = d.body.find((b) => b.ds === "example");
    if (!example) this.err("AXS0604", `data '${d.name}' declares no example construction`, d.loc, ["add an `example { field ... }` block"]);
    const fieldTypes = new Map<string, ValueType>();
    if (example && example.ds === "example")
      for (const fdecl of example.fields) {
        const ty = this.typeOf(fdecl.type);
        // a field may be derived from the fields declared before it (`inputs = ids[0 : T]`)
        const pipeline = this.dataPipeline(fdecl.value, d.name, fdecl.name, "both", fitted, fieldTypes);
        fieldTypes.set(fdecl.name, ty);
        fields.push({ name: fdecl.name, type: ty, pipeline, stochasticInTrain: pipeline.some((p) => p.effects.includes("stochastic")) });
      }
    for (const b of d.body) {
      if (b.ds === "stage") {
        for (const entry of b.entries) {
          const f = fields.find((x) => x.name === entry.field);
          if (!f) {
            this.err("AXS0604", `data '${d.name}' has no field '${entry.field}'`, entry.loc, [
              `declared fields: ${fields.map((x) => x.name).join(", ")}`,
            ]);
            continue;
          }
          const mode = b.mode === "both" ? (b.stage === "augment" ? "train" : "both") : b.mode;
          const steps = this.dataPipeline(entry.pipe, d.name, entry.field, mode, fitted);
          for (const st of steps) {
            if (b.stage === "augment" && mode === "eval" && st.effects.includes("stochastic"))
              this.warn("AXS0621", `stochastic '${st.op}' placed in the eval pipeline of '${entry.field}'`, entry.loc, [
                "evaluation transforms are normally deterministic; move it to `augment train { ... }`",
              ]);
          }
          f.pipeline.push(...steps);
          f.stochasticInTrain = f.pipeline.some((p) => p.effects.includes("stochastic") && p.split !== "eval");
        }
      } else if (b.ds === "split") splits.push(...b.parts);
      else if (b.ds === "batch") batch = b.size;
      else if (b.ds === "shuffle") shuffle = b.value;
    }
    for (const fit of fitted)
      if (fit.split !== "train")
        this.err(
          "AXS0620",
          fit.split === "unspecified"
            ? `statistic '${fit.stat}' for field '${fit.field}' does not say which split it is fitted on`
            : `statistic '${fit.stat}' for field '${fit.field}' is fitted on split '${fit.split}'`,
          fit.loc,
          [
            "fitting normalisation statistics or vocabularies outside the training split leaks information",
            "write `fit: train`" + (fit.split === "unspecified" ? " (a misspelled `fit:` argument is silently ignored)" : ""),
          ]
        );
    this.mod.data.push({ name: d.name, source, fields, splits, batch, shuffle, fitted: fitted.map(({ stat, field, split }) => ({ stat, field, split })) });
  }

  dataPipeline(
    e: Expr,
    dataName: string,
    field: string,
    split: string,
    fitted: { stat: string; field: string; split: string; loc: Loc }[],
    earlierFields?: Map<string, ValueType>
  ): { op: string; args: string; effects: Effect[]; split: string }[] {
    const out: { op: string; args: string; effects: Effect[]; split: string }[] = [];
    const visit = (x: Expr) => {
      if (x.e === "pipe") {
        x.stages.forEach(visit);
        return;
      }
      if (x.e === "name") {
        if (earlierFields?.has(x.name)) {
          out.push({ op: `field.${x.name}`, args: "", effects: ["pure"], split });
          return;
        }
        const spec = DATA_OP_MAP.get(x.name);
        if (!spec) {
          if (!["image", "label", "text", "row", "features", "value"].includes(x.name))
            this.warn("AXS0204", `unknown data operation '${x.name}' in ${dataName}.${field}`, x.loc, [
              `known: ${[...DATA_OP_MAP.keys()].slice(0, 8).join(", ")} ...`,
            ]);
          out.push({ op: `source.${x.name}`, args: "", effects: ["pure"], split });
          return;
        }
        out.push({ op: x.name, args: "", effects: spec.stochastic ? ["stochastic"] : ["pure"], split });
        return;
      }
      if (x.e === "call") {
        if (x.callee.e === "name") {
          const spec = DATA_OP_MAP.get(x.callee.name);
          const argText = x.args.map((a) => `${a.name ? a.name + ": " : ""}${exprText(a.value)}`).join(", ");
          const fitArg = x.args.find((a) => a.name === "fit");
          if (spec?.fits) {
            // `impute(median, fit: train)` fits the median, not "median|mean"
            const strategy = x.args.find((a) => !a.name && a.value.e === "name");
            const stat = spec.fits.includes("|") && strategy && strategy.value.e === "name" ? strategy.value.name : spec.fits;
            fitted.push({ stat, field, split: fitArg ? exprText(fitArg.value) : "unspecified", loc: x.loc });
            // a vocabulary fitted on train meets unseen categories at eval time;
            // what happens then is part of the learning problem (§19)
            if (spec.name === "vocab" && !x.args.some((a) => a.name === "unknown"))
              this.warn("AXS0622", `vocabulary for field '${field}' declares no policy for categories unseen when it was fitted`, x.loc, [
                'write `unknown: "<unk>"` to map them to one token, or `unknown: error` to refuse them',
              ]);
          }
          if (!spec)
            this.warn("AXS0204", `unknown data operation '${x.callee.name}' in ${dataName}.${field}`, x.loc);
          out.push({
            op: x.callee.name,
            args: argText,
            effects: spec?.stochastic ? ["stochastic"] : ["pure"],
            split,
          });
          x.args.forEach((a) => {
            if (a.value.e === "call" || a.value.e === "pipe") visit(a.value);
          });
          return;
        }
      }
      if (x.e === "index") {
        visit(x.obj);
        out.push({ op: "slice", args: sliceText(x.slices), effects: ["pure"], split });
        return;
      }
      if (x.e === "bin") {
        visit(x.l);
        visit(x.r);
        return;
      }
      out.push({ op: "expr", args: exprText(x), effects: ["pure"], split });
    };
    visit(e);
    return out;
  }

  // ---------------------------------------------------------------- training plan
  elabTrain(d: TrainDecl) {
    const plan = {
      name: d.name,
      data: null as string | null,
      models: [] as { alias: string; model: string }[],
      losses: [] as IRLoss[],
      optimizers: [] as IROptimizer[],
      tracks: [] as { name: string; kind: string; region: string; args: string }[],
      phases: [] as IRPhase[],
      events: [] as IRPhase["events"],
      settings: {} as Record<string, string>,
    };
    let dataDecl: IRData | undefined;
    const modelOf = new Map<string, IRGraph>();
    for (const st of d.body) {
      if (st.t === "data") {
        plan.data = st.name;
        dataDecl = this.mod.data.find((x) => x.name === st.name);
        if (!dataDecl) this.err("AXS0201", `unknown data declaration '${st.name}'`, st.loc);
      } else if (st.t === "model") {
        const g = this.mod.graphs.find((x) => x.name === st.name && x.kind === "model");
        if (!g) this.err("AXS0201", `unknown model '${st.name}'`, st.loc);
        else {
          // a model declaration is ONE parameter set: two aliases of it share
          // every parameter, in the IR, the reference runtime and the emitted
          // module alike (F-015)
          const prev = plan.models.find((m) => m.model === st.name);
          if (prev)
            this.warn("AXS0706", `model '${st.name}' is bound as both '${prev.alias}' and '${st.alias}'; the two aliases share one parameter set`, st.loc, [
              `for independent copies declare two \`model\`s that each instantiate the same block (\`model Teacher(x) { Body() }\`)`,
              `drop the second alias if one parameter set applied twice is what you mean`,
            ]);
          modelOf.set(st.alias, g);
          plan.models.push({ alias: st.alias, model: st.name });
        }
      } else if (st.t === "setting") plan.settings[st.key] = exprText(st.value);
    }
    // losses
    for (const st of d.body) {
      if (st.t !== "loss") continue;
      const obj = this.mod.objectives.find((o) => o.name === st.objective);
      if (!obj) {
        this.err("AXS0201", `unknown objective '${st.objective}'`, st.loc, [
          `declared objectives: ${this.mod.objectives.map((o) => o.name).join(", ") || "none"}`,
        ]);
        continue;
      }
      const bindings: IRLoss["bindings"] = [];
      // a misspelled port must not vanish silently behind "port X is not supplied"
      for (const a of st.args)
        if (a.name && !obj.inputs.some((p) => p.name === a.name))
          this.err("AXS0503", `objective '${obj.name}' has no port '${a.name}'`, a.loc, [
            `ports: ${obj.inputs.map((p) => p.name).join(", ")}`,
          ]);
      obj.inputs.forEach((port, i) => {
        const arg = st.args.find((a) => a.name === port.name) ?? st.args.filter((a) => !a.name)[i];
        if (!arg) {
          this.err("AXS0503", `objective '${obj.name}' port '${port.name}' is not supplied`, st.loc);
          return;
        }
        const { type, text, kind } = this.bindingType(arg, modelOf, dataDecl, st.loc);
        bindings.push({ port: port.name, kind, text, type });
        if (type && isTensor(type) && isTensor(port.type)) {
          if (type.shape.length !== port.type.shape.length)
            this.err(
              kind === "model" ? "AXS0602" : "AXS0603",
              `objective '${obj.name}' port '${port.name}' expects ${showType(port.type)} but receives ${showType(type)}`,
              arg.loc
            );
          else {
            for (let k = 1; k < type.shape.length; k++)
              this.eqDim(
                type.shape[k],
                port.type.shape[k],
                `objective '${obj.name}' port '${port.name}' axis ${k}`,
                arg.loc,
                kind === "model" ? "AXS0602" : "AXS0603"
              );
            if (port.type.kind !== "Tensor" && type.kind !== port.type.kind && type.kind !== "Tensor")
              this.warn(
                kind === "model" ? "AXS0602" : "AXS0603",
                `objective '${obj.name}' port '${port.name}' declares ${port.type.kind}[...] but receives ${type.kind}[...]`,
                arg.loc
              );
          }
        }
      });
      plan.losses.push({ name: st.name, objective: st.objective, bindings, weight: st.weight });
    }
    // optimizers
    for (const st of d.body) {
      if (st.t !== "optimizer") continue;
      // no region: the optimizer covers every model bound in the plan, not just the first one
      const regionText = st.region ? exprText(st.region) : "*";
      const params = this.resolveRegion(regionText, modelOf, st.loc);
      if (params.length === 0)
        this.err("AXS0702", `optimizer '${st.name}' covers no trainable parameters (region '${regionText}')`, st.loc, [
          `known parameter regions: ${this.regionNames().join(", ")}`,
        ]);
      const lrArg = st.args.find((a) => a.name === "lr") ?? st.args[0];
      const args: Record<string, number> = {};
      for (const a of st.args) if (a.name && a.value.e === "num") args[a.name] = a.value.value;
      plan.optimizers.push({
        name: st.name,
        kind: st.kind,
        lr: lrArg && lrArg.value.e === "num" ? lrArg.value.value : 1e-3,
        args,
        region: regionText,
        params: params.map((p) => p.id),
        stateSlots: st.kind.startsWith("adam") ? ["exp_avg", "exp_avg_sq", "step"] : st.kind === "sgd" ? ["momentum"] : [],
      });
    }
    for (const st of d.body)
      if (st.t === "track") {
        const regionText = exprText(st.region);
        const params = this.resolveRegion(regionText, modelOf, st.loc);
        if (!params.length) this.err("AXS0701", `track '${st.name}': region '${regionText}' matches no parameters`, st.loc);
        plan.tracks.push({
          name: st.name,
          kind: st.kind,
          region: regionText,
          args: st.args.map((a) => `${a.name ? a.name + ": " : ""}${exprText(a.value)}`).join(", "),
        });
        for (const p of params)
          this.mod.states.push({
            id: `${st.name}.${p.id}`,
            owner: st.name,
            role: `${st.kind} of ${p.id}`,
            shape: p.shape,
            init: "copy",
            update: `${st.kind}(${st.args.map((a) => exprText(a.value)).join(", ")})`,
            checkpointed: true,
            category: "ema",
          });
      }
    for (const st of d.body)
      if (st.t === "every")
        plan.events.push({ every: st.n, unit: st.unit, actions: st.actions.map((a) => ({ kind: a.kind, args: a.args.map((x) => exprText(x.value)).join(", ") })) });

    // phases — `freeze` persists into later phases until an `unfreeze`
    const frozenNow = new Set<string>();
    for (const st of d.body) {
      if (st.t !== "phase") continue;
      let sawEpochs = false;
      let sawSteps = false;
      let sawUntil = false;
      const ph: IRPhase = {
        name: st.name,
        epochs: 1,
        steps: null,
        until: null,
        frozen: [],
        unfrozen: [],
        lrOverrides: [],
        updates: [],
        events: [],
        schedule: null,
      };
      for (const p of st.body) {
        switch (p.p) {
          case "epochs":
            ph.epochs = p.n;
            if (sawSteps)
              this.err("AXS0707", `phase '${st.name}' declares both \`epochs\` and \`steps\`; a phase has one length`, p.loc);
            sawEpochs = true;
            break;
          case "steps":
            ph.steps = p.n;
            if (sawEpochs)
              this.err("AXS0707", `phase '${st.name}' declares both \`epochs\` and \`steps\`; a phase has one length`, p.loc);
            sawSteps = true;
            break;
          case "freeze":
          case "unfreeze": {
            const text = exprText(p.region);
            if (this.resolveRegion(text, modelOf, p.loc).length === 0)
              this.err("AXS0701", `${p.p}: region '${text}' matches no parameters`, p.loc, [
                `known parameter regions: ${this.regionNames().join(", ")}`,
              ]);
            const other = p.p === "freeze" ? ph.unfrozen : ph.frozen;
            if (other.includes(text))
              this.err("AXS0707", `phase '${st.name}' both freezes and unfreezes '${text}'`, p.loc, [
                "a region is frozen or trainable for the whole phase; split the phase if it should change",
              ]);
            (p.p === "freeze" ? ph.frozen : ph.unfrozen).push(text);
            break;
          }
          case "lr": {
            const text = exprText(p.region);
            if (this.resolveRegion(text, modelOf, p.loc).length === 0)
              this.err("AXS0701", `lr: region '${text}' matches no parameters`, p.loc);
            ph.lrOverrides.push({ region: text, lr: p.value });
            break;
          }
          case "update": {
            if (!plan.losses.find((l) => l.name === p.loss))
              this.err("AXS0201", `phase '${st.name}' updates unknown loss '${p.loss}'`, p.loc, [
                `declared losses: ${plan.losses.map((l) => l.name).join(", ") || "none"}`,
              ]);
            const opts = p.opt.length ? p.opt : plan.optimizers.map((o) => o.name);
            for (const o of opts)
              if (!plan.optimizers.find((x) => x.name === o))
                this.err("AXS0201", `unknown optimizer '${o}'`, p.loc);
            ph.updates.push({ loss: p.loss, optimizers: opts, times: p.times });
            break;
          }
          case "until": {
            if (sawUntil)
              this.err("AXS0707", `phase '${st.name}' declares two \`until\` conditions; only one stopping rule is allowed`, p.loc);
            sawUntil = true;
            const metrics = plan.losses.flatMap((l) => [l.name, `train_${l.name}`, `val_${l.name}`]);
            if (!metrics.includes(p.metric))
              this.err("AXS0709", `phase '${st.name}': \`until ${p.metric}\` names a metric no loss or validation produces`, p.loc, [
                `available metrics: ${metrics.join(", ") || "none"}`,
              ]);
            ph.until = { metric: p.metric, cmp: p.cmp, value: p.value };
            break;
          }
          case "every":
            ph.events.push({ every: p.n, unit: p.unit, actions: p.actions.map((a) => ({ kind: a.kind, args: a.args.map((x) => exprText(x.value)).join(", ") })) });
            break;
          case "schedule":
            ph.schedule = `${p.kind}(${p.args.map((a) => `${a.name ? a.name + ": " : ""}${exprText(a.value)}`).join(", ")})`;
            break;
        }
      }
      // parameters claimed twice inside one phase
      const claimed = new Map<string, string>();
      for (const u of ph.updates)
        for (const oname of u.optimizers) {
          const o = plan.optimizers.find((x) => x.name === oname);
          if (!o) continue;
          for (const pid of o.params) {
            const prev = claimed.get(pid);
            if (prev && prev !== oname)
              this.warn("AXS0703", `parameter ${pid} is updated by both '${prev}' and '${oname}' in phase '${st.name}'`, st.loc);
            claimed.set(pid, oname);
          }
        }
      if (ph.updates.length === 0 && plan.losses.length)
        ph.updates.push({ loss: plan.losses[0].name, optimizers: plan.optimizers.map((o) => o.name), times: 1 });
      // frozen set as this phase sees it: inherited, then this phase's changes
      for (const r of ph.frozen) for (const p of this.resolveRegion(r, modelOf, st.loc)) frozenNow.add(p.id);
      for (const r of ph.unfrozen) for (const p of this.resolveRegion(r, modelOf, st.loc)) frozenNow.delete(p.id);
      for (const ov of ph.lrOverrides) {
        const ps = this.resolveRegion(ov.region, modelOf, st.loc);
        if (ps.length && ps.every((p) => frozenNow.has(p.id)))
          this.warn("AXS0707", `phase '${st.name}' sets a learning rate for '${ov.region}' but that region is frozen in this phase`, st.loc);
      }
      // an update whose optimizers reach only frozen parameters trains nothing (F-014)
      for (const u of ph.updates) {
        const reach = u.optimizers.flatMap((n) => plan.optimizers.find((o) => o.name === n)?.params ?? []);
        if (reach.length && reach.every((pid) => frozenNow.has(pid)))
          this.err(
            "AXS0704",
            `phase '${st.name}' updates '${u.loss}' with ${u.optimizers.map((o) => `'${o}'`).join(", ")}, but every parameter ${u.optimizers.length > 1 ? "they cover" : "it covers"} is frozen in this phase`,
            st.loc,
            ["unfreeze the region in this phase, or update with an optimizer over a trainable region"]
          );
      }
      plan.phases.push(ph);
    }
    if (plan.phases.length === 0) {
      const epochs = plan.settings.epochs ? parseFloat(plan.settings.epochs) : 1;
      plan.phases.push({
        name: "main",
        epochs,
        steps: plan.settings.steps ? parseFloat(plan.settings.steps) : null,
        until: null,
        frozen: [],
        unfrozen: [],
        lrOverrides: [],
        updates: plan.losses.map((l) => ({ loss: l.name, optimizers: plan.optimizers.map((o) => o.name), times: 1 })),
        events: [],
        schedule: null,
      });
      if (!plan.settings.epochs && !plan.settings.steps)
        this.info("AXS0705", `train '${d.name}' declares no phase; a single default phase of 1 epoch was derived`, d.loc, [
          "write `epochs 10` for the simple case, or `phase name { ... }` for staged lifecycles",
        ]);
    }
    if (plan.losses.length === 0)
      this.err("AXS0601", `train '${d.name}' declares no loss`, d.loc, ["add `loss main = SomeObjective(...)`"]);
    // an optimizer or loss that no phase applies trains nothing, silently (F-017)
    const applied = plan.phases.flatMap((ph) => ph.updates);
    for (const o of plan.optimizers)
      if (o.params.length && !applied.some((u) => u.optimizers.includes(o.name)))
        this.warn("AXS0708", `optimizer '${o.name}' is not applied by any phase; the ${o.params.length} parameter table(s) it covers are never trained`, d.loc, [
          `add \`update <loss> with ${o.name}\` to a phase, or remove the optimizer`,
        ]);
    for (const l of plan.losses)
      if (!applied.some((u) => u.loss === l.name))
        this.warn("AXS0708", `loss '${l.name}' is declared but no phase updates it`, d.loc);
    // model/data compatibility
    if (dataDecl)
      for (const [alias, g] of modelOf) {
        void alias;
        for (const inp of g.inputs) {
          const fieldName = inp.name ?? "";
          const f = dataDecl.fields.find((x) => x.name === fieldName);
          if (!f) continue;
          const ft = f.type;
          if (isTensor(ft) && isTensor(inp.type)) {
            const batched = [dConst(dataDecl.batch), ...ft.shape];
            if (batched.length !== inp.type.shape.length)
              this.warn(
                "AXS0602",
                `data field '${f.name}' (${showType(ft)} per example, batched ${showShapeList(batched)}) does not match model input '${inp.name}' ${showType(inp.type)}`,
                d.loc
              );
          }
        }
      }
    this.mod.plans.push(plan);
  }

  bindingType(
    arg: Arg,
    modelOf: Map<string, IRGraph>,
    data: IRData | undefined,
    loc: Loc
  ): { type?: ValueType; text: string; kind: "model" | "field" | "expr" } {
    const e = arg.value;
    if (e.e === "call" && e.callee.e === "name" && modelOf.has(e.callee.name)) {
      const g = modelOf.get(e.callee.name)!;
      // check field -> model input contracts
      e.args.forEach((a, i) => {
        const inp = g.inputs[i];
        if (!inp || !data) return;
        const fname = a.value.e === "name" ? a.value.name : null;
        if (!fname) {
          // a nested model application (`D(G(z))`) is checked like a field:
          // the inner model's result must satisfy the outer model's input
          const isModelCall =
            (a.value.e === "call" && a.value.callee.e === "name" && modelOf.has(a.value.callee.name)) ||
            (a.value.e === "index" && a.value.obj.e === "call" && a.value.obj.callee.e === "name" && modelOf.has(a.value.obj.callee.name));
          if (!isModelCall) return;
          const inner = this.bindingType(a, modelOf, data, a.loc);
          if (inner.type && isTensor(inner.type) && isTensor(inp.type)) {
            if (inner.type.shape.length !== inp.type.shape.length)
              this.err("AXS0602", `'${inner.text}' produces ${showType(inner.type)} but model input '${inp.name}' expects ${showType(inp.type)}`, a.loc);
            else
              for (let k = 1; k < inp.type.shape.length; k++)
                this.eqDim(inner.type.shape[k], inp.type.shape[k], `'${inner.text}' -> model input '${inp.name}' axis ${k}`, a.loc, "AXS0602");
          }
          return;
        }
        const f = data.fields.find((x) => x.name === fname);
        if (!f) {
          this.err("AXS0604", `data '${data.name}' has no field '${fname}'`, a.loc, [
            `available fields: ${data.fields.map((x) => x.name).join(", ")}`,
          ]);
          return;
        }
        if (isTensor(f.type) && isTensor(inp.type)) {
          const want = inp.type.shape.slice(1);
          const got = f.type.shape;
          if (want.length !== got.length)
            this.err(
              "AXS0602",
              `field '${fname}' is ${showType(f.type)} per example but model input '${inp.name}' expects ${showShapeList(want)} after the batch axis`,
              a.loc
            );
          else
            // a refuted axis is a data/model contract failure, not a generic
            // shape mismatch: the code names the layer that broke (E-008)
            for (let k = 0; k < want.length; k++)
              this.eqDim(got[k], want[k], `field '${fname}' -> model input '${inp.name}' axis ${k + 1}`, a.loc, "AXS0602");
          if (inp.type.kind !== "Tensor" && f.type.kind !== inp.type.kind && f.type.kind !== "Tensor")
            this.warn(
              "AXS0602",
              `field '${fname}' carries ${f.type.kind}[...] but model input '${inp.name}' expects ${inp.type.kind}[...]`,
              a.loc
            );
        }
      });
      const rt = g.resultType;
      if (rt && rt.t === "tuple") {
        // never pick an output on the user's behalf (§16, finding F-008)
        this.err("AXS0408", `'${e.callee.name}' returns ${showType(rt)}; select the output to bind, e.g. ${e.callee.name}(...)[0]`, arg.loc);
        return { type: rt.items[0], text: exprText(e), kind: "model" };
      }
      return { type: rt, text: exprText(e), kind: "model" };
    }
    if (e.e === "index" && e.obj.e === "call" && e.obj.callee.e === "name" && modelOf.has(e.obj.callee.name)) {
      const g = modelOf.get(e.obj.callee.name)!;
      const rt = g.resultType;
      const idx = e.slices[0] && e.slices[0].s === "index" && e.slices[0].value.d === "num" ? e.slices[0].value.value : 0;
      if (rt && rt.t === "tuple") return { type: rt.items[idx], text: exprText(e), kind: "model" };
      return { type: rt, text: exprText(e), kind: "model" };
    }
    if (e.e === "name" && data) {
      const f = data.fields.find((x) => x.name === e.name);
      if (!f) {
        this.err("AXS0604", `data '${data.name}' has no field '${e.name}'`, loc, [
          `available fields: ${data.fields.map((x) => x.name).join(", ")}`,
        ]);
        return { text: e.name, kind: "field" };
      }
      const t = f.type;
      const batched = isTensor(t) ? tensor([dConst(data.batch), ...t.shape], t.kind, t.dtype) : t;
      return { type: batched, text: e.name, kind: "field" };
    }
    return { text: exprText(e), kind: "expr" };
  }

  regionNames(): string[] {
    const set = new Set<string>();
    for (const p of this.mod.params) {
      const parts = p.owner.split("/");
      for (let i = 1; i <= parts.length; i++) set.add(parts.slice(0, i).join("."));
    }
    return [...set].slice(0, 20);
  }

  resolveRegion(text: string, modelOf: Map<string, IRGraph>, loc: Loc): IRParam[] {
    void loc;
    // `*` — every parameter of every model bound in the plan (the region-less optimizer)
    if (text === "*") {
      const roots = new Set([...modelOf.values()].map((g) => g.name));
      return this.mod.params.filter((p) => roots.has(p.owner.split("/")[0]));
    }
    const parts = text.split(".");
    const alias = parts[0];
    const g = modelOf.get(alias);
    const root = g ? g.name : alias;
    const prefix = [root, ...parts.slice(1)].join("/");
    return this.mod.params.filter((p) => p.owner === prefix || p.owner.startsWith(prefix + "/") || p.owner.startsWith(prefix + "."));
  }
}

function showShapeList(s: DimExpr[]): string {
  return `[${s.map(show).join(", ")}]`;
}

export function sliceText(slices: SliceAst[]): string {
  return slices
    .map((s) =>
      s.s === "all" ? ":" : s.s === "ellipsis" ? ".." : s.s === "index" ? dimText(s.value) : `${s.from ? dimText(s.from) : ""}:${s.to ? dimText(s.to) : ""}`
    )
    .join(", ");
}

export function dimText(d: DimAst): string {
  if (d.d === "num") return String(d.value);
  if (d.d === "name") return d.name;
  return `${dimText(d.l)} ${d.op} ${dimText(d.r)}`;
}

export function exprText(e: Expr): string {
  switch (e.e) {
    case "num":
      return String(e.value);
    case "str":
      return `"${e.value}"`;
    case "bool":
      return String(e.value);
    case "name":
      return e.name;
    case "member":
      return `${exprText(e.obj)}.${e.name}`;
    case "call":
      return `${exprText(e.callee)}(${e.args.map((a) => `${a.name ? a.name + ": " : ""}${exprText(a.value)}`).join(", ")})`;
    case "index":
      return `${exprText(e.obj)}[${sliceText(e.slices)}]`;
    case "bin":
      return `${exprText(e.l)} ${e.op} ${exprText(e.r)}`;
    case "unary":
      return `${e.op}${exprText(e.v)}`;
    case "pipe":
      return e.stages.map(exprText).join(" |> ");
    case "tuple":
      return `(${e.items.map(exprText).join(", ")})`;
    case "list":
      return `[${e.items.map(exprText).join(", ")}]`;
    case "dimref":
      return dimText(e.dim);
    case "scan":
      return `scan over ${exprText(e.over)}`;
    default:
      return "<expr>";
  }
}

export interface CompileResult {
  mod: IRModule;
  ok: boolean;
  errors: Diagnostic[];
  warnings: Diagnostic[];
}

export function compile(src: string, name = "program"): CompileResult {
  uid = 0;
  let an: Analyzer;
  let mod: IRModule;
  try {
    const { program, diags } = parse(src);
    an = new Analyzer(program, name, diags);
  } catch (err) {
    an = new Analyzer({ decls: [] }, name, [
      diag("AXS0102", "error", `internal parser error: ${(err as Error).message}`, NOLOC),
    ]);
  }
  try {
    mod = an.run();
  } catch (err) {
    mod = an.mod;
    mod.diags.push(
      diag("AXS0102", "error", `internal compiler error: ${(err as Error).message}`, NOLOC, undefined, [
        "this is a compiler bug, not a program error",
      ])
    );
  }
  const errors = mod.diags.filter((d) => d.severity === "error");
  return { mod, ok: errors.length === 0, errors, warnings: mod.diags.filter((d) => d.severity === "warning") };
}
