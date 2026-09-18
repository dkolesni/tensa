/**
 * TENSA — recursive descent parser.
 */
import {
  Arg, Branch, CarryDecl, CustomOpDecl, DataDecl, DataStmt, Decl, DimAst, EventAction, Expr,
  FieldDecl, FnLikeDecl, ParamDef, PhaseStmt, Program, SliceAst, SourceDecl, Stmt, TrainDecl,
  TrainStmt, TypeExpr,
} from "./ast";
import { Token, lex } from "./lexer";
import { Diagnostic, Loc, diag } from "./types";

const TENSOR_KINDS = ["Tensor", "Tokens", "Class", "Mask", "Logits", "Probs", "Image"];

class ParseError extends Error {}

export class Parser {
  toks: Token[];
  pos = 0;
  diags: Diagnostic[] = [];

  constructor(src: string) {
    const { tokens, diags } = lex(src);
    this.toks = tokens;
    this.diags = diags;
  }

  // -------------------------------------------------------------- primitives
  peek(k = 0): Token {
    return this.toks[Math.min(this.pos + k, this.toks.length - 1)];
  }
  get loc(): Loc {
    return this.peek().loc;
  }
  at(text: string): boolean {
    const t = this.peek();
    return (t.kind === "op" || t.kind === "kw" || t.kind === "ident") && t.text === text;
  }
  atKind(kind: string): boolean {
    return this.peek().kind === kind;
  }
  next(): Token {
    return this.toks[this.pos++];
  }
  eat(text: string): boolean {
    if (this.at(text)) {
      this.pos++;
      return true;
    }
    return false;
  }
  expect(text: string): Token {
    if (this.at(text)) return this.next();
    this.error(`expected '${text}' but found '${this.peek().text}'`);
    throw new ParseError(text);
  }
  error(msg: string, loc = this.loc) {
    if (this.diags.filter((d) => d.code === "AXS0102").length < 12)
      this.diags.push(diag("AXS0102", "error", msg, loc));
  }
  skipNL() {
    while (this.atKind("nl")) this.pos++;
  }
  endStmt() {
    if (this.atKind("nl")) {
      this.skipNL();
      return;
    }
    if (this.at("}") || this.atKind("eof")) return;
    this.error(`unexpected '${this.peek().text}' after statement`);
    // recover: skip to end of line
    while (!this.atKind("nl") && !this.atKind("eof") && !this.at("}")) this.pos++;
    this.skipNL();
  }
  ident(): string {
    const t = this.peek();
    if (t.kind === "ident" || t.kind === "kw") {
      this.pos++;
      return t.text;
    }
    this.error(`expected identifier, found '${t.text}'`);
    throw new ParseError("ident");
  }
  num(): number {
    const t = this.peek();
    if (t.kind === "num") {
      this.pos++;
      return t.value!;
    }
    if (this.at("-") && this.peek(1).kind === "num") {
      this.pos++;
      return -this.next().value!;
    }
    this.error(`expected number, found '${t.text}'`);
    throw new ParseError("num");
  }

  // -------------------------------------------------------------- program
  parseProgram(): Program {
    const decls: Decl[] = [];
    this.skipNL();
    while (!this.atKind("eof")) {
      const startPos = this.pos;
      try {
        const d = this.parseDecl();
        if (d) decls.push(d);
      } catch {
        // panic-mode recovery: skip to next top-level keyword
        while (
          !this.atKind("eof") &&
          !["dim", "model", "block", "fn", "objective", "source", "data", "train", "custom"].some(
            (k) => this.at(k) && this.peek(-1)?.kind !== "op"
          )
        )
          this.pos++;
        if (this.atKind("eof")) break;
      }
      this.skipNL();
      // never spin on a position the parser cannot consume
      if (this.pos === startPos) this.pos++;
    }
    return { decls };
  }

  parseDecl(): Decl | null {
    const loc = this.loc;
    if (this.at("dim")) {
      this.next();
      const name = this.ident();
      let value: DimAst | null = null;
      if (this.eat("=")) value = this.parseDim();
      this.endStmt();
      return { d: "dim", name, value, loc };
    }
    if (this.at("model") || this.at("block") || this.at("fn") || this.at("objective")) {
      const kw = this.next().text as FnLikeDecl["d"];
      const name = this.ident();
      const params = this.parseParamList();
      let ret: TypeExpr | null = null;
      if (this.eat("->")) ret = this.parseType();
      const body = this.parseBlock();
      return { d: kw, name, params, ret, body, loc };
    }
    if (this.at("source")) {
      this.next();
      const name = this.ident();
      this.expect("=");
      const adapter = this.ident();
      const args = this.at("(") ? this.parseArgs() : [];
      this.endStmt();
      return { d: "source", name, adapter, args, loc } as SourceDecl;
    }
    if (this.at("data")) return this.parseData();
    if (this.at("train")) return this.parseTrain();
    if (this.at("custom")) return this.parseCustomOp();
    this.error(`unexpected '${this.peek().text}' at top level`);
    throw new ParseError("decl");
  }

  parseParamList(): ParamDef[] {
    const out: ParamDef[] = [];
    if (!this.eat("(")) return out;
    this.skipNL();
    while (!this.at(")") && !this.atKind("eof")) {
      const loc = this.loc;
      const name = this.ident();
      this.expect(":");
      const type = this.parseType();
      let def: Expr | undefined;
      if (this.eat("=")) def = this.parseExpr();
      out.push({ name, type, def, loc });
      this.skipNL();
      if (!this.eat(",")) break;
      this.skipNL();
    }
    this.expect(")");
    return out;
  }

  // -------------------------------------------------------------- types
  parseType(): TypeExpr {
    const loc = this.loc;
    if (this.at("[")) {
      const dims = this.parseDimList();
      return { t: "tensorType", kind: "Tensor", dims, loc };
    }
    if (this.at("(")) {
      this.next();
      const items: TypeExpr[] = [];
      this.skipNL();
      while (!this.at(")") && !this.atKind("eof")) {
        items.push(this.parseType());
        this.skipNL();
        if (!this.eat(",")) break;
        this.skipNL();
      }
      this.expect(")");
      return { t: "tupleType", items, loc };
    }
    const name = this.ident();
    if (name === "Scalar") return { t: "scalarType", loc };
    if (name === "Int") return { t: "intType", loc };
    if (name === "Dim") return { t: "dimType", loc };
    if (TENSOR_KINDS.includes(name)) {
      const dims = this.at("[") ? this.parseDimList() : [];
      return { t: "tensorType", kind: name, dims, loc };
    }
    this.error(`unknown type '${name}'`, loc);
    return { t: "tensorType", kind: "Tensor", dims: [], loc };
  }

  parseDimList(): DimAst[] {
    this.expect("[");
    const dims: DimAst[] = [];
    this.skipNL();
    while (!this.at("]") && !this.atKind("eof")) {
      dims.push(this.parseDim());
      this.skipNL();
      if (!this.eat(",")) break;
      this.skipNL();
    }
    this.expect("]");
    return dims;
  }

  parseDim(): DimAst {
    return this.parseDimAdd();
  }
  parseDimAdd(): DimAst {
    let l = this.parseDimMul();
    while (this.at("+") || this.at("-")) {
      const loc = this.loc;
      const op = this.next().text as "+" | "-";
      const r = this.parseDimMul();
      l = { d: "bin", op, l, r, loc };
    }
    return l;
  }
  parseDimMul(): DimAst {
    let l = this.parseDimAtom();
    while (this.at("*") || this.at("/")) {
      const loc = this.loc;
      const op = this.next().text as "*" | "/";
      const r = this.parseDimAtom();
      l = { d: "bin", op, l, r, loc };
    }
    return l;
  }
  parseDimAtom(): DimAst {
    const loc = this.loc;
    if (this.atKind("num")) return { d: "num", value: this.num(), loc };
    if (this.eat("(")) {
      const inner = this.parseDim();
      this.expect(")");
      return inner;
    }
    const name = this.ident();
    return { d: "name", name, loc };
  }

  // -------------------------------------------------------------- statements
  parseBlock(): Stmt[] {
    this.expect("{");
    const out: Stmt[] = [];
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      const before = this.pos;
      try {
        out.push(this.parseStmt());
      } catch {
        while (!this.atKind("nl") && !this.atKind("eof") && !this.at("}")) this.pos++;
      }
      if (this.pos === before) this.pos++;
      this.skipNL();
    }
    if (!this.eat("}")) this.diags.push(diag("AXS0103", "error", "unterminated block", this.loc));
    return out;
  }

  parseStmt(): Stmt {
    const loc = this.loc;
    if (this.at("let")) {
      this.next();
      const names: string[] = [];
      if (this.eat("(")) {
        while (!this.at(")") && !this.atKind("eof")) {
          names.push(this.ident());
          if (!this.eat(",")) break;
        }
        this.expect(")");
      } else names.push(this.ident());
      this.expect("=");
      const value = this.parseExpr();
      this.endStmt();
      return { s: "let", names, value, loc };
    }
    if (this.at("return")) {
      this.next();
      const value = this.atKind("nl") || this.at("}") ? null : this.parseExpr();
      this.endStmt();
      return { s: "return", value, loc };
    }
    if (this.at("yield")) {
      this.next();
      const values = [this.parseExpr()];
      while (this.eat(",")) values.push(this.parseExpr());
      this.endStmt();
      return { s: "yield", values, loc };
    }
    if (this.at("residual")) {
      this.next();
      let via: Expr | null = null;
      if (this.eat("via")) via = this.parsePostfix();
      const body = this.parseBlock();
      this.endStmt();
      return { s: "residual", via, body, loc };
    }
    if (this.at("split")) {
      this.next();
      this.expect("merge");
      const mloc = this.loc;
      const kind = this.ident();
      let axis: number | undefined;
      if (this.eat("(")) {
        if (this.at("axis")) {
          this.next();
          this.expect(":");
        }
        if (!this.at(")")) axis = this.num();
        this.expect(")");
      }
      this.expect("{");
      const branches: Branch[] = [];
      this.skipNL();
      while (!this.at("}") && !this.atKind("eof")) {
        const bloc = this.loc;
        if (this.at("branch")) {
          this.next();
          let label: string | null = null;
          if (!this.at("{")) label = this.ident();
          const body = this.parseBlock();
          branches.push({ label, body, loc: bloc });
        } else {
          const st = this.parseStmt();
          branches.push({ label: null, body: [st], loc: bloc });
        }
        this.skipNL();
      }
      this.expect("}");
      this.endStmt();
      return { s: "split", merge: { kind, axis, loc: mloc }, branches, loc };
    }
    if (this.at("for")) {
      this.next();
      let count: number | null = null;
      let index: string | null = null;
      let from = 0;
      let to = 0;
      if (this.atKind("num")) {
        count = this.num();
        from = 0;
        to = count;
      } else {
        index = this.ident();
        this.expect("in");
        from = this.num();
        this.expect("..");
        to = this.num();
      }
      if (this.eat(":")) {
        // `for 12: <stmt>` — the inner statement already consumed its terminator
        this.skipNL();
        return { s: "for", count, index, from, to, body: [this.parseStmt()], loc };
      }
      const body = this.parseBlock();
      this.endStmt();
      return { s: "for", count, index, from, to, body, loc };
    }
    if (this.at("frozen") || this.at("param")) {
      const frozen = this.eat("frozen");
      this.expect("param");
      const name = this.ident();
      this.expect(":");
      const type = this.parseType();
      let init = "zeros";
      if (this.eat("init")) {
        this.expect(":");
        init = this.ident();
      }
      this.endStmt();
      return { s: "param", name, type, init, frozen, loc };
    }
    if (this.at("state")) {
      this.next();
      const name = this.ident();
      this.expect(":");
      const type = this.parseType();
      let init = "zeros";
      let update: string | null = null;
      if (this.eat("init")) {
        this.expect(":");
        init = this.ident();
      }
      if (this.eat("update")) {
        this.expect(":");
        update = this.ident();
        if (this.eat("(")) {
          const parts: string[] = [];
          while (!this.at(")") && !this.atKind("eof")) parts.push(this.next().text);
          this.expect(")");
          update += `(${parts.join("")})`;
        }
      }
      this.endStmt();
      return { s: "state", name, type, init, update, loc };
    }
    const value = this.parseExpr();
    this.endStmt();
    return { s: "expr", value, loc };
  }

  // -------------------------------------------------------------- expressions
  parseExpr(): Expr {
    return this.parsePipe();
  }

  parsePipe(): Expr {
    const loc = this.loc;
    const first = this.parseCompare();
    if (!this.at("|>")) return first;
    const stages = [first];
    while (this.eat("|>")) {
      this.skipNLInline();
      stages.push(this.parseCompare());
    }
    return { e: "pipe", stages, loc };
  }
  skipNLInline() {
    while (this.atKind("nl") && this.peek().text === "\\n") this.pos++;
  }

  parseCompare(): Expr {
    let l = this.parseAdd();
    while (["==", "!=", "<", ">", "<=", ">="].some((o) => this.at(o))) {
      const loc = this.loc;
      const op = this.next().text;
      const r = this.parseAdd();
      l = { e: "bin", op, l, r, loc };
    }
    return l;
  }
  parseAdd(): Expr {
    let l = this.parseMul();
    while (this.at("+") || this.at("-")) {
      const loc = this.loc;
      const op = this.next().text;
      const r = this.parseMul();
      l = { e: "bin", op, l, r, loc };
    }
    return l;
  }
  parseMul(): Expr {
    let l = this.parseUnary();
    while (this.at("*") || this.at("/") || this.at("%")) {
      const loc = this.loc;
      const op = this.next().text;
      const r = this.parseUnary();
      l = { e: "bin", op, l, r, loc };
    }
    return l;
  }
  parseUnary(): Expr {
    if (this.at("-")) {
      const loc = this.loc;
      this.next();
      return { e: "unary", op: "-", v: this.parseUnary(), loc };
    }
    return this.parsePostfix();
  }

  parsePostfix(): Expr {
    let e = this.parsePrimary();
    for (;;) {
      if (this.at("(")) {
        const loc = this.loc;
        const args = this.parseArgs();
        e = { e: "call", callee: e, args, loc };
      } else if (this.at(".") && this.peek(1).kind !== "num") {
        const loc = this.loc;
        this.next();
        const name = this.ident();
        e = { e: "member", obj: e, name, loc };
      } else if (this.at("[")) {
        const loc = this.loc;
        this.next();
        const slices: SliceAst[] = [];
        this.skipNL();
        while (!this.at("]") && !this.atKind("eof")) {
          slices.push(this.parseSlice());
          this.skipNL();
          if (!this.eat(",")) break;
          this.skipNL();
        }
        this.expect("]");
        e = { e: "index", obj: e, slices, loc };
      } else break;
    }
    return e;
  }

  parseSlice(): SliceAst {
    const loc = this.loc;
    if (this.at("..")) {
      this.next();
      return { s: "ellipsis", loc };
    }
    if (this.at(":")) {
      this.next();
      if (this.at(",") || this.at("]")) return { s: "all", loc };
      const to = this.parseDim();
      return { s: "range", from: null, to, loc };
    }
    const first = this.parseDim();
    if (this.eat(":")) {
      if (this.at(",") || this.at("]")) return { s: "range", from: first, to: null, loc };
      const to = this.parseDim();
      return { s: "range", from: first, to, loc };
    }
    return { s: "index", value: first, loc };
  }

  parseArgs(): Arg[] {
    this.expect("(");
    const args: Arg[] = [];
    this.skipNL();
    while (!this.at(")") && !this.atKind("eof")) {
      const loc = this.loc;
      let name: string | undefined;
      if ((this.peek().kind === "ident" || this.peek().kind === "kw") && this.peek(1).text === ":") {
        name = this.ident();
        this.next();
      }
      const value = this.parseExpr();
      args.push({ name, value, loc });
      this.skipNL();
      if (!this.eat(",")) break;
      this.skipNL();
    }
    this.expect(")");
    return args;
  }

  parsePrimary(): Expr {
    const loc = this.loc;
    const t = this.peek();
    if (t.kind === "num") {
      this.next();
      return { e: "num", value: t.value!, loc };
    }
    if (t.kind === "str") {
      this.next();
      return { e: "str", value: t.text, loc };
    }
    if (this.at("true") || this.at("false")) {
      this.next();
      return { e: "bool", value: t.text === "true", loc };
    }
    if (this.at("scan")) return this.parseScan();
    if (this.at("[")) {
      this.next();
      const items: Expr[] = [];
      this.skipNL();
      while (!this.at("]") && !this.atKind("eof")) {
        items.push(this.parseExpr());
        this.skipNL();
        if (!this.eat(",")) break;
        this.skipNL();
      }
      this.expect("]");
      return { e: "list", items, loc };
    }
    if (this.at("(")) {
      this.next();
      this.skipNL();
      const items: Expr[] = [this.parseExpr()];
      this.skipNL();
      while (this.eat(",")) {
        this.skipNL();
        if (this.at(")")) break;
        items.push(this.parseExpr());
        this.skipNL();
      }
      this.expect(")");
      return items.length === 1 ? items[0] : { e: "tuple", items, loc };
    }
    if (t.kind === "ident" || t.kind === "kw") {
      this.next();
      return { e: "name", name: t.text, loc };
    }
    this.error(`unexpected '${t.text}' in expression`);
    throw new ParseError("expr");
  }

  parseScan(): Expr {
    const loc = this.loc;
    this.expect("scan");
    this.expect("over");
    const over = this.parsePostfix();
    let axis = 1;
    if (this.eat("axis")) {
      this.expect(":");
      axis = this.num();
    }
    const carry: CarryDecl[] = [];
    if (this.eat("carry")) {
      for (;;) {
        const cloc = this.loc;
        const name = this.ident();
        this.expect(":");
        const type = this.parseType();
        let init = "zeros";
        if (this.eat("init")) {
          this.expect(":");
          init = this.ident();
        }
        carry.push({ name, type, init, loc: cloc });
        if (!this.eat(",")) break;
      }
    }
    const body = this.parseBlock();
    return { e: "scan", over, axis, carry, body, loc };
  }

  // -------------------------------------------------------------- data
  parseData(): DataDecl {
    const loc = this.loc;
    this.expect("data");
    const name = this.ident();
    let source = "";
    if (this.eat("from")) source = this.ident();
    this.expect("{");
    const body: DataStmt[] = [];
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      const sloc = this.loc;
      if (this.at("example")) {
        this.next();
        this.expect("{");
        const fields: FieldDecl[] = [];
        this.skipNL();
        while (!this.at("}") && !this.atKind("eof")) {
          const floc = this.loc;
          this.expect("field");
          const fname = this.ident();
          this.expect(":");
          const type = this.parseType();
          this.expect("=");
          const value = this.parseExpr();
          fields.push({ name: fname, type, value, loc: floc });
          this.skipNL();
        }
        this.expect("}");
        body.push({ ds: "example", fields, loc: sloc });
      } else if (this.at("preprocess") || this.at("augment")) {
        const stage = this.next().text as "preprocess" | "augment";
        let mode: "train" | "eval" | "both" = "both";
        if (!this.at("{") && (this.peek().kind === "ident" || this.peek().kind === "kw")) {
          const m = this.ident();
          mode = m === "train" ? "train" : m === "eval" ? "eval" : "both";
        }
        this.expect("{");
        const entries: { field: string; pipe: Expr; loc: Loc }[] = [];
        this.skipNL();
        while (!this.at("}") && !this.atKind("eof")) {
          const eloc = this.loc;
          const field = this.ident();
          this.expect(":");
          const pipe = this.parseExpr();
          entries.push({ field, pipe, loc: eloc });
          this.skipNL();
        }
        this.expect("}");
        body.push({ ds: "stage", stage, mode, entries, loc: sloc });
      } else if (this.at("split")) {
        this.next();
        this.expect("{");
        const parts: { name: string; frac: number }[] = [];
        this.skipNL();
        while (!this.at("}") && !this.atKind("eof")) {
          const pname = this.ident();
          this.expect(":");
          const frac = this.num();
          parts.push({ name: pname, frac });
          this.eat(",");
          this.skipNL();
        }
        this.expect("}");
        body.push({ ds: "split", parts, loc: sloc });
      } else if (this.at("batch")) {
        this.next();
        body.push({ ds: "batch", size: this.num(), loc: sloc });
      } else if (this.at("shuffle")) {
        this.next();
        const v = this.at("true") || this.at("false") ? this.next().text === "true" : true;
        body.push({ ds: "shuffle", value: v, loc: sloc });
      } else {
        this.error(`unexpected '${this.peek().text}' in data block`);
        while (!this.atKind("nl") && !this.atKind("eof") && !this.at("}")) this.pos++;
      }
      this.skipNL();
    }
    this.expect("}");
    return { d: "data", name, source, body, loc };
  }

  // -------------------------------------------------------------- train
  parseEventActions(): EventAction[] {
    const actions: EventAction[] = [];
    this.expect("{");
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      const aloc = this.loc;
      const kind = this.ident();
      const args = this.at("(") ? this.parseArgs() : [];
      actions.push({ kind, args, loc: aloc });
      this.skipNL();
    }
    this.expect("}");
    return actions;
  }

  parseTrain(): TrainDecl {
    const loc = this.loc;
    this.expect("train");
    const name = this.ident();
    this.expect("{");
    const body: TrainStmt[] = [];
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      const sloc = this.loc;
      if (this.at("data")) {
        this.next();
        body.push({ t: "data", name: this.ident(), loc: sloc });
      } else if (this.at("model")) {
        this.next();
        let alias = this.ident();
        let mname = alias;
        if (this.eat("=")) mname = this.ident();
        body.push({ t: "model", alias, name: mname, loc: sloc });
      } else if (this.at("loss")) {
        this.next();
        const lname = this.ident();
        this.expect("=");
        const objective = this.ident();
        const args = this.at("(") ? this.parseArgs() : [];
        let weight = 1;
        if (this.eat("weight")) {
          this.eat(":");
          weight = this.num();
        }
        body.push({ t: "loss", name: lname, objective, args, weight, loc: sloc });
      } else if (this.at("optimizer")) {
        this.next();
        const oname = this.ident();
        let kind = "adamw";
        let args: Arg[] = [];
        if (this.eat("=")) {
          kind = this.ident();
          args = this.at("(") ? this.parseArgs() : [];
        }
        let region: Expr | null = null;
        if (this.eat("over")) region = this.parsePostfix();
        body.push({ t: "optimizer", name: oname, kind, args, region, loc: sloc });
      } else if (this.at("track")) {
        this.next();
        const tname = this.ident();
        this.expect("=");
        const kind = this.ident();
        this.expect("(");
        const region = this.parsePostfix();
        const args: Arg[] = [];
        while (this.eat(",")) {
          const aloc = this.loc;
          let aname: string | undefined;
          if (this.peek(1).text === ":") {
            aname = this.ident();
            this.next();
          }
          args.push({ name: aname, value: this.parseExpr(), loc: aloc });
        }
        this.expect(")");
        body.push({ t: "track", name: tname, kind, region, args, loc: sloc });
      } else if (this.at("phase")) {
        this.next();
        const pname = this.ident();
        body.push({ t: "phase", name: pname, body: this.parsePhase(), loc: sloc });
      } else if (this.at("every")) {
        this.next();
        const n = this.atKind("num") ? this.num() : 1;
        const unit = this.at("epoch") || this.at("epochs") ? "epochs" : "steps";
        this.next();
        body.push({ t: "every", n, unit, actions: this.parseEventActions(), loc: sloc });
      } else {
        const key = this.ident();
        const value = this.parseExpr();
        body.push({ t: "setting", key, value, loc: sloc });
      }
      this.skipNL();
    }
    this.expect("}");
    return { d: "train", name, body, loc };
  }

  parsePhase(): PhaseStmt[] {
    const out: PhaseStmt[] = [];
    this.expect("{");
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      const loc = this.loc;
      if (this.at("epochs")) {
        this.next();
        out.push({ p: "epochs", n: this.num(), loc });
      } else if (this.at("steps")) {
        this.next();
        out.push({ p: "steps", n: this.num(), loc });
      } else if (this.at("freeze")) {
        this.next();
        out.push({ p: "freeze", region: this.parsePostfix(), loc });
      } else if (this.at("unfreeze")) {
        this.next();
        out.push({ p: "unfreeze", region: this.parsePostfix(), loc });
      } else if (this.at("lr")) {
        this.next();
        const region = this.parsePostfix();
        this.expect("=");
        out.push({ p: "lr", region, value: this.num(), loc });
      } else if (this.at("update")) {
        this.next();
        const lossName = this.ident();
        const opts: string[] = [];
        if (this.eat("with")) {
          opts.push(this.ident());
          while (this.eat(",")) opts.push(this.ident());
        }
        let times = 1;
        if (this.eat("times")) times = this.num();
        out.push({ p: "update", loss: lossName, opt: opts, times, loc });
      } else if (this.at("until")) {
        this.next();
        const metric = this.ident();
        const cmp = this.next().text;
        out.push({ p: "until", metric, cmp, value: this.num(), loc });
      } else if (this.at("every")) {
        this.next();
        const n = this.atKind("num") ? this.num() : 1;
        const unit = this.at("epoch") || this.at("epochs") ? "epochs" : "steps";
        this.next();
        out.push({ p: "every", n, unit, actions: this.parseEventActions(), loc });
      } else {
        const kind = this.ident();
        const args = this.at("(") ? this.parseArgs() : [];
        out.push({ p: "schedule", kind, args, loc });
      }
      this.skipNL();
    }
    this.expect("}");
    return out;
  }

  // -------------------------------------------------------------- custom op
  parseCustomOp(): CustomOpDecl {
    const loc = this.loc;
    this.expect("custom");
    this.expect("op");
    const name = this.ident();
    const params = this.parseParamList();
    let ret: TypeExpr | null = null;
    if (this.eat("->")) ret = this.parseType();
    const effects: string[] = [];
    const backend: { target: string; code: string }[] = [];
    let shapeUnknown = ret === null;
    let differentiable = true;
    this.expect("{");
    this.skipNL();
    while (!this.at("}") && !this.atKind("eof")) {
      if (this.at("effects")) {
        this.next();
        this.expect(":");
        effects.push(this.ident());
        while (this.eat(",")) effects.push(this.ident());
      } else if (this.at("backend")) {
        this.next();
        const target = this.ident();
        this.expect(":");
        const code = this.peek().kind === "str" ? this.next().text : "";
        backend.push({ target, code });
      } else {
        const key = this.ident();
        this.expect(":");
        const v = this.peek().kind === "str" ? this.next().text : this.ident();
        if (key === "shape" && v === "unknown") shapeUnknown = true;
        if (key === "differentiable" && v === "false") differentiable = false;
      }
      this.skipNL();
    }
    this.expect("}");
    return { d: "customop", name, params, ret, effects, backend, shapeUnknown, differentiable, loc };
  }
}

export function parse(src: string): { program: Program; diags: Diagnostic[] } {
  const p = new Parser(src);
  const program = p.parseProgram();
  return { program, diags: p.diags };
}
