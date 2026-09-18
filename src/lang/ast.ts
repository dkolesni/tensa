/**
 * TENSA — abstract syntax tree.
 */
import { Loc } from "./types";

export interface Node {
  loc: Loc;
}

// ------------------------------------------------------------------ types

export type TypeExpr =
  | { t: "tensorType"; kind: string; dims: DimAst[]; loc: Loc }
  | { t: "scalarType"; loc: Loc }
  | { t: "intType"; loc: Loc }
  | { t: "dimType"; loc: Loc }
  | { t: "tupleType"; items: TypeExpr[]; loc: Loc };

export type DimAst =
  | { d: "num"; value: number; loc: Loc }
  | { d: "name"; name: string; loc: Loc }
  | { d: "bin"; op: "+" | "-" | "*" | "/"; l: DimAst; r: DimAst; loc: Loc };

// ------------------------------------------------------------------ expressions

export interface Arg {
  name?: string;
  value: Expr;
  loc: Loc;
}

export type Expr =
  | { e: "num"; value: number; loc: Loc }
  | { e: "str"; value: string; loc: Loc }
  | { e: "bool"; value: boolean; loc: Loc }
  | { e: "name"; name: string; loc: Loc }
  | { e: "member"; obj: Expr; name: string; loc: Loc }
  | { e: "call"; callee: Expr; args: Arg[]; loc: Loc }
  | { e: "index"; obj: Expr; slices: SliceAst[]; loc: Loc }
  | { e: "bin"; op: string; l: Expr; r: Expr; loc: Loc }
  | { e: "unary"; op: string; v: Expr; loc: Loc }
  | { e: "pipe"; stages: Expr[]; loc: Loc }
  | { e: "tuple"; items: Expr[]; loc: Loc }
  | { e: "list"; items: Expr[]; loc: Loc }
  | { e: "dimref"; dim: DimAst; loc: Loc }
  | { e: "scan"; over: Expr; axis: number; carry: CarryDecl[]; body: Stmt[]; loc: Loc };

export type SliceAst =
  | { s: "all"; loc: Loc }
  | { s: "ellipsis"; loc: Loc }
  | { s: "index"; value: DimAst; loc: Loc }
  | { s: "range"; from: DimAst | null; to: DimAst | null; loc: Loc };

export interface CarryDecl {
  name: string;
  type: TypeExpr;
  init: string;
  loc: Loc;
}

// ------------------------------------------------------------------ statements

export type Stmt =
  | { s: "let"; names: string[]; value: Expr; loc: Loc }
  | { s: "return"; value: Expr | null; loc: Loc }
  | { s: "yield"; values: Expr[]; loc: Loc }
  | { s: "expr"; value: Expr; loc: Loc }
  | { s: "residual"; via: Expr | null; body: Stmt[]; loc: Loc }
  | { s: "split"; merge: { kind: string; axis?: number; loc: Loc }; branches: Branch[]; loc: Loc }
  | { s: "for"; count: number | null; index: string | null; from: number; to: number; body: Stmt[]; loc: Loc }
  | { s: "param"; name: string; type: TypeExpr; init: string; frozen: boolean; loc: Loc }
  | { s: "state"; name: string; type: TypeExpr; init: string; update: string | null; loc: Loc };

export interface Branch {
  label: string | null;
  body: Stmt[];
  loc: Loc;
}

// ------------------------------------------------------------------ declarations

export interface ParamDef {
  name: string;
  type: TypeExpr;
  def?: Expr;
  loc: Loc;
}

export interface FnLikeDecl {
  d: "model" | "block" | "fn" | "objective";
  name: string;
  params: ParamDef[];
  ret: TypeExpr | null;
  body: Stmt[];
  loc: Loc;
}

export interface SourceDecl {
  d: "source";
  name: string;
  adapter: string;
  args: Arg[];
  loc: Loc;
}

export interface FieldDecl {
  name: string;
  type: TypeExpr;
  value: Expr;
  loc: Loc;
}

export type DataStmt =
  | { ds: "example"; fields: FieldDecl[]; loc: Loc }
  | { ds: "stage"; stage: "preprocess" | "augment"; mode: "train" | "eval" | "both"; entries: { field: string; pipe: Expr; loc: Loc }[]; loc: Loc }
  | { ds: "split"; parts: { name: string; frac: number }[]; loc: Loc }
  | { ds: "batch"; size: number; loc: Loc }
  | { ds: "shuffle"; value: boolean; loc: Loc };

export interface DataDecl {
  d: "data";
  name: string;
  source: string;
  body: DataStmt[];
  loc: Loc;
}

export type PhaseStmt =
  | { p: "epochs"; n: number; loc: Loc }
  | { p: "steps"; n: number; loc: Loc }
  | { p: "freeze"; region: Expr; loc: Loc }
  | { p: "unfreeze"; region: Expr; loc: Loc }
  | { p: "lr"; region: Expr; value: number; loc: Loc }
  | { p: "update"; loss: string; opt: string[]; times: number; loc: Loc }
  | { p: "until"; metric: string; cmp: string; value: number; loc: Loc }
  | { p: "every"; n: number; unit: "steps" | "epochs"; actions: EventAction[]; loc: Loc }
  | { p: "schedule"; kind: string; args: Arg[]; loc: Loc };

export interface EventAction {
  kind: string;
  args: Arg[];
  loc: Loc;
}

export type TrainStmt =
  | { t: "data"; name: string; loc: Loc }
  | { t: "model"; alias: string; name: string; loc: Loc }
  | { t: "loss"; name: string; objective: string; args: Arg[]; weight: number; loc: Loc }
  | { t: "optimizer"; name: string; kind: string; args: Arg[]; region: Expr | null; loc: Loc }
  | { t: "track"; name: string; kind: string; region: Expr; args: Arg[]; loc: Loc }
  | { t: "phase"; name: string; body: PhaseStmt[]; loc: Loc }
  | { t: "every"; n: number; unit: "steps" | "epochs"; actions: EventAction[]; loc: Loc }
  | { t: "setting"; key: string; value: Expr; loc: Loc };

export interface TrainDecl {
  d: "train";
  name: string;
  body: TrainStmt[];
  loc: Loc;
}

export interface DimDeclAst {
  d: "dim";
  name: string;
  value: DimAst | null;
  loc: Loc;
}

export interface CustomOpDecl {
  d: "customop";
  name: string;
  params: ParamDef[];
  ret: TypeExpr | null;
  effects: string[];
  backend: { target: string; code: string }[];
  shapeUnknown: boolean;
  differentiable: boolean;
  loc: Loc;
}

export type Decl = DimDeclAst | FnLikeDecl | SourceDecl | DataDecl | TrainDecl | CustomOpDecl;

export interface Program {
  decls: Decl[];
}
