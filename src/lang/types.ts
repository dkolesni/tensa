/**
 * TENSA — value types, semantic tensor kinds, effects, diagnostics.
 */
import { DimExpr, show } from "./dims";

// ------------------------------------------------------------------ locations

export interface Loc {
  line: number;
  col: number;
  len: number;
}

export const NOLOC: Loc = { line: 1, col: 1, len: 0 };

// ------------------------------------------------------------------ types

/**
 * Semantic tensor kinds.  These are *refinements* of `Tensor`, not a nominal
 * hierarchy: every semantic kind is a tensor and is assignable to `Tensor`.
 * They exist purely because they buy real diagnostics (softmax-of-probabilities,
 * cross-entropy-on-probabilities, embedding of non-token tensors, ...).
 */
export type TensorKind =
  | "Tensor"
  | "Tokens"
  | "Class"
  | "Mask"
  | "Logits"
  | "Probs"
  | "Image";

export interface TensorType {
  t: "tensor";
  kind: TensorKind;
  shape: DimExpr[];
  dtype: "f32" | "i32" | "bool";
  /** shape knowledge was lost (custom op without shape semantics, §5.2 "unknown"); `shape` is empty and must not be reasoned about */
  unknown?: true;
}

export interface ScalarType {
  t: "scalar";
}
export interface TupleType {
  t: "tuple";
  items: ValueType[];
}
export interface StageType {
  t: "stage";
  name: string;
  instanceId: string;
}
export interface IntType {
  t: "int";
}

export type ValueType = TensorType | ScalarType | TupleType | StageType | IntType;

export function tensor(shape: DimExpr[], kind: TensorKind = "Tensor", dtype: TensorType["dtype"] = "f32"): TensorType {
  return { t: "tensor", kind, shape, dtype };
}
export const scalar: ScalarType = { t: "scalar" };
export function unknownTensor(): TensorType {
  return { t: "tensor", kind: "Tensor", shape: [], dtype: "f32", unknown: true };
}

export function isTensor(v: ValueType): v is TensorType {
  return v.t === "tensor";
}

export function rank(t: TensorType): number {
  return t.shape.length;
}

export function showType(t: ValueType): string {
  switch (t.t) {
    case "scalar":
      return "Scalar";
    case "int":
      return "Int";
    case "stage":
      return `Stage<${t.name}>`;
    case "tuple":
      return `(${t.items.map(showType).join(", ")})`;
    case "tensor":
      return t.unknown ? "Tensor[?]" : `${t.kind}[${t.shape.map(show).join(", ")}]`;
  }
}

export function showShape(shape: DimExpr[]): string {
  return `[${shape.map(show).join(", ")}]`;
}

// ------------------------------------------------------------------ effects

export type Effect =
  | "pure"
  | "stochastic"
  | "reads-state"
  | "writes-state"
  | "parameterized"
  | "training-sensitive"
  | "nondiff" // mathematically non-differentiable
  | "grad-stopped"; // differentiation intentionally cut

export const EFFECT_DOC: Record<Effect, string> = {
  pure: "deterministic, no persistent state",
  stochastic: "samples randomness at runtime",
  "reads-state": "reads persistent non-parameter state",
  "writes-state": "updates persistent non-parameter state",
  parameterized: "owns trainable parameters",
  "training-sensitive": "behaves differently in train vs eval context",
  nondiff: "mathematically non-differentiable (no gradient exists)",
  "grad-stopped": "gradient intentionally stopped by the program",
};

// ------------------------------------------------------------------ diagnostics

export type Severity = "error" | "warning" | "info";

export interface Diagnostic {
  code: string;
  severity: Severity;
  message: string;
  loc: Loc;
  where?: string; // architectural location, e.g. "model Net › residual › linear#3"
  notes?: string[];
}

export interface DiagnosticSpec {
  code: string;
  title: string;
  explain: string;
}

/** Stable diagnostic catalogue.  Codes never change meaning. */
export const DIAGNOSTICS: DiagnosticSpec[] = [
  { code: "AXS0101", title: "unexpected character", explain: "The lexer found a character that is not part of TENSA source." },
  { code: "AXS0102", title: "unexpected token", explain: "The parser expected a different token here." },
  { code: "AXS0103", title: "unterminated block", explain: "A `{` was never closed." },
  { code: "AXS0201", title: "unknown name", explain: "The identifier is not a declared dim, value, stage, block, fn, objective or catalog operation." },
  { code: "AXS0202", title: "duplicate declaration", explain: "Two top-level declarations share a name." },
  { code: "AXS0203", title: "unknown dimension", explain: "A dimension symbol used in a type annotation was never declared and is not a declaration-local template variable." },
  { code: "AXS0204", title: "unknown operation", explain: "No catalog operation, block, fn or stage with that name is in scope." },
  { code: "AXS0301", title: "no implicit value", explain: "A bare transformation was written where no implicit value is in scope (multi-input entry point, or start of a block that has not seeded the cursor)." },
  { code: "AXS0302", title: "implicit value unused", explain: "A block produced an implicit value that is never returned or consumed." },
  { code: "AXS0303", title: "missing return", explain: "A declaration with a declared result produced no value." },
  { code: "AXS0304", title: "name rebound in scope", explain: "A `let` rebinds a name already bound by `let` in the same scope. The earlier value becomes unreachable, and a rebound stage application gets its own parameters rather than sharing them (see F-001)." },
  { code: "AXS0305", title: "model input unused", explain: "A model declares more than one input and its body never reads one of them. The data contract binds the field, but nothing consumes it — a padding mask that never reaches attention is the typical case." },
  { code: "AXS0401", title: "shape mismatch", explain: "Two operand shapes are incompatible on a specific axis." },
  { code: "AXS0402", title: "rank mismatch", explain: "An operation received a tensor of the wrong rank." },
  { code: "AXS0403", title: "unprovable constraint", explain: "The checker could not prove a symbolic dimension equality. It is carried as an assumption and verified at runtime." },
  { code: "AXS0404", title: "residual shape mismatch", explain: "A residual branch changed the shape of the tensor and no explicit projection was given." },
  { code: "AXS0405", title: "concat incompatibility", explain: "Branches being concatenated disagree on a non-concat axis." },
  { code: "AXS0406", title: "return contract violated", explain: "The produced shape does not match the declared result shape." },
  { code: "AXS0407", title: "divisibility not provable", explain: "A dimension expression requires exact division that cannot be proven." },
  { code: "AXS0408", title: "argument type mismatch", explain: "An operation received an argument of the wrong type." },
  { code: "AXS0409", title: "semantic kind mismatch", explain: "A tensor carrying a semantic kind was used where another kind is required." },
  { code: "AXS0410", title: "index out of bounds", explain: "A slice or index provably lies outside the axis it addresses (negative start, end beyond the extent, or start after end). Bounds that depend on runtime dimensions are carried as `≤` constraints instead (F-011)." },
  { code: "AXS0501", title: "missing required argument", explain: "A catalog operation requires an argument that describes an architectural choice." },
  { code: "AXS0502", title: "unknown argument", explain: "The operation does not accept that named argument." },
  { code: "AXS0503", title: "arity mismatch", explain: "Wrong number of positional arguments." },
  { code: "AXS0601", title: "objective must return Scalar", explain: "Objectives are typed graphs whose result is a scalar loss." },
  { code: "AXS0602", title: "data/model contract mismatch", explain: "A field fed to a model input has an incompatible shape or semantic kind." },
  { code: "AXS0603", title: "data/objective contract mismatch", explain: "A field fed to an objective target has an incompatible shape or semantic kind." },
  { code: "AXS0604", title: "unknown data field", explain: "The training plan referenced a field the data declaration does not construct." },
  { code: "AXS0620", title: "leakage: statistic fitted outside train split", explain: "Normalisation statistics or vocabularies must be fitted on the training split only." },
  { code: "AXS0621", title: "augmentation in eval pipeline", explain: "A stochastic augmentation was placed in the evaluation pipeline." },
  { code: "AXS0622", title: "unknown-category policy missing", explain: "A vocabulary is fitted on the training split but the pipeline does not say what happens to a category first seen at evaluation time (`unknown: \"<unk>\"` to map it, `unknown: error` to refuse it)." },
  { code: "AXS0701", title: "unknown parameter region", explain: "A lifecycle statement referenced a stage path that does not exist in the elaborated model." },
  { code: "AXS0702", title: "optimizer covers no parameters", explain: "An optimizer was declared over a region that owns no trainable parameters." },
  { code: "AXS0703", title: "parameter in two optimizers", explain: "A parameter is claimed by more than one optimizer in the same phase." },
  { code: "AXS0704", title: "frozen parameters updated", explain: "A phase updates an objective whose optimizer only covers frozen parameters." },
  { code: "AXS0705", title: "no phase declared", explain: "A training plan must contain at least one phase." },
  { code: "AXS0706", title: "one model declaration bound twice", explain: "Two `model` aliases in a training plan name the same model declaration. A model declaration is one parameter set, so the aliases share every parameter (F-015); wrap the shared block in two model declarations when independent copies are meant." },
  { code: "AXS0707", title: "conflicting phase configuration", explain: "A phase freezes and unfreezes the same region, declares both `epochs` and `steps`, states two `until` conditions, or overrides the learning rate of a region it freezes." },
  { code: "AXS0708", title: "optimizer or loss never applied", explain: "An optimizer is declared but no phase's `update` names it, so the parameters it covers are never trained; or a loss is declared but never updated. Both are silent in ordinary training code." },
  { code: "AXS0709", title: "unknown stopping metric", explain: "`until <metric> …` names a metric the lifecycle does not produce. Metrics are the loss names (training value), `train_<loss>` and `val_<loss>` (from `validate`)." },
  { code: "AXS0801", title: "shared stage shape conflict", explain: "A stage value applied twice received incompatible input shapes, so its parameters cannot be shared." },
  { code: "AXS0802", title: "parameter shape conflict", explain: "An explicit parameter declaration conflicts with a previous declaration in the same stage instance." },
  { code: "AXS0901", title: "backend capability missing", explain: "The selected backend cannot execute or differentiate this operation. This is a backend limitation, not a language semantic." },
  { code: "AXS0902", title: "gradient path blocked", explain: "An objective depends on a parameter only through a non-differentiable or gradient-stopped edge." },
  { code: "AXS0903", title: "custom op degrades static knowledge", explain: "A custom operation did not provide output shape semantics, so downstream shapes are unknown." },
  { code: "AXS1001", title: "runtime recurrence contract", explain: "A `scan` carried state whose shape changed between iterations." },
  { code: "AXS1002", title: "dynamic condition not statically bounded", explain: "A data-dependent branch was used where both arms must have identical static contracts." },
];

export function diag(
  code: string,
  severity: Severity,
  message: string,
  loc: Loc,
  where?: string,
  notes?: string[]
): Diagnostic {
  return { code, severity, message, loc, where, notes };
}
