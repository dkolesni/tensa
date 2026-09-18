/**
 * TENSA — lexer.  Newline sensitive: a newline terminates a statement unless the
 * line clearly continues (trailing operator / open bracket).
 */
import { Diagnostic, Loc, diag } from "./types";

export type TokKind =
  | "ident"
  | "kw"
  | "num"
  | "str"
  | "op"
  | "nl"
  | "eof";

export interface Token {
  kind: TokKind;
  text: string;
  value?: number;
  loc: Loc;
}

export const KEYWORDS = new Set([
  "dim", "model", "block", "fn", "objective", "source", "data", "train", "custom", "op",
  "let", "return", "yield", "residual", "via", "split", "merge", "branch", "for", "in",
  "param", "frozen", "state", "init", "update", "scan", "over", "carry", "axis",
  "example", "field", "preprocess", "augment", "batch", "shuffle", "from",
  "phase", "epochs", "steps", "epoch", "freeze", "unfreeze", "lr", "with", "times", "until",
  "every", "optimizer", "loss", "track", "weight", "effects", "backend", "true", "false",
]);

const MULTI_OPS = ["|>", "..", "->", ">=", "<=", "==", "!=", "::"];
const SINGLE_OPS = "+-*/(){}[],:.<>=|@%";

export function lex(src: string): { tokens: Token[]; diags: Diagnostic[] } {
  const tokens: Token[] = [];
  const diags: Diagnostic[] = [];
  let i = 0;
  let line = 1;
  let col = 1;

  const push = (kind: TokKind, text: string, loc: Loc, value?: number) =>
    tokens.push({ kind, text, loc, value });

  while (i < src.length) {
    const c = src[i];
    const loc: Loc = { line, col, len: 1 };

    if (c === "\n") {
      push("nl", "\\n", loc);
      i++;
      line++;
      col = 1;
      continue;
    }
    if (c === " " || c === "\t" || c === "\r") {
      i++;
      col++;
      continue;
    }
    // comments
    if (c === "#" || (c === "/" && src[i + 1] === "/")) {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && src[i + 1] === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) {
        if (src[i] === "\n") {
          line++;
          col = 1;
        }
        i++;
      }
      i += 2;
      continue;
    }
    if (c === ";") {
      push("nl", ";", loc);
      i++;
      col++;
      continue;
    }
    // numbers
    if (/[0-9]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9_]/.test(src[j])) j++;
      if (src[j] === "." && /[0-9]/.test(src[j + 1] ?? "")) {
        j++;
        while (j < src.length && /[0-9_]/.test(src[j])) j++;
      }
      if (src[j] === "e" || src[j] === "E") {
        let k = j + 1;
        if (src[k] === "-" || src[k] === "+") k++;
        if (/[0-9]/.test(src[k] ?? "")) {
          k++;
          while (k < src.length && /[0-9]/.test(src[k])) k++;
          j = k;
        }
      }
      const text = src.slice(i, j).replace(/_/g, "");
      loc.len = j - i;
      push("num", text, loc, parseFloat(text));
      col += j - i;
      i = j;
      continue;
    }
    // identifiers
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_']/.test(src[j])) j++;
      const text = src.slice(i, j);
      loc.len = j - i;
      push(KEYWORDS.has(text) ? "kw" : "ident", text, loc);
      col += j - i;
      i = j;
      continue;
    }
    // strings
    if (c === '"') {
      let j = i + 1;
      let out = "";
      while (j < src.length && src[j] !== '"') {
        if (src[j] === "\\") {
          j++;
          out += src[j] === "n" ? "\n" : src[j];
        } else out += src[j];
        j++;
      }
      loc.len = j - i + 1;
      push("str", out, loc);
      col += j - i + 1;
      i = j + 1;
      continue;
    }
    // operators
    const two = src.slice(i, i + 2);
    if (MULTI_OPS.includes(two)) {
      loc.len = 2;
      push("op", two, loc);
      i += 2;
      col += 2;
      continue;
    }
    if (SINGLE_OPS.includes(c)) {
      push("op", c, loc);
      i++;
      col++;
      continue;
    }
    diags.push(diag("AXS0101", "error", `unexpected character '${c}'`, loc));
    i++;
    col++;
  }
  tokens.push({ kind: "eof", text: "<eof>", loc: { line, col, len: 0 } });
  return { tokens, diags };
}
