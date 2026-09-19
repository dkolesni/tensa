/**
 * TENSA — command line driver.  The portal terminal and a node CLI would call
 * exactly this function; commands operate on the compiled IR only.
 */
import { compile } from "./analyze";
import { CATALOG } from "./catalog";
import { EXAMPLES } from "./examples";
import { emitTorch } from "./emit_torch";
import { runProgram } from "./exec";
import { renderInspect, inspectModule } from "./inspect";
import { printIR } from "./ir";
import { runTests, testSummary } from "./tests";
import { DIAGNOSTICS, Diagnostic } from "./types";

export const HELP = `axis <command> [file.axis]

  check      parse + full static semantic analysis, no execution
  inspect    show what the compiler actually built (graph, shapes, parameters,
             sharing, state, effects, lifecycle, checkpoint coverage)
  ir         print the backend-neutral intermediate representation
  emit       print generated PyTorch for the selected backend
  run        explicitly use the CPU reference interpreter (validation only)
  test       run the semantic / regression suite
  examples   list the bundled examples
  catalog    list the standard catalog with shape rules and effects
  codes      print the diagnostic catalogue
  help       this message

Diagnostics use stable codes (AXS####) with source locations and the
architectural location of the failure.`;

export function formatDiagnostic(d: Diagnostic, src?: string): string {
  const head = `${d.severity === "error" ? "error" : d.severity === "warning" ? "warning" : "note"}[${d.code}] ${d.loc.line}:${d.loc.col}: ${d.message}`;
  const lines: string[] = [head];
  if (src) {
    const line = src.split("\n")[d.loc.line - 1];
    if (line !== undefined) {
      lines.push(`   | ${line}`);
      lines.push(`   | ${" ".repeat(Math.max(0, d.loc.col - 1))}${"^".repeat(Math.max(1, d.loc.len || 1))}`);
    }
  }
  if (d.where) lines.push(`   in ${d.where}`);
  for (const n of d.notes ?? []) lines.push(`   note: ${n}`);
  return lines.join("\n");
}

export function executeCommand(cmd: string, src: string): string {
  const [verb, ...rest] = cmd.trim().split(/\s+/);
  switch (verb) {
    case "":
      return "";
    case "help":
    case "?":
      return HELP;
    case "check": {
      const r = compile(src);
      const out = r.mod.diags.map((d) => formatDiagnostic(d, src)).join("\n\n");
      const summary = `${r.errors.length} error(s), ${r.warnings.length} warning(s)`;
      if (r.ok && r.mod.diags.length === 0)
        return `ok — ${r.mod.graphs.length} graph(s), ${r.mod.objectives.length} objective(s), ${r.mod.params.length} parameter table(s), ${r.mod.states.length} state slot(s)\nno diagnostics`;
      return `${out}\n\n${summary}`;
    }
    case "ir":
      return printIR(compile(src).mod);
    case "inspect":
      return renderInspect(inspectModule(compile(src).mod));
    case "emit":
      return emitTorch(compile(src).mod);
    case "run": {
      const r = compile(src);
      if (!r.ok)
        return `refusing to run: ${r.errors.length} error(s)\n\n${r.errors.map((d) => formatDiagnostic(d, src)).join("\n\n")}`;
      const rep = runProgram(r.mod, { maxSteps: Number(rest[0] ?? 6) });
      const L: string[] = ["CPU reference validation (explicit reference backend); GPU execution uses emitted PyTorch."];
      L.push(`dimension bindings: ${rep.dims.map((d) => `${d.name}=${d.value}`).join(", ")}`);
      L.push(`parameters: ${rep.paramCount.toLocaleString()} values`);
      L.push("");
      for (const f of rep.forward) L.push(`forward ${f.model}(${f.inputs.join("; ")}) -> ${f.output}   [${f.ms} ms]`);
      if (rep.losses.length) {
        L.push("");
        L.push("training:");
        for (const s of rep.losses)
          L.push(
            `  step ${String(s.step).padStart(3)} [${s.phase}] ${Object.entries(s.values)
              .map(([k, v]) => `${k}=${v.toFixed(4)}`)
              .join("  ")}`
          );
      }
      if (rep.phases.length) {
        L.push("");
        L.push("lifecycle:");
        for (const p of rep.phases)
          L.push(`  phase ${p.name}: ${p.steps} steps, stopped by ${p.stoppedBy}, lr × ${p.lrStart.toFixed(3)} → ${p.lrEnd.toFixed(3)}`);
        for (const e of rep.events) L.push(`  step ${String(e.step).padStart(3)} [${e.phase}] ${e.action} ${e.detail}`);
      }
      if (Object.keys(rep.finalMetrics).length) {
        L.push("");
        L.push(`metrics: ${JSON.stringify(rep.finalMetrics)}`);
      }
      if (rep.errors.length) {
        L.push("");
        L.push("backend errors:");
        for (const e of rep.errors) L.push(`  ${e}`);
      }
      L.push("");
      L.push("approximations:");
      for (const a of rep.approximations) L.push(`  - ${a}`);
      return L.join("\n");
    }
    case "test": {
      const rs = runTests();
      const s = testSummary(rs);
      const failed = rs.filter((r) => !r.ok);
      const L = [`${s.passed}/${s.total} tests passed`];
      if (failed.length) {
        L.push("");
        for (const f of failed) L.push(`FAIL [${f.group}] ${f.name}\n      ${f.detail}`);
      }
      return L.join("\n");
    }
    case "examples":
      return EXAMPLES.map((e) => `${e.id.padEnd(16)} ${e.group.padEnd(18)} ${e.title}`).join("\n");
    case "catalog":
      return CATALOG.map(
        (o) =>
          `${o.name.padEnd(18)} ${o.style.padEnd(10)} ports(${o.ports.join(",") || "-"})  args(${o.config
            .map((c) => c.name + (c.required ? "*" : ""))
            .join(",") || "-"})  effects{${o.effects.join(",")}}`
      ).join("\n");
    case "codes":
      return DIAGNOSTICS.map((d) => `${d.code}  ${d.title}\n        ${d.explain}`).join("\n");
    default:
      return `unknown command '${verb}'\n\n${HELP}`;
  }
}
