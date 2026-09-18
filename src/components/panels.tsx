import { useMemo, useRef } from "react";
import { RunReport } from "@/lang/exec";
import { GraphLine, InspectReport } from "@/lang/inspect";
import { TestResult } from "@/lang/tests";
import { Diagnostic } from "@/lang/types";
import { Highlighted, Panel, Tag } from "./ui";

// ------------------------------------------------------------------ editor

export function Editor({
  value,
  onChange,
  diagnostics,
  onGoto,
}: {
  value: string;
  onChange: (v: string) => void;
  diagnostics: Diagnostic[];
  onGoto?: (line: number) => void;
}) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const preRef = useRef<HTMLPreElement>(null);
  const lines = value.split("\n");
  const marks = new Map<number, string>();
  for (const d of diagnostics) {
    const cur = marks.get(d.loc.line);
    if (cur === "error") continue;
    marks.set(d.loc.line, d.severity);
  }
  const sync = () => {
    if (preRef.current && taRef.current) {
      preRef.current.scrollTop = taRef.current.scrollTop;
      preRef.current.scrollLeft = taRef.current.scrollLeft;
    }
  };
  return (
    <div className="relative flex h-full overflow-hidden rounded-xl border border-zinc-800 bg-[#0b0d12]">
      <div className="select-none overflow-hidden border-r border-zinc-800/80 bg-zinc-900/40 py-3 text-right font-mono text-[12px] leading-[1.55] text-zinc-600">
        {lines.map((_, i) => (
          <div
            key={i}
            onClick={() => onGoto?.(i + 1)}
            className={`px-2 ${
              marks.get(i + 1) === "error"
                ? "bg-rose-500/20 text-rose-300"
                : marks.get(i + 1) === "warning"
                ? "bg-amber-500/15 text-amber-300"
                : ""
            }`}
          >
            {i + 1}
          </div>
        ))}
      </div>
      <div className="relative flex-1 overflow-hidden">
        <pre
          ref={preRef}
          aria-hidden
          className="pointer-events-none absolute inset-0 overflow-auto whitespace-pre p-3 font-mono text-[12px] leading-[1.55]"
        >
          <Highlighted code={value + "\n"} />
        </pre>
        <textarea
          ref={taRef}
          value={value}
          spellCheck={false}
          onScroll={sync}
          onChange={(e) => onChange(e.target.value)}
          className="absolute inset-0 h-full w-full resize-none overflow-auto whitespace-pre bg-transparent p-3 font-mono text-[12px] leading-[1.55] text-transparent caret-cyan-300 outline-none"
        />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ diagnostics

export function DiagnosticList({
  diags,
  src,
  onGoto,
}: {
  diags: Diagnostic[];
  src: string;
  onGoto?: (line: number) => void;
}) {
  if (!diags.length)
    return (
      <div className="flex items-center gap-2 text-[12px] text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> no diagnostics — the program is
        statically well-formed
      </div>
    );
  const srcLines = src.split("\n");
  return (
    <div className="space-y-2">
      {diags.map((d, i) => (
        <div
          key={i}
          onClick={() => onGoto?.(d.loc.line)}
          className={`cursor-pointer rounded-lg border px-3 py-2 ${
            d.severity === "error"
              ? "border-rose-500/30 bg-rose-500/5"
              : d.severity === "warning"
              ? "border-amber-500/30 bg-amber-500/5"
              : "border-sky-500/30 bg-sky-500/5"
          }`}
        >
          <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
            <Tag tone={d.severity === "error" ? "rose" : d.severity === "warning" ? "amber" : "cyan"}>
              {d.code}
            </Tag>
            <span className="text-zinc-500">
              line {d.loc.line}:{d.loc.col}
            </span>
            {d.where && <span className="text-zinc-500">in {d.where}</span>}
          </div>
          <div className="mt-1 text-[12.5px] text-zinc-200">{d.message}</div>
          {srcLines[d.loc.line - 1] !== undefined && (
            <pre className="mt-1.5 overflow-x-auto font-mono text-[11px] text-zinc-500">
              {srcLines[d.loc.line - 1]}
              {"\n"}
              {" ".repeat(Math.max(0, d.loc.col - 1))}
              <span className="text-rose-400">{"^".repeat(Math.max(1, d.loc.len || 1))}</span>
            </pre>
          )}
          {(d.notes ?? []).map((n, k) => (
            <div key={k} className="mt-1 text-[11.5px] text-zinc-400">
              → {n}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ inspect

function GraphLines({ lines }: { lines: GraphLine[] }) {
  return (
    <div className="overflow-x-auto font-mono text-[11.5px] leading-[1.6]">
      {lines.map((l, i) => (
        <div
          key={i}
          className="flex gap-2 whitespace-pre"
          style={{ paddingLeft: `${l.depth * 14}px` }}
        >
          {l.kind === "region" ? (
            <span className="text-violet-300">
              ┌ {l.text}
              {l.detail ? <span className="text-zinc-600"> ({l.detail})</span> : null}
            </span>
          ) : (
            <>
              <span className={l.kind === "merge" ? "text-amber-300" : "text-sky-300"}>
                {l.kind === "merge" ? "▼ " : "· "}
                {l.text}
              </span>
              {l.detail && <span className="text-zinc-600">{l.detail}</span>}
              {l.shape && <span className="text-emerald-300/90">{l.shape}</span>}
              {l.params.length > 0 && <span className="text-fuchsia-300/80">params×{l.params.length}</span>}
              {l.effects.map((e) => (
                <span key={e} className="text-amber-300/80">
                  {"{"}
                  {e}
                  {"}"}
                </span>
              ))}
              {l.shared && <span className="text-cyan-300">shared:{l.shared}</span>}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export function InspectView({ report }: { report: InspectReport }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="parameter tables" value={String(report.params.length)} />
        <Stat label="parameter values" value={(report.totalParams ?? 0).toLocaleString()} />
        <Stat label="trainable" value={(report.trainableParams ?? 0).toLocaleString()} />
        <Stat label="state slots" value={String(report.states.length)} />
      </div>

      {report.graphs.map((g) => (
        <Panel
          key={g.name}
          title={`${g.kind} ${g.name}`}
          right={<span className="font-mono text-[11px] text-zinc-500">{g.signature}</span>}
        >
          <GraphLines lines={g.lines} />
        </Panel>
      ))}

      <Panel title="parameters — ownership, sharing, trainability">
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-[11.5px]">
            <thead className="text-zinc-500">
              <tr className="text-left">
                <th className="py-1 pr-3 font-normal">table</th>
                <th className="py-1 pr-3 font-normal">shape</th>
                <th className="py-1 pr-3 font-normal">values</th>
                <th className="py-1 pr-3 font-normal">kind</th>
                <th className="py-1 pr-3 font-normal">state</th>
                <th className="py-1 font-normal">applications</th>
              </tr>
            </thead>
            <tbody>
              {report.params.map((p) => (
                <tr key={p.id} className="border-t border-zinc-800/60">
                  <td className="py-1 pr-3 text-zinc-300">{p.id}</td>
                  <td className="py-1 pr-3 text-emerald-300/90">{p.shape}</td>
                  <td className="py-1 pr-3 text-zinc-400">{p.count?.toLocaleString() ?? "?"}</td>
                  <td className="py-1 pr-3 text-zinc-500">{p.kind}</td>
                  <td className="py-1 pr-3">
                    {p.trainable ? <Tag tone="emerald">trainable</Tag> : <Tag tone="amber">frozen</Tag>}
                  </td>
                  <td className="py-1">
                    {p.applications > 1 ? (
                      <Tag tone="cyan">shared ×{p.applications}</Tag>
                    ) : (
                      <span className="text-zinc-600">1</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {report.sharedGroups.length > 0 && (
          <div className="mt-3 space-y-1 text-[12px] text-cyan-200">
            {report.sharedGroups.map((s) => (
              <div key={s.stage}>
                shared stage <span className="font-mono">{s.stage}</span>: {s.applications} applications over{" "}
                {s.tables} tables — {(s.savedParams ?? 0).toLocaleString()} values not duplicated
              </div>
            ))}
          </div>
        )}
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="persistent state">
          {report.states.length === 0 ? (
            <Empty>no persistent state</Empty>
          ) : (
            <ul className="space-y-1 font-mono text-[11.5px]">
              {report.states.map((s) => (
                <li key={s.id} className="flex flex-wrap gap-2">
                  <span className="text-zinc-300">{s.id}</span>
                  <span className="text-emerald-300/90">{s.shape}</span>
                  <Tag tone="violet">{s.category}</Tag>
                  <span className="text-zinc-500">update={s.update}</span>
                  {s.checkpointed && <Tag tone="cyan">checkpointed</Tag>}
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="effects">
          {report.effects.length === 0 ? (
            <Empty>every operation is pure</Empty>
          ) : (
            <ul className="space-y-1.5 text-[12px]">
              {report.effects.map((e) => (
                <li key={e.effect}>
                  <Tag tone="amber">{e.effect}</Tag>{" "}
                  <span className="font-mono text-[11px] text-zinc-400">{e.sites.slice(0, 4).join(", ")}</span>
                  {e.sites.length > 4 && <span className="text-zinc-600"> +{e.sites.length - 4} more</span>}
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="symbolic dimensions & constraints">
          <div className="flex flex-wrap gap-1.5">
            {report.dims.map((d) => (
              <Tag key={d.name} tone={d.kind === "static" ? "emerald" : "cyan"}>
                {d.name} = {d.value}
              </Tag>
            ))}
          </div>
          {report.constraints.length > 0 && (
            <ul className="mt-3 space-y-1 font-mono text-[11px]">
              {report.constraints.slice(0, 14).map((c, i) => (
                <li key={i} className="flex flex-wrap gap-2">
                  <Tag tone={c.status === "proved" ? "emerald" : c.status === "assumed" ? "amber" : "rose"}>
                    {c.status}
                  </Tag>
                  <span className="text-zinc-300">{c.text}</span>
                  <span className="text-zinc-600">{c.origin}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="checkpoint coverage">
          <ul className="space-y-1 text-[12px]">
            {report.checkpoint.map((c) => (
              <li key={c.kind} className="flex justify-between gap-3">
                <span className="text-zinc-300">{c.kind}</span>
                <span className="font-mono text-[11px] text-zinc-500">
                  {c.items} — {c.detail}
                </span>
              </li>
            ))}
          </ul>
          {report.capabilities.length > 0 && (
            <div className="mt-3 border-t border-zinc-800 pt-2">
              <div className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">backend capabilities</div>
              <ul className="space-y-1 font-mono text-[11px]">
                {report.capabilities.map((c) => (
                  <li key={c.op}>
                    <span className="text-zinc-300">{c.op}</span>{" "}
                    <Tag tone={c.grad ? "emerald" : "amber"}>grad {c.grad ? "yes" : "no"}</Tag>{" "}
                    <span className="text-zinc-500">{c.note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      </div>

      {(report.compatibility.length > 0 || report.lifecycle.length > 0) && (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel title="data / model / objective contracts">
            {report.compatibility.length === 0 ? (
              <Empty>no data or objective bindings</Empty>
            ) : (
              <pre className="whitespace-pre-wrap font-mono text-[11.5px] text-zinc-300">
                {report.compatibility.join("\n")}
              </pre>
            )}
          </Panel>
          <Panel title="training lifecycle">
            {report.lifecycle.length === 0 ? (
              <Empty>no training plan</Empty>
            ) : (
              <pre className="whitespace-pre-wrap font-mono text-[11.5px] text-zinc-300">
                {report.lifecycle.join("\n")}
              </pre>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.12em] text-zinc-500">{label}</div>
      <div className="font-mono text-lg text-cyan-300">{value}</div>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-[12px] text-zinc-600">{children}</div>;
}

// ------------------------------------------------------------------ run

export function RunView({ report }: { report: RunReport }) {
  const series = useMemo(() => {
    const names = new Set<string>();
    for (const l of report.losses) for (const k of Object.keys(l.values)) names.add(k);
    return [...names].map((name) => ({
      name,
      points: report.losses.map((l, i) => ({ x: i, y: l.values[name] })).filter((p) => p.y !== undefined),
    }));
  }, [report]);
  const colors = ["#22d3ee", "#a78bfa", "#fbbf24", "#34d399"];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="parameter values" value={report.paramCount.toLocaleString()} />
        <Stat label="training steps" value={String(report.losses.length)} />
        <Stat
          label="runtime dims"
          value={report.dims
            .slice(0, 3)
            .map((d) => `${d.name}=${d.value}`)
            .join(" ")}
        />
        <Stat label="backend errors" value={String(report.errors.length)} />
      </div>

      <Panel title="forward — CPU reference interpreter (validation)">
        <ul className="space-y-1 font-mono text-[11.5px]">
          {report.forward.map((f) => (
            <li key={f.model} className="flex flex-wrap gap-2">
              <span className="text-sky-300">{f.model}</span>
              <span className="text-zinc-500">({f.inputs.join("; ")})</span>
              <span className="text-zinc-400">→</span>
              <span className="text-emerald-300">{f.output}</span>
              <span className="text-zinc-600">{f.ms} ms</span>
            </li>
          ))}
        </ul>
      </Panel>

      {series.length > 0 && (
        <Panel title="training — CPU reference interpreter (validation)">
          <LossChart series={series} colors={colors} />
          <div className="mt-2 flex flex-wrap gap-3 text-[11px]">
            {series.map((s, i) => (
              <span key={s.name} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: colors[i % colors.length] }} />
                <span className="font-mono text-zinc-300">{s.name}</span>
              </span>
            ))}
          </div>
          <div className="mt-3 max-h-40 overflow-y-auto font-mono text-[11px] text-zinc-400">
            {report.losses.map((l, i) => (
              <div key={i}>
                step {String(l.step).padStart(3)} [{l.phase}]{" "}
                {Object.entries(l.values)
                  .map(([k, v]) => `${k}=${v.toFixed(4)}`)
                  .join("  ")}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {report.phases.length > 0 && (
        <Panel title="lifecycle — phases and events">
          <div className="space-y-1 font-mono text-[11px] text-zinc-400">
            {report.phases.map((p) => (
              <div key={p.name} className="flex flex-wrap items-center gap-2">
                <span className="text-sky-300">{p.name}</span>
                <span>{p.steps} steps</span>
                <Tag tone={p.stoppedBy === "until" ? "emerald" : "zinc"}>stopped by {p.stoppedBy}</Tag>
                {(p.lrStart !== 1 || p.lrEnd !== 1) && (
                  <span className="text-zinc-600">
                    lr × {p.lrStart.toFixed(3)} → {p.lrEnd.toFixed(3)}
                  </span>
                )}
              </div>
            ))}
            {report.events.map((e, i) => (
              <div key={i} className="text-zinc-500">
                step {String(e.step).padStart(3)} [{e.phase}] {e.action} {e.detail}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {report.gradCoverage.length > 0 && (
        <Panel title="gradient coverage">
          <div className="grid gap-1 font-mono text-[11px] sm:grid-cols-2">
            {report.gradCoverage.map((g) => (
              <div key={g.param} className="flex items-center gap-2">
                <Tag tone={g.updated ? "emerald" : "amber"}>{g.updated ? "updated" : "no update"}</Tag>
                <span className="text-zinc-400">{g.param}</span>
                {g.reason && <span className="text-zinc-600">{g.reason}</span>}
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel title="honest reporting">
        <ul className="space-y-1 text-[12px] text-zinc-400">
          {report.errors.map((e, i) => (
            <li key={`e${i}`} className="text-rose-300">
              error: {e}
            </li>
          ))}
          {report.notes.map((n, i) => (
            <li key={`n${i}`}>note: {n}</li>
          ))}
          {report.approximations.map((a, i) => (
            <li key={`a${i}`}>approximation: {a}</li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}

function LossChart({
  series,
  colors,
}: {
  series: { name: string; points: { x: number; y: number }[] }[];
  colors: string[];
}) {
  const all = series.flatMap((s) => s.points.map((p) => p.y));
  if (!all.length) return null;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const n = Math.max(...series.map((s) => s.points.length), 2);
  const W = 640;
  const H = 150;
  const x = (i: number) => (i / (n - 1)) * (W - 30) + 25;
  const y = (v: number) => H - 20 - ((v - min) / Math.max(max - min, 1e-9)) * (H - 35);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1="25" y1={H - 20} x2={W - 5} y2={H - 20} stroke="#3f3f46" strokeWidth="1" />
      <line x1="25" y1="10" x2="25" y2={H - 20} stroke="#3f3f46" strokeWidth="1" />
      <text x="2" y="16" fill="#71717a" fontSize="9" fontFamily="monospace">
        {max.toFixed(2)}
      </text>
      <text x="2" y={H - 22} fill="#71717a" fontSize="9" fontFamily="monospace">
        {min.toFixed(2)}
      </text>
      {series.map((s, si) => (
        <polyline
          key={s.name}
          fill="none"
          stroke={colors[si % colors.length]}
          strokeWidth="1.5"
          points={s.points.map((p, i) => `${x(i)},${y(p.y)}`).join(" ")}
        />
      ))}
    </svg>
  );
}

// ------------------------------------------------------------------ tests

export function TestsView({ results }: { results: TestResult[] }) {
  const groups = useMemo(() => {
    const m = new Map<string, TestResult[]>();
    for (const r of results) m.set(r.group, [...(m.get(r.group) ?? []), r]);
    return [...m.entries()];
  }, [results]);
  const passed = results.filter((r) => r.ok).length;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <Stat label="tests" value={String(results.length)} />
        <Stat label="passed" value={String(passed)} />
        <Stat label="failed" value={String(results.length - passed)} />
      </div>
      {groups.map(([group, rs]) => (
        <Panel
          key={group}
          title={group}
          right={
            <span className="font-mono text-[11px] text-zinc-500">
              {rs.filter((r) => r.ok).length}/{rs.length}
            </span>
          }
        >
          <ul className="space-y-1">
            {rs.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-[12px]">
                <span className={`mt-[3px] font-mono text-[10px] ${r.ok ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.ok ? "PASS" : "FAIL"}
                </span>
                <span className="text-zinc-300">{r.name}</span>
                <span className={`font-mono text-[11px] ${r.ok ? "text-zinc-600" : "text-rose-300"}`}>
                  {r.detail}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      ))}
    </div>
  );
}
