import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { compile } from "@/lang/analyze";
import { CATALOG } from "@/lang/catalog";
import { executeCommand } from "@/lang/cli";
import { COMPARISONS, COMPRESSION, DOCS, FINDING_KINDS, HARDENING, JOURNAL, LIMITATIONS, REPORT, toneOfKind } from "@/lang/docs";
import { emitTorch } from "@/lang/emit_torch";
import { EXAMPLES } from "@/lang/examples";
import { RunReport, runProgram } from "@/lang/exec";
import { inspectModule } from "@/lang/inspect";
import { printIR } from "@/lang/ir";
import { TestResult, runTests } from "@/lang/tests";
import { DIAGNOSTICS } from "@/lang/types";
import { DiagnosticList, Editor, InspectView, RunView, TestsView } from "@/components/panels";
import { CodeBlock, Markdown, Panel, Tag } from "@/components/ui";

type Page = "play" | "learn" | "compare" | "hardening" | "diagnostics" | "tests" | "report";
type Tab = "check" | "inspect" | "ir" | "emit" | "run" | "cli";

function safeText(fn: () => string): string {
  try {
    return fn();
  } catch (e) {
    return `internal error while producing this view: ${(e as Error).message}`;
  }
}

const NAV: { id: Page; label: string; hint: string }[] = [
  { id: "play", label: "Playground", hint: "edit, check, inspect, run" },
  { id: "learn", label: "Language", hint: "reference & semantics" },
  { id: "compare", label: "vs PyTorch", hint: "semantic compression" },
  { id: "hardening", label: "Hardening", hint: "where abstractions fail" },
  { id: "diagnostics", label: "Diagnostics", hint: "codes & catalog" },
  { id: "tests", label: "Tests", hint: "semantic + regression suite" },
  { id: "report", label: "Report", hint: "journal, limits, findings" },
];

export default function App() {
  const [page, setPage] = useState<Page>("play");
  const [exampleId, setExampleId] = useState(EXAMPLES[0].id);
  const [source, setSource] = useState(EXAMPLES[0].code);
  const [tab, setTab] = useState<Tab>("check");
  const [runReport, setRunReport] = useState<RunReport | null>(null);
  const [running, setRunning] = useState(false);
  const [tests, setTests] = useState<TestResult[] | null>(null);
  const [cliLog, setCliLog] = useState<{ cmd: string; out: string }[]>([]);
  const [cliInput, setCliInput] = useState("");
  const editorRef = useRef<HTMLDivElement>(null);

  const result = useMemo(() => compile(source, "playground"), [source]);
  const example = EXAMPLES.find((e) => e.id === exampleId);

  const inspectReport = useMemo(() => {
    try {
      return inspectModule(result.mod);
    } catch (e) {
      return `inspection failed: ${(e as Error).message}`;
    }
  }, [result]);
  const irText = useMemo(() => safeText(() => printIR(result.mod)), [result]);
  const torchText = useMemo(() => safeText(() => emitTorch(result.mod)), [result]);

  const loadExample = (id: string) => {
    const ex = EXAMPLES.find((e) => e.id === id);
    if (!ex) return;
    setExampleId(id);
    setSource(ex.code);
    setRunReport(null);
    setTab("check");
  };

  const doRun = useCallback(() => {
    setTab("run");
    setRunning(true);
    setTimeout(() => {
      try {
        setRunReport(runProgram(result.mod, { maxSteps: 6, dims: { B: 2 } }));
      } catch (e) {
        setRunReport({
          dims: [],
          forward: [],
          losses: [],
          finalMetrics: {},
          paramCount: 0,
          notes: [],
          errors: [`reference backend crashed: ${(e as Error).message}`],
          approximations: [],
          gradCoverage: [],
          phases: [],
          events: [],
        });
      }
      setRunning(false);
    }, 30);
  }, [result]);

  const doTests = useCallback(() => {
    setTests(null);
    setTimeout(() => setTests(runTests()), 30);
  }, []);

  useEffect(() => {
    if (page === "tests" && tests === null) doTests();
  }, [page, tests, doTests]);

  const errorCount = result.errors.length;
  const warnCount = result.warnings.length;

  return (
    <div className="min-h-screen bg-[#07080c] text-zinc-200 selection:bg-cyan-500/30">
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-[#07080c]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-cyan-400 to-violet-500 font-mono text-[13px] font-bold text-black">
              tn
            </div>
            <div>
              <div className="font-mono text-[15px] font-semibold tracking-tight text-zinc-100">TENSA</div>
              <div className="-mt-0.5 text-[10px] uppercase tracking-[0.18em] text-zinc-500">
                an ML-native language
              </div>
            </div>
          </div>
          <nav className="flex flex-1 flex-wrap gap-1">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => setPage(n.id)}
                title={n.hint}
                className={`rounded-md px-2.5 py-1.5 text-[12.5px] transition ${
                  page === n.id
                    ? "bg-zinc-800 text-cyan-300"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                }`}
              >
                {n.label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
            <span>tensors · graphs · state · lifecycle</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-4 py-4">
        {page === "play" && (
          <div className="mb-3 rounded-xl border border-zinc-800 bg-gradient-to-r from-cyan-500/[0.07] via-violet-500/[0.05] to-transparent px-4 py-3">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h1 className="text-[15px] font-semibold text-zinc-100">
                A language where the ML concepts are the language.
              </h1>
              <p className="text-[12.5px] text-zinc-400">
                Symbolic shapes, graph topology, parameter identity, effects, objectives, data semantics
                and training lifecycle are compiler objects — not framework conventions.
              </p>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {[
                "linear(256) — you state choices, the compiler infers consequences",
                "reusing a bound stage shares parameters",
                "split/merge is explicit in the IR",
                "fit: train is checked",
                "phases, not loops",
              ].map((s) => (
                <Tag key={s} tone="cyan">
                  {s}
                </Tag>
              ))}
            </div>
          </div>
        )}
        {page === "play" && (
          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="flex flex-col gap-3">
              <Panel
                title="program"
                right={
                  <div className="flex items-center gap-2">
                    <select
                      value={exampleId}
                      onChange={(e) => loadExample(e.target.value)}
                      className="max-w-[260px] rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-200 outline-none"
                    >
                      {[...new Set(EXAMPLES.map((e) => e.group))].map((g) => (
                        <optgroup key={g} label={g}>
                          {EXAMPLES.filter((e) => e.group === g).map((e) => (
                            <option key={e.id} value={e.id}>
                              {e.title}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                  </div>
                }
                className="flex-1"
              >
                {example && (
                  <p className="mb-2 text-[12px] leading-relaxed text-zinc-400">
                    <span className="text-zinc-300">{example.title}.</span> {example.summary}
                  </p>
                )}
                <div ref={editorRef} className="h-[52vh] min-h-[380px]">
                  <Editor value={source} onChange={setSource} diagnostics={result.mod.diags} />
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  {(
                    [
                      ["check", "Check"],
                      ["inspect", "Inspect"],
                      ["ir", "IR"],
                      ["emit", "Emit"],
                      ["run", "Run"],
                      ["cli", "CLI"],
                    ] as [Tab, string][]
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      onClick={() => (id === "run" ? doRun() : setTab(id))}
                      className={`rounded-md border px-3 py-1.5 text-[12px] transition ${
                        tab === id
                          ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-300"
                          : "border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-zinc-600"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                  <div className="ml-auto flex items-center gap-2 font-mono text-[11px]">
                    <Tag tone={errorCount ? "rose" : "emerald"}>{errorCount} errors</Tag>
                    <Tag tone={warnCount ? "amber" : "zinc"}>{warnCount} warnings</Tag>
                    <Tag tone="cyan">{result.mod.params.length} param tables</Tag>
                  </div>
                </div>
              </Panel>

              {example?.friction && example.friction.length > 0 && (
                <Panel title="friction analysis for this example">
                  <ul className="space-y-1.5 text-[12px]">
                    {example.friction.map((f, i) => (
                      <li key={i} className="flex gap-2">
                        <Tag tone={toneOfKind(f.kind)}>{f.kind}</Tag>
                        <span className="text-zinc-300">{f.note}</span>
                      </li>
                    ))}
                  </ul>
                </Panel>
              )}
            </div>

            <div className="min-w-0">
              {tab === "check" && (
                <Panel title="check — static semantic analysis">
                  <DiagnosticList diags={result.mod.diags} src={source} />
                  {result.mod.diags.length > 0 && (
                    <div className="mt-3 border-t border-zinc-800 pt-2 font-mono text-[11px] text-zinc-500">
                      {errorCount} error(s), {warnCount} warning(s) — warnings are constraints the compiler
                      carried rather than proved
                    </div>
                  )}
                </Panel>
              )}
              {tab === "inspect" &&
                (typeof inspectReport === "string" ? (
                  <Panel title="inspect">
                    <div className="text-[12px] text-rose-300">{inspectReport}</div>
                  </Panel>
                ) : (
                  <InspectView report={inspectReport} />
                ))}
              {tab === "ir" && (
                <Panel title="ir — backend-neutral intermediate representation">
                  <pre className="max-h-[75vh] overflow-auto whitespace-pre font-mono text-[11.5px] leading-[1.55] text-zinc-300">
                    {irText}
                  </pre>
                </Panel>
              )}
              {tab === "emit" && (
                <Panel title="emit — PyTorch backend">
                  <pre className="max-h-[75vh] overflow-auto whitespace-pre font-mono text-[11.5px] leading-[1.55] text-zinc-300">
                    {torchText}
                  </pre>
                </Panel>
              )}
              {tab === "run" &&
                (running ? (
                  <Panel title="run — reference backend">
                    <div className="animate-pulse text-[12px] text-cyan-300">executing…</div>
                  </Panel>
                ) : runReport ? (
                  <RunView report={runReport} />
                ) : (
                  <Panel title="run — reference backend">
                    <div className="text-[12px] text-zinc-500">press Run to execute with the reference backend</div>
                  </Panel>
                ))}
              {tab === "cli" && (
                <Panel title="cli — the same driver the portal uses">
                  <div className="max-h-[60vh] space-y-2 overflow-auto font-mono text-[11.5px]">
                    <div className="text-zinc-500">
                      try: check · inspect · ir · emit · run 4 · test · catalog · codes · examples · help
                    </div>
                    {cliLog.map((l, i) => (
                      <div key={i}>
                        <div className="text-cyan-300">axis {l.cmd}</div>
                        <pre className="whitespace-pre-wrap text-zinc-300">{l.out}</pre>
                      </div>
                    ))}
                  </div>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      const cmd = cliInput.trim();
                      if (!cmd) return;
                      setCliLog((l) => [...l, { cmd, out: executeCommand(cmd, source) }]);
                      setCliInput("");
                    }}
                    className="mt-2 flex items-center gap-2 border-t border-zinc-800 pt-2"
                  >
                    <span className="font-mono text-[12px] text-cyan-400">axis</span>
                    <input
                      value={cliInput}
                      onChange={(e) => setCliInput(e.target.value)}
                      placeholder="check"
                      className="flex-1 bg-transparent font-mono text-[12px] text-zinc-200 outline-none"
                    />
                  </form>
                </Panel>
              )}
            </div>
          </div>
        )}

        {page === "learn" && <LearnPage />}
        {page === "compare" && <ComparePage />}
        {page === "hardening" && <HardeningPage />}
        {page === "diagnostics" && <DiagnosticsPage />}
        {page === "tests" && (
          <div className="space-y-3">
            <Panel
              title="semantic & regression suite"
              right={
                <button
                  onClick={doTests}
                  className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:border-cyan-500/50 hover:text-cyan-300"
                >
                  re-run
                </button>
              }
            >
              <p className="text-[12px] text-zinc-400">
                Every bundled example is compiled here, together with targeted semantic tests: symbolic
                constraint propagation, residual and concat shape rules, the parallel-branch lowering
                regression, stage-identity parameter counting, effects and state, objective/data contracts,
                lifecycle resolution, backend lowering and end-to-end execution.
              </p>
            </Panel>
            {tests === null ? (
              <Panel>
                <div className="animate-pulse text-[12px] text-cyan-300">running suite…</div>
              </Panel>
            ) : (
              <TestsView results={tests} />
            )}
          </div>
        )}
        {page === "report" && <ReportPage />}
      </main>

      <footer className="mx-auto max-w-[1500px] px-4 pb-8 pt-4 text-[11px] text-zinc-600">
        TENSA — consolidated from the Vect, Axon and Lumen prototypes. Compiler, IR, reference backend,
        PyTorch emitter, test suite and portal all run in this page.
      </footer>
    </div>
  );
}

// ------------------------------------------------------------------ pages

function LearnPage() {
  const [sel, setSel] = useState(DOCS[0].id);
  const doc = DOCS.find((d) => d.id === sel)!;
  return (
    <div className="grid gap-3 lg:grid-cols-[220px_minmax(0,1fr)]">
      <nav className="space-y-1">
        {DOCS.map((d) => (
          <button
            key={d.id}
            onClick={() => setSel(d.id)}
            className={`w-full rounded-md px-2.5 py-2 text-left text-[12.5px] transition ${
              sel === d.id ? "bg-zinc-800 text-cyan-300" : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
            }`}
          >
            <div>{d.title}</div>
            <div className="text-[10.5px] text-zinc-600">{d.blurb}</div>
          </button>
        ))}
      </nav>
      <Panel title={doc.title}>
        <Markdown text={doc.body} />
      </Panel>
    </div>
  );
}

function ComparePage() {
  return (
    <div className="space-y-3">
      <Panel title="semantic compression — what actually shrinks">
        <table className="w-full text-[12px]">
          <thead className="text-zinc-500">
            <tr className="text-left">
              <th className="py-1 pr-3 font-normal">category</th>
              <th className="py-1 pr-3 font-normal">TENSA</th>
              <th className="py-1 font-normal">idiomatic PyTorch</th>
            </tr>
          </thead>
          <tbody>
            {COMPRESSION.categories.map((c) => (
              <tr key={c.id} className="border-t border-zinc-800/60">
                <td className="py-1.5 pr-3 text-zinc-200">
                  {c.id}. {c.name}
                </td>
                <td className="py-1.5 pr-3 text-cyan-300">{c.axis}</td>
                <td className="py-1.5 text-zinc-400">{c.torch}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[12px] leading-relaxed text-zinc-400">{COMPRESSION.note}</p>
      </Panel>
      {COMPARISONS.map((c) => (
        <Panel key={c.id} title={c.title}>
          <div className="grid gap-3 lg:grid-cols-2">
            <div>
              <div className="mb-1 text-[11px] uppercase tracking-wider text-cyan-400">TENSA</div>
              <CodeBlock code={c.axis} />
            </div>
            <div>
              <div className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">PyTorch</div>
              <CodeBlock code={c.torch} lang="py" />
            </div>
          </div>
          <p className="mt-3 text-[12px] leading-relaxed text-zinc-400">{c.note}</p>
        </Panel>
      ))}
    </div>
  );
}

function HardeningPage() {
  const kinds = FINDING_KINDS;
  return (
    <div className="space-y-3">
      <Panel title="semantic hardening — architectures chosen to break the abstraction">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {kinds.map((k) => (
            <div key={k.k} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
              <Tag tone={k.tone}>{k.k}</Tag>
              <div className="mt-1 text-[12.5px] text-zinc-200">{k.label}</div>
              <div className="text-[11px] text-zinc-500">{k.desc}</div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[12px] text-zinc-400">
          Difficulty was never answered by adding syntax on the spot. A finding was promoted to language
          semantics only when it recurred across several architectures.
        </p>
      </Panel>
      <Panel title="findings">
        <div className="space-y-2">
          {HARDENING.map((h, i) => (
            <div key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Tag tone={toneOfKind(h.kind)}>{h.kind}</Tag>
                <span className="text-[13px] text-zinc-100">{h.model}</span>
              </div>
              <div className="mt-1 text-[12px] text-zinc-400">{h.finding}</div>
              <div className="mt-1 text-[12px] text-cyan-200">→ {h.action}</div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function DiagnosticsPage() {
  const [q, setQ] = useState("");
  const list = DIAGNOSTICS.filter(
    (d) => !q || d.code.toLowerCase().includes(q.toLowerCase()) || d.title.toLowerCase().includes(q.toLowerCase())
  );
  return (
    <div className="space-y-3">
      <Panel
        title="diagnostic catalogue"
        right={
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="filter…"
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-200 outline-none"
          />
        }
      >
        <div className="space-y-1.5">
          {list.map((d) => (
            <div key={d.code} className="flex flex-wrap items-baseline gap-2 border-b border-zinc-800/60 pb-1.5">
              <Tag tone={d.code >= "AXS0900" ? "violet" : d.code >= "AXS0600" ? "amber" : "cyan"}>{d.code}</Tag>
              <span className="text-[12.5px] text-zinc-200">{d.title}</span>
              <span className="text-[11.5px] text-zinc-500">{d.explain}</span>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="standard catalog — library vocabulary, not language semantics">
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-[11.5px]">
            <thead className="text-zinc-500">
              <tr className="text-left">
                <th className="py-1 pr-3 font-normal">operation</th>
                <th className="py-1 pr-3 font-normal">style</th>
                <th className="py-1 pr-3 font-normal">ports</th>
                <th className="py-1 pr-3 font-normal">arguments</th>
                <th className="py-1 font-normal">effects</th>
              </tr>
            </thead>
            <tbody>
              {CATALOG.map((o) => (
                <tr key={o.name} className="border-t border-zinc-800/60">
                  <td className="py-1 pr-3 text-sky-300">{o.name}</td>
                  <td className="py-1 pr-3 text-zinc-500">{o.style}</td>
                  <td className="py-1 pr-3 text-zinc-400">
                    {[...o.ports, ...(o.optionalPorts ?? []).map((p) => `${p}?`)].join(", ") || "—"}
                  </td>
                  <td className="py-1 pr-3 text-zinc-400">
                    {o.config.map((c) => `${c.name}${c.required ? "*" : ""}`).join(", ") || "—"}
                  </td>
                  <td className="py-1 text-amber-300/80">{o.effects.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function ReportPage() {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel title="final report">
        <Markdown text={REPORT} />
      </Panel>
      <div className="space-y-3">
        <Panel title="design journal">
          <Markdown text={JOURNAL} />
        </Panel>
        <Panel title="known limitations">
          <Markdown text={LIMITATIONS} />
        </Panel>
      </div>
    </div>
  );
}
