import { ReactNode } from "react";

// ------------------------------------------------------------------ syntax highlighting

const KEYWORDS = [
  "dim", "model", "block", "fn", "objective", "source", "data", "train", "custom", "op", "let",
  "return", "yield", "residual", "via", "split", "merge", "branch", "for", "in", "param", "frozen",
  "state", "init", "update", "scan", "over", "carry", "axis", "example", "field", "preprocess",
  "augment", "batch", "shuffle", "from", "phase", "epochs", "steps", "epoch", "freeze", "unfreeze",
  "lr", "with", "times", "until", "every", "optimizer", "loss", "track", "weight", "effects",
  "backend", "shape", "differentiable", "device", "precision", "clip_grad", "validate", "checkpoint",
];
const TYPES = ["Tensor", "Scalar", "Tokens", "Class", "Mask", "Logits", "Probs", "Image", "Int", "Dim"];

interface Tok {
  t: string;
  c: string;
}

export function tokenizeLine(line: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const push = (t: string, c: string) => out.push({ t, c });
  while (i < line.length) {
    const rest = line.slice(i);
    if (rest.startsWith("#") || rest.startsWith("//")) {
      push(rest, "text-zinc-500 italic");
      break;
    }
    const str = /^"[^"]*"/.exec(rest);
    if (str) {
      push(str[0], "text-amber-300");
      i += str[0].length;
      continue;
    }
    const num = /^\d[\d_.]*(e-?\d+)?/.exec(rest);
    if (num) {
      push(num[0], "text-orange-300");
      i += num[0].length;
      continue;
    }
    const word = /^[A-Za-z_][A-Za-z0-9_']*/.exec(rest);
    if (word) {
      const w = word[0];
      const after = rest.slice(w.length);
      let cls = "text-zinc-100";
      if (KEYWORDS.includes(w)) cls = "text-fuchsia-300 font-medium";
      else if (TYPES.includes(w)) cls = "text-emerald-300";
      else if (/^\s*\(/.test(after)) cls = "text-sky-300";
      else if (/^[A-Z]/.test(w)) cls = "text-emerald-200";
      push(w, cls);
      i += w.length;
      continue;
    }
    const op = /^(\|>|->|\.\.|==|!=|>=|<=|[+\-*/(){}\[\],:.<>=|@%;])/.exec(rest);
    if (op) {
      push(op[0], op[0] === "|>" ? "text-cyan-300 font-semibold" : "text-zinc-400");
      i += op[0].length;
      continue;
    }
    push(rest[0], "text-zinc-300");
    i++;
  }
  return out;
}

export function Highlighted({ code, className = "" }: { code: string; className?: string }) {
  return (
    <code className={className}>
      {code.split("\n").map((line, li) => (
        <span key={li}>
          {tokenizeLine(line).map((tk, i) => (
            <span key={i} className={tk.c}>
              {tk.t}
            </span>
          ))}
          {"\n"}
        </span>
      ))}
    </code>
  );
}

export function CodeBlock({ code, lang = "axis" }: { code: string; lang?: string }) {
  return (
    <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-[#0b0d12] p-3 text-[12px] leading-[1.55]">
      {lang === "axis" ? (
        <Highlighted code={code} />
      ) : (
        <code className="text-zinc-300">{code}</code>
      )}
    </pre>
  );
}

// ------------------------------------------------------------------ markdown-lite

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) body.push(lines[i++]);
      i++;
      blocks.push(
        <div key={key++} className="my-3">
          <CodeBlock code={body.join("\n")} />
        </div>
      );
      continue;
    }
    if (line.startsWith("## ")) {
      blocks.push(
        <h3 key={key++} className="mt-6 mb-2 text-[13px] font-semibold uppercase tracking-wider text-cyan-300">
          {line.slice(3)}
        </h3>
      );
      i++;
      continue;
    }
    if (line.startsWith("> ")) {
      const body: string[] = [];
      while (i < lines.length && lines[i].startsWith("> ")) body.push(lines[i++].slice(2));
      blocks.push(
        <blockquote
          key={key++}
          className="my-3 border-l-2 border-cyan-500/60 bg-cyan-500/5 py-2 pl-3 text-[13px] italic text-cyan-100"
        >
          {inline(body.join(" "))}
        </blockquote>
      );
      continue;
    }
    if (/^[-*] /.test(line)) {
      const items: string[] = [];
      while (i < lines.length && (/^[-*] /.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
        if (/^[-*] /.test(lines[i])) items.push(lines[i].slice(2));
        else items[items.length - 1] += " " + lines[i].trim();
        i++;
      }
      blocks.push(
        <ul key={key++} className="my-2 space-y-1.5 pl-1 text-[13px] text-zinc-300">
          {items.map((it, n) => (
            <li key={n} className="flex gap-2">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-cyan-400" />
              <span>{inline(it)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }
    if (/^\d+\. /.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\. /.test(lines[i])) items.push(lines[i++].replace(/^\d+\. /, ""));
      blocks.push(
        <ol key={key++} className="my-2 list-decimal space-y-1.5 pl-5 text-[13px] text-zinc-300">
          {items.map((it, n) => (
            <li key={n}>{inline(it)}</li>
          ))}
        </ol>
      );
      continue;
    }
    if (line.trim() === "") {
      i++;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```") && !lines[i].startsWith("## ") && !/^[-*] /.test(lines[i]) && !lines[i].startsWith("> "))
      para.push(lines[i++]);
    blocks.push(
      <p key={key++} className="my-2 text-[13px] leading-relaxed text-zinc-300">
        {inline(para.join(" "))}
      </p>
    );
  }
  return <div>{blocks}</div>;
}

function inline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("`"))
      parts.push(
        <code key={k++} className="rounded bg-zinc-800/80 px-1 py-[1px] font-mono text-[12px] text-cyan-200">
          {tok.slice(1, -1)}
        </code>
      );
    else
      parts.push(
        <strong key={k++} className="font-semibold text-zinc-100">
          {tok.slice(2, -2)}
        </strong>
      );
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// ------------------------------------------------------------------ atoms

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-zinc-800 bg-zinc-900/40 ${className}`}>
      {title && (
        <header className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-400">{title}</h2>
          {right}
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Tag({ tone = "zinc", children }: { tone?: string; children: ReactNode }) {
  const tones: Record<string, string> = {
    zinc: "bg-zinc-800 text-zinc-300 border-zinc-700",
    cyan: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
    emerald: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    rose: "bg-rose-500/10 text-rose-300 border-rose-500/30",
    violet: "bg-violet-500/10 text-violet-300 border-violet-500/30",
    sky: "bg-sky-500/10 text-sky-300 border-sky-500/30",
    fuchsia: "bg-fuchsia-500/10 text-fuchsia-300 border-fuchsia-500/30",
    orange: "bg-orange-500/10 text-orange-300 border-orange-500/30",
  };
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-[1px] font-mono text-[10px] ${tones[tone] ?? tones.zinc}`}>
      {children}
    </span>
  );
}
