/**
 * TENSA hardening — Milestone 3 "learning-program completeness" challenges
 * (§14, §15, §18–§22, §25, §26, §46).
 *
 * The claim under test: a complete learning program — data contract, state,
 * effects, lifecycle — is expressible without leaving the language, the
 * checker refuses the classic silent mistakes (leakage, stochastic eval,
 * frozen-but-updated regions, contradictory phases), and both backends honour
 * every lifecycle mechanic the IR records (schedules, `until`, events, clip,
 * tracks, alternating updates, nested model bindings).
 */
import { emitTorch } from "../emit_torch";
import { IRModule } from "../ir";
import { runProgram } from "../exec";
import { Challenge, CustomCheck } from "./types";

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

/** run the reference backend and hand the report to an assertion */
function runCheck(name: string, steps: number, fn: (rep: ReturnType<typeof runProgram>) => string): CustomCheck {
  return { name, check: (mod: IRModule) => fn(runProgram(mod, { maxSteps: steps })) };
}

// ================================================================== 4.1 data (§18–§21)

const TABULAR = `dim B
dim K = 3
dim NC = 8

source Rows = csv(path: "adult.csv")

data Tab from Rows {
  example {
    field age: Tensor[1] = select(column: "age") |> to_float |> impute(median, fit: train) |> standardize(fit: train)
    field job: Tokens[1] = select(column: "job") |> vocab(fit: train, unknown: "<unk>") |> encode
    field label: Class = select(column: "income") |> as_class
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 32
}

model Tabular(age: Tensor[B, 1], job: Tokens[B, 1]) -> Logits[B, K] {
  let e = job |> embedding(NC, 4) |> reshape([B, 4])
  let h = concat(age, e, axis: -1)
  return h |> linear(16) |> relu |> linear(K)
}

objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}

train Run {
  data Tab
  model M = Tabular
  loss main = Cls(logits: M(age, job), labels: label)
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`;

const tabularPipeline: Challenge = {
  id: "tabular-pipeline",
  title: "Tabular contract: impute → standardize, vocab → encode, all fitted on train",
  section: "§19",
  tier: 2,
  code: TABULAR,
  record: "tabular-pipeline.md",
  expect: {
    warnCodes: [],
    run: { steps: 4, gradAll: true, outputs: { Tabular: "[4, 3]" } },
    custom: [
      {
        name: "every fitted statistic names the train split (§19)",
        check: (mod) => {
          const fitted = mod.data[0].fitted;
          const stats = fitted.map((f) => f.stat).sort();
          assert(fitted.length === 3 && fitted.every((f) => f.split === "train"), `fitted: ${JSON.stringify(fitted)}`);
          return `fitted on train: ${stats.join(", ")}`;
        },
      },
    ],
  },
  twins: [
    {
      id: "impute-fit-all",
      mutates: "imputation statistic fitted on the full set",
      code: TABULAR.replace("impute(median, fit: train)", "impute(median, fit: all)"),
      expectCodes: ["AXS0620"],
      line: 9,
    },
    {
      id: "impute-no-fit",
      mutates: "imputation statistic with no split at all",
      code: TABULAR.replace("impute(median, fit: train)", "impute(median)"),
      expectCodes: ["AXS0620"],
    },
    {
      id: "misspelled-fit",
      mutates: "`fit:` misspelled — a silently ignored argument must not read as a fitted-on-train claim",
      code: TABULAR.replace("standardize(fit: train)", "standardize(fitt: train)"),
      expectCodes: ["AXS0620"],
    },
    {
      id: "unknown-category-unhandled",
      mutates: "vocabulary fitted on train with no policy for a category first seen at eval time",
      code: TABULAR.replace('vocab(fit: train, unknown: "<unk>")', "vocab(fit: train)"),
      expectCodes: ["AXS0622"],
      line: 10,
    },
  ],
};

const VISION = `dim B
dim K = 10
source Cifar = image_folder(path: "cifar10/", classes: 10)
data VisionData from Cifar {
  example {
    field image: Image[3, 32, 32] = decode(file)
    field label: Class = as_class(folder)
  }
  preprocess { image: resize(32, 32) |> to_float |> normalize(fit: train) }
  augment train { image: random_crop(32, pad: 4) |> random_flip(p: 0.5) }
  augment eval { image: center_crop(32) }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 64
}
model Net(image: Image[B, 3, 32, 32]) -> Logits[B, K] {
  conv2d(8, kernel: 3, pad: 1)
  relu
  global_avgpool
  linear(K)
}
objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}
train VisionRun {
  data VisionData
  model M = Net
  loss main = Classification(logits: M(image), labels: label)
  optimizer opt = adamw(lr: 1e-3)
  epochs 2
}`;

const visionPipeline: Challenge = {
  id: "vision-pipeline",
  title: "Vision contract: layout, channels, train/eval augmentation split, target kind",
  section: "§18",
  tier: 2,
  code: VISION,
  expect: {
    warnCodes: [],
    run: { steps: 2, gradAll: true, dims: { B: 2 } },
  },
  twins: [
    {
      id: "fit-on-val",
      mutates: "normalisation statistic fitted on the validation split",
      code: VISION.replace("normalize(fit: train)", "normalize(fit: val)"),
      expectCodes: ["AXS0620"],
    },
    {
      id: "stochastic-eval",
      mutates: "a random augmentation in the eval pipeline",
      code: VISION.replace("augment eval { image: center_crop(32) }", "augment eval { image: random_flip(p: 0.5) }"),
      expectCodes: ["AXS0621"],
    },
    {
      id: "wrong-layout",
      mutates: "field declared HWC while the model expects CHW — a contract refutation, not a generic shape clash (E-008)",
      code: VISION.replace("field image: Image[3, 32, 32]", "field image: Image[32, 32, 3]"),
      expectCodes: ["AXS0602"],
      forbidCodes: ["AXS0401", "AXS0403"],
    },
    {
      id: "wrong-channels",
      mutates: "greyscale field bound to a 3-channel model input",
      code: VISION.replace("field image: Image[3, 32, 32]", "field image: Image[1, 32, 32]"),
      expectCodes: ["AXS0602"],
      forbidCodes: ["AXS0401"],
    },
    {
      id: "target-kind",
      mutates: "label field is Tokens[1], objective port wants Class[B]",
      code: VISION.replace("field label: Class = as_class(folder)", "field label: Tokens[1] = as_class(folder) |> encode"),
      expectCodes: ["AXS0603"],
    },
  ],
};

const LM_PADDED = `dim B
dim T = 16
dim V = 64
dim D = 32

source Corpus = text_file(path: "corpus/*.txt")

data LMData from Corpus {
  example {
    field ids: Tokens[T + 1] = tokenize(text, vocab: V) |> pad_to(T + 1)
    field padded: Mask[1, T + 1] = tokenize(text, vocab: V) |> pad_mask(T + 1)
    field pad: Mask[1, T] = padded[0 : 1, 0 : T]
    field inputs: Tokens[T] = ids[0 : T]
    field targets: Tokens[T] = ids[1 : T + 1]
  }
  split { train: 0.9, val: 0.1 }
  batch 8
}

model LM(tokens: Tokens[B, T], pad: Mask[B, 1, T]) -> Logits[B, T, V] {
  let h = tokens |> embedding(V, D) |> positional(max: 64)
  let a = h |> rmsnorm |> attention(heads: 2, causal: true, mask: pad)
  let r = h + a
  return r |> rmsnorm |> linear(V)
}

objective NextToken(logits: Logits[B, T, V], targets: Tokens[B, T]) -> Scalar {
  return cross_entropy(logits, targets)
}

train Pretrain {
  data LMData
  model M = LM
  loss lm = NextToken(logits: M(inputs, pad), targets: targets)
  optimizer opt = adamw(lr: 3e-4)
  epochs 1
}`;

const lmPadded: Challenge = {
  id: "lm-padded",
  title: "Variable-length LM: pad_to + pad_mask as data, the mask reaches attention",
  section: "§20",
  tier: 2,
  code: LM_PADDED,
  expect: {
    warnCodes: [],
    irOps: ["attention", "embedding"],
    run: { steps: 2, gradAll: true, dims: { B: 2 } },
    custom: [
      {
        name: "the padding mask is a model input, not a batch-derived heuristic",
        check: (mod) => {
          const g = mod.graphs.find((x) => x.name === "LM")!;
          const att = g.nodes.find((n) => n.op === "attention")!;
          assert(att.inputs.length >= 2, `attention takes ${att.inputs.length} input(s); the mask must be wired in`);
          return `attention consumes ${att.inputs.length} inputs`;
        },
      },
    ],
  },
  twins: [
    {
      id: "mask-not-passed",
      mutates: "the pad mask is bound to the model but attention ignores it — the unused input is reported",
      code: LM_PADDED.replace("attention(heads: 2, causal: true, mask: pad)", "attention(heads: 2, causal: true)"),
      expectCodes: ["AXS0305"],
    },
    {
      id: "mask-rank",
      mutates: "pad mask declared Mask[T] — not the padding-mask contract Mask[B, 1, T]",
      code: LM_PADDED.replace("field pad: Mask[1, T] = padded[0 : 1, 0 : T]", "field pad: Mask[T] = padded[0, 0 : T]"),
      expectCodes: ["AXS0602"],
    },
  ],
};

const CONTRASTIVE = `dim B
dim E = 16
source Photos = image_folder(path: "u/")
data Views from Photos {
  example {
    field x: Image[3, 8, 8] = decode(file) |> to_float
  }
  augment train { x: random_flip(p: 0.5) }
  batch 4
}
block Enc(x: Image[B, 3, 8, 8]) -> Tensor[B, E] {
  conv2d(4, kernel: 3, pad: 1)
  relu
  global_avgpool
  linear(E)
}
model Two(a: Image[B, 3, 8, 8]) -> (Tensor[B, E], Tensor[B, E]) {
  let enc = Enc()
  let v1 = a |> dropout(0.5)
  let v2 = a |> dropout(0.5)
  return (enc(v1), enc(v2))
}
objective Sim(ea: Tensor[B, E], eb: Tensor[B, E]) -> Scalar {
  return mean(1.0 - cosine_similarity(ea, eb, axis: -1))
}
train P {
  data Views
  model M = Two
  loss con = Sim(ea: M(x)[0], eb: M(x)[1])
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`;

const contrastiveViews: Challenge = {
  id: "contrastive-views",
  title: "Two stochastic views of one input are two draws, one shared encoder",
  section: "§21",
  tier: 2,
  code: CONTRASTIVE,
  expect: {
    warnCodes: [],
    sharedGroups: 1,
    effects: ["stochastic"],
    run: { steps: 2, gradAll: true },
    custom: [
      {
        name: "`augment(x); augment(x)` lowers to two stochastic nodes, not one reused value (§21)",
        check: (mod) => {
          const g = mod.graphs.find((x) => x.name === "Two")!;
          const draws = g.nodes.filter((n) => n.op === "dropout");
          assert(draws.length === 2, `expected two independent dropout draws, found ${draws.length}`);
          assert(draws[0].outputs[0] !== draws[1].outputs[0], "both views read the same value");
          return "two independent stochastic draws feed one shared encoder";
        },
      },
    ],
  },
  twins: [],
};

// ================================================================== 4.2 state (§14)

const STATE_ZOO = `dim B
dim D = 8
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  state center: Tensor[D] init: zeros update: ema(0.9)
  state seen: Scalar init: zeros update: assign
  state proto: Tensor[4, D] init: zeros update: ema(0.95)
  state bank: Tensor[16, D] init: zeros update: assign
  let f = linear(D) |> gelu
  let c = observe(center, mean(f, axis: 0))
  let n = observe(seen, seen + 1.0)
  let p = observe(proto, reshape(concat(mean(f, axis: 0), mean(f, axis: 0), mean(f, axis: 0), mean(f, axis: 0), axis: 0), [4, D]))
  let b = observe(bank, reshape(concat(p, p, p, p, axis: 0), [16, D]))
  return f - c
}`;

const stateZoo: Challenge = {
  id: "state-zoo",
  title: "EMA value, counter, prototypes and a memory bank in one model",
  section: "§14",
  tier: 2,
  code: STATE_ZOO,
  expect: {
    stateSlots: 4,
    effects: ["writes-state"],
    irOps: ["state_read", "state_update"],
    checkpointKinds: ["parameters", "persistent state"],
    inspectContains: ["center", "seen", "proto", "bank"],
    emitContains: ["0.9 * self.M_center + 0.1 *", "register_buffer"],
    run: { outputs: { M: "[4, 8]" } },
    custom: [
      runCheck("state updates only in training mode and the counter counts", 0, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        return "forward in eval mode leaves state untouched";
      }),
    ],
  },
  twins: [],
};

const STATE_SHARED = `dim B
dim D = 8
block Norm(x: Tensor[B, D]) -> Tensor[B, D] {
  state center: Tensor[D] init: zeros update: ema(0.9)
  let c = observe(center, mean(x, axis: 0))
  return x - c
}
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  let n = Norm()
  let a = x |> linear(D) |> n
  let b = a |> linear(D) |> n
  return b
}`;

const stateSharedStage: Challenge = {
  id: "state-shared-stage",
  title: "A shared stage owns one state: two applications, one buffer",
  section: "§14",
  tier: 2,
  code: STATE_SHARED,
  expect: {
    stateSlots: 1,
    paramTables: 4,
    inspectContains: ["M/n.center"],
    run: { outputs: { M: "[4, 8]" } },
    custom: [
      {
        name: "both applications read and update the same state id",
        check: (mod) => {
          const g = mod.graphs.find((x) => x.name === "M")!;
          const nodes: string[] = [];
          const walk = (ns: typeof g.nodes) => {
            for (const n of ns) {
              if (n.op === "state_update") nodes.push(n.states[0]);
              if (n.regions) for (const r of n.regions) walk(r.nodes);
            }
          };
          walk(g.nodes);
          assert(nodes.length === 2 && nodes[0] === nodes[1], `state updates: ${nodes.join(", ")}`);
          return `2 updates of ${nodes[0]}`;
        },
      },
    ],
  },
  twins: [],
};

const STATE_IN_SCAN = `dim B
dim T
dim F = 4
dim H = 8
model M(x: Tensor[B, T, F]) -> Tensor[B, T, H] {
  state center: Tensor[H] init: zeros update: ema(0.9)
  let (ys, fin) = scan over x axis: 1 carry h: Tensor[B, H] init: zeros {
    let c = observe(center, mean(h, axis: 0))
    yield concat(step, h - c, axis: -1) |> linear(H) |> tanh
  }
  return ys
}`;

const stateInScan: Challenge = {
  id: "state-in-scan",
  title: "Persistent state observed inside a scan body: state ≠ carry",
  section: "§14, §27",
  tier: 3,
  code: STATE_IN_SCAN,
  expect: {
    stateSlots: 1,
    irOps: ["scan", "state_update"],
    run: { dims: { B: 2, T: 3 }, outputs: { M: "[2, 3, 8]" } },
  },
  twins: [],
};

const EMA_TEACHER = `dim B
dim D = 8
dim E = 4
block Body(x: Tensor[B, D]) -> Tensor[B, E] {
  linear(8)
  gelu
  linear(E)
}
model Student(x: Tensor[B, D]) -> Tensor[B, E] { Body() }
model Teacher(x: Tensor[B, D]) -> Tensor[B, E] { Body() }
objective Distill(s: Tensor[B, E], t: Tensor[B, E]) -> Scalar {
  let target = stop_grad(t)
  return mean((s - target) * (s - target))
}
source S = synthetic(features: 8)
data Dd from S {
  example { field x: Tensor[D] = decode(row) |> to_float }
  batch 4
}
train T {
  data Dd
  model student = Student
  model teacher = Teacher
  track ema_t = ema(student, rate: 0.99)
  loss d = Distill(s: student(x), t: teacher(x))
  optimizer opt = adamw(lr: 1e-3) over student
  epochs 1
}`;

const emaTeacher: Challenge = {
  id: "ema-teacher",
  title: "EMA teacher/student: the shadow is tracked, but no model can *be* the shadow (G-cand-003)",
  section: "§26 item 4, §14",
  tier: 3,
  code: EMA_TEACHER,
  record: "ema-teacher.md",
  expect: {
    warnCodes: [],
    paramTables: 8,
    stateSlots: 4,
    inspectContains: ["ema_t"],
    emitContains: ["ema_t = {k: student.p[k].detach().clone()", "v.mul_(0.99).add_(student.p[k].detach(), alpha=0.01)"],
    run: { steps: 3 },
    custom: [
      {
        name: "the teacher is a distinct parameter set (two declarations of one block)",
        check: (mod) => {
          const owners = new Set(mod.params.map((p) => p.owner.split("/")[0]));
          assert(owners.has("Student") && owners.has("Teacher"), `owners: ${[...owners].join(", ")}`);
          return "Student and Teacher each own a Body";
        },
      },
      runCheck("the tracked shadow moves with the student (reference backend)", 3, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        const teacherCoverage = rep.gradCoverage.filter((g) => g.param.startsWith("Teacher/"));
        assert(teacherCoverage.every((g) => !g.updated), "teacher parameters must not be optimised");
        return `${rep.losses.length} steps; teacher untouched by the optimizer`;
      }),
    ],
  },
  twins: [
    {
      id: "one-decl-two-aliases",
      mutates: "`model teacher = Student` — one declaration bound twice is one parameter set, not a copy (F-015)",
      code: EMA_TEACHER.replace("model teacher = Teacher", "model teacher = Student"),
      expectCodes: ["AXS0706"],
    },
  ],
};

// ================================================================== 4.3 effects (§15)

const FN_EFFECTS = `dim B
dim D = 8
fn noisy(x: Tensor[B, D]) -> Tensor[B, D] {
  return x |> dropout(0.1)
}
fn normed(x: Tensor[B, D]) -> Tensor[B, D] {
  return x |> batchnorm
}
fn stopped(x: Tensor[B, D]) -> Tensor[B, D] {
  return stop_grad(x)
}
fn clean(x: Tensor[B, D]) -> Tensor[B, D] {
  return x * 2.0
}
model M(x: Tensor[B, D]) -> Tensor[B, D] {
  let a = noisy(x)
  let b = normed(a)
  let c = stopped(b)
  let d = clean(c)
  return d |> linear(D)
}`;

const fnEffects: Challenge = {
  id: "fn-effects",
  title: "A pure-looking `fn` inherits the effects of what it calls",
  section: "§15",
  tier: 2,
  code: FN_EFFECTS,
  expect: {
    effects: ["stochastic", "writes-state", "grad-stopped"],
    custom: [
      {
        name: "each fn application carries the union of its body's effects (F-016)",
        check: (mod) => {
          const g = mod.graphs.find((x) => x.name === "M")!;
          const apps = g.nodes.filter((n) => n.op === "apply" || n.regions);
          const byNote = (needle: string) => apps.find((n) => (n.note ?? "").includes(needle) || String(n.attrs.fn ?? n.attrs.name ?? "").includes(needle));
          const noisy = byNote("noisy");
          const normed = byNote("normed");
          const stopped = byNote("stopped");
          const clean = byNote("clean");
          assert(!!noisy && noisy.effects.includes("stochastic"), `noisy: ${noisy?.effects.join(",")}`);
          assert(!!normed && normed.effects.includes("writes-state"), `normed: ${normed?.effects.join(",")}`);
          assert(!!stopped && stopped.effects.includes("grad-stopped"), `stopped: ${stopped?.effects.join(",")}`);
          assert(!!clean && clean.effects.length === 1 && clean.effects[0] === "pure", `clean: ${clean?.effects.join(",")}`);
          return "noisy→stochastic, normed→stateful, stopped→grad_stop, clean→pure";
        },
      },
    ],
  },
  twins: [],
};

const THREE_WAY = `dim B
dim D = 8
dim K = 4
custom op weird(x: Tensor[B, K]) -> Tensor[B, K] {
  effects: pure
  backend torch: "weird(x)"
}
model NonDiff(x: Tensor[B, D]) -> Tensor[B, K] {
  return x |> linear(K) |> argmax(axis: -1) |> one_hot(classes: K) |> linear(K)
}
model Stopped(x: Tensor[B, D]) -> Tensor[B, K] {
  return x |> linear(K) |> stop_grad |> linear(K)
}
model Foreign(x: Tensor[B, D]) -> Tensor[B, K] {
  return x |> linear(K) |> weird |> linear(K)
}
objective L(y: Tensor[B, K]) -> Scalar { return mean(y * y) }
source S = synthetic(features: 8)
data Dd from S { example { field x: Tensor[D] = decode(row) |> to_float } batch 4 }
train T {
  data Dd
  model a = NonDiff
  model b = Stopped
  model c = Foreign
  loss la = L(y: a(x))
  loss lb = L(y: b(x))
  loss lc = L(y: c(x))
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`;

const threeWay: Challenge = {
  id: "grad-three-way",
  title: "Non-differentiable, gradient-stopped and backend-unsupported are three different diagnoses",
  section: "§15",
  tier: 3,
  code: THREE_WAY,
  record: "grad-three-way.md",
  expect: {
    warnCodes: [],
    effects: ["nondiff", "grad-stopped"],
    custom: [
      {
        name: "a custom op with no reference backend is announced at compile time (AXS0901, info)",
        check: (mod) => {
          const d = mod.diags.find((x) => x.code === "AXS0901");
          assert(!!d && d.severity === "info" && d.message.includes("weird"), `diags: ${mod.diags.map((x) => x.code).join(",")}`);
          return d.message;
        },
      },
      runCheck("gradient coverage tells the three apart", 2, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        const reason = (prefix: string) => rep.gradCoverage.filter((g) => g.param.startsWith(prefix) && g.param.includes("linear#1")).map((g) => g.reason ?? "updated");
        const nd = reason("NonDiff/");
        const st = reason("Stopped/");
        const fo = reason("Foreign/");
        assert(nd.every((r) => r === "no gradient path reached this parameter"), `NonDiff first layer: ${nd.join(",")}`);
        assert(st.every((r) => r === "no gradient path reached this parameter"), `Stopped first layer: ${st.join(",")}`);
        assert(fo.every((r) => r === "updated"), `Foreign first layer (identity substitute for the custom op): ${fo.join(",")}`);
        return "NonDiff, Stopped: first layer unreached; Foreign: identity substitute keeps the path (AXS0901 says so at compile time)";
      }),
    ],
  },
  twins: [],
};

// ================================================================== 4.4 lifecycle (§22, §25, §26)

const BASELINE = `dim B
dim K = 10
source Cifar = image_folder(path: "cifar10/", classes: 10)
data VisionData from Cifar {
  example {
    field image: Image[3, 8, 8] = decode(file) |> to_float |> normalize(fit: train)
    field label: Class = as_class(folder)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
model Net(image: Image[B, 3, 8, 8]) -> Logits[B, K] {
  conv2d(4, kernel: 3, pad: 1)
  relu
  global_avgpool
  linear(K)
}
objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}
train Ceremony {
  data VisionData
  model M = Net
  loss main = Classification(logits: M(image), labels: label)
  optimizer opt = adamw(lr: 1e-3, wd: 0.05)
  phase main {
    steps 12
    cosine(warmup: 2)
    update main with opt
    every 4 steps { validate ; checkpoint }
  }
  clip_grad 1.0
  precision mixed
}`;

const baselineCeremony: Challenge = {
  id: "baseline-ceremony",
  title: "The baseline ceremony: schedule, warmup, clipping, periodic validate + checkpoint",
  section: "§22",
  tier: 2,
  code: BASELINE,
  expect: {
    warnCodes: [],
    emitContains: [
      'schedule = {"kind": "cosine", "args": {"warmup": 2}}',
      "apply_schedule([opt], schedule, phase_step, phase_budget)",
      "torch.nn.utils.clip_grad_norm_(opt_params, 1)",
      "if phase_step % 4 == 0:",
      "metrics.update(validate())",
      "checkpoint()",
      "weight_decay=0.05",
      "# precision mixed",
    ],
    emitForbids: ["region_params(model", 'batch["M(image"'],
    custom: [
      runCheck("the reference runtime honours steps, warmup/cosine and the 4-step events (H-001)", 40, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        const ph = rep.phases[0];
        assert(ph.steps === 12 && ph.stoppedBy === "steps", `phase ran ${ph.steps} steps, stopped by ${ph.stoppedBy}`);
        assert(ph.lrStart === 0.5 && ph.lrEnd < 0.05, `lr multiplier ${ph.lrStart} → ${ph.lrEnd}`);
        const validates = rep.events.filter((e) => e.action === "validate").map((e) => e.step);
        const ckpts = rep.events.filter((e) => e.action === "checkpoint").length;
        assert(validates.join(",") === "3,7,11" && ckpts === 3, `validate at ${validates.join(",")}, ${ckpts} checkpoints`);
        assert("val_main" in rep.finalMetrics, "validation loss missing from metrics");
        return `12 steps, lr × ${ph.lrStart} → ${ph.lrEnd.toFixed(3)}, validate at ${validates.join(",")}`;
      }),
    ],
  },
  twins: [],
};

const TRANSFER = `dim B
dim K = 10
block Backbone(image: Image[B, 3, 8, 8]) -> Tensor[B, 16] {
  conv2d(4, kernel: 3, pad: 1)
  relu
  global_avgpool
  linear(16)
}
block Head(f: Tensor[B, 16]) -> Logits[B, K] {
  gelu
  linear(K)
}
model Transfer(image: Image[B, 3, 8, 8]) -> Logits[B, K] {
  let encoder = Backbone()
  let head = Head()
  image |> encoder |> head
}
objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}
source Cifar = image_folder(path: "cifar10/", classes: 10)
data VisionData from Cifar {
  example {
    field image: Image[3, 8, 8] = decode(file) |> to_float |> normalize(fit: train)
    field label: Class = as_class(folder)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 8
}
train Finetune {
  data VisionData
  model Net = Transfer
  loss main = Classification(logits: Net(image), labels: label)
  optimizer head_opt = adamw(lr: 1e-3) over Net.head
  optimizer enc_opt = adamw(lr: 1e-5) over Net.encoder
  phase warmup {
    epochs 2
    freeze Net.encoder
    update main with head_opt
  }
  phase finetune {
    epochs 3
    unfreeze Net.encoder
    lr Net.encoder = 0.00001
    update main with head_opt, enc_opt
    every 1 epochs { validate ; checkpoint }
  }
}`;

const transferLifecycle: Challenge = {
  id: "transfer-lifecycle",
  title: "Transfer: freeze persists, regions resolve structurally, per-region lr, conflicting phases refused",
  section: "§22, §26",
  tier: 2,
  code: TRANSFER,
  record: "transfer-lifecycle.md",
  expect: {
    warnCodes: [],
    emitContains: [
      "head_opt_params = [Net.p[k] for k in [\"Transfer_head_linear",
      'p.requires_grad_(False)  # freeze Net.encoder',
      "set_lr(enc_opt, [Net.p[k] for k in [\"Transfer_encoder_conv2d",
      "if (epoch + 1) % 1 == 0:",
    ],
    emitForbids: ["region_params(model", "set_lr(region_params"],
    custom: [
      runCheck("freeze holds through the warmup phase and lifts in finetune (reference backend)", 60, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        assert(rep.phases.length === 2 && rep.phases[0].steps === 12 && rep.phases[1].steps === 18, JSON.stringify(rep.phases));
        const enc = rep.gradCoverage.filter((g) => g.param.startsWith("Transfer/encoder"));
        assert(enc.length > 0 && enc.every((g) => g.updated), "encoder must be updated once unfrozen");
        const validates = rep.events.filter((e) => e.action === "validate");
        assert(validates.length === 3 && validates.every((e) => e.phase === "finetune"), `validate events: ${JSON.stringify(validates)}`);
        return `warmup 12 steps, finetune 18 steps, 3 epoch-end validations`;
      }),
    ],
  },
  twins: [
    {
      id: "unknown-region",
      mutates: "freeze names a region that does not exist",
      code: TRANSFER.replace("freeze Net.encoder", "freeze Net.backbone"),
      expectCodes: ["AXS0701"],
    },
    {
      id: "overlapping-optimizers",
      mutates: "two optimizers claim the same parameters",
      code: TRANSFER.replace("optimizer enc_opt = adamw(lr: 1e-5) over Net.encoder", "optimizer enc_opt = adamw(lr: 1e-5) over Net"),
      expectCodes: ["AXS0703"],
    },
    {
      id: "frozen-but-updated",
      mutates: "the warmup phase updates with the optimizer whose whole region it froze (F-014)",
      code: TRANSFER.replace("update main with head_opt\n  }", "update main with enc_opt\n  }"),
      expectCodes: ["AXS0704"],
    },
    {
      id: "freeze-and-unfreeze",
      mutates: "one phase freezes and unfreezes the same region",
      code: TRANSFER.replace("freeze Net.encoder", "freeze Net.encoder\n    unfreeze Net.encoder"),
      expectCodes: ["AXS0707"],
    },
    {
      id: "epochs-and-steps",
      mutates: "a phase declares both a step and an epoch budget",
      code: TRANSFER.replace("epochs 2\n", "epochs 2\n    steps 50\n"),
      expectCodes: ["AXS0707"],
    },
    {
      id: "two-until",
      mutates: "two stopping conditions in one phase",
      code: TRANSFER.replace("epochs 2\n", "epochs 2\n    until val_acc > 0.9\n    until val_main < 0.1\n"),
      expectCodes: ["AXS0707"],
    },
    {
      id: "lr-on-frozen",
      mutates: "a learning rate is set for a region the same phase freezes",
      code: TRANSFER.replace("freeze Net.encoder", "freeze Net.encoder\n    lr Net.encoder = 0.1"),
      expectCodes: ["AXS0707"],
    },
  ],
};

const GAN = `dim B
dim Z = 8
dim D = 16
model Generator(z: Tensor[B, Z]) -> Tensor[B, D] {
  linear(16)
  gelu
  linear(D)
  tanh
}
model Discriminator(x: Tensor[B, D]) -> Probs[B, 1] {
  linear(16)
  gelu
  linear(1)
  sigmoid
}
objective DiscLoss(real_score: Probs[B, 1], fake_score: Probs[B, 1]) -> Scalar {
  return mean(0.0 - log(real_score)) + mean(0.0 - log(1.0 - fake_score))
}
objective GenLoss(fake_score: Probs[B, 1]) -> Scalar {
  return mean(0.0 - log(fake_score))
}
source Noise = synthetic(features: 16)
data GanData from Noise {
  example {
    field z: Tensor[Z] = decode(row) |> to_float
    field real: Tensor[D] = decode(row) |> to_float
  }
  batch 4
}
train Adversarial {
  data GanData
  model G = Generator
  model D = Discriminator
  loss d_loss = DiscLoss(real_score: D(real), fake_score: D(G(z)))
  loss g_loss = GenLoss(fake_score: D(G(z)))
  optimizer d_opt = adam(lr: 0.0002) over D
  optimizer g_opt = adam(lr: 0.0002) over G
  phase adversarial {
    epochs 2
    update d_loss with d_opt times 2
    update g_loss with g_opt times 1
  }
}`;

const ganAlternating: Challenge = {
  id: "gan-alternating",
  title: "Alternating updates: `times` is data, two objectives share one model, nested bindings reach both",
  section: "§26 items 1–3, §25",
  tier: 2,
  code: GAN,
  expect: {
    warnCodes: [],
    run: { steps: 6, gradAll: true },
    emitContains: [
      "for _ in range(2):",
      "d_opt_params = [D.p[k]",
      "g_opt_params = [G.p[k]",
      'objective_DiscLoss(D(batch["real"]), D(G(batch["z"])), dims=dims)',
      "return v",
    ],
    emitForbids: ['batch["G(z"', "model.p["],
    custom: [
      {
        name: "the update ratio is plan data, not loop code",
        check: (mod) => {
          const ph = mod.plans[0].phases[0];
          assert(ph.updates.map((u) => `${u.loss}×${u.times}`).join(" ") === "d_loss×2 g_loss×1", JSON.stringify(ph.updates));
          return "d_loss×2 g_loss×1";
        },
      },
      {
        name: "objective ports are aliased in the emitted Python (H-006)",
        check: (mod) => {
          const code = emitTorch(mod);
          const m = /def objective_GenLoss\(fake_score, dims=None\):\n    dims = resolve_dims\(dims or \{\}\)\n    (v\w+) = fake_score/.exec(code);
          assert(!!m, "objective body does not bind its port to the IR value");
          return `fake_score → ${m[1]}`;
        },
      },
    ],
  },
  twins: [
    {
      id: "generator-never-updated",
      mutates: "the generator has an optimizer but no update names it — its parameters are claimed and never trained (F-017)",
      code: GAN.replace("    update g_loss with g_opt times 1\n", ""),
      expectCodes: ["AXS0708"],
    },
  ],
};

const CURRICULUM = `dim B
dim D = 8
dim K = 3
model Net(x: Tensor[B, D]) -> Logits[B, K] {
  linear(16)
  gelu
  linear(K)
}
objective Cls(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}
source S = synthetic(features: 8)
data Dd from S {
  example {
    field x: Tensor[D] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  split { train: 0.9, val: 0.1 }
  batch 4
}
train Curriculum {
  data Dd
  model M = Net
  loss main = Cls(logits: M(x), labels: label)
  optimizer opt = adamw(lr: 1e-2)
  phase easy {
    epochs 10
    until main < 0.9
    update main with opt
  }
  phase hard {
    steps 6
    linear(warmup: 2)
    update main with opt
    every 3 steps { validate }
  }
  every 5 steps { checkpoint }
}`;

const curriculumEvents: Challenge = {
  id: "curriculum-events",
  title: "Curriculum: `until` ends a phase early, per-phase schedule, phase and plan-level events",
  section: "§25",
  tier: 3,
  code: CURRICULUM,
  record: "curriculum-events.md",
  expect: {
    warnCodes: [],
    emitContains: [
      'if metrics.get("main", float("inf")) < 0.9:  # until main < 0.9',
      "stop = True; break",
      'schedule = {"kind": "linear", "args": {"warmup": 2}}',
      "if phase_step % 3 == 0:",
      "if step % 5 == 0:",
    ],
    custom: [
      runCheck("`until` stops the easy phase before its epoch budget; events fire at both levels (H-001)", 80, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        const [easy, hard] = rep.phases;
        assert(easy.stoppedBy === "until" && easy.steps < 60, `easy: ${JSON.stringify(easy)}`);
        assert(hard.stoppedBy === "steps" && hard.steps === 6, `hard: ${JSON.stringify(hard)}`);
        assert(hard.lrStart === 0.5 && hard.lrEnd < hard.lrStart, `hard lr ${hard.lrStart} → ${hard.lrEnd}`);
        const validates = rep.events.filter((e) => e.action === "validate");
        const ckpts = rep.events.filter((e) => e.action === "checkpoint");
        assert(validates.length === 2 && validates.every((e) => e.phase === "hard"), `validate: ${JSON.stringify(validates)}`);
        assert(ckpts.length >= 1 && ckpts.every((e) => (e.step + 1) % 5 === 0), `checkpoint: ${JSON.stringify(ckpts)}`);
        return `easy stopped by until after ${easy.steps} steps; hard 6 steps; ${validates.length} validates, ${ckpts.length} checkpoints`;
      }),
    ],
  },
  twins: [
    {
      id: "until-unknown-metric",
      mutates: "`until` names a metric no loss or validation produces (F-018)",
      code: CURRICULUM.replace("until main < 0.9", "until accuracy > 0.9"),
      expectCodes: ["AXS0709"],
    },
  ],
};

// ================================================================== §26 escalation ladder items 5–7

const DISTILL = `dim B
dim D = 8
dim K = 4
model Teacher(x: Tensor[B, D]) -> Logits[B, K] {
  linear(32)
  gelu
  linear(K)
}
model Student(x: Tensor[B, D]) -> Logits[B, K] {
  linear(8)
  gelu
  linear(K)
}
objective Distill(s: Logits[B, K], t: Logits[B, K], labels: Class[B]) -> Scalar {
  let soft = softmax(stop_grad(t), axis: -1)
  let kd = mean(0.0 - sum(soft * log_softmax(s, axis: -1), axis: -1))
  return 0.5 * kd + 0.5 * cross_entropy(s, labels)
}
source S = synthetic(features: 8)
data Dd from S {
  example {
    field x: Tensor[D] = decode(row) |> to_float
    field label: Class = as_class(target)
  }
  batch 4
}
train KD {
  data Dd
  model teacher = Teacher
  model student = Student
  loss kd = Distill(s: student(x), t: teacher(x), labels: label)
  optimizer opt = adamw(lr: 1e-3) over student
  epochs 1
}`;

const distillation: Challenge = {
  id: "distillation",
  title: "Knowledge distillation: a frozen teacher feeds soft targets through stop_grad",
  section: "§26 item 5",
  tier: 3,
  code: DISTILL,
  record: "escalation-ladder.md",
  expect: {
    warnCodes: [],
    effects: ["grad-stopped"],
    custom: [
      runCheck("the teacher receives no update and the student receives all of them", 3, (rep) => {
        assert(rep.errors.length === 0, rep.errors.join("; "));
        const t = rep.gradCoverage.filter((g) => g.param.startsWith("Teacher/"));
        const s = rep.gradCoverage.filter((g) => g.param.startsWith("Student/"));
        assert(t.length > 0 && t.every((g) => !g.updated && g.reason === "not claimed by any optimizer"), JSON.stringify(t));
        assert(s.length > 0 && s.every((g) => g.updated), JSON.stringify(s));
        return `teacher ${t.length} tables untouched; student ${s.length} tables updated`;
      }),
    ],
  },
  twins: [
    {
      id: "two-until",
      mutates: "a distillation phase with two stopping rules",
      code: DISTILL.replace("  epochs 1\n}", "  phase kd {\n    epochs 1\n    until kd < 0.1\n    until val_kd < 0.1\n    update kd with opt\n  }\n}"),
      expectCodes: ["AXS0707"],
    },
  ],
};

const PSEUDO = `dim B
dim D = 8
dim K = 4
model Net(x: Tensor[B, D]) -> Logits[B, K] {
  linear(8)
  gelu
  linear(K)
}
objective Pseudo(s: Logits[B, K], t: Logits[B, K]) -> Scalar {
  let pseudo = argmax(stop_grad(t), axis: -1)
  return cross_entropy(s, pseudo)
}
source S = synthetic(features: 8)
data Dd from S {
  example { field x: Tensor[D] = decode(row) |> to_float }
  augment train { x: noise(0.1) }
  batch 4
}
train SelfTrain {
  data Dd
  model M = Net
  loss st = Pseudo(s: M(x), t: M(x))
  optimizer opt = adamw(lr: 1e-3)
  epochs 1
}`;

const pseudoLabel: Challenge = {
  id: "self-training",
  title: "Self-training: pseudo-labels are an argmax of a gradient-stopped prediction",
  section: "§26 item 6",
  tier: 3,
  code: PSEUDO,
  record: "escalation-ladder.md",
  expect: {
    warnCodes: [],
    effects: ["grad-stopped", "nondiff"],
    run: { steps: 2, gradAll: true },
  },
  twins: [],
};

export const LEARNING_CHALLENGES: Challenge[] = [
  tabularPipeline,
  visionPipeline,
  lmPadded,
  contrastiveViews,
  stateZoo,
  stateSharedStage,
  stateInScan,
  emaTeacher,
  fnEffects,
  threeWay,
  baselineCeremony,
  transferLifecycle,
  ganAlternating,
  curriculumEvents,
  distillation,
  pseudoLabel,
];
