/**
 * TENSA — example suite.  Every example is compiled by the test runner.
 */
import type { FindingKind } from "./docs";

export interface Example {
  id: string;
  title: string;
  group: string;
  summary: string;
  code: string;
  /** semantic-hardening classification of the residual friction */
  friction?: { kind: FindingKind; note: string }[];
}

export const EXAMPLES: Example[] = [
  {
    id: "mlp",
    title: "MLP classifier",
    group: "Architecture",
    summary: "Implicit sequential flow, inferred input widths, objective as a typed graph, derived training loop.",
    code: `# The whole program: shapes, model, objective, data and lifecycle.
dim B                      # free at compile time, bound at runtime (batch)
dim K = 10

model MLP(x: Tensor[B, 784]) -> Logits[B, K] {
  linear(256)              # the input width 784 is a consequence, not a choice
  gelu
  dropout(0.1)
  linear(128)
  gelu
  linear(K)
}

objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}

source Digits = synthetic(features: 784, classes: 10)

data DigitData from Digits {
  example {
    field x: Tensor[784] = decode(row) |> to_float |> standardize(fit: train)
    field label: Class = as_class(target)
  }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 32
}

train DigitRun {
  data DigitData
  model Net = MLP
  loss main = Classification(logits: Net(x), labels: label)
  optimizer opt = adamw(lr: 3e-4, wd: 0.01)
  epochs 3
  every 1 epochs { validate ; checkpoint }
}`,
    friction: [
      { kind: "B", note: "Widths 256/128/10 are genuine architectural choices and must be written." },
      { kind: "A", note: "Zero-grad / forward / backward / step / eval-mode switching are removed entirely." },
    ],
  },
  {
    id: "cnn",
    title: "CNN classifier",
    group: "Architecture",
    summary: "Channel counts are choices; spatial shapes are computed symbolically through conv and pool.",
    code: `dim B
dim K = 10

model SmallCNN(image: Image[B, 3, 32, 32]) -> Logits[B, K] {
  conv2d(32, kernel: 3, pad: 1)
  batchnorm                 # owns running statistics: persistent state, train/eval sensitive
  relu
  maxpool2d(2)
  conv2d(64, kernel: 3, pad: 1)
  batchnorm
  relu
  maxpool2d(2)
  conv2d(128, kernel: 3, pad: 1)
  relu
  global_avgpool            # [B, 128, 8, 8] -> [B, 128]
  linear(K)
}

objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}`,
    friction: [
      { kind: "B", note: "Kernel/stride/padding are mathematical choices." },
      { kind: "A", note: "Flattened feature counts never appear; the compiler derives them." },
    ],
  },
  {
    id: "resnet",
    title: "Residual CNN",
    group: "Topology",
    summary: "Residual topology with an explicit projection. The compiler never invents a learned projection.",
    code: `dim B

block ResBlock(x: Tensor[B, C, H, W], out: Dim = 64, stride: Dim = 1) {
  residual via conv2d(out, kernel: 1, stride: stride) {
    conv2d(out, kernel: 3, stride: stride, pad: 1)
    batchnorm
    relu
    conv2d(out, kernel: 3, pad: 1)
    batchnorm
  }
  relu
}

model ResNet(image: Image[B, 3, 32, 32]) -> Logits[B, 10] {
  conv2d(32, kernel: 3, pad: 1)
  batchnorm
  relu
  ResBlock(out: 32)
  ResBlock(out: 64, stride: 2)
  ResBlock(out: 64)
  global_avgpool
  linear(10)
}`,
    friction: [
      { kind: "B", note: "The 1x1 projection is information: it says *how* the skip path changes shape." },
      { kind: "A", note: "No `nn.Sequential`, no `downsample=None` plumbing, no manual channel bookkeeping." },
    ],
  },
  {
    id: "autoencoder",
    title: "Autoencoder",
    group: "Architecture",
    summary: "Named bindings where identity matters; reconstruction objective over the original input.",
    code: `dim B
dim D = 784
dim Code = 32

model AutoEncoder(x: Tensor[B, D]) -> Tensor[B, D] {
  let code = linear(256) |> gelu |> linear(Code)
  let out = code |> linear(256) |> gelu |> linear(D)
  return out
}

objective Reconstruction(pred: Tensor[B, D], original: Tensor[B, D]) -> Scalar {
  return mse(pred, original)
}

source Vectors = synthetic(features: 784)

data VecData from Vectors {
  example {
    field x: Tensor[D] = decode(row) |> to_float |> standardize(fit: train)
  }
  split { train: 0.9, val: 0.1 }
  batch 32
}

train AE {
  data VecData
  model Net = AutoEncoder
  loss rec = Reconstruction(pred: Net(x), original: x)
  optimizer opt = adamw(lr: 1e-3)
  epochs 2
}`,
    friction: [{ kind: "B", note: "The objective must say that the target *is* the input — that is learning-problem information." }],
  },
  {
    id: "inception",
    title: "Multi-scale branching",
    group: "Topology",
    summary: "Parallel split with an explicit merge. Every branch receives the same incoming tensor.",
    code: `dim B

model MultiScale(image: Image[B, 3, 32, 32]) -> Logits[B, 10] {
  conv2d(16, kernel: 3, pad: 1)
  relu

  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3, pad: 1)
    conv2d(32, kernel: 5, pad: 2)
    branch pooled {                 # multi-statement branch body
      maxpool2d(3, stride: 1, pad: 1)
      conv2d(32, kernel: 1)
    }
  }

  batchnorm
  relu
  global_avgpool
  linear(10)
}`,
    friction: [
      { kind: "B", note: "Which scales to use, and how to merge them, is architecture." },
      { kind: "A", note: "No branch variables, no manual torch.cat bookkeeping, no accidental sequential composition." },
    ],
  },
  {
    id: "siamese",
    title: "Siamese network (tied encoder)",
    group: "Parameter identity",
    summary: "Two applications of one bound stage share exactly one parameter set — no registration machinery.",
    code: `dim B
dim E = 64

block Encoder(x: Image[B, 1, 28, 28]) -> Tensor[B, E] {
  conv2d(16, kernel: 3, stride: 2, pad: 1)
  relu
  conv2d(32, kernel: 3, stride: 2, pad: 1)
  relu
  global_avgpool
  linear(E)
}

model Siamese(a: Image[B, 1, 28, 28], b: Image[B, 1, 28, 28]) -> (Tensor[B, E], Tensor[B, E]) {
  let encoder = Encoder()        # one configured stage value
  return (encoder(a), encoder(b))
}

# Recreating the stage would create independent parameters:
model TwoTowers(a: Image[B, 1, 28, 28], b: Image[B, 1, 28, 28]) -> (Tensor[B, E], Tensor[B, E]) {
  return (Encoder(a), Encoder(b))
}

objective Contrastive(ea: Tensor[B, E], eb: Tensor[B, E], same: Class[B]) -> Scalar {
  let sim = cosine_similarity(ea, eb, axis: -1)
  let diff = sim - same
  return mean(diff * diff)
}`,
    friction: [
      { kind: "B", note: "Sharing or not sharing is a modelling decision; the language makes it a value-identity decision." },
      { kind: "A", note: "ModuleList / register_parameter / shared-name strings all disappear." },
    ],
  },
  {
    id: "transformer",
    title: "Transformer block",
    group: "Architecture",
    summary: "Pre-norm residual blocks with catalog attention; symbolic widths (D*4) throughout.",
    code: `dim B
dim T
dim D = 128
dim Heads = 4

block TransformerBlock(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  residual {
    layernorm
    attention(heads: Heads, causal: true)
    dropout(0.1)
  }
  residual {
    layernorm
    linear(D * 4)
    gelu
    linear(D)
    dropout(0.1)
  }
}

model Encoder4(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  for 4: TransformerBlock()     # 4 independent parameter sets
  layernorm
}

model Weighted(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let shared = TransformerBlock()
  for 4: shared                 # one parameter set, applied 4 times
  layernorm
}`,
    friction: [
      { kind: "B", note: "Pre-norm vs post-norm, causal masking and the 4x MLP ratio are all real choices." },
      { kind: "C", note: "`for 6:` could hide whether parameters are shared — so `inspect` reports the mode explicitly." },
    ],
  },
  {
    id: "gpt",
    title: "Decoder-only language model",
    group: "Architecture",
    summary: "Tokens in, logits out; target shifting is explicit in the data declaration, never inferred from a dataset name.",
    code: `dim B
dim T
dim V = 256
dim D = 64
dim Heads = 4

block Block(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  residual {
    rmsnorm
    attention(heads: Heads, causal: true)
  }
  residual {
    rmsnorm
    linear(D * 4)
    gelu
    linear(D)
  }
}

model TinyLM(tokens: Tokens[B, T]) -> Logits[B, T, V] {
  embedding(V, D)
  positional(max: 256)
  for 2: Block()
  rmsnorm
  linear(V)
}

objective NextToken(logits: Logits[B, T, V], targets: Tokens[B, T]) -> Scalar {
  return cross_entropy(logits, targets)
}

source Corpus = text_file(path: "corpus/*.txt")

data LMData from Corpus {
  example {
    field window: Tokens[T + 1] = tokenize(text, vocab: V) |> window(T + 1, stride: T)
    field inputs: Tokens[T] = window[0 : T]        # shifting is written down …
    field targets: Tokens[T] = window[1 : T + 1]   # … not derived from a dataset name
  }
  split { train: 0.98, val: 0.02 }
  batch 16
}

train Pretrain {
  data LMData
  model LM = TinyLM
  loss lm = NextToken(logits: LM(inputs), targets: targets)
  optimizer opt = adamw(lr: 3e-4, wd: 0.1)
  phase main {
    steps 200
    cosine(warmup: 20)
    update lm with opt
    every 50 steps { validate ; checkpoint }
  }
  precision mixed
  clip_grad 1.0
}`,
    friction: [
      { kind: "B", note: "Sequence windowing and shifting are learning-problem definitions and stay visible." },
      { kind: "D", note: "`precision mixed` is execution policy — annotated, not baked into semantics." },
    ],
  },
  {
    id: "attention-math",
    title: "Tensor-level attention",
    group: "Tensor level",
    summary: "The same architecture, descended into explicit tensor mathematics — same language, same IR.",
    code: `dim B
dim T
dim D = 128
dim Heads = 4
dim DH = D / Heads          # symbolic division, proved exact: 32

fn split_heads(x: Tensor[B, T, D]) -> Tensor[B, Heads, T, DH] {
  return transpose(reshape(x, [B, T, Heads, DH]), 1, 2)
}

fn merge_heads(x: Tensor[B, Heads, T, DH]) -> Tensor[B, T, D] {
  return reshape(transpose(x, 1, 2), [B, T, D])
}

fn attention_core(q: Tensor[B, Heads, T, DH],
                  k: Tensor[B, Heads, T, DH],
                  v: Tensor[B, Heads, T, DH]) -> Tensor[B, Heads, T, DH] {
  let scores = matmul(q, transpose(k, 2, 3)) / sqrt(DH)
  let masked = masked_fill(scores, causal_mask(T), value: -1000000.0)
  let weights = softmax(masked, axis: -1)
  return matmul(weights, v)
}

block ManualAttention(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param wq: Tensor[D, D] init: xavier
  param wk: Tensor[D, D] init: xavier
  param wv: Tensor[D, D] init: xavier
  param wo: Tensor[D, D] init: xavier

  let q = split_heads(matmul(x, wq))
  let k = split_heads(matmul(x, wk))
  let v = split_heads(matmul(x, wv))
  let ctx = attention_core(q, k, v)
  return matmul(merge_heads(ctx), wo)
}

model Manual(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  residual {
    layernorm
    ManualAttention()
  }
  residual {
    layernorm
    linear(D * 4)
    gelu
    linear(D)
  }
}`,
    friction: [
      { kind: "B", note: "This is the mathematics itself — nothing here should be shorter." },
      { kind: "A", note: "`view/permute/contiguous` juggling is replaced by shape-checked reshape/transpose." },
    ],
  },
  {
    id: "rmsnorm-fn",
    title: "Custom normalisation & gating",
    group: "Tensor level",
    summary: "A missing layer is written in the language — no compiler change required.",
    code: `dim B
dim T
dim D = 128

fn my_rmsnorm(x: Tensor[B, T, D], scale: Tensor[D]) -> Tensor[B, T, D] {
  let ms = mean(x * x, axis: -1, keep: true)
  return x * rsqrt(ms + 0.000001) * scale
}

block RMSNorm(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param scale: Tensor[D] init: ones
  return my_rmsnorm(x, scale)
}

block SwiGLU(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let gate = x |> linear(D * 2) |> silu
  let up = x |> linear(D * 2)
  return gate * up |> linear(D)
}

model Gated(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  residual {
    RMSNorm()
    SwiGLU()
  }
}`,
  },
  {
    id: "multi-input",
    title: "Multi-input model (cross attention)",
    group: "Multi-stream",
    summary: "With several inputs there is no implicit stream until the program seeds one.",
    code: `dim B
dim T
dim S
dim D = 128

model Fusion(text: Tensor[B, T, D], image: Tensor[B, S, D]) -> Tensor[B, D] {
  # no bare transformation may appear here: two inputs, no single implicit value
  let t = text |> layernorm
  let i = image |> layernorm
  let fused = attention(query: t, key: i, value: i, heads: 8)
  return mean(fused, axis: 1)
}

model SeqPair(a: Tensor[B, T, D], b: Tensor[B, T, D]) -> Tensor[B, T, D] {
  let joined = concat(a, b, axis: -1)     # explicit multi-stream join
  joined |> linear(D) |> gelu             # from here a single stream exists again
}`,
    friction: [
      { kind: "B", note: "Which stream is query and which is key/value is a modelling decision — it must be written." },
      { kind: "C", note: "Rejected alternative: auto-seeding the cursor with the first input would silently guess." },
    ],
  },
  {
    id: "multi-objective",
    title: "Multi-output / multi-objective",
    group: "Objectives",
    summary: "Tuple results, destructuring, weighted losses and an auxiliary regulariser.",
    code: `dim B
dim K = 10

model MultiTask(x: Tensor[B, 128]) -> (Logits[B, K], Tensor[B, 1]) {
  let h = linear(256) |> gelu |> linear(128) |> gelu
  let cls = h |> linear(K)
  let reg = h |> linear(1)
  return (cls, reg)
}

objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}

objective Value(pred: Tensor[B, 1], target: Tensor[B, 1]) -> Scalar {
  return mse(pred, target)
}

source Tab = csv(path: "table.csv")

data TaskData from Tab {
  example {
    field x: Tensor[128] = select(features) |> impute(strategy: median, fit: train) |> standardize(fit: train)
    field label: Class = as_class(category)
    field value: Tensor[1] = decode(price) |> to_float
  }
  split { train: 0.7, val: 0.15, test: 0.15 }
  batch 32
}

train Joint {
  data TaskData
  model Net = MultiTask
  loss cls = Classification(logits: Net(x)[0], labels: label) weight: 1.0
  loss val = Value(pred: Net(x)[1], target: value) weight: 0.3
  optimizer opt = adamw(lr: 1e-3)
  phase main {
    epochs 3
    update cls with opt
    update val with opt
  }
}`,
    friction: [{ kind: "B", note: "Loss weights are research choices and stay explicit." }],
  },
  {
    id: "explicit-param",
    title: "Explicit learned parameters",
    group: "Parameters",
    summary: "Parameters are mathematics; registration is plumbing. Frozen parameters are declared, not filtered by name.",
    code: `dim B
dim D = 128
dim K = 10

model Calibrated(x: Tensor[B, D]) -> Logits[B, K] {
  param temperature: Scalar init: ones
  frozen param prior: Tensor[K] init: zeros

  let logits = linear(256) |> gelu |> linear(K)
  return logits / temperature + prior
}

objective Classification(logits: Logits[B, K], labels: Class[B]) -> Scalar {
  return cross_entropy(logits, labels)
}`,
  },
  {
    id: "state",
    title: "Persistent non-parameter state",
    group: "State",
    summary: "An EMA feature centre: persistent, checkpointed, updated only in training context.",
    code: `dim B
dim D = 128

model Centered(x: Tensor[B, D]) -> Tensor[B, D] {
  state center: Tensor[D] init: zeros update: ema(0.9)

  let feats = linear(D) |> gelu
  let batch_mean = mean(feats, axis: 0)
  let c = observe(center, batch_mean)   # reads state, schedules its update
  return feats - c
}

model Counted(x: Tensor[B, D]) -> Tensor[B, D] {
  state seen: Scalar init: zeros update: assign
  let n = observe(seen, mean(x))
  return x * 1.0
}`,
    friction: [
      { kind: "B", note: "Whether a statistic is an EMA or an exact batch value is algorithmic content." },
      { kind: "A", note: "No register_buffer, no manual train/eval branching, no checkpoint bookkeeping." },
    ],
  },
  {
    id: "vision-data",
    title: "Vision data semantics",
    group: "Data",
    summary: "Preprocessing that can change what the model learns lives in the program, with leakage checks.",
    code: `dim B
dim K = 10

source Cifar = image_folder(path: "cifar10/", classes: 10)

data VisionData from Cifar {
  example {
    field image: Image[3, 32, 32] = decode(file)
    field label: Class = as_class(folder)
  }

  preprocess {
    image: resize(32, 32) |> to_float |> normalize(fit: train)
  }

  augment train {
    image: random_crop(32, pad: 4) |> random_flip(p: 0.5) |> color_jitter(0.2)
  }

  augment eval {
    image: center_crop(32)
  }

  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 64
  shuffle true
}

model Net(image: Image[B, 3, 32, 32]) -> Logits[B, K] {
  conv2d(32, kernel: 3, pad: 1)
  relu
  maxpool2d(2)
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
}`,
    friction: [
      { kind: "B", note: "Augmentation policy changes what is learned — it belongs to the program." },
      { kind: "A", note: "Split-aware statistics are checked instead of being an unwritten convention." },
    ],
  },
  {
    id: "contrastive-data",
    title: "Two-view contrastive construction",
    group: "Data",
    summary: "Example construction can emit several views of one source record.",
    code: `dim B
dim E = 64

source Photos = image_folder(path: "unlabelled/")

data Views from Photos {
  example {
    field view1: Image[3, 32, 32] = decode(file) |> to_float |> random_crop(32, pad: 4) |> random_flip(p: 0.5)
    field view2: Image[3, 32, 32] = decode(file) |> to_float |> random_crop(32, pad: 4) |> color_jitter(0.4)
    field same: Class = as_class(index)
  }
  split { train: 0.95, val: 0.05 }
  batch 32
}

block Encoder(x: Image[B, 3, 32, 32]) -> Tensor[B, E] {
  conv2d(32, kernel: 3, stride: 2, pad: 1)
  relu
  global_avgpool
  linear(E)
}

model TwoView(a: Image[B, 3, 32, 32], b: Image[B, 3, 32, 32]) -> (Tensor[B, E], Tensor[B, E]) {
  let encoder = Encoder()
  return (encoder(a), encoder(b))
}

objective NTXent(ea: Tensor[B, E], eb: Tensor[B, E]) -> Scalar {
  let sim = cosine_similarity(ea, eb, axis: -1)
  return mean(1.0 - sim)
}

train Pretext {
  data Views
  model Enc = TwoView
  loss con = NTXent(ea: Enc(view1, view2)[0], eb: Enc(view1, view2)[1])
  optimizer opt = adamw(lr: 1e-3)
  epochs 2
}`,
  },
  {
    id: "transfer",
    title: "Transfer learning lifecycle",
    group: "Lifecycle",
    summary: "Phases, freeze/unfreeze, per-region learning rates and structurally resolved parameter regions.",
    code: `dim B
dim K = 10

block Backbone(image: Image[B, 3, 32, 32]) -> Tensor[B, 128] {
  conv2d(32, kernel: 3, pad: 1)
  batchnorm
  relu
  maxpool2d(2)
  conv2d(64, kernel: 3, pad: 1)
  batchnorm
  relu
  global_avgpool
  linear(128)
}

block Head(f: Tensor[B, 128]) -> Logits[B, K] {
  gelu
  linear(K)
}

model Transfer(image: Image[B, 3, 32, 32]) -> Logits[B, K] {
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
    field image: Image[3, 32, 32] = decode(file) |> to_float |> normalize(fit: train)
    field label: Class = as_class(folder)
  }
  augment train { image: random_flip(p: 0.5) }
  split { train: 0.8, val: 0.1, test: 0.1 }
  batch 64
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
    lr Net.head = 0.001
    update main with head_opt, enc_opt
    every 1 epochs { validate ; checkpoint }
  }

  device auto
  precision mixed
}`,
    friction: [
      { kind: "B", note: "Which region thaws when, and at which learning rate, is the algorithm." },
      { kind: "A", note: "Parameter groups are resolved from structure, not from fragile parameter-name substrings." },
    ],
  },
  {
    id: "gan",
    title: "Alternating optimisation (GAN)",
    group: "Lifecycle",
    summary: "Two models, two objectives, two optimizers and an explicit update ratio.",
    code: `dim B
dim Z = 64
dim D = 128

model Generator(z: Tensor[B, Z]) -> Tensor[B, D] {
  linear(128)
  gelu
  linear(D)
  tanh
}

model Discriminator(x: Tensor[B, D]) -> Probs[B, 1] {
  linear(128)
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

source Noise = synthetic(features: 128)

data GanData from Noise {
  example {
    field z: Tensor[Z] = decode(row) |> to_float
    field real: Tensor[D] = decode(row) |> to_float
  }
  batch 32
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
}`,
    friction: [
      { kind: "B", note: "The 2:1 update ratio is the algorithm." },
      { kind: "D", note: "The reference backend approximates nested model composition with synthetic inputs." },
    ],
  },
  {
    id: "recurrence",
    title: "Runtime recurrence (scan)",
    group: "Dynamic",
    summary: "Static repetition and temporal recurrence are different constructs with different parameter semantics.",
    code: `dim B
dim T
dim F = 16
dim Hid = 64

model RNNTagger(x: Tensor[B, T, F]) -> Logits[B, T, 4] {
  let (states, final) = scan over x axis: 1 carry h: Tensor[B, Hid] init: zeros {
    let combined = concat(step, h, axis: -1)
    yield combined |> linear(Hid) |> tanh
  }
  return states |> linear(4)
}

# iterative refinement: the same block applied repeatedly to its own output
block Refine(x: Tensor[B, Hid]) -> Tensor[B, Hid] {
  residual {
    layernorm
    linear(Hid)
    gelu
  }
}

model Iterative(x: Tensor[B, Hid]) -> Tensor[B, Hid] {
  let step = Refine()
  for 5: step          # runtime-independent repetition of ONE parameter set
}`,
    friction: [
      { kind: "B", note: "Recurrence is genuinely temporal; the carried state and its initialisation are meaningful." },
      { kind: "A", note: "No hidden-state plumbing, no manual stacking of per-step outputs." },
    ],
  },
  {
    id: "custom-op",
    title: "Custom / foreign operation",
    group: "Escape hatches",
    summary: "Declare contracts, effects and backend implementations. Lost static knowledge is reported, not hidden.",
    code: `dim B
dim T
dim Heads = 4
dim D = 128
dim DH = D / Heads

custom op flash_attention(q: Tensor[B, Heads, T, DH],
                          k: Tensor[B, Heads, T, DH],
                          v: Tensor[B, Heads, T, DH]) -> Tensor[B, Heads, T, DH] {
  effects: pure
  backend torch: "flash_attn_func(q, k, v, causal=True)"
}

custom op mystery(x: Tensor[B, T, D]) {
  effects: pure
  shape: unknown              # the compiler reports the loss of static knowledge
  differentiable: false
}

fn heads(x: Tensor[B, T, D]) -> Tensor[B, Heads, T, DH] {
  return transpose(reshape(x, [B, T, Heads, DH]), 1, 2)
}

block FlashAttn(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  param wq: Tensor[D, D] init: xavier
  param wk: Tensor[D, D] init: xavier
  param wv: Tensor[D, D] init: xavier
  let out = flash_attention(heads(matmul(x, wq)), heads(matmul(x, wk)), heads(matmul(x, wv)))
  return reshape(transpose(out, 1, 2), [B, T, D])
}

model WithFlash(x: Tensor[B, T, D]) -> Tensor[B, T, D] {
  residual {
    layernorm
    FlashAttn()
  }
}`,
  },
  {
    id: "diagnostics",
    title: "Diagnostics tour (intentionally wrong)",
    group: "Diagnostics",
    summary: "Every error below is deliberate — run Check to see codes, locations and architectural context.",
    code: `dim B
dim S
dim T
dim D = 64

model Mismatch(a: Tensor[B, S, D], b: Tensor[B, T, D]) -> Tensor[B, S, D] {
  return a + b                       # AXS0403: cannot prove S = T
}

model BadResidual(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 16, 16] {
  residual {                         # AXS0404: shape changes, no projection given
    conv2d(64, kernel: 3, stride: 2, pad: 1)
  }
}

model BadConcat(x: Tensor[B, 3, 32, 32]) -> Tensor[B, 64, 32, 32] {
  split merge concat(1) {
    conv2d(32, kernel: 1)
    conv2d(32, kernel: 3)            # AXS0405: 30x30 vs 32x32 on axis 2
  }
}

model MissingChoice(x: Tensor[B, 16]) -> Tensor[B, 32] {
  linear()                           # AXS0501: output width is a choice
}

model NoStream(a: Tensor[B, 16], b: Tensor[B, 16]) -> Tensor[B, 32] {
  linear(32)                         # AXS0301: which input did you mean?
}

source S1 = csv(path: "t.csv")

data Leaky from S1 {
  example {
    field x: Tensor[8] = decode(row) |> standardize(fit: val)   # AXS0620: leakage
  }
  batch 8
}`,
  },
];

export const EXAMPLE_MAP = new Map(EXAMPLES.map((e) => [e.id, e]));
