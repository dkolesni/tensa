# CPU vs GPU timing experiment — 2026-09-18

Requested follow-up to M4's correctness-only validation. These are measured timings of **TENSA-emitted
PyTorch models on both devices**, not a comparison against the much slower TypeScript interpreter.
Raw confirmed-run measurements: `cpu-gpu-benchmark-2026-09.json` beside this record.

## Hardware and method

- CPU: Intel Core i9-12900K, 24 logical CPUs; PyTorch's default **16 intra-op threads**, one inter-op thread.
- GPU: **NVIDIA GeForce RTX 5090**, explicitly selected as **cuda:1** in this Python environment.
- PyTorch: **2.11.0+cu128**, float32, TF32 disabled, eager execution (no torch.compile, mixed precision or CUDA graphs).
- Same generated architecture, initial weights, inputs and targets on CPU/GPU. Untimed forward outputs
  verified equal within 5e-4 absolute/relative tolerance.
- Five warm-up calls plus one untimed pilot. Five timed groups, reporting median per-batch wall time.
  CUDA synchronized at group boundaries; GPU work cannot escape the timer.
- Inference: eval + inference_mode, resident inputs. Adaptive repetitions per group amortize timer noise.
- Training: zero_grad + forward + MSE + backward + AdamW step. **20 steps/group on BOTH devices**,
  so both complete the same 106 steps including warm-up/pilot. Losses remained finite and nearly identical.
- Excludes imports, code generation, initial allocation, data loading and host-to-device transfer.
  Pageable batch upload (input + target, including allocation) is measured separately.
- Shared workstation, not an isolated benchmark host. Other processes were using both GPUs before the run;
  no processes were stopped or hardware settings changed. Group variation below matters.

## Models

| Workload | Batch | Image | Tokens | Width | Encoder blocks | Parameters |
|---|---:|---|---:|---:|---:|---:|
| Tiny M4 ViT | 2 | 4×6 RGB | 7 | 8 | 1 | 736 |
| Medium ViT | 8 | 128×128 RGB | 65 | 128 | 2 | 506,784 |
| Larger ViT | 8 | 224×224 RGB | 197 | 256 | 4 | 3,411,232 |

Larger fixtures retain patchify, class token, learned positions, pre-norm residual attention and 4× MLPs;
they use 16×16 patches and a 32-wide output head. The tiny fixture is the actual M4 ViT, not a rewritten
PyTorch approximation. Both larger fixtures are compiled by TENSA before benchmarking.

## Confirmed results (milliseconds per batch)

Speedup = CPU time / GPU time. Greater than one favors GPU.

| Workload | Mode | CPU median | RTX 5090 median | GPU speedup |
|---|---|---:|---:|---:|
| Tiny M4 ViT | inference | 0.285 | 0.845 | 0.34× (CPU ~3× faster) |
| Tiny M4 ViT | training | 3.318 | 3.205 | 1.04× (effectively tied amid noise) |
| Medium ViT | inference | 2.865 | 1.180 | 2.43× |
| Medium ViT | training | 15.284 | 7.059 | 2.17× |
| Larger ViT | inference | 30.935 | 2.520 | **12.27×** |
| Larger ViT | training | 106.895 | 10.077 | **10.61×** |

Larger workload training throughput: about **75 images/s CPU vs 794 images/s GPU**.
GPU peak PyTorch-allocated memory during its training measurement: **182.4 MiB** (not total device VRAM usage).
Separate median batch-upload time: tiny 0.034 ms, medium 0.300 ms, larger **0.546 ms**.
These upload numbers are not an end-to-end dataloader benchmark and should not simply be advertised
as perfectly overlappable or always additive.

### Variability: min–max group averages (ms)

| Workload | CPU inference | GPU inference | CPU training | GPU training |
|---|---|---|---|---|
| Tiny | 0.245–0.311 | 0.606–1.102 | 2.796–3.459 | 3.052–4.030 |
| Medium | 2.697–3.138 | 1.092–1.504 | 14.719–17.914 | 6.076–7.600 |
| Larger | 27.683–54.019 | 2.311–3.315 | 105.028–132.080 | 8.954–11.243 |

An earlier exploratory run used adaptive training repetition counts and showed ~10.7× larger-model
training speedup. It is not pooled into the table: the confirmed run above uses identical training
step counts to avoid comparing different optimizer evolution lengths.

## Interpretation

The GPU is not inherently faster for every TENSA program. Tiny eager graphs are dominated by Python,
launch and synchronization overhead; CPU wins tiny inference here. Increasing arithmetic per batch
makes the GPU advantage clear: approximately **10–12×** for the 3.4M-parameter workload under these
conditions. This is not a throughput guarantee for all architectures, full-size ViT, mixed precision,
compiled execution or another CPU thread count.

GPU remains the execution default as requested; CPU is explicitly selected only for the benchmark
baseline. No CPU fallback or changes to model-language semantics were introduced.

## Reproduce

```sh
npx tsx hardening/benchmark.ts
K:/Anima-TrainFlow/python_embeded/python.exe hardening/benchmark.py --gpu cuda:1 --threads 16
```

Query `torch.cuda.get_device_name(i)` before choosing an index. CUDA and nvidia-smi enumerate the two
cards differently here. Generated source/fixtures and fresh timing JSON go under ignored
`hardening/.m4-validation/benchmark/`.

## Correction to earlier M4 hardware reporting

The saved M4 correctness report identifies **RTX 4090** (default cuda:0), not RTX 5090. Earlier prose
incorrectly inferred the card from nvidia-smi index zero. The M4 report, revalidation record and portal
docs have been corrected. **This performance experiment explicitly selected and reported RTX 5090.**
