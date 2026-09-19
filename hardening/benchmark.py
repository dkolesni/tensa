"""Benchmark emitted TENSA on explicit CPU and required GPU, with warm-up/synchronization.
This intentionally compares two PyTorch devices, NOT PyTorch against the TS reference interpreter.
No silent CPU fallback. Imports, allocations, data generation and compilation are outside timings.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="cuda:0")
parser.add_argument("--threads", type=int, default=torch.get_num_threads())
parser.add_argument("--groups", type=int, default=5)
args = parser.parse_args()
gpu = torch.device(args.gpu)
if gpu.type != "cuda" or not torch.cuda.is_available():
    raise RuntimeError("This comparison requires a CUDA GPU; CPU is used only as the explicitly selected baseline.")
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(123)
root = Path(__file__).resolve().parent / ".m4-validation" / "benchmark"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, iterations=None):
    for _ in range(5):
        fn()
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    pilot = time.perf_counter() - start
    if iterations is None:
        iterations = max(3, min(100, int(0.3 / max(pilot, 1e-6))))
    samples = []
    for _ in range(args.groups):
        sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "iterations_per_group": iterations, "group_ms_per_step": samples}


report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(gpu), "gpu_device": str(gpu),
          "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "cpu_threads": args.threads,
          "dtype": "float32", "tf32": False, "compile": False, "groups": args.groups,
          "timing": "synchronized wall clock; 5 warmups + pilot, median of groups; training 20 steps/group on both devices; resident inputs; no imports/allocation/data I/O",
          "results": []}
print(json.dumps({k:v for k,v in report.items() if k != "results"}), flush=True)
for index, w in enumerate(json.loads((root / "manifest.json").read_text())):
    spec = importlib.util.spec_from_file_location("bench_" + w["id"].replace("-", "_"), root / (w["id"] + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu = torch.device("cpu")  # explicit CPU opt-in, never automatic fallback
    model_cpu = module.ViT(device="cpu", B=w["batch"]).eval()
    model_gpu = module.ViT(device=gpu, B=w["batch"]).eval()
    model_gpu.load_state_dict(model_cpu.state_dict())
    initial = {k:v.detach().clone() for k,v in model_cpu.state_dict().items()}
    x_cpu = torch.randn(w["batch"], 3, w["height"], w["width"], device=cpu)
    x_gpu = x_cpu.to(gpu)
    with torch.inference_mode():
        expected, actual = model_cpu(x_cpu), model_gpu(x_gpu)
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)
        target_cpu = torch.randn_like(expected)
    target_gpu = target_cpu.to(gpu)
    row = {**w, "parameters": sum(p.numel() for p in model_cpu.parameters()), "forward_parity": "pass"}
    # Alternate which device runs first to reduce systematic warm/background ordering bias.
    devices = [("cpu", cpu, model_cpu, x_cpu, target_cpu), ("gpu", gpu, model_gpu, x_gpu, target_gpu)]
    if index % 2:
        devices.reverse()
    for name, device, model, x, target in devices:
        model.eval()
        with torch.inference_mode():
            row[name + "_inference"] = measure(lambda: model(x), device)
        model.load_state_dict(initial)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_value = None
        def step():
            global loss_value
            opt.zero_grad(set_to_none=True)
            output = model(x)
            loss_value = (output - target).square().mean()
            loss_value.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        row[name + "_training"] = measure(step, device, iterations=20)
        assert torch.isfinite(loss_value).item(), "non-finite training loss"
        row[name + "_last_loss"] = loss_value.item()
        if device.type == "cuda":
            row["gpu_peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
        del opt
        model.zero_grad(set_to_none=True)
    # Pageable host-to-device upload cost measured separately; includes allocation and sync.
    def upload():
        return x_cpu.to(gpu), target_cpu.to(gpu)
    row["gpu_upload"] = measure(upload, gpu)
    for mode in ("inference", "training"):
        row[mode + "_speedup"] = row["cpu_" + mode]["median_ms"] / row["gpu_" + mode]["median_ms"]
        row["gpu_" + mode + "_images_per_second"] = w["batch"] * 1000 / row["gpu_" + mode]["median_ms"]
    report["results"].append(row)
    print(json.dumps(row), flush=True)
    del step, upload, model_cpu, model_gpu, model, x_cpu, x_gpu, x, target_cpu, target_gpu, target, devices, initial, expected, actual, loss_value
    gc.collect()
    torch.cuda.empty_cache()
output = root / f"timings-{gpu.index or 0}-{args.threads}threads.json"
output.write_text(json.dumps(report, indent=2))
print(f"Saved {output}", flush=True)
