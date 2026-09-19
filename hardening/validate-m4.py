"""M4 fidelity gate: GPU when available, CPU otherwise (--device forces one). H-011.
Run validate-m4.ts first. No dependencies beyond PyTorch and the standard library.
Fixtures align weights, inputs and stochastic draws; RNG algorithms need not match.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default="hardening/.m4-validation")
parser.add_argument("--device", default=None, help="GPU when available, CPU otherwise; pass cpu/cuda to force one")
args = parser.parse_args()
directory = Path(args.directory)
device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
torch.set_num_threads(1)
py = lambda name: re.sub(r"[^A-Za-z0-9_]", "_", name)

def tensor(t, kind=None):
    dtype = torch.bool if kind == "Mask" else torch.long if kind in ("Tokens", "Class") else torch.float32
    return torch.tensor(t["data"], dtype=dtype, device=device).reshape(t["shape"])

results = []
for entry in json.loads((directory / "manifest.json").read_text()):
    name = entry["id"]
    if "rejected" in entry:
        results.append({"id": name, "expected_rejection": entry["rejected"]})
        continue
    spec = importlib.util.spec_from_file_location(py(name), directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads((directory / (name + ".json")).read_text())
    assert module.execution_device("cpu").type == "cpu"
    # H-011: GPU when available, CPU otherwise — never an error for a missing accelerator.
    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.backends.mps, "is_available", return_value=False):
        assert module.execution_device().type == "cpu"
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert module.execution_device().type == "cuda"
    checked = 0
    for g in data["graphs"]:
        model = getattr(module, g["name"])(device=args.device, **data["dims"]).eval()
        assert model._device_anchor.device.type == device.type
        with torch.no_grad():
            for key, value in data["params"].items():
                if py(key) in model.p: model.p[py(key)].copy_(tensor(value))
            for key, value in data["states"].items():
                if hasattr(model, py(key)): getattr(model, py(key)).copy_(tensor(value))
        inputs = [tensor(t, t["kind"]) for t in g["inputs"]]
        saved = {k: v.clone() for k, v in model.named_buffers()}
        # Controlled draws isolate backend math from RNG implementation differences.
        draws = iter(g["draws"])
        with patch.object(torch, "randn_like", side_effect=lambda x, **kwargs: tensor(next(draws))):
            out = model(*inputs)
        assert next(draws, None) is None, "emitter dropped a stochastic draw"
        out = out if isinstance(out, tuple) else (out,)
        for actual, expected in zip(out, g["output"], strict=True):
            assert actual.device.type == device.type
            torch.testing.assert_close(actual, tensor(expected), atol=3e-4, rtol=3e-4, msg=lambda msg: f"{name}/{g['name']} forward: {msg}")
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, saved[k], msg=f"{name} mutated state in eval")
        loss = sum(t.mean() for t in out)
        if loss.requires_grad:
            loss.backward()
            for key, expected in g["gradients"].items():
                if py(key) in model.p:
                    actual = model.p[py(key)].grad
                    assert actual is not None, key
                    torch.testing.assert_close(actual.flatten(), torch.tensor(expected, dtype=torch.float32, device=device), atol=5e-4, rtol=5e-4, msg=f"{name}/{key} gradient")
        # State is checkpointed and restorable, including device movement.
        clone = getattr(module, g["name"])(device=args.device, **data["dims"])
        clone.load_state_dict(model.state_dict())
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, clone.state_dict()[k])
        if g["draws"]:
            a, b = model(*inputs), model(*inputs)
            a = a if isinstance(a, tuple) else (a,)
            b = b if isinstance(b, tuple) else (b,)
            assert any(not torch.equal(x, y) for x, y in zip(a, b)), "eval noise disabled"
        model.train()
        draws = iter(g["trainingDraws"])
        with patch.object(torch, "randn_like", side_effect=lambda x, **kwargs: tensor(next(draws))):
            train_out = model(*inputs)
        assert next(draws, None) is None, "emitter dropped a training draw"
        train_out = train_out if isinstance(train_out, tuple) else (train_out,)
        for actual, expected in zip(train_out, g["trainingOutput"], strict=True):
            torch.testing.assert_close(actual, tensor(expected), atol=5e-4, rtol=5e-4, msg=lambda msg: f"{name} train forward: {msg}")
        for key, expected in g["trainingStates"].items():
            torch.testing.assert_close(getattr(model, py(key)), tensor(expected), atol=5e-4, rtol=5e-4, msg=lambda msg: f"{name}/{key} state: {msg}")
        clone.load_state_dict(model.state_dict())
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, clone.state_dict()[k])
        checked += 1
    batch = {k: tensor(v, v["kind"]) for k, v in data["batch"].items()}
    for plan in data["plans"]:
        # Incoming host batches exercise generated batch transfer, too.
        host_batch = {k: v.cpu() for k, v in batch.items()}
        models, metrics = getattr(module, "train_" + plan)([host_batch] * 3, device=args.device, **data["dims"])
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
        assert all(p.device.type == device.type for m in models.values() for p in m.parameters())
    result = {"id": name, "graphs": checked, "plans": len(data["plans"]), "device": str(device), "forward_and_gradients": "pass"}
    results.append(result)
    print(json.dumps(result))
report = {"torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "results": results}
(directory / f"results-{device.type}.json").write_text(json.dumps(report, indent=2))
print(f"PASS: {len(results)} challenges, torch {torch.__version__}, device={device}")
