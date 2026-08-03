"""Does the engine compute the same thing PyTorch does?

Every optimization in this repo is only meaningful if the answer stays yes, so
this runs before any benchmark and is wired into CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.export import export  # noqa: E402
from tools.models import REGISTRY, build  # noqa: E402

BIN = Path("build/ni_run")


def engine_output(model_path: Path, input_path: Path, threads: int = 1) -> np.ndarray:
    out = subprocess.run(
        [str(BIN), str(model_path), str(input_path), "--threads", str(threads)],
        capture_output=True, text=True, check=True)
    return np.array([float(v) for v in out.stdout.split()], dtype=np.float64)


def check(which: str, quantize: bool = False, n_inputs: int = 5, tol: float = 2e-4) -> dict:
    model, example, name = build(which)
    tag = f"{name}_i8" if quantize else name
    model_path = Path(f"models/{tag}.ngm")
    calib = torch.randn(16, *example.shape[1:])
    export(model, example, model_path, tag, quantize=quantize, calib=calib)

    worst_abs, agree, ref_all, got_all = 0.0, 0, [], []
    for i in range(n_inputs):
        torch.manual_seed(1000 + i)
        x = torch.randn(*example.shape)
        inp = Path(f"models/{tag}_input_{i}.bin")
        inp.write_bytes(x.numpy().astype("<f4").tobytes())
        with torch.no_grad():
            ref = model(x).numpy().astype(np.float64).ravel()
        got = engine_output(model_path, inp, threads=1)
        inp.unlink()
        if got.shape != ref.shape:
            raise SystemExit(f"{tag}: shape mismatch {got.shape} vs {ref.shape}")
        worst_abs = max(worst_abs, float(np.max(np.abs(got - ref))))
        agree += int(np.argmax(got) == np.argmax(ref))
        ref_all.append(ref)
        got_all.append(got)

    threaded = engine_output(model_path, Path(f"models/{name}_input.bin"), threads=4) \
        if Path(f"models/{name}_input.bin").exists() else None

    res = {
        "model": tag,
        "quantized": quantize,
        "max_abs_diff": worst_abs,
        "top1_agreement": f"{agree}/{n_inputs}",
        "mean_abs_ref": float(np.mean(np.abs(np.concatenate(ref_all)))),
    }
    if not quantize:
        res["ok"] = worst_abs < tol
    else:
        # int8 changes the arithmetic, so the bar is prediction agreement rather
        # than numerical closeness
        res["ok"] = agree >= n_inputs - 1
    if threaded is not None:
        res["threads_match"] = "n/a"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--quantized", action="store_true")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"{BIN} not built; run cmake --build build first")

    rows = []
    for which in args.models:
        rows.append(check(which, quantize=False))
        if args.quantized:
            rows.append(check(which, quantize=True))

    print()
    print(f"{'model':22s} {'max |diff|':>12s} {'top-1':>8s}  verdict")
    print("-" * 56)
    bad = 0
    for r in rows:
        verdict = "ok" if r["ok"] else "MISMATCH"
        bad += 0 if r["ok"] else 1
        print(f"{r['model']:22s} {r['max_abs_diff']:12.2e} {r['top1_agreement']:>8s}  {verdict}")
    if bad:
        raise SystemExit(f"{bad} model(s) disagree with PyTorch")
    print("\nall models agree with PyTorch")


if __name__ == "__main__":
    main()
