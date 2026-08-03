"""Run the optimization ladder and compare against ONNX Runtime.

Same weights, same inputs, same machine, same measurement protocol: warmup,
then the median of many timed single-inference calls. ORT gets its own thread
setting rather than being pinned to one, so the comparison is against how you
would actually deploy it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.export import export  # noqa: E402
from tools.models import REGISTRY, build  # noqa: E402

RESULTS = Path("results")
BIN = Path("build/ni_bench")


def engine_rows(model_path: Path, threads: list[int], iters: int, ladder: bool) -> list[dict]:
    cmd = [str(BIN), str(model_path), "--iters", str(iters),
           "--threads", ",".join(map(str, threads))]
    if ladder:
        cmd.append("--ladder")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def onnx_bench(model: torch.nn.Module, example: torch.Tensor, name: str,
               iters: int, threads: int) -> dict:
    import onnxruntime as ort

    path = RESULTS / f"{name}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, str(path), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = example.numpy()

    for _ in range(20):
        sess.run(None, {"x": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"x": x})
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return {
        "variant": "onnxruntime",
        "threads": threads,
        "median_us": round(times[len(times) // 2], 1),
        "p99_us": round(times[int(len(times) * 0.99)], 1),
        "mean_us": round(sum(times) / len(times), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=RESULTS / "bench.json")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"{BIN} not built; run cmake --build build first")

    all_rows: list[dict] = []
    for which in args.models:
        model, example, name = build(which)
        f32_path = Path(f"models/{name}.ngm")
        if not f32_path.exists():
            export(model, example, f32_path, name)
        i8_path = Path(f"models/{name}_i8.ngm")
        calib = torch.randn(16, *example.shape[1:])
        export(model, example, i8_path, f"{name}_i8", quantize=True, calib=calib)

        rows = engine_rows(f32_path, args.threads, args.iters, ladder=True)
        rows += engine_rows(i8_path, [1], max(args.iters // 4, 30), ladder=False)
        for r in rows:
            r["model"] = name
            if r["variant"] == "engine":
                r["variant"] = "int8"
        all_rows += rows

        for th in args.threads:
            r = onnx_bench(model, example, name, args.iters, th)
            r["model"] = name
            r["arena_kb"] = ""
            all_rows.append(r)

        print(f"  {name}: {len(rows)} engine rows + {len(args.threads)} ORT rows")

    RESULTS.mkdir(exist_ok=True)
    meta = {
        "machine": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "iters": args.iters,
    }
    args.out.write_text(json.dumps({"meta": meta, "rows": all_rows}, indent=2))

    print(f"\n{'model':16s} {'variant':14s} {'thr':>4s} {'median us':>10s} {'p99 us':>9s}")
    print("-" * 60)
    for r in sorted(all_rows, key=lambda r: (r["model"], r["variant"], int(r["threads"]))):
        print(f"{r['model']:16s} {r['variant']:14s} {int(r['threads']):4d} "
              f"{float(r['median_us']):10.1f} {float(r['p99_us']):9.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
