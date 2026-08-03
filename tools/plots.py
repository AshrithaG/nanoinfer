"""Figures for the README: the optimization ladder and the ONNX Runtime gap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

MODELS = ["mnist_cnn", "kws_dscnn", "vww_mobilenet"]
PRETTY = {
    "mnist_cnn": "MNIST CNN\n(dense, 28x28)",
    "kws_dscnn": "keyword spotting\n(depthwise, 49x10)",
    "vww_mobilenet": "visual wake words\n(depthwise, 96x96)",
}
VARIANTS = [
    ("naive_conv", "naive conv loops", "#8a8f98"),
    ("im2col_gemm", "im2col + GEMM, fused, pooled", "#4b7ade"),
    ("int8", "int8 weights", "#3f9d54"),
    ("onnxruntime", "ONNX Runtime", "#d9772e"),
]


def load() -> tuple[dict, dict]:
    blob = json.loads((RESULTS / "bench.json").read_text())
    best: dict[tuple[str, str], float] = {}
    by_thread: dict[tuple[str, str], dict[int, float]] = {}
    for r in blob["rows"]:
        key = (r["model"], r["variant"])
        v = float(r["median_us"])
        t = int(r["threads"])
        if key not in best or v < best[key]:
            best[key] = v
        by_thread.setdefault(key, {})[t] = v
    return best, by_thread


def ladder(best: dict, by_thread: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    x = np.arange(len(MODELS))
    width = 0.2
    ax = axes[0]
    for i, (variant, label, color) in enumerate(VARIANTS):
        vals = [best.get((m, variant), np.nan) for m in MODELS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("median latency per inference (us, log scale)")
    ax.set_xticks(x, [PRETTY[m] for m in MODELS], fontsize=9)
    ax.set_title("Where the time goes, best thread count per configuration", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_ylim(top=max(best.get((m, "naive_conv"), 1) for m in MODELS) * 3)
    for i, m in enumerate(MODELS):
        n = best.get((m, "naive_conv"))
        f = best.get((m, "im2col_gemm"))
        if n and f:
            ax.annotate(f"{n / f:.0f}x faster", xy=(i, n * 1.25), ha="center",
                        fontsize=9, color="#4b7ade", fontweight="bold")

    ax = axes[1]
    for m, marker in zip(MODELS, ["o", "s", "^"]):
        d = by_thread.get((m, "im2col_gemm"), {})
        if not d:
            continue
        ts = sorted(d)
        base = d[ts[0]]
        ax.plot(ts, [base / d[t] for t in ts], marker + "-", label=PRETTY[m].split("\n")[0])
    ts = sorted({t for m in MODELS for t in by_thread.get((m, "im2col_gemm"), {})})
    if ts:
        ax.plot(ts, ts, "--", color="#8a8f98", lw=1, label="perfect scaling")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("Thread scaling: small models have nothing to parallelize", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "latency.png", dpi=150)
    print("wrote", FIGURES / "latency.png")


def table(best: dict):
    lines = ["| model | naive conv | this engine | ONNX Runtime | ladder speedup | vs ORT |",
             "|---|---|---|---|---|---|"]
    for m in MODELS:
        n = best.get((m, "naive_conv"), float("nan"))
        f = best.get((m, "im2col_gemm"), float("nan"))
        o = best.get((m, "onnxruntime"), float("nan"))
        lines.append(f"| {m} | {n:.0f} us | {f:.0f} us | {o:.0f} us | "
                     f"{n / f:.1f}x | {o / f:.2f}x |")
    out = "\n".join(lines)
    (RESULTS / "latency_table.md").write_text(out + "\n")
    print("\n" + out)


if __name__ == "__main__":
    best, by_thread = load()
    ladder(best, by_thread)
    table(best)
