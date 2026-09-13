"""Triton int8 GEMM next to the hand-written CUDA kernels and cuBLAS.

The CUDA backend's best int8 GEMM (wmma+smem) trails cuBLAS by about 30% at
n >= 2048, and the README lists the causes in the order it would attack them: no
cp.async overlap between loading and computing, a 64x64 block tile, unswizzled
shared memory, one K-stage in flight. In Triton the compiler chooses the
shared-memory swizzle, and pipelining depth and tile shape are one launch
parameter each, so the list can be tested rather than argued:

1. At exactly the wmma+smem tiling, how fast is a Triton kernel, and what did the
   compiler emit that the hand-written kernel does not? Answered from the compiled
   artifacts (TTGIR shared-memory layouts, PTX, SASS).
2. Which of the listed causes carry the gap? The ladder turns them on one at a time.
3. Was cuBLAS measured on its fast path? gemm_i8.cu passes B row-major (NN), while
   int8 tensor-core GEMMs are built around both operands being K-contiguous (TN).

Every implementation runs in this one process, on one stream, under one timer, and
must match an exact CPU reference at every timed size before it is timed. Time is
reported two ways: back-to-back launches between CUDA events, which is what
ni_bench_cuda measures, and CUDA-graph replay, which takes the host's launch cost
out of the window. Where the two disagree, the difference is launch overhead.

On the GPU machine, from the repo root: bash tools/triton_gemm_vm.sh
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

SIZES = [256, 512, 1024, 2048, 4096]
# Launches per timing window, the same counts ni_bench_cuda uses.
ITERS = {256: 200, 512: 100, 1024: 50, 2048: 20, 4096: 10}
# M, N and K all differ, so an implementation that mixes two of them up cannot pass.
VERIFY_SHAPE = (256, 512, 768)
# The most shared memory one block can opt into on sm_89.
SMEM_LIMIT = 99 * 1024
LAYOUTS = ("NN", "TN")


@triton.jit
def int8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Which output tile this program owns. GROUP_M=1 is plain row-major tile order,
    # the order gemm_i8.cu gets from blockIdx. A larger GROUP_M walks GROUP_M rows of
    # tiles together, so programs running at the same time share A and B tiles in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # No masks: the launcher only takes shapes the tiles divide, the same restriction
    # wmma+smem has. Staging through shared memory, its layout, and any overlap of
    # loads with math are left entirely to the compiler.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@dataclass(frozen=True)
class Cfg:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    group_m: int

    @property
    def name(self) -> str:
        return f"{self.bm}x{self.bn}x{self.bk} w{self.warps} s{self.stages} g{self.group_m}"

    def smem_estimate(self) -> int:
        # one int8 A tile and one B tile per pipeline stage
        return self.stages * (self.bm * self.bk + self.bk * self.bn)


# One change per step. Swizzling is not a step because Triton has no switch for it:
# it is in every row, and the codegen table shows the layout the compiler picked.
LADDER = [
    ("same tiling as wmma+smem", Cfg(64, 64, 32, 4, 1, 1)),
    ("+ pipelining, 3 stages", Cfg(64, 64, 32, 4, 3, 1)),
    ("+ 128x128 tile, 8 warps", Cfg(128, 128, 32, 8, 3, 1)),
    ("+ grouped tile order", Cfg(128, 128, 32, 8, 3, 8)),
]

# The search the "best" rows come from. Modelled on the configs in Triton's matmul
# tutorial, with larger K tiles because an int8 element is a quarter of an fp32 one.
SWEEP = [Cfg(bm, bn, bk, w, s, 8) for bm, bn, bk, w, s in [
    (64, 64, 32, 4, 3), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 64, 64, 4, 4),
    (128, 128, 32, 4, 4), (128, 128, 32, 8, 3), (128, 128, 64, 4, 3), (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (256, 128, 64, 8, 3), (64, 256, 64, 4, 4),
    (256, 64, 64, 4, 4), (128, 256, 32, 8, 4), (64, 128, 128, 4, 4), (128, 64, 128, 4, 4),
]]


class Declined(Exception):
    """The implementation does not take this shape, which is not the same as failing it."""


@dataclass
class Impl:
    name: str
    layout: str  # "NN": B is [K, N] row-major, as gemm_i8.cu takes it. "TN": K-contiguous.
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]
    cfg: Cfg | None = None


def first_line(e: BaseException) -> str:
    text = str(e).strip()
    return (text.splitlines()[0] if text else type(e).__name__)[:200]


# ------------------------------------------------------------------ implementations
def triton_impl(name: str, cfg: Cfg, layout: str) -> Impl:
    def run(a, b, c):
        m, k = a.shape
        n = b.shape[1]
        if m % cfg.bm or n % cfg.bn or k % cfg.bk:
            raise Declined
        grid = (triton.cdiv(m, cfg.bm) * triton.cdiv(n, cfg.bn),)
        # B's strides carry the layout. Triton specializes on a stride of 1, so the
        # NN and TN calls compile to different kernels from the same source.
        return int8_gemm_kernel[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=cfg.bm, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk, GROUP_M=cfg.group_m,
            num_warps=cfg.warps, num_stages=cfg.stages,
        )

    return Impl(name, layout, run, cfg)


def load_shim(path: Path) -> tuple[ctypes.CDLL | None, str]:
    if not path.exists():
        return None, f"{path} not found, build it with --target ni_gemm_shim"
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        return None, f"could not load {path}: {first_line(e)}"
    for fn in (lib.ni_gemm_i8, lib.ni_cublas_i8):
        fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4
        fn.restype = ctypes.c_int
    lib.ni_cublas_set_workspace.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ni_cublas_set_workspace.restype = ctypes.c_int
    return lib, "loaded"


def shim_impl(name: str, layout: str, fn, selector: int) -> Impl:
    def run(a, b, c):
        m, k = a.shape
        n = b.shape[1]
        rc = fn(selector, m, n, k, a.data_ptr(), b.data_ptr(), c.data_ptr(),
                torch.cuda.current_stream().cuda_stream)
        if rc == 1:
            raise Declined
        if rc:
            raise RuntimeError(f"{name}: shim returned {rc}")

    return Impl(name, layout, run)


def int_mm(a, b, c):
    torch._int_mm(a, b, out=c)


def build_impls(lib) -> list[Impl]:
    impls = []
    if lib is not None:
        # The hand-written kernels only take B row-major, so they only run NN.
        impls += [shim_impl(name, "NN", lib.ni_gemm_i8, i)
                  for i, name in enumerate(["tiled+dp4a", "wmma", "wmma+smem"])]
        impls += [shim_impl("cuBLAS", "NN", lib.ni_cublas_i8, 0),
                  shim_impl("cuBLAS", "TN", lib.ni_cublas_i8, 1)]
    # torch._int_mm goes through cuBLASLt rather than cublasGemmEx.
    impls += [Impl("torch._int_mm", "NN", int_mm), Impl("torch._int_mm", "TN", int_mm)]
    impls += [triton_impl(f"Triton: {label}", cfg, "NN") for label, cfg in LADDER]
    return impls


# ------------------------------------------------------------------------- timing
def time_eager(fn: Callable[[], object], iters: int, windows: int) -> list[float]:
    """ni_bench_cuda's protocol: one warmup, then `iters` back-to-back launches between
    two CUDA events, reported per launch. Repeated `windows` times."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        out.append(start.elapsed_time(stop) / iters)
    return out


def time_graph(fn: Callable[[], object], iters: int, windows: int) -> list[float]:
    """The same launches captured once into a CUDA graph and replayed. The CPU work of
    each launch happens at capture, outside the timed window."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        out.append(start.elapsed_time(stop) / iters)
    del graph
    return out


def operands(m: int, n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=g)
    b = torch.randint(-127, 128, (k, n), dtype=torch.int8, generator=g)
    # float64 BLAS gives the exact integer answer: every product and every partial sum
    # is an integer far below 2**53, so no summation order can round anything.
    ref = (a.double() @ b.double()).to(torch.int32).cuda()
    by_layout = {"NN": b.cuda(), "TN": b.t().contiguous().cuda().t()}
    return a.cuda(), by_layout, ref


def check(impl: Impl, a, b, ref) -> tuple[str, torch.Tensor]:
    """Run once and compare with the exact reference. Only "exact" passes."""
    # Pre-filled, so a kernel that writes nothing cannot pass on a stale allocation
    # that still holds the previous implementation's correct answer.
    c = torch.full(ref.shape, -1, dtype=torch.int32, device="cuda")
    try:
        impl.run(a, b, c)
        torch.cuda.synchronize()
    except Declined:
        return "declined", c
    except Exception as e:  # compile errors and resource limits are results too
        return f"error: {first_line(e)}", c
    bad = int((c != ref).sum())
    return ("exact" if bad == 0 else f"FAIL: {bad} of {ref.numel()} outputs wrong"), c


def measure(impl: Impl, a, b, ref, iters: int, windows: int) -> dict:
    status, c = check(impl, a, b, ref)
    if status != "exact":
        return {"status": status}

    def call():
        return impl.run(a, b, c)

    row: dict = {"status": "ok"}
    eager = time_eager(call, iters, windows)
    row.update(eager_ms=statistics.median(eager), eager_windows=eager)
    try:
        graph = time_graph(call, iters, windows)
        row.update(graph_ms=statistics.median(graph), graph_windows=graph)
    except Exception as e:
        row["graph_status"] = f"capture failed: {first_line(e)}"
    return row


def summary(r: dict) -> str:
    if r["status"] != "ok":
        return r["status"]
    g = r.get("graph_ms")
    graph = r.get("graph_status", "n/a") if g is None else f"{g:.4f} ms"
    return f"eager {r['eager_ms']:.4f} ms   graph {graph}"


# ------------------------------------------------------------------------ codegen
SASS_OPS = ["IMMA", "LDSM", "LDGSTS", "LDS", "STS", "LDG", "BAR", "DEPBAR", "PRMT", "STG"]
SASS_LINE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]\s+)?([A-Z][A-Z0-9_]*)")
PTX_PATTERNS = {
    "cp.async": r"\bcp\.async\.c[ag]\.",
    "ldmatrix": r"\bldmatrix\.",
    "mma.sync": r"\bmma\.sync\.",
    "ld.shared": r"\bld\.shared\.",
    "st.shared": r"\bst\.shared\.",
}
HAND_KERNELS = {"tiled+dp4a": "k_tiled_dp4aE", "wmma": "k_wmmaE", "wmma+smem": "k_wmma_smemE"}


def find_cuobjdump() -> str | None:
    # Triton ships its own, matched to the ptxas that built its kernels.
    bundled = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
    return str(bundled) if bundled.exists() else shutil.which("cuobjdump")


def cuobjdump(tool: str, flag: str, binary: Path) -> str:
    return subprocess.run([tool, flag, str(binary)], capture_output=True, text=True,
                          check=True).stdout


def split_functions(sass: str) -> dict[str, str]:
    parts = re.split(r"\n\s*Function : (\S+)\n", sass)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def sass_counts(sass: str) -> dict[str, int]:
    ops = Counter(m.group(1) for m in SASS_LINE.finditer(sass))
    return {"instructions": sum(ops.values()), **{op: ops[op] for op in SASS_OPS}}


def triton_codegen(kernel, tool: str | None, save_to: Path | None) -> dict:
    asm = getattr(kernel, "asm", None)
    if not asm:
        return {"status": "this Triton version exposes no compiled artifacts"}
    ptx, ttgir = asm.get("ptx", ""), asm.get("ttgir", "")
    encodings = sorted(set(re.findall(r"#[\w.]*shared<\{[^}]*\}>", ttgir)))
    phases = [int(p) for enc in encodings for p in re.findall(r"maxPhase = (\d+)", enc)]
    async_copies = re.findall(r"async_copy_global_to_local|insert_slice_async", ttgir)
    rec = {
        "regs": getattr(kernel, "n_regs", None),
        "local_bytes": getattr(kernel, "n_spills", None),
        "smem_bytes": getattr(getattr(kernel, "metadata", None), "shared", None),
        "shared_encodings": encodings,
        "swizzled": any(p > 1 for p in phases),
        "async_copies_ttgir": len(async_copies),
        "ptx": {k: len(re.findall(p, ptx)) for k, p in PTX_PATTERNS.items()},
        "mma_shapes": sorted(set(re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)", ptx))),
    }
    sass = ""
    if tool and asm.get("cubin"):
        with tempfile.TemporaryDirectory() as d:
            cubin = Path(d) / "kernel.cubin"
            cubin.write_bytes(asm["cubin"])
            sass = cuobjdump(tool, "-sass", cubin)
        rec["sass"] = sass_counts(sass)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        for ext, text in (("ttgir", ttgir), ("ptx", ptx), ("sass", sass)):
            if text:
                save_to.with_suffix(f".{ext}").write_text(text)
    return rec


def shim_codegen(shim: Path, tool: str, save_dir: Path | None) -> dict:
    funcs = split_functions(cuobjdump(tool, "-sass", shim))
    usage_pat = r"Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
    usage = re.findall(usage_pat, cuobjdump(tool, "-res-usage", shim))
    out = {}
    for name, tag in HAND_KERNELS.items():
        fn = next((f for f in funcs if tag in f), None)
        if fn is None:
            continue
        rec: dict = {"sass": sass_counts(funcs[fn]), "swizzled": False}
        for f, regs, _stack, smem, local in usage:
            if tag in f:
                rec.update(regs=int(regs), smem_bytes=int(smem), local_bytes=int(local))
        out[name] = rec
        if save_dir is not None and name == "wmma+smem":
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "hand_wmma_smem.sass").write_text(funcs[fn])
    return out


# ---------------------------------------------------------------------- reporting
def gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi unavailable"]
    me = f"{os.getpid()},"
    return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith(me)]


def environment(shim_status: str, windows: int) -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        driver = smi[0] if smi else "unknown"
    except (OSError, subprocess.SubprocessError):
        driver = "unknown"
    return {"device": p.name, "sm": f"sm_{p.major}{p.minor}", "sms": p.multi_processor_count,
            "driver": driver, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "triton": triton.__version__, "python": platform.python_version(),
            "shim": shim_status, "windows": windows}


def markdown(res: dict, sizes: list[int]) -> str:
    rows, best, env = res["rows"], res["best"], res["env"]

    def ms(name: str, layout: str, n: int, key: str = "graph_ms") -> float | None:
        for r in rows:
            if (r["impl"], r["layout"], r["n"]) == (name, layout, n) and r["status"] == "ok":
                return r.get(key)
        return None

    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    def table(header: list[str], body: list[list[str]]) -> list[str]:
        return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                *("| " + " | ".join(r) + " |" for r in body), ""]

    ladder = [(label, f"Triton: {label}") for label, _ in LADDER]
    out = [
        f"# int8 GEMM: hand-written CUDA, Triton and cuBLAS on {env['device']} ({env['sm']})",
        "",
        f"torch {env['torch']} (CUDA {env['torch_cuda']}), Triton {env['triton']}, "
        f"driver {env['driver']}. Hand-written kernels: {env['shim']}.",
        "",
        "C[n,n] int32 = A[n,n] int8 x B[n,n] int8. Milliseconds per GEMM, median of "
        f"{env['windows']} timing windows. Every number passed an exact CPU check at its own "
        "size first. Unless a column says eager, times are CUDA-graph replay, which leaves "
        "host launch cost out.",
        "",
        "## Ladder, B row-major (NN)",
        "",
    ]
    body = []
    for n in sizes:
        top = best["NN"].get(n, {})
        cub = ms("cuBLAS", "NN", n)
        ratio = cub / top["graph_ms"] if cub and top.get("graph_ms") else None
        body.append([str(n), f(ms("wmma+smem", "NN", n)),
                     *(f(ms(name, "NN", n)) for _, name in ladder),
                     f"{f(top.get('graph_ms'))} ({top.get('cfg', 'n/a')})", f(cub),
                     "n/a" if ratio is None else f"{ratio:.2f}x"])
    out += table(["n", "wmma+smem", *(label for label, _ in ladder), "Triton best",
                  "cuBLAS", "Triton best vs cuBLAS"], body)

    out += ["## Layout: B row-major (NN) or K-contiguous (TN)", ""]
    cols = [("cuBLAS NN", "cuBLAS", "NN"), ("cuBLAS TN", "cuBLAS", "TN"),
            ("torch._int_mm NN", "torch._int_mm", "NN"),
            ("torch._int_mm TN", "torch._int_mm", "TN")]
    body = []
    for n in sizes:
        cells = {label: ms(name, lay, n) for label, name, lay in cols}
        for lay in LAYOUTS:
            cells[f"Triton best {lay}"] = best[lay].get(n, {}).get("graph_ms")
        timed = {k: v for k, v in cells.items() if v}
        fastest = min(timed, key=timed.get) if timed else None
        tops = 2 * n**3 / (timed[fastest] * 1e-3) / 1e12 if fastest else None
        body.append([str(n), *(f(v) for v in cells.values()), fastest or "n/a",
                     "n/a" if tops is None else f"{tops:.1f}"])
    header = ["n", *(label for label, _, _ in cols), "Triton best NN", "Triton best TN"]
    out += table(header + ["fastest", "TOPS"], body)

    out += ["## Back-to-back launches (ni_bench_cuda's method) against CUDA-graph replay", ""]
    who = [("wmma+smem", "NN"), ("cuBLAS", "NN"), ("torch._int_mm", "NN"), (ladder[0][1], "NN")]
    header = ["n"]
    for name, _ in who:
        header += [f"{name} eager", "graph"]
    body = [[str(n), *(f(ms(name, lay, n, key)) for name, lay in who
                       for key in ("eager_ms", "graph_ms"))] for n in sizes]
    out += table(header, body)

    out += ["## What each kernel compiled to", "",
            "Static counts from cuobjdump -sass: which instructions exist, not how often they "
            "run. Triton kernels as compiled for B row-major. The hand-written kernels stage "
            "through plain __shared__ arrays, so their shared memory is unswizzled by "
            "construction.", ""]
    ops = ["IMMA", "LDSM", "LDGSTS", "LDS", "STS", "LDG", "BAR", "PRMT"]
    body = []
    for name, rec in res["codegen"].items():
        sass = rec.get("sass", {})
        body.append([name, str(rec.get("regs", "n/a")), str(rec.get("local_bytes", "n/a")),
                     str(rec.get("smem_bytes", "n/a")), str(rec.get("swizzled", "n/a")),
                     ", ".join(rec.get("mma_shapes", [])) or "n/a",
                     *(str(sass.get(op, "n/a")) for op in ops)])
    out += table(["kernel", "regs", "local B", "smem B", "swizzled", "mma (PTX)", *ops], body)

    ver = Counter(v["status"].split(":")[0] for v in res["verify"])
    shape = "x".join(map(str, VERIFY_SHAPE))
    out += ["## Correctness", "",
            f"At M x N x K = {shape}: "
            + ", ".join(f"{k} {c}" for k, c in sorted(ver.items())) + ".", ""]
    bad = [r for r in res["verify"] + rows + res["sweep"]
           if r["status"].startswith(("FAIL", "error"))]
    out += [f"- {r['impl']} ({r['layout']}, {r.get('n', shape)}): {r['status']}" for r in bad]

    inter = res["interference"]
    others = sorted(set(inter.get("before", []) + inter.get("after", [])))
    out += ["", "Other processes on the GPU during the run: "
            + ("none." if not others else "; ".join(others))]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shim", type=Path, default=Path("build-triton/libni_gemm_shim.so"))
    ap.add_argument("--no-shim", action="store_true", help="skip the hand-written kernels")
    ap.add_argument("--out", type=Path, default=Path("results/triton_gemm_i8.json"))
    ap.add_argument("--windows", type=int, default=5, help="timing windows per measurement")
    ap.add_argument("--quick", action="store_true", help="two sizes and four sweep configs")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    sizes = [256, 1024] if args.quick else SIZES
    sweep = [c for c in (SWEEP[:4] if args.quick else SWEEP) if c.smem_estimate() <= SMEM_LIMIT]
    lib, shim_status = (None, "skipped (--no-shim)") if args.no_shim else load_shim(args.shim)
    workspace = None
    if lib is not None:
        # cuBLAS may not allocate during graph capture, so it gets a workspace up front.
        workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        lib.ni_cublas_set_workspace(workspace.data_ptr(), workspace.numel())
    impls = build_impls(lib)
    tool = find_cuobjdump()

    res: dict = {"env": environment(shim_status, args.windows), "cuobjdump": tool,
                 "interference": {"before": gpu_processes()}, "verify": [], "rows": [],
                 "sweep": [], "best": {lay: {} for lay in LAYOUTS}, "codegen": {}}
    print(json.dumps(res["env"], indent=1))
    if res["interference"]["before"]:
        print("WARNING: other processes are using the GPU:", res["interference"]["before"])

    m, n, k = VERIFY_SHAPE
    a, b, ref = operands(m, n, k, seed=0)
    candidates = impls + [triton_impl(f"Triton {cfg.name}", cfg, lay)
                          for cfg in sweep for lay in LAYOUTS]
    for impl in candidates:
        status, _ = check(impl, a, b[impl.layout], ref)
        res["verify"].append({"impl": impl.name, "layout": impl.layout, "status": status})
        if status not in ("exact", "declined"):
            print(f"verify {impl.layout} {impl.name}: {status}")
    print(f"verified {len(candidates)} implementations at {m}x{n}x{k}")

    for n in sizes:
        a, b, ref = operands(n, n, n, seed=n)
        for impl in impls:
            r = measure(impl, a, b[impl.layout], ref, ITERS[n], args.windows)
            res["rows"].append({"impl": impl.name, "layout": impl.layout, "n": n, **r})
            print(f"{n:5d} {impl.layout} {impl.name:<36} {summary(r)}")
        for lay in LAYOUTS:
            top = None
            for cfg in sweep:
                impl = triton_impl(f"Triton {cfg.name}", cfg, lay)
                r = measure(impl, a, b[lay], ref, ITERS[n], args.windows)
                res["sweep"].append({"impl": impl.name, "layout": lay, "n": n,
                                     "cfg": cfg.name, **r})
                if r.get("graph_ms") and (top is None or r["graph_ms"] < top["graph_ms"]):
                    top = {"cfg": cfg.name, **r}
            if top:
                res["best"][lay][n] = top
                print(f"{n:5d} {lay} Triton best {top['cfg']:<24} {summary(top)}")
        del a, b, ref
        torch.cuda.empty_cache()

    # Save the timings before anything else can go wrong: codegen is a report, and a
    # failure in it should never cost the measurements.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str))

    # Codegen. Every Triton kernel here is already compiled, so these calls hit the cache.
    ir_dir = args.out.parent / "triton_ir"
    if lib is not None and tool:
        try:
            res["codegen"].update(shim_codegen(args.shim, tool, ir_dir))
        except Exception as e:
            res["codegen"]["hand-written"] = {"status": f"codegen failed: {first_line(e)}"}
    a, b, ref = operands(1024, 1024, 1024, seed=1)
    c = torch.empty(ref.shape, dtype=torch.int32, device="cuda")
    targets = [(f"Triton: {label}", cfg, "NN") for label, cfg in LADDER]
    for lay in LAYOUTS:
        top = res["best"][lay].get(max(sizes))
        if top:
            cfg = next(x for x in sweep if x.name == top["cfg"])
            targets.append((f"Triton best at {max(sizes)}, {lay}: {cfg.name}", cfg, lay))
    for i, (name, cfg, lay) in enumerate(targets):
        try:
            kernel = triton_impl(name, cfg, lay).run(a, b[lay], c)
            save = ir_dir / f"ladder{i}" if i < 2 else None
            res["codegen"][name] = triton_codegen(kernel, tool, save)
        except Exception as e:
            res["codegen"][name] = {"status": f"codegen failed: {first_line(e)}"}

    res["interference"]["after"] = gpu_processes()
    args.out.write_text(json.dumps(res, indent=1, default=str))
    md = markdown(res, sizes)
    args.out.with_suffix(".md").write_text(md)
    print("\n" + md)
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
