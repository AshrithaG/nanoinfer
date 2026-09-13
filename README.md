# nanoinfer

A neural-network inference engine written from scratch in C++: own tensors, own
convolution kernels, own memory planner, own int8 quantization. No runtime
dependencies beyond a BLAS `sgemm` and a thread library.

There is also an optional CUDA backend, built separately: a hand-written int8
GEMM, the same GEMM written in Triton, and a fused fp32 convolution measured
against cuDNN, ahead on three of four edge-CNN layer shapes and 3x behind on the
fourth. The int8 result contains a correction worth reading first: the
hand-written kernel was ahead of cuBLAS up to n=1024 only as I was calling
cuBLAS. Given its preferred operand layout, cuBLAS is 1.6x to 4.5x faster at
every size, and the Triton version matches it at n=4096. See
[on the GPU](#on-the-gpu).

It targets edge-sized CNNs, the kind that run on a phone or a
microcontroller-class budget, and it is fast enough to be measured honestly
against ONNX Runtime rather than against a strawman.

![latency and thread scaling](results/figures/latency.png)

## Where it lands

Apple M4 Pro, single inference, median of 600 timed runs at the best thread
count for each configuration. Same weights and same inputs through both engines,
verified numerically identical first.

| model | naive conv | this engine | ONNX Runtime | ladder speedup | vs ORT |
|---|---|---|---|---|---|
| MNIST CNN (dense, 28x28) | 287 us | **49 us** | 48 us | 5.9x | 0.99x |
| keyword spotting (depthwise, 49x10) | 1054 us | **110 us** | 67 us | 9.5x | 0.60x |
| visual wake words (depthwise, 96x96) | 3929 us | **240 us** | 184 us | 16.4x | 0.77x |

Level with the state of the art on the dense model, 1.3–1.7x behind it on the
depthwise ones. ONNX Runtime is years of hand-tuned kernels from a team; the
interesting part is not the gap, it is knowing exactly what is on the other side
of it (see [what is still slow](#what-is-still-slow)).

## What the speedup is made of

The engine keeps its naive reference kernels compiled in and selectable, so the
before/after is a real measurement rather than a memory of one. Median latency:

| configuration | MNIST | KWS | VWW |
|---|---|---|---|
| naive nested-loop convolution, 1 thread | 912 us | 3522 us | 13974 us |
| naive, best thread count | 287 us | 1054 us | 3929 us |
| im2col + `sgemm` + fusion + arena, 1 thread | **49 us** | 198 us | 440 us |
| the same, best thread count | 49 us | **110 us** | **240 us** |

The kernel work is worth 6–18x; threads add another 1.8x on the two larger
models and nothing at all on MNIST. The individual optimizations below the
kernel switch (fusion, weight pre-transposing, the 1x1 special case, the arena)
are not separately toggleable, so I am not going to attribute a number to each
one, they are inside the 49/198/440 column together.

Two of them are worth calling out because they were not obvious:

**Pre-transposing Linear weights.** `linear` needs `w` as `[K, N]` but stores it
as `[N, K]`. Transposing inside the call costs an O(K·N) copy *per inference* , 
on MNIST that is 100k floats moved to do a 100k-MAC matmul. Hoisting it to load
time is free at runtime.

**1x1 convolutions are already a GEMM.** The patch matrix im2col would build for
a 1x1 stride-1 unpadded convolution is bit-for-bit the input tensor. Detecting
that and passing the input straight to `sgemm` removes a full copy of the
activations, and pointwise convolutions are half the layers in a
depthwise-separable network.

## Three bugs that produced plausible wrong answers

Each of these passed every test I had at the time, which is the point.

**The memory planner aliased a node's input with its own output.** Liveness
analysis frees a buffer after its last read. My reuse check accepted a buffer
whose last read was *the current node*, so a node could be handed its own input
as its output. Elementwise ops do not care. A GEMM reads its input while writing
its output, so two chained `Linear` layers silently corrupted each other , 
6.5e-7 error on a single layer, 3.5e-2 on two. Fixed by requiring the buffer to
have been released strictly before the current node, and the plan now
[asserts the invariant](src/graph.cpp) at load time for every non-in-place op.

**Asymmetric padding collapsed to one axis.** The exporter wrote a single `pad`
and `stride` attribute. The keyword-spotting stem is a 10x4 kernel with padding
(5, 1), so the width axis got the height's padding. Output shapes still matched,
because those came from PyTorch, only the values were wrong. Now both axes are
written separately, and the conv equivalence test carries that exact
configuration as a case.

**Pooling read int8 tensors through a `float*`.** `maxpool2d` and
`avgpool_global` were written before quantization existed and unconditionally
called `.f32()`. In a quantized graph they reinterpreted int8 bytes as floats,
which read like a quantization accuracy problem rather than a type error, every
int8 model was wrong (max error 2.3e-1 on MNIST). Fixing it improved int8
accuracy by 40–145x. The graph now
[validates op/dtype combinations](src/graph.cpp) at load, so an unsupported pair
is a named load failure instead of quiet garbage.

The lesson I would carry forward: for a numerical library, "the output looks
plausible" is not evidence of anything. Every fast path is tested against a
reference implementation that is too slow to ship and too simple to be wrong,
and the invariants that are easy to violate are asserted in code rather than
trusted.

## Correctness

`tools/parity.py` runs five random inputs per model through both PyTorch and the
engine and compares.

| model | max abs difference | top-1 agreement |
|---|---|---|
| MNIST CNN | 5.2e-07 | 5/5 |
| keyword spotting | 4.7e-07 | 5/5 |
| visual wake words | 4.9e-07 | 5/5 |
| MNIST CNN, int8 | 6.0e-03 | 4/5 |
| keyword spotting, int8 | 1.4e-03 | 5/5 |
| visual wake words, int8 | 1.0e-03 | 5/5 |

Float agreement is at the level of GEMM accumulation-order noise. int8 changes
the arithmetic, so it is held to prediction agreement instead.

The unit tests (25 checks, no dependencies, no network) check the fast paths
against reference implementations: im2col GEMM vs naive convolution across six
shape configurations including grouped and non-square, the NEON depthwise path
vs naive grouped convolution, fused vs separate ReLU, and single- vs
multi-threaded output equality.

## Memory planning

Activations are carved out of one arena, with offsets assigned by liveness so
buffers get recycled. There are no allocations in the forward pass after load.

| model | arena | sum of activations | saved | buffers reused |
|---|---|---|---|---|
| MNIST CNN | 64 KB | 101 KB | 37% | 5 |
| keyword spotting | 64 KB | 284 KB | 77% | 10 |
| visual wake words | 684 KB | 1441 KB | 53% | 11 |

## int8: smaller, not faster

Post-training quantization with symmetric per-output-channel weight scales and
calibrated activation ranges. Weights shrink about 4x:

| model | f32 weights | int8 weights | f32 latency | int8 latency |
|---|---|---|---|---|
| MNIST CNN | 414 KB | 105 KB | 49 us | 74 us |
| keyword spotting | 90 KB | 28 KB | 110 us | 337 us |
| visual wake words | 140 KB | 42 KB | 240 us | 1426 us |

int8 is *slower* here, and that is not a bug, it is what the hardware says. The
float path calls Accelerate's `sgemm`, which on Apple Silicon reaches the AMX
matrix coprocessor. The int8 path is my own SDOT kernel (`vdotq_s32`, 16
multiply-accumulates per instruction), which is a good scalar-to-vector win , 
early versions were 4–7x slower still, before the depthwise fast path and the
padded-stride packing, but it is a hand-written loop competing against
dedicated matrix silicon. On a target without an AMX-class unit, or where model
size is the binding constraint, the tradeoff flips. Quantization here buys 4x
smaller weights, and I would ship it for that reason, not for latency.

## What is still slow

Being specific about the remaining gap to ONNX Runtime, in the order I would
attack it:

- **Depthwise convolution is the whole gap.** The dense model is at parity; both
  models where I lose are dominated by depthwise layers. My depthwise kernel
  vectorizes across the width axis with a scalar fallback near borders, and
  processes one channel at a time. ORT blocks over channels and keeps several
  rows of accumulators live.
- **No cache blocking in the epilogue.** The bias+ReLU pass walks the whole
  output tensor again after the GEMM writes it. For layers where the output does
  not fit in L1 that is a second trip through memory; it should be tiled into the
  GEMM's N loop.
- **Thread scaling stops at about 1.8x on four cores** (right-hand chart), and
  MNIST is fastest single-threaded. Parallelism is per-op over output channels,
  so each op pays a barrier and small ops have too few channels to divide. Real
  runtimes parallelize over a fused region, not one kernel.
- **No layout choice.** Everything is NCHW because that is what the exporter
  emits. Depthwise kernels generally prefer NHWC, where the channel axis is
  contiguous and a single vector load covers 16 channels of one pixel.

Measurement caveat: run-to-run variation on an unpinned laptop is roughly ±10%,
and the ORT numbers moved by that much across runs. Differences below about 15%
in the tables above should be read as a tie.

## On the GPU

The CPU engine above is the whole project; this is a separate backend that
answers a different question. int8 is the case where quantization should pay off
on a GPU, so: how close can a hand-written int8 GEMM get to cuBLAS, and what
exactly is on the other side of the gap?

Build it with `cmake -B build-cuda -DNI_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`
and run `./build-cuda/ni_bench_cuda`. Every implementation is checked against a
CPU int32 reference on three shapes before anything is timed.

RTX 4090, sm_89, C[n,n] int32 = A[n,n] int8 * B[n,n] int8, mean of 10 to 200
back-to-back launches between two CUDA events, depending on size.

| n | naive | tiled | tiled+dp4a | wmma | wmma+smem | cuBLAS (NN) | best TOPS | vs cuBLAS (NN) |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.015 ms | 0.017 ms | 0.011 ms | **0.004 ms** | 0.006 ms | 0.009 ms | 7.7 | **2.09x** |
| 512 | 0.055 ms | 0.062 ms | 0.039 ms | **0.008 ms** | 0.010 ms | 0.010 ms | 33.1 | **1.26x** |
| 1024 | 0.423 ms | 0.485 ms | 0.303 ms | 0.044 ms | **0.024 ms** | 0.029 ms | 90.9 | **1.24x** |
| 2048 | 3.251 ms | 3.552 ms | 2.208 ms | 0.329 ms | **0.145 ms** | 0.106 ms | 118.7 | 0.73x |
| 4096 | 24.746 ms | 30.464 ms | 18.344 ms | 5.142 ms | **1.119 ms** | 0.797 ms | 122.8 | 0.71x |

Ahead of cuBLAS as called here up to 1024, and about 30% behind at 2048 and
4096. "As called here" turned out to matter; see
[against cuBLAS's fast path](#against-cublass-fast-path).

### What the ladder is made of

**Shared-memory tiling made it slower.** This is the textbook optimization and
it lost to the naive kernel at every size. Naive is accidentally well behaved
here: consecutive threads read consecutive `B` elements so those loads coalesce,
every thread in a row reads the same `A` element so that broadcasts, and the
4090's 72 MB L2 absorbs most of the reuse that shared memory was supposed to
provide. Against that, `int8` shared-memory tiles introduce 4-way bank conflicts,
because four consecutive bytes share a bank. The tiling paid a cost for a
benefit the cache was already giving away.

**`__dp4a` was worth 1.35x to 1.6x.** Four int8 multiply-accumulates into an
int32 in one instruction. `A` is contiguous along K so a thread can reinterpret
four consecutive bytes directly; `B` is not, so it is staged transposed into
shared memory to make its K axis contiguous there.

**Tensor cores were worth another 3.6x, and they were the whole story.** Every
kernel above runs on the CUDA cores. cuBLAS does not: it issues IMMA, the
integer tensor-core instruction, which is a different and much wider unit of the
chip. One IMMA is a 16x16x16 matrix multiply per warp, against four MACs per
thread for `__dp4a`. No amount of tuning on the CUDA cores closes that, which is
why the naive tensor-core kernel already beat the best CUDA-core kernel by 3.6x
while doing nothing else clever.

**Staging was worth 4.6x on top of that, and the diagnosis came first.** The
first tensor-core kernel loaded every fragment straight from global memory, so at
4096 each `A` row was re-read by all 256 blocks along N. That works out to about
8.6 GB of requested traffic against a 1008 GB/s bus, which accounted for
essentially the entire 5.1 ms runtime: the kernel was bandwidth-bound while
issuing tensor-core instructions. Staging tiles through shared memory and giving
each warp a 32x32 output tile (four accumulator fragments instead of one) made
each fetched byte feed sixteen times more math, and the runtime fell to 1.119 ms.

### What was still slow, tested

Roughly 30% behind cuBLAS at 2048 and above. I ranked the causes in the order I
would attack them, then tested the ranking instead of attacking: a Triton kernel
can be written at exactly this kernel's tiling (64x64 blocks, 32-wide K tiles, 4
warps) and changed one knob at a time. The ranking was:

1. No overlap between loading and computing (`cp.async`), "almost certainly the
   largest single item."
2. The block tile is too small.
3. No shared-memory swizzling.
4. One K-stage in flight.

B row-major as before. Microseconds per GEMM, CUDA-graph replay, all in one
process with the hand-written kernel:

| n | wmma+smem | Triton, same tiling | + 3 pipeline stages | + 128x128 tile, 8 warps | + grouped tile order | Triton best |
|---|---|---|---|---|---|---|
| 256 | 5.0 | 4.3 | 3.7 | 5.5 | 5.5 | 2.6 |
| 512 | 8.4 | 6.9 | 5.7 | 8.9 | 8.9 | 4.1 |
| 1024 | 21.1 | 15.9 | 14.2 | 16.4 | 16.4 | 9.2 |
| 2048 | 144.8 | 95.0 | 97.7 | 55.8 | 55.8 | 45.1 |
| 4096 | 1124.8 | 699.2 | 689.3 | 414.2 | 416.5 | 333.4 |

**The first item was wrong.** Three pipeline stages at this tiling changed the
runtime by 1.01x at 4096 and made it slightly slower at 2048. The compiled kernel
really does overlap loads with math (its SASS issues `cp.async` loads and the
one-stage kernel's does not), so this measures the idea, not a knob that silently
did nothing. Once tiles are staged, this kernel is not waiting on its loads.

**The second was the largest single step.** A 128x128 tile with 8 warps was worth
1.66x at 4096 and 1.75x at 2048. It is slower at 1024 and below, where there are
fewer blocks than SMs.

**The third cannot be isolated, but it sits inside the largest gap.** At identical
tiling and the same register budget (71 registers per thread against 70), the
Triton kernel is already 1.61x faster at 4096, 1.53x at 2048 and 1.33x at 1024.
Two differences show in the compiled output: its shared memory is swizzled where
mine is plain arrays, and it issues a different int8 MMA shape (m16n8k32 against
wmma's 16x16x16). That is an upper bound on what the two are worth together, not
a measurement of either.

**The fourth is the same experiment as the first, with the same answer.** Grouped
tile order, which was not on the list, was worth nothing.

The best Triton kernel (128x128 blocks, 64-wide K tiles, 4 warps, 3 stages,
grouped order) reaches 412 TOPS at 4096, 3.4x this kernel. And the thing that
decided the comparison with cuBLAS was not on the list at all.

### Against cuBLAS's fast path

`gemm_i8.cu` calls cuBLAS with B row-major (NN), which is how this backend
stores it. Int8 tensor-core GEMMs are built around both operands being
K-contiguous (TN), and an engine chooses its weight layout at load time, the
same way the CPU engine pre-transposes Linear weights. Same data in both layouts,
one process, CUDA-graph replay, microseconds:

| n | best hand-written | cuBLAS NN | cuBLAS TN | torch._int_mm TN | Triton best NN | Triton best TN |
|---|---|---|---|---|---|---|
| 256 | 3.7 (wmma) | 7.8 | 2.3 | 3.0 | 2.6 | 2.0 |
| 512 | 6.9 (wmma) | 8.1 | 2.9 | 3.7 | 4.1 | 2.8 |
| 1024 | 21.1 (wmma+smem) | 24.1 | 9.2 | 7.9 | 9.2 | 7.8 |
| 2048 | 144.8 (wmma+smem) | 105.0 | 57.5 | 50.1 | 45.1 | 34.5 |
| 4096 | 1124.8 (wmma+smem) | 857.9 | 249.9 | 251.2 | 333.4 | 254.3 |

**cuBLAS is 1.8x to 3.4x faster in TN,** and in TN it is 1.6x to 4.5x faster than
the best hand-written kernel at every size. The first table's "ahead of cuBLAS up
to 1024" is true only of cuBLAS as I called it. At 4096 cuBLAS reaches 550 TOPS.

**The layout costs cuBLAS far more than it costs Triton.** Triton's NN kernel is
about 1.3x slower than its TN kernel at 2048 and 4096, and the compiled output
shows a difference: the best NN kernel issues byte permutes and the TN kernel
issues none. cuBLAS loses 3.4x at 4096 to the same change, so its row-major path
is a slow path, more than a transposition.

**Triton, given TN, matches the vendor.** Within 2% of cuBLAS at 4096 (540 against
550 TOPS), and 1.45x faster at 2048 than the fastest vendor path there
(`torch._int_mm`, which goes through cuBLASLt), at 498 TOPS. At 1024 and below
every TN implementation is within 1.5 microseconds of the others.

### Back-to-back timing at small sizes

`ni_bench_cuda` times back-to-back launches, which includes each call's host
cost. From n=2048 up that does not matter: back-to-back and replayed times agree
within 0.6% for the hand-written kernels and within 7% for everything else. Below
that it matters a great deal. At 256, cuBLAS TN takes 12.3 microseconds
back-to-back and 2.3 replayed, so about 10 microseconds of every call is host
work, enough to make it look 1.5x slower than wmma (7.9 microseconds) when it is
1.6x faster (2.3 against 3.7). The first table's 2.09x at 256 survives against
cuBLAS NN either way (2.12x replayed), because that path is slow on the GPU
itself.

Reproduce with `bash tools/triton_gemm_vm.sh` on a CUDA machine. It builds
`bench/gemm_shim.cu`, which puts the hand-written kernels in the same process as
Triton and cuBLAS so all of them share one stream and one timer, and it writes
`results/triton_gemm_i8.md`. Every implementation must match an exact CPU
reference at every timed size before it is timed.

The bandwidth figure above is computed from launch geometry and measured time,
not read off a profiler; see [resource use and occupancy](#resource-use-and-occupancy)
for what could and could not be measured on this machine.

### Resource use and occupancy

GPU performance counters are restricted to root on the machine this was measured
on, so there is no Nsight Compute output here. Theoretical occupancy and per
kernel resource use do not need counters: `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` report them from the runtime,
and that is what the benchmark prints.

| kernel | regs/thread | shared B | block | blocks/SM | occupancy |
|---|---|---|---|---|---|
| gemm naive | 38 | 0 | 1024 | 1 | 67% |
| gemm tiled | 37 | 2048 | 1024 | 1 | 67% |
| gemm tiled+dp4a | 37 | 2048 | 1024 | 1 | 67% |
| gemm wmma | 40 | 0 | 128 | 12 | 100% |
| gemm wmma+smem | 70 | 4096 | 128 | 7 | 58% |
| conv direct | 38 | 0 | 256 | 6 | 100% |
| conv direct+fused | 38 | 0 | 256 | 6 | 100% |
| conv smem+fused | 40 | 4608 | 256 | 6 | 100% |

Three things fall out of this table.

**The fastest kernel has the lowest occupancy.** `wmma+smem` runs at 58% against
`wmma` at 100%, and is 4.6x faster. The four accumulator fragments per warp that
make it fast are also what push it to 70 registers per thread, which is what
limits it to 7 blocks per SM. Occupancy is a means of hiding latency, not a
goal; here the extra work per thread hides more latency than extra threads would
have. Tuning for the occupancy number alone would have made this kernel slower.

**The CUDA-core kernels are capped at 67% by an arbitrary choice.** A 32x32 tile
is 1024 threads, and an Ada SM holds 1536, so exactly one block fits and a third
of the SM is idle by construction. A 16x16 tile would fit six blocks and reach
100%. Whether that would actually be faster is untested: those kernels are
bandwidth-bound, so more resident warps may not help. It is the first thing to
try if anyone wants to push the CUDA-core ladder further.

**The convolution kernels are all at 100%,** which locates the 3x loss to cuDNN
on the mid 3x3 somewhere other than occupancy. It is algorithmic: a direct
convolution against cuDNN's implicit GEMM, which turns the problem into a matrix
multiply and gets to reuse a tiled GEMM that is far better optimized than
anything here.

**What is out of scope.** Achieved occupancy, DRAM throughput, tensor-core
utilization and warp stall reasons all require hardware counters, and therefore
root, which was not available. The bandwidth figures quoted above are computed
from launch geometry and measured time rather than read off a profiler, and the
occupancy figures are theoretical rather than achieved. Both are stated as such
wherever they appear.

### Fused convolution against cuDNN

The CPU engine runs bias and activation as a separate pass over the output
tensor, which the [what is still slow](#what-is-still-slow) section above lists
as one of its weaknesses. On a GPU that cost is easy to isolate: the unfused
path writes the output, then reads and writes it again for bias, then a third
time for ReLU. The fused and unfused kernels here are the same template with the
epilogue switched on or off, so the difference between those two rows is only
the memory traffic.

Shapes are the kind an edge CNN is made of, not the kind a benchmark suite
usually reports. fp32, NCHW, batch 1. `./build-cuda/ni_bench_conv`.

| layer | direct | +fused | +smem | cuDNN unfused | cuDNN fused | cuDNN fused f32 | mine vs cuDNN |
|---|---|---|---|---|---|---|---|
| first 3x3, 3 to 32, 96x96 | 0.0158 | 0.0106 | **0.0096** | 0.0312 | 0.0305 | 0.0302 | **3.15x** |
| mid 3x3, 64 to 64, 56x56 | 0.1283 | 0.1189 | **0.1062** | 0.0586 | 0.0635 | **0.0355** | 0.33x |
| pointwise 1x1, 128 to 128, 28x28 | 0.0542 | 0.0490 | **0.0353** | 0.0421 | 0.0435 | 0.0431 | **1.22x** |
| depthwise 3x3, 128, 28x28 | 0.0072 | **0.0035** | n/a | 0.0219 | 0.0215 | 0.0211 | **6.02x** |

Milliseconds, mean of 50 back-to-back launches. Every implementation is checked against the
direct kernel's output before timing. These are small layers timed back-to-back,
the method that on the GEMM included about 10 microseconds of host cost per
cuBLAS call at n=256, and cuDNN has per-call host cost too. The three wins have
not yet been re-timed with host cost excluded.

**Fusion is worth most where the convolution is smallest.** 1.08x on the mid 3x3
but 2.06x on the depthwise layer. That ordering is the whole point: the
depthwise layer is only 1.8 MFLOP against the same output tensor size, so it is
almost entirely epilogue, and removing two passes over that tensor nearly halves
the runtime. The compute-heavy 3x3 hides its epilogue behind arithmetic.

**Staging the filter in shared memory is worth up to 1.39x.** Every thread in
the direct kernel re-reads the same filter from global memory; a 128-channel
pointwise layer is 64 KB of weights re-read per block. Staging it once per block
fixes that. The kernel declines depthwise layers, where each output channel has
its own 3x3 filter and there is nothing for a block to share.

**Three wins and one clear loss, and the pattern is not an accident.** The
hand-written kernels beat cuDNN on the first layer, the pointwise layer and the
depthwise layer, and lose by 3x on the conventional 64-to-64 3x3. cuDNN's
algorithms assume large batches and wide channels; on a 3-channel input or a
depthwise layer it is running machinery shaped for a workload that is not there.
On the one shape its heuristics were built for, it is three times faster than
anything here, and closing that would mean implicit GEMM with proper tiling
rather than a direct convolution.

**cuDNN's default math mode was both slower and less accurate.** On the mid 3x3,
leaving cuDNN on `CUDNN_DEFAULT_MATH` produced 7.76e-03 relative error against
our fp32 reference, because on Ampere and later it silently promotes fp32
convolution to TF32 tensor cores, which keep 10 mantissa bits instead of 23.
Forcing `CUDNN_FMA_MATH` made it exact *and* dropped it from 0.0635 ms to
0.0355 ms. The math type does not only choose the arithmetic, it changes which
internal kernel gets selected, and here the default picked a worse one. The
comparison column above uses the true-fp32 row, since that is the only vendor
result computing the same arithmetic these kernels do.

### Where the vendor library declines

cuBLAS's int8 path requires every leading dimension to be a multiple of 4,
because the IMMA instructions underneath load four bytes at a time, and it
returns `CUBLAS_STATUS_NOT_SUPPORTED` otherwise. The tensor-core kernels here
inherit stricter versions of the same constraint: 16 for `wmma`, 64 for the
staged variant. The CUDA-core kernels take any shape. This is visible in the
correctness output, where the ragged 61x76x132 case runs on three
implementations and is declined by three others.

## Building and running

Needs CMake, a C++20 compiler, and [uv](https://docs.astral.sh/uv/) for the
Python tooling.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/ni_test                        # unit tests, no dependencies

uv sync
uv run python -m tools.parity --quantized    # engine vs PyTorch, all models
uv run python -m tools.benchmark             # ladder + ONNX Runtime comparison
uv run python -m tools.plots                 # figures
```

```
include/nanoinfer/   tensor, ops, graph, thread pool headers
src/
  tensor.cpp         tensor views and the arena allocator
  ops.cpp            conv (naive / im2col+GEMM / depthwise NEON), gemm, int8 SDOT,
                     pooling, linear, activations
  graph.cpp          model loading, liveness-based memory planning, load-time
                     invariant checks, the execution loop
  pool.cpp           persistent worker pool
tools/
  models.py          the three benchmark models
  export.py          PyTorch -> .ngm, with BatchNorm folded at export time
  parity.py          engine vs PyTorch
  benchmark.py       ladder + ONNX Runtime, same protocol for both
  plots.py           figures
tests/test_main.cpp  fast paths vs reference implementations
```

The model format is deliberately boring: a text header describing tensors and
nodes, then one binary blob of weights. It is readable with `head`, diffable, and
needs no parser library on either side.

## References

- Chellapilla et al., [High Performance Convolutional Neural Networks for
  Document Processing](https://inria.hal.science/inria-00112631/document), im2col
- Jacob et al., [Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- Zhang et al., [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/abs/1711.07128), the DS-CNN model
- Chowdhery et al., [Visual Wake Words Dataset](https://arxiv.org/abs/1906.05721)
- [MLPerf Tiny](https://github.com/mlcommons/tiny) for the benchmark model choices
