#!/usr/bin/env bash
# Build the GEMM shim and run tools/triton_gemm.py on a CUDA machine.
#
#   bash tools/triton_gemm_vm.sh            # full run, around ten minutes on a 4090
#   bash tools/triton_gemm_vm.sh --quick    # two sizes, to check the setup first
#
# PYTHON picks the interpreter; otherwise the first venv under $HOME that can
# import torch and triton with a GPU visible is used. NVCC picks the compiler for
# the shim (default /usr/bin/nvcc) and CUDA_ARCH its target (default 89).
set -euo pipefail
cd "$(dirname "$0")/.."

py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for cand in "$HOME"/cuda-env/bin/python "$HOME"/.venv/bin/python \
              "$HOME"/*/.venv/bin/python "$HOME"/*-env/bin/python "$HOME"/*/venv/bin/python; do
    [[ -x "$cand" ]] || continue
    if "$cand" -c 'import torch, triton; assert torch.cuda.is_available()' 2>/dev/null; then
      py="$cand"
      break
    fi
  done
fi
if [[ -z "$py" ]]; then
  echo "No Python with torch, triton and a visible GPU found. Set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "python: $py"

echo "other processes on the GPU (timings are only clean if this is empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

extra=()
if cmake -S . -B build-triton -DCMAKE_BUILD_TYPE=Release -DNI_WITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
      -DCMAKE_CUDA_COMPILER="${NVCC:-/usr/bin/nvcc}" >/dev/null &&
   cmake --build build-triton --target ni_gemm_shim -j; then
  echo "shim: build-triton/libni_gemm_shim.so"
else
  echo "shim build failed; running without the hand-written kernels" >&2
  extra=(--no-shim)
fi

"$py" tools/triton_gemm.py ${extra[@]+"${extra[@]}"} "$@"
