#!/usr/bin/env bash
# Run from the A100 checkout. Each mode uses a fresh process/output directory.
# No --fast-infer: release prompt models before DiT, then DiT before decoding.
set -euo pipefail
cd "$(dirname "$0")/.."

mode=${1:-regional}
case "$mode" in
  eager) compile_args=() ;;
  regional|autotune) compile_args=(--compile-transformer regional --compile-video-decoder) ;;
  *) echo 'Usage: benchmarks/run_a100_compile.sh eager|regional|autotune' >&2; exit 2 ;;
esac

out="outputs/a100-compile/${mode}-matched"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out. Preserve it and choose a fresh experiment directory." >&2
  exit 1
fi
mkdir -p "$out"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/a100-compile/inductor"
export TORCH_LOGS=graph_breaks,recompiles
# Set explicitly so a caller's environment cannot change the comparison.
export TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=0
# A100 probe: Triton 3D-convolution candidates were much slower than cuDNN,
# with tens of seconds of search per convolution. Still compile the entire
# decoder graph, but retain the ATen/cuDNN convolution implementation.
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN
if [[ "$mode" == autotune ]]; then
  export TORCHINDUCTOR_MAX_AUTOTUNE=1
fi
printf '%s\n' \
  "mode=$mode" \
  "TORCHINDUCTOR_MAX_AUTOTUNE=$TORCHINDUCTOR_MAX_AUTOTUNE" \
  "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=$TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS" \
  "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR" \
  'Sequential model loading; one full warmup and three measured requests.' \
  > "$out/launcher-config.txt"

.venv/bin/python benchmarks/latency.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio outputs/a100-setup/jfk.flac \
  --reference outputs/a100-setup/reference.png \
  --quantization fp8-cast "${compile_args[@]}" \
  --frames 257 --runs 3 --warmup-runs 1 --cache on --save-latents \
  --output-dir "$out"
