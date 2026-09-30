#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?Usage: run_a100_profile.sh eager|compiled}
case "$mode" in
  eager) compile_args=() ;;
  compiled) compile_args=(--compile-transformer regional --compile-video-decoder) ;;
  *) exit 2 ;;
esac
out="outputs/a100-profile/$mode"
if [[ -e "$out" ]]; then
  echo "Preserve existing results and choose a fresh output path: $out" >&2
  exit 1
fi
mkdir -p "$out"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/a100-compile/inductor"
export TORCH_LOGS=graph_breaks,recompiles
export TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=0
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN
export TORCHINDUCTOR_CUDAGRAPHS=0
.venv/bin/python benchmarks/profile_launch_overhead.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio outputs/a100-setup/jfk.flac --reference outputs/a100-setup/reference.png \
  --quantization fp8-cast "${compile_args[@]}" \
  --frames 257 --runs 4 --warmup-runs 1 --cache on --output-dir "$out"
