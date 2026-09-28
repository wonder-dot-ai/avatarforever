#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/work/avatarforever
root=outputs/fp8-long-comparison
trap 'status=$?; if (( status != 0 )); then echo "FAILED $status" > "$root/driver-status.txt"; fi' EXIT
for mode in none fp8-cast fp8-dynamic; do
    echo "RUNNING $mode" > "$root/driver-status.txt"
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python benchmarks/latency.py \
      --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
      --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
      --audio data/jfk-american-university.ogg \
      --audio-latents "$root/audio-latents.pt" \
      --reference outputs/stage-comparison/reference.png \
      --quantization "$mode" --fp8-activation-backend auto \
      --frames 31505 --runs 1 --warmup-runs 1 --warmup-frames 257 \
      --cache on --output-dir "$root/long/$mode" > "$root/long-$mode.log" 2>&1
done
echo ANALYZING > "$root/driver-status.txt"
.venv/bin/python benchmarks/analyze_long_comparison.py --root "$root" > "$root/analysis.log" 2>&1
echo COMPLETE > "$root/driver-status.txt"
