#!/usr/bin/env bash
cd /home/ubuntu/work/avatarforever
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u benchmarks/latency.py --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized --audio data/jfk.flac --frames 257 --runs 3 --warmup-runs 1 --cache both --fast-infer --output-dir outputs/benchmark-warm-257
result=$?
printf '%s\n' "$result" > benchmarks/logs/warm.exit
exit "$result"
