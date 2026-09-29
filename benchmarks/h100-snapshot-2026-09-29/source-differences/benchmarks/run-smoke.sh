#!/usr/bin/env bash
cd /home/ubuntu/work/avatarforever
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u benchmarks/latency.py --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized --audio data/jfk.flac --frames 65 --runs 1 --warmup-runs 0 --cache off --output-dir outputs/benchmark-smoke
result=$?
printf '%s\n' "$result" > benchmarks/logs/smoke.exit
exit "$result"
