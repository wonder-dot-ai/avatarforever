#!/usr/bin/env bash
set -eu
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/work/avatarforever
python3 -m pip install --user uv
uv sync --all-packages --python 3.11
uv run python -c 'import torch, transformers; print("torch",torch.__version__,"cuda",torch.version.cuda,"transformers",transformers.__version__); print(torch.cuda.get_device_name()); from ltx_pipelines import ARA2VidDistilledPipeline; print("pipeline import OK")'
