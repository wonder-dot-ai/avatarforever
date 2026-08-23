# Model checkpoints

Model weights are intentionally excluded from Git. Place the required Avatar-Forever and Gemma files here:

```text
checkpoints/
├── avatarforever-ltx-2.3-22b.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/
    └── ...
```

Download [`avatarforever-ltx-2.3-22b.safetensors`](https://huggingface.co/LetsThink/AvatarForever/blob/main/avatarforever-ltx-2.3-22b.safetensors) from the [AvatarForever model repository](https://huggingface.co/LetsThink/AvatarForever).

Gemma is available from [`google/gemma-3-12b-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized). You must accept Google's Gemma license before downloading it.

See the repository [README](../README.md#model-checkpoints) for download and inference commands.
