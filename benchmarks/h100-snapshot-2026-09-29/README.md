# H100 working-copy snapshot before shutdown

Captured 2026-09-29 from `/home/ubuntu/work/avatarforever` on the experiment H100.
The development branch `codex/fp8-inference-comparison` contains the complete
inference/quantization/compilation work through `647422a`. This backup branch
adds the remaining server-specific source and environment records.

The server's Git HEAD was `4dfc42b0e2dbbded4d148387d186219bd7601279`, with newer
files synced into its working tree without commits. Its dirty status therefore
does not mean the main implementation was missing from the development branch.

- `source-differences/` preserves remote source files that differed from or were
  absent in laptop commit `647422a`, including setup/run scripts and `uv.lock`.
  These are archival copies, not updates applied to the current implementation.
  Some remote README files are older than the laptop's versions.
- `remote-working-tree.patch` records the server's tracked-file changes relative
  to its original HEAD. Do not apply it on top of the newer development branch.
- `remote-source-inventory.json` records every source file's size and SHA-256.
  Files without an archived difference were byte-identical to the laptop copy.
- `environment.json` records the actual installed package versions and hardware.
  The archived lockfile/setup scripts are historical; the installed environment
  is authoritative for reproducing the measurements. Do not assume running a
  fresh `uv sync` will recreate the tested PyTorch 2.14 environment.

Videos, latents, audio inputs, raw logs, and the complete verification index are
saved on the laptop under `outputs/h100-backup-2026-09-29/`. Previously downloaded
comparison videos remain in their existing `outputs/` directories. Additional
artifacts were also copied to their original relative paths when no local file
would be overwritten. The local backup includes all unique source/result/input
files found by the audit; `verified-files.json` maps remote paths to verified
local copies. Raw logs and AppleDouble metadata are retained locally, not added
as source files in this snapshot.

The 67 GB checkpoint directory, 5.8 GB virtual environment, and approximately
2.9 GB of compiler caches were not downloaded. These are reproducible downloads
or caches. Account credentials, SSH keys and unrelated home-directory files are
not part of this snapshot. If the machine's disk is deleted, reinstall the
recorded environment and download the checkpoint/Gemma weights before resuming.
No shutdown command was issued by the backup procedure.
