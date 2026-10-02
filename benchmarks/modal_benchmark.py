"""On-demand Modal baseline. No deployed endpoint and at most one GPU container.

From the repository root, after Modal login and creating the HF Secret:
  .venv-modal/bin/modal run --env dev benchmarks/modal_benchmark.py --action prepare
  .venv-modal/bin/modal run --env dev benchmarks/modal_benchmark.py --action baseline

The baseline remains a single-request correctness/performance control. It does
not claim to implement concurrent serving or streaming decoding.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal

if modal.is_local():
    from modal.config import config
    if config.get('environment') != 'dev':
        raise RuntimeError('This experiment uses the dev environment. Run modal with --env dev.')

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/workspace/avatarforever')
CACHE = Path('/cache')
app = modal.App('avatarforever-benchmarks')
volume = modal.Volume.from_name('avatarforever-benchmarks', create_if_missing=True)
hf_secret = modal.Secret.from_name(os.environ.get('AVATAR_MODAL_HF_SECRET', 'huggingface-secret'))
cpu_image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install('huggingface-hub==0.36.2')
             .env({'HF_XET_HIGH_PERFORMANCE': '1'}))

# Match the previous H100 environment. TorchAudio was independently installed
# there; retain that version and explicitly verify both imports during build.
gpu_image = (
    modal.Image.from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.11')
    .apt_install('git', 'ffmpeg', 'build-essential')
    .pip_install('torch==2.14.0', 'transformers==4.55.4', 'huggingface-hub==0.36.2',
                 'accelerate==1.15.0', 'safetensors==0.8.0', 'einops==0.8.2',
                 'numpy==2.4.6', 'scipy==1.17.1', 'av==18.1.0', 'pillow==12.3.0',
                 'tqdm==4.70.1', 'ninja==1.13.2', 'psutil==7.2.2')
    .pip_install('torchaudio==2.11.0', extra_options='--no-deps')
    .run_commands('python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"')
    .env({'PYTHONPATH': f'{REMOTE}:{REMOTE}/packages/ltx-core/src:{REMOTE}/packages/ltx-pipelines/src',
          'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'MAX_JOBS': '4',
          'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': '/cache/huggingface',
          'TORCHINDUCTOR_CACHE_DIR': '/cache/compiler/h100-torch214',
          'TRITON_CACHE_DIR': '/cache/triton/h100-torch214'})
)
# Explicit source/input allowlist: never upload credentials, local environments,
# checkpoint files, .git, or unrelated workspace contents.
for name in ('packages',):
    gpu_image = gpu_image.add_local_dir(ROOT / name, str(REMOTE / name),
                                      ignore=['**/__pycache__/**', '**/._*', '**/*.pyc'])
for name in ('inference.py', 'util.py'):
    gpu_image = gpu_image.add_local_file(ROOT / name, str(REMOTE / name))
for path in (ROOT / 'benchmarks' / name for name in ('latency.py', 'paired_inference.py', 'test_batch_sigma.py')):
    gpu_image = gpu_image.add_local_file(path, str(REMOTE / 'benchmarks' / path.name))
gpu_image = gpu_image.add_local_file(ROOT / 'data/jfk-american-university.ogg', '/inputs/speech.ogg')
gpu_image = gpu_image.add_local_file(ROOT / 'outputs/stage-comparison/reference.png', '/inputs/reference.png')


@app.function(image=cpu_image, secrets=[hf_secret], timeout=60, retries=0)
def check_access() -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ['HF_TOKEN']
    result = {'token_has_surrounding_whitespace': token != token.strip()}
    try:
        HfApi(token=token).whoami()
        result['token_valid'] = True
    except Exception as exc:
        response = getattr(exc, 'response', None)
        result.update(token_valid=False, authentication_status=getattr(response, 'status_code', None))
        return result
    try:
        path = hf_hub_download('google/gemma-3-12b-it-qat-q4_0-unquantized', 'config.json',
                               token=token, force_download=True)
        result.update(gemma_access=True, config_bytes=Path(path).stat().st_size)
    except Exception as exc:
        response = getattr(exc, 'response', None)
        result.update(gemma_access=False, download_status=getattr(response, 'status_code', None),
                      exception_type=type(exc).__name__)
    return result


@app.function(image=gpu_image, cpu=2, memory=8192, timeout=120, retries=0)
def check_runtime() -> dict:
    # CPU-only import check: avoid spending GPU time on packaging mistakes.
    for name in ('latency.py', 'paired_inference.py'):
        result = subprocess.run([sys.executable, str(REMOTE / 'benchmarks' / name), '--help'],
                                cwd=REMOTE, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{name} import failed:\n{result.stderr}')
    result = subprocess.run([sys.executable, str(REMOTE / 'benchmarks/test_batch_sigma.py')],
                            cwd=REMOTE, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Batch regression failed:\n{result.stderr}')
    return {'imports': 'passed', 'batch_regression': result.stderr.strip(), 'gpu_allocated': False}


@app.function(image=gpu_image, cpu=2, memory=8192, volumes={str(CACHE): volume}, timeout=300, retries=0)
def compare_latents(reference_run: str, candidate_run: str) -> dict:
    import torch
    for name in (reference_run, candidate_run):
        if Path(name).name != name or not name.startswith('h100-'):
            raise ValueError('Expected a benchmark run ID')
    reference = CACHE / 'runs' / reference_run
    candidate = CACHE / 'runs' / candidate_run
    rows = []
    for index in range(2):
        source = reference / f'alternating-{index}.pt'
        if not source.exists():
            if index:
                continue  # A single-request control covers request zero only.
            source = next(reference.glob('cache-on-measured-*-latent.pt'))
        ref = torch.load(source, map_location='cpu', weights_only=True).float()
        for mode in ('alternating', 'batched'):
            result = torch.load(candidate / f'{mode}-{index}.pt', map_location='cpu', weights_only=True).float()
            assert result.shape == ref.shape
            rows.append({'request': index, 'mode': mode, 'finite': bool(torch.isfinite(result).all()),
                         'exact': bool(torch.equal(result, ref)),
                         'relative_rms': float((result-ref).square().mean().sqrt()/ref.square().mean().sqrt()),
                         'max_abs': float((result-ref).abs().max())})
    return {'reference_run': reference_run, 'candidate_run': candidate_run, 'comparisons': rows,
            'note': 'Numerical latent comparison, not a perceptual quality score.'}


@app.function(image=cpu_image, secrets=[hf_secret], volumes={str(CACHE): volume},
              cpu=8, memory=16384, timeout=3600, max_containers=1, retries=0)
def prepare() -> dict:
    """Download weights without reserving a GPU; record resolved HF revisions."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('The Modal HF Secret must contain HF_TOKEN.')
    api = HfApi(token=token)
    checkpoint_repo = 'LetsThink/AvatarForever'
    gemma_repo = 'google/gemma-3-12b-it-qat-q4_0-unquantized'
    checkpoint_revision = api.model_info(checkpoint_repo).sha
    gemma_revision = api.model_info(gemma_repo).sha
    # Fail early for gated access, before the large public checkpoint transfer.
    hf_hub_download(gemma_repo, 'config.json', revision=gemma_revision,
                    token=token, cache_dir='/cache/hub')
    print('Downloading AvatarForever checkpoint', checkpoint_revision, flush=True)
    checkpoint = hf_hub_download(checkpoint_repo, 'avatarforever-ltx-2.3-22b.safetensors',
                                 revision=checkpoint_revision, token=token, cache_dir='/cache/hub')
    volume.commit()
    print('Checkpoint complete; downloading Gemma', gemma_revision, flush=True)
    gemma = snapshot_download(gemma_repo, revision=gemma_revision, token=token, cache_dir='/cache/hub')
    record = {'checkpoint': checkpoint, 'gemma_root': gemma,
              'checkpoint_revision': checkpoint_revision, 'gemma_revision': gemma_revision}
    (CACHE / 'weights.json').write_text(json.dumps(record, indent=2) + '\n')
    volume.commit()
    return record


@app.function(image=gpu_image, gpu='H100', cpu=8, memory=131072,
              volumes={str(CACHE): volume}, timeout=1800, startup_timeout=1800,
              max_containers=1, scaledown_window=2, retries=0)
def baseline(run_id: str, compile_mode: str = 'regional', frames: int = 257,
             benchmark: str = 'baseline', quantization: str = 'fp8-cast') -> dict:
    """Run real inference, including VAE and MP4; persist even failed run logs."""
    weights = json.loads((CACHE / 'weights.json').read_text())
    output = CACHE / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    script = 'paired_inference.py' if benchmark == 'paired' else 'latency.py'
    command = [sys.executable, str(REMOTE / 'benchmarks' / script),
               '--checkpoint', weights['checkpoint'], '--gemma-root', weights['gemma_root'],
               '--audio', '/inputs/speech.ogg', '--reference', '/inputs/reference.png',
               '--quantization', 'fp8-cast' if quantization == 'fp8-preexpanded' else quantization,
               '--frames', str(frames), '--runs', '3',
               '--output-dir', str(output)]
    if quantization == 'fp8-preexpanded':
        command.append('--preexpand-fp8')
    if benchmark == 'baseline':
        command += ['--compile-transformer', compile_mode, '--warmup-runs', '1',
                    '--cache', 'on', '--fast-infer', '--save-latents']
        if compile_mode != 'none':
            command.append('--compile-video-decoder')
    (output / 'weights.json').write_text(json.dumps(weights, indent=2) + '\n')
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'environment.txt').write_text(freeze.stdout)
    try:
        with (output / 'run.log').open('w') as log:
            with subprocess.Popen(command, cwd=REMOTE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                status = process.wait()
        if status:
            raise RuntimeError(f'Inference failed with exit {status}; see runs/{run_id}/run.log')
        results = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
        if any('error' in row for row in results):
            raise RuntimeError(f'Inference recorded errors; see runs/{run_id}/results.jsonl')
        return {'run_id': run_id, 'summary': json.loads((output / 'summary.json').read_text())}
    finally:
        volume.commit()


@app.local_entrypoint()
def main(action: str = 'prepare', compile_mode: str = 'regional', frames: int = 257,
         quantization: str = 'fp8-cast', reference_run: str = '', candidate_run: str = ''):
    if action == 'compare':
        result = compare_latents.remote(reference_run, candidate_run)
        target = ROOT / 'outputs/modal' / candidate_run
        target.mkdir(parents=True, exist_ok=True)
        (target / f'comparison-with-{reference_run}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    elif action == 'check-runtime':
        print(json.dumps(check_runtime.remote(), indent=2))
    elif action == 'check-access':
        print(json.dumps(check_access.remote(), indent=2))
    elif action == 'prepare':
        print(json.dumps(prepare.remote(), indent=2))
    elif action in ('baseline', 'paired'):
        if quantization not in ('none', 'fp8-cast', 'fp8-dynamic', 'fp8-preexpanded'):
            raise ValueError('Unsupported quantization')
        if quantization == 'fp8-preexpanded' and action != 'paired':
            raise ValueError('Preexpanded weights are currently supported by the paired benchmark only')
        if compile_mode not in ('none', 'regional'):
            raise ValueError('compile_mode must be none or regional')
        if frames < 257 or (frames - 1) % 8:
            raise ValueError('Use at least 257 frames and an 8n+1 frame count')
        run_id = f'h100-{action}-{quantization}-{compile_mode}-{uuid.uuid4().hex[:12]}'
        # Reserve both startup and execution limits, including failed calls.
        # This ledger covers launches through this entrypoint, not other team jobs.
        import fcntl

        budget = ROOT / 'outputs/modal/h100-budget.json'
        budget.parent.mkdir(parents=True, exist_ok=True)
        with budget.open('a+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            data = json.loads(file.read() or '{"reservations": []}')
            reservation = 3602  # 1800 startup + 1800 execution + 2 idle
            if sum(r['reserved_seconds'] for r in data['reservations']) + reservation > 5 * 3600:
                raise RuntimeError('Five H100-hour experiment cap reached; inspect budget before more runs.')
            data['reservations'].append({'run_id': run_id, 'reserved_seconds': reservation,
                                         'created_at_unix': time.time()})
            file.seek(0)
            file.truncate()
            json.dump(data, file, indent=2)
            file.flush()
        print(f'RUN_ID={run_id}', flush=True)
        result = baseline.remote(run_id, compile_mode, frames, action, quantization)
        target = ROOT / 'outputs/modal' / run_id
        target.mkdir(parents=True, exist_ok=True)
        # Pull all artifacts, including actual videos, without keeping a GPU alive.
        for entry in volume.listdir(f'runs/{run_id}', recursive=True):
            if entry.type == modal.volume.FileEntryType.FILE:
                relative = Path(entry.path).relative_to(f'runs/{run_id}')
                dest = target / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('wb') as file:
                    for chunk in volume.read_file(entry.path):
                        file.write(chunk)
        print(json.dumps(result, indent=2))
        print(f'Local results: {target}')
    else:
        raise ValueError('action must be check-runtime, check-access, prepare, baseline, or paired')
