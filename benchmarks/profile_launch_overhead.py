"""Capture one steady AR chunk without per-forward synchronization.

Wrap the existing inference benchmark, replacing its synchronized Recorder.
Request 0 warms the complete pipeline; requests 1-3 are unprofiled controls;
request 4 captures calls 16-19 (zero based), one four-step steady AR chunk.
The only synchronization in that window is its explicitly marked final drain.
Use window-results.json as the timing record, not latency.py's generic summary.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import latency


class LaunchRecorder(latency.Recorder):
    def __init__(self):
        super().__init__()
        self.request = -1
        self.call = 0
        self.profiler = None
        self.window = None
        self.rows = []
        self.output = Path(sys.argv[sys.argv.index("--output-dir") + 1])

    def begin_window(self):
        # Drain earlier chunks before enabling CUPTI, outside the measured range.
        torch.cuda.synchronize()
        if self.request == 4:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
            )
            self.profiler.start()
            self.marker = record_function("AR_CHUNK_4_STEPS")
            self.marker.__enter__()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.wall_start = time.perf_counter()
        self.start_event.record()
        self.window = True

    def end_window(self):
        self.end_event.record()
        if self.profiler is not None:
            with record_function("WINDOW_END_DRAIN_NOT_INTERNAL_BARRIER"):
                torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self.wall_start) * 1000
        event_ms = self.start_event.elapsed_time(self.end_event)
        profiled = self.profiler is not None
        if profiled:
            self.marker.__exit__(None, None, None)
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output / "trace.json"))
            (self.output / "operator-table.txt").write_text(
                self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
            )
            self.profiler = None
        self.rows.append({
            "request": self.request, "warmup": self.request == 0,
            "profiled": profiled, "forward_indices": [16, 17, 18, 19],
            "window_cuda_event_ms": event_ms, "window_wall_ms": wall_ms,
        })
        (self.output / "window-results.json").write_text(json.dumps({
            "protocol": "One full warmup; three unprofiled controls; one profiled request. "
                        "Calls 16-19: four denoising steps of a steady AR chunk. "
                        "Synchronization only before and after the window, never between forwards. "
                        "CUDA event elapsed time includes GPU idle gaps, not just active kernels. "
                        "Profiler disables stack/shape/memory recording. "
                        "Generic latency.py DiT fields are intentionally unmeasured; use this file.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "windows": self.rows,
        }, indent=2) + "\n")
        self.window = None

    def install(self):
        original_sampling = latency.a2v.A2VidDistilledPipeline._denoise_video_only_ar
        original_forward = latency.X0Model.forward

        @functools.wraps(original_sampling)
        def sampling(*args, **kwargs):
            self.request += 1
            self.call = 0
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_sampling(*args, **kwargs)
            assert self.call > 20 and self.window is None, self.call
            torch.cuda.synchronize()
            self.samples["ar_sampling"].append(time.perf_counter() - started)
            return result

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            index = self.call
            if index == 16:
                self.begin_window()
            elif index == 20:
                self.end_window()
            self.call += 1
            if self.profiler is not None:
                phase = "populate" if index % 4 == 0 else "reuse"
                with record_function(f"DIT_STEP_{index % 4}_{phase}"):
                    return original_forward(*args, **kwargs)
            return original_forward(*args, **kwargs)

        latency.a2v.A2VidDistilledPipeline._denoise_video_only_ar = sampling
        latency.X0Model.forward = forward


if __name__ == "__main__":
    # Fixed request layout is intentional: never accidentally profile compilation.
    for option, expected in (("--warmup-runs", "1"), ("--runs", "4"), ("--cache", "on")):
        if option not in sys.argv or sys.argv[sys.argv.index(option) + 1] != expected:
            raise SystemExit(f"This experiment requires {option} {expected}")
    latency.Recorder = LaunchRecorder
    latency.main()
