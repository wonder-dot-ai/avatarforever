"""Validate matched videos and create sampled-frame comparisons.

Pixel difference metrics quantify agreement with the baseline, not perceptual
quality or lip-sync accuracy: the two runs can generate different valid motion.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            digest.update(frame.to_ndarray().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = (32, 96, 160, 224)
    sheet = Image.new("RGB", (768, len(sample_indices) * 280), "#151922")
    draw = ImageDraw.Draw(sheet)
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    frames = 0
    with av.open(str(args.baseline)) as left, av.open(str(args.candidate)) as right:
        lv, rv = left.streams.video[0], right.streams.video[0]
        assert (lv.width, lv.height, lv.average_rate) == (rv.width, rv.height, rv.average_rate)
        fps = float(lv.average_rate)
        for a, b in zip(left.decode(video=0), right.decode(video=0), strict=True):
            assert a.time == b.time, (a.time, b.time)
            aa = a.to_ndarray(format="rgb24")
            bb = b.to_ndarray(format="rgb24")
            diff = aa.astype(np.float32) - bb.astype(np.float32)
            squared_error += float(np.square(diff).sum(dtype=np.float64))
            absolute_error += float(np.abs(diff).sum(dtype=np.float64))
            elements += diff.size
            if frames in sample_indices:
                row = sample_indices.index(frames)
                for col, (array, label) in enumerate(((aa, "BF16"), (bb, "FP8"))):
                    image = Image.fromarray(array)
                    image.save(args.output_dir / f"{label.lower()}-frame-{frames}.png")
                    sheet.paste(image.resize((384, 256)), (col * 384, row * 280 + 24))
                    draw.text((col * 384 + 10, row * 280 + 5),
                              f"{label} | frame {frames} | {frames / fps:.2f}s", fill="white")
            frames += 1
        assert frames == 257, frames
        metadata = {"frames": frames, "width": lv.width, "height": lv.height, "fps": fps,
                    "duration_seconds": frames / fps, "video_codec": lv.codec_context.name}
    mse = squared_error / elements
    ah, bh = audio_hash(args.baseline), audio_hash(args.candidate)
    report = {
        **metadata,
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "rgb_pixel_mae_0_to_255": absolute_error / elements,
        "rgb_pixel_psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
        "baseline_decoded_audio_sha256": ah, "candidate_decoded_audio_sha256": bh,
        "decoded_audio_identical": ah == bh,
        "sample_frame_indices_zero_based": sample_indices,
        "interpretation": "Pixel metrics measure output difference, not quality or lip-sync. "
        "A quantized diffusion model can follow a different motion trajectory with the same seed.",
    }
    sheet.save(args.output_dir / "sampled-comparison.png")
    (args.output_dir / "quality-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
