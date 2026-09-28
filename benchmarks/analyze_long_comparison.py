"""Sample continuous long runs and measure basic image statistics over time.

Image statistics are diagnostics, not perceptual-quality or lip-sync scores.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

MODES = ('none', 'fp8-cast', 'fp8-dynamic')
LABELS = ('BF16', 'FP8 storage / BF16 compute', 'FP8 matrix multiplication')
TIMES = (10, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'quality'
    out.mkdir(exist_ok=True)
    reports = {}
    samples = {}
    for mode in MODES:
        path = args.root / 'long' / mode / 'cache-on-measured-1.mp4'
        per_minute = defaultdict(list)
        previous = None
        count = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            fps = float(stream.average_rate)
            requested = {round(t*fps):t for t in TIMES}
            for frame in container.decode(video=0):
                if count in requested:
                    t = requested[count]
                    pic = frame.to_image()
                    pic.save(out/f'{mode}-{t:04d}s.png')
                    samples[mode,t] = pic.resize((256,171))
                if count % round(fps) == 0:
                    image = np.asarray(frame.to_image().resize((192,128)),dtype=np.float32)
                    gray = image.mean(axis=2)
                    stats = {'time_seconds':count/fps, 'mean_luma':float(gray.mean()),
                             'luma_std':float(gray.std()),
                             'spatial_gradient':float((np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2),
                             'change_from_previous_second':float(np.abs(gray-previous).mean()) if previous is not None else 0}
                    per_minute[int(count/fps//60)].append(stats)
                    previous = gray
                count += 1
        assert count == 31505, (mode,count)
        digest = hashlib.sha256()
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                digest.update(frame.to_ndarray().tobytes())
        reports[mode] = {'frames':count, 'fps':fps, 'duration_seconds':count/fps,
                         'decoded_audio_sha256':digest.hexdigest(), 'per_minute':{}}
        for minute, rows in per_minute.items():
            reports[mode]['per_minute'][minute] = {key:float(np.mean([r[key] for r in rows]))
                                                 for key in rows[0] if key != 'time_seconds'}
            reports[mode]['per_minute'][minute]['sample_count'] = len(rows)
        print(f'Analyzed {mode}: {count} frames',flush=True)
    assert len({r['decoded_audio_sha256'] for r in reports.values()}) == 1
    for page in range(3):
        canvas = Image.new('RGB',(768,4*199),'#151922')
        draw = ImageDraw.Draw(canvas)
        for row,t in enumerate(TIMES[page*4:page*4+4]):
            for col,(mode,label) in enumerate(zip(MODES,LABELS)):
                draw.text((col*256+5,row*199+5),f'{label} | {t//60}:{t%60:02d}',fill='white')
                canvas.paste(samples[mode,t],(col*256,row*199+24))
        canvas.save(out/f'contact-{page+1}.png')
    report = {'modes':reports,'sample_seconds':TIMES,'audio_identical':True,
              'interpretation':'Statistics describe brightness, spatial detail and motion change; they are not identity, perceptual quality or lip-sync scores. Review synchronized videos and sampled frames.'}
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
