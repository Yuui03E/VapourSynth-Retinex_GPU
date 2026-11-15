"""vspipe --info examples/mask.vpy -
CPU-only: add --arg dll=dist/cpu-only/retinex_gpu.dll.
"""
from pathlib import Path
import vapoursynth as vs
script = Path(__file__)
if not script.is_file():
    script = Path.cwd() / script.name
root = script.resolve().parents[1]
vs.core.std.LoadPlugin(str(root / globals().get('dll', 'dist/retinex_gpu.dll')))
clip = vs.core.std.BlankClip(width=1920, height=1080, format=vs.YUV420P10, length=24)
# Replace BlankClip with your video source filter.
vs.core.retinex_gpu.RetinexEdgeMask(
    clip, mode=globals().get('mode', 'fast'), backend=globals().get('backend', 'auto')
).set_output()
