"""Named, range-aligned outputs for VSView and its slow.pics extension.

vsview --arg source=D:/video.mkv --arg atom=D:/atomchtools.py --arg stage=mask examples/compare_vsview.vpy
Set stage=retinex for the standalone prefilters. Review frames before uploading.
"""
from pathlib import Path
import sys
import vapoursynth as vs
from vsview.api import set_output

script = Path(__file__)
if not script.is_file():
    script = Path.cwd() / script.name
root = script.resolve().parents[1]
sys.path.insert(0, str(root / 'tests'))
from compare_implementations import LABELS, VISUAL_NAMES, limited_retinex, load_atom, load_plugins, make

if 'source' not in globals() or 'atom' not in globals():
    raise ValueError('Pass source=VIDEO and atom=ATOMCHTOOLS with --arg')
stage = globals().get('stage', 'mask')
if stage not in ('mask', 'retinex'):
    raise ValueError('stage must be mask or retinex')
core = vs.core
load_plugins(core)
atom_function = load_atom(Path(atom), core)
src = core.bs.VideoSource(source, variableformat=0, threads=4, gpu=False,
    cachemode=2, cachepath=str(root / 'build/bestsource-index'), showprogress=False)
y = src.std.ShufflePlanes(0, vs.GRAY).resize.Point(format=vs.GRAY16)


def display(clip, limited):
    expression = 'x 4096 - 56064 / 65535 * 0 max 65535 min' if limited else 'x'
    return clip.std.Expr(expression).std.SetFrameProps(_Range=0)


set_output(display(y, True), 'Source luma')
for name in VISUAL_NAMES:
    node = make(core, y, stage, name, atom_function)
    limited = stage == 'retinex' and limited_retinex(name)
    set_output(display(node, limited), LABELS[name])
