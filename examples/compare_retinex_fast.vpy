"""BestSource comparison. Select ONE output at the bottom, or pass --arg mode=...

vspipe --info examples/compare_retinex_fast.vpy -
vspipe --arg mode=gpu --requests 2 --start 100 --end 399 examples/compare_retinex_fast.vpy --
vspipe --arg mode=cpu --requests 2 --start 100 --end 399 examples/compare_retinex_fast.vpy --
vspipe --arg mode=exact_gpu --requests 2 --start 100 --end 399 examples/compare_retinex_fast.vpy --
vspipe --arg mode=fastest_cpu --arg threads=32 --requests 32 --start 100 --end 399 examples/compare_retinex_fast.vpy --
vspipe --arg mode=fastest_gpu --requests 8 --start 100 --end 399 examples/compare_retinex_fast.vpy --

Add --arg stage=MSRCP to compare Retinex alone instead of complete edge masks.
Add --arg source="D:\\your_video.mkv" to use another video.
Strict GPU benchmark mode reports unavailable GPUs instead of silently timing CPU.
Use mode=auto in a heavy filtering script to enable CPU fallback.
"""
from pathlib import Path
import vapoursynth as vs

core = vs.core
core.num_threads = int(globals().get('threads', 8))
script = Path(__file__)
if not script.is_file():
    script = Path.cwd() / script.name
project = script.resolve().parents[1]
SOURCE = globals().get('source', 'input.mkv')
STAGE = globals().get('stage', 'RetinexEdgeMask')
cache = project / 'build/bestsource-index'
cache.mkdir(parents=True, exist_ok=True)
src = core.bs.VideoSource(SOURCE, variableformat=0, threads=2, gpu=False,
                         cachemode=2, cachepath=str(cache), showprogress=False)
# All candidates receive exactly the same 16-bit luma.
src = src.std.ShufflePlanes(0, vs.GRAY).resize.Point(format=vs.GRAY16)

def fast(clip, backend='auto'):
    if not hasattr(core, 'retinex_gpu'):
        dll = Path(globals().get('fast_dll', project / 'dist/retinex_gpu.dll'))
        core.std.LoadPlugin(str(dll if dll.is_absolute() else project / dll))
    return getattr(core.retinex_gpu, STAGE)(
        clip, mode="fast", backend=backend, fallback=backend != 'gpu', sigma=[50, 200, 350],
        downscale=int(globals().get('downscale', 8)),
        max_gpu_mb=int(globals().get('max_gpu_mb', 128)),
        reserve_gpu_mb=int(globals().get('reserve_gpu_mb', 512)),
        max_cpu_mb=int(globals().get('max_cpu_mb', 128)),
    )

def exact_gpu(clip):
    if not hasattr(core, 'retinex_gpu'):
        core.std.LoadPlugin(str(project / 'dist/retinex_gpu.dll'))
    return getattr(core.retinex_gpu, STAGE)(clip, mode='exact', backend='gpu', fallback=False, sigma=[50, 200, 350], workers=2)

def fastest(clip, backend='auto'):
    if not hasattr(core, 'retinex_gpu'):
        core.std.LoadPlugin(str(project / 'dist/retinex_gpu.dll'))
    return getattr(core.retinex_gpu, STAGE)(clip, mode='fastest', backend=backend, fallback=backend != 'gpu')

if 'mode' in globals():
    choices = {'gpu': lambda c: fast(c, 'gpu'), 'cpu': lambda c: fast(c, 'cpu'),
               'auto': fast, 'exact_gpu': exact_gpu, 'source': lambda c: c,
               'fastest_cpu': lambda c: fastest(c, 'cpu'),
               'fastest_gpu': lambda c: fastest(c, 'gpu'), 'fastest_auto': fastest}
    output = choices[mode](src)
else:
    # SELECT ONE: comment out the others.
    output = fast(src, 'auto')
    # output = fast(src, 'gpu')
    # output = fast(src, 'cpu')
    # output = exact_gpu(src)
    # output = fastest(src, 'cpu')
    # output = fastest(src, 'gpu')

core.std.SetVideoCache(output, mode=0)
output.set_output()
