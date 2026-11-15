"""Exact native stage verification against the original CPU references."""
from __future__ import annotations
from project_paths import REPORTS
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
MATRICES = [[5,5,5,-3,0,-3,-3,-3,-3], [-3,5,5,-3,0,5,-3,-3,-3],
            [-3,-3,5,-3,0,5,-3,-3,5], [-3,-3,-3,-3,0,5,-3,5,5]]

def load(build: Path) -> None:
    core = vs.core
    core.std.LoadPlugin(str((build / 'retinex_gpu.dll').resolve()))
    for name in ('retinex', 'tcanny'):
        if not hasattr(core, name):
            core.std.LoadPlugin(str((build / 'tests/reference' / f'{name}_reference.dll').resolve()))

def source(w: int, h: int, bits: int, length: int = 8) -> vs.VideoNode:
    floating = bits == 32
    fmt = vs.core.query_video_format(vs.GRAY, vs.FLOAT if floating else vs.INTEGER, bits, 0, 0)
    blank = vs.core.std.BlankClip(width=w, height=h, format=fmt.id, length=length)
    yy, xx = np.mgrid[:h, :w]
    rng = np.random.default_rng(20261114)
    patterns = [np.zeros((h,w)), np.full((h,w),.5),
                (xx >= w//2).astype(float),
                .02 + .06*xx/max(w-1,1) + .03*np.sin(yy*.09),
                rng.random((h,w)),
                .5+.15*np.sin(xx*.025)+.14*np.cos(yy*.019)+.04*rng.random((h,w)),
                ((xx//7+yy//5)%2).astype(float),
                np.where((xx==w//2)&(yy==h//2),1.,0.)]
    peak = 1 if floating else (1<<bits)-1
    dtype = np.float32 if floating else np.uint8 if bits==8 else np.uint16
    arrays = [(np.clip(v,0,1)*peak + (0 if floating else .5)).astype(dtype) for v in patterns]
    def fill(n, f):
        out = f.copy()
        np.copyto(np.asarray(out[0]), arrays[n % len(arrays)])
        return out
    return blank.std.ModifyFrame(blank, fill)

def cpu_stage(src: vs.VideoNode, stage: str, *, sigma=(50.,200.,350.), lower=.001, upper=.005,
              blur=1., op=1) -> vs.VideoNode:
    core = vs.core
    if src.format.color_family == vs.YUV:
        src = core.std.ShufflePlanes(src, 0, vs.GRAY)
    if stage == 'Kirsch':
        peak = 1 if src.format.sample_type == vs.FLOAT else (1<<src.format.bits_per_sample)-1
        return core.std.Expr([src.std.Convolution(matrix=m, saturate=False) for m in MATRICES],
                             f'x y max z max a max {peak} min 0 max')
    if stage == 'TCanny':
        return src.tcanny.TCanny(mode=1, sigma=blur, op=op)
    if src.format.sample_type == vs.FLOAT:
        from float_reference import float_msrcp
        ret = float_msrcp(src, sigma, lower, upper)
    else:
        ret = src.retinex.MSRCP(sigma=list(sigma), lower_thr=lower, upper_thr=upper)
    if stage == 'MSRCP':
        return ret
    tc = cpu_stage(ret, 'TCanny', blur=blur, op=op).std.Minimum(coordinates=[1,0,1,0,0,1,0,1])
    peak = 1 if src.format.sample_type == vs.FLOAT else (1<<src.format.bits_per_sample)-1
    return core.std.Expr([cpu_stage(src, 'Kirsch'), tc], f'x y + {peak} min 0 max')

def compare(cpu: vs.VideoNode, gpu: vs.VideoNode, stage: str, bits: int, frames: int) -> dict:
    # Work in float with native code values: MakeDiff on integers clips large errors.
    a = cpu.std.Expr('x', format=vs.GRAYS)
    b = gpu.std.Expr('x', format=vs.GRAYS)
    diff = vs.core.std.Expr([a,b], 'x y - abs').std.PlaneStats()
    maximum = mean = 0.
    peak = 1 if bits==32 else (1<<bits)-1
    # Uniform input is covered by the correctness regression, not by the upstream
    # oracle, whose recursive roundoff can turn a constant frame into false edges.
    indices = [i for i in range(frames) if i % 8 != 1]
    for i in indices:
        with diff.get_frame(i) as f:
            maximum = max(maximum, float(f.props.PlaneStatsMax))
            mean += float(f.props.PlaneStatsAverage)
        with gpu.get_frame(i) as f:
            arr = np.asarray(f[0])
            if not np.isfinite(arr).all() or arr.min()<0 or arr.max()>peak:
                raise AssertionError(f'{stage} {bits}: invalid output range')
    tolerance = 2e-5 if bits==32 else 2
    row = dict(stage=stage, bits=bits, frames=len(indices), frame_indices=indices,
               excluded_uniform_pattern=1, max_abs=maximum, mean_abs=mean/len(indices),
               tolerance=tolerance, passed=maximum<=tolerance,
               reference='float extension' if bits==32 and stage in ('MSRCP','RetinexEdgeMask') else 'upstream')
    print(json.dumps(row), flush=True)
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/all-in-one')
    parser.add_argument('--stages',nargs='+',default=['MSRCP','Kirsch','TCanny','RetinexEdgeMask'])
    parser.add_argument('--bits',nargs='+',type=int,default=[8,10,12,16,32])
    parser.add_argument('--width',type=int,default=320)
    parser.add_argument('--height',type=int,default=180)
    parser.add_argument('--frames',type=int,default=7,
                        help='7 defined patterns; 8 adds the upstream undefined collapsed-percentile impulse')
    parser.add_argument('--output',type=Path,default=REPORTS / 'validation.json')
    args = parser.parse_args()
    load(args.build)
    vs.core.num_threads = 4
    rows=[]
    for stage in args.stages:
        for bits in args.bits:
            src=source(args.width,args.height,bits,args.frames)
            cpu=cpu_stage(src,stage)
            gpu=getattr(vs.core.retinex_gpu,stage)(src,workers=1)
            rows.append(compare(cpu,gpu,stage,bits,args.frames))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(python=platform.python_version(),vapoursynth=str(vs.__version__),
                                          width=args.width,height=args.height,results=rows),indent=2)+'\n')
    if not all(r['passed'] for r in rows):
        raise SystemExit(1)

if __name__=='__main__':
    main()
