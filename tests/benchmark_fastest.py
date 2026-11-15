"""Native candidate benchmark and dark-detail checks against current exact."""
from project_paths import ROOT, use_vsjet
use_vsjet()
import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import vapoursynth as vs
from compare_implementations import consume, disable_caches
from compare_quality import array, normalize, difference, summarize
from vsmasktools import retinex, dre_edgemask
from vsmasktools.spat_funcs import gauss_blur


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--dll',type=Path,default=ROOT/'dist/retinex_gpu.dll')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--skip-timing',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'build/reports/fastest-workers.json')
    args=p.parse_args()
    core=vs.core; core.num_threads=32
    core.std.LoadPlugin(str(ROOT/'dist/retinex_gpu.dll'))
    core.std.LoadPlugin(str(args.dll.resolve()),forcens='candidate',forceid='org.retinex.candidate')
    source=core.bs.VideoSource(str(args.input),gpu=False,variableformat=0,threads=4,
        cachemode=2,cachepath=str(ROOT/'build/bestsource-index'),showprogress=False)
    y=source.std.ShufflePlanes(0,vs.GRAY).resize.Point(format=vs.GRAY16)
    settings={f'cpu_{w}':dict(backend='cpu',workers=w) for w in (1,2,4,8,16)}
    settings.update({f'gpu_{w}':dict(backend='gpu',workers=w) for w in (1,2,4)})
    names=list(settings)+['vsjet_cpu','vsjet_cuda','fast_cpu','fast_gpu']
    def make(clip,stage,name):
        func='MSRCP' if stage=='retinex' else 'RetinexEdgeMask'
        if name in settings:
            return getattr(core.candidate,func)(clip,mode='fastest',fallback=False,**settings[name])
        if name.startswith('fast_'):
            return getattr(core.retinex_gpu,func)(clip,mode='fast',backend='cpu' if name.endswith('cpu') else 'gpu',fallback=False)
        backend=gauss_blur.Backend.CPU if name.endswith('cpu') else gauss_blur.Backend.CUDA
        with backend():
            return retinex(clip,sigma=[50,200,350],lower_thr=.001,upper_thr=.005,fast=True) if stage=='retinex' else dre_edgemask.RETINEX(clip,brz=0,sigmas=[50,200,350],lower_thr=.001,upper_thr=.005,fast=True)
    warm=list(range(1432,1440)); timed=[n for start in (1440,10080,30240) for n in range(start,start+20)]
    held=[y.get_frame(n) for n in warm+timed];blank=y.std.BlankClip(length=len(held))
    memory=blank.std.ModifyFrame(blank,lambda n,f:held[n])
    report=dict(source=args.input.name,candidate_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
        anchor_sha256=hashlib.sha256((ROOT/'dist/retinex_gpu.dll').read_bytes()).hexdigest(),
        trials=[],timing=[],quality=[],dark=[],summary={})
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    if not args.skip_timing:
        for stage in ('retinex','mask'):
            for trial in range(args.repeats):
                order=names[trial:]+names[:trial]
                for name in order:
                    node=make(memory,stage,name);disable_caches(node)
                    w=node[:8];t=node[8:];core.std.SetVideoCache(w,mode=0);core.std.SetVideoCache(t,mode=0)
                    start=time.perf_counter()
                    while time.perf_counter()-start<.5:consume(w,32)
                    start=time.perf_counter();frames=cycles=0
                    while cycles==0 or time.perf_counter()-start<1:
                        frames+=consume(t,32);cycles+=1
                    elapsed=time.perf_counter()-start
                    report['trials'].append(dict(stage=stage,candidate=name,repeat=trial+1,frames=frames,cycles=cycles,seconds=elapsed,fps=frames/elapsed))
                    print(stage,name,trial+1,f'{frames/elapsed:.1f} FPS',flush=True)
                    save();del node,w,t;gc.collect()
            for name in names:
                report['timing'].append(dict(stage=stage,candidate=name,fps=statistics.median(r['fps'] for r in report['trials'] if r['stage']==stage and r['candidate']==name)))
    del memory,blank
    for f in held:f.close()
    numbers=sorted(set(np.linspace(128,y.num_frames-128,24,dtype=int).tolist())|{1440,7200,10080,18720,30240})
    for stage in ('retinex','mask'):
        func='MSRCP' if stage=='retinex' else 'RetinexEdgeMask'
        ref=getattr(core.retinex_gpu,func)(y,backend='gpu',fallback=False)
        cpu=make(y,stage,'cpu_16');gpu=make(y,stage,'gpu_2')
        rows=[]
        for n in numbers:
            a=normalize(array(ref,n),stage=='retinex');b=normalize(array(gpu,n),stage=='retinex')
            c=normalize(array(cpu,n),stage=='retinex');dark=normalize(array(y,n),True)<.15
            row=dict(stage=stage,frame=n,**difference(a,b,dict(dark=dark)),cpu_gpu_max_abs=float(np.abs(b-c).max()))
            report['quality'].append(row);rows.append(row)
            if stage=='mask':
                for threshold in (.03,.122):
                    active=(a>=threshold)&dark
                    report['dark'].append(dict(frame=n,threshold=threshold,reference_pixels=int(active.sum()),
                        missed=int(np.count_nonzero(active&(b<threshold))),extra=int(np.count_nonzero(dark&(a<threshold)&(b>=threshold)))))
        report['summary'][stage]=summarize(rows)
        report['summary'][stage]['cpu_gpu_max_abs']=max(r['cpu_gpu_max_abs'] for r in rows)
        del ref,cpu,gpu;gc.collect();save()
    report['passed']=all(report['summary'][stage]['cpu_gpu_max_abs'] <= 2/(56064 if stage=='retinex' else 65535) for stage in ('retinex','mask'));save()
    print('PASSED' if report['passed'] else 'FAILED',json.dumps(report['summary'],indent=2),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':main()
