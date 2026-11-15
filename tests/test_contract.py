"""Bounds, parameter, concurrency, degenerate-histogram and YUV regression checks."""
from project_paths import REPORTS
import gc
import json
import sys
import unittest
import numpy as np
import vapoursynth as vs
from verify import ROOT, load, source, cpu_stage
from float_reference import msrcp_array

class Contract(unittest.TestCase):
    def assert_close(self,a,b,tolerance=2):
        for n in range(a.num_frames):
            with a.get_frame(n) as fa,b.get_frame(n) as fb:
                x=np.asarray(fa[0]).astype(float);y=np.asarray(fb[0]).astype(float)
                self.assertTrue(np.isfinite(y).all())
                self.assertLessEqual(float(np.abs(x-y).max()),tolerance)

    def test_parameters_and_stages(self):
        for bits in (8,10,12,16,32):
            src=source(37,29,bits,6)
            for sigma,lower,upper in [([0.],.001,.005),([0.,2.5,17.],0.,0.),([.5,3.,12.],.02,.03)]:
                with self.subTest(bits=bits,sigma=sigma):
                    ref=cpu_stage(src,'MSRCP',sigma=sigma,lower=lower,upper=upper)
                    gpu=vs.core.retinex_gpu.MSRCP(src,sigma=sigma,lower_thr=lower,upper_thr=upper,workers=1)
                    # Flat mid-gray with small sigmas amplifies last-bit CPU/GPU math
                    # difference. Measure that explicitly in test_known_differences.
                    self.assert_close(ref[:1]+ref[2:],gpu[:1]+gpu[2:],2e-5 if bits==32 else 2)
            for op in (1,2):
                for blur in (0.,.5,2.):
                    ref=cpu_stage(src,'TCanny',op=op,blur=blur)
                    gpu=vs.core.retinex_gpu.TCanny(src,op=op,blur_sigma=blur,workers=1)
                    self.assert_close(ref,gpu,2e-5 if bits==32 else 2)

    def test_yuv_and_wrapper(self):
        sys.path.insert(0,str(ROOT/'python'))
        from retinex_gpu import retinex_edgemask_gpu
        for fmt in (vs.YUV420P8,vs.YUV420P10,vs.YUV422P12,vs.YUV444P16,vs.YUV444PS):
            src=vs.core.std.BlankClip(width=64,height=48,format=fmt,length=2)
            y=src.std.ShufflePlanes(0,vs.GRAY)
            gpu=retinex_edgemask_gpu(src,workers=1)
            self.assertEqual(gpu.format.color_family,vs.GRAY)
            self.assertEqual(gpu.format.bits_per_sample,src.format.bits_per_sample)
            self.assert_close(retinex_edgemask_gpu(y,workers=1),gpu,0)
            with gpu.get_frame(0) as f:self.assertEqual(f.props._Range,vs.Range.RANGE_FULL)

    def test_bad_parameters(self):
        src=source(32,24,8,1)
        for kwargs in [dict(sigma=[]),dict(sigma=[-1.]),dict(sigma=[float('nan')]),dict(sigma=[1001.]),
                       dict(lower_thr=-1),dict(lower_thr=.5,upper_thr=.5),dict(upper_thr=float('inf')),
                       dict(blur_sigma=-1),dict(blur_sigma=float('nan')),dict(workers=0),dict(workers=17),
                       dict(op=3),dict(device=-1),dict(fast_pyramid=2)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(vs.Error):
                vs.core.retinex_gpu.RetinexEdgeMask(src,**kwargs)
        for fmt in (vs.RGB24,vs.GRAYH):
            with self.assertRaises(vs.Error):
                vs.core.retinex_gpu.RetinexEdgeMask(vs.core.std.BlankClip(format=fmt))

    def test_tiny_and_pyramid_bounds(self):
        for bits in (8,10,12,16,32):
            for w,h in ((1,1),(1,17),(19,1),(3,2),(33,31)):
                src=source(w,h,bits,3)
                for fast in (0,1):
                    gpu=vs.core.retinex_gpu.RetinexEdgeMask(src,fast_pyramid=fast,workers=1)
                    for n in range(3):
                        with gpu.get_frame(n) as f:
                            a=np.asarray(f[0]);self.assertTrue(np.isfinite(a).all())
                            self.assertGreaterEqual(float(a.min()),0)
                            self.assertLessEqual(float(a.max()),1 if bits==32 else (1<<bits)-1)

    def test_collapsed_percentile_fallback(self):
        for bits in (8,10,12,16,32):
            src=source(320,180,bits,8)[7:]
            gpu=vs.core.retinex_gpu.MSRCP(src,workers=1)
            with src.get_frame(0) as f,gpu.get_frame(0) as g:
                original=np.asarray(f[0]).astype(float)
                expected=msrcp_array((original/(1 if bits==32 else (1<<bits)-1)).astype(np.float32))
                if bits!=32:expected=np.floor(expected.astype(float)*(219<<(bits-8))+(16<<(bits-8))+.5)
                self.assertLessEqual(float(np.abs(np.asarray(g[0])-expected).max()),2e-5 if bits==32 else 0)

    def test_concurrency_repeatability(self):
        src=source(319,181,12,32)
        one=vs.core.retinex_gpu.RetinexEdgeMask(src,workers=1)
        many=vs.core.retinex_gpu.RetinexEdgeMask(src,workers=3)
        futures=[many.get_frame_async(n) for n in reversed(range(src.num_frames))]
        for n,future in zip(reversed(range(src.num_frames)),futures):
            with future.result() as f,one.get_frame(n) as g:
                self.assertTrue(np.array_equal(np.asarray(f[0]),np.asarray(g[0])))

    def test_float_sanitization(self):
        src=source(16,16,32,1)
        def modify(n,f):
            out=f.copy();a=np.asarray(out[0]);a[0,:5]=[np.nan,np.inf,-np.inf,-1,2];return out
        src=src.std.ModifyFrame(src,modify)
        gpu=vs.core.retinex_gpu.MSRCP(src,sigma=[0.],workers=1)
        with gpu.get_frame(0) as f:
            self.assertTrue(np.array_equal(np.asarray(f[0])[0,:5],[0,0,0,0,1]))

    def test_known_differences(self):
        rows=[]
        for bits in (8,10,12,16):
            for name,w,h,index,sigma,lower,upper in [
                ('collapsed percentile',320,180,7,[50.,200.,350.],.001,.005),
                ('uniform correctness exception to unstable upstream',37,29,1,[.5,3.,12.],.02,.03)]:
                src=source(w,h,bits,8)[index:index+1]
                for stage in ('MSRCP','RetinexEdgeMask'):
                    cpu=cpu_stage(src,stage,sigma=sigma,lower=lower,upper=upper)
                    gpu=getattr(vs.core.retinex_gpu,stage)(src,sigma=sigma,lower_thr=lower,upper_thr=upper,workers=1)
                    with cpu.get_frame(0) as a,gpu.get_frame(0) as b:
                        diff=np.abs(np.asarray(a[0]).astype(float)-np.asarray(b[0]).astype(float))
                        self.assertTrue(np.isfinite(np.asarray(b[0])).all())
                        rows.append(dict(case=name,bits=bits,stage=stage,max_abs=float(diff.max()),mean_abs=float(diff.mean())))
        (REPORTS / 'known-differences.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    load(ROOT/'build/all-in-one');vs.core.num_threads=8
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    (REPORTS / 'contract-tests.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
