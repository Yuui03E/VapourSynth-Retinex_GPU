"""Fastest local-illumination contracts, both builds and dark-detail regression."""
import concurrent.futures
import hashlib
import json
import unittest
import numpy as np
import vapoursynth as vs
from verify import ROOT, source
from project_paths import REPORTS

core = vs.core
core.num_threads = 8
for ns, kind in [('retinex_gpu','all-in-one'),('cpu_only','cpu-only'),('failure_test','testing')]:
    core.std.LoadPlugin(str(ROOT/'build'/kind/'retinex_gpu.dll'),forcens=ns,forceid='org.fastest.test.'+ns)


def array(node,n=0):
    with node.get_frame(n) as f:return np.array(f[0]),dict(f.props)


class Fastest(unittest.TestCase):
    def test_formats_borders_and_cpu_only(self):
        for w,h in [(1,1),(1,17),(19,1),(3,2),(63,35),(257,129)]:
            for bits in (8,10,12,16,32):
                src=source(w,h,bits)
                for stage in ('MSRCP','RetinexEdgeMask'):
                    cpu=getattr(core.retinex_gpu,stage)(src,mode='fastest',backend='cpu',workers=1)
                    gpu=getattr(core.retinex_gpu,stage)(src,mode='fastest',backend='gpu',fallback=False,workers=1)
                    only=getattr(core.cpu_only,stage)(src,mode='fastest',workers=1)
                    for n in range(8):
                        a,props=array(cpu,n);b,_=array(gpu,n);c,_=array(only,n)
                        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
                        self.assertGreaterEqual(float(a.min()),0)
                        self.assertLessEqual(float(a.max()),1 if bits==32 else (1<<bits)-1)
                        np.testing.assert_array_equal(a,c)
                        self.assertLessEqual(float(np.abs(a.astype(float)-b).max()),2e-5 if bits==32 else 2,(w,h,bits,stage,n))
                        self.assertEqual(props['RetinexMode'],'fastest')
                        self.assertEqual(props['RetinexFastestTone'],'local-ratio')

    def test_options_bypass_and_range(self):
        for bits in (8,16,32):
            src=source(257,129,bits)[3:4]
            for stage in ('MSRCP','RetinexEdgeMask'):
                for options in (dict(sigma=[0]),dict(sigma=[0,2.5,17,80],op=2,blur_sigma=0),
                        dict(sigma=[50]*32),dict(lower_thr=0,upper_thr=0),
                        dict(lower_thr=.45,upper_thr=.45),dict(blur_sigma=1e-30),dict(blur_sigma=4)):
                    a,_=array(getattr(core.retinex_gpu,stage)(src,mode='fastest',backend='cpu',workers=1,**options))
                    b,_=array(getattr(core.retinex_gpu,stage)(src,mode='fastest',backend='gpu',fallback=False,workers=1,**options))
                    self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
                    self.assertLessEqual(float(np.abs(a.astype(float)-b).max()),2e-5 if bits==32 else 2,(bits,stage,options))
            bypass=core.retinex_gpu.MSRCP(src.std.SetFrameProps(_Range=0),mode='fastest',backend='cpu',sigma=[0])
            a,props=array(bypass);np.testing.assert_array_equal(a,array(src)[0]);self.assertEqual(props['_Range'],0)

    def test_uniform_and_nonfinite(self):
        for bits,fmt,peak in [(8,vs.GRAY8,255),(16,vs.GRAY16,65535),(32,vs.GRAYS,1.)]:
            for value in (0,peak/2,peak):
                src=core.std.BlankClip(width=1920,height=1080,format=fmt,length=1,color=value)
                for backend in ('cpu','gpu'):
                    mask=core.retinex_gpu.RetinexEdgeMask(src,mode='fastest',backend=backend,fallback=False,workers=1)
                    self.assertLessEqual(float(array(mask)[0].max()),2e-6 if bits==32 else 0)
        src=core.std.BlankClip(width=64,height=48,format=vs.GRAYS,length=1)
        def fill(n,f):
            out=f.copy();np.asarray(out[0])[:]=np.resize(np.array([np.nan,np.inf,-np.inf,-1,2,.3,.8],np.float32),(48,64));return out
        src=src.std.ModifyFrame(src,fill)
        for backend in ('cpu','gpu'):
            for stage in ('MSRCP','RetinexEdgeMask'):
                a,_=array(getattr(core.retinex_gpu,stage)(src,mode='fastest',backend=backend,fallback=False,workers=1))
                self.assertTrue(np.isfinite(a).all());self.assertGreaterEqual(a.min(),0);self.assertLessEqual(a.max(),1)

    def test_validation_defaults_and_budgets(self):
        src=source(63,35,16)[3:4]
        for options in (dict(downscale=8),dict(fast_pyramid=True),dict(workers=0),dict(workers=17),dict(max_cpu_mb=0)):
            with self.assertRaises(vs.Error):core.retinex_gpu.MSRCP(src,mode='fastest',**options)
        a,props=array(core.retinex_gpu.MSRCP(src,mode='fastest',backend='cpu'))
        self.assertEqual(props['RetinexWorkers'],8)
        np.testing.assert_array_equal(a,array(core.cpu_only.MSRCP(src,mode='fastest'))[0])
        self.assertEqual(array(core.cpu_only.MSRCP(src,mode='fastest'))[1]['RetinexWorkers'],8)
        fallback=core.retinex_gpu.MSRCP(src,mode='fastest',backend='gpu',max_gpu_mb=0,workers=1)
        np.testing.assert_array_equal(a,array(fallback)[0]);self.assertTrue(array(fallback)[1]['RetinexFallback'])
        with self.assertRaises(vs.Error):core.retinex_gpu.MSRCP(src,mode='fastest',backend='gpu',max_gpu_mb=0,fallback=False)
        with self.assertRaises(vs.Error):core.cpu_only.MSRCP(src,mode='fastest',backend='gpu',fallback=False)

    def test_runtime_failure_and_concurrency(self):
        src=source(257,129,16,32)
        node=core.failure_test.RetinexEdgeMask(src,mode='fastest',backend='gpu',workers=1,test_fail_after=1)
        core.std.SetVideoCache(node,mode=0)
        self.assertEqual(array(node,3)[1]['RetinexBackend'],'gpu')
        cpu=core.retinex_gpu.RetinexEdgeMask(src,mode='fastest',backend='cpu',workers=1)
        for n in (4,2,7):
            a,props=array(node,n);np.testing.assert_array_equal(a,array(cpu,n)[0]);self.assertEqual(props['RetinexBackend'],'cpu')
        for backend in ('cpu','gpu'):
            parallel=core.retinex_gpu.RetinexEdgeMask(src,mode='fastest',backend=backend,fallback=False,workers=4)
            core.std.SetVideoCache(parallel,mode=0)
            order=[13,2,25,4,31,7,0,17,30,6,29,18]*2
            with concurrent.futures.ThreadPoolExecutor(8) as pool:actual=list(pool.map(lambda n:array(parallel,n)[0],order))
            for n,a in zip(order,actual):
                self.assertLessEqual(float(np.abs(a.astype(float)-array(cpu,n)[0]).max()),2)

    def test_synthetic_dark_lines_beside_bright_region(self):
        w,h=1920,1080
        yy,xx=np.mgrid[:h,:w]
        base=.005+.0002*xx/w+.0001*np.sin(yy*.025)
        base[:150,:150]=.6
        line=(abs(xx-1300)<=1)&(yy>250)&(yy<900)
        image=(base+.0006*line).astype(np.float32)
        src=core.std.BlankClip(width=w,height=h,format=vs.GRAYS,length=1)
        def fill(n,f):
            out=f.copy();np.copyto(np.asarray(out[0]),image);return out
        src=src.std.ModifyFrame(src,fill)
        exact=array(core.retinex_gpu.RetinexEdgeMask(src,backend='gpu',fallback=False,workers=1))[0]
        quick=array(core.retinex_gpu.RetinexEdgeMask(src,mode='fastest',backend='gpu',fallback=False,workers=1))[0]
        roi=(abs(xx-1300)<=4)&(yy>255)&(yy<895)
        active=(exact>=.03)&roi
        self.assertGreater(int(active.sum()),100)
        recall=float(np.mean(quick[active]>=.025))
        self.assertGreaterEqual(recall,.9)
        (REPORTS/'fastest-dark-line.json').write_text(json.dumps(dict(reference_pixels=int(active.sum()),
            reference_threshold=.03,candidate_threshold=.025,recall=recall,source_line_contrast=.0006,
            bright_region_level=.6),indent=2)+'\n')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fastest))
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        sha256={kind:hashlib.sha256((ROOT/'build'/kind/'retinex_gpu.dll').read_bytes()).hexdigest() for kind in ('all-in-one','cpu-only','testing')})
    (REPORTS/'fastest-contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
