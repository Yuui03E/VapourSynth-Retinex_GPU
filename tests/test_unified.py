"""Unified builds: native CPU/CUDA parity, all stages and correctness regressions."""
import concurrent.futures
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import vapoursynth as vs
from verify import ROOT, source, cpu_stage
from project_paths import REPORTS

core=vs.core
core.num_threads=8
for ns, file in [('retinex_gpu','all-in-one/retinex_gpu.dll'), ('cpu_only','cpu-only/retinex_gpu.dll'),
                 ('failure_test','testing/retinex_gpu.dll')]:
    core.std.LoadPlugin(str(ROOT/'build'/file),forcens=ns,forceid='org.retinex.test.'+ns)
for ns in ('retinex','tcanny'):
    if not hasattr(core,ns): core.std.LoadPlugin(str(ROOT/'build/all-in-one/tests/reference'/f'{ns}_reference.dll'))

def array(node,n=0):
    with node.get_frame(n) as f: return np.array(f[0]),dict(f.props)

class Unified(unittest.TestCase):
    def test_uniform_input_has_no_visible_edges(self):
        for bits, fmt, peak in ((8,vs.GRAY8,255),(10,vs.GRAY10,1023),(16,vs.GRAY16,65535),(32,vs.GRAYS,1.0)):
            for value in (0,peak/2 if bits==32 else peak//2,peak):
                src=core.std.BlankClip(width=1920,height=1080,format=fmt,length=1,color=value)
                for backend in ('cpu','gpu'):
                    mask=core.retinex_gpu.RetinexEdgeMask(src,backend=backend,fallback=False,workers=1)
                    self.assertLessEqual(float(array(mask)[0].max()),2e-6 if bits==32 else 0,(bits,value,backend))
                    ret=core.retinex_gpu.MSRCP(src,backend=backend,fallback=False,workers=1)
                    expected=value if bits==32 else int(value/peak*(219<<(bits-8))+(16<<(bits-8))+.5)
                    self.assertLessEqual(float(np.abs(array(ret)[0].astype(float)-expected).max()),2e-6 if bits==32 else 0)

    def test_tiny_blur_is_finite_identity_blur(self):
        for bits in (8,16,32):
            src=source(63,35,bits)[3:4]
            for stage in ('TCanny','RetinexEdgeMask'):
                for mode in ('exact','fast'):
                    for backend in ('cpu','gpu'):
                        zero=getattr(core.retinex_gpu,stage)(src,mode=mode,backend=backend,fallback=False,blur_sigma=0,workers=1,sigma=[0])
                        expected=array(zero)[0]
                        for blur in (1e-30,1e-300):
                            actual=array(getattr(core.retinex_gpu,stage)(src,mode=mode,backend=backend,fallback=False,blur_sigma=blur,workers=1,sigma=[0]))[0]
                            self.assertTrue(np.isfinite(actual).all())
                            self.assertLessEqual(float(np.abs(actual.astype(float)-expected).max()),2e-6 if bits==32 else 0)

    def test_same_interface_and_cpu_only(self):
        self.assertEqual([(f.name,f.signature) for f in core.retinex_gpu.functions()],[(f.name,f.signature) for f in core.cpu_only.functions()])
        for bits in (8,10,12,16,32):
            src=source(37,23,bits)
            for stage in ('MSRCP','Kirsch','TCanny','RetinexEdgeMask'):
                for mode in ('exact','fast'):
                    a=getattr(core.retinex_gpu,stage)(src,mode=mode,backend='cpu')
                    b=getattr(core.cpu_only,stage)(src,mode=mode)
                    for n in range(8):
                        np.testing.assert_array_equal(array(a,n)[0],array(b,n)[0])
                        self.assertEqual(array(b,n)[1]['RetinexBackend'],'cpu')

    def test_exact_cpu_against_independent_references(self):
        for bits in (8,10,12,16,32):
            src=source(43,31,bits)
            for stage in ('MSRCP','Kirsch','TCanny','RetinexEdgeMask'):
                for options in (dict(),dict(sigma=[0.,2.5,17.],lower_thr=0.,upper_thr=0.,blur_sigma=0.,op=2)):
                    a=getattr(core.cpu_only,stage)(src,**options)
                    ref=cpu_stage(src,stage,sigma=options.get('sigma',[50.,200.,350.]),
                        lower=options.get('lower_thr',.001),upper=options.get('upper_thr',.005),
                        blur=options.get('blur_sigma',1.),op=options.get('op',1))
                    # Flat and impulse histograms are known reference exceptions.
                    for n in (0,2,3,4,5,6):
                        aa=array(a,n)[0]; bb=array(ref,n)[0]
                        diff=float(np.abs(aa.astype(float)-bb).max())
                        self.assertLessEqual(diff,2e-5 if bits==32 else 2,(bits,stage,options,n,diff))

    def test_exact_cpu_cuda_parity_and_borders(self):
        for w,h in ((1,1),(1,17),(19,1),(3,2),(63,35)):
            for bits in (8,10,12,16,32):
                src=source(w,h,bits)
                for stage in ('MSRCP','Kirsch','TCanny','RetinexEdgeMask'):
                    for options in (dict(),dict(sigma=[0]),dict(fast_pyramid=1),
                                    dict(sigma=[0,2.5,17,80],blur_sigma=.5,op=2)):
                        a=getattr(core.retinex_gpu,stage)(src,backend='cpu',workers=1,**options)
                        b=getattr(core.retinex_gpu,stage)(src,backend='gpu',fallback=False,workers=1,**options)
                        for n in range(8):
                            aa=array(a,n)[0]; bb=array(b,n)[0]; diff=float(np.abs(aa.astype(float)-bb).max())
                            self.assertTrue(np.isfinite(aa).all())
                            self.assertGreaterEqual(float(aa.min()),0)
                            self.assertLessEqual(float(aa.max()),1 if bits==32 else (1<<bits)-1)
                            self.assertLessEqual(diff,2e-5 if bits==32 else 2,(w,h,bits,stage,options,n,diff))

    def test_fallback_and_concurrency(self):
        src=source(129,65,16,24)
        for mode in ('exact','fast'):
            cpu=core.retinex_gpu.RetinexEdgeMask(src,mode=mode,backend='cpu',workers=1)
            auto=core.retinex_gpu.RetinexEdgeMask(src,mode=mode,device=9999,workers=1)
            aa,props=array(auto,3); np.testing.assert_array_equal(aa,array(cpu,3)[0])
            self.assertEqual(props['RetinexBackend'],'cpu'); self.assertTrue(props['RetinexFallback'])
            with self.assertRaises(vs.Error): core.retinex_gpu.MSRCP(src,mode=mode,device=9999,fallback=False)
            with self.assertRaises(vs.Error): core.cpu_only.MSRCP(src,mode=mode,backend='gpu',fallback=False)
            node=core.failure_test.RetinexEdgeMask(src,mode=mode,workers=1,test_fail_after=1)
            core.std.SetVideoCache(node,mode=0)
            self.assertEqual(array(node,3)[1]['RetinexBackend'],'gpu')
            for n in (4,2,7):
                aa,props=array(node,n); np.testing.assert_array_equal(aa,array(cpu,n)[0])
                self.assertEqual(props['RetinexBackend'],'cpu'); self.assertIn('simulated',props['RetinexFallback'])
        many=core.cpu_only.RetinexEdgeMask(src,workers=3)
        order=[23,1,8,16,2,7,13,0]*3
        with concurrent.futures.ThreadPoolExecutor(8) as ex: actual=list(ex.map(lambda n:array(many,n)[0],order))
        for n,aa in zip(order,actual): np.testing.assert_array_equal(aa,array(core.cpu_only.RetinexEdgeMask(src,workers=1),n)[0])

    def test_validation_sanitization_yuv(self):
        src=source(16,8,32,1)
        def fill(n,f):
            out=f.copy(); np.asarray(out[0])[:]=np.resize(np.array([np.nan,np.inf,-np.inf,-1,2,.3,.8],np.float32),(8,16)); return out
        src=src.std.ModifyFrame(src,fill)
        for stage in ('MSRCP','Kirsch','TCanny','RetinexEdgeMask'):
            for backend in ('cpu','gpu'):
                a=getattr(core.retinex_gpu,stage)(src,backend=backend,sigma=[0])
                self.assertTrue(np.isfinite(array(a)[0]).all())
        for options in (dict(mode='invalid'),dict(backend='invalid'),dict(mode='exact',downscale=8),
                        dict(mode='fast',fast_pyramid=1),dict(mode='fast',workers=2),dict(mode='exact',max_cpu_mb=1)):
            with self.assertRaises(vs.Error): core.retinex_gpu.MSRCP(src,**options)
        yuv=core.std.BlankClip(width=64,height=48,format=vs.YUV420P10,length=1).std.SetFrameProps(_Range=0,_ChromaLocation=0)
        for plugin in (core.retinex_gpu,core.cpu_only):
            for mode in ('exact','fast'):
                node=plugin.MSRCP(yuv,backend='cpu',mode=mode,sigma=[0])
                self.assertEqual(node.format.color_family,vs.GRAY)
                self.assertEqual(array(node)[1]['_Range'],0)
                self.assertNotIn('_ChromaLocation',array(node)[1])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Unified))
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        sha256={kind:hashlib.sha256((ROOT/'build'/kind/'retinex_gpu.dll').read_bytes()).hexdigest()
                for kind in ('all-in-one','cpu-only','testing')})
    (REPORTS/'unified-contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
