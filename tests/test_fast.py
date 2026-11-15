"""Fast companion: format, parity, budget, fallback and scheduling contracts."""
from project_paths import REPORTS
import concurrent.futures
import json
from pathlib import Path
import unittest
import numpy as np
import vapoursynth as vs
from verify import ROOT, source

core = vs.core
core.num_threads = 8
from functools import partial
class FastAPI:
    def __init__(self, plugin): self.plugin = plugin
    def __getattr__(self, name): return partial(getattr(self.plugin, name), mode='fast')
core.std.LoadPlugin(str(ROOT / 'build/all-in-one/retinex_gpu.dll'))
core.std.LoadPlugin(str(ROOT / 'build/cpu-only/retinex_gpu.dll'), forcens='cpu_only', forceid='org.retinex.cpu-test')
core.std.LoadPlugin(str(ROOT / 'build/testing/retinex_gpu.dll'), forcens='failure_test', forceid='org.retinex.failure-test')
fast = FastAPI(core.retinex_gpu)
fast_cpu = FastAPI(core.cpu_only)
fast_test = FastAPI(core.failure_test)

def array(node, n=0):
    with node.get_frame(n) as frame:
        return np.array(frame[0]), dict(frame.props)

class FastContracts(unittest.TestCase):
    def assert_close(self, a, b, bits):
        aa, _ = array(a)
        bb, _ = array(b)
        self.assertLessEqual(float(np.max(np.abs(aa.astype(float)-bb))), 2e-6 if bits == 32 else 1)

    def test_formats_borders_and_patterns(self):
        for w, h in [(1, 1), (1, 9), (9, 1), (3, 2), (63, 35), (257, 129)]:
            for bits in (8, 10, 12, 16, 32):
                src = source(w, h, bits)
                for stage in ('MSRCP', 'RetinexEdgeMask'):
                    cpu = getattr(fast, stage)(src, backend='cpu')
                    gpu = getattr(fast, stage)(src, backend='gpu', fallback=False)
                    for n in range(8):
                        aa, props = array(cpu, n)
                        bb, _ = array(gpu, n)
                        self.assertTrue(np.isfinite(aa).all() and np.isfinite(bb).all())
                        self.assertGreaterEqual(float(aa.min()), 0)
                        self.assertLessEqual(float(aa.max()), 1 if bits == 32 else (1 << bits)-1)
                        self.assertLessEqual(float(np.max(np.abs(aa.astype(float)-bb))), 2e-6 if bits == 32 else 1,
                                             (w, h, bits, stage, n))
                        self.assertEqual(props['RetinexFastBackend'], 'cpu')

    def test_options_and_streaming(self):
        src = source(721, 481, 16)[3:4]
        for stage in ('MSRCP', 'RetinexEdgeMask'):
            for options in (dict(), dict(sigma=[0]), dict(sigma=[0, 1e-12, 1000]),
                            dict(sigma=[1, 8, 25, 80, 250], downscale=4), dict(sigma=[80], downscale=16),
                            dict(sigma=[50]*32), dict(blur_sigma=0), dict(blur_sigma=2.5, op=2),
                            dict(lower_thr=0, upper_thr=0), dict(lower_thr=.45, upper_thr=.45)):
                cpu = getattr(fast, stage)(src, backend='cpu', **options)
                gpu = getattr(fast, stage)(src, backend='gpu', fallback=False, **options)
                self.assert_close(cpu, gpu, 16)
            cached = getattr(fast, stage)(src, backend='cpu')
            streaming = getattr(fast, stage)(src, backend='cpu', max_cpu_mb=1)
            self.assertEqual(array(streaming)[1]['RetinexFastScoreCache'], 0)
            self.assertEqual(array(cached)[1]['RetinexFastScoreCache'], 1)
            self.assert_close(cached, streaming, 16)
        tiny = source(5, 3, 32)[3:4]
        self.assert_close(fast.RetinexEdgeMask(tiny, backend='cpu', blur_sigma=100),
                          fast.RetinexEdgeMask(tiny, backend='gpu', fallback=False, blur_sigma=100), 32)

    def test_nonfinite_and_yuv(self):
        src = source(16, 8, 32, 1)
        def fill(n, f):
            out = f.copy()
            np.asarray(out[0])[:] = np.resize(np.array([np.nan, np.inf, -np.inf, -1, 2, .3, .8], np.float32), (8, 16))
            return out
        src = src.std.ModifyFrame(src, fill)
        for stage in ('MSRCP', 'RetinexEdgeMask'):
            a = getattr(fast, stage)(src, backend='cpu')
            b = getattr(fast, stage)(src, backend='gpu', fallback=False)
            self.assertTrue(np.isfinite(array(a)[0]).all())
            self.assert_close(a, b, 32)
        def subnormal(n, f):
            out = f.copy()
            np.asarray(out[0])[:] = np.resize(np.array([0, 1e-45, 2e-45, 1e-44], np.float32), (8, 16))
            return out
        tiny_float = src.std.ModifyFrame(src, subnormal)
        a = fast.MSRCP(tiny_float, backend='cpu')
        b = fast.MSRCP(tiny_float, backend='gpu', fallback=False)
        self.assertTrue(np.isfinite(array(a)[0]).all() and np.isfinite(array(b)[0]).all())
        self.assert_close(a, b, 32)
        yuv = core.std.BlankClip(width=64, height=48, format=vs.YUV420P10, length=1)
        yuv = yuv.std.SetFrameProps(_Range=0, _ChromaLocation=0)
        out = fast.MSRCP(yuv, backend='cpu')
        self.assertEqual(out.format.color_family, vs.GRAY)
        self.assertEqual(array(out)[1]['_Range'], 0)
        self.assertNotIn('_ChromaLocation', array(out)[1])
        bypass = fast.MSRCP(yuv, backend='cpu', sigma=[0])
        self.assertEqual(array(bypass)[1]['_Range'], 0)

    def test_invalid_arguments(self):
        src = source(128, 64, 8)
        for options in (dict(sigma=[]), dict(sigma=[-1]), dict(sigma=[float('nan')]), dict(sigma=[1]*33),
                        dict(downscale=0), dict(downscale=3), dict(backend='invalid'), dict(device=-1),
                        dict(fallback=2), dict(max_cpu_mb=0), dict(max_gpu_mb=-1), dict(op=3),
                        dict(blur_sigma=float('inf')), dict(lower_thr=.5, upper_thr=.5), dict(device=2**40)):
            with self.assertRaises(vs.Error): fast.MSRCP(src, **options)
        with self.assertRaises(vs.Error): fast.MSRCP(core.std.BlankClip(format=vs.RGB24))
        with self.assertRaises(vs.Error): fast.MSRCP(source(1920, 1080, 16), backend='cpu', max_cpu_mb=1, downscale=2)

    def test_creation_fallback_and_strict_errors(self):
        src = source(129, 65, 16)[3:4]
        cpu = fast.MSRCP(src, backend='cpu')
        for options in (dict(max_gpu_mb=0), dict(reserve_gpu_mb=1048576), dict(device=9999)):
            auto = fast.MSRCP(src, **options)
            self.assertEqual(array(auto)[1]['RetinexFastBackend'], 'cpu')
            self.assertTrue(array(auto)[1]['RetinexFastFallback'])
            self.assert_close(cpu, auto, 16)
            with self.assertRaises(vs.Error): fast.MSRCP(src, fallback=False, **options)
        standalone = fast_cpu.MSRCP(src)
        self.assert_close(cpu, standalone, 16)
        self.assertEqual(array(standalone)[1]['RetinexFastGPUBytes'], 0)
        with self.assertRaises(vs.Error): fast_cpu.MSRCP(src, backend='gpu', fallback=False)
        constrained = fast.RetinexEdgeMask(source(721, 481, 16), max_cpu_mb=1)
        _, props = array(constrained, 3)
        self.assertEqual(props['RetinexFastBackend'], 'cpu')
        self.assertIn('staging', props['RetinexFastFallback'])
        self.assertLessEqual(props['RetinexFastCPUBytes'], 1024*1024)
        # Creation failures must not contaminate a subsequent valid CUDA call.
        self.assert_close(cpu, fast.MSRCP(src, backend='gpu', fallback=False), 16)

    def test_runtime_fallback_sticks_and_retries_current_frame(self):
        src = source(257, 129, 16)
        cpu = fast.RetinexEdgeMask(src, backend='cpu')
        node = fast_test.RetinexEdgeMask(src, backend='gpu', test_fail_after=1)
        core.std.SetVideoCache(node, mode=0)
        _, first = array(node, 3)
        self.assertEqual(first['RetinexFastBackend'], 'gpu')
        for n in (4, 2, 7, 5):
            aa, props = array(node, n)
            bb, _ = array(cpu, n)
            np.testing.assert_array_equal(aa, bb)
            self.assertEqual(props['RetinexFastBackend'], 'cpu')
            self.assertIn('simulated', props['RetinexFastFallback'])
            self.assertLess(props['RetinexFastCPUBytes'], first['RetinexFastCPUBytes'])
        strict = fast_test.MSRCP(src, backend='gpu', fallback=False, test_fail_after=0)
        with self.assertRaises(vs.Error): array(strict)

    def test_concurrent_out_of_order_requests(self):
        src = source(257, 131, 12, 32)
        for backend in ('cpu', 'gpu'):
            node = fast.RetinexEdgeMask(src, backend=backend, fallback=False)
            core.std.SetVideoCache(node, mode=0)
            expected = [array(node, n)[0] for n in range(8)]
            order = [13, 2, 25, 4, 31, 7, 0, 17, 30, 6, 29, 18]*2
            with concurrent.futures.ThreadPoolExecutor(8) as executor:
                actual = list(executor.map(lambda n: array(node, n)[0], order))
            for n, aa in zip(order, actual): np.testing.assert_array_equal(aa, expected[n % 8])

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FastContracts))
    (REPORTS / 'fast-contracts.json').write_text(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful()), indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
