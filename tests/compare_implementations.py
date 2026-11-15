"""Reproduce Atom/VSJet comparisons; save range-aligned PNGs and raw metrics.

Run from the repository root. No uploads occur in this script.
"""
from __future__ import annotations
from project_paths import use_vsjet
use_vsjet()

import argparse
import ast
import gc
import hashlib
import inspect
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
import vapoursynth as vs
from vsmasktools import dre_edgemask, retinex
from vsmasktools.spat_funcs import gauss_blur

from compare_quality import array, binary_agreement, difference, normalize, summarize

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(sigma=[50, 200, 350], lower_thr=.001, upper_thr=.005)
LABELS = {
    'our_fastest': 'Our fastest CUDA (2 workers)', 'our_fastest_cpu': 'Our fastest CPU (up to 16 workers)',
    'our_exact': 'Our exact CUDA', 'our_fast8': 'Our fast CUDA (downscale 8)',
    'our_exact_cpu': 'Our exact CPU (2 workers)', 'our_fast_cpu': 'Our fast CPU (downscale 8)',
    'atom_normal': 'Atom normal (CPU)', 'atom_draft': 'Atom draft (square root)',
    'vsjet_cpu_fast': 'VSJet fast (CPU)', 'vsjet_cpu_full': 'VSJet full (CPU)',
    'vsjet_cuda_fast': 'VSJet fast (CUDA blurs + CPU stages)', 'vsjet_cuda_full': 'VSJet full (CUDA blurs + CPU stages)',
    'vsjet_opencl_fast': 'VSJet fast (OpenCL blurs + CPU stages)', 'vsjet_opencl_full': 'VSJet full (OpenCL blurs + CPU stages)',
}
VISUAL_NAMES = ['our_exact', 'our_fast8', 'our_fastest', 'atom_normal', 'atom_draft',
                'vsjet_cpu_fast', 'vsjet_cuda_fast', 'vsjet_cuda_full']


def load_plugins(core):
    for namespace, path in [('retinex_gpu', 'dist/retinex_gpu.dll'),
                            ('retinex', 'build/all-in-one/tests/reference/retinex_reference.dll'),
                            ('tcanny', 'build/all-in-one/tests/reference/tcanny_reference.dll')]:
        if not hasattr(core, namespace):
            core.std.LoadPlugin(str(ROOT / path))


def load_atom(path, core):
    """Use the supplied functions without importing unrelated Atom dependencies."""
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('retinex_edgemask', 'kirsch')]
    if len(nodes) != 2:
        raise ValueError('Expected retinex_edgemask and kirsch in the Atom file')
    scope = dict(core=core, VideoNode=vs.VideoNode, GRAY=vs.GRAY)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['retinex_edgemask']


def make(core, clip, stage, name, atom):
    if name in ('our_fastest', 'our_fastest_cpu'):
        return getattr(core.retinex_gpu, 'MSRCP' if stage == 'retinex' else 'RetinexEdgeMask')(
            clip, mode='fastest', backend='cpu' if name.endswith('_cpu') else 'gpu', fallback=False, **OPTIONS)
    if name == 'our_exact':
        return getattr(core.retinex_gpu, 'MSRCP' if stage == 'retinex' else 'RetinexEdgeMask')(
            clip, mode='exact', backend='gpu', fallback=False, workers=2, **OPTIONS)
    if name == 'our_fast8':
        return getattr(core.retinex_gpu, 'MSRCP' if stage == 'retinex' else 'RetinexEdgeMask')(
            clip, mode='fast', backend='gpu', fallback=False, downscale=8, **OPTIONS)
    if name in ('our_exact_cpu', 'our_fast_cpu'):
        return getattr(core.retinex_gpu, 'MSRCP' if stage=='retinex' else 'RetinexEdgeMask')(
            clip, mode='exact' if name=='our_exact_cpu' else 'fast', backend='cpu', **OPTIONS)
    if name.startswith('atom_'):
        if stage == 'mask':
            return atom(clip, draft=name == 'atom_draft')
        return (clip.std.Expr('x 65535 / sqrt 65535 *') if name == 'atom_draft'
                else core.retinex.MSRCP(clip, **OPTIONS))
    backend = {'cpu': gauss_blur.Backend.CPU, 'cuda': gauss_blur.Backend.CUDA,
               'opencl': gauss_blur.Backend.GPU}[name.split('_')[1]]
    with backend():
        if stage == 'retinex':
            return retinex(clip, fast=name.endswith('_fast'), **OPTIONS)
        return dre_edgemask.RETINEX(clip, brz=0, fast=name.endswith('_fast'),
                                    sigmas=OPTIONS['sigma'], lower_thr=.001, upper_thr=.005)


def consume(node, requests):
    completed = 0
    for frame in node.frames(prefetch=requests, backlog=requests):
        frame.close()
        completed += 1
    return completed


def disable_caches(node):
    # Disable native intermediate caches too: repeated batches must recompute.
    vs.core.std.SetVideoCache(node, mode=0)
    for child in node.dependencies:
        disable_caches(child)


def limited_retinex(name):
    return name.startswith('our_') or name == 'atom_normal'


def save_png(a, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--atom', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/reports/implementation-comparison')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Reviewed frames for PNG export')
    parser.add_argument('--samples', type=int, default=24, help='Additional distributed quality frames')
    parser.add_argument('--starts', type=int, nargs='+', required=True, help='Benchmark sequence starts')
    parser.add_argument('--per-sequence', type=int, default=100)
    parser.add_argument('--requests', type=int, nargs='+', default=[32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--minimum-seconds', type=float, default=1.0)
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core = vs.core
    core.num_threads = 32
    core.max_cache_size = 16384
    load_plugins(core)
    atom = load_atom(args.atom, core)
    src = core.bs.VideoSource(str(args.input.resolve()), variableformat=0, threads=4,
        gpu=False, cachemode=2, cachepath=str(ROOT / 'build/bestsource-index'), showprogress=False)
    y = src.std.ShufflePlanes(0, vs.GRAY).resize.Point(format=vs.GRAY16)
    if min(args.frames + args.starts) < 0 or max(args.frames) >= y.num_frames:
        raise ValueError('Frame selections are outside the source')
    if any(start + max(args.per_sequence, 24) > y.num_frames for start in args.starts):
        raise ValueError('A sequence extends past the source')
    names = list(LABELS)
    timed_names = names
    report = dict(schema=2, source=args.input.name, date=time.strftime('%Y-%m-%d'),
        width=y.width, height=y.height, source_format=src.format.name,
        source_frames=y.num_frames, fps=float(y.fps), processing_format=y.format.name,
        options=OPTIONS, core_threads=32, max_cache_mib=16384,
        versions={n: version(n) for n in ('VapourSynth', 'vsjetpack', 'vapoursynth-bestsource', 'vapoursynth-vszip', 'vapoursynth-vszipcu', 'vapoursynth-vszipcl')},
        platform=platform.platform(), visual_frames=args.frames, sequence_starts=args.starts,
        method=dict(timing='Identical predecoded frames; unique warmup; fresh graph and disabled caches; '
            'rotating trial order; median FPS. Transfers and output allocation included. Decode, '
            'depth conversion, graph construction and warmup excluded. All native intermediate caches are disabled; whole batches repeat until the minimum duration. CPU full mode is timed.',
            quality='Exact GPU is a compatibility anchor, not perceptual truth. Soft masks /65535. '
            'Original/our Retinex (x-4096)/56064; VSJet and Atom draft x/65535. PNGs use the same '
            'fixed display mapping, rounded to 8-bit with no per-image contrast normalization.',
            atom='Only two supplied AST function definitions executed unchanged; upstream CPU API4 '
            'reference adapters; CPU TCanny. No Atom OpenCL timing.',
            temporal='Three or more 24-frame sequences; change in candidate-minus-exact residual; '
            'not motion compensated and not an absolute flicker score.'),
        sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            'atomchtools.py': args.atom, 'source': args.input,
            'retinex_gpu.dll': ROOT / 'dist/retinex_gpu.dll',
            'retinex_cpu_only.dll': ROOT / 'dist/cpu-only/retinex_gpu.dll',
            'vszipcu.dll': Path(core.vszipcu.plugin_path),
            'vszipcl.dll': Path(core.vszipcl.plugin_path),
            'vszip.dll': Path(core.vszip.plugin_path),
            'akarin.dll': Path(core.akarin.plugin_path),
            'resize2.dll': Path(core.resize2.plugin_path),
            'retinex_reference.dll': ROOT / 'build/all-in-one/tests/reference/retinex_reference.dll',
            'tcanny_reference.dll': ROOT / 'build/all-in-one/tests/reference/tcanny_reference.dll'}.items()},
        minimum_seconds=args.minimum_seconds,
        implementation_sources=dict(retinex=inspect.getsource(retinex),
                                    gauss_blur=inspect.getsource(gauss_blur._func),
                                    dre=inspect.getsource(dre_edgemask)),
        timing=[], trials=[], quality=[], binary=[], statistics=[], temporal=[], dark_agreement=[], failures=[])

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')

    if not args.skip_timing:
        timed = [n for start in args.starts for n in range(start, start + args.per_sequence)]
        warm = list(range(max(0, args.starts[0] - args.warmup), args.starts[0]))
        if len(warm) != args.warmup or set(warm) & set(timed):
            raise ValueError('Warmup indices must be unique and outside timed sequences')
        report['timed_source_frames'] = timed
        print('Predecoding benchmark frames', flush=True)
        held = [y.get_frame(n) for n in warm + timed]
        blank = y.std.BlankClip(length=len(held))
        memory = blank.std.ModifyFrame(blank, lambda n, f: held[n])
        core.std.SetVideoCache(memory, mode=0)
        for requests in args.requests:
            for stage in ('retinex', 'mask'):
                trials = {n: [] for n in timed_names}
                for repeat in range(args.repeats):
                    offset = repeat * 2 % len(timed_names)
                    order = timed_names[offset:] + timed_names[:offset]
                    if repeat % 2:
                        order = order[::-1]
                    for name in order:
                        node = make(core, memory, stage, name, atom)
                        disable_caches(node)
                        warm_node=node[:len(warm)]; timed_node=node[len(warm):]
                        core.std.SetVideoCache(warm_node, mode=0); core.std.SetVideoCache(timed_node, mode=0)
                        native_workers = None
                        if name.startswith('our_'):
                            with warm_node.get_frame(0) as first:
                                native_workers = int(first.props.RetinexWorkers)
                        warm_begin=time.perf_counter()
                        while time.perf_counter()-warm_begin < .5:
                            consume(warm_node, requests)
                        begin, cpu_begin = time.perf_counter(), time.process_time()
                        completed=cycles=0
                        while cycles==0 or time.perf_counter()-begin < args.minimum_seconds:
                            completed += consume(timed_node, requests); cycles += 1
                        elapsed = time.perf_counter() - begin
                        row = dict(stage=stage, candidate=name, requests=requests, repeat=repeat + 1,
                            frames=completed, batch_frames=len(timed), cycles=cycles, seconds=elapsed, fps=completed / elapsed,
                            cpu_cores=(time.process_time() - cpu_begin) / elapsed, native_workers=native_workers)
                        report['trials'].append(row)
                        trials[name].append(row)
                        save()
                        print(stage, name, requests, repeat + 1, f"{row['fps']:.1f} FPS", flush=True)
                        del node, warm_node, timed_node
                        gc.collect()
                for name, rows in trials.items():
                    report['timing'].append(dict(stage=stage, candidate=name, requests=requests,
                        fps=statistics.median(r['fps'] for r in rows),
                        cpu_cores=statistics.median(r['cpu_cores'] for r in rows)))
                save()
        del memory, blank
        for f in held:
            f.close()
        del held
        gc.collect()

    numbers = sorted(set(np.linspace(128, y.num_frames - 128, args.samples, dtype=int)) | set(args.frames))
    report['quality_frames'] = [int(n) for n in numbers]
    manifest = dict(source=args.input.name, fps=float(y.fps), frames=args.frames, stages={})
    for stage in ('retinex', 'mask'):
        nodes = {name: make(core, y, stage, name, atom) for name in names}
        kirsch = core.retinex_gpu.Kirsch(y, backend='gpu', fallback=False, workers=1) if stage=='mask' else None
        manifest['stages'][stage] = dict(labels={'source': 'Source luma', **{n: LABELS[n] for n in VISUAL_NAMES}}, images={})
        for n in numbers:
            n = int(n)
            raw = array(y, n)
            uniform = bool(raw.min() == raw.max())
            values = {name: normalize(array(node, n), stage == 'retinex' and limited_retinex(name)) for name, node in nodes.items()}
            if stage=='mask':
                dark = normalize(raw, True) < .15
                contributed = values['our_exact'] - normalize(array(kirsch, n)) > .01
            for name, a in values.items():
                if name != 'our_exact':
                    report['quality'].append(dict(stage=stage, frame=n, uniform=uniform, candidate=name,
                                                 **difference(values['our_exact'], a)))
                if stage == 'mask':
                    report['statistics'].append(dict(frame=n, candidate=name, uniform=uniform,
                        mean=float(a.mean()), maximum=float(a.max()), active_5pct=float(np.mean(a > .05))))
                    if name != 'our_exact':
                        report['binary'].append(dict(frame=n, candidate=name, uniform=uniform,
                            **binary_agreement(values['our_exact'], a, .122)))
                        for region, selection in [('dark',dark),('dark_retinex_contribution',dark & contributed)]:
                            for threshold in (.03,.122):
                                reference = (values['our_exact'] >= threshold) & selection
                                candidate = (a >= threshold) & selection
                                report['dark_agreement'].append(dict(frame=n,candidate=name,region=region,threshold=threshold,
                                    region_pixels=int(selection.sum()),reference_pixels=int(reference.sum()),
                                    candidate_pixels=int(candidate.sum()),missed=int(np.count_nonzero(reference & ~candidate)),
                                    extra=int(np.count_nonzero(candidate & ~reference))))
            if n in args.frames:
                exports = {'source': normalize(raw, True), **{name: values[name] for name in VISUAL_NAMES}}
                paths = {}
                for name, a in exports.items():
                    relative = f'images/{stage}/{n}-{name}.png'
                    save_png(a, args.output / relative)
                    paths[name] = relative
                manifest['stages'][stage]['images'][str(n)] = paths
            print('quality', stage, n, flush=True)
        if stage == 'mask':
            for start in args.starts:
                previous = None
                for n in range(start, start + 24):
                    vals = {name: normalize(array(nodes[name], n)) for name in
                            ('our_exact', 'our_fast8', 'our_fastest', 'atom_normal', 'atom_draft', 'vsjet_cpu_fast')}
                    if previous is not None:
                        for name in vals:
                            if name != 'our_exact':
                                delta = (vals[name] - vals['our_exact']) - (previous[name] - previous['our_exact'])
                                report['temporal'].append(dict(start=start, frame=n, candidate=name,
                                    residual_change_mae=float(np.abs(delta).mean())))
                    previous = vals
        del nodes
        del kirsch
        gc.collect()
        save()
    report['summary'] = {stage: {name: summarize([r for r in report['quality'] if r['stage'] == stage
        and r['candidate'] == name and not r['uniform']]) for name in names if name != 'our_exact'}
        for stage in ('retinex', 'mask')}
    report['passed'] = not report['failures']
    report['summary_cpu_parity'] = {stage: report['summary'][stage]['our_exact_cpu'] for stage in ('retinex','mask')}
    report['dark_summary'] = {}
    for name in names:
        if name=='our_exact': continue
        report['dark_summary'][name] = {}
        for region in ('dark','dark_retinex_contribution'):
            report['dark_summary'][name][region] = {}
            for threshold in (.03,.122):
                rows=[r for r in report['dark_agreement'] if r['candidate']==name and r['region']==region and r['threshold']==threshold]
                totals={key:sum(r[key] for r in rows) for key in ('region_pixels','reference_pixels','candidate_pixels','missed','extra')}
                totals['recall']=1-totals['missed']/max(1,totals['reference_pixels'])
                totals['precision']=1-totals['extra']/max(1,totals['candidate_pixels'])
                report['dark_summary'][name][region][str(threshold)] = totals
    save()
    (args.output / 'images.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
