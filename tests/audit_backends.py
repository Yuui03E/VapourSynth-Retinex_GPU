"""Prove which native filters VSJet executes; compare source and binary provenance."""
import argparse
import hashlib
import io
import inspect
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/vsjet'))
import vapoursynth as vs

class InspectionPolicy(vs.EnvironmentPolicy):
    def on_policy_registered(self, api):
        self.api = api
        self.env = api.create_environment(vs.ENABLE_GRAPH_INSPECTION)
    def on_policy_cleared(self):
        self.api.destroy_environment(self.env)
    def get_current_environment(self):
        return self.env
    def set_environment(self, env):
        self.env = env

# Register before any library can initialize the standalone core.
vs.register_policy(InspectionPolicy())
from importlib.metadata import version
from vsmasktools import retinex, dre_edgemask
from vsmasktools.spat_funcs import gauss_blur
from compare_implementations import make, LABELS, load_plugins, load_atom

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def official_binary(core, name):
    """Verify installed native code against the official PyPI Windows wheel."""
    package = 'vapoursynth-' + name
    installed_version = version(package)
    with urlopen(f'https://pypi.org/pypi/{package}/{installed_version}/json') as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata['urls']
                 if item['filename'].endswith('win_amd64.whl'))
    with urlopen(wheel['url']) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == wheel['digests']['sha256']
    path = Path(getattr(core, name).plugin_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        binary = next(n for n in archive.namelist() if Path(n).name == path.name)
        expected = hashlib.sha256(archive.read(binary)).hexdigest()
    record = dict(package=package, version=installed_version, wheel=wheel['filename'],
        wheel_sha256=wheel['digests']['sha256'], dll=path.name,
        dll_sha256=digest(path), matches_official_wheel=digest(path)==expected)
    assert record['matches_official_wheel'], (name, record)
    return record

def graph(node):
    seen = set()
    records = []
    def walk(n):
        if n in seen:
            return
        seen.add(n)
        record = dict(namespace=n._plugin_ns, function=n._name, width=n.width,
            height=n.height, format=n.format.name)
        record['parameters'] = {k: v for k, v in n._inputs.items()
            if k in ('sigma', 'num_streams', 'device_id', 'mode', 'backend', 'downscale', 'fallback', 'workers')}
        records.append(record)
        for child in n.dependencies:
            walk(child)
    walk(node)
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--atom', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=1440)
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 32
    load_plugins(core)
    atom = load_atom(args.atom, core)
    src = core.bs.VideoSource(str(args.input.resolve()), gpu=False, variableformat=0,
        cachemode=2, cachepath=str(ROOT / 'build/bestsource-index'), showprogress=False)
    y = src.std.ShufflePlanes(0, vs.GRAY).resize.Point(format=vs.GRAY16)
    records = []
    for stage in ('retinex', 'mask'):
        for name in LABELS:
            node = make(core, y, stage, name, atom)
            with node.get_frame(args.frame) as f:
                # Fetching the final frame forces all relevant native dependencies.
                backend = f.props.get('RetinexBackend')
            nodes = graph(node)
            gpu_nodes = [n for n in nodes if n['namespace'] in ('vszipcu', 'vszipcl')]
            if name.startswith('vsjet_'):
                selected = name.split('_')[1]
                expected = 0 if selected == 'cpu' else (2 if name.endswith('_fast') else 3) + (stage == 'mask')
                assert len(gpu_nodes) == expected, (name, stage, gpu_nodes)
                assert all(n['namespace'] == ('vszipcu' if selected == 'cuda' else 'vszipcl') for n in gpu_nodes)
                assert all((n['width'], n['height']) == (y.width, y.height) for n in gpu_nodes)
                assert any(n['namespace']=='vszip' and n['function']=='PlaneMinMax' for n in nodes)
            elif name.startswith('our_'):
                assert backend == ('cpu' if name.endswith('_cpu') else 'gpu')
            records.append(dict(candidate=name, stage=stage, frame=args.frame, executed=True,
                native_gpu_filters=gpu_nodes, graph=nodes, reported_backend=backend))
    sources = {}
    for name, obj, relative in [('retinex', retinex, 'vsmasktools/spat_funcs.py'),
                                ('gauss_blur', gauss_blur._func, 'vsrgtools/blur.py'),
                                ('dre', dre_edgemask, 'vsmasktools/masks.py')]:
        path = Path(inspect.getfile(obj))
        repo_file = ROOT / 'refs/vsjet' / path.relative_to(ROOT / '.tools/vsjet')
        assert repo_file.is_file() and digest(path)==digest(repo_file), (name, repo_file)
        sources[name] = dict(file=path.relative_to(ROOT / '.tools/vsjet').as_posix(),
            sha256=digest(path), line=inspect.getsourcelines(obj)[1], matches_pinned_upstream=True)
    revisions = {name: subprocess.check_output(['git', '-C', str(ROOT/'refs'/name), 'rev-parse', 'HEAD'], text=True).strip()
        for name in ('vsjet', 'vszip', 'vszipcu', 'vszipcl', 'retinex', 'tcanny')}
    report = dict(passed=True, vsjet_version=version('vsjetpack'), source_frames=[args.frame],
        project_binary_sha256=digest(Path(core.retinex_gpu.plugin_path)),
        python_sources=sources, upstream_revisions=revisions,
        official_binary_checks={name: official_binary(core, name)
            for name in ('vszip', 'vszipcu', 'vszipcl')}, executions=records,
        conclusions=dict(gpu_retinex='Hybrid pipeline: native CUDA/OpenCL Gaussian filters plus CPU expression/statistics stages.',
            fast_cpu='Largest sigma replaced by frame mean; remaining blurs downsample for CPU fast mode.',
            fast_gpu='Largest sigma replaced by frame mean; remaining Gaussian blurs retain full resolution.',
            native_code='vszipcu/vszipcl GaussBlur uses separable finite Gaussian taps; project exact uses double recursive Gaussian.'))
    assert all(p['matches_official_wheel'] for p in report['official_binary_checks'].values())
    dest = ROOT / 'build/audit/backend-audit.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(f'Passed: actual native dependencies executed for all {len(records)} candidate/stage combinations.')
if __name__ == '__main__':
    main()
