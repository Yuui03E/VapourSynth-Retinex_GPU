"""Package the two verified v2.1 Windows builds, each with exactly one DLL."""
import hashlib
import json
from pathlib import Path
import shutil
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist/releases'
KINDS = ('all-in-one', 'cpu-only')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(kind):
    lines = (ROOT / 'build' / kind / 'CMakeCache.txt').read_text().splitlines()
    keys = ('RETINEX_CUDA:', 'RETINEX_TESTING:', 'CMAKE_CUDA_ARCHITECTURES:')
    return {line.split(':', 1)[0]: line.split('=', 1)[1] for line in lines if line.startswith(keys)}


def compiler_versions(kind):
    versions = {}
    for language in ('CXX', 'CUDA'):
        candidates = sorted((ROOT / 'build' / kind / 'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake'))
        if candidates:
            text = candidates[-1].read_text()
            match = re.search(r'set\(CMAKE_' + language + r'_COMPILER_VERSION "([^"]+)"\)', text)
            if match:
                versions[language] = match[1]
    return versions


def main():
    binaries = {kind: ROOT / 'build' / kind / 'retinex_gpu.dll' for kind in KINDS}
    configs = {kind: configuration(kind) for kind in KINDS}
    if configs['all-in-one'].get('RETINEX_CUDA') != 'ON' or configs['cpu-only'].get('RETINEX_CUDA') != 'OFF':
        raise SystemExit('Expected one all-in-one and one CPU-only normal build.')
    if any(c.get('RETINEX_TESTING') != 'OFF' for c in configs.values()):
        raise SystemExit('Test builds must never be packaged.')
    report = json.loads((ROOT / 'build/reports/unified-contracts.json').read_text())
    for kind, p in binaries.items():
        if not report.get('passed') or report['sha256'][kind] != digest(p):
            raise SystemExit('Run passing unified tests on current binaries before packaging.')
    comparison = json.loads((ROOT / 'docs/results/comparison.json').read_text(encoding='utf8'))
    audit = json.loads((ROOT / 'docs/results/backend-audit.json').read_text(encoding='utf8'))
    if not comparison.get('passed') or not audit.get('passed'):
        raise SystemExit('Current comparison and backend audit must pass before packaging.')
    if comparison['sha256']['retinex_gpu.dll'] != digest(binaries['all-in-one']):
        raise SystemExit('Comparison was not measured with the current all-in-one binary.')
    if comparison['sha256']['retinex_cpu_only.dll'] != digest(binaries['cpu-only']):
        raise SystemExit('Comparison does not identify the current CPU-only binary.')
    if audit['project_binary_sha256'] != digest(binaries['all-in-one']):
        raise SystemExit('Backend audit does not identify the current all-in-one binary.')
    for name in ('contract-tests', 'fast-contracts', 'fastest-contracts'):
        if not json.loads((ROOT / 'build/reports' / f'{name}.json').read_text()).get('passed'):
            raise SystemExit(name + ' has not passed.')
    fastest = json.loads((ROOT / 'docs/results/fastest-contracts.json').read_text(encoding='utf8'))
    if not fastest.get('passed') or any(fastest['sha256'][kind] != digest(p) for kind, p in binaries.items()):
        raise SystemExit('Fastest contracts do not identify both current production binaries.')
    parity = json.loads((ROOT / 'docs/results/fastest-video-parity.json').read_text(encoding='utf8'))
    if not parity.get('passed') or parity['candidate_sha256'] != parity['anchor_sha256'] or parity['candidate_sha256'] != digest(binaries['all-in-one']):
        raise SystemExit('Fastest video parity does not identify the current binary.')
    images = json.loads((ROOT / 'docs/results/comparison-images.json').read_text(encoding='utf8'))
    if not images.get('passed') or images.get('verified_images') != 90 or images['frames'] != comparison['visual_frames']:
        raise SystemExit('Current five-scene galleries must contain 90 verified images.')
    uploaded = [image for stage in images['stages'].values() for scene in stage['scenes'] for image in scene['images']]
    if len(uploaded) != 90 or any(not image.get('verified') or image['decoded_pixel_sha256'] != image['download_pixel_sha256'] for image in uploaded):
        raise SystemExit('Uploaded image pixels do not match reviewed originals.')
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    files = sorted({ROOT / name.decode('utf8') for name in raw.split(b'\0') if name})
    files = [p for p in files if p.is_file()]
    for name in ('refs/vapoursynth/include/VapourSynth4.h', 'refs/vapoursynth/include/VSHelper4.h', 'refs/vapoursynth/COPYING.LESSER'):
        p = ROOT / name
        if p.is_file():
            files.append(p)
    files = sorted(set(files))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    manifest = dict(tag='v2.1', version='2.1', platform='Windows x64', source_commit=commit, source_dirty=dirty,
        compiler_versions={kind: compiler_versions(kind) for kind in KINDS}, configurations=configs,
        binaries={kind: dict(bytes=p.stat().st_size, sha256=digest(p)) for kind, p in binaries.items()},
        requirements='VapourSynth API 4. All-in-one includes exact/fast/fastest CPU and CUDA modes; '
            'CPU-only includes exact/fast/fastest CPU modes and no NVIDIA dependency. Both expose core.retinex_gpu. '
            'CUDA targets 75/86/89/120 plus compute_75 PTX; GPU use requires a compatible NVIDIA driver. '
            'Physical testing: Windows x64 / RTX 5070.',
        verification=dict(contracts_passed=True, backend_audit_passed=True,
            comparison_passed=True, fastest_video_parity_passed=True,
            uploaded_images_verified=images['verified_images'], vsjet_version=comparison['versions']['vsjetpack'],
            quality_frames=len(comparison['quality_frames']), timed_batch_frames=len(comparison['timed_source_frames'])),
        source_files={p.relative_to(ROOT).as_posix(): digest(p) for p in files})
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = OUT / 'release-build-info.json'
    metadata.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    packages = []
    for kind, p in binaries.items():
        dest = ROOT / 'dist' / ('retinex_gpu.dll' if kind == 'all-in-one' else 'cpu-only/retinex_gpu.dll')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        path = OUT / f'retinex-2.1-{kind}-windows-x64.zip'
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for source in files:
                archive.write(source, source.relative_to(ROOT).as_posix())
            archive.write(p, 'dist/retinex_gpu.dll')
            archive.write(metadata, 'dist/release-build-info.json')
        with zipfile.ZipFile(path) as archive:
            dlls = [n for n in archive.namelist() if n.lower().endswith('.dll')]
            if dlls != ['dist/retinex_gpu.dll'] or archive.testzip() is not None:
                raise RuntimeError('Package must contain exactly one intact DLL.')
            if hashlib.sha256(archive.read(dlls[0])).hexdigest() != digest(p):
                raise RuntimeError('Packaged DLL checksum mismatch.')
        packages.append(path)
        print(path.relative_to(ROOT))
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in packages + [metadata]), encoding='utf8')


if __name__ == '__main__':
    main()
