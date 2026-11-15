"""Verify public slow.pics collections against every reviewed local PNG."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image


def download(url):
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/results/comparison-images.json'))
    args = parser.parse_args()
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    links = json.loads((root / 'slowpics-links.json').read_text(encoding='utf8'))
    report = dict(passed=True, source=manifest['source'], frames=manifest['frames'],
        fps=manifest['fps'], stages={})
    jobs = []
    for stage, data in manifest['stages'].items():
        url = links['links'][stage]
        html = download(url).decode('utf8')
        marker = 'var collection = '
        collection, _ = json.JSONDecoder().raw_decode(html.split(marker, 1)[1])
        assert len(collection['comparisons']) == len(manifest['frames'])
        scenes = []
        for n, scene in zip(manifest['frames'], collection['comparisons']):
            assert scene['name'].endswith(f'/ {n}')
            assert len(scene['images']) == len(data['labels'])
            images = []
            for (name, label), remote in zip(data['labels'].items(), scene['images']):
                assert remote['name'] == label and not remote['broken']
                assert (remote['width'], remote['height']) == (1920, 1080)
                local = root / data['images'][str(n)][name]
                expected = hashlib.sha256(local.read_bytes()).hexdigest()
                with Image.open(local) as png:
                    assert png.mode == 'L' and png.size == (1920, 1080)
                    pixel_hash = hashlib.sha256(png.tobytes()).hexdigest()
                image = dict(candidate=name, label=label, sha256=expected,
                    decoded_pixel_sha256=pixel_hash,
                    url='https://i.slow.pics/' + remote['path'], width=1920, height=1080)
                images.append(image)
                jobs.append(image)
            scenes.append(dict(frame=n, url='https://slow.pics/c/' + scene['key'], images=images))
        report['stages'][stage] = dict(url=url, title=collection['name'], scenes=scenes)
    def verify(image):
        data = download(image['url'])
        actual = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as png:
            assert png.size == (image['width'], image['height'])
            # Lossless optimization may replace L with a grayscale palette.
            rgb = png.convert('RGB')
            channels = [rgb.getchannel(c).tobytes() for c in ('R', 'G', 'B')]
            assert channels[0] == channels[1] == channels[2], image['url']
            pixels = hashlib.sha256(channels[0]).hexdigest()
            image['download_pixel_format'] = png.mode
        assert pixels == image['decoded_pixel_sha256'], image['url']
        image['download_sha256'] = actual
        image['download_pixel_sha256'] = pixels
        image['encoding_bytes_match'] = actual == image['sha256']
        image['verified'] = True
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(verify, jobs))
    report['verified_images'] = len(jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(f'Verified identical decoded pixels in all {len(jobs)} uploaded PNGs.')


if __name__ == '__main__':
    main()
