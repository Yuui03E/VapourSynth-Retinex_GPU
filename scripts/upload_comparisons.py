"""Upload reviewed comparison PNGs with VSView's installed slow.pics extension.

Requires vsview-comp. Uploads are link-only, have no scheduled expiry, and use
an anonymous browser identity. The private upload record stays under build/.
"""
import argparse
import asyncio
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QObject
from vsview_comp.models import ComparisonImage, ComparisonSource, SlowPicsSources
from vsview_comp.worker import SlowPicsWorker


class MemorySecrets:
    def __init__(self):
        self.values = {}

    def set_json(self, namespace, key, value):
        self.values[namespace, key] = value

    def get_json(self, namespace, key):
        return self.values.get((namespace, key))

    def delete_json(self, namespace, key):
        self.values.pop((namespace, key), None)

    def get_credential(self, context):
        return None


async def upload(args):
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    root = args.manifest.parent
    record_path = root / 'upload-record-private.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        'browser_id': str(uuid4()), 'collections': {}}
    settings = SimpleNamespace(global_=SimpleNamespace(browser_id=record['browser_id']))
    parent = QObject()
    worker = SlowPicsWorker(None, settings, MemorySecrets(), parent)
    links = {}
    for stage, data in manifest['stages'].items():
        if stage in record['collections']:
            links[stage] = record['collections'][stage]
            print('Already uploaded', stage, links[stage], flush=True)
            continue
        sources = []
        for name, label in data['labels'].items():
            images = []
            for n in manifest['frames']:
                path = (root / data['images'][str(n)][name]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                seconds = n / manifest['fps']
                stamp = f'{int(seconds // 60):02d}:{seconds % 60:06.3f}'
                images.append(ComparisonImage(path, '?', n, stamp))
            sources.append(ComparisonSource(label, images))
        src = SlowPicsSources(f'{args.name} - {stage}', sources, False, False, None, -1, [])
        if src.total_images > 50:
            raise ValueError('Split generated images into collections of at most 50')

        def progress(**values):
            if 'increment' in values:
                print('Uploaded an image', flush=True)

        for attempt in range(3):
            try:
                url = await inspect.unwrap(SlowPicsWorker.upload)(worker, src=src, cookies={}, progress_cb=progress)
                break
            except asyncio.CancelledError as error:
                # VSView reports the service's retry timer as CancelledError.
                # Preserve real cancellation, and retry only that explicit response.
                message = str(error)
                match = re.search(r'Slowpics rate limited\. Retry after (\d+)s', message)
                if not match or attempt == 2:
                    raise
                delay = int(match[1])
                print(f'Service rate limit: waiting {delay} seconds before retry.', flush=True)
                await asyncio.sleep(delay)
        record['browser_id'] = settings.global_.browser_id
        record['collections'][stage] = links[stage] = url
        record_path.write_text(json.dumps(record, indent=2), encoding='utf8')
        print('UPLOAD COMPLETE', stage, url, flush=True)
    (root / 'slowpics-links.json').write_text(json.dumps(dict(
        uploader='VSView slow.pics extension', visibility='link-only',
        comparisons_per_collection=len(manifest['frames']), links=links), indent=2), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--name', default='Retinex GPU vs Atom vs VSJet')
    parser.add_argument('--upload', action='store_true', help='Explicitly enable transmission to slow.pics')
    args = parser.parse_args()
    if not args.upload:
        raise SystemExit('Review the PNGs, then pass --upload to publish the comparisons.')
    app = QCoreApplication([])
    asyncio.run(upload(args))


if __name__ == '__main__':
    main()
