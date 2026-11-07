"""Write paradee_addon/pins.json: the int8 ONNX file and config.json, pinned at a revision.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

    python tools/pin_model.py                 # the revision upstream's v1.2 tag points to
    python tools/pin_model.py <sha>

Pins each file by size and sha256. The ONNX file is in Git LFS, whose metadata
gives its sha256, and the download is checked against it; config.json is a
plain file, so it is downloaded and hashed.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODEL = 'paradee-1.2'
REPO = 'sahilmahendrakar/Paradee-8M-v1.2'
TAG = 'v1.2'
FILES = ('config.json', 'onnx/paradee_int8.onnx')
HERE = Path(__file__).resolve().parent.parent


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def tag_revision() -> str:
    refs = json.loads(get(f'https://huggingface.co/api/models/{REPO}/refs'))
    return next(tag['targetCommit'] for tag in refs['tags'] if tag['name'] == TAG)


def pin(revision: str) -> dict:
    tree = {item['path']: item for item in json.loads(get(
        f'https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=1'))}
    files = []
    for path in FILES:
        data = get(f'https://huggingface.co/{REPO}/resolve/{revision}/{path}')
        sha = hashlib.sha256(data).hexdigest()
        lfs = (tree[path].get('lfs') or {}).get('oid')
        if lfs and lfs != sha:
            raise SystemExit(f'{path}: Hugging Face has {lfs}, the download {sha}')
        if len(data) != tree[path]['size']:
            raise SystemExit(f'{path}: Hugging Face says {tree[path]["size"]} bytes, the download has {len(data)}')
        files.append({'path': path, 'size': len(data), 'sha256': sha})
    print(f"{MODEL}: {sum(f['size'] for f in files) / 1e6:.1f} MB at {revision}")
    return {'repo': REPO, 'revision': revision, 'files': files}


def main(argv: list[str]) -> int:
    revision = argv[0] if argv else tag_revision()
    out = {'models': {MODEL: pin(revision)}}
    (HERE / 'paradee_addon' / 'pins.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
