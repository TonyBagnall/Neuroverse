"""Download pinned model artifacts on the desktop for offline cluster transfer.

No model imports, fitting, installation or cluster-side network access. The output
is a Hugging Face hub cache, including pinned refs/main and model-card attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify(path, spec):
    data = Path(path).read_bytes()
    if len(data) != spec['size']:
        raise ValueError(f"Size mismatch: {path}")
    sha256 = hashlib.sha256(data).hexdigest()
    if spec.get('sha256') and sha256 != spec['sha256']:
        raise ValueError(f"SHA256 mismatch: {path}")
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if spec.get('git_blob_sha1') and blob != spec['git_blob_sha1']:
        raise ValueError(f"Git blob mismatch: {path}")
    return sha256


def prepare(output, lock, opener=urlopen):
    if os.environ.get('SLURM_JOB_ID') or any(
        name in socket.gethostname().lower() for name in ('hali', 'iridis')
    ):
        raise RuntimeError('Download on the desktop, then transfer the cache offline.')
    output = Path(output)
    receipt = []
    for model in lock['models']:
        revision = model['revision']
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Expected a pinned commit SHA')
        base = output / ('models--' + model['repo'].replace('/', '--'))
        snapshot = base / 'snapshots' / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for spec in model['files']:
            filename = spec['path']
            if Path(filename).name != filename or filename in ('.', '..'):
                raise ValueError('Artifact paths must be simple filenames')
            path = snapshot / filename
            url = f"https://huggingface.co/{model['repo']}/resolve/{revision}/{filename}"
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.partial')
                with opener(url, timeout=120) as response, temporary.open('wb') as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                verify(temporary, spec)
                temporary.replace(path)
            digest = verify(path, spec)
            receipt.append(dict(repo=model['repo'], revision=revision, file=filename,
                                bytes=path.stat().st_size, sha256=digest, source=url))
        (base / 'refs').mkdir(exist_ok=True)
        (base / 'refs' / 'main').write_text(revision, encoding='utf8')
        print(model['repo'], revision, 'verified', flush=True)
    (output / 'download-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=ROOT/'docs/foundation-model-source-lock.json')
    args = parser.parse_args()
    prepare(args.output, json.loads(args.lock.read_text()))
