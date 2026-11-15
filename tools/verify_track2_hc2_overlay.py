"""Lightweight per-job verification of the isolated standard-runner extension."""
import hashlib
import json
from pathlib import Path

def verify(stage, repo):
    manifest = json.loads((Path(stage) / 'manifest.json').read_text())
    for rel, entry in manifest['files'].items():
        if not rel.startswith('overlay/') or not rel.endswith('.py'):
            continue
        path = Path(repo) / rel.removeprefix('overlay/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Isolated source overlay changed: ' + rel)
    print('Verified competition source overlay; base commit', manifest['source_tsml_commit'])


if __name__ == '__main__':
    verify(Path('/home/ajb2u23/Competitions/NeuralInterfaces2026Track2/HC2'),
           Path('/home/ajb2u23/Code/tsml-eval-proteus-hc2-20261110'))
