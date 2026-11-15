"""Validate uploaded public data and native experiment persistence on a compute node."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(stage, repo):
    stage, repo = Path(stage), Path(repo)
    manifest = json.loads((stage / 'manifest.json').read_text())
    for rel, entry in manifest['files'].items():
        p = stage / rel
        if p.stat().st_size != entry['bytes'] or digest(p) != entry['sha256']:
            raise ValueError('Payload mismatch: ' + rel)
    for rel, entry in manifest['files'].items():
        if rel.startswith('overlay/') and rel.endswith('.py') and digest(repo / rel.removeprefix('overlay/')) != entry['sha256']:
            raise ValueError('Installed overlay differs: ' + rel)
    sys.path.insert(0, str(repo))
    import numpy as np
    from aeon.datasets import load_from_ts_file
    folder = stage / 'data' / manifest['dataset']
    shapes = {}
    for partition in ['TRAIN', 'TEST']:
        x, y = load_from_ts_file(folder / (manifest['dataset'] + '_' + partition + '.ts'))
        with (folder / ('cases_' + partition.lower() + '.tsv')).open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        expected = manifest['train_cases'] if partition == 'TRAIN' else manifest['test_cases']
        assert x.shape == (expected, 41, 480)
        assert len(y) == len(rows) == expected
        assert np.isfinite(x).all()
        for i, row in enumerate(rows):
            if str(y[i]) != row['label']:
                raise ValueError('Case/label alignment differs')
            if hashlib.sha256(np.asarray(x[i], dtype='<f8').tobytes()).hexdigest() != row['signal_sha256']:
                raise ValueError('Case/signal alignment differs')
        shapes[partition] = list(x.shape)
        del x, y
    from check_track2_hc2_experiment import check
    result = check(repo, stage / 'preflight-models')
    result.update(data_shapes=shapes, all_case_hashes_and_labels_verified=True)
    (stage / 'preflight-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.stage, args.repo)
