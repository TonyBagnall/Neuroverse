"""Stage a standard tsml-eval HC2 component experiment for the public competition.

No SSH or job submission. Keeps model persistence opt-in in an isolated checkout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile

import numpy as np
from tools.competition_track2 import DEFAULT_OUTPUT, ROOT, dump, sha, table, write_rows, write_ts

COMPONENTS = {'ConvolutionBased': 'Arsenal', 'IntervalBased': 'DrCIF-500',
              'ShapeletBased': 'STC-aeon', 'DictionaryBased': 'TDE'}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source revision differs at: ' + old[:100])
    return text.replace(old, new, 1)


def patched_sources(source):
    """Small opt-in extension; preserve native experiment fitting/results code."""
    files = {}
    rel = 'tsml_eval/utils/arguments.py'
    text = (source / rel).read_text(encoding='utf-8')
    text = replace_once(text, '    parser.add_argument(\n        "-chi",',
        '    parser.add_argument("--keep_checkpoint", action="store_true",\n'
        '        help="classification: retain the completed fitted model checkpoint; requires -ch")\n'
        '    parser.add_argument(\n        "-chi",')
    files[rel] = text
    rel = 'tsml_eval/experiments/experiments.py'
    text = (source / rel).read_text(encoding='utf-8')
    # Exactly two classification signatures precede other task implementations.
    first, rest = text.split('def run_regression_experiment(', 1)
    if first.count('    checkpoint_interval=None,\n):') != 2:
        raise ValueError('Classification signatures changed')
    first = first.replace('    checkpoint_interval=None,\n):',
                          '    checkpoint_interval=None,\n    keep_checkpoint=False,\n):')
    first = replace_once(first, '    resuming = False\n    if checkpoint_path is not None:',
        '    if keep_checkpoint and (checkpoint_path is None or not classifier.get_tag(\n'
        '            "capability:checkpointing", False, False)):\n'
        '        raise ValueError("keep_checkpoint requires a checkpointable classifier and -ch")\n'
        '    resuming = False\n    if checkpoint_path is not None:')
    first = replace_once(first, '        if attribute_file_path is not None:\n',
        '        if keep_checkpoint and not os.path.isfile(checkpoint_path):\n'
        '            raise RuntimeError("Completed fitted checkpoint was not written")\n'
        '        if attribute_file_path is not None:\n')
    first = replace_once(first, '    if checkpoint_path is not None and os.path.isfile(checkpoint_path):\n',
        '    if not keep_checkpoint and checkpoint_path is not None and os.path.isfile(checkpoint_path):\n')
    first = replace_once(first, '        checkpoint_interval=checkpoint_interval,\n    )\n',
        '        checkpoint_interval=checkpoint_interval,\n        keep_checkpoint=keep_checkpoint,\n    )\n')
    files[rel] = first + 'def run_regression_experiment(' + rest
    for rel in ['tsml_eval/experiments/classification_experiments.py',
                'tsml_eval/experiments/threaded_classification_experiments.py']:
        text = (source / rel).read_text(encoding='utf-8')
        if 'threaded_' in rel:
            text = replace_once(text, '                predefined_resample=args.predefined_resample,\n',
                '                predefined_resample=args.predefined_resample,\n'
                '                transform_train_only=args.transform_train_only,\n'
                '                cv_folds=args.cv_folds,\n'
                '                checkpoint=args.checkpoint,\n'
                '                checkpoint_interval=args.checkpoint_interval,\n'
                '                keep_checkpoint=args.keep_checkpoint,\n')
        else:
            text = replace_once(text, '                checkpoint_interval=args.checkpoint_interval,\n',
                '                checkpoint_interval=args.checkpoint_interval,\n'
                '                keep_checkpoint=args.keep_checkpoint,\n')
        files[rel] = text
    rel = '_tsml_research_resources/slurm/tsml_slurm.py'
    text = (source / rel).read_text(encoding='utf-8')
    text = replace_once(text, '        "#SBATCH --mail-type=FAIL",',
                        '        "#SBATCH --mail-type=FAIL" if e.email else "#SBATCH --mail-type=NONE",')
    files[rel] = text
    for rel, text in files.items():
        compile(text, rel, 'exec')
    return files


def prepare(data=DEFAULT_OUTPUT, source=Path('C:/Code/tsml-eval')):
    source = Path(source)
    stage = Path(data) / 'hc2-cluster-stage'
    stage.mkdir(parents=True, exist_ok=True)
    dataset = 'ProteusCommandsAllPublic'
    folder = stage / 'data' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    x = np.load(Path(data) / 'prepared/X.npy', mmap_mode='r')
    y = np.load(Path(data) / 'prepared/y.npy')
    rows = table(Path(data) / 'prepared/cases.tsv')
    smoke = [int(np.flatnonzero(y == label)[0]) for label in sorted(set(y))]
    train = folder / (dataset + '_TRAIN.ts')
    if not train.exists():
        write_ts(train.with_suffix('.partial'), x, y)
        train.with_suffix('.partial').replace(train)
    write_ts(folder / (dataset + '_TEST.ts'), x[smoke], y[smoke])
    write_rows(folder / 'cases_train.tsv', rows)
    write_rows(folder / 'cases_test.tsv', [rows[i] for i in smoke])
    for partition, indices in [('TRAIN', range(len(y))), ('TEST', smoke)]:
        (folder / (dataset + '_id_' + partition + '.txt')).write_text(
            ''.join(rows[i]['subject'] + '\n' for i in indices), encoding='utf-8')
    for name in ['metadata.json', 'channels.tsv']:
        shutil.copy2(Path(data) / 'prepared' / name, folder / name)
    (stage / 'datasets.txt').write_text(dataset + '\n', encoding='utf-8')
    patched = patched_sources(source)
    for rel, text in patched.items():
        path = stage / 'overlay' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
    manifest = dict(dataset=dataset, train_cases=len(y), test_cases=len(smoke),
        test_case_indices=smoke, labels=[int(v) for v in sorted(set(y))],
        shape=list(x.shape), seed=33, components=COMPONENTS,
        split='TRAIN is all public cases in original case order. TEST repeats one TRAIN case per class for diagnostic prediction only; not a generalisation evaluation.',
        train_estimate='Native component fit_predict_proba estimates, not resubstitution probabilities; these estimates supply HC2 weights and do not measure unseen-subject generalisation.',
        source_tsml_commit='ce0d1131dca7c2f432ae005162284acfe28ce4be',
        files={str(p.relative_to(stage)).replace('\\','/'): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in stage.rglob('*') if p.is_file() and p.name not in ['manifest.json', 'payload.tar']
               and '__pycache__' not in p.parts and p.suffix != '.pyc'},
        training_array_sha256=sha(Path(data) / 'prepared/X.npy'))
    dump(stage / 'manifest.json', manifest)
    dump(ROOT / 'docs/competition-track2-hc2-stage.json', manifest)
    return stage


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--source',type=Path,default=Path('C:/Code/tsml-eval'))
    args=parser.parse_args();print(prepare(args.data,args.source))
