"""Resumable, subject-isolated public PROTEUS evaluation. No submissions."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from tools.competition_track2 import DEFAULT_OUTPUT, ROOT, dump, score, sha, table
from tools.train_track2_selected import native_scores


def subject_folds(rows):
    subjects = np.asarray([str(row['subject']) for row in rows])
    for subject in sorted(set(subjects)):
        train = np.flatnonzero(subjects != subject)
        test = np.flatnonzero(subjects == subject)
        if not len(train) or not len(test):
            raise ValueError('LOSO requires at least two subjects')
        assert set(subjects[train]).isdisjoint(subjects[test])
        yield subject, train, test


def run(data=DEFAULT_OUTPUT, output=None, jobs=8, classifier=None, subject_only=None):
    import aeon
    import sklearn
    import torch
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    from aeon.classification.interval_based import QUANTClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    torch.set_num_threads(jobs)
    folder = Path(data) / 'prepared'
    out = Path(output or Path(data) / 'selected-loso')
    out.mkdir(parents=True, exist_ok=True)
    x = np.load(folder / 'X.npy', mmap_mode='r')
    y = np.load(folder / 'y.npy')
    rows = table(folder / 'cases.tsv')
    assert len(x) == len(y) == len(rows)
    identity = dict(cases_sha256=sha(folder / 'cases.tsv'),
                    x_sha256=sha(folder / 'X.npy'), y_sha256=sha(folder / 'y.npy'),
                    random_state=33, jobs=jobs, aeon=aeon.__version__,
                    sklearn=sklearn.__version__, torch=torch.__version__)
    config = out / 'configuration.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8')) != identity:
        raise ValueError('Cannot resume with different data, versions or settings')
    dump(config, identity)
    state = out / 'state.json'
    report = dict(protocol='Leave one whole subject out; all sessions/runs/interfaces together. '
                  'Fresh fixed-parameter model per fold; no held-out subject used for fitting. '
                  'All public subjects participate, including the previous development/test subjects.',
                  configuration=identity, folds={}, status='running')
    all_scores = {}
    all_predictions = {}
    try:
        # Complete MRHydra first, so its overall result is available before QUANT.
        for name in ['MRHydra', 'QUANT']:
            if classifier is not None and name != classifier:
                continue
            values = np.full((len(y), 3), np.nan)
            predictions = np.full(len(y), -1, dtype=int)
            for subject, train, test in subject_folds(rows):
                if subject_only is not None and subject != subject_only:
                    continue
                path = out / name / subject
                path.mkdir(parents=True, exist_ok=True)
                receipt = path / 'result.json'
                dump(state, dict(status='running', classifier=name, subject=subject,
                                 completed=report['folds'], updated=time.time()))
                if receipt.exists():
                    result = json.loads(receipt.read_text(encoding='utf-8'))
                    saved = np.load(path / 'predictions.npz', allow_pickle=False)
                    if not np.array_equal(saved['case_index'], test):
                        raise ValueError('Saved case indices differ')
                    if not np.array_equal(saved['y_true'], y[test]):
                        raise ValueError('Saved labels differ')
                    values[test] = saved['scores']
                    predictions[test] = saved['predictions']
                else:
                    print('Fitting', name, 'held-out', subject, len(train), len(test), flush=True)
                    if name == 'MRHydra':
                        model = MultiRocketHydraClassifier(n_jobs=jobs, random_state=33)
                    else:
                        model = QUANTClassifier(random_state=33, estimator=ExtraTreesClassifier(
                            n_estimators=200, max_features=.1, criterion='entropy',
                            n_jobs=jobs, random_state=33))
                    start = time.perf_counter()
                    model.fit(x[train], y[train])
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    predictions[test] = model.predict(x[test])
                    values[test] = native_scores(model, name, x[test])
                    result = dict(subject=subject, train_cases=len(train), test_cases=len(test),
                                  train_subjects=sorted({rows[i]['subject'] for i in train}),
                                  fit_seconds=fit_seconds,
                                  prediction_seconds=time.perf_counter() - start,
                                  score_argmax_agrees_with_native=bool(np.array_equal(
                                      np.argmax(values[test], axis=1), predictions[test])),
                                  metrics=score(y[test], predictions[test], [rows[i] for i in test]))
                    if not np.isfinite(values[test]).all():
                        raise ValueError('Non-finite prediction scores')
                    np.savez_compressed(path / 'predictions.npz', case_index=test,
                                        y_true=y[test], predictions=predictions[test], scores=values[test])
                    dump(receipt, result)  # Only a completed receipt enables resume.
                    del model
                    gc.collect()
                report['folds'].setdefault(name, {})[subject] = result
                dump(out / 'results.json', report)
                print(name, subject, result['metrics']['balanced_accuracy_cell_mean'], flush=True)
            if subject_only is not None:
                if not report['folds'].get(name):
                    raise ValueError('Unknown held-out subject: ' + subject_only)
                return report
            assert np.isfinite(values).all() and (predictions >= 0).all()
            all_scores[name] = values
            all_predictions[name] = predictions
            report.setdefault('overall', {})[name] = score(y, predictions, rows)
            report['overall'][name]['subject_macro_cell_balanced_accuracy'] = float(np.mean([
                f['metrics']['balanced_accuracy_cell_mean'] for f in report['folds'][name].values()]))
            dump(out / 'results.json', report)
            dump(ROOT / 'docs/competition-track2-loso-results.json', report)
        ensemble = (all_scores['MRHydra'] + all_scores['QUANT']) / 2
        pred = np.argmax(ensemble, axis=1)
        report['ensemble_definition'] = 'Equal mean of QUANT probabilities and clipped/normalised MRHydra ridge scores; no fitted calibration or weights.'
        report['overall']['ensemble'] = score(y, pred, rows)
        report['folds']['ensemble'] = {}
        for subject, train, test in subject_folds(rows):
            report['folds']['ensemble'][subject] = dict(subject=subject, test_cases=len(test),
                metrics=score(y[test], pred[test], [rows[i] for i in test]))
        report['overall']['ensemble']['subject_macro_cell_balanced_accuracy'] = float(np.mean([
            f['metrics']['balanced_accuracy_cell_mean'] for f in report['folds']['ensemble'].values()]))
        np.savez_compressed(out / 'out-of-fold.npz', case_index=np.arange(len(y)), y_true=y,
                            mrhydra_predictions=all_predictions['MRHydra'],
                            quant_predictions=all_predictions['QUANT'],
                            mrhydra_scores=all_scores['MRHydra'], quant_scores=all_scores['QUANT'],
                            ensemble_predictions=pred)
        report['status'] = 'complete'
        dump(out / 'results.json', report)
        dump(ROOT / 'docs/competition-track2-loso-results.json', report)
        dump(state, dict(status='complete', updated=time.time(), overall=report['overall']))
        return report
    except BaseException as exc:
        dump(state, dict(status='failed', error=repr(exc), updated=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--classifier', choices=['MRHydra', 'QUANT'])
    parser.add_argument('--subject')
    args = parser.parse_args()
    run(args.data, args.output, args.jobs, args.classifier, args.subject)
