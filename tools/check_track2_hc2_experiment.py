"""Small offline persistence/experiment checks; on clusters use a compute job."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def check(repo, output, experiment_module=None):
    sys.path.insert(0, str(repo))
    import aeon
    import sklearn
    from aeon.classification.convolution_based import Arsenal
    from aeon.classification.interval_based import DrCIFClassifier
    from aeon.classification.shapelet_based import ShapeletTransformClassifier
    from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
    from aeon.classification.sklearn import RotationForestClassifier
    from tsml_eval.experiments import get_classifier_by_name
    from tsml_eval.evaluation.storage import ClassifierResults
    if experiment_module:
        spec = importlib.util.spec_from_file_location('patched_experiment', experiment_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from tsml_eval.experiments import experiments as module
    configured = get_classifier_by_name('DrCIF-500', random_state=33, n_jobs=1)
    assert configured.n_estimators == 500
    rng = np.random.RandomState(33)
    x = rng.normal(size=(42, 3, 48))
    y = np.arange(42) % 3
    for i in range(len(y)):
        x[i, y[i], 10:20] += 2
    models = {
        'Arsenal': Arsenal(n_estimators=2, n_kernels=84, random_state=33),
        'DrCIF-500': DrCIFClassifier(n_estimators=2, random_state=33),
        'STC-aeon': ShapeletTransformClassifier(n_shapelet_samples=50, max_shapelets=6,
            estimator=RotationForestClassifier(n_estimators=5, random_state=33), random_state=33),
        'TDE': TemporalDictionaryEnsemble(n_parameter_samples=8, max_ensemble_size=3,
            randomly_selected_params=4, random_state=33),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, model in models.items():
        path = output / (name + '.ckpt')
        module.run_classification_experiment(x[:36], y[:36], x[36:], y[36:], model,
            str(output), classifier_name=name, dataset_name='PersistenceSmoke', resample_id=0,
            build_train_file=True, checkpoint_path=str(path), checkpoint_interval=30,
            keep_checkpoint=True, benchmark_time=False)
        assert path.is_file()
        restored = type(model).load_checkpoint(path)
        assert restored.is_fitted
        result_path = output / name / 'Predictions/PersistenceSmoke'
        result = ClassifierResults().load_from_file(str(result_path / 'testResample0.csv'))
        assert np.allclose(restored.predict_proba(x[36:]), result.probabilities)
        train = ClassifierResults().load_from_file(str(result_path / 'trainResample0.csv'))
        assert len(train.class_labels) == 36
        reports[name] = dict(checkpoint_kept=True, loaded_model_matches_test_probabilities=True,
                             train_cases=36, test_cases=6)
        print('Passed', name, flush=True)
    result = dict(passed=True, aeon=aeon.__version__, sklearn=sklearn.__version__,
                  drcif_production_trees=configured.n_estimators, reduced_smoke_models=True,
                  components=reports)
    (output / 'check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-module', type=Path)
    args = parser.parse_args()
    print(check(args.repo, args.output, args.experiment_module))
