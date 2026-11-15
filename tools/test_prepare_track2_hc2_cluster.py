"""Offline guards for the standard-runner competition overlay/configuration."""
from pathlib import Path
import tomllib
import hashlib
import json

import pytest

from tools.prepare_track2_hc2_cluster import patched_sources, replace_once, ROOT
from tools.verify_track2_hc2_overlay import verify


def test_guard_refuses_unknown_or_ambiguous_source():
    with pytest.raises(ValueError):
        replace_once('different source', 'expected source', 'replacement')
    with pytest.raises(ValueError):
        replace_once('a a', 'a', 'replacement')


def test_source_verification_ignores_generated_bytecode_but_rejects_source_changes(tmp_path):
    stage = tmp_path / 'stage';stage.mkdir()
    repo = tmp_path / 'repo';repo.mkdir()
    source = repo / 'experiment.py';source.write_text('correct source')
    entries = {'overlay/experiment.py': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
               'overlay/__pycache__/experiment.cpython-312.pyc': {'sha256': 'generated-cache-not-portable'}}
    (stage / 'manifest.json').write_text(json.dumps(dict(files=entries, source_tsml_commit='test')))
    verify(stage, repo)
    source.write_text('changed source')
    with pytest.raises(RuntimeError, match='source overlay changed'):
        verify(stage, repo)


def test_component_configuration_requires_full_train_estimates_and_models():
    config = tomllib.loads((ROOT / 'tools/slurm/competition17982_hc2_components.toml').read_text())
    assert list(config['estimators'].values()) == [['Arsenal'], ['DrCIF-500'], ['STC-aeon'], ['TDE']]
    assert config['experiment']['train_files'] is True
    assert config['experiment']['resamples'] == 1
    assert '-ch' in config['experiment']['extra_args']
    assert '--keep_checkpoint' in config['experiment']['extra_args']
    assert config['experiment']['email'] == ''
    assert config['experiment']['pin_commit'] is True
    assert config['slurm']['max_active'] == 4


def test_opt_in_overlay_preserves_defaults_and_forwards_checkpointing():
    source = Path('C:/Code/tsml-eval')
    if not source.is_dir():
        pytest.skip('Local source checkout unavailable')
    files = patched_sources(source)
    assert len(files) == 5
    core = files['tsml_eval/experiments/experiments.py']
    assert core.count('    keep_checkpoint=False,') == 2
    assert 'if not keep_checkpoint and checkpoint_path is not None' in core
    threaded = files['tsml_eval/experiments/threaded_classification_experiments.py']
    assert 'checkpoint=args.checkpoint,' in threaded
    assert 'checkpoint_interval=args.checkpoint_interval,' in threaded
    assert 'keep_checkpoint=args.keep_checkpoint,' in threaded
    assert '"#SBATCH --mail-type=NONE"' in files['_tsml_research_resources/slurm/tsml_slurm.py']
