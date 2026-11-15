"""Parity and input-contract checks for the dependency-free aeon export."""
import copy
import json
import numpy as np
import pytest
from aeon.classification.convolution_based import MiniRocketClassifier
from tools.build_aeon_track2_submission import export,load_submission
from tools.aeon_track2_submission import PortableMiniRocket


@pytest.fixture(scope='module')
def fitted():
    rng=np.random.default_rng(33)
    x=rng.normal(size=(24,41,480)).astype(np.float64)
    y=np.tile(np.arange(3),8)
    x[y==1,0]*=3;x[y==2,1]*=3
    classifier=MiniRocketClassifier(n_kernels=168,n_jobs=1,random_state=33).fit(x,y)
    return classifier,x,y


def test_export_preserves_fitted_model_predictions_and_licence(fitted,tmp_path):
    classifier,x,y=fitted;folder=export(classifier,tmp_path/'model');model=load_submission(folder)
    assert np.array_equal(model.predict(x),classifier.predict(x))
    assert (folder/'LICENSE.txt').is_file()
    assert sorted(p.name for p in folder.iterdir())==['LICENSE.txt','submission.py','weights.npz']
    assert np.allclose(model.features(x),classifier._transformer.transform(x),atol=1/480)


def test_torch_input_and_session_copy_preserve_predictions(fitted,tmp_path):
    import torch
    classifier,x,y=fitted;folder=export(classifier,tmp_path/'model');model=load_submission(folder)
    expected=model.predict(x[:1]);fresh=copy.deepcopy(model);fresh.reset_state()
    assert np.array_equal(fresh.predict(torch.tensor(x[:1],dtype=torch.float32)),expected)
    assert np.array_equal(model.predict(x[:1]),expected)


def test_wrong_shape_and_nonfinite_input_are_rejected(fitted,tmp_path):
    classifier,x,y=fitted;folder=export(classifier,tmp_path/'model');model=PortableMiniRocket(folder/'weights.npz')
    with pytest.raises(ValueError,match='Invalid'):model.predict(x[:1,:,:100])
    bad=x[:1].copy();bad[0,0,0]=np.nan
    with pytest.raises(ValueError,match='Invalid'):model.predict(bad)
