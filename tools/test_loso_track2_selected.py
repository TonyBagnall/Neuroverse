"""Check the critical split guarantee independently of expensive training."""
import numpy as np
import pytest
from tools.loso_track2_selected import subject_folds


def test_all_sessions_of_a_subject_stay_together():
    rows = [dict(subject=s, session=t) for s, t in [
        ('01', '1'), ('02', '1'), ('01', '2'), ('03', '1'), ('02', '2')]]
    folds = list(subject_folds(rows))
    assert [s for s, _, _ in folds] == ['01', '02', '03']
    assert np.array_equal(np.sort(np.concatenate([te for _, _, te in folds])), np.arange(5))
    for subject, train, test in folds:
        assert all(rows[i]['subject'] == subject for i in test)
        assert all(rows[i]['subject'] != subject for i in train)
        assert sorted([*train, *test]) == list(range(5))


def test_single_subject_is_rejected():
    with pytest.raises(ValueError, match='at least two'):
        list(subject_folds([dict(subject='001')]))
