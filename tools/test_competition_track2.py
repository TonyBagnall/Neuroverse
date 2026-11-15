"""Competition-critical splitting, grouping, source and stream-contract checks."""
import numpy as np
import pytest
from tools.competition_track2 import (balanced_accuracy, cue_windows,
                                     development_split, predict_streams,
                                     safe_target, score, source_digest,transfer_mode,dump)


def test_native_cue_grid_and_end_exclusive_window():
    events=[dict(trial_type='mi',onset='6.992',sample='3496',value='33024'),
            dict(trial_type='start_feedback',onset='10',sample='5000',value='781')]
    row=cue_windows(events,20000)[0]
    assert row['start_sample']==3996 and row['end_sample']==5996
    assert row['processed_start_sample']==959 and row['label']==0
    with pytest.raises(ValueError,match='Incomplete'):cue_windows(events,5500)
    events[0]['value']='33026'
    with pytest.raises(ValueError,match='Conflicting'):cue_windows(events,20000)


def test_development_split_is_subject_grouped_order_independent_and_fixed():
    people=[f'P{i:02d}' for i in range(1,11)]
    a=development_split(people);assert a==development_split(people[::-1]+people)
    assert [list(a.values()).count(k) for k in ['train','val','test']]==[6,2,2]
    assert a==development_split(people)


def test_cells_count_equally_not_by_number_of_epochs():
    rows=[dict(subject='P01',session='1',interface='Graz')]*6+[dict(subject='P02',session='1',interface='BrainHero')]*3
    y=np.array([0,1,2]*3);p=np.array([0,1,2]*2+[1,2,0])
    result=score(y,p,rows)
    assert result['balanced_accuracy_cell_mean']==.5
    assert result['accuracy']==pytest.approx(2/3)


def test_predicted_absent_class_matches_official_metric_semantics():
    assert balanced_accuracy([0,0],[0,1])==.25


def test_original_git_blob_checksum_and_path_confinement(tmp_path):
    import hashlib
    p=tmp_path/'test';p.write_bytes(b'abc')
    assert source_digest(p,'git')==hashlib.sha1(b'blob 3\0abc').hexdigest()
    with pytest.raises(ValueError,match='Unsafe'):safe_target(tmp_path,'../escaped')


class StatefulCounter:
    def reset_state(self):self.calls=0
    def predict(self,x):
        assert x.shape==(1,41,480)
        result=np.array([self.calls%3],dtype=int);self.calls+=1;return result


def test_streams_reset_per_session_use_chronological_order_and_no_metadata():
    rows=[dict(subject='P1',session='1',run='2',cue_onset_seconds='1'),
          dict(subject='P1',session='1',run='1',cue_onset_seconds='2'),
          dict(subject='P1',session='2',run='1',cue_onset_seconds='1')]
    model=StatefulCounter();out=predict_streams(model,np.zeros((3,41,480)),rows)
    assert out.tolist()==[1,0,0]
    assert not hasattr(model,'calls')


def test_invalid_class_or_alignment_is_rejected():
    with pytest.raises(ValueError,match='Invalid'):score(np.array([0]),np.array([3]),[{}])


def test_partial_download_appends_only_on_verified_range():
    assert transfer_mode(206,'bytes 5-9/10',5,10)=='ab'
    assert transfer_mode(200,None,5,10)=='wb'  # server ignored Range
    with pytest.raises(ValueError,match='range'):transfer_mode(206,'bytes 0-9/10',5,10)
    with pytest.raises(ValueError,match='range'):transfer_mode(206,'bytes 5-8/10',5,10)


def test_native_numpy_header_numbers_are_json_serialisable(tmp_path):
    import json
    p=tmp_path/'metadata.json';dump(p,{'native_samples':np.int64(210000)})
    assert json.loads(p.read_text())=={'native_samples':210000}
