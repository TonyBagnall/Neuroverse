"""Local result checks and protection against replacing newer desktop copies."""
import json
import os
import time

import pytest

from tools.pull_iridis_predictions import pull, validate_csv


def test_prediction_checks_validate_probabilities_and_stored_accuracy(tmp_path):
    path=tmp_path/'result.csv'
    path.write_text('data,classifier,TEST,0\nparameters\n1,0,0,-1,0,2,N/A,-1,-1\n0,0,,0.8,0.2\n1,1,,0.3,0.7\n')
    assert validate_csv(path)==dict(cases=2,classes=2,accuracy=1.0)
    path.write_text(path.read_text().replace('0.3,0.7','0.3,0.6'))
    with pytest.raises(ValueError,match='Probability sum'):validate_csv(path)


def test_prediction_checks_reject_truncated_rows(tmp_path):
    path=tmp_path/'result.csv'
    path.write_text('data,classifier,TEST,0\nparameters\n1,0,0,-1,0,2,N/A,-1,-1\n0,0,,0.8\n')
    with pytest.raises(ValueError,match='Incomplete result row'):validate_csv(path)


def test_older_remote_file_never_overwrites_newer_local_copy(tmp_path,monkeypatch):
    local=tmp_path/'results';local.mkdir()
    path=local/'result.csv';path.write_text('newer result')
    now=int(time.time());os.utime(path,(now-1000,now-1000))
    inventory=tmp_path/'inventory.json'
    inventory.write_text(json.dumps(dict(files=[dict(path='result.csv',bytes=999,mtime_ns=(now-2000)*10**9)])))
    def unexpected_network(*args,**kwargs):raise AssertionError('No transfer needed')
    monkeypatch.setattr('tools.pull_iridis_predictions.subprocess.Popen',unexpected_network)
    result=pull(inventory,'/remote/results',local,tmp_path/'report.json')
    assert result['skipped_older']==['result.csv']
    assert result['requested_files']==0
    assert path.read_text()=='newer result'
