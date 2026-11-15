"""Event pulse interpretation, native offsets and leakage detection."""
import numpy as np
import pytest
from tools.prepare_small_moabb import (
    pulse_onsets,
    slice_window,
    overlapping_intervals,
    case_row,
    select_channels,
    mamem_trials,
    check_amplitude_scale,
    validate,
    header,
)
from tools.prepare_wang import digest_case, format_case


def test_pulse_onsets_do_not_repeat_samples_and_keep_changed_codes():
    assert pulse_onsets([0,1,1,0,2,2,3,3,0,1]).tolist()==[1,4,6,9]
    assert pulse_onsets([2,2,0]).tolist()==[0]


def test_native_window_is_half_open_and_preserves_channel_order():
    x=np.arange(60,dtype=np.float64).reshape(20,3)
    result,start,end=slice_window(x,4,3,5)
    np.testing.assert_array_equal(result,x[7:12].T)
    assert (start,end)==(7,12) and result.dtype==np.float64
    with pytest.raises(ValueError,match='Incomplete'):slice_window(x,19,0,5)


def test_cross_split_interval_overlap_respects_recording_identity():
    train={'same':[(0,5),(10,15)],'other':[(0,20)]}
    assert overlapping_intervals(train,{'same':[(5,10),(15,20)]})==0
    assert overlapping_intervals(train,{'same':[(4,11)]})==2
    assert overlapping_intervals(train,{'different':[(0,20)]})==0


def test_participant_ids_remain_strings_and_original_partition_is_retained():
    row=case_row('speller',1,'0','train','s1.mat',5,2,'train')
    assert row['participant_id']=='01' and row['session_id']=='0'
    assert row['original_partition']=='train' and row['label_description']=='Target'


def test_global_channel_selection_retains_original_axis_mapping():
    x=np.arange(15,dtype=np.float64).reshape(3,5)
    result,names,keep=select_channels(x,['Fp1','Fp2','Cz'],[1])
    assert names==['Fp1','Cz'] and keep==[0,2]
    np.testing.assert_array_equal(result,x[[0,2]])
    with pytest.raises(ValueError):select_channels(x,['Fp1','Fp2','Cz'],[3])


def test_mamem_trials_use_exact_markers_and_one_based_positions():
    eeg=np.zeros((15,1000));eeg[-1,10]=33028;eeg[-1,138]=32779;eeg[-1,778]=32780
    events=np.array([[1,33028,11,1],[2,32779,139,2],[3,32780,779,3]])
    assert list(mamem_trials(eeg,events))==[(33028,138,778,0,1,2)]
    events[2,2]=778
    with pytest.raises(ValueError):list(mamem_trials(eeg,events))


@pytest.mark.parametrize('scale_factor', [1e-3, 1e6])
def test_validate_rejects_median_channel_sd_outside_plausible_microvolt_bounds(scale_factor):
    """Prove validate() enforces [0.5, 500] uV median SD bounds and reports |x| > 1000 uV case fraction."""
    import csv
    import tempfile
    from pathlib import Path

    n_cases = 4
    n_channels = 3
    n_points = 64
    n_classes = 2
    spec = dict(name='SyntheticScaleCheck', source='Synthetic', channels=n_channels, points=n_points, classes=n_classes)
    rng = np.random.default_rng(17)
    base_cases = rng.normal(loc=0.0, scale=25.0, size=(n_cases, n_channels, n_points))
    # Give exactly one case an excursion > 1000 uV while keeping median channel SD ~25 uV
    base_cases[0, 0, 0] = 1500.0
    n_high_excursion = 1

    def _write_export(folder: Path, factor: float) -> None:
        ts_path = folder / f'{spec["name"]}_TRAIN.ts'
        tsv_path = folder / 'cases_train.tsv'
        fields = ['split', 'row', 'participant_id', 'recording_id', 'start_sample', 'end_sample', 'sample_reference', 'label', 'signal_sha256']
        with ts_path.open('w', encoding='ascii', newline='\n') as ts_out, tsv_path.open('w', encoding='utf-8', newline='') as tsv_out:
            ts_out.write(header(spec))
            writer = csv.DictWriter(tsv_out, fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            for idx in range(n_cases):
                scaled = np.ascontiguousarray(base_cases[idx] * factor, dtype='<f8')
                lbl = str((idx % n_classes) + 1)
                ts_out.write(format_case(scaled) + ':' + lbl + '\n')
                writer.writerow(
                    dict(
                        split='TRAIN',
                        row=idx,
                        participant_id=f'{idx + 1:02d}',
                        recording_id=f'rec-{idx}',
                        start_sample=0,
                        end_sample=n_points,
                        sample_reference='Samples within continuous source',
                        label=lbl,
                        signal_sha256=digest_case(scaled),
                    )
                )

    with tempfile.TemporaryDirectory() as tmp_ok:
        folder_ok = Path(tmp_ok)
        _write_export(folder_ok, 1.0)
        report, _, _ = validate(folder_ok, spec, 'TRAIN')
        assert 0.5 <= report['median_channel_sd_uV'] <= 500.0
        assert report['cases_max_abs_gt_1000_uV'] == n_high_excursion
        assert report['fraction_cases_max_abs_gt_1000_uV'] == pytest.approx(n_high_excursion / n_cases)

    with tempfile.TemporaryDirectory() as tmp_bad:
        folder_bad = Path(tmp_bad)
        _write_export(folder_bad, scale_factor)
        with pytest.raises(ValueError, match='Amplitude scale check failed'):
            validate(folder_bad, spec, 'TRAIN')

