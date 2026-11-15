"""Unit tests for Group A cognitive and affective scalp-EEG preparation helpers."""
from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest

from tools.prepare_cognitive_group_a import (
    DS007169_V105_EEG_SCALE,
    arithmetic_trials,
    decode_brainvision,
    decode_eeglab_set,
    dens_emotion_windows,
    fixed_resting_windows,
    harmonise_sampling_rate,
    nback_records,
    nback_trials,
    parse_clateeg_txt,
    sternberg_trials,
    whole_subject_split,
)


def test_whole_subject_split_sorts_deterministically_and_rejects_duplicates():
    """Verify 75/25 subject partition uses sorted unique IDs and rejects duplicates."""
    n_subjects = 8
    people = [f'{i:02d}' for i in range(n_subjects, 0, -1)]
    expected_train_count = int(np.floor(0.75 * n_subjects))
    train, ordered = whole_subject_split(people)
    assert ordered == [f'{i:02d}' for i in range(1, n_subjects + 1)]
    assert len(train) == expected_train_count
    assert train == set(ordered[:expected_train_count])

    with pytest.raises(ValueError, match='Duplicate'):
        whole_subject_split(['01', '02', '01', '03'])


def test_arithmetic_trials_lock_to_stimulus_onset_and_skip_tutorials():
    """Prove epochs start at question onset rather than post-answer marker and exclude tutorials."""
    hz = 250.0
    window_sec = 6.0
    points = int(hz * window_sec)
    n_samples = int(hz * 60.0)
    stim_onset = 20.0
    rt_sec = 2.4
    resp_onset = stim_onset + rt_sec
    next_stim_onset = resp_onset + 6.5

    events = [
        dict(
            onset='5.0',
            trial_type='1_5_2_4',
            marker='1.5-2.4',
            difficulty_range='n/a',
            istutorial='n/a',
            marker_stream='arithmetic-Markers',
        ),
        dict(
            onset='7.0',
            trial_type='1.5-2.4',
            marker='1.5-2.4 Correct tutorial',
            difficulty_range='1.5-2.4',
            outcome='Correct',
            response_accuracy='1',
            istutorial='true',
            marker_stream='arithmetic-Markers',
        ),
        dict(
            onset='15.0',
            trial_type='dropped_samples',
            marker='Dropped 1 samples',
            difficulty_range='n/a',
            istutorial='n/a',
            marker_stream='UoHDataOffsetStream',
        ),
        dict(
            onset=str(stim_onset),
            trial_type='4_2_5_1',
            marker='4.2-5.1',
            difficulty_range='n/a',
            istutorial='n/a',
            marker_stream='arithmetic-Markers',
        ),
        dict(
            onset=str(resp_onset),
            trial_type='4.2-5.1',
            marker='4.2-5.1 Wrong',
            difficulty_range='4.2-5.1',
            outcome='Wrong',
            response_accuracy='0',
            istutorial='false',
            marker_stream='arithmetic-Markers',
        ),
        dict(
            onset=str(next_stim_onset),
            trial_type='0_6_1_5',
            marker='0.6-1.5',
            difficulty_range='n/a',
            istutorial='n/a',
            marker_stream='arithmetic-Markers',
        ),
        dict(
            onset=str(next_stim_onset + 1.5),
            trial_type='0.6-1.5',
            marker='0.6-1.5 Correct',
            difficulty_range='0.6-1.5',
            outcome='Correct',
            response_accuracy='1',
            istutorial='false',
            marker_stream='arithmetic-Markers',
        ),
    ]
    retained = arithmetic_trials(events, hz=hz, points=points, n_samples=n_samples)
    assert len(retained) == 2
    assert retained[0]['start_sample'] == int(round(stim_onset * hz))
    assert retained[0]['end_sample'] - retained[0]['start_sample'] == points
    assert np.isclose(retained[0]['response_time'], rt_sec)
    assert retained[0]['label'] == '5'
    assert retained[1]['label'] == '1'


def test_nback_and_sternberg_trials_enforce_non_overlap_and_bad_trial_exclusion():
    """Verify n-back SOA non-overlap and Sternberg exclusion of bad_trial IDs."""
    hz = 250.0
    nback_points = 375
    n_samples = 4000
    nback_events = [
        dict(onset='1.0', trial_type='1-back', istutorial='true', marker_stream='n-backMarkers'),
        dict(onset='3.0', trial_type='2-back', istutorial='false', marker_stream='n-backMarkers'),
        dict(onset='4.7', trial_type='4-back', istutorial='false', marker_stream='n-backMarkers'),
    ]
    nb = nback_trials(nback_events, hz=hz, points=nback_points, n_samples=n_samples)
    assert [r['label'] for r in nb] == ['2', '4']
    assert nb[0]['end_sample'] <= nb[1]['start_sample']

    stern_points = 500
    stern_events = [
        dict(onset='2.0', task_role='work_memory', trial='1', memory_cond='3'),
        dict(onset='4.3', task_role='probe_target', trial='1', memory_cond='3'),
        dict(onset='8.0', task_role='work_memory', trial='2', memory_cond='7'),
        dict(onset='9.0', task_role='bad_trial', trial='2', memory_cond='7'),
        dict(onset='10.5', task_role='probe_not_shown', trial='2', memory_cond='7'),
    ]
    st = sternberg_trials(stern_events, target_hz=hz, points=stern_points, n_samples=n_samples)
    assert len(st) == 1
    assert st[0]['trial_id'] == '1' and st[0]['label'] == '1'


@pytest.mark.parametrize('source_hz', [500.0, 500.059, 5000.0])
def test_harmonise_sampling_rate_preserves_inband_tone_and_attenuates_above_nyquist(source_hz):
    """Prove both polyphase integer and reflected FFT resampling preserve low-band amplitude and suppress alias frequencies."""
    target_hz = 250.0 if source_hz < 1000.0 else 500.0
    duration_sec = 4.0
    n_channels = 2
    n_source = int(round(source_hz * duration_sec))
    t_src = np.arange(n_source, dtype=np.float64) / source_hz
    inband_freq = 10.0
    alias_freq = target_hz * 0.75  # Above target Nyquist (0.5 * target_hz), below source Nyquist
    clean = np.sin(2.0 * np.pi * inband_freq * t_src)
    high = np.sin(2.0 * np.pi * alias_freq * t_src)
    signal = np.vstack([clean, clean + high])

    out = harmonise_sampling_rate(
        signal,
        source_hz=source_hz,
        target_hz=target_hz,
        allowed_rates=(250.0, 500.0, 500.059, 5000.0),
    )
    expected_len = int(round(n_source * target_hz / source_hz))
    assert out.shape == (n_channels, expected_len)
    t_dst = np.arange(expected_len, dtype=np.float64) / target_hz
    ref_clean = np.sin(2.0 * np.pi * inband_freq * t_dst)
    # Interior region away from boundary transients matches reference tone closely and suppresses alias component
    margin = int(0.25 * target_hz)
    interior = slice(margin, expected_len - margin)
    assert np.max(np.abs(out[0, interior] - ref_clean[interior])) < 2e-2
    assert np.std(out[1, interior] - ref_clean[interior]) < 5e-2


def test_dens_emotion_windows_compute_numeric_quadrants_and_skip_impedance_checks():
    """Verify DENS 1-based sample latencies, numeric circumplex quadrants, neutral filtering, and IBEG/IEND masking."""
    n_channels = 4
    points = 1000
    n_samples = 12000
    rng = np.random.default_rng(42)
    eeg = rng.standard_normal((n_channels, n_samples))

    events = [
        dict(onset='101.0', trial_type='stm', label='neutral_1_1'),
        dict(onset='2501.0', trial_type='vlnc', label='Valence'),
        # Emotional clip 1: samples [3000, 6200) -> 3 windows, middle window [4000, 5000) masked by IBEG..IEND
        dict(onset='3001.0', trial_type='stm', label='17_2'),
        dict(onset='4201.0', trial_type='IBEG', label='Impedances Begin'),
        dict(onset='4501.0', trial_type='IEND', label='Impedances End'),
        dict(onset='6201.0', trial_type='vlnc', label='Valence'),
        # Emotional clip 2: samples [7000, 9200) -> 2 windows, HVLA
        dict(onset='7001.0', trial_type='stm', label='8_3'),
        dict(onset='9201.0', trial_type='vlnc', label='Valence'),
    ]
    beh = [
        dict(stimuliName='neutral_1.mp4', valence='7.0', arousal='3.0'),
        dict(stimuliName='17.mp4', valence='3.2', arousal='7.8'),  # LVHA -> code '3'
        dict(stimuliName='8.m4v', valence='6.5', arousal='2.5'),   # HVLA -> code '2'
    ]
    windows, skipped = dens_emotion_windows(events, beh, eeg, points=points)
    assert skipped == []
    assert [(w['start_sample'], w['quadrant'], w['label']) for w in windows] == [
        (3000, 'LVHA', '3'),
        (5000, 'LVHA', '3'),
        (7000, 'HVLA', '2'),
        (8000, 'HVLA', '2'),
    ]


def test_parse_clateeg_txt_and_fixed_resting_windows_respect_cap_and_zero_dropouts():
    """Verify OpenBCI Cyton columns 1..8 extraction and resting window cap around multi-sample zero dropouts."""
    n_channels = 8
    points = 10
    max_windows = 3
    n_rows = 45
    lines = []
    for i in range(n_rows):
        vals = [0.0] * n_channels if 12 <= i <= 15 else [float(i + c + 1) for c in range(n_channels)]
        cols = [float(i % 256), *vals, 0.04, -0.01, 0.98, 192.0, 1.78e9 + i * 0.004, 0.0]
        lines.append(', '.join(f'{v:.4f}' for v in cols) + ', 2026-06-10 10:32:03.000')
    arr = parse_clateeg_txt('\n'.join(lines).encode('utf-8'))
    assert arr.shape == (n_channels, n_rows)
    assert np.allclose(arr[:, 0], np.arange(1, n_channels + 1, dtype=np.float64))

    windows, zero_runs = fixed_resting_windows(arr, points=points, max_windows=max_windows)
    assert zero_runs == [(12, 16)]
    assert len(windows) == max_windows
    assert [(a, b) for _, a, b in windows] == [(0, 10), (16, 26), (26, 36)]


def test_decode_eeglab_set_handles_matlab_v73_userblock_and_fdt():
    """Verify MATLAB v7.3 512-byte userblock HDF5 .set files with paired .fdt decode in Fortran sample order."""
    n_channels = 3
    n_samples = 6
    srate = 250.0
    ch_labels = ['FP1', 'FZ', 'CZ']
    expected = np.arange(n_channels * n_samples, dtype=np.float32).reshape(n_channels, n_samples)

    with tempfile.TemporaryDirectory() as tmp:
        set_path = Path(tmp) / 'sample_eeg.set'
        fdt_path = Path(tmp) / 'sample_eeg.fdt'
        # EEGLAB writes float32 samples in Fortran (column-major) order: all channels for sample 0, then sample 1, ...
        expected.T.astype('<f4').tofile(fdt_path)
        with h5py.File(set_path, 'w', userblock_size=512) as h5:
            h5.create_dataset('srate', data=np.array([[srate]]))
            h5.create_dataset('nbchan', data=np.array([[float(n_channels)]]))
            h5.create_dataset('pnts', data=np.array([[float(n_samples)]]))
            h5.create_dataset('trials', data=np.array([[1.0]]))
            d_str = h5.create_dataset('data', data=np.array([ord(c) for c in 'sample_eeg.fdt'], dtype=np.uint16))
            d_str.attrs.create('MATLAB_class', np.bytes_('char'))
            locs = h5.create_group('chanlocs')
            refs = []
            for idx, lbl in enumerate(ch_labels):
                ds = h5.create_dataset(f'_lbl_{idx}', data=np.array([ord(c) for c in lbl], dtype=np.uint16))
                ds.attrs.create('MATLAB_class', np.bytes_('char'))
                refs.append(ds.ref)
            locs.create_dataset('labels', data=np.array(refs, dtype=h5py.ref_dtype))
        with set_path.open('r+b') as raw:
            raw.write(b'MATLAB 7.3 MAT-file, Platform: GLNXA64')

        decoded, info = decode_eeglab_set(set_path)
        assert decoded.shape == (n_channels, n_samples)
        assert np.array_equal(decoded, expected.astype(np.float64))
        assert info['srate'] == srate
        assert info['ch_names'] == ch_labels
        assert info['storage'] == 'paired_fdt_hdf5_set'


def test_nback_records_corrects_upstream_1e6_scale_and_rejects_already_calibrated_inputs():
    """Prove nback_records rescales ds007169 v1.0.5 1e6*uV BrainVision channels to uV and guards against double-scaling."""
    import json

    n_eeg = 19
    n_aux = 5
    n_total = n_eeg + n_aux
    hz = 250.0
    points = 375
    n_samples = 1200
    resolution = 0.1
    eeg_names = [f'EEG{i + 1:02d}' for i in range(n_eeg)]
    aux_names = [f'D{i + 1}' for i in range(n_aux)]
    all_names = eeg_names + aux_names
    rng = np.random.default_rng(99)
    true_uV = rng.normal(loc=2.5e5, scale=50.0, size=(n_eeg, n_samples))
    aux_vals = rng.normal(loc=0.0, scale=100.0, size=(n_aux, n_samples))

    def _populate_uoh_cache(root: Path, upstream_gain: float) -> None:
        (root / 'participants.tsv').write_text(
            'participant_id\tage\tsex\thandedness\tanalysis_included\tdata_quality_overall\n'
            'sub-001\t24\tF\tR\ttrue\tpass\n',
            encoding='utf-8',
        )
        eeg_dir = root / 'sub-001' / 'eeg'
        eeg_dir.mkdir(parents=True, exist_ok=True)
        stem = eeg_dir / 'sub-001_task-nback_'

        stored_phys = np.vstack([true_uV * upstream_gain, aux_vals])
        # BrainVision binary stores digital values such that physical = digital * resolution
        digital_f32 = (stored_phys / resolution).T.astype('<f4')
        Path(str(stem) + 'eeg.eeg').write_bytes(digital_f32.tobytes())

        ch_infos = '\n'.join(
            f'Ch{i + 1}={name},,{resolution},{"µV" if i < n_eeg else "n/a"}'
            for i, name in enumerate(all_names)
        )
        Path(str(stem) + 'eeg.vhdr').write_text(
            'Brain Vision Data Exchange Header File Version 1.0\n'
            '[Common Infos]\n'
            'DataFile=sub-001_task-nback_eeg.eeg\n'
            'DataFormat=BINARY\n'
            'DataOrientation=MULTIPLEXED\n'
            f'NumberOfChannels={n_total}\n'
            f'SamplingInterval={int(1e6 / hz)}\n'
            '[Binary Infos]\n'
            'BinaryFormat=IEEE_FLOAT_32\n'
            '[Channel Infos]\n'
            f'{ch_infos}\n',
            encoding='utf-8',
        )
        ch_rows = ['name\ttype\tunits'] + [
            f'{name}\t{"EEG" if i < n_eeg else "MISC"}\t{"µV" if i < n_eeg else "n/a"}'
            for i, name in enumerate(all_names)
        ]
        Path(str(stem) + 'channels.tsv').write_text('\n'.join(ch_rows) + '\n', encoding='utf-8')
        Path(str(stem) + 'eeg.json').write_text(json.dumps({'SamplingFrequency': hz}), encoding='utf-8')
        Path(str(stem) + 'events.tsv').write_text(
            'onset\tduration\ttrial_type\tistutorial\tmarker_stream\tnback_level\tmatched\tkey_press\toutcome\tresponse_accuracy\n'
            '0.4\t1.5\t1-back\tfalse\tn-backMarkers\t1\ttrue\tspace\tCorrect\t1\n'
            '2.2\t1.5\t3-back\tfalse\tn-backMarkers\t3\tfalse\tnone\tCorrect\t1\n',
            encoding='utf-8',
        )

    with tempfile.TemporaryDirectory() as tmp_v105:
        cache_v105 = Path(tmp_v105)
        _populate_uoh_cache(cache_v105, upstream_gain=1.0 / DS007169_V105_EEG_SCALE)
        raw_decoded, _ = decode_brainvision(cache_v105 / 'sub-001/eeg/sub-001_task-nback_eeg.vhdr')
        assert np.median(raw_decoded[:n_eeg].std(axis=1)) > 1e7

        spec = dict(points=points, metadata_updates={})
        batches = list(nback_records(cache_v105, spec))
        assert len(batches) == 1
        epochs_arr, rows, names, info = batches[0]
        assert names == eeg_names
        assert epochs_arr.shape == (2, n_eeg, points)
        assert [r['label'] for r in rows] == ['1', '3']
        assert info['upstream_eeg_scale_correction'] == DS007169_V105_EEG_SCALE
        # Restored float64 epochs match original true_uV within float32 storage precision
        start0 = rows[0]['start_sample']
        np.testing.assert_allclose(epochs_arr[0], true_uV[:, start0 : start0 + points], rtol=1e-6, atol=1e-2)
        assert 20.0 <= float(np.median(epochs_arr.std(axis=2))) <= 100.0

    with tempfile.TemporaryDirectory() as tmp_patched:
        cache_patched = Path(tmp_patched)
        _populate_uoh_cache(cache_patched, upstream_gain=1.0)
        spec_patched = dict(points=points, metadata_updates={})
        with pytest.raises(ValueError, match='Expected unpatched'):
            list(nback_records(cache_patched, spec_patched))


