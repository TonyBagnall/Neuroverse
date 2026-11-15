"""Unit tests for Diagnostic-to-30 Phase 1 decoders, splitters, and window selectors."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

import tempfile

from tools.prepare_diagnostic_to_30 import (
    DS003523_EXCLUSIONS,
    DS004367_CANONICAL_61,
    DS004367_FIXED_59,
    DS004367_FIXED_INDICES,
    SIENA_CANONICAL_29,
    build_spec,
    decode_brainvision_safe,
    extract_clean_windows,
    parse_siena_seizure_list,
    plan_ds003523_participants,
    select_ds003506_session,
    select_ds004367_channels,
    select_siena_channels,
    select_siena_windows,
    stratified_diagnostic_split,
    whole_patient_split,
)


def test_stratified_diagnostic_split_preserves_class_ratios_and_disjoint_subjects():
    """Verify deterministic 75/25 stratified split per diagnostic class without overlap."""
    n_cases_cls1 = 19
    n_cases_cls2 = 24
    train_ratio = 0.75
    mapping = {
        **{f'sub-P{i:02d}': '1' for i in range(n_cases_cls1)},
        **{f'sub-C{i:02d}': '2' for i in range(n_cases_cls2)},
    }

    train, test = stratified_diagnostic_split(mapping, train_ratio=train_ratio)

    expected_train_c1 = int(n_cases_cls1 * train_ratio)
    expected_train_c2 = int(n_cases_cls2 * train_ratio)
    expected_total_train = expected_train_c1 + expected_train_c2
    expected_total_test = (n_cases_cls1 + n_cases_cls2) - expected_total_train

    assert train.isdisjoint(test)
    assert len(train) == expected_total_train
    assert len(test) == expected_total_test
    assert sum(1 for s in train if mapping[s] == '1') == expected_train_c1
    assert sum(1 for s in train if mapping[s] == '2') == expected_train_c2

    # Verify determinism regardless of input insertion order
    reversed_mapping = dict(reversed(list(mapping.items())))
    train_rev, test_rev = stratified_diagnostic_split(reversed_mapping, train_ratio=train_ratio)
    assert train_rev == train
    assert test_rev == test


def test_whole_patient_split_partitions_sorted_patients_cleanly():
    """Verify 14-patient epilepsy cohort splits into first 10 train and last 4 test."""
    n_patients = 14
    train_ratio = 0.75
    patients = [f'PN{i:02d}' for i in range(n_patients)]

    train, test = whole_patient_split(patients, train_ratio=train_ratio)
    expected_train = int(round(n_patients * train_ratio))

    assert train.isdisjoint(test)
    assert sorted(train) == patients[:expected_train]
    assert sorted(test) == patients[expected_train:]


def test_extract_clean_windows_rejects_boundaries_railed_channels_and_zero_dropouts():
    """Verify boundary-crossing, flat-channel, and zero-dropout windows are skipped up to cap."""
    n_channels = 4
    points = 200
    n_blocks = 6
    max_windows = 2
    total_samples = n_blocks * points
    rng = np.random.default_rng(12345)
    data = rng.normal(loc=0.0, scale=25.0, size=(n_channels, total_samples))

    # Block 0: crosses a segment boundary inside (0, points)
    boundary_sample = points // 2
    # Block 1: clean -> should be accepted (#1)
    # Block 2: one channel railed flat
    data[1, 2 * points : 3 * points] = 5.0
    # Block 3: >5% all-channel simultaneous zero dropout
    dropout_len = points // 5
    data[:, 3 * points : 3 * points + dropout_len] = 0.0
    # Block 4: clean -> should be accepted (#2, hitting max_windows)
    # Block 5: clean -> should not be reached because max_windows is met

    candidates = [b * points for b in range(n_blocks)]
    seen: set[str] = set()
    wins, starts, dropped = extract_clean_windows(
        data,
        candidates,
        points=points,
        boundary_samples=[boundary_sample],
        max_windows=max_windows,
        seen_hashes=seen,
    )

    assert len(wins) == max_windows
    assert starts == [1 * points, 4 * points]
    for w, s in zip(wins, starts, strict=True):
        assert w.shape == (n_channels, points)
        np.testing.assert_array_equal(w, data[:, s : s + points])

    reasons = {d['start_sample']: d['reason'] for d in dropped}
    assert reasons[0 * points] == 'crosses_boundary'
    assert reasons[2 * points] == 'flat_or_railed_channel'
    assert reasons[3 * points] == 'zero_dropout'


def test_parse_siena_seizure_list_handles_midnight_rollover_and_documented_typos():
    """Verify Siena parser handles PN01/PNO6 filenames, midnight rollover, and PN00-3 typo."""
    edf_headers = {
        'PN00/PN00-3.edf': {'start_time': '18.15.44', 'duration_s': 2509.0},
        'PN01/PN01-1.edf': {'start_time': '19.00.44', 'duration_s': 48537.0},
        'PN06/PN06-2.edf': {'start_time': '21.11.29', 'duration_s': 12620.0},
        'PN10/PN10-2.edf': {'start_time': '09.30.15', 'duration_s': 8534.0},
    }

    txt_pn00 = (
        'Seizure n 3\n'
        'File name: PN00-3.edf\n'
        'Registration start time: 18.15.44\n'
        'Registration end time: 18.57.13\n'
        'Seizure start time: 18.28.29\n'
        'Seizure end time: 19.29.29\n'
    )
    parsed_00 = parse_siena_seizure_list(txt_pn00, 'PN00', edf_headers)
    assert parsed_00['PN00/PN00-3.edf'] == [(765.0, 825.0)]

    txt_pn01 = (
        'File name: PN01.edf\n'
        'Registration start time: 19.00.44\n'
        'Registration end time: 08.29.41\n'
        'Seizure n 1\n'
        'Start time: 21:51:02\n'
        'End time: 21.51.56\n'
        'Seizure n 2\n'
        'Start time: 07.53.17\n'
        'End time: 07.54.31\n'
    )
    parsed_01 = parse_siena_seizure_list(txt_pn01, 'PN01', edf_headers)
    assert parsed_01['PN01/PN01-1.edf'] == [(10218.0, 10272.0), (46353.0, 46427.0)]

    txt_pn06 = (
        'Seizure n 2:\n'
        'File name: PNO6-2.edf\n'
        'Registration start time: 21.11.29\n'
        'Registration end time: 00.41.49\n'
        'Seizure start time: 23.39.09\n'
        'Seizure end time: 23.40.18\n'
    )
    parsed_06 = parse_siena_seizure_list(txt_pn06, 'PN06', edf_headers)
    assert parsed_06['PN06/PN06-2.edf'] == [(8860.0, 8929.0)]

    txt_pn10 = (
        'Seizure n 2 (seizure B1):\n'
        'File name: PN10-2.edf\n'
        'Registration start time: 09.30.15\n'
        'Seizure start time: 11.40.13\n'
        'Seizure end time: 11.41.04 opure 11.40.43\n'
    )
    parsed_10 = parse_siena_seizure_list(txt_pn10, 'PN10', edf_headers)
    assert parsed_10['PN10/PN10-2.edf'] == [(7798.0, 7849.0)]


def test_select_siena_windows_enforces_peri_ictal_buffer_and_interictal_cap():
    """Verify ictal windows require >=2 s overlap and interictal windows stay >=60 s away."""
    n_channels = 3
    srate = 64.0
    win_sec = 4
    points = int(win_sec * srate)
    duration_sec = 300
    buffer_sec = 60.0
    max_interictal = 5
    seizure_start_sec = 100.0
    seizure_end_sec = 110.0

    rng = np.random.default_rng(99)
    data = rng.normal(0.0, 20.0, size=(n_channels, int(duration_sec * srate)))

    wins, meta, _ = select_siena_windows(
        data,
        srate=srate,
        points=points,
        seizures_sec=[(seizure_start_sec, seizure_end_sec)],
        max_interictal=max_interictal,
        buffer_sec=buffer_sec,
        min_ictal_overlap_sec=2.0,
    )

    ictal = [s for s, cls in meta if cls == '1']
    inter = [s for s, cls in meta if cls == '2']
    # [100,104) has 4 s overlap; [104,108) has 4 s overlap; [108,112) has 2 s overlap -> 3 ictal windows
    expected_ictal_starts = [int(t * srate) for t in (100, 104, 108)]
    assert ictal == expected_ictal_starts
    assert len(inter) == max_interictal
    assert len(wins) == len(ictal) + len(inter)

    for s in inter:
        w0 = s / srate
        w1 = (s + points) / srate
        assert (w1 <= seizure_start_sec - buffer_sec) or (w0 >= seizure_end_sec + buffer_sec)


@pytest.mark.parametrize(
    ('ch16', 'ch34', 'ch42', 'ch62'),
    [
        ('F7', 'TP7', 'TP8', 'AFz'),
        ('F7', 'AFz', 'TP8', '63'),
        ('F7', 'TP8', 'AFz', '63'),
        ('FC7', 'TP8', 'AFz', '63'),
        ('FT7', 'TP8', 'AFz', '63'),
    ],
)
def test_select_ds004367_channels_aligns_all_five_cap_layouts(ch16, ch34, ch42, ch62):
    """Verify all 5 upstream ds004367 68-channel layouts map onto identical 61 scalp channels."""
    total_native = 68
    names = [''] * total_native
    for idx, fixed_name in zip(DS004367_FIXED_INDICES, DS004367_FIXED_59, strict=True):
        names[idx] = fixed_name
    names[16] = ch16
    names[34] = ch34
    names[42] = ch42
    names[62] = ch62
    names[63:68] = ['A2', 'GSR1', 'GSR2', 'ECG1', 'ECG2']

    picks, canon = select_ds004367_channels(names)

    assert canon == DS004367_CANONICAL_61
    assert len(picks) == len(DS004367_CANONICAL_61)
    assert [names[i] for i in picks] == DS004367_CANONICAL_61
    assert 16 not in picks


def test_select_siena_channels_matches_case_insensitive_reordered_edf_headers():
    """Verify PN14-style uppercase/reordered Siena EDF headers align to canonical 29 order."""
    pn14_labels = [
        'EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T5',
        'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fz', 'EEG Pz',
        'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6',
        'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10', '1', '2',
        'EKG 1', 'EKG 2', 'EEG FP2', 'EEG P9', 'EEG P10', 'SPO2', 'HR',
        'EEG T3', 'EEG CZ', 'MK',
    ]

    picks = select_siena_channels(pn14_labels)

    assert len(picks) == len(SIENA_CANONICAL_29)
    aligned_short = [pn14_labels[i][4:].strip().upper() for i in picks]
    assert aligned_short == [c.upper() for c in SIENA_CANONICAL_29]


def test_build_spec_shares_mutable_state_across_shallow_copy():
    """Verify reader updates on a shallow copy of spec remain visible to splitter and caller."""
    with tempfile.TemporaryDirectory() as tmp:
        spec, _ = build_spec('pardogarcia2026', Path(tmp), deferred_files=[])
        shallow_copy = dict(spec)
        expected_train = {'PAC01', 'CTL01'}
        expected_files = [{'name': 'PAC01.eeg', 'bytes': 1024}]
        shallow_copy['_state']['train_subjects'] = expected_train
        shallow_copy['_state']['all_source_files'] = expected_files

        assert shallow_copy['splitter'](sorted(expected_train | {'PAC02'}), []) == expected_train
        assert spec['_state']['all_source_files'] == expected_files


def test_decode_brainvision_safe_resolves_stale_datafile_without_mutating_header():
    """Verify stale BrainVision DataFile pointer falls back to BIDS .eeg and preserves header bytes."""
    n_channels = 3
    n_samples = 16
    resolution_uv = 0.25
    stale_name = '019_P1.dat'
    stem = 'sub-042_task-proposer_run-1_eeg'

    raw_ints = np.arange(n_channels * n_samples, dtype='<i2').reshape(n_channels, n_samples)
    vhdr_lines = [
        'Brain Vision Data Exchange Header File Version 2.0',
        '[Common Infos]',
        f'DataFile={stale_name}',
        f'MarkerFile={stem}.vmrk',
        'DataFormat=BINARY',
        'DataOrientation=VECTORIZED',
        f'NumberOfChannels={n_channels}',
        f'DataPoints={n_samples}',
        'SamplingInterval=500.0',
        '[Binary Infos]',
        'BinaryFormat=INT_16',
        'UseBigEndianOrder=NO',
        '[Channel Infos]',
        *[f'Ch{i + 1}=E{i + 1},Mastoid,{resolution_uv},uV' for i in range(n_channels)],
    ]
    vhdr_text = '\n'.join(vhdr_lines) + '\n'

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vhdr_path = root / f'{stem}.vhdr'
        eeg_path = root / f'{stem}.eeg'
        vhdr_path.write_text(vhdr_text, encoding='utf-8')
        raw_ints.tofile(eeg_path)
        original_vhdr_bytes = vhdr_path.read_bytes()

        decoded, meta = decode_brainvision_safe(vhdr_path, eeg_path)

        assert vhdr_path.read_bytes() == original_vhdr_bytes
        assert list(root.glob('*.vhdr')) == [vhdr_path]
        assert decoded.shape == (n_channels, n_samples)
        np.testing.assert_allclose(decoded, raw_ints.astype(np.float64) * resolution_uv)
        assert meta['stale_vhdr_datafile'] == stale_name
        assert meta['native_data_file'] == eeg_path.name


def test_select_ds003506_session_selects_off_medication_and_rejects_ambiguous_rows():
    """Verify PD OFF-medication session routing and error on missing/dual OFF flags."""
    assert select_ds003506_session(
        {'participant_id': 'sub-001', 'Group': 'PD', 'sess1_Med': 'OFF', 'sess2_Med': 'ON'}
    ) == ('ses-01', 'PD_OFF', '1')
    assert select_ds003506_session(
        {'participant_id': 'sub-002', 'Group': 'PD', 'sess1_Med': 'ON', 'sess2_Med': 'OFF'}
    ) == ('ses-02', 'PD_OFF', '1')
    assert select_ds003506_session(
        {'participant_id': 'sub-030', 'Group': 'CTL', 'sess1_Med': 'n/a', 'sess2_Med': 'n/a'}
    ) == ('ses-01', 'CTL', '2')

    with pytest.raises(ValueError, match='Ambiguous OFF-medication'):
        select_ds003506_session(
            {'participant_id': 'sub-003', 'Group': 'PD', 'sess1_Med': 'ON', 'sess2_Med': 'ON'}
        )


def test_plan_ds003523_participants_applies_author_exclusions_and_enforces_parity_and_uniqueness():
    """Verify duplicate sub-037 and clinical exclusions are dropped and parity/uniqueness hold."""
    rows = [
        {'participant_id': 'sub-001', 'Original_ID': '3044', 'URSI': 'M87102477', 'Group': '0'},
        {'participant_id': 'sub-002', 'Original_ID': '3009', 'URSI': 'M87104171', 'Group': '1'},
        {'participant_id': 'sub-036', 'Original_ID': '3013', 'URSI': 'M87138342', 'Group': '1'},
        # Duplicate Original_ID=3013 -> excluded via DS003523_EXCLUSIONS
        {'participant_id': 'sub-037', 'Original_ID': '3013', 'URSI': 'M87138432', 'Group': '1'},
        # Author clinical exclusion (no LOC)
        {'participant_id': 'sub-064', 'Original_ID': '3004', 'URSI': 'M87169117', 'Group': '0'},
    ]
    retained, excluded = plan_ds003523_participants(rows)

    expected_retained_count = len(rows) - 2
    assert len(retained) == expected_retained_count
    assert [r[0] for r in retained] == ['sub-001', 'sub-002', 'sub-036']
    assert [r[2] for r in retained] == ['1', '2', '2']
    assert {e['participant_id'] for e in excluded} == {'sub-037', 'sub-064'} & set(DS003523_EXCLUSIONS)

    # Unlisted duplicate Original_ID must raise ValueError
    dup_rows = rows[:2] + [
        {'participant_id': 'sub-099', 'Original_ID': '3044', 'URSI': 'M99999999', 'Group': '0'}
    ]
    with pytest.raises(ValueError, match='Unresolved duplicate'):
        plan_ds003523_participants(dup_rows)


def test_plan_ds006260_participants_separates_math_and_reading_cohorts_and_excludes_missing_baseline():
    """Verify sub-M* vs sub-R* routing on ses-1/run-1 and documented exclusion of sub-Mc22."""
    from tools.prepare_diagnostic_to_30 import plan_ds006260_participants

    rows = [
        {'participant_id': 'sub-Mc01', 'Group': 'cont', 'Age': '9', 'Gender': 'M'},
        {'participant_id': 'sub-Me01', 'Group': 'exp', 'Age': '10', 'Gender': 'F'},
        {'participant_id': 'sub-Rc01', 'Group': 'cont', 'Age': '8', 'Gender': 'M'},
        {'participant_id': 'sub-Mc22', 'Group': 'cont', 'Age': '11', 'Gender': 'M'},
    ]
    available = {
        f'{pid}/ses-1/eeg/{pid}_ses-1_task-SmartickDataset_run-1_eeg.set'
        for pid in ('sub-Mc01', 'sub-Me01', 'sub-Rc01')
    }
    retained, excluded = plan_ds006260_participants(rows, available)

    expected_retained = len(rows) - 1
    assert len(retained) == expected_retained
    assert [(pid, lbl) for pid, _, lbl, _ in retained] == [
        ('sub-Mc01', '1'),
        ('sub-Me01', '1'),
        ('sub-Rc01', '2'),
    ]
    assert [e['participant_id'] for e in excluded] == ['sub-Mc22']

    with pytest.raises(ValueError, match='Inconsistent Group'):
        plan_ds006260_participants(
            [{'participant_id': 'sub-Mc01', 'Group': 'exp', 'Age': '9', 'Gender': 'M'}],
            available,
        )


def test_plan_ds007427_participants_retains_raw_kindred_and_excludes_sporadic_and_derivatives_only():
    """Verify G1/G2 raw Antioquia kindred selection and exclusion of sporadic sub-CTR011 and derivatives-only rows."""
    from tools.prepare_diagnostic_to_30 import plan_ds007427_participants

    rows = [
        {'participant_id': 'sub-CTR001', 'age': '55.0', 'sex': 'M', 'MM_total': '30.0'},
        {'participant_id': 'sub-CTR011', 'age': '50.0', 'sex': 'F', 'MM_total': '29.0'},
        {'participant_id': 'sub-G1001', 'age': '29.0', 'sex': 'F', 'MM_total': '30.0'},
        {'participant_id': 'sub-G1002', 'age': '31.0', 'sex': 'M', 'MM_total': '29.0'},
        {'participant_id': 'sub-G2001', 'age': '30.0', 'sex': 'F', 'MM_total': '30.0'},
    ]
    available = {
        f'{pid}/ses-V0/eeg/{pid}_ses-V0_task-CE_eeg.eeg'
        for pid in ('sub-CTR011', 'sub-G1001', 'sub-G2001')
    }
    retained, excluded = plan_ds007427_participants(rows, available)

    assert [(pid, lbl) for pid, _, lbl, _ in retained] == [
        ('sub-G1001', '1'),
        ('sub-G2001', '2'),
    ]
    assert {e['participant_id'] for e in excluded} == {'sub-CTR001', 'sub-CTR011', 'sub-G1002'}

    with pytest.raises(ValueError, match='Unaccounted raw .eeg'):
        plan_ds007427_participants(
            [{'participant_id': 'sub-DTA001', 'age': '60.0', 'sex': 'M', 'MM_total': '20.0'}],
            {'sub-DTA001/ses-V0/eeg/sub-DTA001_ses-V0_task-CE_eeg.eeg'},
        )




