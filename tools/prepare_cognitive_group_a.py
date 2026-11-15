"""Prepare Group A (+7) cognitive and affective scalp-EEG benchmark problems."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import configparser
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from tools import prepare_open_moabb as core
from tools.download_eegdash_diagnosis import blob_sha
from tools.prepare_moabb import dump, sha
from tools.prepare_small_moabb import fetch
from tools.update_dataset_sidecars import finish

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / 'docs/source-evidence/cognitive-affective-review'

ARITHMETIC_LABELS = {
    '0.6-1.5': ('1', '0.6_to_1.5'),
    '1.5-2.4': ('2', '1.5_to_2.4'),
    '2.4-3.3': ('3', '2.4_to_3.3'),
    '3.3-4.2': ('4', '3.3_to_4.2'),
    '4.2-5.1': ('5', '4.2_to_5.1'),
    '5.1-6.0': ('6', '5.1_to_6.0'),
    '6.0-6.9': ('7', '6.0_to_6.9'),
}
NBACK_LABELS = {
    '1-back': ('1', '1_back'),
    '2-back': ('2', '2_back'),
    '3-back': ('3', '3_back'),
    '4-back': ('4', '4_back'),
}
STERNBERG_LABELS = {
    '3': ('1', 'load_3'),
    '5': ('2', 'load_5'),
    '7': ('3', 'load_7'),
}
MATB_LABELS = {
    'MATBeasy': ('1', 'easy'),
    'MATBmed': ('2', 'medium'),
    'MATBdiff': ('3', 'difficult'),
}
CLATEEG_LABELS = {
    'neutral': ('1', 'neutral'),
    'lowlevel': ('2', 'low'),
    'midlevel': ('3', 'medium'),
    'highlevel': ('4', 'high'),
}
CLATEEG_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C4']
DENS_QUADRANTS = {
    'HVHA': ('1', 'HVHA'),
    'HVLA': ('2', 'HVLA'),
    'LVHA': ('3', 'LVHA'),
    'LVLA': ('4', 'LVLA'),
}
SLEEP_LABELS = {
    'ses-1': ('1', 'normal_sleep'),
    'ses-2': ('2', 'sleep_deprived'),
}

OPENNEURO_RELEASES = {
    'arithmetic_difficulty': dict(
        dataset='ds007262',
        ref='1.1.0',
        commit='b156c6185ba1704107305eb5ab049d090053ddcf',
        licence='CC0-1.0',
    ),
    'nback_difficulty': dict(
        dataset='ds007169',
        ref='1.0.5',
        commit='08649287ba6805b268bb24c507cbda533860ace4',
        licence='CC0-1.0',
    ),
    'sternberg_memory': dict(
        dataset='ds004117',
        ref='1.0.1',
        commit='065e38865296e7707166eb3b1b561044ac62ab4c',
        licence='CC0-1.0',
    ),
    'naturalistic_emotion': dict(
        dataset='ds003751',
        ref='1.0.6',
        commit='5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c',
        licence='CC0-1.0',
    ),
    'sleep_deprivation': dict(
        dataset='ds004902',
        ref='1.0.8',
        commit='fd8c444dc1c1945b2d5add4d858e68df1c925854',
        licence='CC0-1.0',
    ),
}

GROUP_A_ORDER = [
    'arithmetic_difficulty',
    'nback_difficulty',
    'multitask_workload',
    'cognitive_load_tasks',
    'sternberg_memory',
    'naturalistic_emotion',
    'sleep_deprivation',
]


def verify_final(result: dict) -> dict:
    folder = Path(result['path'])
    manifest_path = folder / f"{result['dataset']}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative, expected in manifest['artifacts'].items():
        p = (folder / relative).resolve()
        if not p.is_relative_to(folder.resolve()):
            raise ValueError(f'Unsafe manifest artifact path: {p}')
        if p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError(f'Final artifact mismatch: {p}')
    return result


def whole_subject_split(people: list[str], train_ratio: float = 0.75) -> tuple[set[str], list[str]]:
    unique_sorted = sorted({str(p) for p in people})
    if len(unique_sorted) != len(people):
        raise ValueError('Duplicate participant identifier in split input')
    train_count = int(np.floor(train_ratio * len(unique_sorted)))
    if train_count < 2 or len(unique_sorted) - train_count < 1:
        raise ValueError('Insufficient participant count for whole-subject split')
    return set(unique_sorted[:train_count]), unique_sorted


def harmonise_sampling_rate(
    data: np.ndarray,
    source_hz: float,
    target_hz: float,
    allowed_rates: tuple[float, ...],
) -> np.ndarray:
    """Whole-recording linear-phase FIR anti-alias resampling onto target_hz."""
    if not any(np.isclose(source_hz, known, rtol=1e-4, atol=1e-2) for known in allowed_rates):
        raise ValueError(f'Unreviewed native sampling frequency: {source_hz}')
    values = np.asarray(data, dtype='<f8')
    if np.isclose(source_hz, target_hz, rtol=1e-9, atol=0.0):
        return values
    frac = Fraction(float(target_hz) / float(source_hz)).limit_denominator(10000)
    return np.asarray(
        resample_poly(values, up=frac.numerator, down=frac.denominator, axis=-1, padtype='line'),
        dtype='<f8',
    )


def nonzero_segments(eeg_data: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split continuous recordings around multi-sample all-channel zero runs."""
    zero_mask = np.all(eeg_data == 0, axis=0)
    n_samples = int(eeg_data.shape[1])
    idx = np.flatnonzero(zero_mask)
    if not len(idx):
        return [(0, n_samples)], []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    zero_runs = [(int(r[0]), int(r[-1]) + 1) for r in np.split(idx, splits) if len(r) >= 2]
    if not zero_runs:
        return [(0, n_samples)], []
    segments = []
    cursor = 0
    for a, b in zero_runs:
        if cursor < a:
            segments.append((cursor, a))
        cursor = b
    if cursor < n_samples:
        segments.append((cursor, n_samples))
    return segments, zero_runs


def decode_brainvision(vhdr_path: Path) -> tuple[np.ndarray, dict]:
    """Decode BrainVision .vhdr/.eeg (MULTIPLEXED or VECTORIZED) in float64 microvolts."""
    text = Path(vhdr_path).read_text(encoding='utf-8-sig')
    ini_body = text[text.index('[Common Infos]') :].split('\n[Comment]')[0]
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    config.read_string(ini_body)
    common = config['Common Infos']
    binary = config['Binary Infos']
    if common.get('DataFormat') != 'BINARY' or binary.get('UseBigEndianOrder', 'NO') != 'NO':
        raise ValueError(f'Unreviewed BrainVision DataFormat or endianness in {vhdr_path}')
    orientation = common.get('DataOrientation', '')
    if orientation not in {'MULTIPLEXED', 'VECTORIZED'}:
        raise ValueError(f'Unreviewed BrainVision DataOrientation {orientation} in {vhdr_path}')
    bfmt = binary.get('BinaryFormat', '')
    if bfmt == 'IEEE_FLOAT_32':
        dtype = '<f4'
        itemsize = 4
    elif bfmt == 'INT_16':
        dtype = '<i2'
        itemsize = 2
    else:
        raise ValueError(f'Unreviewed BrainVision BinaryFormat {bfmt} in {vhdr_path}')
    n_channels = int(common['NumberOfChannels'])
    hz = 1e6 / float(common['SamplingInterval'])
    names = []
    resolutions = []
    units = []
    for i in range(n_channels):
        fields = config['Channel Infos'][f'Ch{i + 1}'].split(',')
        names.append(fields[0])
        res = fields[2].strip() if len(fields) > 2 else ''
        unit = fields[3].strip() if len(fields) > 3 else ''
        resolutions.append(float(res) if res else 1.0)
        units.append(unit)
    data_file = Path(vhdr_path).parent / common['DataFile']
    if data_file.resolve().parent != Path(vhdr_path).parent.resolve():
        raise ValueError(f'Unsafe BrainVision DataFile pointer in {vhdr_path}')
    total_bytes = data_file.stat().st_size
    if total_bytes % (n_channels * itemsize) != 0:
        raise ValueError(f'BrainVision byte size not divisible by channel frame in {data_file}')
    n_samples = total_bytes // (n_channels * itemsize)
    if 'DataPoints' in common and int(common['DataPoints']) != n_samples:
        raise ValueError(f'BrainVision DataPoints mismatch in {vhdr_path}')
    mm = np.memmap(data_file, dtype=dtype, mode='r')
    if orientation == 'MULTIPLEXED':
        arr = np.asarray(mm.reshape(n_samples, n_channels).T, dtype='<f8')
    else:
        arr = np.asarray(mm.reshape(n_channels, n_samples), dtype='<f8')
    del mm
    scale = np.asarray(resolutions, dtype='<f8')[:, None]
    return arr * scale, dict(
        native_channel_names=names,
        native_resolutions=resolutions,
        native_declared_units=units,
        native_sampling_frequency_hz=hz,
        native_samples=n_samples,
        native_channels=n_channels,
        native_orientation=orientation,
        native_binary_format=bfmt,
        native_data_file=data_file.name,
    )


def _h5_str_or_val(f: h5py.File, ds):
    if isinstance(ds, h5py.Reference):
        if not ds:
            return None
        ds = f[ds]
    if ds.attrs.get('MATLAB_empty', 0):
        return None
    arr = np.asarray(ds)
    if ds.attrs.get('MATLAB_class') == b'char':
        return ''.join(chr(int(v)) for v in arr.ravel())
    if arr.dtype == object:
        return [_h5_str_or_val(f, r) for r in arr.ravel()]
    if arr.size == 1:
        return arr.item()
    return arr.tolist()


def decode_eeglab_set(set_path: Path) -> tuple[np.ndarray, dict]:
    """Decode EEGLAB .set (MATLAB v5 or HDF5 v7.3, embedded or paired .fdt) as float64."""
    fdt_path = Path(set_path).with_suffix('.fdt')
    if h5py.is_hdf5(set_path):
        with h5py.File(set_path, 'r') as f:
            g = f['EEG'] if 'EEG' in f else f
            srate = float(np.asarray(g['srate']).item())
            nbchan = int(round(float(np.asarray(g['nbchan']).item())))
            pnts = int(round(float(np.asarray(g['pnts']).item())))
            trials = int(round(float(np.asarray(g['trials']).item())))
            ref_val = _h5_str_or_val(f, g['ref']) if 'ref' in g else ''
            locs = g['chanlocs']
            labels_refs = np.asarray(locs['labels']).ravel()
            ch_names = [str(_h5_str_or_val(f, r)) for r in labels_refs]
            ch_types = (
                [str(_h5_str_or_val(f, r) or '') for r in np.asarray(locs['type']).ravel()]
                if 'type' in locs
                else [''] * len(ch_names)
            )
            events = []
            if 'event' in g and isinstance(g['event'], h5py.Group) and 'latency' in g['event']:
                lat_refs = np.asarray(g['event']['latency']).ravel()
                typ_refs = np.asarray(g['event']['type']).ravel() if 'type' in g['event'] else []
                dur_refs = np.asarray(g['event']['duration']).ravel() if 'duration' in g['event'] else []
                for idx in range(len(lat_refs)):
                    ev = dict(
                        latency=float(_h5_str_or_val(f, lat_refs[idx])),
                        type=str(_h5_str_or_val(f, typ_refs[idx])) if idx < len(typ_refs) else '',
                    )
                    if idx < len(dur_refs):
                        dv = _h5_str_or_val(f, dur_refs[idx])
                        ev['duration'] = float(dv) if isinstance(dv, (int, float)) else 0.0
                    events.append(ev)
            data_node = g['data']
            if data_node.attrs.get('MATLAB_class') == b'char' or fdt_path.is_file():
                if not fdt_path.is_file():
                    raise ValueError(f'Missing paired FDT file for {set_path}')
                expected_bytes = nbchan * pnts * trials * 4
                if fdt_path.stat().st_size != expected_bytes:
                    raise ValueError(f'FDT byte count mismatch for {fdt_path}')
                mm = np.memmap(fdt_path, dtype='<f4', mode='r')
                if trials == 1:
                    data = np.asarray(mm.reshape(pnts, nbchan).T, dtype='<f8')
                else:
                    data = np.asarray(mm.reshape(trials, pnts, nbchan).transpose(0, 2, 1), dtype='<f8')
                del mm
                storage = 'paired_fdt_hdf5_set'
            else:
                raw_arr = np.asarray(data_node, dtype='<f8')
                if trials == 1:
                    data = raw_arr.T if raw_arr.shape == (pnts, nbchan) else raw_arr
                else:
                    # HDF5 reverses MATLAB (nbchan, pnts, trials) -> (trials, pnts, nbchan)
                    if raw_arr.shape == (trials, pnts, nbchan):
                        data = raw_arr.transpose(0, 2, 1)
                    elif raw_arr.shape == (nbchan, pnts, trials):
                        data = raw_arr.transpose(2, 0, 1)
                    else:
                        raise ValueError(f'Unexpected HDF5 embedded data shape {raw_arr.shape}')
                storage = 'embedded_hdf5_set'
    else:
        mat = loadmat(set_path, simplify_cells=True)
        eeg = mat.get('EEG', mat)
        srate = float(eeg['srate'])
        nbchan = int(eeg['nbchan'])
        pnts = int(eeg['pnts'])
        trials = int(eeg['trials'])
        ref_val = str(eeg.get('ref', ''))
        chanlocs = eeg['chanlocs']
        if isinstance(chanlocs, dict):
            chanlocs = [chanlocs]
        ch_names = [str(c['labels']) for c in chanlocs]
        ch_types = [str(c.get('type', '')) for c in chanlocs]
        raw_events = eeg.get('event', [])
        if isinstance(raw_events, dict):
            raw_events = [raw_events]
        events = []
        for ev in list(raw_events):
            if isinstance(ev, dict) and 'latency' in ev:
                dur = ev.get('duration', 0.0)
                events.append(
                    dict(
                        latency=float(ev['latency']),
                        type=str(ev.get('type', '')),
                        duration=float(dur) if isinstance(dur, (int, float, np.number)) else 0.0,
                    )
                )
        raw_data = eeg['data']
        if isinstance(raw_data, np.ndarray):
            if trials == 1:
                if raw_data.shape != (nbchan, pnts):
                    raise ValueError(f'Unexpected 2D SET shape {raw_data.shape} in {set_path}')
                data = np.asarray(raw_data, dtype='<f8')
            else:
                if raw_data.shape != (nbchan, pnts, trials):
                    raise ValueError(f'Unexpected 3D SET shape {raw_data.shape} in {set_path}')
                data = np.asarray(raw_data.transpose(2, 0, 1), dtype='<f8')
            storage = 'embedded_mat5_set'
        else:
            if not fdt_path.is_file():
                raise ValueError(f'Missing paired FDT file for {set_path}')
            expected_bytes = nbchan * pnts * trials * 4
            if fdt_path.stat().st_size != expected_bytes:
                raise ValueError(f'FDT byte count mismatch for {fdt_path}')
            mm = np.memmap(fdt_path, dtype='<f4', mode='r')
            if trials == 1:
                data = np.asarray(mm.reshape(pnts, nbchan).T, dtype='<f8')
            else:
                data = np.asarray(mm.reshape(trials, pnts, nbchan).transpose(0, 2, 1), dtype='<f8')
            del mm
            storage = 'paired_fdt_mat5_set'
    return data, dict(
        srate=srate,
        nbchan=nbchan,
        pnts=pnts,
        trials=trials,
        ref=ref_val,
        ch_names=ch_names,
        ch_types=ch_types,
        events=events,
        storage=storage,
    )


def arithmetic_trials(events: list[dict], hz: float, points: int, n_samples: int) -> list[dict]:
    """Extract 6.0 s stimulus-locked non-tutorial arithmetic epochs from ds007262."""
    task_events = [r for r in events if r.get('marker_stream') == 'arithmetic-Markers']
    retained = []
    for idx, row in enumerate(task_events):
        diff = row.get('difficulty_range', 'n/a')
        if diff == 'n/a' or row.get('istutorial', '').strip().lower() != 'false':
            continue
        if diff not in ARITHMETIC_LABELS:
            raise ValueError(f'Unrecognised arithmetic difficulty range: {diff}')
        if idx == 0:
            raise ValueError('Outcome arithmetic marker lacks preceding stimulus marker')
        stim_row = task_events[idx - 1]
        if stim_row.get('marker', '').strip() != diff:
            raise ValueError(f'Stimulus/outcome marker mismatch: {stim_row} vs {row}')
        onset_stim = float(stim_row['onset'])
        onset_resp = float(row['onset'])
        if onset_resp < onset_stim:
            raise ValueError('Negative response latency in arithmetic events')
        if idx + 1 < len(task_events) and task_events[idx + 1].get('trial_type') != 'finished_arithmetic':
            next_onset = float(task_events[idx + 1]['onset'])
            if onset_stim + points / hz > next_onset + 1e-6:
                raise ValueError('Arithmetic 6.0 s epoch overlaps next trial stimulus')
        start = int(round(onset_stim * hz))
        end = start + points
        if start < 0 or end > n_samples:
            raise ValueError('Arithmetic epoch out of continuous recording bounds')
        code, desc = ARITHMETIC_LABELS[diff]
        retained.append(
            dict(
                start_sample=start,
                end_sample=end,
                onset_stim=onset_stim,
                onset_resp=onset_resp,
                response_time=onset_resp - onset_stim,
                difficulty_range=diff,
                outcome=row.get('outcome', ''),
                response_accuracy=row.get('response_accuracy', ''),
                label=code,
                label_name=desc,
            )
        )
    if not retained:
        raise ValueError('No valid non-tutorial arithmetic epochs extracted')
    return retained


def nback_trials(events: list[dict], hz: float, points: int, n_samples: int) -> list[dict]:
    """Extract 1.5 s non-tutorial post-stimulus n-back epochs from ds007169."""
    trials = [
        r
        for r in events
        if r.get('marker_stream') == 'n-backMarkers'
        and r.get('istutorial', '').strip().lower() == 'false'
        and r.get('trial_type') in NBACK_LABELS
    ]
    if not trials:
        raise ValueError('No non-tutorial n-back trials found')
    retained = []
    prev_end = -1
    for row in trials:
        onset = float(row['onset'])
        start = int(round(onset * hz))
        end = start + points
        if start < prev_end:
            raise ValueError(f'Overlapping n-back epochs: {start} < {prev_end}')
        if start < 0 or end > n_samples:
            raise ValueError('N-back epoch exceeds recording bounds')
        prev_end = end
        ttype = row['trial_type']
        code, desc = NBACK_LABELS[ttype]
        retained.append(
            dict(
                start_sample=start,
                end_sample=end,
                onset=onset,
                trial_type=ttype,
                nback_level=row.get('nback_level', ''),
                matched=row.get('matched', ''),
                key_press=row.get('key_press', ''),
                outcome=row.get('outcome', ''),
                response_accuracy=row.get('response_accuracy', ''),
                label=code,
                label_name=desc,
            )
        )
    return retained


def sternberg_trials(events: list[dict], target_hz: float, points: int, n_samples: int) -> list[dict]:
    """Extract 2.0 s maintenance-period epochs from ds004117 WorkingMemory runs."""
    bad_trial_ids = {r.get('trial') for r in events if r.get('task_role') == 'bad_trial'}
    retained = []
    prev_end = -1
    for idx, row in enumerate(events):
        if row.get('task_role') != 'work_memory':
            continue
        if row.get('trial') in bad_trial_ids:
            continue
        cond = str(row.get('memory_cond', '')).strip()
        if cond not in STERNBERG_LABELS:
            continue
        onset = float(row['onset'])
        # Ensure maintenance delay before probe is at least window length
        for nxt in events[idx + 1 :]:
            if nxt.get('task_role', '').startswith('probe_'):
                if float(nxt['onset']) - onset < points / target_hz - 1e-6:
                    raise ValueError('Maintenance delay shorter than 2.0 s before probe')
                break
        start = int(round(onset * target_hz))
        end = start + points
        if start < prev_end or start < 0 or end > n_samples:
            raise ValueError('Sternberg epoch out of bounds or overlapping')
        prev_end = end
        code, desc = STERNBERG_LABELS[cond]
        retained.append(
            dict(
                start_sample=start,
                end_sample=end,
                onset=onset,
                trial_id=str(row.get('trial', '')),
                memory_cond=cond,
                label=code,
                label_name=desc,
            )
        )
    return retained


def dens_emotion_windows(
    events: list[dict],
    beh_rows: list[dict],
    eeg_data: np.ndarray,
    points: int = 1000,
) -> tuple[list[dict], list[dict]]:
    """Extract 4.0 s windows within emotional video intervals in ds003751 (1-based sample latencies)."""
    n_samples = int(eeg_data.shape[1])
    bad = np.zeros(n_samples, dtype=bool)
    zero_mask = np.all(eeg_data == 0, axis=0)
    zero_idx = np.flatnonzero(zero_mask)
    if len(zero_idx):
        splits = np.where(np.diff(zero_idx) > 1)[0] + 1
        for run in np.split(zero_idx, splits):
            if len(run) >= 2:
                bad[int(run[0]) : int(run[-1]) + 1] = True

    # Mark impedance check intervals [IBEG, IEND] and interior boundaries
    ibeg_start = None
    boundary_samples = set()
    for ev in events:
        ttype = ev.get('trial_type', '').strip()
        lat_0 = int(round(float(ev['onset']) - 1.0))
        if ttype == 'IBEG':
            ibeg_start = max(0, min(n_samples, lat_0))
        elif ttype == 'IEND' and ibeg_start is not None:
            iend_stop = max(ibeg_start, min(n_samples, lat_0 + 1))
            bad[ibeg_start:iend_stop] = True
            ibeg_start = None
        elif ttype.lower() == 'boundary' and 0 < lat_0 < n_samples:
            boundary_samples.add(lat_0)

    beh_map = {}
    for br in beh_rows:
        stem = br['stimuliName'].rsplit('.', 1)[0].strip()
        beh_map[stem] = br

    windows = []
    skipped = []
    for idx, ev in enumerate(events):
        if ev.get('trial_type', '').strip() != 'stm':
            continue
        stim_label = ev.get('label', '').strip()
        stem = stim_label.rsplit('_', 1)[0]
        if stem.lower().startswith('neutral'):
            continue
        if stem not in beh_map:
            skipped.append(dict(stimulus_label=stim_label, reason='Missing self-assessment row in beh.tsv'))
            continue
        start_samp = int(round(float(ev['onset']) - 1.0))
        end_samp = None
        for nxt in events[idx + 1 :]:
            if nxt.get('trial_type', '').strip() in {'vlnc', 'arsl', 'trno'}:
                end_samp = int(round(float(nxt['onset']) - 1.0))
                break
        if end_samp is None:
            skipped.append(dict(stimulus_label=stim_label, reason='Missing post-video rating onset marker'))
            continue
        start_samp = max(0, start_samp)
        end_samp = min(n_samples, end_samp)
        if end_samp - start_samp < points:
            skipped.append(dict(stimulus_label=stim_label, reason='Stimulus interval shorter than 4.0 s'))
            continue
        br = beh_map[stem]
        valence = float(br['valence'])
        arousal = float(br['arousal'])
        quad = ('HV' if valence >= 5.0 else 'LV') + ('HA' if arousal >= 5.0 else 'LA')
        code, desc = DENS_QUADRANTS[quad]
        # Cut around any interior boundary or bad mask
        cuts = sorted({start_samp, end_samp} | {b for b in boundary_samples if start_samp < b < end_samp})
        win_within = 0
        for seg_a, seg_b in zip(cuts[:-1], cuts[1:]):
            cursor = seg_a
            while cursor + points <= seg_b:
                if bad[cursor : cursor + points].any():
                    cursor += points
                    continue
                win_arr = eeg_data[:, cursor : cursor + points]
                if (win_arr.std(axis=1) <= 1e-7).all():
                    skipped.append(
                        dict(
                            stimulus_label=stim_label,
                            start_sample=cursor,
                            reason='All channels flat/railed across 4.0 s window (std <= 1e-7 uV)',
                        )
                    )
                    cursor += points
                    continue
                windows.append(
                    dict(
                        start_sample=cursor,
                        end_sample=cursor + points,
                        stimulus_label=stim_label,
                        stimulus_stem=stem,
                        window_within_stimulus=win_within,
                        valence=valence,
                        arousal=arousal,
                        quadrant=quad,
                        label=code,
                        label_name=desc,
                    )
                )
                win_within += 1
                cursor += points
    return windows, skipped


def parse_clateeg_txt(raw_bytes: bytes) -> np.ndarray:
    """Parse OpenBCI Cyton 8-channel EEG columns (cols 1..8) in float64 microvolts."""
    rows = []
    text = raw_bytes.decode('utf-8', errors='replace')
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('%') or s.startswith('Sample'):
            continue
        parts = s.split(',')
        if len(parts) < 9:
            continue
        rows.append([float(parts[c]) for c in range(1, 9)])
    if not rows:
        raise ValueError('Empty OpenBCI text recording in CLATEEG26')
    arr = np.asarray(rows, dtype='<f8').T
    if arr.shape[0] != 8 or not np.isfinite(arr).all():
        raise ValueError('Invalid CLATEEG26 decoded shape or non-finite values')
    return arr


def fixed_resting_windows(
    eeg_data: np.ndarray,
    points: int,
    max_windows: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Extract up to max_windows consecutive non-overlapping windows from nonzero segments."""
    segments, zero_runs = nonzero_segments(eeg_data)
    windows = []
    for seg_idx, (seg_a, seg_b) in enumerate(segments):
        for start in range(seg_a, seg_b - points + 1, points):
            if (eeg_data[:, start : start + points].std(axis=1) <= 1e-7).all():
                continue
            windows.append((seg_idx, start, start + points))
            if len(windows) >= max_windows:
                return windows, zero_runs
    return windows, zero_runs


def lock_and_download_openneuro(key: str, cache_root: Path) -> dict:
    info = OPENNEURO_RELEASES[key]
    ds = info['dataset']
    ref = info['ref']
    commit = info['commit']
    cache = cache_root / 'CognitiveGroupA' / ds
    cache.mkdir(parents=True, exist_ok=True)

    tree_path = cache / 'original-release-tree.json'
    fetch(f'https://api.github.com/repos/OpenNeuroDatasets/{ds}/git/trees/{commit}?recursive=1', tree_path)
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    if tree.get('truncated'):
        raise ValueError(f'Truncated GitHub tree for {ds}')

    archive = cache / 'original-metadata.zip'
    fetch(f'https://codeload.github.com/OpenNeuroDatasets/{ds}/zip/{commit}', archive)

    inline_files = []
    annex_candidates = []
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0]
        for entry in tree['tree']:
            if entry['type'] != 'blob':
                continue
            name = entry['path']
            data = z.read(prefix + name)
            if blob_sha(data) != entry['sha']:
                raise ValueError(f'Git blob SHA1 mismatch for {ds}/{name}')
            if entry['mode'] == '120000':
                pointer = data.decode('utf-8', errors='replace')
                m = re.search(r'(SHA256|MD5)E-s(\d+)--([0-9a-f]+)(?:\.[^/]+)?$', pointer.strip())
                if not m or len(m[3]) != {'SHA256': 64, 'MD5': 32}[m[1]]:
                    continue
                annex_candidates.append(
                    dict(
                        name=name,
                        url=f'https://s3.amazonaws.com/openneuro.org/{ds}/{name}',
                        bytes=int(m[2]),
                        checksum=m[1].lower() + ':' + m[3],
                        git_blob_sha1=entry['sha'],
                        git_annex_pointer=pointer,
                    )
                )
            else:
                if name.startswith(('.', 'sourcedata/', 'code/', 'stimuli/', 'derivatives/')):
                    continue
                if '/pupil/' in name or '/ecg/' in name or 'physio.' in name:
                    continue
                if key == 'sleep_deprivation' and '/eeg/' in name and '_task-eyesopen_' not in name:
                    continue
                target = (cache / name).resolve()
                if not target.is_relative_to(cache.resolve()):
                    raise ValueError(f'Unsafe path {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                inline_files.append(
                    dict(
                        name=name,
                        url=f'https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{commit}/{name}',
                        bytes=len(data),
                        checksum='sha256:' + hashlib.sha256(data).hexdigest(),
                        sha256=hashlib.sha256(data).hexdigest(),
                        git_blob_sha1=entry['sha'],
                    )
                )

    # Filter annex candidates per dataset rules
    if key in {'arithmetic_difficulty', 'nback_difficulty'}:
        pts = core.table(cache / 'participants.tsv')
        included_subs = {r['participant_id'] for r in pts if r.get('analysis_included', '').lower() == 'true'}
        selected = [
            f
            for f in annex_candidates
            if f['name'].split('/')[0] in included_subs
            and '/eeg/' in f['name']
            and f['name'].endswith(('_eeg.vhdr', '_eeg.vmrk', '_eeg.eeg'))
        ]
    elif key == 'sternberg_memory':
        selected = [
            f
            for f in annex_candidates
            if f['name'].startswith('sub-') and '/eeg/' in f['name'] and f['name'].endswith(('_eeg.set', '_eeg.fdt'))
        ]
    elif key == 'naturalistic_emotion':
        beh_subs = {
            f['name'].split('/')[0]
            for f in inline_files
            if f['name'].startswith('sub-') and f['name'].endswith('_beh.tsv')
        }
        selected = [
            f
            for f in annex_candidates
            if f['name'] == 'LICENSE'
            or (
                f['name'].split('/')[0] in beh_subs
                and '/eeg/' in f['name']
                and f['name'].endswith(('_eeg.set', '_eeg.fdt'))
            )
        ]
    elif key == 'sleep_deprivation':
        selected = [
            f
            for f in annex_candidates
            if f['name'].startswith('sub-')
            and '_task-eyesopen_eeg.' in f['name']
            and f['name'].endswith(('.set', '.fdt'))
        ]
    else:
        raise ValueError(f'Unknown OpenNeuro key: {key}')

    def _dl(item: dict) -> dict:
        dst = cache / item['name']
        fetch(item['url'], dst, item['checksum'], item['bytes'])
        print(f'Verified {ds}/{item["name"]}', flush=True)
        return dict(item, sha256=sha(dst))

    with ThreadPoolExecutor(4) as pool:
        downloaded = list(pool.map(_dl, selected))

    licence_evidence = 'Original pinned dataset_description.json License=CC0'
    if key == 'naturalistic_emotion':
        licence_evidence = (
            'dataset_description.json declares License=CC0; root LICENSE file contains ODbL-1.0 '
            '(with DBCL-1.0 for individual contents); both retained in provenance'
        )
    receipt = dict(
        dataset=ds,
        source_record_id=ds,
        source_revision=f'OpenNeuro {ds} v{ref}, Git {commit}',
        original_ref=ref,
        original_commit=commit,
        licence=info['licence'],
        licence_evidence=licence_evidence,
        source_urls=[f'https://openneuro.org/datasets/{ds}/versions/{ref}'],
        metadata_archive_sha256=sha(archive),
        files=inline_files + downloaded,
    )
    dump(cache / 'download-receipt.json', receipt)
    dump(cache / 'source-record.json', receipt)
    dump(cache / 'original-source-lock.json', receipt)
    dump(ROOT / 'docs' / f'{ds}-cognitive-source-lock.json', receipt)
    return receipt


def lock_and_download_neuro2021(cache_root: Path) -> dict:
    cache = cache_root / 'Neuro2021'
    cache.mkdir(parents=True, exist_ok=True)
    record = json.loads((EVIDENCE_DIR / 'Neuro2021.json').read_text(encoding='utf-8'))
    zip_entries = sorted(
        [
            f
            for f in record['files']
            if re.fullmatch(r'P\d{2}\.zip', f.get('name') or f.get('key', ''))
        ],
        key=lambda x: x.get('name') or x['key'],
    )
    if len(zip_entries) != 15:
        raise ValueError(f'Expected 15 subject zips in Neuro2021.json, got {len(zip_entries)}')

    def _dl(entry: dict) -> dict:
        name = entry.get('name') or entry['key']
        url = entry.get('links', {}).get('self') or f'https://zenodo.org/api/records/5055046/files/{name}/content'
        size = int(entry.get('bytes') or entry['size'])
        checksum = entry['checksum']
        dst = cache / name
        fetch(url, dst, checksum, size)
        print(f'Verified Neuro2021/{name}', flush=True)
        return dict(name=name, url=url, bytes=size, checksum=checksum, sha256=sha(dst))

    with ThreadPoolExecutor(3) as pool:
        downloaded = list(pool.map(_dl, zip_entries))

    receipt = dict(
        dataset='Neuro2021',
        source_record_id='5055046',
        source_revision='Zenodo record 5055046 (v2, 2021-07-01)',
        licence='CC-BY-SA-4.0',
        licence_evidence='Zenodo record 5055046 metadata license.id=cc-by-sa-4.0',
        source_urls=['https://zenodo.org/records/5055046', 'https://doi.org/10.5281/zenodo.5055046'],
        files=downloaded,
    )
    dump(cache / 'download-receipt.json', receipt)
    dump(cache / 'source-record.json', receipt)
    dump(ROOT / 'docs/neuro2021-cognitive-source-lock.json', receipt)
    return receipt


def lock_and_download_clateeg26(cache_root: Path) -> dict:
    cache = cache_root / 'CLATEEG26'
    cache.mkdir(parents=True, exist_ok=True)
    entries = json.loads((EVIDENCE_DIR / 'CLATEEG26-files.json').read_text(encoding='utf-8'))
    entry = entries[0]
    details = entry['content_details']
    name = entry['filename']
    url = details['download_url']
    size = int(details['size'])
    checksum = 'sha256:' + details['sha256_hash']
    dst = cache / name
    fetch(url, dst, checksum, size)
    print(f'Verified CLATEEG26/{name}', flush=True)
    file_rec = dict(name=name, url=url, bytes=size, checksum=checksum, sha256=sha(dst))
    receipt = dict(
        dataset='CLATEEG26',
        source_record_id='pxysmjn3j2.3',
        source_revision='Mendeley Data 10.17632/pxysmjn3j2.3 (v3, 2026-10-22)',
        licence='CC-BY-4.0',
        licence_evidence='Mendeley Data public-api data_licence short_name=CC BY 4.0',
        source_urls=['https://data.mendeley.com/datasets/pxysmjn3j2/3', 'https://doi.org/10.17632/pxysmjn3j2.3'],
        files=[file_rec],
    )
    dump(cache / 'download-receipt.json', receipt)
    dump(cache / 'source-record.json', receipt)
    dump(ROOT / 'docs/clateeg26-cognitive-source-lock.json', receipt)
    return receipt


# --- Per-dataset READERS ---


DS007169_V105_EEG_SCALE = 1e-6
DS007169_V105_SCALE_REASON = (
    'OpenNeuro ds007169 v1.0.5 upstream XDF->BIDS converter (LMBooth/QT-nback_study '
    'convert_xdf_to_bids_eeg.py) passed microvolt LSL EEGStream samples directly into '
    'mne.io.RawArray (which expects SI Volts) with ch_types="eeg" before '
    'mne_bids.write_raw_bids multiplied EEG channels by 1e6 (leaving D1..D5 "misc" channels '
    'unscaled). Multiplying decoded BrainVision EEG values by 1e-6 in float64 restores true '
    'microvolts, matching the upstream v1.1.0 fix applied by the same authors to sister dataset '
    'ds007262 (ArithmeticDifficulty).'
)


def _read_uoh_bv(
    cache: Path,
    spec: dict,
    task_name: str,
    trial_fn,
    eeg_scale_factor: float = 1.0,
    scale_correction_reason: str | None = None,
) -> tuple:
    pts = core.table(cache / 'participants.tsv')
    included = {}
    excluded = []
    for row in pts:
        pid = row['participant_id']
        person = pid.removeprefix('sub-')
        if row.get('analysis_included', '').lower() == 'true':
            included[person] = row
        else:
            excluded.append(
                dict(
                    participant_id=person,
                    original_participant_id=pid,
                    data_quality_overall=row.get('data_quality_overall', ''),
                    reason='Marked analysis_included=false by dataset authors due to acquisition/QC failure',
                )
            )
    spec['metadata_updates']['excluded_participants'] = excluded
    canonical = None
    for person in sorted(included):
        participant = included[person]
        rel = f'sub-{person}/eeg/sub-{person}_task-{task_name}_eeg.vhdr'
        vhdr = cache / rel
        stem = str(vhdr)[:-8]
        channels = core.table(stem + 'channels.tsv')
        events = core.table(stem + 'events.tsv')
        bids = json.loads(Path(stem + 'eeg.json').read_text(encoding='utf-8-sig'))
        data_uV, info = decode_brainvision(vhdr)
        if len(channels) != info['native_channels'] or not np.isclose(
            float(bids['SamplingFrequency']), info['native_sampling_frequency_hz'], rtol=1e-9
        ):
            raise ValueError(f'Header/BIDS mismatch in {rel}')
        if not np.isclose(info['native_sampling_frequency_hz'], 250.0, rtol=1e-9):
            raise ValueError(f'Expected 250 Hz in {rel}')
        picks = [i for i, c in enumerate(channels) if c['type'].upper() == 'EEG']
        names = [channels[i]['name'] for i in picks]
        if len(picks) != 19 or info['native_channel_names'] != [c['name'] for c in channels]:
            raise ValueError(f'Unexpected channel layout in {rel}')
        if canonical is None:
            canonical = names
        elif names != canonical:
            raise ValueError(f'Channel ordering differs in {rel}')
        raw_eeg = np.ascontiguousarray(data_uV[picks], dtype='<f8')
        raw_median_sd = float(np.median(raw_eeg.std(axis=1)))
        if eeg_scale_factor != 1.0:
            if raw_median_sd <= 1e5:
                raise ValueError(
                    f'Expected unpatched 1e6*uV upstream BrainVision scale in {rel}, '
                    f'got median channel SD {raw_median_sd:.6g}; review before applying {eeg_scale_factor}'
                )
            eeg_uV = np.ascontiguousarray(raw_eeg * eeg_scale_factor, dtype='<f8')
        else:
            eeg_uV = raw_eeg
        epochs = trial_fn(events, 250.0, spec['points'], eeg_uV.shape[1])
        rows = []
        values = []
        for idx, ep in enumerate(epochs):
            start = ep['start_sample']
            row = core.base_row(
                person,
                '0',
                task_name,
                rel,
                idx,
                start,
                spec['points'],
                250,
                ep['label_name'],
                ep['label'],
            )
            source_expr = (
                f'BrainVision MULTIPLEXED float32 channels {picks}, samples [{start}:{ep["end_sample"]}] in microvolts'
                if eeg_scale_factor == 1.0
                else (
                    f'BrainVision MULTIPLEXED float32 channels {picks}, samples [{start}:{ep["end_sample"]}] '
                    f'multiplied by {eeg_scale_factor:g} in float64 to microvolts (correcting upstream ds007169 v1.0.5 1e6 over-scaling)'
                )
            )
            row.update(
                original_participant_id=participant['participant_id'],
                age=participant.get('age', ''),
                sex=participant.get('sex', ''),
                handedness=participant.get('handedness', ''),
                outcome=ep.get('outcome', ''),
                response_accuracy=ep.get('response_accuracy', ''),
                source_array_index=source_expr,
            )
            if 'response_time' in ep:
                row.update(
                    source_event_onset=ep['onset_stim'],
                    response_onset_seconds=ep['onset_resp'],
                    response_time_seconds=round(ep['response_time'], 6),
                    difficulty_range=ep['difficulty_range'],
                )
            else:
                row.update(
                    source_event_onset=ep['onset'],
                    nback_level=ep['nback_level'],
                    matched=ep['matched'],
                    key_press=ep['key_press'],
                )
            rows.append(row)
            values.append(np.ascontiguousarray(eeg_uV[:, start : ep['end_sample']], dtype='<f8'))
        dropout_count = sum(1 for e in events if e.get('trial_type') == 'dropped_samples')
        info.update(
            source_file=rel,
            source_file_sha256=sha(vhdr),
            participant_id=person,
            participant_metadata=participant,
            source_eeg_indices=picks,
            excluded_non_eeg_channels=[channels[i]['name'] for i in range(len(channels)) if i not in picks],
            lsl_offset_annotation_count=dropout_count,
            all_zero_sample_count=int(np.all(eeg_uV == 0, axis=0).sum()),
            source_case_count=len(rows),
        )
        if eeg_scale_factor != 1.0:
            info.update(
                upstream_eeg_scale_correction=eeg_scale_factor,
                upstream_raw_median_channel_sd=raw_median_sd,
                corrected_median_channel_sd_uV=float(np.median(eeg_uV.std(axis=1))),
                upstream_eeg_scale_correction_reason=scale_correction_reason,
            )
        yield np.stack(values), rows, canonical, info


def arithmetic_records(cache: Path, spec: dict):
    yield from _read_uoh_bv(cache, spec, 'arithmetic', arithmetic_trials)


def nback_records(cache: Path, spec: dict):
    yield from _read_uoh_bv(
        cache,
        spec,
        'nback',
        nback_trials,
        eeg_scale_factor=DS007169_V105_EEG_SCALE,
        scale_correction_reason=DS007169_V105_SCALE_REASON,
    )


def multitask_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    canonical = None
    for f in sorted(receipt['files'], key=lambda item: item['name']):
        person = re.fullmatch(r'P(\d{2})\.zip', f['name'])[1]
        zpath = cache / f['name']
        with zipfile.ZipFile(zpath) as z:
            members = set(z.namelist())
            for sess in ('1', '2'):
                for cond_tag, (code, label_name) in MATB_LABELS.items():
                    rel_in_zip = f'P{person}/S{sess}/eeg/alldata_sbj{person}_sess{sess}_{cond_tag}.set'
                    if rel_in_zip not in members:
                        raise ValueError(f'Missing expected MATB SET {rel_in_zip} in {f["name"]}')
                    raw_bytes = z.read(rel_in_zip)
                    mat = loadmat(io.BytesIO(raw_bytes), simplify_cells=True)
                    eeg = mat.get('EEG', mat)
                    srate = float(eeg['srate'])
                    nbchan = int(eeg['nbchan'])
                    pnts = int(eeg['pnts'])
                    trials = int(eeg['trials'])
                    chanlocs = eeg['chanlocs']
                    names = [str(c['labels']) for c in chanlocs]
                    arr = eeg['data']
                    if srate != 250.0 or nbchan != 61 or pnts != 500 or arr.shape != (61, 500, trials):
                        raise ValueError(f'Unexpected Neuro2021 SET schema in {rel_in_zip}: {arr.shape} @ {srate}')
                    if canonical is None:
                        canonical = names
                    elif names != canonical:
                        raise ValueError(f'Channel ordering differs in {rel_in_zip}')
                    x = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype='<f8')
                    rows = []
                    src_id = f'{f["name"]}:{rel_in_zip}'
                    for ep_idx in range(trials):
                        row = core.base_row(
                            person,
                            f'S{sess}',
                            cond_tag,
                            src_id,
                            ep_idx,
                            ep_idx * 500,
                            500,
                            250,
                            label_name,
                            code,
                        )
                        row.update(
                            source_epoch_index=ep_idx,
                            original_participant_id=f'P{person}',
                            original_session_id=f'S{sess}',
                            matb_condition=cond_tag,
                            source_array_index=f'Embedded float32 SET data[:, :, {ep_idx}] transposed to (61, 500) float64 microvolts',
                        )
                        rows.append(row)
                    info = dict(
                        source_file=src_id,
                        source_zip_sha256=f['sha256'],
                        set_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                        participant_id=person,
                        session_id=f'S{sess}',
                        condition=cond_tag,
                        source_eeg_indices=list(range(61)),
                        native_channel_names=names,
                        native_sampling_frequency_hz=250.0,
                        native_trials=trials,
                        source_case_count=len(rows),
                    )
                    yield x, rows, canonical, info


def clateeg_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    zfile = cache / receipt['files'][0]['name']
    max_windows = 15
    points = spec['points']
    with zipfile.ZipFile(zfile) as z:
        txt_entries = [
            n
            for n in z.namelist()
            if n.endswith('.txt') and '/._' not in n and not n.startswith('__MACOSX')
        ]
        if len(txt_entries) != 320:
            raise ValueError(f'Expected 320 CLATEEG26 .txt files, got {len(txt_entries)}')
        parsed_index = []
        for entry in txt_entries:
            m = re.search(
                r'(Arithmetic_Data|Stroop_Data)/(neutral|lowlevel|midlevel|highlevel)-subject-(\d{2})\.txt$',
                entry,
            )
            if not m:
                raise ValueError(f'Unrecognised CLATEEG26 entry path: {entry}')
            task_dir, cond_key, person = m[1], m[2], m[3]
            parsed_index.append((person, task_dir, cond_key, entry))
        parsed_index.sort()
        for person, task_dir, cond_key, entry in parsed_index:
            raw_bytes = z.read(entry)
            eeg_uV = parse_clateeg_txt(raw_bytes)
            windows, zero_runs = fixed_resting_windows(eeg_uV, points=points, max_windows=max_windows)
            if not windows:
                raise ValueError(f'No complete 4.0 s windows in {entry}')
            code, label_name = CLATEEG_LABELS[cond_key]
            task_short = 'arithmetic' if task_dir.startswith('Arithmetic') else 'stroop'
            src_id = f'CLATEEG26.zip:{entry}'
            rows = []
            values = []
            for idx, (seg_idx, start, end) in enumerate(windows):
                row = core.base_row(
                    person,
                    task_short,
                    cond_key,
                    src_id,
                    idx,
                    start,
                    points,
                    250,
                    label_name,
                    code,
                )
                row.update(
                    original_participant_id=f'subject-{person}',
                    cognitive_task=task_short,
                    source_condition=cond_key,
                    continuous_segment_index=seg_idx,
                    source_array_index=f'OpenBCI Cyton columns 1..8 (8 channels), samples [{start}:{end}] in float64 microvolts',
                )
                rows.append(row)
                values.append(np.ascontiguousarray(eeg_uV[:, start:end], dtype='<f8'))
            info = dict(
                source_file=src_id,
                txt_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                participant_id=person,
                session_id=task_short,
                condition=cond_key,
                source_eeg_indices=list(range(8)),
                native_channel_names=CLATEEG_CHANNELS,
                native_samples=int(eeg_uV.shape[1]),
                native_sampling_frequency_hz=250.0,
                excluded_all_zero_runs=zero_runs,
                retained_windows_cap=max_windows,
                source_case_count=len(rows),
            )
            yield np.stack(values), rows, CLATEEG_CHANNELS, info


def sternberg_records(cache: Path, spec: dict):
    pts = {r['participant_id'].removeprefix('sub-'): r for r in core.table(cache / 'participants.tsv')}
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    set_files = sorted(f['name'] for f in receipt['files'] if f['name'].endswith('_eeg.set'))
    canonical = None
    allowed_rates = (250.0, 500.0, 500.059, 1000.0)
    for rel in set_files:
        m = re.match(r'sub-(\d+)/ses-(\d+)/eeg/sub-\d+_ses-\d+_task-WorkingMemory_run-(\d+)_eeg\.set$', rel)
        if not m:
            raise ValueError(f'Unrecognised ds004117 SET path: {rel}')
        person, ses, run = m[1], m[2], m[3]
        participant = pts[person]
        path = cache / rel
        stem = str(path)[:-7]
        channels = core.table(stem + 'channels.tsv')
        events = core.table(stem + 'events.tsv')
        bids = json.loads(Path(stem + 'eeg.json').read_text(encoding='utf-8-sig'))
        data, set_info = decode_eeglab_set(path)
        hz = set_info['srate']
        picks = [i for i, c in enumerate(channels) if c['type'].upper() == 'EEG']
        names = [channels[i]['name'] for i in picks]
        if len(picks) != 69 or len(channels) != 71:
            raise ValueError(f'Unexpected ds004117 channel layout in {rel}')
        if canonical is None:
            canonical = names
        elif names != canonical:
            raise ValueError(f'Channel ordering differs in {rel}')
        eeg_uV = np.ascontiguousarray(data[picks], dtype='<f8')
        resampled = harmonise_sampling_rate(eeg_uV, hz, 250.0, allowed_rates)
        epochs = sternberg_trials(events, 250.0, spec['points'], resampled.shape[1])
        if not epochs:
            continue
        rows = []
        values = []
        excluded_railed = []
        for ep in epochs:
            start = ep['start_sample']
            seg = np.ascontiguousarray(resampled[:, start : ep['end_sample']], dtype='<f8')
            flat_mask = seg.std(axis=1) <= 1e-7
            if flat_mask.any():
                excluded_railed.append(
                    dict(
                        trial_id=ep['trial_id'],
                        start_sample=start,
                        end_sample=ep['end_sample'],
                        n_flat_channels=int(flat_mask.sum()),
                        flat_channels=[names[k] for k in np.flatnonzero(flat_mask).tolist()],
                        reason='Unmarked amplifier ADC rail saturation / flat signal (std <= 1e-7 uV)',
                    )
                )
                continue
            idx = len(rows)
            native_start = int(round(ep['onset'] * hz))
            native_len = int(round(2.0 * hz))
            row = core.base_row(
                person,
                f'ses-{ses}',
                f'run-{run}',
                rel,
                idx,
                native_start,
                native_len,
                hz,
                ep['label_name'],
                ep['label'],
            )
            row.update(
                original_participant_id=participant['participant_id'],
                age=participant.get('age', ''),
                sex=participant.get('sex', ''),
                original_trial_id=ep['trial_id'],
                memory_load=ep['memory_cond'],
                source_event_onset=ep['onset'],
                original_sampling_frequency_hz=hz,
                processed_sampling_frequency_hz=250.0,
                processed_start_sample=start,
                processed_end_sample=ep['end_sample'],
                resampling_applied=not np.isclose(hz, 250.0, rtol=1e-9),
                source_array_index=f'SET/FDT 69 EEG channels resampled {hz}->250 Hz, processed samples [{start}:{ep["end_sample"]}]',
            )
            rows.append(row)
            values.append(seg)
        if not rows:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(path),
            participant_id=person,
            session_id=f'ses-{ses}',
            run_id=f'run-{run}',
            source_eeg_indices=picks,
            excluded_eog_channels=[channels[i]['name'] for i in range(len(channels)) if i not in picks],
            excluded_railed_trials=excluded_railed,
            native_sampling_frequency_hz=hz,
            processed_sampling_frequency_hz=250.0,
            resampling_applied=not np.isclose(hz, 250.0, rtol=1e-9),
            native_samples=set_info['pnts'],
            processed_samples=int(resampled.shape[1]),
            source_case_count=len(rows),
        )
        yield np.stack(values), rows, canonical, info


def naturalistic_emotion_records(cache: Path, spec: dict):
    # Deduplicate participants.tsv (sub-mit081 appears twice upstream)
    pt_map = {}
    duplicate_pt_rows = []
    for row in core.table(cache / 'participants.tsv'):
        pid = row['participant_id'].strip()
        if pid in pt_map:
            duplicate_pt_rows.append(pid)
        else:
            pt_map[pid] = row

    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    all_ev_subs = sorted({
        f['name'].split('/')[0]
        for f in receipt['files']
        if f['name'].startswith('sub-') and f['name'].endswith('_events.tsv')
    })
    beh_files = {
        f['name'].split('/')[0]: f['name']
        for f in receipt['files']
        if f['name'].startswith('sub-') and f['name'].endswith('_beh.tsv')
    }
    set_files = {
        f['name'].split('/')[0]: f['name']
        for f in receipt['files']
        if f['name'].startswith('sub-') and f['name'].endswith('_eeg.set')
    }
    excluded = [
        dict(
            participant_id=sub.removeprefix('sub-'),
            original_participant_id=sub,
            reason='No beh/*_beh.tsv self-assessment rating table in OpenNeuro ds003751 v1.0.6 release',
        )
        for sub in all_ev_subs
        if sub not in beh_files
    ]
    spec['metadata_updates']['excluded_participants'] = excluded
    spec['metadata_updates']['deduplicated_participants_tsv_ids'] = duplicate_pt_rows

    canonical = [f'E{i}' for i in range(1, 129)]
    for sub in sorted(set_files):
        person = sub.removeprefix('sub-')
        participant = pt_map.get(sub, dict(participant_id=sub))
        rel_set = set_files[sub]
        rel_beh = beh_files[sub]
        set_path = cache / rel_set
        fdt_path = set_path.with_suffix('.fdt')
        hdr_mat = loadmat(set_path, simplify_cells=True)
        hdr_eeg = hdr_mat.get('EEG', hdr_mat)
        expected_fdt_bytes = int(hdr_eeg['nbchan']) * int(hdr_eeg['pnts']) * int(hdr_eeg['trials']) * 4
        actual_fdt_bytes = fdt_path.stat().st_size if fdt_path.is_file() else 0
        if actual_fdt_bytes != expected_fdt_bytes:
            excluded.append(
                dict(
                    participant_id=person,
                    original_participant_id=sub,
                    expected_fdt_bytes=expected_fdt_bytes,
                    actual_fdt_bytes=actual_fdt_bytes,
                    reason='Upstream OpenNeuro ds003751 v1.0.0-v1.0.6 .fdt binary is truncated (incomplete frame/byte count vs .set header)',
                )
            )
            continue
        ev_candidates = [
            f['name']
            for f in receipt['files']
            if f['name'].startswith(sub + '/eeg/') and f['name'].endswith('_events.tsv')
        ]
        ev_path = cache / ev_candidates[0]
        events = core.table(ev_path)
        beh_rows = core.table(cache / rel_beh)
        data, set_info = decode_eeglab_set(set_path)
        if set_info['srate'] != 250.0:
            raise ValueError(f'Unexpected srate {set_info["srate"]} in {rel_set}')
        ch_names = set_info['ch_names']
        picks = [ch_names.index(ch) for ch in canonical]
        excluded_chs = [c for i, c in enumerate(ch_names) if i not in picks]
        eeg_uV = np.ascontiguousarray(data[picks], dtype='<f8')
        windows, skipped = dens_emotion_windows(events, beh_rows, eeg_uV, points=spec['points'])
        if not windows:
            raise ValueError(f'No valid emotional stimulus windows for {sub}')
        rows = []
        values = []
        for idx, w in enumerate(windows):
            start = w['start_sample']
            row = core.base_row(
                person,
                '0',
                w['stimulus_label'],
                rel_set,
                idx,
                start,
                spec['points'],
                250,
                w['label_name'],
                w['label'],
            )
            row.update(
                original_participant_id=sub,
                age=participant.get('age', ''),
                sex=participant.get('sex', ''),
                handedness=participant.get('handedness', ''),
                stimulus_stem=w['stimulus_stem'],
                window_within_stimulus=w['window_within_stimulus'],
                valence_rating=w['valence'],
                arousal_rating=w['arousal'],
                circumplex_quadrant=w['quadrant'],
                source_array_index=f'SET/FDT active E1..E128 channels, 0-based samples [{start}:{w["end_sample"]}] in microvolts',
            )
            rows.append(row)
            values.append(np.ascontiguousarray(eeg_uV[:, start : w['end_sample']], dtype='<f8'))
        info = dict(
            source_file=rel_set,
            source_file_sha256=sha(cache / rel_set),
            beh_file=rel_beh,
            participant_id=person,
            source_eeg_indices=picks,
            excluded_channels=excluded_chs,
            native_sampling_frequency_hz=250.0,
            native_samples=set_info['pnts'],
            skipped_stimuli=skipped,
            source_case_count=len(rows),
        )
        yield np.stack(values), rows, canonical, info


def sleep_deprivation_records(cache: Path, spec: dict):
    pts = {r['participant_id'].removeprefix('sub-'): r for r in core.table(cache / 'participants.tsv')}
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    set_files = sorted(
        f['name']
        for f in receipt['files']
        if f['name'].endswith('_eeg.set') and '_task-eyesopen_' in f['name']
    )
    if len(set_files) != 142:
        raise ValueError(f'Expected 142 eyes-open SET files in ds004902, got {len(set_files)}')

    # Pre-scan paired sessions for upstream OpenNeuro .fdt byte truncation so only intact pairs are exported
    truncated_subs = {}
    for rel in set_files:
        m_pre = re.match(r'sub-(\d+)/(ses-[12])/eeg/sub-\d+_ses-[12]_task-eyesopen_eeg\.set$', rel)
        if not m_pre:
            raise ValueError(f'Unrecognised ds004902 SET path: {rel}')
        person_pre, ses_pre = m_pre[1], m_pre[2]
        sp = cache / rel
        fp = sp.with_suffix('.fdt')
        hdr_mat = loadmat(sp, simplify_cells=True)
        hdr_eeg = hdr_mat.get('EEG', hdr_mat)
        exp_bytes = int(hdr_eeg['nbchan']) * int(hdr_eeg['pnts']) * int(hdr_eeg['trials']) * 4
        act_bytes = fp.stat().st_size if fp.is_file() else 0
        if act_bytes != exp_bytes:
            truncated_subs[person_pre] = dict(
                participant_id=person_pre,
                original_participant_id=f'sub-{person_pre}',
                truncated_session=ses_pre,
                expected_fdt_bytes=exp_bytes,
                actual_fdt_bytes=act_bytes,
                reason='Upstream OpenNeuro ds004902 v1.0.8 .fdt binary is truncated (incomplete frame/byte count vs .set header); paired participant excluded to preserve balanced paired sessions',
            )
    spec['metadata_updates']['excluded_participants'] = [truncated_subs[k] for k in sorted(truncated_subs)]

    canonical = None
    allowed_rates = (500.0, 5000.0)
    points = spec['points']
    max_windows = 30
    for rel in set_files:
        m = re.match(r'sub-(\d+)/(ses-[12])/eeg/sub-\d+_ses-[12]_task-eyesopen_eeg\.set$', rel)
        if not m:
            raise ValueError(f'Unrecognised ds004902 SET path: {rel}')
        person, ses = m[1], m[2]
        if person in truncated_subs:
            continue
        participant = pts[person]
        code, label_name = SLEEP_LABELS[ses]
        path = cache / rel
        stem = str(path)[:-7]
        channels = core.table(stem + 'channels.tsv')
        bids = json.loads(Path(stem + 'eeg.json').read_text(encoding='utf-8-sig'))
        data, set_info = decode_eeglab_set(path)
        hz = set_info['srate']
        if not np.isclose(float(bids['SamplingFrequency']), hz, rtol=1e-9):
            raise ValueError(f'BIDS/SET frequency mismatch in {rel}')
        picks = [i for i, c in enumerate(channels) if c['type'].upper() == 'EEG']
        names = [channels[i]['name'] for i in picks]
        if len(picks) != 61:
            raise ValueError(f'Expected 61 EEG channels in {rel}, got {len(picks)}')
        if canonical is None:
            canonical = names
        elif names != canonical:
            raise ValueError(f'Channel ordering differs in {rel}')
        eeg_uV = np.ascontiguousarray(data[picks], dtype='<f8')
        processed = harmonise_sampling_rate(eeg_uV, hz, 500.0, allowed_rates)
        windows, zero_runs = fixed_resting_windows(processed, points=points, max_windows=max_windows)
        if len(windows) != max_windows:
            raise ValueError(f'Expected {max_windows} windows in {rel}, got {len(windows)}')
        rows = []
        values = []
        factor = int(round(hz / 500.0))
        for idx, (seg_idx, start, end) in enumerate(windows):
            row = core.base_row(
                person,
                ses,
                'eyesopen',
                rel,
                idx,
                start * factor,
                points * factor,
                hz,
                label_name,
                code,
            )
            row.update(
                original_participant_id=participant['participant_id'],
                age=participant.get('Age', ''),
                gender=participant.get('Gender', ''),
                session_order=participant.get('SessionOrder', ''),
                continuous_segment_index=seg_idx,
                original_sampling_frequency_hz=hz,
                processed_sampling_frequency_hz=500.0,
                processed_start_sample=start,
                processed_end_sample=end,
                resampling_applied=not np.isclose(hz, 500.0, rtol=1e-9),
                source_array_index=f'SET/FDT 61 EEG channels ({hz}->500 Hz), processed samples [{start}:{end}] in microvolts',
            )
            rows.append(row)
            values.append(np.ascontiguousarray(processed[:, start:end], dtype='<f8'))
        info = dict(
            source_file=rel,
            source_file_sha256=sha(path),
            participant_id=person,
            session_id=ses,
            sleep_condition=label_name,
            source_eeg_indices=picks,
            native_sampling_frequency_hz=hz,
            processed_sampling_frequency_hz=500.0,
            resampling_applied=not np.isclose(hz, 500.0, rtol=1e-9),
            excluded_all_zero_runs=zero_runs,
            native_samples=set_info['pnts'],
            processed_samples=int(processed.shape[1]),
            source_case_count=len(rows),
        )
        yield np.stack(values), rows, canonical, info


def _build_spec(key: str, cache_root: Path) -> tuple[dict, callable, Path, str]:
    if key == 'arithmetic_difficulty':
        cache = cache_root / 'CognitiveGroupA/ds007262'
        pts = [
            r['participant_id'].removeprefix('sub-')
            for r in core.table(cache / 'participants.tsv')
            if r.get('analysis_included', '').lower() == 'true'
        ]
        train, _ = whole_subject_split(pts)
        labels = {v[0]: v[1] for v in ARITHMETIC_LABELS.values()}
        spec = dict(
            name='ArithmeticDifficulty',
            source='Barras & Booth / University of Hull / OpenNeuro ds007262 v1.1.0',
            cache='CognitiveGroupA/ds007262',
            hz=250,
            points=1500,
            classes=7,
            train_subjects=len(train),
            licence='CC0-1.0',
            licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
            moabb_revision=None,
            doi='10.18112/openneuro.ds007262.v1.1.0',
            paper='Barras & Booth (2026), Cognitive Workload 8-level arithmetic, OpenNeuro ds007262 v1.1.0; Bateson & Asghar (2021), IEEE Access',
            labels=labels,
            splitter=lambda people, chunks: set(people) & train,
            evidence_files=['cognitive-group-a-plan.md', 'ds007262-cognitive-source-lock.json'],
            cache_evidence_files=[p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README.md', 'task-arithmetic_events.json', 'original-source-lock.json'] if (cache / p).is_file()],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='Graded 7-class mental arithmetic workload classification across Q-value difficulty ranges (0.6-1.5 through 6.0-6.9) on 18 QC-passed participants from OpenNeuro ds007262 v1.1.0 (sub-002 and sub-017 excluded per analysis_included=false). 19 10-20 scalp EEG channels at 250 Hz in float64 microvolts (D1 ECG and D2..D5 auxiliary channels excluded); 6.0-second (1,500-sample) non-tutorial stimulus-locked epochs starting at question onset. Shares the University of Hull laboratory cohort with NBackDifficulty (ds007169).',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://openneuro.org/datasets/ds007262/versions/1.1.0',
                paper_reference='Barras & Booth (2026), OpenNeuro ds007262 v1.1.0',
                eeg_reference='Linked-ear reference (README)',
                reference='Linked-ear reference',
                window_overlap_seconds=0,
                cohort_overlap_note='Recorded by Barras & Booth (University of Hull) on the same 20-participant mobile EEG+pupil cohort as NBackDifficulty (ds007169); treat as separate task paradigms on a shared participant cohort.',
            ),
        )
        return spec, arithmetic_records, cache, 'ds007262-cognitive'

    if key == 'nback_difficulty':
        cache = cache_root / 'CognitiveGroupA/ds007169'
        pts = [
            r['participant_id'].removeprefix('sub-')
            for r in core.table(cache / 'participants.tsv')
            if r.get('analysis_included', '').lower() == 'true'
        ]
        train, _ = whole_subject_split(pts)
        labels = {v[0]: v[1] for v in NBACK_LABELS.values()}
        spec = dict(
            name='NBackDifficulty',
            source='Barras & Booth / University of Hull / OpenNeuro ds007169 v1.0.5',
            cache='CognitiveGroupA/ds007169',
            hz=250,
            points=375,
            classes=4,
            train_subjects=len(train),
            licence='CC0-1.0',
            licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
            moabb_revision=None,
            doi='10.18112/openneuro.ds007169.v1.0.5',
            paper='Barras & Booth (2026), Multimodal Cognitive Workload n-back Task, 4 Difficulties, OpenNeuro ds007169 v1.0.5; Bateson & Asghar (2021), IEEE Access',
            labels=labels,
            splitter=lambda people, chunks: set(people) & train,
            evidence_files=['cognitive-group-a-plan.md', 'ds007169-cognitive-source-lock.json'],
            cache_evidence_files=[p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'task-nback_events.json', 'original-source-lock.json'] if (cache / p).is_file()],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note=(
                '4-class visual n-back working-memory load classification (1-back, 2-back, 3-back, 4-back) '
                'on 18 QC-passed participants from OpenNeuro ds007169 v1.0.5 (sub-013 and sub-017 excluded '
                'per analysis_included=false). 19 10-20 scalp EEG channels at 250 Hz in float64 microvolts '
                '(D1 ECG and D2..D5 auxiliary channels excluded); 1.5-second (375-sample) non-overlapping '
                'post-stimulus epochs on non-tutorial trials (1.0 s letter + 0.5 s retention within ~1.7 s SOA). '
                'Decoded BrainVision EEG values are multiplied by 1e-6 in float64 to correct an upstream '
                'XDF->BIDS double-microvolt scaling bug in ds007169 v1.0.5 (convert_xdf_to_bids_eeg.py passed '
                'microvolt LSL data directly into mne.io.RawArray before mne_bids.write_raw_bids multiplied '
                'EEG channels by 1e6; the same bug was corrected upstream in v1.1.0 of sister dataset ds007262). '
                'Case set, case order, labels, participant IDs, and the 13/5 subject split are unchanged. '
                'Shares the University of Hull cohort with ArithmeticDifficulty (ds007262).'
            ),
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://openneuro.org/datasets/ds007169/versions/1.0.5',
                paper_reference='Barras & Booth (2026), OpenNeuro ds007169 v1.0.5',
                eeg_reference='Linked-ear reference (README)',
                reference='Linked-ear reference',
                window_overlap_seconds=0,
                preprocessing=(
                    'Native EEG selection, source-event crops, float64 microvolt scale (including 1e-6 '
                    'correction for upstream ds007169 v1.0.5 XDF->BIDS double-microvolt scaling); '
                    'no additional filter, baseline, resampling or rereference'
                ),
                amplitude_scale_correction=dict(
                    applied_date='2026-11-14',
                    multiplicative_factor=DS007169_V105_EEG_SCALE,
                    computation_dtype='float64',
                    target_units='uV',
                    reason=DS007169_V105_SCALE_REASON,
                    dc_offset_note=(
                        'The University of Hull Bateson & Asghar (2021) mobile EEG front-end is DC-coupled '
                        '(+-2.25 V full scale) without high-pass filtering; standing electrode half-cell '
                        'DC offsets (~0.14-1.50 V = 1.4e5-1.5e6 uV) remain in raw samples on both '
                        'NBackDifficulty and ArithmeticDifficulty while within-epoch per-channel SDs are ~51 uV.'
                    ),
                    case_order_and_ids_preserved=True,
                ),
                cohort_overlap_note='Recorded by Barras & Booth (University of Hull) on the same 20-participant mobile EEG+pupil cohort as ArithmeticDifficulty (ds007262); treat as separate task paradigms on a shared participant cohort.',
            ),
            validation_updates=dict(
                amplitude_scale_correction=dict(
                    applied_date='2026-11-14',
                    multiplicative_factor=DS007169_V105_EEG_SCALE,
                    reason=DS007169_V105_SCALE_REASON,
                    before_median_channel_sd={'TRAIN': 51618040.29671288, 'TEST': 50522676.33334918, 'TEST_first_200': 50578781.33163752},
                    case_set_order_labels_ids_unchanged=True,
                ),
            ),
        )
        return spec, nback_records, cache, 'ds007169-cognitive'

    if key == 'multitask_workload':
        cache = cache_root / 'Neuro2021'
        people = [f'{i:02d}' for i in range(1, 16)]
        train, _ = whole_subject_split(people)
        labels = {v[0]: v[1] for v in MATB_LABELS.values()}
        spec = dict(
            name='MultitaskWorkload',
            source='Hinss, Darmet, Somon, Jahanpour, Lotte, Ladouce & Roy / Neuroergonomics 2021 Grand Challenge / Zenodo 5055046',
            cache='Neuro2021',
            hz=250,
            points=500,
            classes=3,
            train_subjects=len(train),
            licence='CC-BY-SA-4.0',
            licence_url='https://creativecommons.org/licenses/by-sa/4.0/',
            moabb_revision=None,
            doi='10.5281/zenodo.5055046',
            paper='Hinss et al. (2023), An EEG dataset for cross-session mental workload estimation: Passive BCI competition of the Neuroergonomics Conference 2021, Scientific Data 10, 85, DOI 10.1038/s41597-022-01898-y',
            labels=labels,
            splitter=lambda p, chunks: set(p) & train,
            evidence_files=['cognitive-group-a-plan.md', 'neuro2021-cognitive-source-lock.json'],
            cache_evidence_files=[],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='3-class ecological multitasking mental workload classification (easy, medium, difficult MATB-II blocks) across 15 participants and 2 labelled sessions (S1, S2) from the Neuroergonomics 2021 passive-BCI Grand Challenge (Zenodo 5055046). 61 active scalp EEG channels at 250 Hz in float64 microvolts on native released 2.0-second (500-sample) non-overlapping epochs; unlabelled competition test session S3 and resting files excluded. Do not duplicate with the 15-subject subset of the full 29-subject COG-BCI Zenodo archive (6874129).',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://zenodo.org/records/5055046',
                paper_reference='Hinss et al. (2023), Scientific Data 10, 85, DOI 10.1038/s41597-022-01898-y',
                window_overlap_seconds=0,
                excluded_participants=[],
                cohort_overlap_note='Represents the 15-participant Neuroergonomics 2021 competition subset of the broader COG-BCI programme (Hinss et al., Zenodo 6874129); if COG-BCI is later ingested, restrict to non-MATB or disjoint participants.',
            ),
        )
        return spec, multitask_records, cache, 'neuro2021-cognitive'

    if key == 'cognitive_load_tasks':
        cache = cache_root / 'CLATEEG26'
        people = [f'{i:02d}' for i in range(1, 41)]
        train, _ = whole_subject_split(people)
        labels = {v[0]: v[1] for v in CLATEEG_LABELS.values()}
        spec = dict(
            name='CognitiveLoadTasks',
            source='Nirabi, Habaebi, Abd Rahman, Sidek & Yusoff / IIUM / Mendeley Data CLATEEG26 v3',
            cache='CLATEEG26',
            hz=250,
            points=1000,
            classes=4,
            train_subjects=len(train),
            licence='CC-BY-4.0',
            licence_url='https://creativecommons.org/licenses/by/4.0/',
            moabb_revision=None,
            doi='10.17632/pxysmjn3j2.3',
            paper='Nirabi, Habaebi, Abd Rahman, Sidek & Yusoff (2026), Cognitive Load Assessment Through EEG: A Dataset from Arithmetic and Stroop Tasks (CLATEEG26), Mendeley Data v3, DOI 10.17632/pxysmjn3j2.3',
            labels=labels,
            splitter=lambda p, chunks: set(p) & train,
            evidence_files=['cognitive-group-a-plan.md', 'clateeg26-cognitive-source-lock.json'],
            cache_evidence_files=[],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='4-class protocol-induced cognitive load classification (neutral, low, medium, high) across Stroop Colour-Word and Mental Arithmetic tasks in 40 healthy adults from CLATEEG26 (Mendeley Data 10.17632/pxysmjn3j2.3). 8 OpenBCI Cyton 10-20 scalp EEG channels (Fp1, Fp2, F7, F3, Fz, F4, F8, C4) at 250 Hz in float64 microvolts; first 15 consecutive non-overlapping 4.0-second (1,000-sample) windows (60 s) per task-condition recording.',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://data.mendeley.com/datasets/pxysmjn3j2/3',
                paper_reference='Nirabi et al. (2026), Mendeley Data v3, DOI 10.17632/pxysmjn3j2.3',
                window_overlap_seconds=0,
                excluded_participants=[],
            ),
        )
        return spec, clateeg_records, cache, 'clateeg26-cognitive'

    if key == 'sternberg_memory':
        cache = cache_root / 'CognitiveGroupA/ds004117'
        pts = [r['participant_id'].removeprefix('sub-') for r in core.table(cache / 'participants.tsv')]
        train, _ = whole_subject_split(pts)
        labels = {v[0]: v[1] for v in STERNBERG_LABELS.values()}
        spec = dict(
            name='SternbergMemory',
            source='Onton, Delorme & Makeig / SCCN UCSD / OpenNeuro ds004117 v1.0.1',
            cache='CognitiveGroupA/ds004117',
            hz=250,
            points=500,
            classes=3,
            train_subjects=len(train),
            licence='CC0-1.0',
            licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
            moabb_revision=None,
            doi='10.18112/openneuro.ds004117.v1.0.1',
            paper='Onton, Delorme & Makeig (2005), Frontal midline EEG dynamics during working memory, NeuroImage 27(2), 341-356, DOI 10.1016/j.neuroimage.2005.04.014',
            labels=labels,
            splitter=lambda p, chunks: set(p) & train,
            evidence_files=['cognitive-group-a-plan.md', 'ds004117-cognitive-source-lock.json'],
            cache_evidence_files=[p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'original-source-lock.json'] if (cache / p).is_file()],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='3-class modified Sternberg working-memory load classification (3, 5, or 7 memorised letters vs ignored distractors) during the 2.0-second (500-sample) post-encoding maintenance period across 23 participants from OpenNeuro ds004117 v1.0.1. 69 scalp EEG channels (LEYE and REYE EOG excluded) harmonised to 250 Hz in float64 microvolts via whole-recording anti-alias resampling where native runs were sampled at 500 Hz, 500.059 Hz, or 1,000 Hz.',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://openneuro.org/datasets/ds004117/versions/1.0.1',
                paper_reference='Onton, Delorme & Makeig (2005), NeuroImage 27(2), 341-356',
                eeg_reference='Right mastoid (README)',
                reference='Right mastoid',
                window_overlap_seconds=0,
                excluded_participants=[],
            ),
        )
        return spec, sternberg_records, cache, 'ds004117-cognitive'

    if key == 'naturalistic_emotion':
        cache = cache_root / 'CognitiveGroupA/ds003751'
        receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
        beh_subs = {
            f['name'].split('/')[0]
            for f in receipt['files']
            if f['name'].startswith('sub-') and f['name'].endswith('_beh.tsv')
        }
        intact_people = []
        for f in sorted(receipt['files'], key=lambda x: x['name']):
            if f['name'].startswith('sub-') and f['name'].endswith('_eeg.set'):
                sub = f['name'].split('/')[0]
                if sub not in beh_subs:
                    continue
                sp = cache / f['name']
                fp = sp.with_suffix('.fdt')
                hdr_mat = loadmat(sp, simplify_cells=True)
                hdr_eeg = hdr_mat.get('EEG', hdr_mat)
                exp_bytes = int(hdr_eeg['nbchan']) * int(hdr_eeg['pnts']) * int(hdr_eeg['trials']) * 4
                if fp.is_file() and fp.stat().st_size == exp_bytes:
                    intact_people.append(sub.removeprefix('sub-'))
        train, _ = whole_subject_split(intact_people)
        labels = {v[0]: v[1] for v in DENS_QUADRANTS.values()}
        spec = dict(
            name='NaturalisticEmotion',
            source='Mishra, Asif, Tiwary & Srinivasan / IIIT Allahabad / OpenNeuro ds003751 v1.0.6 (DENS)',
            cache='CognitiveGroupA/ds003751',
            hz=250,
            points=1000,
            classes=4,
            train_subjects=len(train),
            licence='CC0-1.0',
            licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
            moabb_revision=None,
            doi='10.18112/openneuro.ds003751.v1.0.6',
            paper='Mishra, Asif, Tiwary & Srinivasan (2023), Dataset on Emotion with Naturalistic Stimuli (DENS), OpenNeuro ds003751 v1.0.6; Asif et al. (2023), IEEE Access',
            labels=labels,
            splitter=lambda p, chunks: whole_subject_split(p)[0],
            evidence_files=['cognitive-group-a-plan.md', 'ds003751-cognitive-source-lock.json'],
            cache_evidence_files=[p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'CHANGES', 'LICENSE', 'original-source-lock.json'] if (cache / p).is_file()],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='4-class affective circumplex classification (HVHA, HVLA, LVHA, LVLA computed directly from participant numeric self-ratings valence >= 5.0 and arousal >= 5.0, avoiding click-only sparse Quadrant strings) during naturalistic emotional multimedia videos across the 9 byte-intact rated participants in OpenNeuro ds003751 v1.0.6 (DENS; 6 recordings lacking beh.tsv and 25 recordings with upstream byte-truncated .fdt binaries in OpenNeuro v1.0.0-v1.0.6 are excluded and logged). 128 active EGI HydroCel scalp EEG channels (E1..E128; vertex reference E129 and peripheral ECG/EMG excluded) at 250 Hz in float64 microvolts on consecutive non-overlapping 4.0-second (1,000-sample) windows. Note upstream licence metadata lists CC0 in dataset_description.json and ODbL-1.0 in LICENSE.',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://openneuro.org/datasets/ds003751/versions/1.0.6',
                paper_reference='Mishra et al. (2023), OpenNeuro ds003751 v1.0.6; Asif et al. (2023), IEEE Access',
                window_overlap_seconds=0,
                upstream_licence_note='dataset_description.json specifies License=CC0 whereas root LICENSE text is ODbL-1.0 / DBCL-1.0; both texts archived in source_metadata.',
            ),
        )
        return spec, naturalistic_emotion_records, cache, 'ds003751-cognitive'

    if key == 'sleep_deprivation':
        cache = cache_root / 'CognitiveGroupA/ds004902'
        receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
        bad_people = set()
        all_people = set()
        for f in receipt['files']:
            if f['name'].endswith('_eeg.set') and '_task-eyesopen_' in f['name']:
                person = f['name'].split('/')[0].removeprefix('sub-')
                all_people.add(person)
                sp = cache / f['name']
                fp = sp.with_suffix('.fdt')
                hdr_mat = loadmat(sp, simplify_cells=True)
                hdr_eeg = hdr_mat.get('EEG', hdr_mat)
                exp_bytes = int(hdr_eeg['nbchan']) * int(hdr_eeg['pnts']) * int(hdr_eeg['trials']) * 4
                if not fp.is_file() or fp.stat().st_size != exp_bytes:
                    bad_people.add(person)
        intact_pts = sorted(all_people - bad_people)
        train, _ = whole_subject_split(intact_pts)
        labels = {v[0]: v[1] for v in SLEEP_LABELS.values()}
        spec = dict(
            name='SleepDeprivation',
            source='Xiang, Fan, Bai, Lv & Lei / Southwest University / OpenNeuro ds004902 v1.0.8',
            cache='CognitiveGroupA/ds004902',
            hz=500,
            points=2000,
            classes=2,
            train_subjects=len(train),
            licence='CC0-1.0',
            licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
            moabb_revision=None,
            doi='10.18112/openneuro.ds004902.v1.0.8',
            paper='Xiang, Fan, Bai, Lv & Lei (2024), A resting-state EEG dataset for sleep deprivation, Scientific Data 11, 427, DOI 10.1038/s41597-024-03268-2',
            labels=labels,
            splitter=lambda p, chunks: whole_subject_split(p)[0],
            evidence_files=['cognitive-group-a-plan.md', 'ds004902-cognitive-source-lock.json'],
            cache_evidence_files=[p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'original-source-lock.json'] if (cache / p).is_file()],
            conversion_files=['prepare_cognitive_group_a.py'],
            intermediate_directory='classification-intermediate-full',
            note='Binary vigilance state classification (normal_sleep ses-1 vs sleep_deprived ses-2; counterbalanced chronological order recorded in SessionOrder) during eyes-open resting-state EEG across the 68 byte-intact paired participants from OpenNeuro ds004902 v1.0.8 (sub-01, sub-28, and sub-44 excluded and logged due to upstream byte-truncated ses-2 .fdt binaries on OpenNeuro). 61 scalp EEG channels harmonised to 500 Hz in float64 microvolts (oversampled 5,000 Hz recordings anti-alias decimated 10x before windowing); first 30 consecutive non-overlapping 4.0-second (2,000-sample) windows (120 s) per session.',
            metadata_updates=dict(
                domain='Psy',
                original_source_url='https://openneuro.org/datasets/ds004902/versions/1.0.8',
                paper_reference='Xiang et al. (2024), Scientific Data 11, 427, DOI 10.1038/s41597-024-03268-2',
                window_overlap_seconds=0,
                excluded_participants=[],
            ),
        )
        return spec, sleep_deprivation_records, cache, 'ds004902-cognitive'

    raise ValueError(f'Unknown Group A dataset key: {key}')


def cleanup_cache(cache: Path, intermediate_dir: str) -> dict:
    """Remove temporary NPZs and heavy binary source payloads after verified export."""
    inter = (cache / intermediate_dir).resolve()
    removed_npz = 0
    removed_npz_bytes = 0
    if inter.is_dir() and inter.is_relative_to(cache.resolve()):
        for f in inter.glob('*.npz'):
            removed_npz_bytes += f.stat().st_size
            f.unlink()
            removed_npz += 1
        if not any(inter.iterdir()):
            inter.rmdir()

    removed_raw = 0
    removed_raw_bytes = 0
    heavy_suffixes = {'.eeg', '.vhdr', '.vmrk', '.fdt', '.set', '.zip'}
    for p in list(cache.rglob('*')):
        if p.is_file() and p.suffix.lower() in heavy_suffixes:
            removed_raw_bytes += p.stat().st_size
            p.unlink()
            removed_raw += 1
    return dict(
        removed_intermediate_npz=removed_npz,
        removed_intermediate_bytes=removed_npz_bytes,
        removed_raw_files=removed_raw,
        removed_raw_bytes=removed_raw_bytes,
    )


def prepare_one(
    key: str,
    cache_root: Path = Path('D:/Data/EEGSources'),
    output: Path = Path('D:/Data/EEG'),
    purge_cache: bool = True,
) -> dict:
    cache_root = Path(cache_root)
    output = Path(output)
    if key in OPENNEURO_RELEASES:
        lock_and_download_openneuro(key, cache_root)
    elif key == 'multitask_workload':
        lock_and_download_neuro2021(cache_root)
    elif key == 'cognitive_load_tasks':
        lock_and_download_clateeg26(cache_root)
    else:
        raise ValueError(f'Unknown key: {key}')

    spec, reader, cache, doc_stem = _build_spec(key, cache_root)
    core.SPECS[key] = spec
    core.READERS[key] = reader
    result = core.prepare(key, cache_root, output)
    folder = Path(result['path'])
    name = result['dataset']
    m = json.loads((folder / f'{name}_metadata.json').read_text(encoding='utf-8'))
    picks = m['recordings'][0]['source_eeg_indices']
    for sensor in m['sensors']:
        sensor['native_channel_index'] = picks[sensor['pre_flat_removal_array_index']]
        sensor['identity_evidence'] = 'Verified native scalp EEG channel order; non-scalp auxiliary channels excluded'
    result = finish(
        result,
        m,
        spec['evidence_files'],
        spec['conversion_files'],
        dict(
            label_mapping=spec['labels'],
            split_definition=m['split_definition'],
            excluded_participants=m.get('excluded_participants', []),
        ),
    )
    verify_final(result)
    if purge_cache:
        result['cache_cleanup'] = cleanup_cache(cache, spec['intermediate_directory'])
    dump(ROOT / 'docs' / f'{doc_stem}-conversion-result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=GROUP_A_ORDER,
        default=GROUP_A_ORDER,
    )
    parser.add_argument('--cache-root', type=Path, default=Path('D:/Data/EEGSources'))
    parser.add_argument('--output', type=Path, default=Path('D:/Data/EEG'))
    parser.add_argument('--keep-cache', action='store_true')
    args = parser.parse_args()
    for key in args.datasets:
        target = args.output / {
            'arithmetic_difficulty': 'ArithmeticDifficulty',
            'nback_difficulty': 'NBackDifficulty',
            'multitask_workload': 'MultitaskWorkload',
            'cognitive_load_tasks': 'CognitiveLoadTasks',
            'sternberg_memory': 'SternbergMemory',
            'naturalistic_emotion': 'NaturalisticEmotion',
            'sleep_deprivation': 'SleepDeprivation',
        }[key]
        if target.exists():
            print(f'Skipping already-exported {target.name}; verifying manifest...', flush=True)
            doc_map = {
                'arithmetic_difficulty': 'ds007262-cognitive',
                'nback_difficulty': 'ds007169-cognitive',
                'multitask_workload': 'neuro2021-cognitive',
                'cognitive_load_tasks': 'clateeg26-cognitive',
                'sternberg_memory': 'ds004117-cognitive',
                'naturalistic_emotion': 'ds003751-cognitive',
                'sleep_deprivation': 'ds004902-cognitive',
            }
            res_path = ROOT / 'docs' / f'{doc_map[key]}-conversion-result.json'
            verify_final(json.loads(res_path.read_text(encoding='utf-8')))
            continue
        prepare_one(key, args.cache_root, args.output, purge_cache=not args.keep_cache)


if __name__ == '__main__':
    main()
