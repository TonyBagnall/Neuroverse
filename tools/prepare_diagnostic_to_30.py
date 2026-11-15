"""Prepare Diagnostic-to-30 Phase 1 (+8 open clinical scalp-EEG problems -> 28 Clinical).

Streams raw upstream recordings in bounded participant batches onto C:/Data/EEGSources/DiagnosticTo30,
verifies cryptographic checksums and byte counts for every file, extracts native-Hz float64 microvolt
scalp-EEG windows without added filtering, baseline subtraction, or rereferencing, checks free space on
D:/Data/EEG before export (overflowing to C:/Data/EEG with an NTFS directory junction when D: headroom
is below threshold), validates exact float64 round-trip and amplitude scale [0.5, 500] uV, and purges
staged raw binaries and intermediate .npz caches immediately after verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import urllib.parse
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat

from tools import prepare_open_moabb as core
from tools.download_eegdash_diagnosis import blob_sha
from tools.prepare_cognitive_group_a import (
    _h5_str_or_val,
    decode_brainvision,
    decode_eeglab_set,
    verify_final,
)
from tools.prepare_cognitive_group_b import (
    SHARED_DS003753,
    create_junction_if_needed,
    decode_edf,
    resolve_storage_destination,
)
from tools.prepare_moabb import dump, sha
from tools.prepare_small_moabb import fetch
from tools.prepare_wang import digest_case
from tools.update_dataset_sidecars import finish

ROOT = Path(__file__).resolve().parents[1]

PHASE1_ORDER = [
    'albrecht2019',
    'pardogarcia2026',
    'ds004595',
    'siena',
    'ds007605',
    'ds004000',
    'ds004367',
    'ds007181',
    'ds003506',
    'ds003523',
]

DATASET_NAMES = {
    'albrecht2019': 'SchizophreniaConflict',
    'pardogarcia2026': 'StrokeMotorImagery',
    'ds004595': 'AlcoholBandit',
    'siena': 'AdultSeizures',
    'ds007605': 'PostCovidVigilance',
    'ds004000': 'SchizophreniaUltimatum',
    'ds004367': 'SchizophreniaMotion',
    'ds007181': 'ZosterNeuralgiaSleep',
    'ds003506': 'ParkinsonsReinforcement',
    'ds003523': 'AcuteTBIVisualMemory',
}

ESTIMATED_EXPORT_GB = {
    'albrecht2019': 10.5,
    'pardogarcia2026': 1.8,
    'ds004595': 3.0,
    'siena': 1.0,
    'ds007605': 18.5,
    'ds004000': 8.0,
    'ds004367': 5.5,
    'ds007181': 1.0,
    'ds003506': 3.6,
    'ds003523': 5.5,
}

ALBRECHT_REPO = 'jalauer/Albrecht2019'
ALBRECHT_REV = '53b7c8732740f56325687eca673ae2eed8a2c5f6'
ALBRECHT_EOG_INDICES = {4, 26, 37, 59}
ALBRECHT_STIM_TRIGGERS = {
    '111', '112', '113', '114',
    '121', '122', '123', '124',
    '211', '212', '213', '214',
    '221', '222', '223', '224',
}

PARDO_RECORD_ID = '19599466'
PARDO_EXCLUDE_INDICES = {28, 29, 30, 31, 62}
PARDO_CONTROL_MAP = {
    'C01': 'CTL01',
    '02': 'CTL02',
    'C03': 'CTL03',
    'CONTROL04': 'CTL04',
    'CONTROL05': 'CTL05',
    'CONTROL06': 'CTL06',
    'CONTROL07': 'CTL07',
    'CONTROL08': 'CTL08',
}

SIENA_BASE_URL = 'https://physionet.org/files/siena-scalp-eeg/1.0.0/'
SIENA_CANONICAL_29 = [
    'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5',
    'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4',
    'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10',
]

DS004367_FIXED_INDICES = (
    list(range(0, 16))
    + list(range(17, 34))
    + list(range(35, 42))
    + list(range(43, 62))
)
DS004367_FIXED_59 = [
    'FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7',
    'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'O9', 'O10',
]
DS004367_CANONICAL_61 = DS004367_FIXED_59 + ['TP8', 'AFz']

DS007181_CANONICAL_19 = [
    'Fp1-M2', 'Fp2-M1', 'F3-M2', 'F4-M1', 'F7-M2', 'F8-M1',
    'C3-M2', 'C4-M1', 'T3-M2', 'T4-M1', 'T5-M2', 'T6-M1',
    'P3-M2', 'P4-M1', 'O1-M2', 'O2-M1', 'Fz', 'Cz', 'Pz',
]
DS007181_TRUNCATED_UPSTREAM = {
    'sub-51': (
        'Upstream byte-truncated EDF on OpenNeuro ds007181 v1.0.1 '
        '(85,674,000 bytes vs 414,256,496 expected; cut mid-record at 6,191.63 / 29,940 records)'
    ),
}

DS003506_NON_EEG_TAIL = ['VEOG', 'X', 'Y', 'Z']
DS003506_PRACTICE_CUES = 6

DS003523_NON_EEG_TAIL = ['VEOG', 'EKG']
DS003523_EXCLUSIONS = {
    'sub-037': (
        'Exact duplicate BIDS export of sub-036 (both Original_ID=3013, identical ses-01 .fdt '
        'annex object md5:07cdb4cabe882d80087a742c0d6e327f)'
    ),
    'sub-064': 'Author clinical exclusion in s2_WM_Kill_Data.m: Original_ID=3004 had no loss of consciousness (no LOC)',
    'sub-084': 'Author clinical exclusion in s2_WM_Kill_Data.m: Original_ID=3056 had no loss of consciousness (no LOC)',
    'sub-068': 'Author clinical exclusion in s2_WM_Kill_Data.m: Original_ID=3024 had pre-existing head injury',
    'sub-022': 'Author clinical exclusion in s2_WM_Kill_Data.m: Original_ID=3053 had pre-existing head injury',
}
DS003523_STIM_MAP = {
    'S  1': '3_red',
    'S  2': '3_red_2_yellow',
    'S  3': '5_red',
}

OPENNEURO_PHASE1 = {
    'ds004595': dict(
        dataset='ds004595',
        ref='1.0.0',
        commit='3417f020ebe1f4ca1f62d3e54d88e89826e8bcec',
        eegdash_id='on004595',
        licence='CC0-1.0',
    ),
    'ds007605': dict(
        dataset='ds007605',
        ref='1.0.0',
        commit='361bc7a3898f8199559e857a887eaf48353101df',
        eegdash_id='on007605',
        licence='CC0-1.0',
    ),
    'ds004000': dict(
        dataset='ds004000',
        ref='1.0.0',
        commit='b857fb0f95a9ad8bc503b7569d379a92a7750ff8',
        eegdash_id='on004000',
        licence='CC0-1.0',
    ),
    'ds004367': dict(
        dataset='ds004367',
        ref='1.0.2',
        commit='c0747aa0c9f87928faa9189ce22e6bece69461b6',
        eegdash_id='on004367',
        licence='CC0-1.0',
    ),
    'ds007181': dict(
        dataset='ds007181',
        ref='1.0.1',
        commit='2499c3d0f3dca9b3e84b8faf01e7838374dcc621',
        eegdash_id='on007181',
        licence='CC0-1.0',
    ),
    'ds003506': dict(
        dataset='ds003506',
        ref='1.1.0',
        commit='15fd795b705aca65c49c4c538a76c06f8b33b84a',
        eegdash_id='on003506',
        licence='CC0-1.0',
    ),
    'ds003523': dict(
        dataset='ds003523',
        ref='1.1.0',
        commit='ae850fd9843ef2ec32346e6183b13a59217d2a55',
        eegdash_id='on003523',
        licence='CC0-1.0',
    ),
}

SAMPLE_REF_CONT = 'Zero-based continuous recording sample index; start inclusive, end exclusive'
SAMPLE_REF_SEG = 'Zero-based concatenated file sample index within native marker-segmented recording; start inclusive, end exclusive'


def sanitise_text(value: str) -> str:
    """Replace any em dash or en dash in upstream free text with ASCII hyphenation."""
    return value.replace('\u2014', ' - ').replace('\u2013', '-')


def decode_brainvision_safe(
    vhdr_path: Path,
    fallback_eeg_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Decode BrainVision .vhdr/.eeg, falling back to BIDS stem .eeg when DataFile header is stale."""
    vhdr_path = Path(vhdr_path)
    try:
        return decode_brainvision(vhdr_path)
    except FileNotFoundError:
        if fallback_eeg_path is None:
            raise
        fallback_eeg = Path(fallback_eeg_path)
        if not fallback_eeg.is_file() or fallback_eeg.resolve().parent != vhdr_path.resolve().parent:
            raise
        raw_text = vhdr_path.read_text(encoding='utf-8-sig')
        stale_match = re.search(r'(?m)^DataFile=(.+)$', raw_text)
        if stale_match is None:
            raise
        stale_datafile = stale_match.group(1).strip()
        patched_text = re.sub(
            r'(?m)^DataFile=.+$',
            lambda _: f'DataFile={fallback_eeg.name}',
            raw_text,
            count=1,
        )
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.vhdr',
            dir=vhdr_path.parent,
            delete=False,
        ) as tmp:
            tmp.write(patched_text)
            tmp_vhdr = Path(tmp.name)
        try:
            arr, meta = decode_brainvision(tmp_vhdr)
        finally:
            if tmp_vhdr.exists():
                tmp_vhdr.unlink()
        meta['stale_vhdr_datafile'] = stale_datafile
        meta['native_data_file'] = fallback_eeg.name
        return arr, meta


def disk_free_gb(path: Path) -> float:
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free / 1e9


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def stratified_diagnostic_split(
    subject_to_label: dict[str, str],
    train_ratio: float = 0.75,
) -> tuple[set[str], set[str]]:
    """Deterministic stratified whole-subject 75/25 split preserving class proportions."""
    if not subject_to_label:
        raise ValueError('Empty subject-to-label mapping')
    by_class: dict[str, list[str]] = {}
    for pid, label in sorted(subject_to_label.items()):
        by_class.setdefault(str(label), []).append(pid)
    train: set[str] = set()
    test: set[str] = set()
    for label, subs in sorted(by_class.items()):
        subs_sorted = sorted(subs)
        if len(subs_sorted) < 2:
            raise ValueError(f'Class {label!r} has fewer than 2 subjects ({len(subs_sorted)})')
        n_train = max(1, min(len(subs_sorted) - 1, int(len(subs_sorted) * train_ratio)))
        train.update(subs_sorted[:n_train])
        test.update(subs_sorted[n_train:])
    if train & test:
        raise ValueError('Subject overlap in stratified split')
    return train, test


def whole_patient_split(patients: list[str], train_ratio: float = 0.75) -> tuple[set[str], set[str]]:
    """Deterministic whole-patient 75/25 split for within-patient multi-state clinical cohorts."""
    ordered = sorted(set(patients))
    if len(ordered) < 2:
        raise ValueError('Need at least two patients for whole-patient split')
    n_train = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_ratio))))
    train = set(ordered[:n_train])
    test = set(ordered[n_train:])
    return train, test


def extract_clean_windows(
    data: np.ndarray,
    candidate_starts: list[int],
    points: int,
    boundary_samples: list[int] | None = None,
    max_windows: int | None = None,
    std_floor: float = 1e-7,
    seen_hashes: set[str] | None = None,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """Extract clean, finite, non-railed, non-overlapping windows without crossing boundaries."""
    n_samples = int(data.shape[1])
    bounds = sorted(set(int(b) for b in (boundary_samples or []) if 0 < int(b) < n_samples))
    accepted_x: list[np.ndarray] = []
    accepted_starts: list[int] = []
    dropped: list[dict] = []
    last_end = -1

    for raw_start in candidate_starts:
        if max_windows is not None and len(accepted_x) >= max_windows:
            break
        s = int(raw_start)
        e = s + int(points)
        if s < 0 or e > n_samples:
            dropped.append(dict(start_sample=s, reason='out_of_bounds'))
            continue
        if s < last_end:
            dropped.append(dict(start_sample=s, reason='overlapping_candidate'))
            continue
        if any(s < b < e for b in bounds):
            dropped.append(dict(start_sample=s, reason='crosses_boundary'))
            continue
        win = np.ascontiguousarray(data[:, s:e], dtype='<f8')
        if not np.isfinite(win).all():
            dropped.append(dict(start_sample=s, reason='nonfinite'))
            continue
        if (np.ptp(win, axis=1) == 0).any() or (win.std(axis=1) <= std_floor).any():
            dropped.append(dict(start_sample=s, reason='flat_or_railed_channel'))
            continue
        # Guard against zero-dropout blocks (>10% exact zeros across all channels simultaneously)
        if np.mean(np.all(win == 0.0, axis=0)) > 0.05:
            dropped.append(dict(start_sample=s, reason='zero_dropout'))
            continue
        h = digest_case(win)
        if seen_hashes is not None:
            if h in seen_hashes:
                dropped.append(dict(start_sample=s, reason='duplicate_signal_hash'))
                continue
            seen_hashes.add(h)
        accepted_x.append(win)
        accepted_starts.append(s)
        last_end = e

    return accepted_x, accepted_starts, dropped


def select_ds004367_channels(ch_names: list[str]) -> tuple[list[int], list[str]]:
    """Select the 61 unambiguous common scalp EEG channels across all 5 ds004367 layouts."""
    cleaned = [str(c).strip() for c in ch_names]
    if len(cleaned) != 68:
        raise ValueError(f'Expected 68 channels in ds004367, got {len(cleaned)}')
    fixed_actual = [cleaned[i] for i in DS004367_FIXED_INDICES]
    if fixed_actual != DS004367_FIXED_59:
        raise ValueError(f'ds004367 fixed 59 channels mismatch: {fixed_actual}')
    picks = list(DS004367_FIXED_INDICES)
    for target in ('TP8', 'AFz'):
        matches = [i for i, name in enumerate(cleaned) if name == target]
        if len(matches) != 1:
            raise ValueError(f'Expected unique channel {target!r} in ds004367, got {matches}')
        picks.append(matches[0])
    return picks, list(DS004367_CANONICAL_61)


def select_siena_channels(edf_labels: list[str]) -> list[int]:
    """Select and order the 29 canonical 10-10 scalp EEG channels across all Siena EDFs."""
    by_upper: dict[str, list[int]] = {}
    for idx, raw_lbl in enumerate(edf_labels):
        lbl = raw_lbl.strip()
        if lbl.upper().startswith('EEG '):
            short = lbl[4:].strip().upper()
            by_upper.setdefault(short, []).append(idx)
    picks: list[int] = []
    for canon in SIENA_CANONICAL_29:
        matches = by_upper.get(canon.upper(), [])
        if len(matches) != 1:
            raise ValueError(f'Siena channel {canon!r} matches={matches} in {edf_labels}')
        picks.append(matches[0])
    return picks


def read_edf_start_and_duration(edf_path: Path) -> dict:
    """Read EDF header start_time, start_date, duration_s, and channel labels without full decode."""
    with Path(edf_path).open('rb') as f:
        hdr = f.read(256)
        if len(hdr) < 256:
            raise ValueError(f'Truncated EDF header in {edf_path}')
        header_bytes = int(hdr[184:192].decode('ascii').strip())
        start_date = hdr[168:176].decode('ascii', errors='replace').strip()
        start_time = hdr[176:184].decode('ascii', errors='replace').strip()
        n_records = int(hdr[236:244].decode('ascii').strip())
        rec_dur = float(hdr[244:252].decode('ascii').strip())
        ns = int(hdr[252:256].decode('ascii').strip())
        labels_raw = f.read(ns * 16)
    labels = [labels_raw[i * 16 : (i + 1) * 16].decode('latin-1').strip() for i in range(ns)]
    return dict(
        header_bytes=header_bytes,
        start_date=start_date,
        start_time=start_time,
        n_records=n_records,
        rec_dur=rec_dur,
        duration_s=n_records * rec_dur,
        labels=labels,
    )


def parse_siena_clock_seconds(raw_value: str) -> int:
    """Parse Siena HH.MM.SS / HH:MM:SS timestamps, handling spaces and annotations."""
    cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', raw_value.strip())
    m = re.search(r'(\d{1,2})[.:](\d{2})[.:](\d{2})', cleaned)
    if not m:
        raise ValueError(f'Cannot parse Siena timestamp from {raw_value!r}')
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        raise ValueError(f'Out-of-range Siena clock time {raw_value!r}')
    return hh * 3600 + mm * 60 + ss


def normalise_siena_filename(raw_name: str, patient_id: str, available_edfs: set[str]) -> str:
    """Resolve documented filename typos in Seizures-list-PNxx.txt to exact EDF paths."""
    token = raw_name.strip().replace('PNO', 'PN0')
    if not token.lower().endswith('.edf'):
        token += '.edf'
    candidates = [
        f'{patient_id}/{token}',
        f'{patient_id}/{token.replace(".edf", "-1.edf")}',
        f'{patient_id}/{token.replace("-.edf", "-1.edf")}',
    ]
    for cand in candidates:
        if cand in available_edfs:
            return cand
    raise ValueError(f'Cannot match Siena EDF {raw_name!r} for {patient_id} in {sorted(available_edfs)}')


def parse_siena_seizure_list(
    text: str,
    patient_id: str,
    edf_headers: dict[str, dict],
) -> dict[str, list[tuple[float, float]]]:
    """Parse Seizures-list-PNxx.txt into verified (start_sec, end_sec) intervals per EDF."""
    available = {k for k in edf_headers if k.startswith(f'{patient_id}/')}
    seizures: dict[str, list[tuple[float, float]]] = {k: [] for k in sorted(available)}
    current_edf: str | None = None
    pending_start: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith('file name'):
            _, val = line.split(':', 1)
            current_edf = normalise_siena_filename(val, patient_id, available)
            pending_start = None
        elif low.startswith('seizure start time') or low.startswith('start time'):
            _, val = line.split(':', 1)
            pending_start = parse_siena_clock_seconds(val)
        elif low.startswith('seizure end time') or low.startswith('end time'):
            if current_edf is None or pending_start is None:
                raise ValueError(f'Unpaired seizure end line {line!r} in {patient_id}')
            _, val = line.split(':', 1)
            end_clock = parse_siena_clock_seconds(val)
            hdr = edf_headers[current_edf]
            reg_start = parse_siena_clock_seconds(hdr['start_time'])
            dur_s = float(hdr['duration_s'])

            start_rel = float((pending_start - reg_start) % 86400)
            end_rel = float((end_clock - reg_start) % 86400)
            if end_rel < start_rel:
                end_rel += 86400.0
            # Documented typo in PN00-3: '19.29.29' instead of '18.29.29' (recording ends 18.57.13)
            if end_rel > dur_s and (end_rel - 3600.0) > start_rel and (end_rel - 3600.0) <= dur_s:
                end_rel -= 3600.0
            if not (0.0 <= start_rel < end_rel <= dur_s):
                raise ValueError(
                    f'Siena seizure interval [{start_rel}, {end_rel}] outside [0, {dur_s}] in {current_edf}'
                )
            seizures[current_edf].append((start_rel, end_rel))
            pending_start = None

    total_sz = sum(len(v) for v in seizures.values())
    if total_sz == 0:
        raise ValueError(f'No seizures parsed for {patient_id}')
    return seizures


def select_siena_windows(
    data: np.ndarray,
    srate: float,
    points: int,
    seizures_sec: list[tuple[float, float]],
    max_interictal: int = 35,
    buffer_sec: float = 60.0,
    min_ictal_overlap_sec: float = 2.0,
    std_floor: float = 1e-7,
    seen_hashes: set[str] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, str]], list[dict]]:
    """Extract clean ictal (>=2 s seizure overlap) and evenly spaced interictal (>=60 s buffer) windows."""
    n_samples = int(data.shape[1])
    win_sec = points / srate
    ictal_candidates: list[tuple[np.ndarray, int]] = []
    inter_candidates: list[tuple[np.ndarray, int]] = []
    dropped: list[dict] = []

    for s in range(0, n_samples - points + 1, points):
        w0 = s / srate
        w1 = (s + points) / srate
        overlap = sum(max(0.0, min(w1, sz1) - max(w0, sz0)) for sz0, sz1 in seizures_sec)
        if overlap >= min_ictal_overlap_sec:
            target_cls = '1'
        else:
            far_from_all = all(
                (w1 <= sz0 - buffer_sec) or (w0 >= sz1 + buffer_sec)
                for sz0, sz1 in seizures_sec
            )
            if not far_from_all:
                continue
            target_cls = '2'

        win = np.ascontiguousarray(data[:, s : s + points], dtype='<f8')
        if not np.isfinite(win).all():
            dropped.append(dict(start_sample=s, reason='nonfinite'))
            continue
        if (np.ptp(win, axis=1) == 0).any() or (win.std(axis=1) <= std_floor).any():
            dropped.append(dict(start_sample=s, reason='flat_or_railed_channel'))
            continue
        if np.mean(np.all(win == 0.0, axis=0)) > 0.05:
            dropped.append(dict(start_sample=s, reason='zero_dropout'))
            continue
        if target_cls == '1':
            ictal_candidates.append((win, s))
        else:
            inter_candidates.append((win, s))

    if len(inter_candidates) > max_interictal:
        idx = np.linspace(0, len(inter_candidates) - 1, max_interictal, dtype=int)
        unique_idx = list(dict.fromkeys(int(i) for i in idx))
        selected_inter = [inter_candidates[i] for i in unique_idx]
    else:
        selected_inter = inter_candidates

    combined = [(w, s, '1') for w, s in ictal_candidates] + [(w, s, '2') for w, s in selected_inter]
    combined.sort(key=lambda item: item[1])

    out_x: list[np.ndarray] = []
    out_meta: list[tuple[int, str]] = []
    for win, s, cls in combined:
        h = digest_case(win)
        if seen_hashes is not None:
            if h in seen_hashes:
                dropped.append(dict(start_sample=s, reason='duplicate_signal_hash'))
                continue
            seen_hashes.add(h)
        out_x.append(win)
        out_meta.append((s, cls))

    if abs(win_sec - 4.0) > 1e-6:
        raise ValueError(f'Unexpected Siena window seconds {win_sec}')
    return out_x, out_meta, dropped


def parse_brainvision_vmrk(vmrk_path: Path) -> list[dict]:
    """Parse BrainVision .vmrk markers into 0-based sample positions."""
    text = Path(vmrk_path).read_text(encoding='utf-8', errors='replace')
    markers: list[dict] = []
    in_infos = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('[') and line.endswith(']'):
            in_infos = line.lower() == '[marker infos]'
            continue
        if in_infos and line.lower().startswith('mk') and '=' in line:
            _, rhs = line.split('=', 1)
            parts = rhs.split(',')
            if len(parts) >= 3:
                mtype = parts[0].strip()
                mdesc = parts[1].strip()
                pos_1based = int(parts[2].strip())
                dur = int(parts[3].strip()) if len(parts) >= 4 and parts[3].strip().isdigit() else 1
                markers.append(
                    dict(
                        type=mtype,
                        description=mdesc,
                        sample=pos_1based - 1,
                        sample_1based=pos_1based,
                        duration=dur,
                    )
                )
    return markers


def _load_albrecht_eeg_mat(mat_path: Path) -> tuple[np.ndarray, dict]:
    """Load Albrecht2019 CC_EEG_*.mat (MATLAB v5 or HDF5 v7.3) into float64 microvolts."""
    if h5py.is_hdf5(mat_path):
        with h5py.File(mat_path, 'r') as f:
            g = f['EEG'] if 'EEG' in f else f
            srate = float(np.asarray(g['srate']).item())
            nbchan = int(round(float(np.asarray(g['nbchan']).item())))
            pnts = int(round(float(np.asarray(g['pnts']).item())))
            ref_val = str(_h5_str_or_val(f, g['ref']) or '') if 'ref' in g else ''
            locs = g['chanlocs']
            ch_names = [str(_h5_str_or_val(f, r)).strip() for r in np.asarray(locs['labels']).ravel()]
            events = []
            if 'event' in g and isinstance(g['event'], h5py.Group) and 'latency' in g['event']:
                lat_refs = np.asarray(g['event']['latency']).ravel()
                typ_refs = np.asarray(g['event']['type']).ravel() if 'type' in g['event'] else []
                code_refs = np.asarray(g['event']['code']).ravel() if 'code' in g['event'] else []
                for idx in range(len(lat_refs)):
                    events.append(
                        dict(
                            latency=float(_h5_str_or_val(f, lat_refs[idx])),
                            type=str(_h5_str_or_val(f, typ_refs[idx])).strip() if idx < len(typ_refs) else '',
                            code=str(_h5_str_or_val(f, code_refs[idx])).strip() if idx < len(code_refs) else '',
                        )
                    )
            raw_arr = np.asarray(g['data'], dtype='<f8')
            data = raw_arr.T if raw_arr.shape == (pnts, nbchan) else raw_arr
    else:
        mat = loadmat(mat_path, simplify_cells=True)
        eeg = mat.get('EEG', mat)
        srate = float(eeg['srate'])
        nbchan = int(eeg['nbchan'])
        pnts = int(eeg['pnts'])
        ref_val = str(eeg.get('ref', ''))
        chanlocs = eeg['chanlocs']
        if isinstance(chanlocs, dict):
            chanlocs = [chanlocs]
        ch_names = [str(c['labels']).strip() for c in chanlocs]
        raw_events = eeg.get('event', [])
        if isinstance(raw_events, dict):
            raw_events = [raw_events]
        events = []
        for ev in list(raw_events):
            if isinstance(ev, dict) and 'latency' in ev:
                events.append(
                    dict(
                        latency=float(ev['latency']),
                        type=str(ev.get('type', '')).strip(),
                        code=str(ev.get('code', '')).strip(),
                    )
                )
        data = np.asarray(eeg['data'], dtype='<f8')
    if data.shape != (nbchan, pnts):
        raise ValueError(f'Unexpected Albrecht2019 EEG shape {data.shape} in {mat_path}')
    return data, dict(srate=srate, nbchan=nbchan, pnts=pnts, ref=ref_val, ch_names=ch_names, events=events)


def _write_receipt(cache: Path, doc_slug: str, receipt: dict) -> None:
    dump(cache / 'download-receipt.json', receipt)
    dump(cache / 'source-record.json', receipt)
    dump(cache / 'original-source-lock.json', receipt)
    dump(ROOT / 'docs' / f'{doc_slug}-diagnosis-source-lock.json', receipt)


def bootstrap_openneuro_metadata(key: str, cache: Path, annex_eager_exts: tuple[str, ...] = ()) -> tuple[dict, list[dict]]:
    """Lock OpenNeuro git tree + inline BIDS metadata, optionally fetching tiny header pointers upfront."""
    info = OPENNEURO_PHASE1[key]
    ds = info['dataset']
    ref = info['ref']
    commit = info['commit']
    cache.mkdir(parents=True, exist_ok=True)

    archive = cache / 'original-metadata.zip'
    fetch(f'https://codeload.github.com/OpenNeuroDatasets/{ds}/zip/{commit}', archive)

    inline_files: list[dict] = []
    annex_candidates: list[dict] = []
    with zipfile.ZipFile(archive) as z:
        archive_commit = z.comment.decode('ascii', errors='replace').strip()
        if archive_commit and archive_commit != commit:
            raise ValueError(f'Commit mismatch in {ds} archive: expected {commit}, got {archive_commit}')
        prefix = f'{ds}-{commit}/'
        for zi in z.infolist():
            if zi.is_dir() or not zi.filename.startswith(prefix):
                continue
            name = zi.filename[len(prefix):]
            if not name:
                continue
            data = z.read(zi)
            entry_sha = blob_sha(data)
            is_symlink = ((zi.external_attr >> 16) & 0o170000) == 0o120000
            if is_symlink:
                pointer = data.decode('utf-8', errors='replace')
                m = re.search(r'(SHA256|MD5)E-s(\d+)--([0-9a-f]+)(?:\.[^/]+)?$', pointer.strip())
                if not m or len(m[3]) != {'SHA256': 64, 'MD5': 32}[m[1]]:
                    continue
                if name.startswith('sub-') and '/eeg/' in name:
                    annex_candidates.append(
                        dict(
                            name=name,
                            url=f'https://s3.amazonaws.com/openneuro.org/{ds}/{name}',
                            bytes=int(m[2]),
                            checksum=m[1].lower() + ':' + m[3],
                            git_blob_sha1=entry_sha,
                            git_annex_pointer=pointer.strip(),
                        )
                    )
            else:
                if name.startswith(('.', 'sourcedata/', 'code/', 'stimuli/', 'derivatives/')):
                    continue
                if name.startswith('sub-') and '/eeg/' not in name:
                    continue
                target = (cache / name).resolve()
                if not target.is_relative_to(cache.resolve()):
                    raise ValueError(f'Unsafe inline path {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                inline_files.append(
                    dict(
                        name=name,
                        url=f'https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{commit}/{name}',
                        bytes=len(data),
                        checksum='sha256:' + hashlib.sha256(data).hexdigest(),
                        sha256=hashlib.sha256(data).hexdigest(),
                        git_blob_sha1=entry_sha,
                    )
                )

    eager_downloaded: list[dict] = []
    deferred_annex: list[dict] = []
    for cand in annex_candidates:
        if annex_eager_exts and cand['name'].lower().endswith(annex_eager_exts):
            eager_downloaded.append(cand)
        else:
            deferred_annex.append(cand)

    if eager_downloaded:
        def _dl_small(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            if not target.is_relative_to(cache.resolve()):
                raise ValueError(f"Unsafe annex path {f['name']}")
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=8) as pool:
            eager_verified = list(pool.map(_dl_small, eager_downloaded))
        inline_files.extend(eager_verified)

    inline_files.sort(key=lambda item: item['name'])
    receipt = dict(
        dataset=ds,
        source_record_id=ds,
        source_revision=f'OpenNeuro {ds} v{ref}, Git {commit}',
        original_ref=ref,
        original_commit=commit,
        eegdash_id=info['eegdash_id'],
        licence=info['licence'],
        licence_evidence='Original pinned dataset_description.json License=CC0',
        source_urls=[f'https://openneuro.org/datasets/{ds}/versions/{ref}'],
        metadata_archive_sha256=sha(archive),
        files=inline_files,
    )
    _write_receipt(cache, ds, receipt)
    return receipt, sorted(deferred_annex, key=lambda item: item['name'])


# ---------------------------------------------------------------------------
# #1 SchizophreniaConflict (Albrecht2019, PRED+CT d004)
# ---------------------------------------------------------------------------
def bootstrap_albrecht2019(cache: Path) -> tuple[dict, list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    tree_path = cache / 'hf-tree.json'
    fetch(
        f'https://huggingface.co/api/datasets/{ALBRECHT_REPO}/tree/{ALBRECHT_REV}?recursive=true',
        tree_path,
    )
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    inline_names = {
        'DataDictionary.txt',
        'DeID_Dems.csv',
        'LICENSE.txt',
        'MMN_ElectrodeCoords_MA2.ced',
        'Pre-processing_Pseudo.txt',
        'README.md',
    }
    inline_files: list[dict] = []
    eeg_mats: list[dict] = []
    for entry in tree:
        if entry.get('type') != 'file':
            continue
        p = entry['path']
        url = f'https://huggingface.co/datasets/{ALBRECHT_REPO}/resolve/{ALBRECHT_REV}/' + urllib.parse.quote(p)
        ck = ('sha256:' + entry['lfs']['oid']) if entry.get('lfs') else ('gitblob:' + entry['oid'])
        rec = dict(name=p, url=url, bytes=int(entry['size']), checksum=ck)
        if p in inline_names:
            target = (cache / p).resolve()
            fetch(url, target, ck, rec['bytes'])
            inline_files.append({**rec, 'sha256': sha(target)})
        elif p.startswith('data/CC_EEG_s') and p.endswith('.mat'):
            eeg_mats.append(rec)

    inline_files.sort(key=lambda item: item['name'])
    eeg_mats.sort(key=lambda item: item['name'])
    receipt = dict(
        dataset='Albrecht2019',
        source_record_id='PRED+CT d004 (jalauer/Albrecht2019)',
        source_revision=f'Hugging Face {ALBRECHT_REPO} commit {ALBRECHT_REV} (PRED+CT d004)',
        predct_accession='d004',
        licence='PDDL-1.0',
        licence_evidence='Original PRED+CT downloads page and mirror root LICENSE.txt grant PDDL v1.0',
        source_urls=[
            'https://predict.cs.unm.edu/downloads.php',
            f'https://huggingface.co/datasets/{ALBRECHT_REPO}/tree/{ALBRECHT_REV}',
        ],
        files=inline_files,
    )
    _write_receipt(cache, 'albrecht2019', receipt)
    return receipt, eeg_mats


def albrecht2019_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    eeg_mats = spec['_deferred_files']
    dems_rows = list(csv.DictReader((cache / 'DeID_Dems.csv').open(encoding='utf-8-sig', newline='')))
    dems_by_id = {r['subno'].strip(): r['group'].strip() for r in dems_rows}

    mat_by_id: dict[str, dict] = {}
    excluded: list[dict] = []
    for f in eeg_mats:
        m = re.match(r'^data/CC_EEG_s(\d+)_([PN])\.mat$', f['name'])
        if not m:
            continue
        sub_id, cond = m.group(1), m.group(2)
        if sub_id not in dems_by_id:
            excluded.append(dict(participant_id=f's{sub_id}', file=f['name'], reason='Absent from DeID_Dems.csv'))
            continue
        grp = dems_by_id[sub_id]
        if (grp == 'SZ' and cond != 'P') or (grp == 'HC' and cond != 'N'):
            raise ValueError(f'Label mismatch for {sub_id}: DeID_Dems={grp}, filename={cond}')
        mat_by_id[sub_id] = f

    for sub_id, grp in sorted(dems_by_id.items()):
        if sub_id not in mat_by_id:
            excluded.append(dict(participant_id=f's{sub_id}', group=grp, reason='No CC_EEG_*.mat file in repository'))

    spec['metadata_updates']['excluded_participants'] = excluded
    subject_labels = {
        f's{sub_id}': ('1' if dems_by_id[sub_id] == 'SZ' else '2')
        for sub_id in sorted(mat_by_id)
    }
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by group (SZ vs HC) in sorted ID order: '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; zero shared participants.'
    )

    stream_dir = cache / '_stream_batch'
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    canonical_names: list[str] | None = None
    ordered_ids = sorted(mat_by_id)

    for batch_ids in batched(ordered_ids, 6):
        stream_dir.mkdir(parents=True, exist_ok=True)
        batch_files = [mat_by_id[sid] for sid in batch_ids]

        def _dl(f: dict) -> tuple[Path, dict]:
            p = stream_dir / Path(f['name']).name
            fetch(f['url'], p, f['checksum'], f['bytes'])
            return p, {**f, 'sha256': sha(p)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = list(pool.map(_dl, batch_files))

        for sid, (local_mat, file_rec) in zip(batch_ids, fetched, strict=True):
            try:
                verified_stream.append(file_rec)
                raw_data, meta = _load_albrecht_eeg_mat(local_mat)
                if round(meta['srate']) != spec['hz']:
                    raise ValueError(f"Expected {spec['hz']} Hz in {file_rec['name']}, got {meta['srate']}")
                all_ch = meta['ch_names']
                if len(all_ch) != 64:
                    raise ValueError(f"Expected 64 channels in {file_rec['name']}, got {len(all_ch)}")
                picks = [i for i in range(64) if i not in ALBRECHT_EOG_INDICES]
                ch_names = [all_ch[i] for i in picks]
                if canonical_names is None:
                    canonical_names = ch_names
                elif ch_names != canonical_names:
                    raise ValueError(f"Channel mismatch in {file_rec['name']}: {ch_names}")

                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                bounds = [
                    int(round(ev['latency'] - 1.0))
                    for ev in meta['events']
                    if ev['type'].lower() == 'boundary' or 'new segment' in ev['code'].lower()
                ]
                stim_samples = [
                    int(round(ev['latency'] - 1.0))
                    for ev in meta['events']
                    if ev['type'] in ALBRECHT_STIM_TRIGGERS
                ]
                start0 = min(stim_samples) if stim_samples else 0
                candidates = list(range(start0, scalp.shape[1] - spec['points'] + 1, spec['points']))
                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for Albrecht2019 s{sid}')
                pid = f's{sid}'
                grp = dems_by_id[sid]
                lbl = subject_labels[pid]
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='1',
                        recording_id=pid,
                        source_file=file_rec['name'],
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=grp,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    session_id='1',
                    source_file=file_rec['name'],
                    source_file_sha256=file_rec['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    native_samples=meta['pnts'],
                    native_channels=meta['nbchan'],
                    source_eeg_indices=picks,
                    excluded_channels=[all_ch[i] for i in sorted(ALBRECHT_EOG_INDICES)],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canonical_names, info
            finally:
                if local_mat.exists():
                    local_mat.unlink()
        shutil.rmtree(stream_dir, ignore_errors=True)

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'albrecht2019-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #2 StrokeMotorImagery (Pardo-Garcia et al., 2026; Zenodo 19599466)
# ---------------------------------------------------------------------------
def bootstrap_pardogarcia2026(cache: Path) -> tuple[dict, list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    rec_path = cache / 'zenodo-record.json'
    fetch(f'https://zenodo.org/api/records/{PARDO_RECORD_ID}', rec_path)
    record = json.loads(rec_path.read_text(encoding='utf-8'))

    inline_files: list[dict] = []
    eeg_binaries: list[dict] = []
    baseline_stems = {f'PAC{i:02d}' for i in range(1, 11)} | set(PARDO_CONTROL_MAP.keys())

    for f in record['files']:
        key_name = f['key']
        stem = key_name.rsplit('.', 1)[0] if '.' in key_name else key_name
        ext = key_name.rsplit('.', 1)[1].lower() if '.' in key_name else ''
        url = f['links']['self']
        ck = f['checksum']
        size = int(f['size'])
        item = dict(name=key_name, url=url, bytes=size, checksum=ck)
        if key_name.endswith('.txt') or (stem in baseline_stems and ext in {'vhdr', 'vmrk'}):
            target = (cache / key_name).resolve()
            fetch(url, target, ck, size)
            inline_files.append({**item, 'sha256': sha(target)})
        elif stem in baseline_stems and ext == 'eeg':
            eeg_binaries.append(item)

    inline_files.sort(key=lambda x: x['name'])
    eeg_binaries.sort(key=lambda x: x['name'])
    receipt = dict(
        dataset='PardoGarcia2026',
        source_record_id=PARDO_RECORD_ID,
        source_revision=f'Zenodo record {PARDO_RECORD_ID} (concept 19599465, v1)',
        eegdash_id='nm000309',
        licence='CC-BY-4.0',
        licence_evidence='Zenodo record 19599466 metadata license.id=cc-by-4.0',
        source_urls=[f'https://zenodo.org/records/{PARDO_RECORD_ID}', 'https://doi.org/10.5281/zenodo.19599466'],
        files=inline_files,
    )
    _write_receipt(cache, 'pardogarcia2026', receipt)
    return receipt, eeg_binaries


def pardogarcia2026_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    eeg_binaries = spec['_deferred_files']
    eeg_by_stem = {f['name'].rsplit('.', 1)[0]: f for f in eeg_binaries}

    participants: list[tuple[str, str, str, str]] = []
    for i in range(1, 11):
        stem = f'PAC{i:02d}'
        participants.append((stem, stem, 'chronic_mca_stroke', '1'))
    for raw_stem, canon_id in sorted(PARDO_CONTROL_MAP.items(), key=lambda kv: kv[1]):
        participants.append((canon_id, raw_stem, 'healthy_control', '2'))

    spec['metadata_updates']['excluded_participants'] = [
        dict(
            recording=f'PAC{i:02d}-POST',
            reason='Post-rehabilitation follow-up session excluded to preserve unconfounded pre-intervention baseline',
        )
        for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    subject_labels = {pid: lbl for pid, _, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split across baseline recordings (10 chronic MCA stroke, 8 healthy control): '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; post-rehabilitation recordings excluded.'
    )

    canonical_names: list[str] | None = None
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()

    for batch in batched(participants, 4):
        def _dl(entry):
            _, stem, _, _ = entry
            f = eeg_by_stem[stem]
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return target, {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = list(pool.map(_dl, batch))

        for (pid, stem, src_lbl, lbl), (local_eeg, file_rec) in zip(batch, fetched, strict=True):
            try:
                verified_stream.append(file_rec)
                vhdr_path = cache / f'{stem}.vhdr'
                vmrk_path = cache / f'{stem}.vmrk'
                raw_data, meta = decode_brainvision(vhdr_path)
                if round(meta['native_sampling_frequency_hz']) != spec['hz']:
                    raise ValueError(f"Unexpected sampling rate {meta['native_sampling_frequency_hz']} in {stem}")
                all_ch = [str(c).strip() for c in meta['native_channel_names']]
                if len(all_ch) != 63:
                    raise ValueError(f'Expected 63 channels in {stem}, got {len(all_ch)}')
                picks = [i for i in range(63) if i not in PARDO_EXCLUDE_INDICES]
                ch_names = [all_ch[i] for i in picks]
                if canonical_names is None:
                    canonical_names = ch_names
                elif [c.upper() for c in ch_names] != [c.upper() for c in canonical_names]:
                    raise ValueError(f'Channel order mismatch in {stem}: {ch_names}')

                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                markers = parse_brainvision_vmrk(vmrk_path)
                bounds = [m['sample'] for m in markers if m['type'].lower() == 'new segment']
                cues = []
                for m in markers:
                    desc_norm = re.sub(r'\s+', '', m['description'])
                    if m['type'].lower() == 'stimulus' and desc_norm in {'S1', 'S2'}:
                        cues.append((m['sample'], 'pinch' if desc_norm == 'S1' else 'closed_fist'))
                if not cues:
                    raise ValueError(f'No S1/S2 motor imagery cues in {stem}')
                cue_by_sample = dict(cues)
                wins, starts, dropped = extract_clean_windows(
                    scalp,
                    [s for s, _ in cues],
                    spec['points'],
                    bounds,
                    max_windows=40,
                    seen_hashes=seen_hashes,
                )
                if not wins:
                    raise ValueError(f'Zero clean windows in {stem}')
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='baseline',
                        recording_id=pid,
                        source_file=file_rec['name'],
                        original_recording_stem=stem,
                        motor_imagery_cue=cue_by_sample[s],
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=src_lbl,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    original_recording_stem=stem,
                    session_id='baseline',
                    source_file=file_rec['name'],
                    source_file_sha256=file_rec['sha256'],
                    native_sampling_frequency_hz=meta['native_sampling_frequency_hz'],
                    native_samples=meta['native_samples'],
                    native_channels=meta['native_channels'],
                    source_eeg_indices=picks,
                    excluded_channels=[all_ch[i] for i in sorted(PARDO_EXCLUDE_INDICES)],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canonical_names, info
            finally:
                if local_eeg.exists():
                    local_eeg.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'pardogarcia2026-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #3 AlcoholBandit (OpenNeuro ds004595 v1.0.0)
# ---------------------------------------------------------------------------
def ds004595_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    by_name = {f['name']: f for f in annex}

    part_rows = core.table(cache / 'participants.tsv')
    participants: list[tuple[str, str, str]] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        grp = r['GROUP'].strip()
        if grp not in {'AUD', 'CTRL'}:
            raise ValueError(f'Unexpected GROUP {grp!r} for {pid} in ds004595')
        participants.append((pid, grp, '1' if grp == 'AUD' else '2'))

    subject_labels = {pid: lbl for pid, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by GROUP (AUD vs CTRL) in sorted participant_id order: '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; zero shared participants.'
    )

    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()

    for batch in batched(sorted(participants), 6):
        batch_files = []
        for pid, _, _ in batch:
            stem = f'{pid}/eeg/{pid}_task-ThreeArmedBandit_eeg'
            batch_files.extend([by_name[stem + '.set'], by_name[stem + '.fdt']])

        def _dl(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=6) as pool:
            verified_batch = list(pool.map(_dl, batch_files))
        verified_by_name = {f['name']: f for f in verified_batch}
        verified_stream.extend(verified_batch)

        for pid, grp, lbl in batch:
            stem = f'{pid}/eeg/{pid}_task-ThreeArmedBandit_eeg'
            set_path = cache / (stem + '.set')
            fdt_path = cache / (stem + '.fdt')
            ch_tsv = cache / f'{pid}/eeg/{pid}_task-ThreeArmedBandit_channels.tsv'
            ev_tsv = cache / f'{pid}/eeg/{pid}_task-ThreeArmedBandit_events.tsv'
            try:
                raw_data, meta = decode_eeglab_set(set_path)
                if raw_data.ndim != 2 or round(meta['srate']) != spec['hz']:
                    raise ValueError(f'Unexpected ds004595 array/srate in {pid}')
                tsv_names = [r['name'].strip() for r in core.table(ch_tsv)]
                idx_by_name = {name: i for i, name in enumerate(tsv_names)}
                picks = [idx_by_name[ch] for ch in SHARED_DS003753]
                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')

                ev_rows = core.table(ev_tsv)
                bounds = []
                task_onsets = []
                for er in ev_rows:
                    val = er.get('value', '').strip().lower()
                    ttype = er.get('trial_type', '').strip()
                    onset_str = er.get('onset', '').strip()
                    if not onset_str or onset_str == 'n/a':
                        continue
                    samp = int(round(float(onset_str) * spec['hz']))
                    if val == 'boundary' or ttype.lower() == 'boundary':
                        bounds.append(samp)
                    elif ttype == 'Fixation' or ttype.startswith('CUE'):
                        task_onsets.append(samp)
                for ev in meta.get('events', []):
                    if str(ev.get('type', '')).strip().lower() == 'boundary':
                        bounds.append(int(round(float(ev['latency']) - 1.0)))

                start0 = min(task_onsets) if task_onsets else 0
                candidates = list(range(start0, scalp.shape[1] - spec['points'] + 1, spec['points']))
                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for ds004595 {pid}')
                fdt_rec = verified_by_name[stem + '.fdt']
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='1',
                        recording_id=pid,
                        source_file=fdt_rec['name'],
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=grp,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    session_id='1',
                    source_file=fdt_rec['name'],
                    source_file_sha256=fdt_rec['sha256'],
                    set_file=stem + '.set',
                    set_file_sha256=verified_by_name[stem + '.set']['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    native_samples=int(raw_data.shape[1]),
                    native_channels=int(raw_data.shape[0]),
                    source_eeg_indices=picks,
                    excluded_channels=[n for i, n in enumerate(tsv_names) if i not in set(picks)],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, list(SHARED_DS003753), info
            finally:
                for p in (set_path, fdt_path):
                    if p.exists():
                        p.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds004595-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #4 AdultSeizures (PhysioNet Siena Scalp EEG Database v1.0.0)
# ---------------------------------------------------------------------------
def bootstrap_siena(cache: Path) -> tuple[dict, list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    sums_path = cache / 'SHA256SUMS.txt'
    fetch(SIENA_BASE_URL + 'SHA256SUMS.txt', sums_path)

    inline_files: list[dict] = [
        dict(
            name='SHA256SUMS.txt',
            url=SIENA_BASE_URL + 'SHA256SUMS.txt',
            bytes=sums_path.stat().st_size,
            checksum='sha256:' + sha(sums_path),
            sha256=sha(sums_path),
        )
    ]
    edf_files: list[dict] = []
    for raw_line in sums_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, rel_path = line.split(None, 1)
        rel_path = rel_path.strip()
        url = SIENA_BASE_URL + urllib.parse.quote(rel_path, safe='/')
        ck = f'sha256:{digest}'
        if rel_path.lower().endswith('.edf'):
            edf_files.append(dict(name=rel_path, url=url, checksum=ck))
        elif rel_path in {'LICENSE.txt', 'subject_info.csv'} or 'Seizures-list-' in rel_path:
            target = (cache / rel_path).resolve()
            fetch(url, target, ck)
            inline_files.append(
                dict(name=rel_path, url=url, bytes=target.stat().st_size, checksum=ck, sha256=digest)
            )

    inline_files.sort(key=lambda x: x['name'])
    edf_files.sort(key=lambda x: x['name'])
    receipt = dict(
        dataset='SienaScalpEEG',
        source_record_id='siena-scalp-eeg/1.0.0',
        source_revision='PhysioNet siena-scalp-eeg v1.0.0 (SHA256SUMS-pinned)',
        licence='CC-BY-4.0',
        licence_evidence='PhysioNet siena-scalp-eeg v1.0.0 LICENSE.txt (Creative Commons Attribution 4.0)',
        source_urls=['https://physionet.org/content/siena-scalp-eeg/1.0.0/', 'https://doi.org/10.13026/5d4a-j060'],
        files=inline_files,
    )
    _write_receipt(cache, 'siena', receipt)
    return receipt, edf_files


def siena_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    edf_files = spec['_deferred_files']
    patients = sorted({f['name'].split('/')[0] for f in edf_files})
    train_subs, test_subs = whole_patient_split(patients, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Whole-patient 75/25 split in sorted patient_id order across 14 adult epilepsy patients: '
        f'{len(train_subs)} TRAIN ({", ".join(sorted(train_subs))}), '
        f'{len(test_subs)} TEST ({", ".join(sorted(test_subs))}); zero shared patients.'
    )

    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    edfs_by_patient: dict[str, list[dict]] = {}
    for f in edf_files:
        edfs_by_patient.setdefault(f['name'].split('/')[0], []).append(f)

    for pid in patients:
        p_edfs = sorted(edfs_by_patient[pid], key=lambda x: x['name'])
        sz_txt_path = next((cache / pid).glob('Seizures-list-*.txt'))
        sz_text = sz_txt_path.read_text(encoding='latin-1')

        # Stream EDFs for this patient in bounded batches of at most 3 EDFs at a time
        # Wait: parse_siena_seizure_list needs start_time and duration_s of each EDF for this patient!
        # Let's fetch 256-byte headers if >3 EDFs, or stream in batches after reading tiny headers,
        # OR download this patient's EDFs in sub-batches if needed. Let's check: max EDFs for one patient is
        # PN10 (6 EDFs, ~3.8 GB total) -> easily fits on C: (134 GB free), or we can fetch 2-3 EDFs at a time!
        # Even better: let's read headers via Range (1 KB) or stream in batches of 3 after quick Range header check,
        # OR download up to 3 EDFs at a time! Wait: PN10 has 6 EDFs (~600 MB each = 3.6 GB), totally tiny for 134 GB C:!
        # Let's still cap concurrent EDFs on disk to 3 by fetching headers first via HTTP Range (bytes=0-16383) so
        # parse_siena_seizure_list verifies ALL EDFs of the patient before streaming 3 full EDFs at a time!
        edf_headers: dict[str, dict] = {}
        for f in p_edfs:
            req = urllib.request.Request(f['url'], headers={'Range': 'bytes=0-16383', 'User-Agent': 'EEGBench'})
            import ssl, certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                chunk = resp.read()
            tmp_hdr = cache / '_tmp_hdr.bin'
            tmp_hdr.write_bytes(chunk)
            edf_headers[f['name']] = read_edf_start_and_duration(tmp_hdr)
            tmp_hdr.unlink(missing_ok=True)

        seizures_by_edf = parse_siena_seizure_list(sz_text, pid, edf_headers)

        for batch_edfs in batched(p_edfs, 3):
            def _dl(f: dict) -> tuple[Path, dict]:
                target = (cache / f['name']).resolve()
                fetch(f['url'], target, f['checksum'])
                sz_bytes = target.stat().st_size
                return target, {**f, 'bytes': sz_bytes, 'sha256': f['checksum'].split(':', 1)[1]}

            with ThreadPoolExecutor(max_workers=3) as pool:
                fetched = list(pool.map(_dl, batch_edfs))

            for local_edf, file_rec in fetched:
                try:
                    verified_stream.append(file_rec)
                    hdr_check = read_edf_start_and_duration(local_edf)
                    picks = select_siena_channels(hdr_check['labels'])
                    scalp, meta = decode_edf(local_edf, picks=picks)
                    if round(meta['srate']) != spec['hz']:
                        raise ValueError(f"Expected 512 Hz in {file_rec['name']}, got {meta['srate']}")
                    rec_stem = Path(file_rec['name']).stem
                    sz_intervals = seizures_by_edf[file_rec['name']]
                    wins, win_meta, dropped = select_siena_windows(
                        scalp,
                        meta['srate'],
                        spec['points'],
                        sz_intervals,
                        max_interictal=35,
                        buffer_sec=60.0,
                        min_ictal_overlap_sec=2.0,
                        seen_hashes=seen_hashes,
                    )
                    if not wins:
                        raise ValueError(f"Zero windows extracted from {file_rec['name']}")
                    rows = [
                        dict(
                            participant_id=pid,
                            session_id=rec_stem,
                            recording_id=rec_stem,
                            source_file=file_rec['name'],
                            event_index=idx,
                            start_sample=s,
                            end_sample=s + spec['points'],
                            start_time_seconds=s / spec['hz'],
                            end_time_seconds=(s + spec['points']) / spec['hz'],
                            sample_reference=SAMPLE_REF_CONT,
                            source_label='ictal_seizure' if cls == '1' else 'interictal_non_seizure',
                            label=cls,
                        )
                        for idx, (s, cls) in enumerate(win_meta)
                    ]
                    info = dict(
                        recording_id=rec_stem,
                        participant_id=pid,
                        session_id=rec_stem,
                        source_file=file_rec['name'],
                        source_file_sha256=file_rec['sha256'],
                        native_sampling_frequency_hz=meta['srate'],
                        native_samples=int(scalp.shape[1]),
                        native_channels=len(hdr_check['labels']),
                        source_eeg_indices=picks,
                        seizure_intervals_seconds=sz_intervals,
                        retained_windows=len(wins),
                        ictal_windows=sum(1 for _, c in win_meta if c == '1'),
                        interictal_windows=sum(1 for _, c in win_meta if c == '2'),
                        dropped_windows_count=len(dropped),
                    )
                    yield np.stack(wins, axis=0), rows, list(SIENA_CANONICAL_29), info
                finally:
                    if local_edf.exists():
                        local_edf.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'siena-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #5 PostCovidVigilance (OpenNeuro ds007605 v1.0.0)
# ---------------------------------------------------------------------------
def ds007605_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    eeg_by_sub = {
        f['name'].split('/')[0]: f
        for f in annex
        if f['name'].endswith('_task-pvt_eeg.eeg')
    }

    part_rows = core.table(cache / 'participants.tsv')
    participants: list[tuple[str, str, str]] = []
    excluded: list[dict] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        grp = r['group'].strip()
        if pid not in eeg_by_sub:
            excluded.append(dict(participant_id=pid, group=grp, reason='No task-pvt BrainVision EEG recording in release'))
            continue
        if grp not in {'withPCS', 'withoutPCS'}:
            raise ValueError(f'Unexpected group {grp!r} for {pid} in ds007605')
        participants.append((pid, grp, '1' if grp == 'withPCS' else '2'))

    spec['metadata_updates']['excluded_participants'] = excluded
    subject_labels = {pid: lbl for pid, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by group (withPCS vs withoutPCS) in sorted participant_id order: '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; zero shared participants.'
    )

    canonical_names: list[str] | None = None
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()

    for batch in batched(sorted(participants), 6):
        def _dl(entry):
            pid = entry[0]
            f = eeg_by_sub[pid]
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return target, {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=6) as pool:
            fetched = list(pool.map(_dl, batch))

        for (pid, grp, lbl), (local_eeg, file_rec) in zip(batch, fetched, strict=True):
            try:
                verified_stream.append(file_rec)
                vhdr_path = cache / f'{pid}/eeg/{pid}_task-pvt_eeg.vhdr'
                vmrk_path = cache / f'{pid}/eeg/{pid}_task-pvt_eeg.vmrk'
                ev_tsv = cache / f'{pid}/eeg/{pid}_task-pvt_events.tsv'
                raw_data, meta = decode_brainvision_safe(vhdr_path, local_eeg)
                hz_native = float(meta['native_sampling_frequency_hz'])
                if abs(hz_native - spec['hz']) > 1.0:
                    raise ValueError(f'Unexpected sampling rate {hz_native} in {pid}')

                all_ch = [str(c).strip() for c in meta['native_channel_names']]
                if canonical_names is None:
                    picks = [i for i, c in enumerate(all_ch) if c != 'BrainVision RDA_Gnd']
                    canonical_names = [all_ch[i] for i in picks]
                    if len(canonical_names) != 127:
                        raise ValueError(f'Expected 127 scalp channels in ds007605, got {len(canonical_names)}')
                else:
                    idx_map = {c: i for i, c in enumerate(all_ch)}
                    picks = [idx_map[c] for c in canonical_names]

                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                markers = parse_brainvision_vmrk(vmrk_path) if vmrk_path.is_file() else []
                bounds = [m['sample'] for m in markers if m['type'].lower() == 'new segment']

                ev_rows = core.table(ev_tsv)
                exp_starts = []
                trial_starts = []
                for er in ev_rows:
                    val = er.get('value', '').strip()
                    s_str = er.get('sample', '').strip()
                    samp = int(float(s_str)) if s_str and s_str != 'n/a' else int(round(float(er['onset']) * hz_native))
                    if val == 'Experiment start':
                        exp_starts.append(samp)
                    elif val in {'BL start', 'Trial start'}:
                        trial_starts.append(samp)
                start0 = min(exp_starts) if exp_starts else (min(trial_starts) if trial_starts else 0)
                candidates = list(range(start0, scalp.shape[1] - spec['points'] + 1, spec['points']))
                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=25, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for ds007605 {pid}')
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='1',
                        recording_id=pid,
                        source_file=file_rec['name'],
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=grp,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    session_id='1',
                    source_file=file_rec['name'],
                    source_file_sha256=file_rec['sha256'],
                    native_sampling_frequency_hz=hz_native,
                    nominal_sampling_frequency_hz=spec['hz'],
                    native_samples=meta['native_samples'],
                    native_channels=meta['native_channels'],
                    source_eeg_indices=picks,
                    excluded_channels=['BrainVision RDA_Gnd'],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canonical_names, info
            finally:
                if local_eeg.exists():
                    local_eeg.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds007605-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #6 SchizophreniaUltimatum (OpenNeuro ds004000 v1.0.0)
# ---------------------------------------------------------------------------
def ds004000_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    by_name = {f['name']: f for f in annex}

    part_rows = core.table(cache / 'participants.tsv')
    participants: list[tuple[str, str, str]] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        grp = r['group'].strip()
        if grp not in {'P', 'HC'}:
            raise ValueError(f'Unexpected group {grp!r} for {pid} in ds004000')
        participants.append((pid, grp, '1' if grp == 'P' else '2'))

    subject_labels = {pid: lbl for pid, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by group (P vs HC) in sorted participant_id order: '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; both proposer and responder runs stay in the participant split.'
    )

    # Canonical 128 BioSemi pin labels from sub-000 proposer .vhdr (indices 0..127; 128..131 are EXG1, EXG2, EXG7, EXG8)
    canonical_names: list[str] | None = None
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    picks = list(range(128))

    for batch in batched(sorted(participants), 4):
        batch_eegs = []
        for pid, _, _ in batch:
            for role in ('proposer', 'responder'):
                rel = f'{pid}/eeg/{pid}_task-{role}_run-1_eeg.eeg'
                if rel in by_name:
                    batch_eegs.append(by_name[rel])

        def _dl(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            verified_batch = list(pool.map(_dl, batch_eegs))
        verified_by_name = {f['name']: f for f in verified_batch}
        verified_stream.extend(verified_batch)

        for pid, grp, lbl in batch:
            for role in ('proposer', 'responder'):
                stem = f'{pid}/eeg/{pid}_task-{role}_run-1_eeg'
                if (stem + '.eeg') not in verified_by_name:
                    continue
                local_eeg = cache / (stem + '.eeg')
                vhdr_path = cache / (stem + '.vhdr')
                vmrk_path = cache / (stem + '.vmrk')
                try:
                    raw_data, meta = decode_brainvision_safe(vhdr_path, local_eeg)
                    if round(meta['native_sampling_frequency_hz']) != spec['hz']:
                        raise ValueError(f"Expected 2048 Hz in {stem}, got {meta['native_sampling_frequency_hz']}")
                    all_ch = [str(c).strip() for c in meta['native_channel_names']]
                    if len(all_ch) < 128 or any(c.upper().startswith('EXG') for c in all_ch[:128]):
                        raise ValueError(f'Unexpected BioSemi channel layout in {stem}: {all_ch[:5]}..{all_ch[-6:]}')
                    ch_128 = all_ch[:128]
                    if canonical_names is None:
                        canonical_names = ch_128
                    elif ch_128 != canonical_names:
                        # BioSemi ABCD physical pin order 0..127 verified in sub-*_electrodes.tsv
                        elec_tsv = cache / f'{pid}/eeg/{pid}_electrodes.tsv'
                        if not elec_tsv.is_file():
                            raise ValueError(f'Channel label difference without electrodes.tsv in {stem}')

                    scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                    markers = parse_brainvision_vmrk(vmrk_path)
                    seg_starts = sorted(m['sample'] for m in markers if m['type'].lower() == 'new segment')
                    if not seg_starts:
                        seg_starts = list(range(0, scalp.shape[1], 12288))
                    seg_ends = seg_starts[1:] + [scalp.shape[1]]
                    time0_samples = sorted(m['sample'] for m in markers if m['type'].lower() == 'time 0')
                    blink_intervals = [
                        (m['sample'], m['sample'] + max(1, m['duration']))
                        for m in markers
                        if m['description'].lower() == 'blink'
                    ]

                    blink_free_starts: list[int] = []
                    blink_Fallback_starts: list[int] = []
                    seg_bounds: list[int] = [b for b in seg_starts if b > 0]
                    for seg_s, seg_e in zip(seg_starts, seg_ends, strict=True):
                        in_seg_t0 = [t for t in time0_samples if seg_s <= t < seg_e]
                        offer_s = in_seg_t0[0] if in_seg_t0 else (seg_s + 2048)
                        offer_e = offer_s + spec['points']
                        if offer_e > seg_e:
                            continue
                        has_blink = any(max(offer_s, b0) < min(offer_e, b1) for b0, b1 in blink_intervals)
                        if not has_blink:
                            blink_free_starts.append(offer_s)
                        else:
                            blink_Fallback_starts.append(offer_s)

                    wins, starts, dropped = extract_clean_windows(
                        scalp, blink_free_starts, spec['points'], seg_bounds, max_windows=12, seen_hashes=seen_hashes
                    )
                    if len(wins) < 12 and blink_Fallback_starts:
                        need = 12 - len(wins)
                        extra_w, extra_s, extra_d = extract_clean_windows(
                            scalp, blink_Fallback_starts, spec['points'], seg_bounds, max_windows=need, seen_hashes=seen_hashes
                        )
                        combined = sorted(zip(starts + extra_s, wins + extra_w, strict=True), key=lambda pair: pair[0])
                        starts = [s for s, _ in combined]
                        wins = [w for _, w in combined]
                        dropped.extend(extra_d)

                    if not wins:
                        raise ValueError(f'Zero clean offer epochs in {stem}')
                    rec_id = f'{pid}_{role}'
                    eeg_rec = verified_by_name[stem + '.eeg']
                    rows = [
                        dict(
                            participant_id=pid,
                            session_id=role,
                            recording_id=rec_id,
                            task_role=role,
                            source_file=eeg_rec['name'],
                            event_index=idx,
                            start_sample=s,
                            end_sample=s + spec['points'],
                            start_time_seconds=s / spec['hz'],
                            end_time_seconds=(s + spec['points']) / spec['hz'],
                            sample_reference=SAMPLE_REF_SEG,
                            source_label=grp,
                            label=lbl,
                        )
                        for idx, s in enumerate(starts)
                    ]
                    info = dict(
                        recording_id=rec_id,
                        participant_id=pid,
                        session_id=role,
                        task_role=role,
                        source_file=eeg_rec['name'],
                        source_file_sha256=eeg_rec['sha256'],
                        native_sampling_frequency_hz=meta['native_sampling_frequency_hz'],
                        native_samples=meta['native_samples'],
                        native_channels=meta['native_channels'],
                        native_channel_names_first4=all_ch[:4],
                        excluded_channels=all_ch[128:],
                        source_eeg_indices=picks,
                        retained_windows=len(wins),
                        dropped_windows_count=len(dropped),
                    )
                    if 'stale_vhdr_datafile' in meta:
                        info['stale_vhdr_datafile'] = meta['stale_vhdr_datafile']
                    yield np.stack(wins, axis=0), rows, canonical_names, info
                finally:
                    if local_eeg.exists():
                        local_eeg.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds004000-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #7 SchizophreniaMotion (OpenNeuro ds004367 v1.0.2)
# ---------------------------------------------------------------------------
def ds004367_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    by_name = {f['name']: f for f in annex}

    part_rows = core.table(cache / 'participants.tsv')
    participants: list[tuple[str, str, str]] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        grp = r['Group'].strip()
        if grp not in {'Patient', 'Control'}:
            raise ValueError(f'Unexpected Group {grp!r} for {pid} in ds004367')
        participants.append((pid, grp, '1' if grp == 'Patient' else '2'))

    subject_labels = {pid: lbl for pid, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by Group (Patient vs Control) in sorted participant_id order: '
        f'{len(train_subs)} TRAIN, {len(test_subs)} TEST; zero shared participants.'
    )

    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()

    for batch in batched(sorted(participants), 4):
        batch_files = []
        for pid, _, _ in batch:
            stem = f'{pid}/eeg/{pid}_task-rdk_eeg'
            batch_files.extend([by_name[stem + '.set'], by_name[stem + '.fdt']])

        def _dl(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            verified_batch = list(pool.map(_dl, batch_files))
        verified_by_name = {f['name']: f for f in verified_batch}
        verified_stream.extend(verified_batch)

        for pid, grp, lbl in batch:
            stem = f'{pid}/eeg/{pid}_task-rdk_eeg'
            set_path = cache / (stem + '.set')
            fdt_path = cache / (stem + '.fdt')
            ch_tsv = cache / f'{pid}/eeg/{pid}_task-rdk_channels.tsv'
            ev_tsv = cache / f'{pid}/eeg/{pid}_task-rdk_events.tsv'
            try:
                raw_data, meta = decode_eeglab_set(set_path)
                if raw_data.ndim != 2 or round(meta['srate']) != spec['hz']:
                    raise ValueError(f'Unexpected ds004367 shape/srate in {pid}')
                tsv_names = [r['name'].strip() for r in core.table(ch_tsv)]
                picks, canon_names = select_ds004367_channels(tsv_names)
                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')

                bounds = [
                    int(round(float(ev['latency']) - 1.0))
                    for ev in meta.get('events', [])
                    if str(ev.get('type', '')).strip().lower() == 'boundary'
                ]
                ev_rows = core.table(ev_tsv)
                rdk_cues = [
                    int(float(er['sample']))
                    for er in ev_rows
                    if er.get('value', '').strip() == '3072' and er.get('sample', 'n/a').strip() != 'n/a'
                ]
                if rdk_cues:
                    # Enforce non-overlapping candidate starts in chronological order
                    candidates: list[int] = []
                    next_free = 0
                    for s in sorted(rdk_cues):
                        if s >= next_free:
                            candidates.append(s)
                            next_free = s + spec['points']
                else:
                    candidates = list(range(0, scalp.shape[1] - spec['points'] + 1, spec['points']))

                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for ds004367 {pid}')
                fdt_rec = verified_by_name[stem + '.fdt']
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='1',
                        recording_id=pid,
                        source_file=fdt_rec['name'],
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=grp,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    session_id='1',
                    source_file=fdt_rec['name'],
                    source_file_sha256=fdt_rec['sha256'],
                    set_file=stem + '.set',
                    set_file_sha256=verified_by_name[stem + '.set']['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    native_samples=int(raw_data.shape[1]),
                    native_channels=int(raw_data.shape[0]),
                    source_eeg_indices=picks,
                    excluded_channels=[
                        f'{i}:{n}' for i, n in enumerate(tsv_names) if i not in set(picks)
                    ],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canon_names, info
            finally:
                for p in (set_path, fdt_path):
                    if p.exists():
                        p.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds004367-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #8 ZosterNeuralgiaSleep (OpenNeuro ds007181 v1.0.1)
# ---------------------------------------------------------------------------
def ds007181_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    edf_by_sub = {
        f['name'].split('/')[0]: f
        for f in annex
        if f['name'].endswith('_task-sleep_acq-PSG_eeg.edf')
    }

    part_rows = core.table(cache / 'participants.tsv')
    participants: list[tuple[str, str, str]] = []
    excluded: list[dict] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        grp = r['group'].strip()
        if pid not in edf_by_sub:
            excluded.append(dict(participant_id=pid, group=grp, reason='MRI-only participant without PSG/EEG recording'))
            continue
        if pid in DS007181_TRUNCATED_UPSTREAM:
            excluded.append(dict(participant_id=pid, group=grp, reason=DS007181_TRUNCATED_UPSTREAM[pid]))
            continue
        if grp not in {'ZAN', 'HC'}:
            raise ValueError(f'Unexpected group {grp!r} for {pid} in ds007181')
        participants.append((pid, grp, '1' if grp == 'ZAN' else '2'))

    spec['metadata_updates']['excluded_participants'] = excluded
    subject_labels = {pid: lbl for pid, _, lbl in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by group (ZAN vs HC) across {len(participants)} byte-intact PSG '
        f'participants in sorted ID order: {len(train_subs)} TRAIN, {len(test_subs)} TEST; 7 MRI-only subjects without '
        f'PSG and 1 upstream byte-truncated EDF (sub-51) excluded.'
    )

    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    picks = list(range(19))

    for batch in batched(sorted(participants), 5):
        def _dl(entry):
            pid = entry[0]
            f = edf_by_sub[pid]
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return target, {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=5) as pool:
            fetched = list(pool.map(_dl, batch))

        for (pid, grp, lbl), (local_edf, file_rec) in zip(batch, fetched, strict=True):
            try:
                verified_stream.append(file_rec)
                scalp, meta = decode_edf(local_edf, picks=picks)
                if round(meta['srate']) != spec['hz']:
                    raise ValueError(f"Expected 256 Hz EEG in {pid}, got {meta['srate']}")
                ch_19 = [c.strip() for c in meta['ch_names'][:19]]
                if ch_19 != DS007181_CANONICAL_19:
                    raise ValueError(f'Unexpected PSG EEG channels in {pid}: {ch_19}')

                ev_tsv = cache / f'{pid}/eeg/{pid}_task-sleep_acq-PSG_events.tsv'
                stage_by_epoch30: dict[int, str] = {}
                if ev_tsv.is_file():
                    for er in core.table(ev_tsv):
                        onset_s = int(round(float(er['onset'])))
                        stage_by_epoch30[onset_s // 30] = er.get('stage', 'unknown').strip()

                lead_in_samples = 300 * spec['hz']
                start0 = lead_in_samples if scalp.shape[1] > lead_in_samples + 30 * spec['points'] else 0
                candidates = list(range(start0, scalp.shape[1] - spec['points'] + 1, spec['points']))
                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], boundary_samples=None, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean PSG windows for ds007181 {pid}')
                rows = [
                    dict(
                        participant_id=pid,
                        session_id='sleep',
                        recording_id=pid,
                        source_file=file_rec['name'],
                        sleep_stage=stage_by_epoch30.get(int((s / spec['hz']) // 30), 'unknown'),
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=grp,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=pid,
                    participant_id=pid,
                    session_id='sleep',
                    source_file=file_rec['name'],
                    source_file_sha256=file_rec['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    bids_sidecar_declared_hz=1024,
                    bids_sidecar_note='eeg.json declares 1024 Hz (RIP ECG channel rate); EDF header spr=256 / rec_dur=1.0 confirms 256 Hz on all 19 scalp EEG channels',
                    native_samples=int(scalp.shape[1]),
                    native_channels=len(meta['ch_names']),
                    source_eeg_indices=picks,
                    excluded_channels=meta['ch_names'][19:],
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, list(DS007181_CANONICAL_19), info
            finally:
                if local_edf.exists():
                    local_edf.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds007181-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #9 ParkinsonsReinforcement (OpenNeuro ds003506 v1.1.0)
# ---------------------------------------------------------------------------
def select_ds003506_session(row: dict) -> tuple[str, str, str]:
    """Select each PD patient's OFF-medication session and CTL ses-01 -> (ses, source_label, label)."""
    pid = row['participant_id'].strip()
    grp = row['Group'].strip()
    if grp == 'PD':
        m1 = row.get('sess1_Med', '').strip().upper()
        m2 = row.get('sess2_Med', '').strip().upper()
        if m1 == 'OFF' and m2 != 'OFF':
            return 'ses-01', 'PD_OFF', '1'
        if m2 == 'OFF' and m1 != 'OFF':
            return 'ses-02', 'PD_OFF', '1'
        raise ValueError(f'Ambiguous OFF-medication session for {pid}: sess1={m1!r}, sess2={m2!r}')
    if grp == 'CTL':
        return 'ses-01', 'CTL', '2'
    raise ValueError(f'Unexpected ds003506 Group {grp!r} for {pid}')


def ds003506_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    by_name = {f['name']: f for f in annex}

    part_rows = core.table(cache / 'participants.tsv')
    seen_orig_ids: set[str] = set()
    participants: list[tuple[str, str, str, str, dict]] = []
    excluded_sessions: list[dict] = []
    for r in part_rows:
        pid = r['participant_id'].strip()
        oid = r['Original_ID'].strip()
        if oid in seen_orig_ids:
            raise ValueError(f'Duplicate Original_ID {oid!r} in ds003506 participants.tsv')
        seen_orig_ids.add(oid)
        ses, src_lbl, lbl = select_ds003506_session(r)
        participants.append((pid, ses, src_lbl, lbl, r))
        if r['Group'].strip() == 'PD':
            other_ses = 'ses-02' if ses == 'ses-01' else 'ses-01'
            excluded_sessions.append(
                dict(
                    participant_id=pid,
                    original_id=oid,
                    excluded_session=other_ses,
                    reason='ON-medication visit excluded so every PD participant is evaluated strictly OFF dopaminergic medication',
                )
            )

    spec['metadata_updates']['excluded_participants'] = excluded_sessions
    spec['metadata_updates']['cohort_provenance_note'] = (
        'Same 56-participant UNM Parkinson\'s disease / matched-control cohort (Original_ID 801..829, '
        '890..914, 8010, 8060, 8070) as ParkinsonsRest (ds003490 / Cavanagh2018b), recorded in a '
        'separate active probabilistic reinforcement-learning task (task-ReinforcementLearning).'
    )
    subject_labels = {pid: lbl for pid, _, _, lbl, _ in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split by Group (28 PD OFF-medication vs 28 CTL) in sorted '
        f'participant_id order: {len(train_subs)} TRAIN, {len(test_subs)} TEST; zero shared participants.'
    )

    canonical_names: list[str] | None = None
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    picks = list(range(63))

    for batch in batched(sorted(participants, key=lambda item: item[0]), 6):
        batch_files = []
        for pid, ses, _, _, _ in batch:
            stem = f'{pid}/{ses}/eeg/{pid}_{ses}_task-ReinforcementLearning_eeg'
            batch_files.extend([by_name[stem + '.set'], by_name[stem + '.fdt']])

        def _dl(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=6) as pool:
            verified_batch = list(pool.map(_dl, batch_files))
        verified_by_name = {f['name']: f for f in verified_batch}
        verified_stream.extend(verified_batch)

        for pid, ses, src_lbl, lbl, p_row in batch:
            stem = f'{pid}/{ses}/eeg/{pid}_{ses}_task-ReinforcementLearning_eeg'
            set_path = cache / (stem + '.set')
            fdt_path = cache / (stem + '.fdt')
            ch_tsv = cache / f'{pid}/{ses}/eeg/{pid}_{ses}_task-ReinforcementLearning_channels.tsv'
            ev_tsv = cache / f'{pid}/{ses}/eeg/{pid}_{ses}_task-ReinforcementLearning_events.tsv'
            try:
                raw_data, meta = decode_eeglab_set(set_path)
                if raw_data.ndim != 2 or round(meta['srate']) != spec['hz']:
                    raise ValueError(f'Unexpected ds003506 shape/srate in {pid} {ses}')
                tsv_names = [r['name'].strip() for r in core.table(ch_tsv)]
                if len(tsv_names) != 67 or tsv_names[63:] != DS003506_NON_EEG_TAIL:
                    raise ValueError(f'Unexpected ds003506 channel layout in {pid} {ses}: {tsv_names}')
                ch_63 = tsv_names[:63]
                if canonical_names is None:
                    canonical_names = ch_63
                elif ch_63 != canonical_names:
                    raise ValueError(f'ds003506 scalp channel order mismatch in {pid} {ses}: {ch_63}')

                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                bounds: list[int] = []
                cue_samples: list[tuple[int, str]] = []
                for er in core.table(ev_tsv):
                    val = er.get('value', '').strip()
                    ttype = er.get('trial_type', '').strip()
                    onset_str = er.get('onset', '').strip()
                    if not onset_str or onset_str == 'n/a':
                        continue
                    samp = int(round(float(onset_str) * spec['hz']))
                    if val.lower() == 'boundary' or ttype.lower() == 'boundary':
                        if samp > 0:
                            bounds.append(samp)
                    elif val in {'S  1', 'S  2'}:
                        cue_samples.append((samp, 'choose' if val == 'S  1' else 'match'))
                for ev in meta.get('events', []):
                    if str(ev.get('type', '')).strip().lower() == 'boundary':
                        lat = int(round(float(ev['latency']) - 1.0))
                        if lat > 0:
                            bounds.append(lat)

                # Skip 6 practice cue trials documented in Step1_EEG_VV_Cues_With_Beh_PREPROC.m
                task_cues = (
                    cue_samples[DS003506_PRACTICE_CUES:]
                    if len(cue_samples) > DS003506_PRACTICE_CUES + 30
                    else cue_samples
                )
                candidates: list[int] = []
                cue_type_by_sample: dict[int, str] = {}
                next_free = 0
                for samp, ctype in sorted(task_cues, key=lambda pair: pair[0]):
                    if samp >= next_free:
                        candidates.append(samp)
                        cue_type_by_sample[samp] = ctype
                        next_free = samp + spec['points']

                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for ds003506 {pid} {ses}')
                fdt_rec = verified_by_name[stem + '.fdt']
                rec_id = f'{pid}_{ses}'
                rows = [
                    dict(
                        participant_id=pid,
                        original_id=p_row['Original_ID'].strip(),
                        session_id=ses,
                        medication_state='OFF' if lbl == '1' else 'n/a',
                        recording_id=rec_id,
                        source_file=fdt_rec['name'],
                        rl_trial_mode=cue_type_by_sample.get(s, 'task'),
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=src_lbl,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=rec_id,
                    participant_id=pid,
                    original_id=p_row['Original_ID'].strip(),
                    session_id=ses,
                    medication_state='OFF' if lbl == '1' else 'n/a',
                    source_file=fdt_rec['name'],
                    source_file_sha256=fdt_rec['sha256'],
                    set_file=stem + '.set',
                    set_file_sha256=verified_by_name[stem + '.set']['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    native_samples=int(raw_data.shape[1]),
                    native_channels=int(raw_data.shape[0]),
                    source_eeg_indices=picks,
                    excluded_channels=DS003506_NON_EEG_TAIL,
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canonical_names, info
            finally:
                for p in (set_path, fdt_path):
                    if p.exists():
                        p.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds003506-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# #10 AcuteTBIVisualMemory (OpenNeuro ds003523 v1.1.0)
# ---------------------------------------------------------------------------
def plan_ds003523_participants(part_rows: list[dict]) -> tuple[list[tuple[str, str, str, dict]], list[dict]]:
    """Apply Cavanagh s2_WM_Kill_Data.m clinical exclusions + duplicate Original_ID=3013 guard."""
    retained: list[tuple[str, str, str, dict]] = []
    excluded: list[dict] = []
    seen_orig: set[str] = set()
    seen_ursi: set[str] = set()

    for r in part_rows:
        pid = r['participant_id'].strip()
        oid = r['Original_ID'].strip()
        ursi = r['URSI'].strip()
        grp = str(r['Group']).strip()
        if grp not in {'0', '1'}:
            raise ValueError(f'Unexpected Group {grp!r} for {pid} in ds003523')
        if pid in DS003523_EXCLUSIONS:
            excluded.append(
                dict(
                    participant_id=pid,
                    original_id=oid,
                    ursi=ursi,
                    group=grp,
                    reason=DS003523_EXCLUSIONS[pid],
                )
            )
            continue
        if oid in seen_orig or ursi in seen_ursi:
            raise ValueError(f'Unresolved duplicate Original_ID/URSI in ds003523: {r}')
        # Cross-verify parity rule in s1_WM_Load_Data.m: even Original_ID = mTBI (Group 0), odd = Control (Group 1)
        expected_parity = 0 if grp == '0' else 1
        if int(oid) % 2 != expected_parity:
            raise ValueError(f'Original_ID parity / Group mismatch in ds003523: {r}')
        seen_orig.add(oid)
        seen_ursi.add(ursi)
        src_lbl = 'subacute_mild_tbi' if grp == '0' else 'healthy_control'
        lbl = '1' if grp == '0' else '2'
        retained.append((pid, src_lbl, lbl, r))

    return retained, excluded


def ds003523_records(cache: Path, spec: dict):
    receipt = json.loads((cache / 'download-receipt.json').read_text(encoding='utf-8'))
    annex = spec['_deferred_files']
    by_name = {f['name']: f for f in annex}

    part_rows = core.table(cache / 'participants.tsv')
    participants, excluded = plan_ds003523_participants(part_rows)
    spec['metadata_updates']['excluded_participants'] = excluded
    spec['metadata_updates']['cohort_provenance_note'] = (
        'Baseline Session 1 (3-14 days post-injury) visual working memory recordings (task-VisualWorkingMemory) '
        'from the UNM Center for Brain Recovery and Repair sub-acute mTBI and matched-control cohort '
        '(shared human cohort with TBIAuditoryOddball / ds003522, separate task recordings; longitudinal '
        'Sessions 2 and 3 excluded due to heavy follow-up attrition).'
    )

    subject_labels = {pid: lbl for pid, _, lbl, _ in participants}
    train_subs, test_subs = stratified_diagnostic_split(subject_labels, train_ratio=0.75)
    spec['_state']['train_subjects'] = train_subs
    spec['metadata_updates']['split_definition'] = (
        f'Stratified 75/25 whole-participant split across {len(participants)} clinically validated Session 1 '
        f'participants (54 sub-acute mTBI 3-14 days post-injury vs 32 healthy controls) in sorted participant_id '
        f'order: {len(train_subs)} TRAIN, {len(test_subs)} TEST; duplicate sub-037 (Original_ID=3013) and 4 '
        f's2_WM_Kill_Data.m clinical exclusions (sub-022, sub-064, sub-068, sub-084) removed.'
    )

    canonical_names: list[str] | None = None
    verified_stream: list[dict] = []
    seen_hashes: set[str] = set()
    picks = list(range(63))

    for batch in batched(sorted(participants, key=lambda item: item[0]), 6):
        batch_files = []
        for pid, _, _, _ in batch:
            stem = f'{pid}/ses-01/eeg/{pid}_ses-01_task-VisualWorkingMemory_eeg'
            batch_files.extend([by_name[stem + '.set'], by_name[stem + '.fdt']])

        def _dl(f: dict) -> dict:
            target = (cache / f['name']).resolve()
            fetch(f['url'], target, f['checksum'], f['bytes'])
            return {**f, 'sha256': sha(target)}

        with ThreadPoolExecutor(max_workers=6) as pool:
            verified_batch = list(pool.map(_dl, batch_files))
        verified_by_name = {f['name']: f for f in verified_batch}
        verified_stream.extend(verified_batch)

        for pid, src_lbl, lbl, p_row in batch:
            stem = f'{pid}/ses-01/eeg/{pid}_ses-01_task-VisualWorkingMemory_eeg'
            set_path = cache / (stem + '.set')
            fdt_path = cache / (stem + '.fdt')
            ch_tsv = cache / f'{pid}/ses-01/eeg/{pid}_ses-01_task-VisualWorkingMemory_channels.tsv'
            ev_tsv = cache / f'{pid}/ses-01/eeg/{pid}_ses-01_task-VisualWorkingMemory_events.tsv'
            try:
                raw_data, meta = decode_eeglab_set(set_path)
                if raw_data.ndim != 2 or round(meta['srate']) != spec['hz']:
                    raise ValueError(f'Unexpected ds003523 shape/srate in {pid}')
                tsv_names = [r['name'].strip() for r in core.table(ch_tsv)]
                if len(tsv_names) != 65 or tsv_names[63:] != DS003523_NON_EEG_TAIL:
                    raise ValueError(f'Unexpected ds003523 ses-01 channel layout in {pid}: {tsv_names}')
                ch_63 = tsv_names[:63]
                if canonical_names is None:
                    canonical_names = ch_63
                elif ch_63 != canonical_names:
                    raise ValueError(f'ds003523 scalp channel order mismatch in {pid}: {ch_63}')

                scalp = np.ascontiguousarray(raw_data[picks], dtype='<f8')
                bounds: list[int] = []
                stim_cues: list[tuple[int, str]] = []
                for er in core.table(ev_tsv):
                    val = er.get('value', '').strip()
                    ttype = er.get('trial_type', '').strip()
                    onset_str = er.get('onset', '').strip()
                    if not onset_str or onset_str == 'n/a':
                        continue
                    samp = int(round(float(onset_str) * spec['hz']))
                    if val.lower() == 'boundary' or ttype.lower() == 'boundary':
                        if samp > 0:
                            bounds.append(samp)
                    elif val in DS003523_STIM_MAP:
                        stim_cues.append((samp, DS003523_STIM_MAP[val]))
                for ev in meta.get('events', []):
                    if str(ev.get('type', '')).strip().lower() == 'boundary':
                        lat = int(round(float(ev['latency']) - 1.0))
                        if lat > 0:
                            bounds.append(lat)

                candidates: list[int] = []
                cond_by_sample: dict[int, str] = {}
                next_free = 0
                for samp, cond in sorted(stim_cues, key=lambda pair: pair[0]):
                    if samp >= next_free:
                        candidates.append(samp)
                        cond_by_sample[samp] = cond
                        next_free = samp + spec['points']

                wins, starts, dropped = extract_clean_windows(
                    scalp, candidates, spec['points'], bounds, max_windows=30, seen_hashes=seen_hashes
                )
                if not wins:
                    raise ValueError(f'Zero clean windows for ds003523 {pid}')
                fdt_rec = verified_by_name[stem + '.fdt']
                rec_id = f'{pid}_ses-01'
                rows = [
                    dict(
                        participant_id=pid,
                        original_id=p_row['Original_ID'].strip(),
                        ursi=p_row['URSI'].strip(),
                        session_id='ses-01',
                        recording_id=rec_id,
                        source_file=fdt_rec['name'],
                        vwm_condition=cond_by_sample.get(s, 'vwm_trial'),
                        event_index=idx,
                        start_sample=s,
                        end_sample=s + spec['points'],
                        start_time_seconds=s / spec['hz'],
                        end_time_seconds=(s + spec['points']) / spec['hz'],
                        sample_reference=SAMPLE_REF_CONT,
                        source_label=src_lbl,
                        label=lbl,
                    )
                    for idx, s in enumerate(starts)
                ]
                info = dict(
                    recording_id=rec_id,
                    participant_id=pid,
                    original_id=p_row['Original_ID'].strip(),
                    ursi=p_row['URSI'].strip(),
                    session_id='ses-01',
                    source_file=fdt_rec['name'],
                    source_file_sha256=fdt_rec['sha256'],
                    set_file=stem + '.set',
                    set_file_sha256=verified_by_name[stem + '.set']['sha256'],
                    native_sampling_frequency_hz=meta['srate'],
                    native_samples=int(raw_data.shape[1]),
                    native_channels=int(raw_data.shape[0]),
                    source_eeg_indices=picks,
                    excluded_channels=DS003523_NON_EEG_TAIL,
                    retained_windows=len(wins),
                    dropped_windows_count=len(dropped),
                )
                yield np.stack(wins, axis=0), rows, canonical_names, info
            finally:
                for p in (set_path, fdt_path):
                    if p.exists():
                        p.unlink()

    full_files = sorted(receipt['files'] + verified_stream, key=lambda item: item['name'])
    spec['_state']['all_source_files'] = full_files
    dump(ROOT / 'docs' / 'ds003523-diagnosis-source-lock.json', {**receipt, 'files': full_files})


# ---------------------------------------------------------------------------
# Spec Definitions & Orchestrator
# ---------------------------------------------------------------------------
PHASE1_META = {
    'albrecht2019': dict(
        hz=1000,
        points=4000,
        labels={'1': 'schizophrenia', '2': 'healthy_control'},
        source='Albrecht, Waltz, Cavanagh, Frank & Gold / University of Maryland & UNM / PRED+CT d004 (jalauer/Albrecht2019)',
        doi='10.1016/j.neuropsychologia.2018.12.018',
        paper='Albrecht et al. (2019), Increased conflict-induced slowing, but no differences in conflict-induced positive or negative prediction error learning in patients with schizophrenia, Neuropsychologia 123, 131-140',
        licence='PDDL-1.0',
        licence_url='https://opendatacommons.org/licenses/pddl/1-0/',
        original_source_url=f'https://huggingface.co/datasets/{ALBRECHT_REPO}/tree/{ALBRECHT_REV}',
        eegdash_id='predct_d004',
        doc_slug='albrecht2019',
        reader=albrecht2019_records,
        note=(
            'Binary clinical classification (schizophrenia vs healthy_control) during a modified Simon conflict '
            'reinforcement-learning task across 76 participants (45 SZ, 31 HC) at native 1,000 Hz (60 scalp EEG '
            'channels, up to 30 clean non-overlapping 4.0-second / 4,000-sample task windows starting at first '
            'stimulus trigger; EOG channels AFp9, AFp10, VEOGL, VEOGU excluded; s148 without DeID_Dems row excluded). '
            'No added filtering, baseline subtraction, or rereferencing.'
        ),
    ),
    'pardogarcia2026': dict(
        hz=1000,
        points=4000,
        labels={'1': 'chronic_mca_stroke', '2': 'healthy_control'},
        source='Pardo-Garcia et al. / Zenodo 19599466 (EEGDash nm000309)',
        doi='10.5281/zenodo.19599466',
        paper='Pardo-Garcia et al. (2026), EEG motor imagery dataset in chronic middle cerebral artery stroke patients and healthy controls, Zenodo 19599466',
        licence='CC-BY-4.0',
        licence_url='https://creativecommons.org/licenses/by/4.0/',
        original_source_url='https://zenodo.org/records/19599466',
        eegdash_id='nm000309',
        doc_slug='pardogarcia2026',
        reader=pardogarcia2026_records,
        note=(
            'Binary clinical classification (chronic_mca_stroke vs healthy_control) during upper-limb motor imagery '
            '(pinch S1 and closed fist S2) across 18 baseline participants (10 chronic MCA stroke PAC01-PAC10, '
            '8 healthy controls CTL01-CTL08) at native 1,000 Hz (58 scalp EEG channels, up to 40 clean non-overlapping '
            '4.0-second / 4,000-sample cue-locked windows; HEOGn, HEOGp, VEOGn, VEOGp, and A1 excluded; post-rehabilitation '
            'follow-up recordings excluded to avoid intervention confounding). Float64 microvolts.'
        ),
    ),
    'ds004595': dict(
        hz=500,
        points=2000,
        labels={'1': 'alcohol_use_group', '2': 'control'},
        source='Campbell & Cavanagh / University of New Mexico / OpenNeuro ds004595 v1.0.0',
        doi='10.18112/openneuro.ds004595.v1.0.0',
        paper='Campbell, Singh, Claus, Witkiewitz, Costa, Hogeveen & Cavanagh (2023), Electrophysiological markers of aberrant cue-specific exploration in hazardous drinkers, OpenNeuro ds004595 v1.0.0',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds004595/versions/1.0.0',
        eegdash_id='on004595',
        doc_slug='ds004595',
        reader=ds004595_records,
        note=(
            'Binary clinical group classification (alcohol_use_group AUD vs control CTRL) during a three-armed bandit '
            'reinforcement-learning task with alcohol cues across 53 participants (27 AUD, 26 CTRL) at native 500 Hz '
            '(59 common 10-10 scalp EEG channels shared across both cap configurations; mastoid, ocular, EKG, SCR, and '
            'AudioOutput channels excluded; up to 30 clean non-overlapping 4.0-second / 2,000-sample task windows). Float64 microvolts.'
        ),
    ),
    'siena': dict(
        hz=512,
        points=2048,
        labels={'1': 'ictal_seizure', '2': 'interictal_non_seizure'},
        source='Detti, Vatti & Zabalo Manrique de Lara / University of Siena & PhysioNet siena-scalp-eeg v1.0.0',
        doi='10.13026/5d4a-j060',
        paper='Detti, Vatti & Zabalo Manrique de Lara (2020), EEG Synchronization Analysis for Seizure Prediction: A Study on Data of Noninvasive Recordings, Processes 8(7), 846; PhysioNet DOI 10.13026/5d4a-j060',
        licence='CC-BY-4.0',
        licence_url='https://creativecommons.org/licenses/by/4.0/',
        original_source_url='https://physionet.org/content/siena-scalp-eeg/1.0.0/',
        eegdash_id='physionet_siena_1.0.0',
        doc_slug='siena',
        reader=siena_records,
        note=(
            'Binary clinical seizure classification (ictal_seizure vs interictal_non_seizure) across 14 adult epilepsy '
            'patients (41 EDF recordings, 47 annotated seizures) at native 512 Hz (29 canonical 10-10 scalp EEG channels '
            'aligned by exact label; EKG, SPO2, HR, MK, and auxiliary channels excluded). Non-overlapping 4.0-second '
            '(2,048-sample) windows: all clean ictal windows (>=2.0 s overlap with clinician seizure interval) and up to '
            '35 evenly spaced clean interictal windows (>=60.0 s buffer from every seizure) per EDF recording. Replaces '
            'duplicate slot #4 Cavanagh2019 (verified identical cohort to TBIAuditoryOddball / ds003522).'
        ),
    ),
    'ds007605': dict(
        hz=1000,
        points=4000,
        labels={'1': 'post_covid_syndrome', '2': 'recovered_without_pcs'},
        source='Neumann, Geritz, Keil, Maetzler & Welzel / UKSH Kiel COVIDOM / OpenNeuro ds007605 v1.0.0',
        doi='10.18112/openneuro.ds007605.v1.0.0',
        paper='Neumann, Geritz, Keil, Maetzler & Welzel (2026), EPOC: Neurophysiological correlates of Post-COVID-19 during psychomotor vigilance, OpenNeuro ds007605 v1.0.0',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds007605/versions/1.0.0',
        eegdash_id='on007605',
        doc_slug='ds007605',
        reader=ds007605_records,
        note=(
            'Binary clinical classification (post_covid_syndrome withPCS vs recovered_without_pcs withoutPCS) in the '
            'COVIDOM/NAPKON-POP cohort during the Psychomotor Vigilance Task (PVT) across 97 participants (60 withPCS, '
            '37 withoutPCS) at nominal 1,000 Hz (127 equidistant scalp EEG channels; BrainVision RDA_Gnd excluded; '
            'up to 25 consecutive clean non-overlapping 4.0-second / 4,000-sample windows starting at Experiment start). '
            'Native sample values retained in float64 microvolts without resampling.'
        ),
    ),
    'ds004000': dict(
        hz=2048,
        points=3072,
        labels={'1': 'schizophrenia', '2': 'healthy_control'},
        source='Padee, Missonnier, Prevot, Favre, Gothuey, Merlo & Richiardi / University of Fribourg / OpenNeuro ds004000 v1.0.0',
        doi='10.18112/openneuro.ds004000.v1.0.0',
        paper='Padee et al. (2022), Fribourg Ultimatum Game in Schizophrenia Study, OpenNeuro ds004000 v1.0.0',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds004000/versions/1.0.0',
        eegdash_id='on004000',
        doc_slug='ds004000',
        reader=ds004000_records,
        note=(
            'Binary clinical classification (schizophrenia P vs healthy_control HC) during proposer and responder runs '
            'of the Ultimatum Game across 43 participants (19 schizophrenia, 24 healthy control) at native 2,048 Hz '
            '(128 BioSemi ABCD scalp EEG channels; external EXG1, EXG2, EXG7, EXG8 excluded). Up to 12 clean 1.5-second '
            '(3,072-sample) offer-locked epochs [Time 0, Time 0 + 1.5 s) strictly inside each 6.0 s trial segment per '
            'run (up to 24 trials per participant), prioritising blink-free segments with zero resampling. Float64 microvolts.'
        ),
    ),
    'ds004367': dict(
        hz=1200,
        points=3600,
        labels={'1': 'schizophrenia', '2': 'healthy_control'},
        source='Rouy, Roger, Goueytes, Pereira, Roux & Faivre / OpenNeuro ds004367 v1.0.2',
        doi='10.18112/openneuro.ds004367.v1.0.2',
        paper='Rouy et al. (2022), Meta-RDK: Visual motion confidence and perceptual decision-making in schizophrenia, OpenNeuro ds004367 v1.0.2',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds004367/versions/1.0.2',
        eegdash_id='on004367',
        doc_slug='ds004367',
        reader=ds004367_records,
        note=(
            'Binary clinical classification (schizophrenia Patient vs healthy_control Control) during random-dot '
            'kinematogram (RDK) visual motion discrimination across 40 participants (17 schizophrenia, 23 control) '
            'at native 1,200 Hz (61 unambiguous common scalp EEG channels harmonised by exact label across all 5 cap '
            'layouts; A2, GSR1, GSR2, ECG1, ECG2, unlabelled 63, partial TP7, and ambiguous channel 17 excluded; '
            'up to 30 clean non-overlapping 3.0-second / 3,600-sample stimulus-locked windows). Float64 microvolts.'
        ),
    ),
    'ds007181': dict(
        hz=256,
        points=1024,
        labels={'1': 'zoster_associated_neuralgia', '2': 'healthy_control'},
        source='Li, Han, Bai, Zhang, Liu & He / Army Medical University / OpenNeuro ds007181 v1.0.1',
        doi='10.18112/openneuro.ds007181.v1.0.1',
        paper='Li, Han, Bai, Zhang, Liu & He (2026), Structural MRI, Resting-state fMRI, and PSG/EEG Dataset of Zoster-associated Neuralgia, OpenNeuro ds007181 v1.0.1',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds007181/versions/1.0.1',
        eegdash_id='on007181',
        doc_slug='ds007181',
        reader=ds007181_records,
        note=(
            'Binary clinical classification (zoster_associated_neuralgia ZAN vs healthy_control HC) from overnight '
            'polysomnography scalp EEG across 58 byte-intact PSG participants (26 ZAN, 32 HC; 7 MRI-only subjects '
            'without PSG and sub-51 with an upstream byte-truncated EDF excluded) '
            'at true native 256 Hz EDF EEG rate (19 linked-mastoid/midline scalp EEG derivations Fp1-M2..Pz; EOG, chin '
            'EMG, and 1,024 Hz RIP ECG excluded; first 30 clean non-overlapping 4.0-second / 1,024-sample windows '
            'following the initial 300 s settling period, annotated with AASM sleep stage). Float64 microvolts.'
        ),
    ),
    'ds003506': dict(
        hz=500,
        points=2000,
        labels={'1': 'parkinsons_off_medication', '2': 'healthy_control'},
        source='Brown, Pirio Richardson & Cavanagh / University of New Mexico / OpenNeuro ds003506 v1.1.0',
        doi='10.18112/openneuro.ds003506.v1.1.0',
        paper='Brown, Pirio Richardson & Cavanagh (2020), An electrophysiological signature of motivational salience in Parkinson\'s disease, Brain Research 1727:146541, DOI 10.1016/j.brainres.2019.146541; PMID 31704082',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds003506/versions/1.1.0',
        eegdash_id='on003506',
        doc_slug='ds003506',
        reader=ds003506_records,
        note=(
            'Binary clinical classification (parkinsons_off_medication vs healthy_control) during a probabilistic '
            'value-and-valence reinforcement-learning task across 56 participants (28 PD evaluated strictly OFF '
            'dopaminergic medication and 28 matched controls; ON-medication PD sessions excluded) at native 500 Hz '
            '(63 scalp EEG channels referenced online to CPz; VEOG and X/Y/Z hand-tremor accelerometer channels '
            'excluded; up to 30 clean non-overlapping 4.0-second / 2,000-sample cue-locked task windows after the '
            '6 initial practice trials). Shared UNM PD cohort with ParkinsonsRest (ds003490), separate active task '
            'recordings. Float64 microvolts.'
        ),
    ),
    'ds003523': dict(
        hz=500,
        points=2000,
        labels={'1': 'subacute_mild_tbi', '2': 'healthy_control'},
        source='Cavanagh / UNM Center for Brain Recovery and Repair / OpenNeuro ds003523 v1.1.0',
        doi='10.18112/openneuro.ds003523.v1.1.0',
        paper='Cavanagh (2021), Visual Working Memory in Sub-Acute Mild Traumatic Brain Injury, OpenNeuro ds003523 v1.1.0; cohort protocol in Cavanagh et al. (2019), Neuropsychologia 132:107125',
        licence='CC0-1.0',
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        original_source_url='https://openneuro.org/datasets/ds003523/versions/1.1.0',
        eegdash_id='on003523',
        doc_slug='ds003523',
        reader=ds003523_records,
        note=(
            'Binary clinical classification (subacute_mild_tbi 3-14 days post-injury vs healthy_control) during '
            'baseline Session 1 change-detection visual working memory (set sizes 3, 3+2 distractors, and 5) across '
            '86 strictly disjoint, clinically validated participants (54 sub-acute mTBI, 32 healthy controls; exact '
            'BIDS duplicate sub-037 [Original_ID=3013] and 4 author s2_WM_Kill_Data.m clinical exclusions [sub-022, '
            'sub-064, sub-068, sub-084] removed; longitudinal Sessions 2 and 3 excluded) at native 500 Hz (63 scalp '
            'EEG channels referenced online to CPz; VEOG and EKG excluded; up to 30 clean non-overlapping 4.0-second '
            '/ 2,000-sample array-locked task windows). Shared UNM mTBI cohort with TBIAuditoryOddball (ds003522), '
            'separate task recordings. Float64 microvolts.'
        ),
    ),
}


def bootstrap_dataset(key: str, cache: Path) -> tuple[dict, list[dict]]:
    if key == 'albrecht2019':
        return bootstrap_albrecht2019(cache)
    if key == 'pardogarcia2026':
        return bootstrap_pardogarcia2026(cache)
    if key == 'siena':
        return bootstrap_siena(cache)
    eager = ('.vhdr', '.vmrk') if key in {'ds007605', 'ds004000'} else ()
    return bootstrap_openneuro_metadata(key, cache, annex_eager_exts=eager)


def build_spec(key: str, cache_root: Path, deferred_files: list[dict]) -> tuple[dict, Path]:
    meta = PHASE1_META[key]
    name = DATASET_NAMES[key]
    sub_cache = f'DiagnosticTo30/{key}'
    cache = Path(cache_root) / sub_cache
    doc_slug = meta['doc_slug']

    cache_ev = [
        p for p in [
            'dataset_description.json',
            'participants.tsv',
            'participants.json',
            'README',
            'README.md',
            'CHANGES',
            'DeID_Dems.csv',
            'DataDictionary.txt',
            'Pre-processing_Pseudo.txt',
            'MMN_ElectrodeCoords_MA2.ced',
            'LICENSE.txt',
            'SHA256SUMS.txt',
            'subject_info.csv',
            'bdf_IMAGEN.txt',
            'nota post 02 y post 03.txt',
            'original-source-lock.json',
        ]
        if (cache / p).is_file()
    ]
    evidence_files = ['diagnostic-to-30-plan.md', f'{doc_slug}-diagnosis-source-lock.json']
    if key == 'siena':
        evidence_files.append('cavanagh2019-diagnosis-excluded.json')

    state: dict = {}
    spec = dict(
        name=name,
        source=meta['source'],
        cache=sub_cache,
        hz=meta['hz'],
        points=meta['points'],
        classes=len(meta['labels']),
        train_subjects=0,
        licence=meta['licence'],
        licence_url=meta['licence_url'],
        moabb_revision=None,
        doi=meta['doi'],
        paper=meta['paper'],
        labels=meta['labels'],
        splitter=lambda people, chunks, st=state: st['train_subjects'],
        evidence_files=evidence_files,
        cache_evidence_files=cache_ev,
        conversion_files=['prepare_diagnostic_to_30.py', 'prepare_cognitive_group_a.py', 'prepare_cognitive_group_b.py'],
        intermediate_directory='classification-intermediate-full',
        note=sanitise_text(meta['note']),
        _deferred_files=deferred_files,
        _state=state,
        metadata_updates=dict(
            domain='Med',
            eegdash_id=meta['eegdash_id'],
            original_source_url=meta['original_source_url'],
            paper_reference=sanitise_text(meta['paper']),
            excluded_participants=[],
        ),
    )
    return spec, cache


def prepare_one(
    key: str,
    cache_root: Path = Path('C:/Data/EEGSources'),
    primary_output: Path = Path('D:/Data/EEG'),
    overflow_output: Path | None = Path('C:/Data/EEG'),
    min_free_gb: float = 5.0,
    purge_cache: bool = True,
) -> dict:
    cache_root = Path(cache_root)
    primary_output = Path(primary_output)
    name = DATASET_NAMES[key]
    primary_final = primary_output / name
    est_gb = ESTIMATED_EXPORT_GB[key]

    c_before = disk_free_gb(cache_root)
    d_before = disk_free_gb(primary_output)
    dest_root, use_junction = resolve_storage_destination(
        name, est_gb, primary_output, overflow_output, min_free_gb=min_free_gb
    )
    if not use_junction and overflow_output is not None and (d_before - est_gb) < min_free_gb:
        Path(overflow_output).mkdir(parents=True, exist_ok=True)
        dest_root, use_junction = Path(overflow_output), True
    print(
        f'[{name}] Starting Phase 1 conversion ({key}): '
        f'D: free={d_before:.2f} GB, C: free={c_before:.2f} GB, '
        f'target={dest_root / name} (overflow={use_junction})',
        flush=True,
    )

    cache = cache_root / 'DiagnosticTo30' / key
    _, deferred = bootstrap_dataset(key, cache)
    spec, cache = build_spec(key, cache_root, deferred)
    core.SPECS[key] = spec
    core.READERS[key] = PHASE1_META[key]['reader']

    result = core.prepare(key, cache_root, dest_root)
    folder = Path(result['path'])
    m = json.loads((folder / f'{name}_metadata.json').read_text(encoding='utf-8'))

    first_picks = m['recordings'][0]['source_eeg_indices']
    varying_picks = any(rec['source_eeg_indices'] != first_picks for rec in m['recordings'])
    for sensor in m['sensors']:
        pre_idx = sensor['pre_flat_removal_array_index']
        sensor['native_channel_index'] = first_picks[pre_idx]
        sensor['identity_evidence'] = (
            'Aligned by verified canonical scalp EEG channel name per recording; see recordings[*].source_eeg_indices'
            if varying_picks
            else 'Verified uniform native scalp EEG channel index; non-scalp auxiliary channels excluded'
        )

    full_files = spec['_state'].get('all_source_files')
    manifest_updates = dict(
        label_mapping=spec['labels'],
        split_definition=m['split_definition'],
        excluded_participants=m.get('excluded_participants', []),
    )
    if full_files is not None:
        manifest_updates['source_files'] = full_files
        lock_doc = ROOT / 'docs' / f"{PHASE1_META[key]['doc_slug']}-diagnosis-source-lock.json"
        if lock_doc.is_file():
            for alias in ('download-receipt.json', 'source-record.json', 'original-source-lock.json'):
                shutil.copy2(lock_doc, folder / 'source_metadata' / alias)

    result = finish(
        result,
        m,
        spec['evidence_files'],
        spec['conversion_files'],
        manifest_updates,
    )
    # Refresh source_metadata hashes in manifest after syncing full receipt aliases
    manifest_path = folder / f'{name}_manifest.json'
    manifest_obj = json.loads(manifest_path.read_text(encoding='utf-8'))
    for alias in ('download-receipt.json', 'source-record.json', 'original-source-lock.json'):
        ap = folder / 'source_metadata' / alias
        if ap.is_file():
            manifest_obj['artifacts'][ap.relative_to(folder).as_posix()] = dict(
                bytes=ap.stat().st_size, sha256=sha(ap)
            )
    dump(manifest_path, manifest_obj)

    verify_final(result)
    if use_junction:
        result['physical_path'] = str(dest_root / name)
        if create_junction_if_needed(primary_final, dest_root / name):
            result['junction_path'] = str(primary_final)
            result['path'] = str(primary_final)
            verify_final(result)

    if purge_cache and cache.is_dir():
        shutil.rmtree(cache)
        result['cache_purged'] = True

    result['storage'] = dict(
        physical_root=str(dest_root),
        used_overflow=use_junction,
        d_free_gb_before=round(d_before, 3),
        d_free_gb_after=round(disk_free_gb(primary_output), 3),
        c_free_gb_before=round(c_before, 3),
        c_free_gb_after=round(disk_free_gb(cache_root), 3),
    )
    doc_slug = PHASE1_META[key]['doc_slug']
    dump(ROOT / 'docs' / f'{doc_slug}-diagnosis-conversion-result.json', result)
    print(
        f'[{name}] Verified & complete: {result["train_cases"]} TRAIN / {result["test_cases"]} TEST, '
        f'{result["channels"]} ch x {result["timepoints"]} pts @ {result["frequency"]} Hz, '
        f'{result["size_GB"]:.2f} GB | D: free={result["storage"]["d_free_gb_after"]:.2f} GB, '
        f'C: free={result["storage"]["c_free_gb_after"]:.2f} GB',
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=PHASE1_ORDER, default=PHASE1_ORDER)
    parser.add_argument('--cache-root', type=Path, default=Path('C:/Data/EEGSources'))
    parser.add_argument('--output', type=Path, default=Path('D:/Data/EEG'))
    parser.add_argument('--overflow-output', type=Path, default=Path('C:/Data/EEG'))
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--keep-cache', action='store_true')
    args = parser.parse_args()

    for key in args.datasets:
        name = DATASET_NAMES[key]
        primary_final = args.output / name
        overflow_final = args.overflow_output / name if args.overflow_output else None
        if primary_final.exists():
            print(f'Skipping already-exported {name} at {primary_final}; verifying manifest...', flush=True)
            verify_final({'dataset': name, 'path': str(primary_final)})
            continue
        if overflow_final is not None and overflow_final.exists():
            print(f'Found {name} on overflow {overflow_final}; ensuring junction & verifying...', flush=True)
            create_junction_if_needed(primary_final, overflow_final)
            verify_final({'dataset': name, 'path': str(primary_final if primary_final.exists() else overflow_final)})
            continue
        prepare_one(
            key,
            cache_root=args.cache_root,
            primary_output=args.output,
            overflow_output=args.overflow_output,
            min_free_gb=args.min_free_gb,
            purge_cache=not args.keep_cache,
        )


if __name__ == '__main__':
    main()
