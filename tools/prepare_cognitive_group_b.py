"""Prepare Cognitive Group B (+15 scalp-EEG problems -> 30 Cognitive) at native Hz."""
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
import subprocess
import zipfile

import numpy as np

from tools import prepare_open_moabb as core
from tools.download_eegdash_diagnosis import blob_sha
from tools.prepare_cognitive_group_a import (
    decode_brainvision,
    decode_eeglab_set,
    harmonise_sampling_rate,
    verify_final,
    whole_subject_split,
)
from tools.prepare_erpcore_batch2 import ERPCORE_SPECS, convert_erpcore_component
from tools.prepare_moabb import dump, sha
from tools.prepare_small_moabb import fetch
from tools.prepare_wang import digest_case
from tools.update_dataset_sidecars import finish

ROOT = Path(__file__).resolve().parents[1]

PYTHON_CODE_BINS = {
    110: ('1', 'well_formed'),
    115: ('1', 'well_formed'),
    130: ('2', 'semantically_implausible'),
    135: ('2', 'semantically_implausible'),
    220: ('3', 'structurally_invalid'),
    225: ('3', 'structurally_invalid'),
    240: ('4', 'doubly_anomalous'),
    245: ('4', 'doubly_anomalous'),
}
FUZZY_LABELS = {
    'target': ('1', 'semantic_target'),
    'nontarget': ('2', 'semantic_nontarget'),
}
INTERNAL_ATTN_LABELS = {
    'concentrate': ('1', 'concentrate'),
    'relax': ('2', 'relax'),
}
SYMMETRY_LABELS = {
    '1': ('1', 'regular'),
    '3': ('2', 'random'),
}
MORAL_LABELS = {
    '3': ('1', 'accidental_congruent'),
    '4': ('2', 'accidental_incongruent'),
    '5': ('3', 'intentional_congruent'),
    '6': ('4', 'intentional_incongruent'),
}
PROB_FB_LABELS = {
    'FB: WIN PUPPY': ('1', 'win_puppy'),
    'FB: WIN COW': ('2', 'win_cow'),
    'FB: LOSE PUPPY': ('3', 'lose_puppy'),
    'FB: LOSE COW': ('4', 'lose_cow'),
}
SENTENCE_LABELS = {
    '14': ('1', 'control'),
    '54': ('1', 'control'),
    '124': ('1', 'control'),
    '24': ('2', 'syntactic_violation'),
    '64': ('2', 'syntactic_violation'),
    '34': ('3', 'semantic_violation'),
    '74': ('3', 'semantic_violation'),
    '44': ('4', 'double_violation'),
    '84': ('4', 'double_violation'),
    '114': ('5', 'word_order_anomaly'),
}
NARRATIVE_LABELS = {
    'S  7': ('1', 'narrative_win'),
    'S  8': ('2', 'narrative_loss'),
    'S 17': ('3', 'nonnarrative_win'),
    'S 18': ('4', 'nonnarrative_loss'),
}
FLANKER_LABELS = {
    '11': ('1', 'word_none'),
    '12': ('2', 'word_same'),
    '13': ('3', 'word_different'),
    '21': ('4', 'pseudoword_none'),
    '22': ('5', 'pseudoword_same'),
    '23': ('6', 'pseudoword_different'),
}
BANDIT_LABELS = {
    'S  5': ('1', 'feedback_win'),
    'S  6': ('2', 'feedback_loss'),
}
MUSIC_QUADRANTS = {
    'HPHE': ('1', 'HPHE'),
    'HPLE': ('2', 'HPLE'),
    'LPHE': ('3', 'LPHE'),
    'LPLE': ('4', 'LPLE'),
}
VERBAL_WM_LABELS = {
    'delay period: 5R': ('1', 'retention_5'),
    'delay period: 5M': ('2', 'manipulation_5'),
    'delay period: 6R': ('3', 'retention_6'),
    'delay period: 6M': ('4', 'manipulation_6'),
    'delay period: 7R': ('5', 'retention_7'),
    'delay period: 7M': ('6', 'manipulation_7'),
}

SHARED_DS003753 = [
    'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
    'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
    'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
    'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3',
    'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4',
    'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8',
]
SHARED_DS003458 = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
    'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
    'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz',
    'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
    'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
    'F2', 'AF4', 'AF8',
]

OPENNEURO_RELEASES = {
    'python_code_reading': dict(
        dataset='ds004771',
        ref='1.0.0',
        commit='472d2c04f91e65f103927aa123e334eed93d148e',
        licence='CC0-1.0',
        name='PythonCodeReading',
        est_gb=1.2,
    ),
    'fuzzy_semantics': dict(
        dataset='ds005383',
        ref='1.0.0',
        commit='ad78f3db430e636595b1b1c08417492f3067a25e',
        licence='CC0-1.0',
        name='FuzzySemantics',
        est_gb=0.9,
    ),
    'internal_attention': dict(
        dataset='ds002691',
        ref='1.1.0',
        commit='bde891f03ab06f65b1eb8f00d3814cf63255e77c',
        licence='CC0-1.0',
        name='InternalAttention',
        est_gb=0.7,
    ),
    'visual_symmetry': dict(
        dataset='ds004347',
        ref='1.0.0',
        commit='d11ed957532a829c69727763a6db00162ea37101',
        licence='CC0-1.0',
        name='VisualSymmetry',
        est_gb=1.4,
    ),
    'moral_conflict': dict(
        dataset='ds004860',
        ref='1.0.0',
        commit='0a0919761bc97af2875bfb312f9ea5c70499367a',
        licence='CC0-1.0',
        name='MoralConflict',
        est_gb=1.3,
    ),
    'probabilistic_feedback': dict(
        dataset='ds003753',
        ref='1.1.0',
        commit='c261d49df8cff6bb6f0ab1966508fa303a9c33e2',
        licence='CC0-1.0',
        name='ProbabilisticFeedback',
        est_gb=1.5,
    ),
    'sentence_violations': dict(
        dataset='ds008711',
        ref='1.0.0',
        commit='4f93a25d93de4855645ff2450965e4211b0621a4',
        licence='CC0-1.0',
        name='SentenceViolations',
        est_gb=1.9,
    ),
    'narrative_reward': dict(
        dataset='ds007968',
        ref='1.0.0',
        commit='c80c884d04011fb42a0672cfd247dafbaed6d378',
        licence='CC0-1.0',
        name='NarrativeReward',
        est_gb=2.1,
    ),
    'flanker_conflict': dict(
        dataset='ds005866',
        ref='1.0.1',
        commit='5b5be67a95dd0f38f802405eb56840d0ef4f2223',
        licence='CC0-1.0',
        name='FlankerConflict',
        est_gb=2.6,
    ),
    'bandit_gambling': dict(
        dataset='ds003458',
        ref='1.1.0',
        commit='417ba109d89c92e58252a18f36f2dff4351b2ab2',
        licence='CC0-1.0',
        name='BanditGambling',
        est_gb=2.7,
    ),
    'affective_music': dict(
        dataset='ds002721',
        ref='1.0.3',
        commit='3972f19d2846be4f283fe21c7fe39a1a11efda99',
        licence='CC0-1.0',
        name='AffectiveMusic',
        est_gb=3.0,
    ),
    'verbal_working_memory': dict(
        dataset='ds003655',
        ref='1.0.2',
        commit='4807ef6e9930eddb6baa531be18118035f3e827c',
        licence='CC0-1.0',
        name='VerbalWorkingMemory',
        est_gb=4.0,
    ),
}

ERPCORE_EST_GB = {
    'face_stimuli': 4.8,
    'auditory_deviance': 6.0,
    'spatial_attention': 4.8,
}

GROUP_B_ORDER = [
    'python_code_reading',
    'fuzzy_semantics',
    'internal_attention',
    'visual_symmetry',
    'moral_conflict',
    'probabilistic_feedback',
    'sentence_violations',
    'narrative_reward',
    'flanker_conflict',
    'bandit_gambling',
    'affective_music',
    'verbal_working_memory',
    'face_stimuli',
    'auditory_deviance',
    'spatial_attention',
]

DATASET_NAMES = {
    **{k: v['name'] for k, v in OPENNEURO_RELEASES.items()},
    **{k: v['name'] for k, v in ERPCORE_SPECS.items()},
}


def _unit_scale_to_uv(unit: str) -> float:
    token = unit.strip().lower().replace('µ', 'u')
    if token in {'uv', 'microv', 'microvolt', 'microvolts', ''}:
        return 1.0
    if token in {'mv', 'millivolt', 'millivolts'}:
        return 1e3
    if token in {'v', 'volt', 'volts'}:
        return 1e6
    raise ValueError(f'Unreviewed physical unit {unit!r}')


def decode_edf(edf_path: Path, picks: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """Decode continuous 16-bit EDF in float64 microvolts."""
    raw = Path(edf_path).read_bytes()
    if len(raw) < 256:
        raise ValueError(f'Truncated EDF header in {edf_path}')
    header_bytes = int(raw[184:192].decode('ascii').strip())
    n_records = int(raw[236:244].decode('ascii').strip())
    rec_dur = float(raw[244:252].decode('ascii').strip())
    ns = int(raw[252:256].decode('ascii').strip())
    if header_bytes != 256 * (ns + 1) or len(raw) < header_bytes:
        raise ValueError(f'Invalid EDF header size in {edf_path}')

    pos = 256
    def _fields(width: int) -> list[str]:
        nonlocal pos
        out = [raw[pos + i * width : pos + (i + 1) * width].decode('latin-1').strip() for i in range(ns)]
        pos += ns * width
        return out

    labels = _fields(16)
    _fields(80)  # transducer
    phys_dims = _fields(8)
    phys_mins = [float(x) for x in _fields(8)]
    phys_maxs = [float(x) for x in _fields(8)]
    dig_mins = [float(x) for x in _fields(8)]
    dig_maxs = [float(x) for x in _fields(8)]
    _fields(80)  # prefiltering
    spr = [int(x) for x in _fields(8)]

    total_spr = sum(spr)
    body = raw[header_bytes:]
    expected_body = n_records * total_spr * 2
    if len(body) != expected_body:
        raise ValueError(f'EDF byte mismatch in {edf_path}: got {len(body)}, expected {expected_body}')

    sel = list(range(ns)) if picks is None else list(picks)
    spr_sel = {spr[i] for i in sel}
    if len(spr_sel) != 1 or rec_dur <= 0:
        raise ValueError(f'Inconsistent EDF sampling rate across selected channels in {edf_path}')
    srate = spr[sel[0]] / rec_dur

    rec_arr = np.frombuffer(body, dtype='<i2').reshape(n_records, total_spr)
    offsets = np.cumsum([0] + spr)
    n_samples = n_records * spr[sel[0]]
    out = np.empty((len(sel), n_samples), dtype='<f8')
    for out_i, ch_i in enumerate(sel):
        dig = rec_arr[:, offsets[ch_i] : offsets[ch_i + 1]].reshape(-1).astype('<f8')
        denom = dig_maxs[ch_i] - dig_mins[ch_i]
        if denom == 0:
            raise ValueError(f'Zero digital range in {edf_path} ch {labels[ch_i]}')
        gain = (phys_maxs[ch_i] - phys_mins[ch_i]) / denom
        phys = (dig - dig_mins[ch_i]) * gain + phys_mins[ch_i]
        out[out_i] = phys * _unit_scale_to_uv(phys_dims[ch_i])

    return out, dict(
        srate=srate,
        n_records=n_records,
        rec_dur=rec_dur,
        ch_names=labels,
        phys_dims=phys_dims,
        samples_per_record=spr,
    )


def decode_bdf(bdf_path: Path, picks: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """Decode continuous BioSemi 24-bit BDF in float64 microvolts."""
    raw = Path(bdf_path).read_bytes()
    if len(raw) < 256 or raw[0] != 0xFF or raw[1:8] != b'BIOSEMI':
        raise ValueError(f'Invalid BioSemi BDF magic in {bdf_path}')
    header_bytes = int(raw[184:192].decode('ascii').strip())
    n_records = int(raw[236:244].decode('ascii').strip())
    rec_dur = float(raw[244:252].decode('ascii').strip())
    ns = int(raw[252:256].decode('ascii').strip())
    if header_bytes != 256 * (ns + 1) or len(raw) < header_bytes:
        raise ValueError(f'Invalid BDF header size in {bdf_path}')

    pos = 256
    def _fields(width: int) -> list[str]:
        nonlocal pos
        out = [raw[pos + i * width : pos + (i + 1) * width].decode('latin-1').strip() for i in range(ns)]
        pos += ns * width
        return out

    labels = _fields(16)
    _fields(80)
    phys_dims = _fields(8)
    phys_mins = [float(x) for x in _fields(8)]
    phys_maxs = [float(x) for x in _fields(8)]
    dig_mins = [float(x) for x in _fields(8)]
    dig_maxs = [float(x) for x in _fields(8)]
    _fields(80)
    spr = [int(x) for x in _fields(8)]

    total_spr = sum(spr)
    body = raw[header_bytes:]
    expected_body = n_records * total_spr * 3
    if len(body) != expected_body:
        raise ValueError(f'BDF byte mismatch in {bdf_path}: got {len(body)}, expected {expected_body}')

    sel = list(range(ns)) if picks is None else list(picks)
    spr_sel = {spr[i] for i in sel}
    if len(spr_sel) != 1 or rec_dur <= 0:
        raise ValueError(f'Inconsistent BDF sampling rate across selected channels in {bdf_path}')
    srate = spr[sel[0]] / rec_dur

    triplets = np.frombuffer(body, dtype=np.uint8).reshape(n_records, total_spr, 3)
    offsets = np.cumsum([0] + spr)
    n_samples = n_records * spr[sel[0]]
    out = np.empty((len(sel), n_samples), dtype='<f8')
    for out_i, ch_i in enumerate(sel):
        seg = triplets[:, offsets[ch_i] : offsets[ch_i + 1], :]
        dig_i32 = (
            seg[:, :, 0].astype('<i4')
            | (seg[:, :, 1].astype('<i4') << 8)
            | (seg[:, :, 2].view('<i1').astype('<i4') << 16)
        )
        dig = dig_i32.reshape(-1).astype('<f8')
        denom = dig_maxs[ch_i] - dig_mins[ch_i]
        if denom == 0:
            raise ValueError(f'Zero digital range in {bdf_path} ch {labels[ch_i]}')
        gain = (phys_maxs[ch_i] - phys_mins[ch_i]) / denom
        phys = (dig - dig_mins[ch_i]) * gain + phys_mins[ch_i]
        out[out_i] = phys * _unit_scale_to_uv(phys_dims[ch_i])

    return out, dict(
        srate=srate,
        n_records=n_records,
        rec_dur=rec_dur,
        ch_names=labels,
        phys_dims=phys_dims,
        samples_per_record=spr,
    )


def resolve_storage_destination(
    dataset_name: str,
    est_gb: float,
    primary_root: Path,
    overflow_root: Path | None,
    min_free_gb: float = 3.5,
) -> tuple[Path, bool]:
    """Choose primary_root if sufficient headroom exists, else overflow_root + junction."""
    primary_root = Path(primary_root)
    primary_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(primary_root).free
    required_bytes = max(min_free_gb * 1e9, est_gb * 1.35 * 1e9)
    if overflow_root is None or free_bytes >= required_bytes:
        return primary_root, False
    overflow_root = Path(overflow_root)
    overflow_root.mkdir(parents=True, exist_ok=True)
    return overflow_root, True


def create_junction_if_needed(primary_dir: Path, target_dir: Path) -> bool:
    """Create a Windows directory junction primary_dir -> target_dir when supported (NTFS)."""
    primary_dir = Path(primary_dir)
    target_dir = Path(target_dir).resolve()
    if primary_dir.exists():
        return True
    proc = subprocess.run(
        ['cmd', '/c', 'mklink', '/J', str(primary_dir), str(target_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        if 'NTFS' in msg.upper():
            print(
                f'Note: skipping junction {primary_dir} -> {target_dir} ({primary_dir.drive} is non-NTFS: {msg})',
                flush=True,
            )
            return False
        raise RuntimeError(f'Failed to create directory junction {primary_dir} -> {target_dir}: {msg}')
    if not primary_dir.exists():
        raise RuntimeError(f'Failed to create directory junction {primary_dir} -> {target_dir}')
    return True


def lock_and_download_openneuro_b(key: str, cache_root: Path) -> dict:
    info = OPENNEURO_RELEASES[key]
    ds = info['dataset']
    ref = info['ref']
    commit = info['commit']
    cache = Path(cache_root) / 'CognitiveGroupB' / ds
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
                        git_blob_sha1=entry['sha'],
                    )
                )

    def _keep_annex(f: dict) -> bool:
        n = f['name']
        if n == 'LICENSE':
            return True
        if not n.startswith('sub-'):
            return False
        if key == 'narrative_reward' and ('/beh/' in n or n.endswith(('_eeg.vhdr', '_eeg.vmrk', '_eeg.eeg'))):
            return True
        if '/eeg/' not in n:
            return False
        if key == 'affective_music' and ('_task-run1_' in n or '_task-run6_' in n):
            return False
        return n.lower().endswith(
            ('_eeg.set', '_eeg.fdt', '_eeg.edf', '_eeg.bdf', '_eeg.vhdr', '_eeg.vmrk', '_eeg.eeg', '_events.tsv', '_channels.tsv')
        )

    selected = [f for f in annex_candidates if _keep_annex(f)]

    def _dl(f: dict) -> dict:
        target = (cache / f['name']).resolve()
        if not target.is_relative_to(cache.resolve()):
            raise ValueError(f"Unsafe annex path {f['name']}")
        fetch(f['url'], target, f['checksum'], f['bytes'])
        return {**f, 'sha256': sha(target)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(_dl, selected))

    files = sorted(inline_files + downloaded, key=lambda item: item['name'])
    licence_evidence = 'Original pinned dataset_description.json License=CC0'
    if key == 'affective_music':
        licence_evidence = (
            'dataset_description.json declares License=CC0; README specifies CC-BY-4.0; '
            'both retained in provenance'
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
        files=files,
    )
    dump(cache / 'download-receipt.json', receipt)
    dump(cache / 'source-record.json', receipt)
    dump(cache / 'original-source-lock.json', receipt)
    dump(ROOT / 'docs' / f'{ds}-cognitive-source-lock.json', receipt)
    return receipt


def _filter_clean_cases(
    cases: list[np.ndarray],
    rows: list[dict],
    seen_hashes: set[str],
    dropped_log: list[dict],
) -> tuple[list[np.ndarray], list[dict]]:
    clean_x = []
    clean_r = []
    for x, r in zip(cases, rows, strict=True):
        arr = np.ascontiguousarray(x, dtype='<f8')
        if not np.isfinite(arr).all() or (arr.std(axis=1) <= 1e-7).any():
            dropped_log.append(dict(recording=r['recording_id'], event=r['event_index'], reason='nonfinite_or_railed'))
            continue
        h = digest_case(arr)
        if h in seen_hashes:
            dropped_log.append(dict(recording=r['recording_id'], event=r['event_index'], reason='duplicate_signal_sha256'))
            continue
        seen_hashes.add(h)
        clean_x.append(arr)
        clean_r.append(r)
    return clean_x, clean_r


def extract_python_code_epochs(epochs_meta: list[dict] | dict) -> list[tuple[int, int, str, str]]:
    """Select critical-token epochs (eventlatency == 0 in PYTHON_CODE_BINS) from ds004771."""
    if isinstance(epochs_meta, dict):
        epochs_meta = [epochs_meta]
    selected = []
    for ep_idx, ep in enumerate(epochs_meta):
        lats = ep.get('eventlatency', [])
        types = ep.get('eventtype', [])
        if not isinstance(lats, (list, tuple, np.ndarray)):
            lats = [lats]
            types = [types]
        else:
            lats = list(lats)
            types = list(types) if isinstance(types, (list, tuple, np.ndarray)) else [types]
        for lat, ev_type in zip(lats, types):
            try:
                lat_val = float(lat)
            except (TypeError, ValueError):
                continue
            if abs(lat_val) > 1e-6:
                continue
            m = re.search(r'\((\d+)\)', str(ev_type))
            if not m:
                continue
            code_num = int(m.group(1))
            if code_num in PYTHON_CODE_BINS:
                lbl, desc = PYTHON_CODE_BINS[code_num]
                selected.append((ep_idx, code_num, lbl, desc))
                break
    return selected


def python_code_reading_records(cache: Path, spec: dict):
    from scipy.io import loadmat

    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(32))
    for set_path in sorted(cache.glob('sub-*/eeg/*_task-PY_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-')
        fdt_path = set_path.with_suffix('.fdt')
        mat = loadmat(set_path, simplify_cells=True)
        eeg_hdr = mat.get('EEG', mat)
        exp_bytes = int(eeg_hdr['nbchan']) * int(eeg_hdr['pnts']) * int(eeg_hdr['trials']) * 4
        if not fdt_path.is_file() or fdt_path.stat().st_size != exp_bytes:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(
                dict(participant_id=person, reason='truncated_or_missing_fdt')
            )
            continue
        data3d, hdr = decode_eeglab_set(set_path)
        names = hdr['ch_names'][:32]
        if hdr['ch_names'][32:] != ['LIO', 'ROC'] or round(hdr['srate']) != 256 or hdr['pnts'] != 334:
            raise ValueError(f'Unexpected ds004771 schema in {set_path}')
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {set_path}')
        sel = extract_python_code_epochs(eeg_hdr['epoch'])
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        for ep_idx, code_num, lbl, desc in sel:
            cases.append(data3d[ep_idx, picks, :])
            row = core.base_row(person, '0', '0', rel, ep_idx, ep_idx * 334, 334, 256, code_num, lbl)
            row.update(trigger_code=code_num, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def fuzzy_semantics_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(30))
    for edf_path in sorted(cache.glob('sub-*/ses-*/eeg/*_eeg.edf')):
        person = edf_path.parts[-4].removeprefix('sub-')
        ses = edf_path.parts[-3].removeprefix('ses-')
        try:
            eeg, hdr = decode_edf(edf_path, picks=picks)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_recordings', []).append(edf_path.name)
            continue
        names = hdr['ch_names'][:30]
        if round(hdr['srate']) != 200:
            raise ValueError(f'Unexpected ds005383 srate in {edf_path}')
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {edf_path}')
        ev_rows = core.table(edf_path.with_name(edf_path.name.replace('_eeg.edf', '_events.tsv')))
        rel = edf_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            tt = ev.get('trial_type', '').strip()
            key = 'target' if tt.startswith('target') else 'nontarget' if tt.startswith('nontarget') else None
            if key is None:
                continue
            onset = float(ev['onset'])
            start = int(round(onset * 200.0)) - 40
            end = start + 200
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = FUZZY_LABELS[key]
            cases.append(eeg[:, start:end])
            row = core.base_row(person, ses, '0', rel, idx, start, 200, 200, tt, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(edf_path),
            participant_id=person,
            session_id=ses,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def extract_internal_attention_windows(ev_rows: list[dict], n_samples: int, hz: float = 250.0, points: int = 500) -> list[tuple[str, int, int, float]]:
    """Slice non-overlapping 2.0 s windows strictly within concentrate and relax blocks."""
    blocks: list[list] = []
    cur = None
    for r in ev_rows:
        tt = r.get('trial_type', '').strip()
        onset = float(r['onset'])
        if cur is None or cur[0] != tt or onset - cur[2] > 2.5:
            if cur is not None:
                blocks.append(cur)
            cur = [tt, onset, onset]
        else:
            cur[2] = onset
    if cur is not None:
        blocks.append(cur)

    windows = []
    for i, (tt, t_first, t_last) in enumerate(blocks):
        if tt not in INTERNAL_ATTN_LABELS:
            continue
        next_t = blocks[i + 1][1] if i + 1 < len(blocks) else (t_last + 1.0)
        stop_t = min(next_t, t_last + 1.0)
        start_s = int(round(t_first * hz))
        stop_s = min(n_samples, int(round(stop_t * hz)))
        for s in range(max(0, start_s), stop_s - points + 1, points):
            windows.append((tt, s, s + points, s / hz))
    return windows


def internal_attention_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(32))
    for set_path in sorted(cache.glob('sub-*/eeg/*_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_eeglab_set(set_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if eeg.ndim != 2 or round(hdr['srate']) != 250 or hdr['nbchan'] != 32:
            raise ValueError(f'Unexpected ds002691 schema in {set_path}')
        names = hdr['ch_names']
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {set_path}')
        ev_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_events.tsv')))
        wins = extract_internal_attention_windows(ev_rows, eeg.shape[1], 250.0, 500)
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        for idx, (tt, start, end, onset_s) in enumerate(wins):
            lbl, desc = INTERNAL_ATTN_LABELS[tt]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 500, 250, tt, lbl)
            row.update(source_event_onset=onset_s, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def visual_symmetry_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(64))
    for bdf_path in sorted(cache.glob('sub-*/eeg/*_task-jacobsen_eeg.bdf')):
        person = bdf_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_bdf(bdf_path, picks=picks)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if round(hdr['srate']) != 512:
            raise ValueError(f'Unexpected ds004347 srate in {bdf_path}')
        names = [n.strip("'") for n in hdr['ch_names'][:64]]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {bdf_path}')
        ev_rows = core.table(bdf_path.with_name(bdf_path.name.replace('_eeg.bdf', '_events.tsv')))
        rel = bdf_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            val = ev.get('value', '').strip()
            if val not in SYMMETRY_LABELS:
                continue
            onset = float(ev['onset'])
            start = int(round(onset * 512.0)) - 102
            end = start + 512
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = SYMMETRY_LABELS[val]
            cases.append(eeg[:, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 512, 512, val, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(bdf_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def moral_conflict_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(32))
    for bdf_path in sorted(cache.glob('sub-*/eeg/*_task-HarmN400_eeg.bdf')):
        person = bdf_path.parts[-3].removeprefix('sub-')
        try:
            eeg_raw, hdr = decode_bdf(bdf_path, picks=picks)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        eeg = harmonise_sampling_rate(eeg_raw, hdr['srate'], 512.0, (512.0, 2048.0))
        names = hdr['ch_names'][:32]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {bdf_path}')
        ev_rows = core.table(bdf_path.with_name(bdf_path.name.replace('_eeg.bdf', '_events.tsv')))
        rel = bdf_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            code = ev.get('type', '').strip()
            if code not in MORAL_LABELS:
                continue
            # ds004860 onset column is in milliseconds
            onset_s = float(ev['onset']) / 1000.0
            start = int(round(onset_s * 512.0)) - 102
            end = start + 512
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = MORAL_LABELS[code]
            cases.append(eeg[:, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 512, 512, code, lbl)
            row.update(source_event_onset=onset_s, condition_name=desc, native_srate_hz=hdr['srate'])
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(bdf_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def probabilistic_feedback_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    for set_path in sorted(cache.glob('sub-*/eeg/*_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_eeglab_set(set_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if round(hdr['srate']) != 500:
            raise ValueError(f'Unexpected ds003753 srate in {set_path}')
        ch_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_channels.tsv')))
        ch_names = [r['name'].strip() for r in ch_rows]
        name_to_idx = {n: i for i, n in enumerate(ch_names)}
        picks = [name_to_idx[n] for n in SHARED_DS003753]
        ev_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_events.tsv')))
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            tt = ev.get('trial_type', '').strip()
            if tt not in PROB_FB_LABELS:
                continue
            onset = float(ev['onset'])
            start = int(round(onset * 500.0)) - 100
            end = start + 500
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = PROB_FB_LABELS[tt]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 500, 500, tt, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, SHARED_DS003753, info
    spec['metadata_updates']['dropped_trials'] = dropped


def _holcomb_set_records(cache: Path, spec: dict, label_map: dict[str, tuple[str, str]]):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    for set_path in sorted(cache.glob('sub-*/eeg/*_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-').zfill(2)
        try:
            eeg, hdr = decode_eeglab_set(set_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if round(hdr['srate']) != 500:
            raise ValueError(f'Unexpected srate in {set_path}')
        ch_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_channels.tsv')))
        ch_names = [r['name'].strip() for r in ch_rows]
        picks = [i for i, n in enumerate(ch_names) if n not in {'LE', 'HE', 'A2'}]
        names = [ch_names[i] for i in picks]
        if len(names) != 29:
            raise ValueError(f'Expected 29 scalp channels in {set_path}, got {len(names)}')
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {set_path}')
        ev_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_events.tsv')))
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            val = ev.get('value', '').strip()
            if val not in label_map:
                continue
            onset = float(ev['onset'])
            start = int(round(onset * 500.0)) - 100
            end = start + 500
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = label_map[val]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 500, 500, val, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def sentence_violations_records(cache: Path, spec: dict):
    yield from _holcomb_set_records(cache, spec, SENTENCE_LABELS)


def flanker_conflict_records(cache: Path, spec: dict):
    yield from _holcomb_set_records(cache, spec, FLANKER_LABELS)


def narrative_reward_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    for vhdr_path in sorted(cache.glob('sub-*/eeg/*_eeg.vhdr')):
        person = vhdr_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_brainvision(vhdr_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        srate = float(hdr['native_sampling_frequency_hz'])
        ch_names = list(hdr['native_channel_names'])
        if round(srate) != 1000:
            raise ValueError(f'Unexpected ds007968 srate in {vhdr_path}: {srate}')
        picks = [i for i, n in enumerate(ch_names) if n not in {'M1', 'M2'}]
        names = [ch_names[i] for i in picks]
        if len(names) != 30:
            raise ValueError(f'Expected 30 scalp channels in {vhdr_path}, got {len(names)}')
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {vhdr_path}')
        ev_rows = core.table(vhdr_path.with_name(vhdr_path.name.replace('_eeg.vhdr', '_events.tsv')))
        beh_path = cache / f'sub-{person}/beh/sub-{person}_task-narrativereward_beh.tsv'
        beh_rows = core.table(beh_path) if beh_path.is_file() else []
        fb_events = [(idx, ev) for idx, ev in enumerate(ev_rows) if ev.get('value', '') in NARRATIVE_LABELS]
        if beh_rows and len(beh_rows) == len(fb_events):
            for (_, ev), br in zip(fb_events, beh_rows):
                val = ev['value']
                exp_val = (
                    ('S  7' if br['outcome'].strip() == '1' else 'S  8')
                    if br['locType'].strip() == '1'
                    else ('S 17' if br['outcome'].strip() == '1' else 'S 18')
                )
                if val != exp_val:
                    raise ValueError(f'beh.tsv vs events.tsv feedback mismatch in {vhdr_path}: {val} != {exp_val}')
        rel = vhdr_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in fb_events:
            val = ev['value']
            onset = float(ev['onset'])
            start = int(round(onset * 1000.0)) - 200
            end = start + 1000
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = NARRATIVE_LABELS[val]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 1000, 1000, val, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(vhdr_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=srate,
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def bandit_gambling_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    for set_path in sorted(cache.glob('sub-*/eeg/*_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_eeglab_set(set_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if round(hdr['srate']) != 500:
            raise ValueError(f'Unexpected ds003458 srate in {set_path}')
        ch_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_channels.tsv')))
        ch_names = [r['name'].strip() for r in ch_rows]
        name_to_idx = {n: i for i, n in enumerate(ch_names)}
        picks = [name_to_idx[n] for n in SHARED_DS003458]
        ev_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_events.tsv')))
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            val = ev.get('value', '')
            if val not in BANDIT_LABELS:
                continue
            onset = float(ev['onset'])
            start = int(round(onset * 500.0)) - 100
            end = start + 500
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = BANDIT_LABELS[val]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 500, 500, val, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, SHARED_DS003458, info
    spec['metadata_updates']['dropped_trials'] = dropped


def extract_music_clip_windows(
    ev_rows: list[dict],
    n_samples: int,
    hz: float = 1000.0,
    points: int = 4000,
    max_windows_per_clip: int = 4,
) -> list[dict]:
    """Extract 4.0 s windows from rated non-neutral music clips in ds002721."""
    starts = [i for i, r in enumerate(ev_rows) if r.get('trial_type', '').strip() == '788']
    windows = []
    for k, idx in enumerate(starts):
        end_i = starts[k + 1] if k + 1 < len(starts) else len(ev_rows)
        seg = ev_rows[idx:end_i]
        clip_onset = float(seg[0]['onset'])
        clip_stop = clip_onset + 16.0
        q_scores: dict[int, int] = {}
        cur_q = None
        for sr in seg[1:]:
            try:
                code = int(sr.get('trial_type', '').strip())
            except ValueError:
                continue
            if code == 1092:
                clip_stop = min(clip_stop, float(sr['onset']))
            elif 800 <= code <= 807:
                cur_q = code
                clip_stop = min(clip_stop, float(sr['onset']))
            elif cur_q is not None and 901 <= code <= 909:
                q_scores.setdefault(cur_q, code - 900)
            elif cur_q is not None and 833 <= code <= 841:
                q_scores.setdefault(cur_q, code - 832)
        pleasant = q_scores.get(800)
        energy = q_scores.get(801)
        if pleasant is None or energy is None or pleasant == 5 or energy == 5:
            continue
        quad = ('HP' if pleasant > 5 else 'LP') + ('HE' if energy > 5 else 'LE')
        lbl, desc = MUSIC_QUADRANTS[quad]
        s0 = int(round(clip_onset * hz))
        s1 = min(n_samples, int(round(clip_stop * hz)))
        w_count = 0
        for s in range(max(0, s0), s1 - points + 1, points):
            windows.append(
                dict(
                    start_sample=s,
                    end_sample=s + points,
                    clip_index=k,
                    window_within_clip=w_count,
                    pleasantness=pleasant,
                    energy=energy,
                    quadrant=quad,
                    label=lbl,
                    label_name=desc,
                )
            )
            w_count += 1
            if w_count >= max_windows_per_clip:
                break
    return windows


def affective_music_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(19))
    for edf_path in sorted(cache.glob('sub-*/eeg/*_eeg.edf')):
        if '_task-run1_' in edf_path.name or '_task-run6_' in edf_path.name:
            continue
        person = edf_path.parts[-3].removeprefix('sub-')
        run_m = re.search(r'_task-run(\d+)_', edf_path.name)
        run_id = run_m.group(1) if run_m else '0'
        try:
            eeg, hdr = decode_edf(edf_path, picks=picks)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_recordings', []).append(edf_path.name)
            continue
        if round(hdr['srate']) != 1000:
            raise ValueError(f'Unexpected ds002721 srate in {edf_path}')
        names = hdr['ch_names'][:19]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {edf_path}')
        ev_rows = core.table(edf_path.with_name(edf_path.name.replace('_eeg.edf', '_events.tsv')))
        wins = extract_music_clip_windows(ev_rows, eeg.shape[1], 1000.0, 4000, 4)
        rel = edf_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        for idx, w in enumerate(wins):
            cases.append(eeg[:, w['start_sample'] : w['end_sample']])
            row = core.base_row(person, '0', run_id, rel, idx, w['start_sample'], 4000, 1000, w['quadrant'], w['label'])
            row.update(
                pleasantness=w['pleasantness'],
                energy=w['energy'],
                window_within_clip=w['window_within_clip'],
                condition_name=w['label_name'],
            )
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(edf_path),
            participant_id=person,
            run_id=run_id,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


def verbal_working_memory_records(cache: Path, spec: dict):
    seen_hashes: set[str] = set()
    dropped: list[dict] = []
    expected_names = None
    picks = list(range(19))
    for set_path in sorted(cache.glob('sub-*/eeg/*_eeg.set')):
        person = set_path.parts[-3].removeprefix('sub-')
        try:
            eeg, hdr = decode_eeglab_set(set_path)
        except ValueError:
            spec['metadata_updates'].setdefault('excluded_participants', []).append(person)
            continue
        if round(hdr['srate']) != 500:
            raise ValueError(f'Unexpected ds003655 srate in {set_path}')
        names = hdr['ch_names'][:19]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel mismatch in {set_path}')
        ev_rows = core.table(set_path.with_name(set_path.name.replace('_eeg.set', '_events.tsv')))
        rel = set_path.relative_to(cache).as_posix()
        cases = []
        rows = []
        prev_end = -1
        for idx, ev in enumerate(ev_rows):
            tt = ev.get('trial_type', '').strip()
            if tt not in VERBAL_WM_LABELS:
                continue
            onset = float(ev['onset'])
            if idx + 1 < len(ev_rows):
                next_onset = float(ev_rows[idx + 1]['onset'])
                if next_onset - onset < 2.0 - 1e-3:
                    continue
            start = int(round(onset * 500.0))
            end = start + 1000
            if start < prev_end or start < 0 or end > eeg.shape[1]:
                continue
            prev_end = end
            lbl, desc = VERBAL_WM_LABELS[tt]
            cases.append(eeg[picks, start:end])
            row = core.base_row(person, '0', '0', rel, idx, start, 1000, 500, tt, lbl)
            row.update(source_event_onset=onset, condition_name=desc)
            rows.append(row)
        cases, rows = _filter_clean_cases(cases, rows, seen_hashes, dropped)
        if not cases:
            continue
        info = dict(
            source_file=rel,
            source_file_sha256=sha(set_path),
            participant_id=person,
            source_eeg_indices=picks,
            native_sampling_rate_hz=hdr['srate'],
        )
        yield np.stack(cases), rows, expected_names, info
    spec['metadata_updates']['dropped_trials'] = dropped


OPENNEURO_META = {
    'python_code_reading': dict(
        hz=256, points=334, labels=PYTHON_CODE_BINS, reader=python_code_reading_records,
        source='Kuo & Prat / University of Washington / OpenNeuro ds004771 v1.0.0',
        doi='10.18112/openneuro.ds004771.v1.0.0',
        paper='Kuo & Prat (2024), Programmers show distinct, language-like brain responses to violations in form and meaning when reading code, DOI 10.18112/openneuro.ds004771.v1.0.0',
        note='Four-class Python code comprehension ERP classification (well_formed, semantically_implausible, structurally_invalid, doubly_anomalous critical tokens) across 61 participants at native 256 Hz (32 scalp EEG channels, 334 samples; LIO/ROC ocular channels excluded).',
    ),
    'fuzzy_semantics': dict(
        hz=200, points=200, labels=FUZZY_LABELS, reader=fuzzy_semantics_records,
        source='TMNRED / OpenNeuro ds005383 v1.0.0',
        doi='10.18112/openneuro.ds005383.v1.0.0',
        paper='TMNRED: A Chinese Language EEG Dataset for Fuzzy Semantic Target Identification in Natural Reading Environments, OpenNeuro ds005383 v1.0.0',
        note='Binary semantic target vs non-target natural reading classification across 30 participants at native 200 Hz (30 scalp EEG channels, 200 samples [-0.2, 0.8) s; Status trigger channel excluded).',
    ),
    'internal_attention': dict(
        hz=250, points=500, labels=INTERNAL_ATTN_LABELS, reader=internal_attention_records,
        source='Delorme et al. / OpenNeuro ds002691 v1.1.0',
        doi='10.18112/openneuro.ds002691.v1.1.0',
        paper='Internal attention study, OpenNeuro ds002691 v1.1.0, DOI 10.18112/openneuro.ds002691.v1.1.0',
        note='Binary internal mental concentration vs relaxation classification across 20 participants at native 250 Hz (32 scalp EEG channels, non-overlapping 2.0-second / 500-sample windows strictly within instructed blocks).',
    ),
    'visual_symmetry': dict(
        hz=512, points=512, labels=SYMMETRY_LABELS, reader=visual_symmetry_records,
        source='Makin et al. / University of Liverpool SPN1 / OpenNeuro ds004347 v1.0.0',
        doi='10.18112/openneuro.ds004347.v1.0.0',
        paper='Makin et al. (2022), Symmetry perception and affective responses (SPN1 Jacobsen task), OpenNeuro ds004347 v1.0.0',
        note='Binary regular vs random visual symmetry perception classification across 24 participants from raw BioSemi BDF recordings at native 512 Hz (64 scalp EEG channels, 512 samples [-0.2, 0.8) s; EXG1-8 and Status excluded).',
    ),
    'moral_conflict': dict(
        hz=512, points=512, labels=MORAL_LABELS, reader=moral_conflict_records,
        source='OpenNeuro ds004860 v1.0.0',
        doi='10.18112/openneuro.ds004860.v1.0.0',
        paper='Investigating the cognitive conflict triggered by moral judgment of accidental harm: an ERP study, OpenNeuro ds004860 v1.0.0',
        note='Four-class moral judgement x semantic congruency target ERP classification (accidental/intentional harm x congruent/incongruent target) across 31 participants at 512 Hz (32 BioSemi scalp EEG channels, 512 samples [-0.2, 0.8) s; sub-110 decimated 4x from 2,048 Hz to match the 30 participants recorded at 512 Hz).',
    ),
    'probabilistic_feedback': dict(
        hz=500, points=500, labels=PROB_FB_LABELS, reader=probabilistic_feedback_records,
        source='Brown, Jackson & Cavanagh / University of New Mexico / OpenNeuro ds003753 v1.1.0',
        doi='10.18112/openneuro.ds003753.v1.1.0',
        paper='Brown, Jackson & Cavanagh (2022), The Reward Positivity is sensitive to affective liking, Cognitive, Affective, & Behavioral Neuroscience',
        note='Four-class reinforcement learning outcome x affective stimulus classification (win/lose x puppy/cow) across 25 participants at native 500 Hz (59 shared 10-10 scalp EEG channels, 500 samples [-0.2, 0.8) s).',
    ),
    'sentence_violations': dict(
        hz=500, points=500, labels=SENTENCE_LABELS, reader=sentence_violations_records,
        source='Holcomb / San Diego State University NCL / OpenNeuro ds008711 v1.0.0',
        doi='10.18112/openneuro.ds008711.v1.0.0',
        paper='RSVP with flankers: sentences with semantic and syntactic violations, OpenNeuro ds008711 v1.0.0',
        note='Five-class psycholinguistic critical-word RSVP classification (control, syntactic_violation, semantic_violation, double_violation, word_order_anomaly) across 64 participants at native 500 Hz (29 scalp EEG channels, 500 samples [-0.2, 0.8) s; LE, HE, A2 excluded).',
    ),
    'narrative_reward': dict(
        hz=1000, points=1000, labels=NARRATIVE_LABELS, reader=narrative_reward_records,
        source='MacEwan University / OpenNeuro ds007968 v1.0.0',
        doi='10.18112/openneuro.ds007968.v1.0.0',
        paper='Narrative Reward, OpenNeuro ds007968 v1.0.0, DOI 10.18112/openneuro.ds007968.v1.0.0',
        note='Four-class narrative context x reward feedback classification (narrative/nonnarrative x win/loss, cross-verified 1-to-1 against beh/*_beh.tsv) across 40 participants at native 1,000 Hz (30 scalp EEG channels, 1,000 samples [-0.2, 0.8) s; M1, M2 mastoids excluded).',
    ),
    'flanker_conflict': dict(
        hz=500, points=500, labels=FLANKER_LABELS, reader=flanker_conflict_records,
        source='Holcomb & Emmorey / San Diego State University NCL / OpenNeuro ds005866 v1.0.1',
        doi='10.18112/openneuro.ds005866.v1.0.1',
        paper='Flankers-NEAR lexical and flanker interference study, OpenNeuro ds005866 v1.0.1',
        note='Six-class lexicality x flanker interference classification (word/pseudoword x none/same/different flankers) across 60 participants at native 500 Hz (29 scalp EEG channels, 500 samples [-0.2, 0.8) s; LE, HE, A2 excluded).',
    ),
    'bandit_gambling': dict(
        hz=500, points=500, labels=BANDIT_LABELS, reader=bandit_gambling_records,
        source='Cavanagh / University of New Mexico / OpenNeuro ds003458 v1.1.0',
        doi='10.18112/openneuro.ds003458.v1.1.0',
        paper='Cavanagh, Three-armed oscillating bandit gambling task, OpenNeuro ds003458 v1.1.0',
        note='Binary reinforcement-learning reward vs loss feedback classification (S  5 win vs S  6 loss) in an oscillating three-armed bandit task across 23 participants at native 500 Hz (63 shared scalp EEG channels, 500 samples [-0.2, 0.8) s).',
    ),
    'affective_music': dict(
        hz=1000, points=4000, labels=MUSIC_QUADRANTS, reader=affective_music_records,
        source='Daly, Nicolaou, Williams, Hwang, Kirke, Miranda & Nasuto / University of Reading / OpenNeuro ds002721 v1.0.3',
        doi='10.18112/openneuro.ds002721.v1.0.3',
        paper='Daly et al. (2018/2020), Neural and physiological data from participants listening to affective music, Scientific Data 7, 177, DOI 10.1038/s41597-020-0507-6',
        note='Four-class music-induced affect classification across self-reported pleasantness (Q800) x energy (Q801) circumplex quadrants (HPHE, HPLE, LPHE, LPLE; neutral midpoint 5 excluded) across 31 participants at native 1,000 Hz (19 scalp EEG channels, up to four non-overlapping 4.0-second / 4,000-sample windows strictly within each 16 s music listening clip). README specifies CC-BY-4.0 while OpenNeuro dataset_description.json declares CC0.',
    ),
    'verbal_working_memory': dict(
        hz=500, points=1000, labels=VERBAL_WM_LABELS, reader=verbal_working_memory_records,
        source='Pavlov & Kotchoubey / Ural Federal University / OpenNeuro ds003655 v1.0.2',
        doi='10.18112/openneuro.ds003655.v1.0.2',
        paper='Pavlov & Kotchoubey (2022), Oscillatory brain activity and maintenance of verbal working memory, eLife 11, e74755',
        note='Six-class modified Sternberg verbal working-memory maintenance classification (5, 6, or 7 letters x simple retention vs alphabetical manipulation: 5R, 5M, 6R, 6M, 7R, 7M) across 156 participants at native 500 Hz (19 scalp EEG channels, 2.0-second / 1,000-sample delay-period windows; EOGv and EOGh excluded).',
    ),
}


def build_openneuro_spec(key: str, cache_root: Path) -> tuple[dict, callable, Path, str]:
    rel_info = OPENNEURO_RELEASES[key]
    meta = OPENNEURO_META[key]
    ds = rel_info['dataset']
    ref = rel_info['ref']
    cache = Path(cache_root) / 'CognitiveGroupB' / ds
    labels = {v[0]: v[1] for v in meta['labels'].values()}
    spec = dict(
        name=rel_info['name'],
        source=meta['source'],
        cache=f'CognitiveGroupB/{ds}',
        hz=meta['hz'],
        points=meta['points'],
        classes=len(labels),
        train_subjects=0,
        licence=rel_info['licence'],
        licence_url='https://creativecommons.org/publicdomain/zero/1.0/',
        moabb_revision=None,
        doi=meta['doi'],
        paper=meta['paper'],
        labels=labels,
        splitter=lambda p, chunks: whole_subject_split(p)[0],
        evidence_files=['cognitive-group-b-plan.md', f'{ds}-cognitive-source-lock.json'],
        cache_evidence_files=[
            p for p in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'README.txt', 'original-source-lock.json']
            if (cache / p).is_file()
        ],
        conversion_files=['prepare_cognitive_group_b.py'],
        intermediate_directory='classification-intermediate-full',
        note=meta['note'],
        metadata_updates=dict(
            domain='Psy',
            original_source_url=f'https://openneuro.org/datasets/{ds}/versions/{ref}',
            paper_reference=meta['paper'],
            excluded_participants=[],
        ),
    )
    return spec, meta['reader'], cache, f'{ds}-cognitive'


def prepare_one(
    key: str,
    cache_root: Path = Path('C:/Data/EEGSources'),
    primary_output: Path = Path('D:/Data/EEG'),
    overflow_output: Path | None = Path('C:/Data/EEG'),
    purge_cache: bool = True,
) -> dict:
    cache_root = Path(cache_root)
    primary_output = Path(primary_output)
    name = DATASET_NAMES[key]
    primary_final = primary_output / name

    if key in ERPCORE_SPECS:
        est_gb = ERPCORE_EST_GB[key]
        dest_root, use_junction = resolve_storage_destination(name, est_gb, primary_output, overflow_output)
        comp = ERPCORE_SPECS[key]['component']
        cache = cache_root / 'CognitiveGroupB' / f'ERPCORE_{comp}'
        result = convert_erpcore_component(key, cache, dest_root, purge_cache=purge_cache)
        verify_final(result)
        if use_junction:
            result['physical_path'] = str(dest_root / name)
            if create_junction_if_needed(primary_final, dest_root / name):
                result['junction_path'] = str(primary_final)
                result['path'] = str(primary_final)
                verify_final(result)
        dump(ROOT / 'docs' / f'erpcore-{comp.lower()}-cognitive-conversion-result.json', result)
        return result

    est_gb = OPENNEURO_RELEASES[key]['est_gb']
    dest_root, use_junction = resolve_storage_destination(name, est_gb, primary_output, overflow_output)
    lock_and_download_openneuro_b(key, cache_root)
    spec, reader, cache, doc_stem = build_openneuro_spec(key, cache_root)
    core.SPECS[key] = spec
    core.READERS[key] = reader
    result = core.prepare(key, cache_root, dest_root)
    folder = Path(result['path'])
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
    if use_junction:
        result['physical_path'] = str(dest_root / name)
        if create_junction_if_needed(primary_final, dest_root / name):
            result['junction_path'] = str(primary_final)
            result['path'] = str(primary_final)
            verify_final(result)
    if purge_cache and cache.is_dir():
        shutil.rmtree(cache)
        result['cache_purged'] = True
    dump(ROOT / 'docs' / f'{doc_stem}-conversion-result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=GROUP_B_ORDER, default=GROUP_B_ORDER)
    parser.add_argument('--cache-root', type=Path, default=Path('C:/Data/EEGSources'))
    parser.add_argument('--output', type=Path, default=Path('D:/Data/EEG'))
    parser.add_argument('--overflow-output', type=Path, default=Path('C:/Data/EEG'))
    parser.add_argument('--keep-cache', action='store_true')
    args = parser.parse_args()

    for key in args.datasets:
        name = DATASET_NAMES[key]
        primary_final = args.output / name
        if primary_final.exists():
            print(f'Skipping already-exported {name}; verifying manifest...', flush=True)
            verify_final({'dataset': name, 'path': str(primary_final)})
            continue
        prepare_one(
            key,
            cache_root=args.cache_root,
            primary_output=args.output,
            overflow_output=args.overflow_output,
            purge_cache=not args.keep_cache,
        )


if __name__ == '__main__':
    main()
