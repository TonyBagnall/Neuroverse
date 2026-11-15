"""Prepare ERP CORE N170 (FaceStimuli), MMN (AuditoryDeviance), and N2pc (SpatialAttention) at native 1024 Hz."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.io import loadmat

from tools.prepare_moabb import MOABB_REVISION, dump, fetch, lines, sha, signal_hash
from tools.prepare_small_moabb import check_amplitude_scale

SFREQ = 1024
POINTS = 1024
PRE = 205
MANIFEST_URL = 'https://zenodo.org/records/14866307/files/erpcore_manifest.csv?download=1'
MANIFEST_SHA256 = '16ffd88374e29f121e7fd77cade429d25704526baac11ede027a484c50af5cff'
LOCAL_MANIFEST_FALLBACK = Path('D:/Data/EEG/WordRelatedness/source_metadata/erpcore_manifest.csv')

CHANNELS = [
    'FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7',
    'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'FP2', 'Fz', 'F4', 'F8', 'FC4',
    'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

ERPCORE_SPECS = {
    'face_stimuli': dict(
        name='FaceStimuli',
        component='N170',
        bids_doi='10.18112/openneuro.ds003061.v1.0.0',
        classes=4,
        label_names={1: 'faces', 2: 'cars', 3: 'scrambled_faces', 4: 'scrambled_cars'},
        target='Four-class visual category perception (faces, cars, scrambled_faces, scrambled_cars)',
    ),
    'auditory_deviance': dict(
        name='AuditoryDeviance',
        component='MMN',
        bids_doi='10.18112/openneuro.ds003064.v1.0.0',
        classes=2,
        label_names={1: 'pre_deviant_standard', 2: 'deviant'},
        target='Passive auditory oddball mismatch negativity (80 dB pre-deviant standard vs 70 dB deviant tone)',
    ),
    'spatial_attention': dict(
        name='SpatialAttention',
        component='N2pc',
        bids_doi='10.18112/openneuro.ds003065.v1.0.0',
        classes=2,
        label_names={1: 'target_left', 2: 'target_right'},
        target='Covert visual spatial attention target hemifield (target_left vs target_right)',
    ),
}

FIELDS = [
    'split', 'row', 'example_id', 'group', 'participant_id', 'source_participant_id',
    'session_id', 'run_id', 'loader_session_id', 'loader_run_id', 'recording_id', 'source_file',
    'source_events_row_index', 'original_eeglab_event_index', 'original_urevent_id', 'target_trial_index',
    'source_sample_1based', 'event_sample_0based', 'start_sample', 'end_sample', 'source_onset_seconds',
    'event_time_seconds', 'start_time_seconds', 'end_time_seconds', 'source_label', 'label',
    'label_description', 'source_trial_type', 'original_partition', 'signal_sha256',
]


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sources(cache: Path, component: str, subjects: tuple[int, ...], download: bool = True) -> list[dict]:
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'erpcore_manifest.csv'
    if not manifest.is_file() and LOCAL_MANIFEST_FALLBACK.is_file():
        shutil.copy2(LOCAL_MANIFEST_FALLBACK, manifest)
    elif download and not manifest.is_file():
        fetch(MANIFEST_URL, manifest)
    if sha(manifest) != MANIFEST_SHA256:
        raise ValueError('ERP CORE manifest SHA256 mismatch against pinned revision')

    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    sub_tokens = {f'sub-{s:03d}' for s in subjects}
    selected = frame[
        (frame.component == component)
        & ((frame.participant_id == '') | frame.participant_id.isin(sub_tokens))
    ]
    prefix = f'erpcore/{component}/'

    def ensure(row: dict) -> dict:
        relative = Path(row['local_path'].removeprefix(prefix))
        path = cache / relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError(f'Unsafe ERP CORE source path: {relative}')
        if download:
            fetch(row['url'], path, row['hash'], int(row['size']))
        if not path.is_file() or path.stat().st_size != int(row['size']) or 'md5:' + sha(path, 'md5') != row['hash']:
            raise ValueError(f'Missing or modified ERP CORE source: {relative}')
        return dict(
            path=relative.as_posix(),
            url=row['url'],
            checksum=row['hash'],
            sha256=sha(path),
            bytes=path.stat().st_size,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        inventory = list(pool.map(ensure, selected.to_dict('records')))

    if 'CC BY-SA 4.0' not in (cache / 'LICENSE').read_text(encoding='utf-8', errors='replace'):
        raise ValueError('Original ERP CORE LICENSE text changed')

    loader = cache / 'moabb-erpcore-loader.py'
    local_loader = LOCAL_MANIFEST_FALLBACK.parent / 'moabb-erpcore-loader.py'
    if not loader.is_file() and local_loader.is_file():
        shutil.copy2(local_loader, loader)
    elif download and not loader.is_file():
        fetch(
            f'https://raw.githubusercontent.com/NeuroTechX/moabb/{MOABB_REVISION}/moabb/datasets/erpcore2021.py',
            loader,
        )
    if not loader.is_file():
        raise ValueError('Missing pinned MOABB ERP CORE loader evidence')
    return inventory


def read_recording(cache: Path, component: str, subject: int):
    stem = f'sub-{subject:03d}_task-{component}'
    directory = Path(cache) / f'sub-{subject:03d}' / 'eeg'
    file = directory / f'{stem}_eeg.set'
    mat = loadmat(file, simplify_cells=True)
    eeg = mat.get('EEG', mat)
    names = [ch['labels'] for ch in eeg['chanlocs']]
    channels = pd.read_csv(directory / f'{stem}_channels.tsv', sep='\t', dtype=str, keep_default_na=False)
    if names != channels.name.tolist():
        raise ValueError('EEGLAB and sidecar channel order disagree')
    picks = [i for i, row in channels.iterrows() if row['type'].upper() == 'EEG']
    if [names[i] for i in picks] != CHANNELS or len(channels) != 33 or sum(channels.type.str.upper() == 'EOG') != 3:
        raise ValueError('Inconsistent ERP CORE EEG/EOG channel names/types/order')
    if channels.units.tolist() != ['microV'] * 33:
        raise ValueError('Unexpected original ERP CORE signal units')
    if int(eeg['nbchan']) != 33 or int(eeg['trials']) != 1 or float(eeg['srate']) != SFREQ:
        raise ValueError('Unexpected ERP CORE EEGLAB recording schema')
    binary = directory / f'{stem}_eeg.fdt'
    if eeg['data'] != binary.name or binary.stat().st_size != int(eeg['pnts']) * 33 * 4:
        raise ValueError('Incorrect ERP CORE FDT binary reference or size')
    data = np.memmap(binary, dtype='<f4', mode='r', shape=(int(eeg['pnts']), 33))
    events = pd.read_csv(directory / f'{stem}_events.tsv', sep='\t', dtype=str, keep_default_na=False)
    native = eeg['event'] if isinstance(eeg['event'], list) else [eeg['event']]
    lookup = {
        (float(event['latency']), str(event['type'])): (i + 1, event.get('urevent', ''))
        for i, event in enumerate(native)
    }
    positions = pd.read_csv(directory / f'{stem}_electrodes.tsv', sep='\t', dtype=str, keep_default_na=False)
    if positions.name.tolist() != names:
        raise ValueError('Position sidecar order disagrees with signal axes')
    coord = json.loads((directory / f'{stem}_coordsystem.json').read_text(encoding='utf-8'))
    if coord.get('EEGCoordinateUnits') != 'mm':
        raise ValueError('Unexpected ERP CORE position units')
    eeg_json = json.loads((directory / f'{stem}_eeg.json').read_text(encoding='utf-8'))
    return eeg, data, events, lookup, channels, positions, coord, eeg_json, picks


def map_code_for_component(component: str, code: int) -> int | None:
    if component == 'N170':
        if 1 <= code <= 40:
            return 1
        if 41 <= code <= 80:
            return 2
        if 101 <= code <= 140:
            return 3
        if 141 <= code <= 180:
            return 4
        return None
    if component == 'N2pc':
        if code in {111, 112, 211, 212}:
            return 1
        if code in {121, 122, 221, 222}:
            return 2
        return None
    raise ValueError(f'Direct code map not applicable to {component}')


def target_rows(component: str, events: pd.DataFrame, lookup: dict | None = None) -> list[dict]:
    """Extract verified zero-based target events for N170, MMN, or N2pc."""
    rows = events.to_dict('records')
    selected_pairs: list[tuple[int, dict, int, int]] = []

    if component == 'MMN':
        stim_indices: list[tuple[int, dict, int]] = []
        for idx, row in enumerate(rows):
            if row.get('trial_type', '').strip().lower() != 'stimulus':
                continue
            try:
                code = int(row['value'])
            except ValueError:
                continue
            if code in {70, 80, 180}:
                stim_indices.append((idx, row, code))
        used_rows: set[int] = set()
        for pos, (idx, row, code) in enumerate(stim_indices):
            if code != 70:
                continue
            # Pair with immediately preceding non-habituation standard (80) if unused
            if pos > 0:
                prev_idx, prev_row, prev_code = stim_indices[pos - 1]
                if prev_code == 80 and prev_idx not in used_rows:
                    selected_pairs.append((prev_idx, prev_row, prev_code, 1))
                    used_rows.add(prev_idx)
            selected_pairs.append((idx, row, code, 2))
            used_rows.add(idx)
        selected_pairs.sort(key=lambda item: item[0])
    else:
        for idx, row in enumerate(rows):
            if row.get('trial_type', '').strip().lower() != 'stimulus':
                continue
            try:
                code = int(row['value'])
            except ValueError:
                continue
            mapped = map_code_for_component(component, code)
            if mapped is not None:
                selected_pairs.append((idx, row, code, mapped))

    targets = []
    for source_index, row, code, label in selected_pairs:
        sample = float(row['sample'])
        if sample != round(sample):
            raise ValueError('Non-integer ERP CORE event sample')
        if lookup is not None:
            if (sample, str(code)) not in lookup:
                raise ValueError('Original EEGLAB target event does not match BIDS sample/code')
            original_index, urevent = lookup[(sample, str(code))]
        else:
            original_index, urevent = source_index + 1, ''
        zero = int(sample) - 1
        if abs(float(row['onset']) * SFREQ - zero) > 0.052:
            raise ValueError('Source event sample base/time mismatch')
        targets.append(
            dict(
                source_index=source_index,
                row=row,
                code=code,
                label=label,
                sample=int(sample),
                zero=zero,
                original_index=original_index,
                urevent=str(int(urevent)) if isinstance(urevent, (float, int)) else str(urevent),
            )
        )
    if not targets:
        raise ValueError(f'No target events extracted for ERP CORE {component}')
    return targets


def header(spec: dict) -> str:
    cls_tokens = ' '.join(str(i) for i in range(1, spec['classes'] + 1))
    return (
        f"# ERP CORE {spec['component']}, CC BY-SA 4.0. See provenance, sensor and case sidecars.\n"
        f"@problemName {spec['name']}\n@timestamps false\n@missing false\n@univariate false\n"
        f"@dimensions 30\n@equalLength true\n@seriesLength {POINTS}\n@classLabel true {cls_tokens}\n@data\n"
    )


def validate_split(folder: Path, spec: dict, split: str):
    name = spec['name']
    count = 0
    counts = Counter()
    people = set()
    hashes = set()
    last = (-1, -1)
    channel_sds = []
    case_peaks = []
    with (folder / f'cases_{split.lower()}.tsv').open(newline='', encoding='utf-8') as file:
        rows = iter(csv.DictReader(file, delimiter='\t'))
        for count, line in enumerate(lines(folder / f'{name}_{split}.ts'), 1):
            row = next(rows, None)
            if row is None or row['split'] != split or int(row['row']) != count - 1:
                raise ValueError('Case metadata row alignment failed')
            parts = line.strip().split(':')
            if len(parts) != 31:
                raise ValueError('Incorrect exported EEG channel count')
            array = np.stack([np.fromstring(ch, dtype='<f4', sep=',') for ch in parts[:-1]])
            label = int(parts[-1])
            hashed = signal_hash(array)
            if array.shape != (30, POINTS) or hashed != row['signal_sha256'] or int(row['label']) != label:
                raise ValueError('Signal/label differs from source epoch')
            event = int(row['event_sample_0based'])
            key = (int(row['participant_id']), event)
            if (
                event != int(row['source_sample_1based']) - 1
                or int(row['start_sample']) != event - PRE
                or int(row['end_sample']) != event - PRE + POINTS
                or key <= last
            ):
                raise ValueError('Sample reference, crop or source order changed')
            last = key
            counts[label] += 1
            people.add(row['participant_id'])
            hashes.add(hashed)
            arr64 = array.astype(np.float64)
            channel_sds.append(arr64.std(axis=1))
            case_peaks.append(float(np.abs(arr64).max()))
        if next(rows, None) is not None:
            raise ValueError('Extra metadata rows')
    amplitude = check_amplitude_scale(np.concatenate(channel_sds), case_peaks)
    return (
        dict(
            cases=count,
            shape=[count, 30, POINTS],
            class_counts=dict(sorted(counts.items())),
            subjects=sorted(people),
            exact_float32_roundtrip=True,
            metadata_rows_verified=True,
            **amplitude,
        ),
        hashes,
    )


def smoke_test(folder: Path, spec: dict) -> dict:
    import aeon
    from aeon.classification.feature_based import SummaryClassifier
    from aeon.datasets import load_from_ts_file

    name = spec['name']
    loaded = {}
    with tempfile.TemporaryDirectory(dir=folder, prefix='aeon-check-') as tmp:
        for split in ['TRAIN', 'TEST']:
            examples = {c: [] for c in range(1, spec['classes'] + 1)}
            for line in lines(folder / f'{name}_{split}.ts'):
                label = int(line.rsplit(':', 1)[1])
                if len(examples[label]) < 3:
                    examples[label].append(line)
                if all(len(v) == 3 for v in examples.values()):
                    break
            path = Path(tmp) / f'{split}.ts'
            path.write_text(
                header(spec) + ''.join(line for rows in examples.values() for line in rows),
                encoding='ascii',
            )
            loaded[split] = load_from_ts_file(path)
        classifier = SummaryClassifier(random_state=0, n_jobs=1)
        classifier.fit(*loaded['TRAIN'])
        predicted = classifier.predict(loaded['TEST'][0])
        if len(predicted) != len(loaded['TEST'][1]):
            raise ValueError('aeon classifier compatibility failed')
    return dict(
        passed=True,
        aeon_version=aeon.__version__,
        classifier='SummaryClassifier',
        train_cases=len(loaded['TRAIN'][1]),
        test_cases=len(predicted),
        note='Representative compatibility check; full numeric readback checked separately.',
    )


def convert_erpcore_component(
    key: str,
    cache: Path,
    output: Path,
    subjects: tuple[int, ...] = tuple(range(1, 41)),
    train: tuple[int, ...] = tuple(range(1, 31)),
    download: bool = True,
    smoke: bool = True,
    purge_cache: bool = True,
) -> dict:
    spec = ERPCORE_SPECS[key]
    name = spec['name']
    component = spec['component']
    cache, output = Path(cache).resolve(), Path(output).resolve()
    subjects = tuple(sorted(subjects))
    train_set = set(train)
    folder = output / name
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite {folder}')
    inventory = sources(cache, component, subjects, download)
    demographics = pd.read_csv(
        cache / 'participants.tsv', sep='\t', dtype=str, keep_default_na=False
    ).set_index('participant_id')

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=output) as tmp:
        stage = Path(tmp)
        evidence = stage / 'source_metadata'
        evidence.mkdir()
        handles = {}
        tables = {}
        writers = {}
        counts = Counter()
        all_rows = []
        records = []
        discarded = []
        sensor_rows = None
        position_sets = []
        seen_hashes: set[str] = set()
        fmt = ','.join(['%.9g'] * POINTS)
        try:
            for split in ['TRAIN', 'TEST']:
                handles[split] = (stage / f'{name}_{split}.ts').open('w', encoding='ascii', newline='\n')
                handles[split].write(header(spec))
                tables[split] = (stage / f'cases_{split.lower()}.tsv').open('w', encoding='utf-8', newline='')
                writers[split] = csv.DictWriter(tables[split], FIELDS, delimiter='\t', lineterminator='\n')
                writers[split].writeheader()
            for subject in subjects:
                eeg, data, events, lookup, channels, positions, coord, eeg_json, picks = read_recording(
                    cache, component, subject
                )
                try:
                    targets = target_rows(component, events, lookup)
                    subject_id = f'{subject:03d}'
                    stem = f'sub-{subject_id}_task-{component}'
                    split = 'TRAIN' if subject in train_set else 'TEST'
                    raw_date = safe(eeg.get('etc', {}).get('T0', []))
                    measurement = ''
                    if isinstance(raw_date, list) and len(raw_date) == 6:
                        try:
                            measurement = datetime.datetime(*[int(v) for v in raw_date]).isoformat()
                        except ValueError:
                            pass
                    n = 0
                    for trial, item in enumerate(targets):
                        zero = item['zero']
                        start = zero - PRE
                        end = start + POINTS
                        code = item['code']
                        label = item['label']
                        if start < 0 or end > len(data):
                            discarded.append(
                                dict(
                                    participant_id=subject_id,
                                    source_event_row=item['source_index'],
                                    reason='incomplete one-second epoch',
                                )
                            )
                            continue
                        case = np.asarray(data[start:end][:, picks].T, dtype='<f4')
                        if not np.isfinite(case).all() or (case.std(axis=1) <= 1e-7).any():
                            discarded.append(
                                dict(
                                    participant_id=subject_id,
                                    source_event_row=item['source_index'],
                                    reason='nonfinite or railed/flat channel on trial',
                                )
                            )
                            continue
                        sig_sha = signal_hash(case)
                        if sig_sha in seen_hashes:
                            discarded.append(
                                dict(
                                    participant_id=subject_id,
                                    source_event_row=item['source_index'],
                                    reason='duplicate signal sha256',
                                )
                            )
                            continue
                        seen_hashes.add(sig_sha)
                        source = item['row']
                        row = dict(
                            split=split,
                            row=counts[split],
                            example_id=f'ERP-CORE/{component}/sub-{subject_id}/events-row-{item["source_index"]}',
                            group=subject_id,
                            participant_id=subject_id,
                            source_participant_id=f'sub-{subject_id}',
                            session_id=measurement,
                            run_id='',
                            loader_session_id='0',
                            loader_run_id='0',
                            recording_id=stem,
                            source_file=stem + '_eeg.fdt',
                            source_events_row_index=item['source_index'],
                            original_eeglab_event_index=item['original_index'],
                            original_urevent_id=item['urevent'],
                            target_trial_index=trial,
                            source_sample_1based=item['sample'],
                            event_sample_0based=zero,
                            start_sample=start,
                            end_sample=end,
                            source_onset_seconds=source['onset'],
                            event_time_seconds=zero / SFREQ,
                            start_time_seconds=start / SFREQ,
                            end_time_seconds=end / SFREQ,
                            source_label=code,
                            label=label,
                            label_description=spec['label_names'][label],
                            source_trial_type=source['trial_type'],
                            original_partition='unpartitioned',
                            signal_sha256=sig_sha,
                        )
                        writers[split].writerow(row)
                        all_rows.append(row)
                        handles[split].write(
                            ':'.join(fmt % tuple(channel.tolist()) for channel in case) + f':{label}\n'
                        )
                        counts[split] += 1
                        n += 1
                    compact = safe(
                        {
                            k: eeg.get(k)
                            for k in [
                                'nbchan', 'pnts', 'trials', 'srate', 'xmin', 'xmax',
                                'ref', 'setname', 'filename', 'data', 'subject', 'session', 'run', 'etc', 'chanlocs',
                            ]
                        }
                    )
                    dump(evidence / f'{stem}_eeglab-header.json', compact)
                    subdirectory = evidence / f'sub-{subject_id}'
                    subdirectory.mkdir()
                    for path in (cache / f'sub-{subject_id}' / 'eeg').iterdir():
                        if path.suffix.lower() not in ['.set', '.fdt']:
                            shutil.copy2(path, subdirectory / path.name)
                    position_sets.append(positions.to_dict('records'))
                    if sensor_rows is None:
                        sensor_rows = [
                            dict(
                                original_array_index=i,
                                name=r['name'],
                                type=r['type'],
                                units=r['units'],
                                included_in_arrays=i in picks,
                                exported_array_index=picks.index(i) if i in picks else None,
                                position=positions.iloc[i].to_dict(),
                                coordinate_system=coord,
                            )
                            for i, r in channels.iterrows()
                        ]
                    records.append(
                        dict(
                            participant_id=subject_id,
                            recording_id=stem,
                            samples=int(eeg['pnts']),
                            frequency=SFREQ,
                            channels=channels.to_dict('records'),
                            positions=positions.to_dict('records'),
                            coordinate_system=coord,
                            measurement_datetime_unzoned=measurement,
                            original_T0=raw_date,
                            reference=eeg.get('ref'),
                            bids=eeg_json,
                            demographics=demographics.loc[f'sub-{subject_id}'].to_dict(),
                            target_events=len(targets),
                            exported_cases=n,
                        )
                    )
                    print(f'{name} sub-{subject_id}: {n} {split} cases', flush=True)
                finally:
                    del data
        finally:
            for file in [*handles.values(), *tables.values()]:
                file.close()

        reports = {}
        hashes = {}
        for split in ['TRAIN', 'TEST']:
            reports[split], hashes[split] = validate_split(stage, spec, split)
            print(f'{name}: full {split} readback passed', flush=True)
        if hashes['TRAIN'] & hashes['TEST'] or set(reports['TRAIN']['subjects']) & set(reports['TEST']['subjects']):
            raise ValueError('Signals or participants cross TRAIN/TEST')
        all_rows.sort(key=lambda row: (row['split'] != 'TRAIN', row['row']))
        with (stage / f'{name}_rows.tsv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, FIELDS, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(all_rows)
        aeon_check = smoke_test(stage, spec) if smoke else dict(passed=False, status='not run')
        validation = dict(
            quality_status='passed' if smoke else 'signal validation passed; aeon not run',
            partitions=reports,
            aeon=aeon_check,
            metadata_rows_verified=len(all_rows),
            subject_overlap=[],
            cross_split_duplicate_signal_hashes=0,
            source_channel_order_verified=True,
            source_binary_dtype='little-endian float32',
            one_based_events_verified_against_eeglab=True,
            dropped_events=discarded,
        )
        split_definition = dict(
            unit='participant',
            train=[f'{s:03d}' for s in sorted(train_set)],
            test=[f'{s:03d}' for s in sorted(set(subjects) - train_set)],
            validation='none',
            origin='Fixed whole-subject 30/10 split matching WordRelatedness (ERP CORE N400)',
        )
        metadata = dict(
            name=name,
            source_dataset_name=f'ERP CORE {component}',
            source_aliases=[f'ErpCore2021_{component}', 'ERP-CORE', spec['bids_doi'], 'NEMAR nm000132'],
            target=spec['target'],
            domain='Psy',
            subjects=len(subjects),
            recordings=len(records),
            sessions_per_subject=1,
            source_url='https://osf.io/29xpq/',
            source_doi='10.18115/D5JW4R',
            original_bids_doi=spec['bids_doi'],
            paper_doi='10.1016/j.neuroimage.2020.117465',
            reference=(
                'Kappenman, Farrens, Zhang, Stewart and Luck (2021). '
                'ERP CORE: An open resource for human event-related potential research. NeuroImage 225, 117465.'
            ),
            license='CC-BY-SA-4.0',
            license_url='https://creativecommons.org/licenses/by-sa/4.0/',
            license_evidence='Original LICENSE and https://erpinfo.org/erp-core',
            licence_conflict=(
                'Original BIDS dataset_description says CC0; explicit LICENSE and authors website specify CC BY-SA 4.0; '
                'share-alike retained for this derivative.'
            ),
            shared_cohort_note=(
                'Recorded on the same 40 UC Davis ERP CORE participants as WordRelatedness (N400), '
                'FaceStimuli (N170), AuditoryDeviance (MMN), and SpatialAttention (N2pc), '
                'using distinct experimental paradigms and separate continuous recordings.'
            ),
            frequency=SFREQ,
            sampling_frequency_hz=SFREQ,
            original_sampling_frequency_hz=SFREQ,
            channel_count=30,
            channel_names=CHANNELS,
            channels=CHANNELS,
            sensors=sensor_rows,
            source_channel_count=33,
            excluded_channel_types={'EOG': 3},
            units='microV, original FDT/sidecar scale',
            dtype='float32',
            axis_order=['case', 'channel', 'time'],
            timepoints=POINTS,
            classes=spec['classes'],
            label_map={str(k): v for k, v in spec['label_names'].items()},
            shapes={s.lower(): reports[s]['shape'] for s in reports},
            class_counts={s.lower(): reports[s]['class_counts'] for s in reports},
            window_seconds=1,
            epoch_definition='Zero-based event = original EEGLAB/BIDS sample - 1; start=event-205; end=start+1024 exclusive.',
            preprocessing=dict(
                filtering='none added',
                resampling='none',
                baseline='none',
                referencing='unchanged',
                scaling='none',
                dtype='source float32 retained; .ts 9 significant digits',
            ),
            split_policy=split_definition,
        )
        dump(stage / f'{name}_metadata.json', metadata)
        dump(stage / f'{name}_validation.json', validation)
        dump(stage / 'recordings.json', records)
        dump(stage / 'dropped-events.json', discarded)
        (stage / 'channels.txt').write_text(', '.join(CHANNELS) + '\n', encoding='utf-8')
        (stage / 'channels.tsv').write_text(
            'array_index\tname\ttype\tunits\n'
            + ''.join(f'{i}\t{ch}\tEEG\tmicroV\n' for i, ch in enumerate(CHANNELS)),
            encoding='utf-8',
        )
        for path in cache.iterdir():
            if path.is_file() and path.suffix.lower() not in ['.gdf', '.fdt', '.set']:
                shutil.copy2(path, evidence / path.name)
        shutil.copy2(cache / 'LICENSE', stage / 'LICENSE')
        (stage / 'ATTRIBUTION.txt').write_text(
            'Kappenman, Emily; Farrens, Jaclyn; Zhang, Wendy; Stewart, Andrew X.; Luck, Steven J. (2020/2021). '
            'ERP CORE: An Open Resource for Human Event-Related Potential Research. '
            'Data DOI 10.18115/D5JW4R; paper DOI 10.1016/j.neuroimage.2020.117465.\n'
            'Original LICENSE: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/.\n',
            encoding='utf-8',
        )
        (stage / f'README_{name}.md').write_text(
            f'# {name}\n\nERP CORE {component}: {spec["target"]}.\n'
            f'One-second 30-channel scalp EEG epochs at native 1,024 Hz (CC-BY-SA-4.0).\n',
            encoding='utf-8',
        )
        code_dir = stage / 'conversion_code'
        code_dir.mkdir()
        shutil.copy2(__file__, code_dir / Path(__file__).name)
        manifest_obj = dict(
            name=name,
            status=validation['quality_status'],
            source=dict(
                url=metadata['source_url'],
                data_doi=metadata['source_doi'],
                original_bids_doi=metadata['original_bids_doi'],
                licence='CC-BY-SA-4.0',
                download_manifest_url=MANIFEST_URL,
                download_manifest_sha256=MANIFEST_SHA256,
                files=inventory,
            ),
            moabb_revision=MOABB_REVISION,
            split=split_definition,
            conversion=metadata['preprocessing'],
            validation=validation,
            artifacts={
                p.relative_to(stage).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in stage.rglob('*')
                if p.is_file()
            },
        )
        dump(stage / f'{name}_manifest.json', manifest_obj)
        stage.rename(folder)

    result = dict(
        dataset=name,
        component=component,
        path=str(folder),
        train_cases=counts['TRAIN'],
        test_cases=counts['TEST'],
        classes=spec['classes'],
        channels=30,
        points=POINTS,
        hz=SFREQ,
        subjects=len(subjects),
        status=validation['quality_status'],
    )
    if purge_cache and cache.is_dir():
        shutil.rmtree(cache)
        result['cache_purged'] = True
    return result
