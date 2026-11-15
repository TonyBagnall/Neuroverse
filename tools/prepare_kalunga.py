"""Kalunga2016 / FlickerAttention source audit and deterministic .ts export.

Use through python -m tools.prepare_moabb --dataset kalunga2016. Requires MNE.
"""

from __future__ import annotations

from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from tools.prepare_moabb import MOABB_REVISION, dump, fetch, sha

NAME, RECORD, SFREQ, POINTS = "FlickerAttention", 2392979, 256, 512
CHANNELS = ["Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4"]
LABELS = {1: "rest", 2: "13Hz", 3: "21Hz", 4: "17Hz"}
RUNS = [2] * 6 + [3] + [2] * 2 + [4, 2, 5]
KALUNGA_MV_TO_UV_SCALE = 1e3
KALUNGA_RAW_MV_SD_BOUNDS = (1e-4, 0.5)
AMPLITUDE_SD_BOUNDS_UV = (0.5, 500.0)
AMPLITUDE_EXCURSION_UV = 1000.0
KALUNGA_SCALE_REASON = (
    "Distributed Kalunga2016 FIF/MAT/GDF files store EEG amplitudes in millivolts (mV): "
    "(1) the 16-bit g.tec g.MOBIlab+ front-end has a fixed +-500 uV (+-0.5 mV) ADC range "
    "and subject11 clips at -0.499984771 / +0.499969512 (-32767 / +32766 * 0.5 / 32768 mV); "
    "(2) OpenViBE GDF headers wrote mV values under a hardcoded 'uV' tag while FIF headers "
    "set FIFF_UNIT_V with cal=1.0, range=1.0; and (3) Sylvain Chevallier's reference script "
    "(author-mne-example.txt) explicitly applies 1e3 * epochs.get_data() to convert to microvolts. "
    "Multiplying raw FIF values by 1e3 in float64 converts mV to archive-standard microvolts (uV)."
)
FIELDS = [
    "split",
    "row",
    "example_id",
    "group",
    "participant_id",
    "source_subject_id",
    "session_id",
    "session_date",
    "run_id",
    "loader_session_id",
    "loader_run_id",
    "recording_id",
    "source_file",
    "source_epoch_index",
    "event_sample",
    "start_sample",
    "end_sample",
    "event_time_seconds",
    "start_time_seconds",
    "end_time_seconds",
    "source_label",
    "label",
    "label_description",
    "original_partition",
    "source_mV_float32_sha256",
    "signal_sha256",
    "signal_hash_dtype",
]


def serial(value):
    """Make a lossless-as-available JSON view of native FIF metadata; null nonfinite positions."""
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def header():
    return (
        f"# Kalunga2016 / Zenodo {RECORD}, CC BY 4.0; rescaled x1e3 from mV to uV (float64); see metadata and source audit.\n"
        "# Codes: 1 rest, 2 13Hz, 3 21Hz, 4 17Hz. Native channel order retained.\n"
        f"@problemName {NAME}\n@timestamps false\n@missing false\n@univariate false\n"
        "@dimensions 8\n@equalLength true\n@seriesLength 512\n@classLabel true 1 2 3 4\n@data\n"
    )


def signal_hash(array, dtype="<f8"):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def format_case(array):
    """Shortest exact IEEE-754 binary64 decimal representation per sample."""
    return ":".join(",".join(map(repr, channel.tolist())) for channel in array)


def onset_events(raw):
    import mne

    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose="ERROR")
    if len(events) != 32 or Counter(events[:, 2]) != Counter({1: 8, 2: 8, 3: 8, 4: 8}):
        raise ValueError("Unexpected Kalunga event codes or class counts")
    return events


def audit_sources(cache, download=True):
    """Audit all release aliases and compare actual data, not only file checksums."""
    import mne

    cache = Path(cache)
    record_path = cache / f"zenodo-record-{RECORD}.json"
    if download:
        fetch(f"https://zenodo.org/api/records/{RECORD}", record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["id"] != RECORD or record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected source version or data licence")
    url = f"https://raw.githubusercontent.com/NeuroTechX/moabb/{MOABB_REVISION}/moabb/datasets/ssvep_exo.py"
    if download:
        fetch(url, cache / "moabb-kalunga-loader.py")
    if not (cache / "moabb-kalunga-loader.py").is_file():
        raise ValueError("Missing pinned loader evidence")
    rows, canonical = [], {}
    for item in sorted(record["files"], key=lambda f: f["key"]):
        name = item["key"]
        match = re.fullmatch(r"subject(\d{2})_run(\d+)_raw\.fif", name)
        if not match:
            raise ValueError(f"Unexpected release member: {name}")
        subject, run = map(int, match.groups())
        path = cache / name
        if download:
            fetch(item["links"]["self"], path, item["checksum"], item["size"])
        if not path.is_file() or path.stat().st_size != item["size"] or "md5:" + sha(path, "md5") != item["checksum"]:
            raise ValueError(f"Missing or modified source: {name}")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        if (
            raw.ch_names != CHANNELS + ["STI 014"]
            or raw.get_channel_types() != ["eeg"] * 8 + ["stim"]
            or raw.info["sfreq"] != SFREQ
        ):
            raise ValueError("Inconsistent actual channel order/types/frequency")
        data = raw.get_data()
        if not np.isfinite(data).all():
            raise ValueError("Missing or nonfinite source values")
        hashed = hashlib.sha256(np.asarray(data, dtype="<f8").tobytes()).hexdigest()
        selected = run <= RUNS[subject - 1]
        embedded = int(raw.info["subject_info"]["id"])
        if selected and embedded != subject:
            raise ValueError("Selected source filename disagrees with embedded participant ID")
        reason = "not selected by pinned MOABB run inventory" if not selected else "retained"
        keep = selected
        duplicate_of = None
        if selected:
            if hashed in canonical:
                duplicate_of = canonical[hashed]
                keep = False
                reason = "exact EEG and stimulus duplicate despite different participant headers"
            else:
                canonical[hashed] = name
        events = onset_events(raw)
        row = dict(
            file=name,
            filename_subject=f"{subject:02d}",
            filename_run=str(run),
            embedded_subject_id=str(embedded),
            source_subject_info=serial(dict(raw.info["subject_info"])),
            selected_by_moabb=selected,
            retained=keep,
            decision=reason,
            duplicate_of=duplicate_of,
            source_md5=item["checksum"],
            source_sha256=sha(path),
            source_url=item["links"]["self"],
            signal_and_stim_sha256=hashed,
            samples=int(raw.n_times),
            first_sample=int(raw.first_samp),
            measurement_datetime_utc=serial(raw.info["meas_date"]),
            channel_names=raw.ch_names,
            channel_types=raw.get_channel_types(),
            frequency=SFREQ,
            event_count=len(events),
            class_counts={str(k): int(v) for k, v in sorted(Counter(events[:, 2]).items())},
            channels=serial(raw.info["chs"]),
            bad_channels=raw.info["bads"],
            highpass_tag_hz=raw.info["highpass"],
            lowpass_tag_hz=raw.info["lowpass"],
            custom_reference_applied=bool(raw.info["custom_ref_applied"]),
            line_frequency_hz=raw.info["line_freq"],
            experimenter=raw.info["experimenter"],
            description=raw.info["description"],
            projections=serial(raw.info["projs"]),
            annotations=serial(
                [
                    dict(onset=o, duration=d, description=s)
                    for o, d, s in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
                ]
            ),
        )
        rows.append(row)
    groups = {}
    for row in rows:
        groups.setdefault(row["signal_and_stim_sha256"], []).append(row["file"])
    audit = dict(
        record=RECORD,
        moabb_revision=MOABB_REVISION,
        mne_version=mne.__version__,
        files=rows,
        signal_duplicate_groups=[names for names in groups.values() if len(names) > 1],
        selection="Pinned MOABB inventory, then retain only the first sorted selected filename for each identical EEG-plus-stimulus array.",
        attribution_caveat="A canonical duplicate filename does not resolve the true attribution of the repeated recording; all aliases and embedded subject IDs are retained here.",
    )
    dump(cache / "source-file-audit.json", audit)
    return record, audit


def ts_lines(path):
    with Path(path).open(encoding="ascii") as file:
        for line in file:
            if line.strip().lower() == "@data":
                break
        else:
            raise ValueError("Missing @data marker")
        yield from file


def validate_split(folder, split):
    count = 0
    counts, people, hashes, recordings = Counter(), set(), set(), set()
    channel_sds = []
    case_max_abs = []
    with (folder / f"cases_{split.lower()}.tsv").open(newline="", encoding="utf-8") as file:
        rows = iter(csv.DictReader(file, delimiter="\t"))
        for count, line in enumerate(ts_lines(folder / f"{NAME}_{split}.ts"), 1):
            row = next(rows, None)
            if row is None or row["split"] != split or int(row["row"]) != count - 1:
                raise ValueError("Case metadata row alignment failed")
            parts = line.strip().split(":")
            if len(parts) != 9:
                raise ValueError("Incorrect exported channel count")
            array = np.stack([np.fromstring(ch, dtype="<f8", sep=",") for ch in parts[:-1]])
            label = int(parts[-1])
            hashed = signal_hash(array, dtype="<f8")
            if (
                array.shape != (8, POINTS)
                or not np.isfinite(array).all()
                or hashed != row["signal_sha256"]
                or row["signal_hash_dtype"] != "<f8"
                or str(label) != row["label"]
                or str(label) != row["source_label"]
                or row["label_description"] != LABELS[label]
            ):
                raise ValueError("Exported signal or label differs from source case metadata")
            if int(row["start_sample"]) != int(row["event_sample"]) + 512 or int(row["end_sample"]) != int(
                row["event_sample"]
            ) + 1024:
                raise ValueError("Incorrect epoch sample bounds")
            channel_sds.append(array.std(axis=1, ddof=0))
            case_max_abs.append(float(np.max(np.abs(array))))
            people.add(row["participant_id"])
            recordings.add(row["recording_id"])
            counts[label] += 1
            hashes.add(hashed)
        if next(rows, None) is not None:
            raise ValueError("Extra case metadata rows")
    all_sds = np.concatenate(channel_sds)
    median_sd = float(np.median(all_sds))
    p05_sd = float(np.quantile(all_sds, 0.05))
    p95_sd = float(np.quantile(all_sds, 0.95))
    max_abs = float(np.max(case_max_abs))
    gt_1000 = int(sum(v > AMPLITUDE_EXCURSION_UV for v in case_max_abs))
    lo_sd, hi_sd = AMPLITUDE_SD_BOUNDS_UV
    if not (lo_sd <= median_sd <= hi_sd):
        raise ValueError(
            f"Amplitude scale check failed on {split}: median channel SD {median_sd:.6g} uV outside [{lo_sd}, {hi_sd}] uV"
        )
    id_lines = (folder / f"{NAME}_id_{split}.txt").read_text(encoding="ascii").splitlines()
    if len(id_lines) != count:
        raise ValueError(f"Participant ID sidecar length mismatch on {split}")
    return (
        dict(
            cases=count,
            shape=[count, 8, POINTS],
            class_counts=dict(sorted(counts.items())),
            subjects=sorted(people),
            recordings=sorted(recordings),
            exact_float64_roundtrip=True,
            metadata_rows_verified=True,
            median_channel_sd_uV=median_sd,
            p05_channel_sd_uV=p05_sd,
            p95_channel_sd_uV=p95_sd,
            max_abs_uV=max_abs,
            cases_max_abs_gt_1000_uV=gt_1000,
            fraction_cases_max_abs_gt_1000_uV=gt_1000 / count if count else 0.0,
        ),
        hashes,
    )


def smoke_test(folder):
    import aeon
    from aeon.classification.feature_based import SummaryClassifier
    from aeon.datasets import load_from_ts_file

    loaded = {}
    with tempfile.TemporaryDirectory(dir=folder, prefix="aeon-check-") as tmp:
        for split in ("TRAIN", "TEST"):
            examples = {label: [] for label in LABELS}
            for line in ts_lines(folder / f"{NAME}_{split}.ts"):
                label = int(line.rsplit(":", 1)[1])
                if len(examples[label]) < 2:
                    examples[label].append(line)
                if all(len(items) == 2 for items in examples.values()):
                    break
            path = Path(tmp) / f"{split}.ts"
            path.write_text(
                header() + "".join(line for items in examples.values() for line in items), encoding="ascii"
            )
            loaded[split] = load_from_ts_file(path)
        model = SummaryClassifier(random_state=0, n_jobs=1)
        model.fit(*loaded["TRAIN"])
        predicted = model.predict(loaded["TEST"][0])
        if len(predicted) != len(loaded["TEST"][1]):
            raise ValueError("aeon compatibility failed")
    return dict(
        passed=True,
        aeon_version=aeon.__version__,
        classifier="SummaryClassifier",
        train_cases=len(loaded["TRAIN"][1]),
        test_cases=len(predicted),
        note="Representative four-class compatibility check, not benchmark results.",
    )


def convert(
    cache,
    output,
    subjects=tuple(range(1, 13)),
    train=tuple(range(1, 10)),
    download=True,
    smoke=True,
    overwrite=False,
):
    import mne

    cache, output = Path(cache).resolve(), Path(output).resolve()
    subjects, train = tuple(sorted(subjects)), set(train)
    if (
        len(subjects) != len(set(subjects))
        or not set(subjects) <= set(range(1, 13))
        or not train
        or not train < set(subjects)
    ):
        raise ValueError("Invalid participant split")
    folder = output / NAME
    if folder.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {folder}")
    if smoke:
        from aeon.classification.feature_based import SummaryClassifier  # noqa: F401
        from aeon.datasets import load_from_ts_file  # noqa: F401
    record, audit = audit_sources(cache, download)
    selected = [row for row in audit["files"] if row["retained"] and int(row["filename_subject"]) in subjects]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{NAME}-", dir=output) as tmp:
        stage = Path(tmp)
        counts = Counter()
        tables = {}
        handles = {}
        writers = {}
        id_lists = {"TRAIN": [], "TEST": []}
        all_rows = []
        recordings = []
        dropped = []
        max_source_f32_diff = 0.0
        info_dir = stage / "source_metadata"
        info_dir.mkdir()
        try:
            for split in ("TRAIN", "TEST"):
                handles[split] = (stage / f"{NAME}_{split}.ts").open("w", encoding="ascii", newline="\n")
                handles[split].write(header())
                tables[split] = (stage / f"cases_{split.lower()}.tsv").open("w", encoding="utf-8", newline="")
                writers[split] = csv.DictWriter(tables[split], FIELDS, delimiter="\t", lineterminator="\n")
                writers[split].writeheader()
            for item in selected:
                item = dict(item)
                filename = item["file"]
                subject = int(item["filename_subject"])
                run = item["filename_run"]
                raw = mne.io.read_raw_fif(cache / filename, preload=True, verbose="ERROR")
                mne.io.write_info(info_dir / filename.replace("_raw.fif", "_info.fif"), raw.info, verbose="ERROR")
                raw_mV = np.ascontiguousarray(raw.get_data(picks="eeg"), dtype="<f8")
                raw_median_sd_mV = float(np.median(raw_mV.std(axis=1, ddof=0)))
                lo_mv, hi_mv = KALUNGA_RAW_MV_SD_BOUNDS
                if not (lo_mv <= raw_median_sd_mV <= hi_mv):
                    raise ValueError(
                        f"Expected raw millivolt FIF scale in {filename} with median channel SD in "
                        f"[{lo_mv}, {hi_mv}] mV, got {raw_median_sd_mV:.6g}; review before applying {KALUNGA_MV_TO_UV_SCALE:g}"
                    )
                max_source_f32_diff = max(
                    max_source_f32_diff,
                    float(np.max(np.abs(raw_mV - raw_mV.astype("<f4").astype("<f8")))),
                )
                data_uV = np.ascontiguousarray(raw_mV * KALUNGA_MV_TO_UV_SCALE, dtype="<f8")
                events = onset_events(raw)
                split = "TRAIN" if subject in train else "TEST"
                for epoch_index, event in enumerate(events):
                    sample = int(event[0])
                    start = sample - int(raw.first_samp) + 512
                    end = start + POINTS
                    label = int(event[2])
                    if start < 0 or end > raw.n_times:
                        dropped.append(
                            dict(source_file=filename, event_index=epoch_index, reason="incomplete [2,4) second epoch")
                        )
                        continue
                    source_case_f32 = np.asarray(raw_mV[:, start:end], dtype="<f4")
                    case_uV = np.ascontiguousarray(data_uV[:, start:end], dtype="<f8")
                    pid = f"{subject:02d}"
                    row = dict(
                        split=split,
                        row=counts[split],
                        example_id=f"zenodo:{RECORD}/{filename}/event-{epoch_index}",
                        group=pid,
                        participant_id=pid,
                        source_subject_id=item["source_subject_info"].get("his_id", ""),
                        session_id=item["measurement_datetime_utc"],
                        session_date=item["measurement_datetime_utc"][:10],
                        run_id=run,
                        loader_session_id="0",
                        loader_run_id=str(int(run) - 1),
                        recording_id=filename.removesuffix("_raw.fif"),
                        source_file=filename,
                        source_epoch_index=epoch_index,
                        event_sample=sample,
                        start_sample=sample + 512,
                        end_sample=sample + 1024,
                        event_time_seconds=(sample - int(raw.first_samp)) / SFREQ,
                        start_time_seconds=start / SFREQ,
                        end_time_seconds=end / SFREQ,
                        source_label=label,
                        label=label,
                        label_description=LABELS[label],
                        original_partition="unpartitioned",
                        source_mV_float32_sha256=signal_hash(source_case_f32, dtype="<f4"),
                        signal_sha256=signal_hash(case_uV, dtype="<f8"),
                        signal_hash_dtype="<f8",
                    )
                    writers[split].writerow(row)
                    all_rows.append(row)
                    id_lists[split].append(pid)
                    handles[split].write(f"{format_case(case_uV)}:{label}\n")
                    counts[split] += 1
                item.update(
                    raw_median_channel_sd_mV=raw_median_sd_mV,
                    rescaled_median_channel_sd_uV=float(np.median(data_uV.std(axis=1, ddof=0))),
                    eeg_scale_factor_applied=KALUNGA_MV_TO_UV_SCALE,
                )
                recordings.append(item)
                print(f"{filename}: {split}, {len(events)} events", flush=True)
        finally:
            for file in [*handles.values(), *tables.values()]:
                file.close()
        for split in ("TRAIN", "TEST"):
            with (stage / f"{NAME}_id_{split}.txt").open("wb") as id_f:
                id_f.write(("\r\n".join(id_lists[split]) + "\r\n").encode("ascii"))
        reports = {}
        hashes = {}
        for split in ("TRAIN", "TEST"):
            reports[split], hashes[split] = validate_split(stage, split)
        if hashes["TRAIN"] & hashes["TEST"] or set(reports["TRAIN"]["subjects"]) & set(reports["TEST"]["subjects"]):
            raise ValueError("Signals or subjects cross TRAIN/TEST")
        if set(reports["TRAIN"]["subjects"]) != {f"{s:02d}" for s in train} or set(reports["TEST"]["subjects"]) != {
            f"{s:02d}" for s in set(subjects) - train
        }:
            raise ValueError("Participant split mismatch")
        if len({row["signal_sha256"] for row in all_rows}) != len(all_rows):
            raise ValueError("Duplicate signal cases remain in the selected collection")
        all_rows.sort(key=lambda r: (r["split"] != "TRAIN", r["row"]))
        with (stage / f"{NAME}_rows.tsv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)
        aeon_check = smoke_test(stage) if smoke else dict(passed=False, status="not run")
        validation = dict(
            quality_status="passed" if smoke else "signal validation passed; aeon not run",
            partitions=reports,
            aeon=aeon_check,
            metadata_rows_verified=len(all_rows),
            subject_overlap=[],
            cross_split_duplicate_signal_hashes=0,
            duplicate_signal_cases_remaining=0,
            dropped_events=dropped,
            channel_order_verified=True,
            eeg_scale_factor_applied=KALUNGA_MV_TO_UV_SCALE,
            max_source_fif_float32_vs_float64_diff_mV=max_source_f32_diff,
        )
        split_definition = dict(
            unit="participant",
            train=[f"{s:02d}" for s in sorted(train)],
            test=[f"{s:02d}" for s in sorted(set(subjects) - train)],
            validation="none",
            origin="EEGBench-defined; no universal MOABB fixed split",
        )
        sensors = [
            dict(
                array_index=i,
                name=name,
                type="EEG",
                units="uV",
                source_units="mV",
                unit_tag_fiff=int(recordings[0]["channels"][i]["unit"]),
                raw_fif_loc=recordings[0]["channels"][i]["loc"],
                raw_fif_coord_frame=recordings[0]["channels"][i]["coord_frame"],
                position_metres_verified=None,
            )
            for i, name in enumerate(CHANNELS)
        ]
        metadata = dict(
            name=NAME,
            source_dataset_name="Kalunga2016",
            source_aliases=["SSVEP-exoskeleton", "SSVEP Exo", "SSVEPEXO", "nm000123"],
            domain="BCI",
            target="Attend 13, 17 or 21 Hz flicker, or rest",
            subjects=len(subjects),
            distributed_participant_ids=[f"{s:02d}" for s in subjects],
            recordings=len(recordings),
            recording_sessions=len(recordings),
            sessions=None,
            session_definition="session_id is the recovered measurement datetime used as a recording-session identifier; physical session labels are not supplied. See separate loader 0/run identifiers.",
            distinct_measurement_dates=len({r["measurement_datetime_utc"][:10] for r in recordings}),
            original_subject_ids={r["filename_subject"]: r["source_subject_info"] for r in recordings},
            source_url=f"https://zenodo.org/records/{RECORD}",
            source_doi=f"10.5281/zenodo.{RECORD}",
            source_version="v1",
            source_concept_doi="10.5281/zenodo.2392978",
            paper_doi="10.1016/j.neucom.2016.01.007",
            paper_url="https://arxiv.org/abs/1501.03227",
            reference="Kalunga et al. (2016), Online SSVEP-based BCI using Riemannian Geometry, Neurocomputing.",
            authors=record["metadata"]["creators"],
            license="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            license_evidence=f"https://zenodo.org/api/records/{RECORD}",
            frequency=SFREQ,
            sampling_frequency_hz=SFREQ,
            original_sampling_frequency_hz=SFREQ,
            channels=CHANNELS,
            channel_names=CHANNELS,
            sensors=sensors,
            channel_count=8,
            timepoints=POINTS,
            classes=4,
            label_map={str(k): v for k, v in LABELS.items()},
            label_correction="Code 3 = 21Hz, code 4 = 17Hz, per original record and pinned MOABB correction; numeric source codes unchanged.",
            dtype="float64",
            axis_order=["case", "channel", "time"],
            shapes={s.lower(): reports[s]["shape"] for s in reports},
            class_counts={s.lower(): reports[s]["class_counts"] for s in reports},
            window_seconds=2,
            epoch_interval="[2,4) seconds after FIF stimulus marker, 512 samples; source end_sample is exclusive",
            preprocessing=dict(
                filtering="none added",
                resampling="none",
                baseline="none",
                referencing="unchanged",
                scaling="x1e3 from source millivolts (mV) to microvolts (uV) in float64",
                scale_factor=KALUNGA_MV_TO_UV_SCALE,
                scale_reason=KALUNGA_SCALE_REASON,
                dtype="float64 exact binary64 decimal representation in .ts",
            ),
            units="uV (microvolts; rescaled x1e3 in float64 from verified source millivolts)",
            unit_provenance=KALUNGA_SCALE_REASON,
            positions_warning="FIF raw loc vectors have magnitudes around 100 despite MNE metre convention, consistent with an unverified millimetre/metre discrepancy. Raw vectors preserved; no verified metre montage asserted.",
            channel_reference="Custom reference flagged by FIF. Original publication says right (or left) mastoid; side not resolved. No re-referencing.",
            channel_order="Verified identical in every released FIF; actual PO3/POz/PO7/PO8/PO4 order retained rather than summary order.",
            split_policy=split_definition,
            case_metadata=dict(
                file=f"{NAME}_rows.tsv",
                ordering="TRAIN then TEST; row zero-based within split; IDs strings",
            ),
            source_selection=audit["selection"],
            source_attribution_caveat=audit["attribution_caveat"],
            missing_metadata=[
                "verified sensor coordinates in metres",
                "original session labels",
                "reference laterality",
                "true subject ownership of duplicate 03/07 recording",
            ],
            metadata_evidence=dict(
                distributed_files=[
                    "channel names/types/order",
                    "subject_info IDs",
                    "measurement timestamps",
                    "marker codes",
                    "sample indices",
                    "raw loc/calibration/reference/filter tags",
                    "16-bit g.MOBIlab+ +-0.5 mV ADC rail saturation confirming millivolt numeric scale",
                ],
                source_publication=[
                    "class frequency definitions and correction",
                    "g.MOBIlab+ hardware and mastoid-reference description",
                    "author MNE reference script applying 1e3 * epochs.get_data() to obtain microvolts",
                    "licence",
                ],
                local_definitions=[
                    "fixed subject split",
                    "x1e3 float64 conversion from source millivolts to microvolts",
                    "recording timestamps used as session identifiers",
                    "duplicate canonicalization",
                    "half-open [2,4) second epochs",
                ],
            ),
        )
        dump(stage / f"{NAME}_metadata.json", metadata)
        dump(stage / f"{NAME}_validation.json", validation)
        dump(stage / "source-file-audit.json", audit)
        dump(stage / "recordings.json", recordings)
        dump(stage / "dropped-events.json", dropped)
        dump(
            stage / "metadata-completeness.json",
            dict(evidence=metadata["metadata_evidence"], missing=metadata["missing_metadata"]),
        )
        (stage / "channels.txt").write_text(", ".join(CHANNELS) + "\n", encoding="utf-8")
        (stage / "channels.tsv").write_text(
            "array_index\tname\ttype\tunits\n" + "".join(f"{i}\t{n}\tEEG\tuV\n" for i, n in enumerate(CHANNELS)),
            encoding="utf-8",
        )
        (stage / "ATTRIBUTION.txt").write_text(
            "Chevallier, Sylvain and Kalunga, Emmanuel (2015). SSVEP-exoskeleton, v1. DOI 10.5281/zenodo.2392979. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n"
            "Reference: Kalunga et al. (2016), Online SSVEP-based BCI using Riemannian Geometry, DOI 10.1016/j.neucom.2016.01.007.\n"
            "EEGBench changes: documented duplicate exclusions, subject split, two-second epochs, x1e3 rescaling from source millivolts to float64 microvolts, and .ts conversion. Channel order and source numeric labels retained.\n",
            encoding="utf-8",
        )
        (stage / f"README_{NAME}.md").write_text(
            f"# {NAME}\n\n"
            "Kalunga2016: four classes, 1 rest, 2 13Hz, 3 21Hz, 4 17Hz. Epochs are [2,4) seconds after the FIF stimulus marker, at 256Hz, with eight EEG channels in microvolts (uV).\n\n"
            "Read IDs as strings. Per-split TSV rows match cases exactly; combined rows are TRAIN then TEST. Sample bounds include the original raw first_samp and end is exclusive. session_id uses measurement datetime as a recording identifier; loader session/run fields are separate.\n\n"
            "Source-file audit documents duplicate filename aliases, the extra numeric duplicate 03/07, and canonical exclusions. Original participant attribution of that duplicate remains uncertain. Distributed FIF millivolt (mV) values are multiplied by 1e3 in float64 to archive-standard microvolts (uV), verified by g.MOBIlab+ +-0.5 mV ADC rail saturation and author reference code; raw sensor loc vectors are preserved with position warnings, and no filtering, resampling or reference changes are added. See metadata, source audit, original info FIFs, validation and manifest. No benchmark scores are included.\n",
            encoding="utf-8",
        )
        for filename in [f"zenodo-record-{RECORD}.json", "moabb-kalunga-loader.py", "author-mne-example.txt"]:
            if (cache / filename).exists():
                shutil.copy2(cache / filename, info_dir / filename)
        code = stage / "conversion_code"
        code.mkdir()
        shutil.copy2(__file__, code / Path(__file__).name)
        from tools import prepare_moabb

        shutil.copy2(prepare_moabb.__file__, code / "prepare_moabb.py")
        manifest = dict(
            name=NAME,
            status=validation["quality_status"],
            source=dict(
                record=RECORD,
                version="v1",
                doi=metadata["source_doi"],
                licence="CC-BY-4.0",
                record_sha256=sha(cache / f"zenodo-record-{RECORD}.json"),
            ),
            mne_version=mne.__version__,
            moabb_revision=MOABB_REVISION,
            split=split_definition,
            conversion=metadata["preprocessing"],
            validation=validation,
            source_files=[
                {k: r[k] for k in ["file", "source_sha256", "source_md5", "source_url", "retained", "decision", "duplicate_of"]}
                for r in audit["files"]
            ],
            artifacts={
                p.relative_to(stage).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in stage.rglob("*")
                if p.is_file()
            },
        )
        dump(stage / f"{NAME}_manifest.json", manifest)
        if folder.exists():
            backup = output / f".{NAME}-prev-backup"
            if backup.exists():
                shutil.rmtree(backup)
            folder.rename(backup)
            try:
                stage.rename(folder)
            except Exception:
                if not folder.exists() and backup.exists():
                    backup.rename(folder)
                raise
            shutil.rmtree(backup)
        else:
            stage.rename(folder)
    print(
        json.dumps(
            dict(
                dataset=NAME,
                output=str(folder),
                train=counts["TRAIN"],
                test=counts["TEST"],
                validation=validation["quality_status"],
            )
        ),
        flush=True,
    )
    return metadata, validation
