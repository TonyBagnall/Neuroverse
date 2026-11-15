"""Check FIF metadata recovery, duplicate aliases, mV->uV rescaling, timing and corrected labels."""

import csv
import datetime
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from tools.prepare_kalunga import (
    CHANNELS,
    KALUNGA_MV_TO_UV_SCALE,
    LABELS,
    NAME,
    POINTS,
    RECORD,
    audit_sources,
    convert,
)


def fixture_source(root: Path, subjects=(1, 2), duplicate=False, wrong_channels=False, sd_mv=0.007):
    import mne

    cache = root / "source"
    cache.mkdir(parents=True, exist_ok=True)
    n_channels = len(CHANNELS) + 1
    n_samples = 42000
    n_events = 32
    for subject in subjects:
        rng = np.random.default_rng(1 if duplicate else subject)
        data = rng.normal(0.0, sd_mv, (n_channels, n_samples)).astype("<f4").astype("<f8")
        data[-1] = 0.0
        for i in range(n_events):
            data[-1, 256 + i * 1280] = i % len(LABELS) + 1
        names = CHANNELS.copy()
        if wrong_channels:
            names[3], names[4] = names[4], names[3]
        info = mne.create_info(names + ["STI 014"], 256, ["eeg"] * len(CHANNELS) + ["stim"])
        info["subject_info"] = dict(id=subject, his_id=f"S{subject}")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_meas_date(datetime.datetime(2015, 1, subject, tzinfo=datetime.timezone.utc))
        raw.save(cache / f"subject{subject:02d}_run1_raw.fif", overwrite=True, verbose="ERROR")
    files = [
        dict(
            key=p.name,
            size=p.stat().st_size,
            checksum="md5:" + hashlib.md5(p.read_bytes()).hexdigest(),
            links={"self": "https://example.invalid/" + p.name},
        )
        for p in cache.glob("*.fif")
    ]
    (cache / f"zenodo-record-{RECORD}.json").write_text(
        json.dumps(dict(id=RECORD, metadata=dict(license={"id": "cc-by-4.0"}, creators=[]), files=files)),
        encoding="utf-8",
    )
    (cache / "moabb-kalunga-loader.py").write_text("# Test evidence\n", encoding="utf-8")
    return cache


def test_full_case_alignment_mv_to_uv_rescaling_and_timebounds():
    """convert rescales mV FIF recordings by 1e3 to float64 uV while preserving case order and labels."""
    import mne
    from aeon.datasets import load_from_ts_file

    expected_cases_per_subject = 32
    raw_sd_mv = 0.007
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = fixture_source(root, subjects=(1, 2), sd_mv=raw_sd_mv)
        out = root / "out"
        metadata, validation = convert(cache, out, subjects=(1, 2), train=(1,), download=False, smoke=False)

        assert metadata["shapes"] == {
            "train": [expected_cases_per_subject, len(CHANNELS), POINTS],
            "test": [expected_cases_per_subject, len(CHANNELS), POINTS],
        }
        assert metadata["channel_names"] == CHANNELS
        assert metadata["dtype"] == "float64"
        assert metadata["preprocessing"]["scale_factor"] == KALUNGA_MV_TO_UV_SCALE
        assert metadata["label_map"] == {"1": "rest", "2": "13Hz", "3": "21Hz", "4": "17Hz"}
        assert validation["duplicate_signal_cases_remaining"] == 0

        raw_sub1 = mne.io.read_raw_fif(cache / "subject01_run1_raw.fif", preload=True, verbose="ERROR")
        expected_first_epoch_uv = raw_sub1.get_data(picks="eeg")[:, 768:1280] * KALUNGA_MV_TO_UV_SCALE
        x_train, _ = load_from_ts_file(str(out / NAME / f"{NAME}_TRAIN.ts"))
        np.testing.assert_array_equal(x_train[0], expected_first_epoch_uv)
        assert 5.0 <= validation["partitions"]["TRAIN"]["median_channel_sd_uV"] <= 10.0

        with (out / NAME / "cases_train.tsv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        assert rows[0]["participant_id"] == "01" and rows[0]["source_subject_id"] == "S1"
        assert rows[0]["event_sample"] == "256" and rows[0]["start_sample"] == "768" and rows[0]["end_sample"] == "1280"
        assert rows[2]["label_description"] == LABELS[3] == "21Hz"
        assert rows[0]["session_id"].startswith("2015-01-01")
        assert rows[0]["loader_session_id"] == "0"
        assert rows[0]["signal_hash_dtype"] == "<f8"
        assert (out / NAME / f"{NAME}_id_TRAIN.txt").read_text(encoding="ascii").splitlines() == [
            "01"
        ] * expected_cases_per_subject


def test_mv_scale_guard_rejects_already_microvolt_input():
    """convert refuses to multiply by 1e3 when the source FIF is not in the expected mV range."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = fixture_source(root, subjects=(1, 2), sd_mv=7.0)
        with pytest.raises(ValueError, match="millivolt"):
            convert(cache, root / "out", subjects=(1, 2), train=(1,), download=False, smoke=False)


def test_numeric_duplicate_with_different_participant_headers_is_flagged():
    """Exact EEG+stimulus duplicates with different subject headers retain only the first canonical file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, audit = audit_sources(fixture_source(root, subjects=(3, 7), duplicate=True), download=False)
        assert len(audit["signal_duplicate_groups"]) == 1
        assert [r["file"] for r in audit["files"] if r["retained"]] == ["subject03_run1_raw.fif"]
        assert audit["files"][1]["duplicate_of"] == "subject03_run1_raw.fif"


def test_changed_fif_sensor_order_rejected():
    """Changed channel ordering in source FIF files raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError, match="channel order"):
            audit_sources(fixture_source(root, wrong_channels=True), download=False)


def test_modified_fif_fails_publisher_checksum():
    """Corrupted FIF bytes fail publisher MD5 verification."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = fixture_source(root)
        with (cache / "subject01_run1_raw.fif").open("ab") as file:
            file.write(b"modified")
        with pytest.raises(ValueError, match="modified source"):
            audit_sources(cache, download=False)
