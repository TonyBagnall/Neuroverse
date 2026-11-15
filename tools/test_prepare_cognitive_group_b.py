"""Unit tests for Cognitive Group B (+15 scalp-EEG datasets) decoders and extractors."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from tools.prepare_cognitive_group_b import (
    create_junction_if_needed,
    decode_bdf,
    decode_edf,
    extract_internal_attention_windows,
    extract_music_clip_windows,
    extract_python_code_epochs,
    resolve_storage_destination,
)
from tools.prepare_erpcore_batch2 import target_rows


def _pad_ascii(text: str, width: int) -> bytes:
    encoded = text.encode('latin-1')
    assert len(encoded) <= width
    return encoded + b' ' * (width - len(encoded))


def _write_synthetic_edf(
    path: Path,
    ch_names: list[str],
    phys_dims: list[str],
    phys_mins: list[float],
    phys_maxs: list[float],
    dig_mins: list[int],
    dig_maxs: list[int],
    spr: list[int],
    rec_dur: float,
    digital_records: np.ndarray,
) -> None:
    ns = len(ch_names)
    n_records = int(digital_records.shape[0])
    header_bytes = 256 * (ns + 1)
    hdr = bytearray(b' ' * 256)
    hdr[0:8] = _pad_ascii('0', 8)
    hdr[184:192] = _pad_ascii(str(header_bytes), 8)
    hdr[236:244] = _pad_ascii(str(n_records), 8)
    hdr[244:252] = _pad_ascii(f'{rec_dur:.4f}', 8)
    hdr[252:256] = _pad_ascii(str(ns), 4)

    sig_hdr = bytearray()
    sig_hdr.extend(b''.join(_pad_ascii(c, 16) for c in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii('', 80) for _ in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii(u, 8) for u in phys_dims))
    sig_hdr.extend(b''.join(_pad_ascii(f'{v:.8g}', 8) for v in phys_mins))
    sig_hdr.extend(b''.join(_pad_ascii(f'{v:.8g}', 8) for v in phys_maxs))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in dig_mins))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in dig_maxs))
    sig_hdr.extend(b''.join(_pad_ascii('', 80) for _ in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in spr))
    sig_hdr.extend(b''.join(_pad_ascii('', 32) for _ in ch_names))

    body = np.asarray(digital_records, dtype='<i2').tobytes()
    path.write_bytes(bytes(hdr) + bytes(sig_hdr) + body)


def _write_synthetic_bdf(
    path: Path,
    ch_names: list[str],
    phys_dims: list[str],
    phys_mins: list[float],
    phys_maxs: list[float],
    dig_mins: list[int],
    dig_maxs: list[int],
    spr: list[int],
    rec_dur: float,
    digital_records_i32: np.ndarray,
) -> None:
    ns = len(ch_names)
    n_records = int(digital_records_i32.shape[0])
    header_bytes = 256 * (ns + 1)
    hdr = bytearray(b' ' * 256)
    hdr[0] = 0xFF
    hdr[1:8] = b'BIOSEMI'
    hdr[184:192] = _pad_ascii(str(header_bytes), 8)
    hdr[236:244] = _pad_ascii(str(n_records), 8)
    hdr[244:252] = _pad_ascii(f'{rec_dur:.4f}', 8)
    hdr[252:256] = _pad_ascii(str(ns), 4)

    sig_hdr = bytearray()
    sig_hdr.extend(b''.join(_pad_ascii(c, 16) for c in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii('', 80) for _ in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii(u, 8) for u in phys_dims))
    sig_hdr.extend(b''.join(_pad_ascii(f'{v:.8g}', 8) for v in phys_mins))
    sig_hdr.extend(b''.join(_pad_ascii(f'{v:.8g}', 8) for v in phys_maxs))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in dig_mins))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in dig_maxs))
    sig_hdr.extend(b''.join(_pad_ascii('', 80) for _ in ch_names))
    sig_hdr.extend(b''.join(_pad_ascii(str(v), 8) for v in spr))
    sig_hdr.extend(b''.join(_pad_ascii('', 32) for _ in ch_names))

    flat_i32 = np.asarray(digital_records_i32, dtype='<i4').ravel()
    triplets = np.empty((flat_i32.size, 3), dtype=np.uint8)
    triplets[:, 0] = flat_i32 & 0xFF
    triplets[:, 1] = (flat_i32 >> 8) & 0xFF
    triplets[:, 2] = (flat_i32 >> 16) & 0xFF
    path.write_bytes(bytes(hdr) + bytes(sig_hdr) + triplets.tobytes())


def test_decode_edf_calibrates_physical_units_and_picks_scalp_channels():
    """Verify EDF linear calibration, volt-to-microvolt scaling, and channel selection."""
    n_records = 3
    spr_eeg = 4
    ch_names = ['Fp1', 'Cz', 'Status']
    phys_dims = ['uV', 'V', 'uV']
    phys_mins = [-100.0, -0.001, 0.0]
    phys_maxs = [100.0, 0.001, 100.0]
    dig_mins = [-1000, -1000, 0]
    dig_maxs = [1000, 1000, 1000]
    spr = [spr_eeg, spr_eeg, 2]

    rng = np.random.default_rng(42)
    dig_fp1 = rng.integers(-800, 800, size=(n_records, spr_eeg), dtype='<i2')
    dig_cz = rng.integers(-500, 500, size=(n_records, spr_eeg), dtype='<i2')
    dig_trig = np.zeros((n_records, 2), dtype='<i2')
    records = np.concatenate([dig_fp1, dig_cz, dig_trig], axis=1)

    with tempfile.TemporaryDirectory() as tmp:
        edf_path = Path(tmp) / 'test.edf'
        _write_synthetic_edf(
            edf_path, ch_names, phys_dims, phys_mins, phys_maxs, dig_mins, dig_maxs, spr, 0.5, records
        )
        decoded, meta = decode_edf(edf_path, picks=[0, 1])

    expected_samples = n_records * spr_eeg
    assert decoded.shape == (2, expected_samples)
    assert meta['srate'] == pytest.approx(spr_eeg / 0.5)
    np.testing.assert_allclose(decoded[0], dig_fp1.ravel() * 0.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(decoded[1], dig_cz.ravel() * 1.0, rtol=0, atol=1e-12)


def test_decode_bdf_unpacks_signed_24bit_triplets_accurately():
    """Verify BioSemi 24-bit two's-complement unpacking across positive and negative extremes."""
    n_records = 2
    spr_ch = 5
    ch_names = ['Fp1', 'Oz', 'Status']
    phys_dims = ['uV', 'uV', '']
    phys_mins = [-262144.0, -262144.0, -8388608.0]
    phys_maxs = [262144.0, 262144.0, 8388607.0]
    dig_mins = [-8388608, -8388608, -8388608]
    dig_maxs = [8388607, 8388607, 8388607]
    spr = [spr_ch, spr_ch, spr_ch]

    dig_fp1 = np.array([[-320000, -1, 0, 1, 320000], [-65536, -256, 128, 4096, 131072]], dtype='<i4')
    dig_oz = -dig_fp1
    dig_stat = np.zeros_like(dig_fp1)
    records = np.concatenate([dig_fp1, dig_oz, dig_stat], axis=1)

    with tempfile.TemporaryDirectory() as tmp:
        bdf_path = Path(tmp) / 'test.bdf'
        _write_synthetic_bdf(
            bdf_path, ch_names, phys_dims, phys_mins, phys_maxs, dig_mins, dig_maxs, spr, 1.0, records
        )
        decoded, meta = decode_bdf(bdf_path, picks=[0, 1])

    gain = (phys_maxs[0] - phys_mins[0]) / (dig_maxs[0] - dig_mins[0])
    expected_fp1 = (dig_fp1.ravel() - dig_mins[0]) * gain + phys_mins[0]
    expected_oz = (dig_oz.ravel() - dig_mins[1]) * gain + phys_mins[1]
    assert decoded.shape == (2, n_records * spr_ch)
    assert meta['srate'] == pytest.approx(float(spr_ch))
    np.testing.assert_allclose(decoded[0], expected_fp1, rtol=0, atol=1e-9)
    np.testing.assert_allclose(decoded[1], expected_oz, rtol=0, atol=1e-9)


def test_extract_python_code_epochs_excludes_overlapping_followup_tokens():
    """Verify ds004771 selector keeps only latency-0 critical code bins and drops follow-up tokens."""
    epochs_meta = [
        {'eventlatency': [0, 902.34], 'eventtype': ['B2,4,8,16(240)', 'B18,20,24,32(241)']},
        {'eventlatency': [0, 898.44], 'eventtype': ['B18,20,24,32(241)', 'B34,36,40,48(242)']},
        {'eventlatency': 0, 'eventtype': 'B34,36,40,48(242)'},
        {'eventlatency': [0, 902.34], 'eventtype': ['B1,3,5,13(110)', 'B17,19,21,29(111)']},
        {'eventlatency': [0, 900.00], 'eventtype': ['B1,4,6,14(225)', 'B17,20,22,30(226)']},
        {'eventlatency': [0, 900.00], 'eventtype': ['B2,3,7,15(135)', 'B18,19,23,31(136)']},
    ]
    selected = extract_python_code_epochs(epochs_meta)
    assert selected == [
        (0, 240, '4', 'doubly_anomalous'),
        (3, 110, '1', 'well_formed'),
        (4, 225, '3', 'structurally_invalid'),
        (5, 135, '2', 'semantically_implausible'),
    ]


def test_extract_internal_attention_windows_stays_within_block_boundaries():
    """Verify 2.0 s windows do not cross into adjacent getready or opposite condition blocks."""
    hz = 250.0
    points = 500
    ev_rows = [
        {'onset': '1.0', 'trial_type': 'getready'},
        {'onset': '4.0', 'trial_type': 'concentrate'},
        {'onset': '5.0', 'trial_type': 'concentrate'},
        {'onset': '6.0', 'trial_type': 'concentrate'},
        {'onset': '7.0', 'trial_type': 'concentrate'},
        {'onset': '8.2', 'trial_type': 'relax'},
        {'onset': '9.2', 'trial_type': 'relax'},
        {'onset': '10.2', 'trial_type': 'relax'},
    ]
    windows = extract_internal_attention_windows(ev_rows, n_samples=4000, hz=hz, points=points)
    assert windows == [
        ('concentrate', 1000, 1500, 4.0),
        ('concentrate', 1500, 2000, 6.0),
        ('relax', 2050, 2550, 8.2),
    ]


def test_extract_music_clip_windows_stops_before_rating_questions_and_drops_neutral():
    """Verify ds002721 windowing stops at marker 1092 / question onset and skips neutral 5 ratings."""
    hz = 1000.0
    points = 4000
    ev_rows = [
        # Clip 0: pleasant=8 (908), energy=2 (834) -> HPLE ('2'), 16 s duration -> 4 windows
        {'onset': '5.0', 'trial_type': '788'},
        {'onset': '21.0', 'trial_type': '1092'},
        {'onset': '22.0', 'trial_type': '800'},
        {'onset': '23.0', 'trial_type': '908'},
        {'onset': '25.0', 'trial_type': '801'},
        {'onset': '26.0', 'trial_type': '834'},
        # Clip 1: pleasant=5 (neutral) -> excluded
        {'onset': '30.0', 'trial_type': '788'},
        {'onset': '46.0', 'trial_type': '1092'},
        {'onset': '47.0', 'trial_type': '800'},
        {'onset': '48.0', 'trial_type': '905'},
        {'onset': '50.0', 'trial_type': '801'},
        {'onset': '51.0', 'trial_type': '909'},
    ]
    windows = extract_music_clip_windows(ev_rows, n_samples=60000, hz=hz, points=points, max_windows_per_clip=4)
    assert len(windows) == 4
    assert [w['start_sample'] for w in windows] == [5000, 9000, 13000, 17000]
    assert all(w['end_sample'] <= 21000 for w in windows)
    assert all(w['quadrant'] == 'HPLE' and w['label'] == '2' for w in windows)


@pytest.mark.parametrize(
    ('component', 'rows', 'expected_codes_and_labels'),
    [
        (
            'N170',
            [
                {'trial_type': 'stimulus', 'value': '12', 'sample': '1025', 'onset': '1.0'},
                {'trial_type': 'response', 'value': '201', 'sample': '1537', 'onset': '1.5'},
                {'trial_type': 'stimulus', 'value': '55', 'sample': '2049', 'onset': '2.0'},
                {'trial_type': 'stimulus', 'value': '120', 'sample': '3073', 'onset': '3.0'},
                {'trial_type': 'stimulus', 'value': '160', 'sample': '4097', 'onset': '4.0'},
            ],
            [(12, 1), (55, 2), (120, 3), (160, 4)],
        ),
        (
            'MMN',
            [
                {'trial_type': 'stimulus', 'value': '180', 'sample': '1025', 'onset': '1.0'},
                {'trial_type': 'stimulus', 'value': '70', 'sample': '2049', 'onset': '2.0'},
                {'trial_type': 'stimulus', 'value': '80', 'sample': '3073', 'onset': '3.0'},
                {'trial_type': 'stimulus', 'value': '80', 'sample': '4097', 'onset': '4.0'},
                {'trial_type': 'stimulus', 'value': '70', 'sample': '5121', 'onset': '5.0'},
            ],
            [(70, 2), (80, 1), (70, 2)],
        ),
        (
            'N2pc',
            [
                {'trial_type': 'stimulus', 'value': '111', 'sample': '1025', 'onset': '1.0'},
                {'trial_type': 'stimulus', 'value': '222', 'sample': '2049', 'onset': '2.0'},
            ],
            [(111, 1), (222, 2)],
        ),
    ],
)
def test_erpcore_batch2_target_rows_mapping_and_mmn_pairing(component, rows, expected_codes_and_labels):
    """Verify ERP CORE N170, MMN pre-deviant standard pairing, and N2pc event mappings."""
    df = pd.DataFrame(rows)
    extracted = target_rows(component, df, lookup=None)
    assert [(item['code'], item['label']) for item in extracted] == expected_codes_and_labels
    assert all(item['zero'] == item['sample'] - 1 for item in extracted)


def test_storage_overflow_junction_roundtrip():
    """Verify overflow destination selection and Windows directory junction creation."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        primary = base / 'primary'
        overflow = base / 'overflow'
        dest, use_junc = resolve_storage_destination(
            'SyntheticCognitive', est_gb=1e9, primary_root=primary, overflow_root=overflow
        )
        assert dest == overflow
        assert use_junc is True

        target_dir = overflow / 'SyntheticCognitive'
        target_dir.mkdir(parents=True)
        sentinel = target_dir / 'hello.txt'
        sentinel.write_text('junction_ok', encoding='utf-8')

        create_junction_if_needed(primary / 'SyntheticCognitive', target_dir)
        assert (primary / 'SyntheticCognitive' / 'hello.txt').read_text(encoding='utf-8') == 'junction_ok'
