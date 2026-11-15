"""Tests for programmatic and interactive Neuroverse result downloading."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuroverse as nv
from neuroverse.datasets import by_type, neuroverse_90, neuroverse_114, neuroverse_paper
from neuroverse.estimators import CANONICAL_ESTIMATORS, HEADLINE_ESTIMATORS
from neuroverse.results import (
    VALID_METRICS,
    download_results,
    interactive_download,
    list_result_options,
    load_results,
    main,
)


def test_load_results_splits_pools_and_legacy_aliases():
    """load_results slices splits, classifier tiers, Default baseline, and legacy directory aliases."""
    core_wide = load_results(metric="balacc", split="core90", estimators="headline")
    assert isinstance(core_wide, pd.DataFrame)
    assert list(core_wide.index) == neuroverse_90
    assert list(core_wide.columns) == HEADLINE_ESTIMATORS

    mi_with_default = load_results(
        metric="balacc",
        split="MI",
        estimators="canonical",
        include_default=True,
        drop_empty_rows=True,
    )
    assert isinstance(mi_with_default, pd.DataFrame)
    assert mi_with_default.columns[0] == "Default"
    assert list(mi_with_default.columns[1:]) == CANONICAL_ESTIMATORS
    assert set(mi_with_default.index) <= set(by_type("MI"))
    assert len(mi_with_default) < len(by_type("MI")) or mi_with_default.iloc[:, 1:].notna().any(axis=1).all()
    assert mi_with_default["Default"].notna().all()

    legacy_names = ["HC2-noTDE-500-pc", "DrCIF-500", "STC-aeon", "IT"]
    clean_names = ["HC2", "DrCIF", "STC", "InceptionTime"]
    via_legacy = load_results(metric="balacc", split="all114", estimators=legacy_names)
    via_clean = load_results(metric="balacc", split="all114", estimators=clean_names)
    assert isinstance(via_legacy, pd.DataFrame) and isinstance(via_clean, pd.DataFrame)
    assert list(via_legacy.index) == neuroverse_114
    assert list(via_legacy.columns) == clean_names
    pd.testing.assert_frame_equal(via_legacy, via_clean)

    multi = load_results(metric=["balacc", "accuracy"], split="ME", estimators=["HC2", "Arsenal"])
    assert isinstance(multi, dict)
    assert list(multi) == ["balacc", "accuracy"]
    for frame in multi.values():
        assert list(frame.index) == by_type("ME")
        assert list(frame.columns) == ["HC2", "Arsenal"]


def test_load_results_long_league_and_paper2026_formats():
    """Long-form tidy records and ranked league tables preserve underlying wide matrix values."""
    subset_estimators = ["HC2", "Arsenal", "MRHydra", "DrCIF", "STC"]
    wide_balacc = load_results(metric="balacc", split="MI", estimators=subset_estimators)
    long_df = load_results(
        metric=["balacc", "accuracy"],
        split="MI",
        estimators=subset_estimators,
        format="long",
    )
    assert isinstance(wide_balacc, pd.DataFrame) and isinstance(long_df, pd.DataFrame)
    expected_non_null = int(wide_balacc.notna().sum().sum())
    assert len(long_df) == expected_non_null
    assert {"dataset", "problem_type", "estimator", "paradigm", "is_headline", "balacc", "accuracy"} <= set(
        long_df.columns
    )
    sample_row = long_df.iloc[0]
    assert np.isclose(
        float(sample_row["balacc"]),
        float(wide_balacc.loc[sample_row["dataset"], sample_row["estimator"]]),
    )

    league_df = load_results(
        metric="balacc",
        split="core90",
        estimators="headline",
        format="league",
        league_mode="available",
    )
    assert isinstance(league_df, pd.DataFrame)
    assert set(league_df["estimator"]) == set(HEADLINE_ESTIMATORS)
    assert list(league_df["position"]) == list(range(1, len(HEADLINE_ESTIMATORS) + 1))
    assert league_df["mean_rank"].is_monotonic_increasing

    paper_df = load_results(metric="balacc", split="paper2026", estimators="headline")
    assert isinstance(paper_df, pd.DataFrame)
    assert list(paper_df.index) == sorted(neuroverse_paper)
    assert paper_df.notna().all().all()


def test_download_results_all_formats_and_multiverse_roundtrip():
    """download_results writes lossless wide, multiverse, long, league, and JSON exports."""
    chosen_datasets = ["ImaginedFeetHands", "FeetHands", "OpenCloseFist"]
    chosen_estimators = ["HC2", "Arsenal", "DrCIF"]
    chosen_metrics = ["balacc", "accuracy"]

    reference = load_results(
        metric=chosen_metrics,
        split=chosen_datasets,
        estimators=chosen_estimators,
        include_default=True,
    )
    assert isinstance(reference, dict)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        wide_dir = root / "wide_export"
        wide_files = download_results(
            wide_dir,
            format="wide",
            metrics=chosen_metrics,
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=True,
        )
        expected_wide_count = len(chosen_metrics) + 2  # metric CSVs + datasets.csv + classifiers.csv
        assert len(wide_files) == expected_wide_count
        reloaded_wide = load_results(
            metric="balacc",
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=True,
            results_dir=wide_dir,
        )
        assert isinstance(reloaded_wide, pd.DataFrame)
        pd.testing.assert_frame_equal(reloaded_wide, reference["balacc"], atol=1e-6)

        mv_dir = root / "mv_export"
        mv_files = download_results(
            mv_dir,
            format="multiverse",
            metrics=chosen_metrics,
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=True,
        )
        expected_mv_count = (len(chosen_estimators) + 1) * len(chosen_metrics)
        assert len(mv_files) == expected_mv_count
        reloaded_mv = load_results(
            metric="balacc",
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=True,
            results_dir=mv_dir,
        )
        assert isinstance(reloaded_mv, pd.DataFrame)
        pd.testing.assert_frame_equal(reloaded_mv, reference["balacc"], atol=1e-6)

        long_file = root / "slice_long.csv"
        download_results(
            long_file,
            format="long",
            metrics=chosen_metrics,
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=False,
        )
        long_read = pd.read_csv(long_file)
        assert set(long_read["estimator"]) == set(chosen_estimators)
        assert len(long_read) == len(chosen_datasets) * len(chosen_estimators)

        league_file = root / "slice_league.csv"
        download_results(
            league_file,
            format="league",
            metrics=chosen_metrics,
            split=chosen_datasets,
            estimators=chosen_estimators,
        )
        league_read = pd.read_csv(league_file)
        assert set(league_read["estimator"]) == set(chosen_estimators)

        json_file = root / "slice.json"
        download_results(
            json_file,
            format="json",
            metrics=chosen_metrics,
            split=chosen_datasets,
            estimators=chosen_estimators,
            include_default=False,
        )
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        assert payload["datasets"] == chosen_datasets
        assert [entry["name"] for entry in payload["estimators"]] == chosen_estimators
        for ds in chosen_datasets:
            for est in chosen_estimators:
                assert np.isclose(
                    payload["scores"]["balacc"][ds][est],
                    float(reference["balacc"].loc[ds, est]),
                    atol=1e-6,
                )


def test_remote_source_caching_interactive_wizard_and_cli(monkeypatch, capsys):
    """Remote caching fetches once into cache_dir, and both CLI and interactive wizard export cleanly."""
    import neuroverse.results as res_mod

    repo_neuroverse = Path(__file__).resolve().parents[2] / "results" / "neuroverse"
    fetched_urls: list[str] = []

    def fake_fetch(url: str, target: Path, refresh: bool = False) -> Path:
        if target.is_file() and target.stat().st_size > 0 and not refresh:
            return target
        fetched_urls.append(url)
        rel = url.split("/results/neuroverse/", 1)[1]
        src = repo_neuroverse / rel
        if not src.is_file():
            raise FileNotFoundError(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        return target

    monkeypatch.setattr(res_mod, "_fetch_remote_file", fake_fetch)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache_dir = root / "cache"

        first = load_results(
            metric="balacc",
            split="ME",
            estimators="headline",
            source="remote",
            cache_dir=cache_dir,
        )
        count_after_first = len(fetched_urls)
        assert count_after_first >= 1

        second = load_results(
            metric="balacc",
            split="ME",
            estimators="headline",
            source="remote",
            cache_dir=cache_dir,
        )
        assert len(fetched_urls) == count_after_first
        assert isinstance(first, pd.DataFrame) and isinstance(second, pd.DataFrame)
        pd.testing.assert_frame_equal(first, second)

        wizard_dest = root / "wizard_league.csv"
        simulated_stdin = io.StringIO(f"2\n1\n2\n3\n{wizard_dest}\n")
        wizard_stdout = io.StringIO()
        written = interactive_download(stdin=simulated_stdin, stdout=wizard_stdout)
        assert written == [wizard_dest]
        assert wizard_dest.is_file()
        assert "Equivalent Python snippet:" in wizard_stdout.getvalue()

        cli_dest = root / "cli_long.csv"
        rc = main(
            [
                "--split",
                "ME",
                "--estimators",
                "HC2,STC",
                "--metrics",
                "balacc",
                "--format",
                "long",
                "--no-default",
                "--out",
                str(cli_dest),
            ]
        )
        assert rc == 0
        cli_df = pd.read_csv(cli_dest)
        assert set(cli_df["estimator"]) == {"HC2", "STC"}

        assert main(["--list"]) == 0
        out_text = capsys.readouterr().out
        assert "core90" in out_text and "headline" in out_text

    opts = list_result_options()
    assert opts["metrics"] == list(VALID_METRICS)
    assert nv.load_results is load_results
    assert nv.download_results is download_results


def test_results_validation_errors():
    """Invalid splits, metrics, formats, or estimator names raise clear ValueErrors."""
    with pytest.raises(ValueError, match="Unknown split"):
        load_results(split="NotASplit")
    with pytest.raises(ValueError, match="Unknown metric"):
        load_results(metric="f1_macro")
    with pytest.raises(ValueError, match="format"):
        load_results(format="parquet")
    with pytest.raises(ValueError, match="estimator"):
        load_results(estimators="NonExistentClassifierXYZ")
