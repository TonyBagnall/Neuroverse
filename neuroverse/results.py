"""Programmatic and interactive result downloader for Neuroverse.

Users can query or download Neuroverse benchmark scores either in Python or
from the terminal, working transparently from a local repository checkout or by
fetching the published CSV tables from GitHub when installed standalone::

    import neuroverse as nv

    # Load the 12 headline classifiers on the primary 30/30/30 (90-problem) split
    balacc = nv.load_results(metric="balacc", split="core90", estimators="headline")

    # Load a tidy long-form table across all 81 distinct classifiers on Motor Imagery
    tidy = nv.load_results(metric="all", split="MI", estimators="canonical", format="long")

    # Download a self-contained folder of CSVs (or Multiverse per-estimator layout)
    nv.download_results("my_eeg_results", format="wide", split="core90", estimators="canonical")

Run from the command line with no arguments (or ``--interactive``) to launch an
interactive step-by-step selector, or pass flags directly::

    neuroverse-results
    python -m neuroverse.results --split core90 --estimators headline --format league --out headline.csv
"""

from __future__ import annotations

__all__ = [
    "RAW_RESULTS_URL",
    "VALID_METRICS",
    "VALID_FORMATS",
    "list_result_options",
    "resolve_split_datasets",
    "resolve_estimator_selection",
    "load_results",
    "download_results",
    "interactive_download",
    "main",
]

import argparse
import json
import os
import ssl
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from neuroverse.datasets import (
    STUDY_TAXONOMY,
    by_type,
    metadata as dataset_metadata,
    neuroverse_90,
    neuroverse_114,
    neuroverse_paper,
)
from neuroverse.estimators import (
    BAKEOFF_ESTIMATORS,
    CANONICAL_ESTIMATORS,
    HEADLINE_ESTIMATORS,
    PAPER_ESTIMATORS,
    PARADIGMS,
    estimator_info,
    resolve_estimator_name,
)

RAW_RESULTS_URL = "https://raw.githubusercontent.com/TonyBagnall/Neuroverse/main/results"
VALID_METRICS = ("balacc", "accuracy", "auroc", "logloss")
VALID_FORMATS = ("wide", "long", "league", "multiverse", "json")

_REPO_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

_SPLIT_ALIASES: dict[str, str] = {
    "core90": "core90",
    "core": "core90",
    "study": "core90",
    "90": "core90",
    "30/30/30": "core90",
    "all114": "all114",
    "all": "all114",
    "extended": "all114",
    "114": "all114",
    "mi": "MI",
    "motorimagery": "MI",
    "clinical": "Clinical",
    "cognitive": "Cognitive",
    "bci": "BCI",
    "me": "ME",
    "erp": "ERP",
    "p300": "ERP",
    "vep": "VEP",
    "ssvep": "VEP",
    "speech": "Speech",
    "paper2026": "paper2026",
    "paper": "paper2026",
    "30": "paper2026",
}


def _default_cache_dir() -> Path:
    env = os.environ.get("NEUROVERSE_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / "neuroverse" / "Cache" / "results"
    return Path.home() / ".cache" / "neuroverse" / "results"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch_remote_file(url: str, target: Path, refresh: bool = False) -> Path:
    if target.is_file() and target.stat().st_size > 0 and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "neuroverse-results/0.1"})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=45) as resp:
        payload = resp.read()
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return target


def list_result_options() -> dict[str, object]:
    """Return available splits, estimator pools, paradigms, metrics, and formats."""
    return {
        "splits": {
            "core90": f"Primary 30/30/30 benchmark ({len(neuroverse_90)} problems: 30 MI + 30 Clinical + 30 Cognitive)",
            "MI": f"Motor Imagery ({len(by_type('MI'))} problems)",
            "Clinical": f"Clinical and physiological ({len(by_type('Clinical'))} problems)",
            "Cognitive": f"Cognitive and affective ({len(by_type('Cognitive'))} problems)",
            "all114": f"All scalp EEG benchmark problems ({len(neuroverse_114)} problems)",
            "BCI": f"All BCI paradigms ({len(by_type('BCI'))} problems)",
            "ME": f"Motor Execution ({len(by_type('ME'))} problems)",
            "ERP": f"ERP / P300 ({len(by_type('ERP'))} problems)",
            "VEP": f"VEP / SSVEP ({len(by_type('VEP'))} problems)",
            "Speech": f"Decoded / imagined speech ({len(by_type('Speech'))} problems)",
            "paper2026": f"Archived 2026 DMLR paper benchmark ({len(neuroverse_paper)} problems)",
        },
        "estimator_pools": {
            "headline": f"Headline leaderboard ({len(HEADLINE_ESTIMATORS)} classifiers)",
            "canonical": f"Expanded distinct classifiers ({len(CANONICAL_ESTIMATORS)} classifiers)",
            "all": f"All evaluated configurations ({len(BAKEOFF_ESTIMATORS)} configs incl. variants & controls)",
            **{
                par: f"{label} canonical classifiers"
                for par, label in PARADIGMS.items()
            },
            "paper": f"2026 DMLR Table 2 classifiers ({len(PAPER_ESTIMATORS)} classifiers)",
        },
        "metrics": list(VALID_METRICS),
        "formats": list(VALID_FORMATS),
    }


def resolve_split_datasets(split: str | Sequence[str] = "core90") -> tuple[str, list[str]]:
    """Resolve ``split`` to ``(canonical_split_key, dataset_names)``."""
    if not isinstance(split, str):
        names = list(dict.fromkeys(str(x) for x in split))
        if not names:
            raise ValueError("split sequence must not be empty")
        return "custom", names

    key = _SPLIT_ALIASES.get(split.strip().lower().replace(" ", ""), split.strip())
    if key == "core90":
        return "core90", list(neuroverse_90)
    if key == "all114":
        return "all114", list(neuroverse_114)
    if key == "paper2026":
        return "paper2026", sorted(neuroverse_paper)
    if key in {"MI", "Clinical", "Cognitive", "BCI", "ME", "ERP", "VEP", "Speech"}:
        return key, by_type(key)
    raise ValueError(
        f"Unknown split {split!r}. Valid splits: "
        "core90, MI, Clinical, Cognitive, all114, BCI, ME, ERP, VEP, Speech, paper2026, "
        "or an explicit list of dataset names."
    )


def resolve_estimator_selection(
    estimators: str | Sequence[str] = "headline",
    available: Sequence[str] | None = None,
    for_paper: bool = False,
) -> list[str]:
    """Resolve an estimator tier, paradigm code, or explicit name list."""
    avail_list = list(available) if available is not None else None
    avail_set = set(avail_list) if avail_list is not None else None

    if isinstance(estimators, str):
        token = estimators.strip()
        low = token.lower()
        if for_paper and low in {"headline", "paper"}:
            wanted = list(PAPER_ESTIMATORS)
        elif for_paper and low in {"canonical", "all", "bakeoff", "configs"}:
            wanted = list(avail_list) if avail_list is not None else list(PAPER_ESTIMATORS)
        elif low == "headline":
            wanted = list(HEADLINE_ESTIMATORS)
        elif low in {"canonical", "distinct", "expanded"}:
            wanted = list(CANONICAL_ESTIMATORS)
        elif low in {"all", "bakeoff", "configs"}:
            wanted = list(avail_list) if avail_list is not None else list(BAKEOFF_ESTIMATORS)
        elif low == "paper":
            wanted = list(PAPER_ESTIMATORS)
        else:
            par_match = next(
                (p for p in PARADIGMS if p.lower() == low),
                None,
            )
            if par_match is not None:
                wanted = [
                    name
                    for name in CANONICAL_ESTIMATORS
                    if estimator_info(name)["paradigm"] == par_match
                ]
            elif "," in token:
                parts = [p.strip() for p in token.split(",") if p.strip()]
                return resolve_estimator_selection(parts, available=available, for_paper=for_paper)
            else:
                resolved = token if (avail_set and token in avail_set) else resolve_estimator_name(token)
                if avail_set is not None and resolved not in avail_set:
                    raise ValueError(
                        f"Unknown or unavailable estimator {token!r}. Pass 'headline', "
                        "'canonical', 'all', a paradigm name, or valid classifier names."
                    )
                return [resolved]
    else:
        wanted = []
        for raw in estimators:
            s = str(raw).strip()
            if not s:
                continue
            cand = s if (avail_set and s in avail_set) else resolve_estimator_name(s)
            wanted.append(cand)
        if not wanted:
            raise ValueError("estimators sequence must not be empty")

    if avail_set is not None:
        filtered = [e for e in wanted if e in avail_set and e != "Default"]
        if not filtered and not isinstance(estimators, str):
            raise ValueError(f"None of requested estimators {list(estimators)!r} found in results.")
        return filtered
    return [e for e in wanted if e != "Default"]


def _resolve_metrics(metric: str | Sequence[str]) -> tuple[list[str], bool]:
    if isinstance(metric, str):
        tok = metric.strip().lower()
        if tok == "all":
            return list(VALID_METRICS), False
        if "," in tok:
            parts = [p.strip() for p in tok.split(",") if p.strip()]
            for m in parts:
                if m not in VALID_METRICS:
                    raise ValueError(f"Unknown metric {m!r}; valid: {', '.join(VALID_METRICS)}")
            return parts, False
        if tok not in VALID_METRICS:
            raise ValueError(f"Unknown metric {metric!r}; valid: {', '.join(VALID_METRICS)}, or 'all'")
        return [tok], True
    parts = [str(m).strip().lower() for m in metric if str(m).strip()]
    if not parts:
        raise ValueError("metric sequence must not be empty")
    for m in parts:
        if m not in VALID_METRICS:
            raise ValueError(f"Unknown metric {m!r}; valid: {', '.join(VALID_METRICS)}")
    return parts, False


def _prepare_results_dir(
    split_key: str,
    metrics_needed: Sequence[str],
    include_default: bool,
    source: str,
    refresh: bool,
    cache_dir: Path | None,
    results_dir: Path | str | None,
    base_url: str = RAW_RESULTS_URL,
) -> Path:
    if results_dir is not None:
        root = Path(results_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"results_dir does not exist: {root}")
        return root

    subdir = "neuroverse_paper_2026" if split_key == "paper2026" else "neuroverse"
    local_candidate = _REPO_RESULTS_ROOT / subdir

    if source not in {"auto", "local", "remote"}:
        raise ValueError(f"source must be 'auto', 'local', or 'remote', not {source!r}")

    if source == "local":
        if not local_candidate.is_dir():
            raise FileNotFoundError(
                f"Local results directory not found at {local_candidate}. "
                "Use source='auto' or source='remote' to fetch from GitHub."
            )
        return local_candidate

    if source == "auto" and not refresh and local_candidate.is_dir():
        if subdir == "neuroverse" and (local_candidate / "balacc.csv").is_file():
            return local_candidate
        if subdir == "neuroverse_paper_2026" and any(local_candidate.iterdir()):
            return local_candidate

    cache_root = (cache_dir or _default_cache_dir()) / subdir
    remote_sub = f"{base_url.rstrip('/')}/{subdir}"

    if subdir == "neuroverse":
        for m in metrics_needed:
            _fetch_remote_file(f"{remote_sub}/{m}.csv", cache_root / f"{m}.csv", refresh=refresh)
        for aux in ("datasets.csv", "classifiers.csv"):
            try:
                _fetch_remote_file(f"{remote_sub}/{aux}", cache_root / aux, refresh=refresh)
            except Exception:  # noqa: BLE001
                pass
        if include_default:
            for m in metrics_needed:
                try:
                    _fetch_remote_file(
                        f"{remote_sub}/Default/Default_{m}.csv",
                        cache_root / "Default" / f"Default_{m}.csv",
                        refresh=refresh,
                    )
                except Exception:  # noqa: BLE001
                    pass
    else:
        ests_to_fetch = list(PAPER_ESTIMATORS) + (["Default"] if include_default else [])
        for est in ests_to_fetch:
            for m in metrics_needed:
                try:
                    _fetch_remote_file(
                        f"{remote_sub}/{est}/{est}_{m}.csv",
                        cache_root / est / f"{est}_{m}.csv",
                        refresh=refresh,
                    )
                except Exception:  # noqa: BLE001
                    pass
    return cache_root


def _read_metric_frame(
    results_dir: Path,
    metric: str,
    datasets: list[str],
    estimators: list[str] | str,
    include_default: bool,
    for_paper: bool,
) -> pd.DataFrame:
    from neuroverse.experiments.tables import available_estimators, load_metric

    wide_path = results_dir / f"{metric}.csv"
    if wide_path.is_file():
        wide = pd.read_csv(wide_path, index_col=0)
        wide.index = wide.index.astype(str)
        avail = [str(c) for c in wide.columns]
        chosen = resolve_estimator_selection(estimators, available=avail, for_paper=for_paper)
        out = wide.reindex(index=datasets, columns=chosen).astype(float)
        if include_default:
            def_path = results_dir / "Default" / f"Default_{metric}.csv"
            if "Default" in wide.columns:
                out.insert(0, "Default", wide.reindex(datasets)["Default"].astype(float))
            elif def_path.is_file():
                d_ser = pd.read_csv(def_path, index_col=0).mean(axis=1)
                d_ser.index = d_ser.index.astype(str)
                out.insert(0, "Default", d_ser.reindex(datasets).astype(float))
        out.index.name = "dataset"
        return out

    avail = available_estimators(results_dir, exclude=("Default",))
    chosen = resolve_estimator_selection(estimators, available=avail, for_paper=for_paper)
    cols: dict[str, pd.Series] = {}
    if include_default:
        try:
            cols["Default"] = load_metric("Default", metric, results_dir)
        except FileNotFoundError:
            pass
    for est in chosen:
        try:
            cols[est] = load_metric(est, metric, results_dir)
        except FileNotFoundError:
            continue
    out = pd.DataFrame(cols).reindex(datasets)
    out.index.name = "dataset"
    return out


def _frames_to_long(
    frames: dict[str, pd.DataFrame],
    include_empty: bool = False,
) -> pd.DataFrame:
    meta_df = dataset_metadata()
    metrics_list = list(frames)
    first = frames[metrics_list[0]]
    rows: list[dict[str, object]] = []
    for ds in first.index:
        fam, bci = STUDY_TAXONOMY.get(ds, ("", ""))
        prob_type = bci if bci else (fam if fam else "Archive")
        ds_row = meta_df.loc[ds] if ds in meta_df.index else None
        for est in first.columns:
            vals = {m: frames[m].at[ds, est] for m in metrics_list}
            if not include_empty and all(pd.isna(v) for v in vals.values()):
                continue
            einfo = estimator_info(est)
            rec: dict[str, object] = {
                "dataset": ds,
                "family": fam,
                "bci_type": bci,
                "problem_type": prob_type,
                "channels": int(ds_row["channels"]) if ds_row is not None else None,
                "length": int(ds_row["length"]) if ds_row is not None else None,
                "classes": int(ds_row["classes"]) if ds_row is not None else None,
                "estimator": est,
                "paradigm": einfo["paradigm"],
                "paradigm_label": einfo["paradigm_label"],
                "tsml_family": einfo["tsml_family"],
                "is_headline": bool(einfo["is_headline"]),
                "is_canonical": bool(einfo["is_canonical"]),
            }
            for m, v in vals.items():
                rec[m] = None if pd.isna(v) else float(v)
            rows.append(rec)
    return pd.DataFrame(rows)


def load_results(
    metric: str | Sequence[str] = "balacc",
    split: str | Sequence[str] = "core90",
    estimators: str | Sequence[str] = "headline",
    *,
    format: str = "wide",
    include_default: bool = False,
    drop_empty_rows: bool = False,
    league_mode: str = "common_block",
    min_coverage_ratio: float = 0.8,
    source: str = "auto",
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Load Neuroverse benchmark results in Python (from local files or GitHub).

    Parameters
    ----------
    metric : str or sequence of str, default="balacc"
        ``"balacc"``, ``"accuracy"``, ``"auroc"``, ``"logloss"``, ``"all"``,
        or a sequence of metric names.
    split : str or sequence of str, default="core90"
        Problem split: ``"core90"`` (primary 30 MI + 30 Clinical + 30 Cognitive
        benchmark), ``"MI"``, ``"Clinical"``, ``"Cognitive"``, ``"all114"``,
        ``"BCI"``, ``"ME"``, ``"ERP"``, ``"VEP"``, ``"Speech"``, ``"paper2026"``,
        or an explicit sequence of dataset names.
    estimators : str or sequence of str, default="headline"
        Classifier pool: ``"headline"`` (12 primary classifiers), ``"canonical"``
        (81 distinct classifiers), ``"all"`` (all 109 configurations), a
        paradigm key (``"TSML"``, ``"EEG-Classical"``, ``"EEG-DL"``, ``"EEG-FM"``,
        ``"Tabular"``), or an explicit sequence of estimator names.
    format : {"wide", "long", "league"}, default="wide"
        - ``"wide"``: ``dataset x estimator`` DataFrame (or ``dict[metric, DataFrame]``
          when multiple metrics are requested).
        - ``"long"``: tidy DataFrame with one row per ``(dataset, estimator)``
          plus problem and algorithm metadata.
        - ``"league"``: ranked summary DataFrame produced by
          :func:`neuroverse.experiments.tables.compute_league_table`.
    include_default : bool, default=False
        Whether to include the constant majority-class ``Default`` baseline.
    drop_empty_rows : bool, default=False
        In ``"wide"`` mode, drop dataset rows that have no scores yet across the
        selected estimators.
    league_mode : {"common_block", "available"}, default="common_block"
        Coverage mode when ``format="league"``.
    min_coverage_ratio : float, default=0.8
        Minimum coverage fraction when ``format="league"`` and
        ``league_mode="common_block"``.
    source : {"auto", "local", "remote"}, default="auto"
        Where to read CSVs from. ``"auto"`` uses local repository CSVs when
        present and downloads from GitHub into ``cache_dir`` otherwise.
    refresh : bool, default=False
        Re-download remote CSVs even when cached locally.
    cache_dir : Path or str or None, default=None
        Override local cache directory for remote downloads.
    results_dir : Path or str or None, default=None
        Explicit directory of CSVs to read instead of default local/remote paths.

    Returns
    -------
    pd.DataFrame or dict of str -> pd.DataFrame
    """
    if format not in {"wide", "long", "league"}:
        raise ValueError(f"load_results format must be 'wide', 'long', or 'league', not {format!r}")

    split_key, datasets = resolve_split_datasets(split)
    for_paper = split_key == "paper2026"
    metrics_list, is_single = _resolve_metrics("all" if format == "league" and metric == "all" else metric)
    if format == "league" and "balacc" not in metrics_list:
        primary_metric = metrics_list[0]
    else:
        primary_metric = metrics_list[0]

    need_metrics = list(VALID_METRICS) if format == "league" else metrics_list
    resolved_dir = _prepare_results_dir(
        split_key=split_key,
        metrics_needed=need_metrics,
        include_default=include_default,
        source=source,
        refresh=refresh,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        results_dir=results_dir,
    )

    frames: dict[str, pd.DataFrame] = {}
    for m in need_metrics:
        df = _read_metric_frame(
            resolved_dir,
            m,
            datasets=datasets,
            estimators=estimators,
            include_default=include_default and format != "league",
            for_paper=for_paper,
        )
        if drop_empty_rows and format == "wide":
            comp_cols = [c for c in df.columns if c != "Default"] or list(df.columns)
            df = df.loc[df[comp_cols].notna().any(axis=1)]
        frames[m] = df

    if format == "long":
        return _frames_to_long({m: frames[m] for m in metrics_list}, include_empty=False)

    if format == "league":
        from neuroverse.experiments.tables import compute_league_table

        tbl, _ = compute_league_table(
            datasets=datasets,
            frames=frames,
            primary_metric=primary_metric,
            min_coverage_ratio=min_coverage_ratio,
            mode=league_mode,
            canonical_only=False,
        )
        return tbl

    if is_single:
        return frames[metrics_list[0]]
    return {m: frames[m] for m in metrics_list}


def download_results(
    dest: Path | str = "neuroverse_results",
    *,
    format: str = "wide",
    metrics: str | Sequence[str] = VALID_METRICS,
    split: str | Sequence[str] = "core90",
    estimators: str | Sequence[str] = "headline",
    include_default: bool = True,
    league_mode: str = "common_block",
    min_coverage_ratio: float = 0.8,
    source: str = "auto",
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> list[Path]:
    """Download and export Neuroverse results to disk in the requested format.

    Parameters
    ----------
    dest : Path or str, default="neuroverse_results"
        Target directory (for ``"wide"`` and ``"multiverse"``) or target file /
        directory (for ``"long"``, ``"league"``, and ``"json"``).
    format : {"wide", "long", "league", "multiverse", "json"}, default="wide"
        Export layout.
    metrics : str or sequence of str, default=("balacc", "accuracy", "auroc", "logloss")
        Metrics to export.
    split : str or sequence of str, default="core90"
        Dataset split or explicit dataset list.
    estimators : str or sequence of str, default="headline"
        Classifier pool, paradigm code, or explicit estimator list.
    include_default : bool, default=True
        Include the majority-class ``Default`` baseline where applicable.

    Returns
    -------
    list of Path
        Paths of all files written.
    """
    if format not in VALID_FORMATS:
        raise ValueError(f"format must be one of {', '.join(VALID_FORMATS)}, not {format!r}")

    split_key, datasets = resolve_split_datasets(split)
    for_paper = split_key == "paper2026"
    metrics_list, _ = _resolve_metrics(metrics)
    resolved_dir = _prepare_results_dir(
        split_key=split_key,
        metrics_needed=metrics_list,
        include_default=include_default,
        source=source,
        refresh=refresh,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        results_dir=results_dir,
    )

    frames = {
        m: _read_metric_frame(
            resolved_dir,
            m,
            datasets=datasets,
            estimators=estimators,
            include_default=include_default,
            for_paper=for_paper,
        )
        for m in metrics_list
    }

    target = Path(dest)
    written: list[Path] = []
    meta_df = dataset_metadata()

    if format == "wide":
        out_dir = target
        out_dir.mkdir(parents=True, exist_ok=True)
        for m, df in frames.items():
            p = out_dir / f"{m}.csv"
            df.to_csv(p, float_format="%.6f")
            written.append(p)

        ds_sub = meta_df.reindex([d for d in datasets if d in meta_df.index]).copy()
        ds_path = out_dir / "datasets.csv"
        ds_sub.to_csv(ds_path)
        written.append(ds_path)

        est_cols = [c for c in frames[metrics_list[0]].columns if c != "Default"]
        clf_rows = []
        for est in est_cols:
            info = estimator_info(est)
            clf_rows.append(
                {
                    "estimator": est,
                    "paradigm": info["paradigm"],
                    "paradigm_label": info["paradigm_label"],
                    "tsml_family": info["tsml_family"],
                    "library": info["library"],
                    "is_headline": bool(info["is_headline"]),
                    "is_canonical": bool(info["is_canonical"]),
                    "is_ablation": bool(info["is_ablation"]),
                    "description": info["description"],
                }
            )
        clf_path = out_dir / "classifiers.csv"
        pd.DataFrame(clf_rows).to_csv(clf_path, index=False)
        written.append(clf_path)
        return written

    if format == "multiverse":
        out_dir = target
        out_dir.mkdir(parents=True, exist_ok=True)
        for m, df in frames.items():
            for est in df.columns:
                ser = df[est].dropna()
                if ser.empty:
                    continue
                est_dir = out_dir / str(est)
                est_dir.mkdir(parents=True, exist_ok=True)
                ser.index.name = "Resamples:"
                ser.name = "0"
                p = est_dir / f"{est}_{m}.csv"
                ser.to_csv(p, float_format="%.6f")
                if p not in written:
                    written.append(p)
        return written

    if format == "long":
        out_file = target if target.suffix.lower() == ".csv" else target / f"neuroverse_{split_key}_long.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tidy = _frames_to_long(frames, include_empty=False)
        tidy.to_csv(out_file, index=False, float_format="%.6f")
        return [out_file]

    if format == "league":
        from neuroverse.experiments.tables import compute_league_table

        out_file = target if target.suffix.lower() == ".csv" else target / f"neuroverse_{split_key}_league.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        comp_frames = {
            m: df[[c for c in df.columns if c != "Default"]]
            for m, df in frames.items()
        }
        tbl, _ = compute_league_table(
            datasets=datasets,
            frames=comp_frames,
            primary_metric=metrics_list[0],
            min_coverage_ratio=min_coverage_ratio,
            mode=league_mode,
            canonical_only=False,
        )
        tbl.to_csv(out_file, index=False, float_format="%.6f")
        return [out_file]

    # format == "json"
    out_file = target if target.suffix.lower() == ".json" else target / f"neuroverse_{split_key}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    est_cols = list(frames[metrics_list[0]].columns)
    bundle = {
        "split": split_key,
        "datasets": datasets,
        "estimators": [
            {"name": est, **estimator_info(est)}
            for est in est_cols
        ],
        "scores": {
            m: {
                ds: {
                    est: (None if pd.isna(df.at[ds, est]) else round(float(df.at[ds, est]), 6))
                    for est in df.columns
                }
                for ds in df.index
            }
            for m, df in frames.items()
        },
    }
    out_file.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return [out_file]


def _ask_choice(
    prompt: str,
    options: list[tuple[str, str]],
    default_idx: int = 1,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> str:
    stdout.write(f"\n{prompt}\n")
    for idx, (_, label) in enumerate(options, start=1):
        mark = " (default)" if idx == default_idx else ""
        stdout.write(f"  [{idx}] {label}{mark}\n")
    stdout.write(f"Select [1-{len(options)}] (Enter for {default_idx}): ")
    stdout.flush()
    raw = stdin.readline()
    if not raw or not raw.strip():
        return options[default_idx - 1][0]
    val = raw.strip()
    if val.isdigit():
        num = int(val)
        if 1 <= num <= len(options):
            return options[num - 1][0]
    for key, _ in options:
        if key.lower() == val.lower():
            return key
    stdout.write(f"  Unrecognised input {val!r}; using default [{default_idx}].\n")
    return options[default_idx - 1][0]


def interactive_download(
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    results_dir: Path | str | None = None,
) -> list[Path]:
    """Run an interactive terminal wizard to filter and download Neuroverse results."""
    stdout.write("=== Neuroverse Interactive Result Downloader ===\n")

    split = _ask_choice(
        "1. Choose benchmark problem split:",
        [
            ("core90", f"Core 90 (30 MI + 30 Clinical + 30 Cognitive) [{len(neuroverse_90)} problems]"),
            ("MI", f"Motor Imagery only [{len(by_type('MI'))} problems]"),
            ("Clinical", f"Clinical only [{len(by_type('Clinical'))} problems]"),
            ("Cognitive", f"Cognitive only [{len(by_type('Cognitive'))} problems]"),
            ("all114", f"All Scalp EEG (incl. ME, ERP, VEP, Speech) [{len(neuroverse_114)} problems]"),
            ("BCI", f"BCI All [{len(by_type('BCI'))} problems]"),
            ("paper2026", f"Archived 2026 DMLR Paper [{len(neuroverse_paper)} problems]"),
        ],
        default_idx=1,
        stdin=stdin,
        stdout=stdout,
    )

    est_choice = _ask_choice(
        "2. Choose classifier pool:",
        [
            ("headline", f"Headline Leaderboard ({len(HEADLINE_ESTIMATORS)} classifiers: HC2, Arsenal, MRHydra, QUANT, DrCIF, STC, RDST, TDE, Catch22, FreshPRINCE, InceptionTime, RotationForest)"),
            ("canonical", f"Expanded Canonical Classifiers ({len(CANONICAL_ESTIMATORS)} distinct across all 5 paradigms)"),
            ("all", f"All Configurations ({len(BAKEOFF_ESTIMATORS)} configs incl. variants & random controls)"),
            ("TSML", "General TSML paradigm only"),
            ("EEG-Classical", "EEG Classical (Riemannian / CSP / xDAWN / TRCA) only"),
            ("EEG-DL", "EEG Supervised Deep Learning (Braindecode) only"),
            ("EEG-FM", "EEG Foundation Models only"),
            ("custom", "Enter a custom comma-separated classifier list"),
        ],
        default_idx=1,
        stdin=stdin,
        stdout=stdout,
    )
    if est_choice == "custom":
        stdout.write("Enter classifier names separated by commas (e.g. HC2,MRHydra,DrCIF,STC): ")
        stdout.flush()
        custom_raw = stdin.readline().strip()
        estimators: str = custom_raw if custom_raw else "headline"
    else:
        estimators = est_choice

    metric_choice = _ask_choice(
        "3. Choose evaluation metrics:",
        [
            ("all", "All 4 metrics (balacc, accuracy, auroc, logloss)"),
            ("balacc", "Balanced accuracy only"),
            ("accuracy", "Accuracy only"),
        ],
        default_idx=1,
        stdin=stdin,
        stdout=stdout,
    )
    metrics_arg: str | Sequence[str] = VALID_METRICS if metric_choice == "all" else [metric_choice]

    fmt = _ask_choice(
        "4. Choose export format:",
        [
            ("wide", "Wide CSV matrices (one CSV per metric + datasets.csv + classifiers.csv)"),
            ("long", "Tidy long-form CSV (single file with dataset & algorithm metadata)"),
            ("league", "Ranked league table CSV (mean ranks, scores, wins, coverage)"),
            ("multiverse", "Multiverse per-estimator directory tree (<estimator>/<estimator>_<metric>.csv)"),
            ("json", "Structured JSON bundle"),
        ],
        default_idx=1,
        stdin=stdin,
        stdout=stdout,
    )

    default_dest = (
        f"neuroverse_{split}_{fmt}.csv"
        if fmt in {"long", "league"}
        else (f"neuroverse_{split}.json" if fmt == "json" else f"neuroverse_{split}_{fmt}")
    )
    stdout.write(f"\n5. Destination path [Enter for {default_dest}]: ")
    stdout.flush()
    dest_raw = stdin.readline().strip()
    dest = dest_raw if dest_raw else default_dest

    written = download_results(
        dest=dest,
        format=fmt,
        metrics=metrics_arg,
        split=split,
        estimators=estimators,
        results_dir=results_dir,
    )

    stdout.write(f"\nWrote {len(written)} file(s):\n")
    for p in written[:10]:
        stdout.write(f"  - {p.resolve()}\n")
    if len(written) > 10:
        stdout.write(f"  ... and {len(written) - 10} more files in {Path(dest).resolve()}\n")

    m_repr = '"all"' if metric_choice == "all" else f'"{metric_choice}"'
    f_repr = "wide" if fmt in {"wide", "multiverse", "json"} else fmt
    stdout.write(
        "\nEquivalent Python snippet:\n"
        f'  import neuroverse as nv\n'
        f'  df = nv.load_results(metric={m_repr}, split="{split}", estimators="{estimators}", format="{f_repr}")\n'
        "\nEquivalent CLI command:\n"
        f'  neuroverse-results --split {split} --estimators "{estimators}" --metrics {metric_choice} --format {fmt} --out "{dest}"\n'
    )
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroverse-results",
        description="Interactively browse, filter, and download Neuroverse EEG benchmark results.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive step-by-step terminal wizard (default when no action flags are given).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available splits, classifier pools, metrics, and formats.",
    )
    parser.add_argument(
        "--split",
        default="core90",
        help="Benchmark split: core90 (default 30/30/30), MI, Clinical, Cognitive, all114, BCI, ME, ERP, VEP, Speech, or paper2026.",
    )
    parser.add_argument(
        "--estimators",
        default="headline",
        help="Classifier pool: headline (default, 12), canonical (81), all (109), a paradigm key, or comma-separated names.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Metric(s): all (default), balacc, accuracy, auroc, logloss, or comma-separated list.",
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="wide",
        help="Output layout: wide (default), long, league, multiverse, or json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory or file path.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a ranked league-table preview to stdout without writing files.",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Omit the Default majority-class floor from exported CSVs.",
    )
    parser.add_argument(
        "--league-mode",
        choices=("common_block", "available"),
        default="common_block",
        help="Coverage rule when --format league or --preview is used.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "local", "remote"),
        default="auto",
        help="Result source: auto (local repo if present, else GitHub), local, or remote.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh download when using remote GitHub cache.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Explicit local results directory override.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``neuroverse-results`` and ``python -m neuroverse.results``."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_args)

    if args.list:
        opts = list_result_options()
        print("Available benchmark splits:")
        for k, desc in opts["splits"].items():  # type: ignore[union-attr]
            print(f"  {k:12s} : {desc}")
        print("\nAvailable classifier pools:")
        for k, desc in opts["estimator_pools"].items():  # type: ignore[union-attr]
            print(f"  {k:14s} : {desc}")
        print(f"\nMetrics : {', '.join(VALID_METRICS)}")
        print(f"Formats : {', '.join(VALID_FORMATS)}")
        return 0

    if args.interactive or not raw_args:
        interactive_download(results_dir=args.results_dir)
        return 0

    if args.preview and args.out is None:
        tbl = load_results(
            metric=args.metrics,
            split=args.split,
            estimators=args.estimators,
            format="league",
            league_mode=args.league_mode,
            source=args.source,
            refresh=args.refresh,
            results_dir=args.results_dir,
        )
        assert isinstance(tbl, pd.DataFrame)
        cols = [
            c
            for c in ("position", "estimator", "paradigm_label", "coverage", "mean_rank", "balacc", "accuracy", "wins")
            if c in tbl.columns
        ]
        print(tbl[cols].to_string(index=False))
        return 0

    dest = args.out
    if dest is None:
        if args.format in {"long", "league"}:
            dest = Path(f"neuroverse_{args.split}_{args.format}.csv")
        elif args.format == "json":
            dest = Path(f"neuroverse_{args.split}.json")
        else:
            dest = Path(f"neuroverse_{args.split}_{args.format}")

    written = download_results(
        dest=dest,
        format=args.format,
        metrics=args.metrics,
        split=args.split,
        estimators=args.estimators,
        include_default=not args.no_default,
        league_mode=args.league_mode,
        source=args.source,
        refresh=args.refresh,
        results_dir=args.results_dir,
    )
    print(f"Wrote {len(written)} file(s) -> {dest}")
    for p in written[:8]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
