"""Registry of expected, complete and missing Neuroverse results, and a pull that uses it.

The registry lists every (classifier, dataset) cell the benchmark expects, whether its
test predictions are in the master results copy (the desktop), and what each cluster
says about it. It answers "what is still missing, and where is it coming from?", and
it makes pulling cheap: `pull` asks a cluster only for the files of cells that the
cluster reports as done and the desktop lacks, in two ssh logins, instead of listing
every file under the remote results directory.

    python tools/results_registry.py refresh iridis   # snapshot cluster task states
    python tools/results_registry.py refresh hali
    python tools/results_registry.py build            # rebuild from desktop + snapshots
    python tools/results_registry.py pull iridis      # refresh, fetch, rebuild
    python tools/results_registry.py pull hali --dry-run

`refresh` runs a short read-only probe on the cluster login node. For every
experiment config with runner state (its state_dir, by default ~/tsml-slurm/<name>),
it loads the config with the runner's own code (tsml_slurm.load_experiment,
all_tasks, is_complete, so completed.txt counts too) and reports each task as done,
running, pending, failed (with the runner's reason) or not submitted. Snapshots are
kept in results/registry/clusters/<cluster>.json, so `build` needs no connection.

Expected cells are every classifier in results/neuroverse/classifiers.csv on every
dataset in eeg.txt, plus every task of a started train/test experiment that writes to
Results/EEG, on datasets in eeg.txt. Cross-validation experiments are ignored, and so
are dormant ones (state exists but nothing was ever submitted or done).

Outputs in results/registry/: registry.csv (one row per expected cell), missing.csv
(cells not on the desktop, with the action needed) and summary.md.

Train files (trainResample0.csv) have their own registry in results/registry/train/,
with the same three files. Only the potential HC2 components write them, so its
expected cells are TRAIN_CLASSIFIERS on every dataset in eeg.txt. A cluster counts
toward a train cell only through an experiment with train_files = true. `pull` fetches
the test files of the main registry and the train files of this one.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "registry"
SNAPSHOTS = OUT / "clusters"
DEFAULTS = {
    "results": Path("D:/Results/EEG"),
    "datasets": ROOT / "eeg.txt",
    "classifiers": ROOT / "results" / "neuroverse" / "classifiers.csv",
}
# Iridis 6 and Iridis X share one home, so "iridis" covers both.
CLUSTERS = {
    "iridis": {
        "host": "ajb2u23@iridis6_b.soton.ac.uk",
        "python": "module load conda/python3 >/dev/null 2>&1; "
        "source activate tsml-eval >/dev/null 2>&1; python -",
    },
    "hali": {
        "host": "ajb@hali.uea.ac.uk",
        # login shell: Slurm (squeue) is only on PATH there
        "python": "bash -lc 'PYTHONNOUSERSITE=1 ~/.conda/envs/tsml-eval/bin/python -'",
    },
}
SKIP_FAMILIES = {"NBack", "_superseded"}
# Dropped from the benchmark (docs/evaluation.md, "Dropped Classifiers"); not expected
# even where a cluster experiment still lists them.
DROPPED = {"TRCSP-LDA-MOABB", "SSTDPN-NB", "FBLightConvNet-NB", "SleepStagerChambon2018-NB"}
TEST, TRAIN = "testResample0.csv", "trainResample0.csv"
# The four HC2 components and TRACE: the only classifiers we generate train files for.
TRAIN_CLASSIFIERS = ("Arsenal", "DrCIF", "DrCIF-500", "STC-aeon", "TDE", "TRACE")
TRAIN_OUT = OUT / "train"
SAFE = re.compile(r"^[\w.+-]+/[\w.+-]+/Predictions/[\w.+-]+/(test|train)Resample0\.csv$")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

# Read-only, stdlib plus the runner itself. Prints one JSON document.
PROBE = r'''
import glob, json, os, re, subprocess, sys
home = os.path.expanduser("~")
sys.path.insert(0, home + "/Code/tsml-eval/_tsml_research_resources/slurm")
import tsml_slurm as ts
configs = set()
for pattern in ("/Code/tsml-eval*/_tsml_research_resources/slurm/experiments/*.toml",
                "/tsml-*/*.toml", "/tsml-*/*/*.toml"):
    configs.update(glob.glob(home + pattern))
by_name = {}
for c in sorted(configs):
    try:
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', open(c).read(), re.M)
    except OSError:
        continue
    if m:
        by_name.setdefault(m.group(1), []).append(c)
queue = {}
try:
    lines = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T"],
                           capture_output=True, text=True, timeout=60).stdout.splitlines()
    for line in lines:
        if line.strip():
            job, state = line.split()
            queue[job.split("_")[0]] = state.lower()
except Exception:
    pass
out = []
# Experiments with runner state: from each config's own state_dir (which may be set
# in the config), plus any ~/tsml-slurm/<name> left without a config.
seen = set()
targets = []
for name, cfgs in sorted(by_name.items()):
    for c in cfgs:
        try:
            e = ts.load_experiment(c)
        except Exception:
            continue
        sd = str(e.state_dir)
        if sd in seen or not os.path.exists(os.path.join(sd, "state.json")):
            continue
        seen.add(sd)
        targets.append((name, c, e))
for path in glob.glob(home + "/tsml-slurm/*/state.json") + glob.glob(home + "/.tsml-slurm/*/state.json"):
    sd = os.path.dirname(path)
    if not any(os.path.samefile(sd, x) for x in seen if os.path.exists(x)):
        targets.append((os.path.basename(sd), None, None))
for name, c, e in targets:
    rec = {"experiment": name, "config": c, "error": None, "cells": []}
    if e is None:
        rec["error"] = "no config with this name found"
        out.append(rec)
        continue
    try:
        ts.read_datasets(e)
        sd = str(e.state_dir)
        state = json.load(open(os.path.join(sd, "state.json")))
        rec["cluster"] = e.cluster
        rec["results_dir"] = str(e.results_dir)
        rec["cv"] = bool(getattr(e, "cv_folds", None))
        rec["train_files"] = bool(getattr(e, "train_files", False))
        rec["stopped"] = os.path.exists(os.path.join(sd, "STOP"))
        rec["submissions"] = len(state.get("submissions", []))
        tasks = state.get("tasks", {})
        for t in ts.all_tasks(e):
            key = f"{t.estimator}|{t.dataset}|{t.resample}"
            if ts.is_complete(e, t):
                s = "done"
            else:
                r = tasks.get(key)
                if r is None:
                    s = "not submitted"
                elif r.get("terminal"):
                    s = "failed: " + str(r["terminal"]).splitlines()[0][:80]
                elif r.get("job"):
                    s = queue.get(str(r["job"]).split("_")[0], "submitted")
                else:
                    s = "waiting"
            rec["cells"].append([t.category, t.estimator, t.dataset, t.resample, s])
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    out.append(rec)
print("@@JSON@@" + json.dumps(out))
'''


def refresh(cluster: str) -> dict:
    spec = CLUSTERS[cluster]
    print(f"{datetime.now():%H:%M} probing {spec['host']}")
    run = subprocess.run(
        SSH + [spec["host"], spec["python"]], input=PROBE, capture_output=True, text=True
    )
    marker = run.stdout.rfind("@@JSON@@")
    if run.returncode != 0 or marker < 0:
        raise SystemExit(f"probe failed on {cluster}: {run.stderr.strip()[-500:]}")
    experiments = json.loads(run.stdout[marker + len("@@JSON@@"):])
    snap = {"cluster": cluster, "taken": f"{datetime.now():%Y-%m-%d %H:%M}",
            "experiments": experiments}
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    (SNAPSHOTS / f"{cluster}.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    for x in experiments:
        if x["error"]:
            print(f"  {x['experiment']}: {x['error']}")
        else:
            n = len(x["cells"])
            done = sum(c[4] == "done" for c in x["cells"])
            note = " (dormant: never submitted, ignored)" if _dormant(x) else ""
            print(f"  {x['experiment']}: {done}/{n} done{note}")
    return snap


def _dormant(x: dict) -> bool:
    """Never started (or a retired early run): nothing done and nothing ever submitted."""
    states = [c[4] for c in x["cells"]]
    return not x.get("submissions") and all(s == "not submitted" for s in states)


def _cluster_cells(datasets: set[str], train: bool = False) -> pd.DataFrame:
    """Cluster task states; with train=True only experiments that write train files."""
    rows = []
    for path in sorted(SNAPSHOTS.glob("*.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        for x in snap["experiments"]:
            if x["error"] or x.get("cv") or not x.get("results_dir", "").rstrip("/").endswith("Results/EEG"):
                continue
            if _dormant(x) or (train and not x.get("train_files")):
                continue
            for category, est, ds, resample, state in x["cells"]:
                if resample == 0 and ds in datasets:
                    rows.append((est, category, ds, snap["cluster"], x["experiment"], state, snap["taken"]))
    return pd.DataFrame(rows, columns=["classifier", "family", "dataset", "cluster",
                                       "experiment", "state", "taken"])


def _action(states: list[tuple[str, str, str]]) -> str:
    """What a missing cell needs, from its (cluster, experiment, state) entries."""
    for want, label in (("done", "pull"), ("running", "running"), ("pending", "pending"),
                        ("submitted", "submitted"), ("waiting", "waiting"),
                        ("not submitted", "queued")):
        hits = [f"{c}:{e}" for c, e, s in states if s == want]
        if hits:
            return f"{label} ({', '.join(hits)})"
    failed = [f"{c}:{e} {s[8:]}" for c, e, s in states if s.startswith("failed")]
    return f"failed ({'; '.join(failed)})" if failed else "unscheduled"


def build(results: Path, datasets_file: Path, classifiers_file: Path) -> pd.DataFrame:
    """Rebuild the test registry, then the train registry."""
    cells = _build(results, datasets_file, classifiers_file, train=False)
    _build(results, datasets_file, classifiers_file, train=True, universe=cells)
    return cells


def _build(results, datasets_file, classifiers_file, train, universe=None) -> pd.DataFrame:
    datasets = datasets_file.read_text().split()
    name, out = (TRAIN, TRAIN_OUT) if train else (TEST, OUT)
    clus = _cluster_cells(set(datasets), train)
    if train:
        # families come from the test registry, so TRACE needs no classifiers.csv row
        keep = universe.classifier.isin(TRAIN_CLASSIFIERS)
        cells = universe.loc[keep, ["classifier", "family", "dataset"]].reset_index(drop=True)
    else:
        raw_clf = pd.read_csv(classifiers_file)
        if "source_classifier" in raw_clf.columns:
            raw_clf = raw_clf.assign(classifier=raw_clf["source_classifier"])
        base = raw_clf.merge(pd.Series(datasets, name="dataset"), how="cross")
        cells = pd.concat([base[["classifier", "family", "dataset"]],
                           clus[["classifier", "family", "dataset"]]])
        cells = cells.drop_duplicates(["classifier", "dataset"])
        cells = cells[~cells.family.isin(SKIP_FAMILIES) & ~cells.classifier.isin(DROPPED)]
        cells = cells.reset_index(drop=True)

    def done(r):
        f = results / r.family / r.classifier / "Predictions" / r.dataset / name
        return f.is_file() and f.stat().st_size > 0

    cells["status"] = ["complete" if done(r) else "missing" for r in cells.itertuples()]
    info = {k: list(zip(g.cluster, g.experiment, g.state))
            for k, g in clus.groupby(["classifier", "dataset"])}
    cells["clusters"] = [";".join(f"{c}:{e}={s}" for c, e, s in info.get((r.classifier, r.dataset), []))
                         for r in cells.itertuples()]
    cells["action"] = ["" if r.status == "complete" else _action(info.get((r.classifier, r.dataset), []))
                       for r in cells.itertuples()]
    cells = cells.sort_values(["classifier", "dataset"], key=lambda s: s.str.lower())
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "registry.csv", index=False)
    missing = cells[cells.status == "missing"]
    missing.to_csv(out / "missing.csv", index=False)
    _summary(cells, missing, results, out, train)
    pulls = missing.action.str.startswith("pull").sum()
    print(f"{'train' if train else 'test'}: {len(cells)} expected cells, "
          f"{len(cells) - len(missing)} on the desktop, {len(missing)} missing, "
          f"{pulls} of them ready to pull -> {out}")
    return cells


def _summary(cells, missing, results, out=OUT, train=False):
    kind = missing.action.str.extract(r"^(\w[\w ]*?)(?: \(|$)")[0].fillna("unscheduled")
    by_kind = kind.value_counts()
    by_clf = cells.groupby("classifier").agg(
        expected=("status", "size"), complete=("status", lambda s: (s == "complete").sum()))
    by_clf["missing"] = by_clf.expected - by_clf.complete
    gaps = by_clf[by_clf.missing > 0].sort_values("missing", ascending=False)
    kinds = missing.assign(kind=kind).groupby(["classifier", "kind"]).size().unstack(fill_value=0)
    snaps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SNAPSHOTS.glob("*.json"))]
    cols = [k for k in ("pull", "running", "pending", "submitted", "waiting", "queued", "failed",
                        "unscheduled")
            if k in kinds.columns]
    lines = [
        "# Train file registry" if train else "# Results registry", "",
        *([f"`{TRAIN}` for {', '.join(TRAIN_CLASSIFIERS)} on every dataset. Only experiments",
           "with train_files = true count. The test registry is one level up.", ""]
          if train else []),
        f"Built {datetime.now():%Y-%m-%d %H:%M} from `{results.as_posix()}` by "
        "`tools/results_registry.py build`. Regenerate rather than edit.", "",
        "Cluster snapshots: " + (", ".join(f"{s['cluster']} {s['taken']}" for s in snaps) or "none"), "",
        f"- Expected cells: {len(cells)}",
        f"- On the desktop: {len(cells) - len(missing)} ({1 - len(missing) / len(cells):.1%})",
        f"- Missing: {len(missing)}", "",
        "## Missing cells by state", "",
        "`pull`: done on a cluster, not yet copied. `running`/`pending`: in the Slurm queue.",
        "`submitted`: submitted, not in the queue at snapshot time (finishing or lost; the",
        "runner resolves it next cycle). `waiting`: due for resubmission (e.g. a higher memory",
        "tier). `queued`: in an experiment, not yet submitted. `failed`: the runner gave up",
        "(reason in missing.csv). `unscheduled`: no started experiment on a snapshotted",
        "cluster runs it (Hali experiments count only after `refresh hali`).", "",
        "| state | cells |", "| :--- | ---: |",
        *[f"| {k} | {v} |" for k, v in by_kind.items()], "",
        "## Classifiers with missing cells", "",
        "| classifier | on desktop | expected | " + " | ".join(cols) + " |",
        "| :--- | ---: | ---: | " + " | ".join("---:" for _ in cols) + " |",
        *[f"| {k} | {r.complete} | {r.expected} | "
          + " | ".join(str(kinds.loc[k, c]) if k in kinds.index else "0" for c in cols) + " |"
          for k, r in gaps.iterrows()], "",
        "## Use", "",
        "```",
        "python tools/results_registry.py refresh iridis   # or hali: snapshot task states",
        "python tools/results_registry.py pull iridis      # refresh, fetch ready cells, rebuild",
        "python tools/results_registry.py build            # offline rebuild",
        "```", "",
        "`pull` fetches only the files of cells the cluster reports as done and the desktop",
        "lacks: test files for this registry, train files for registry/train. It never changes anything remote and never overwrites a local file.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def pull(cluster: str, results: Path, dry_run: bool) -> int:
    host = CLUSTERS[cluster]["host"]
    wanted, ready = [], []
    for out, name, label in ((OUT, TEST, "test"), (TRAIN_OUT, TRAIN, "train")):
        path = out / "missing.csv"
        if not path.exists():
            continue
        missing = pd.read_csv(path).fillna("")
        hits = missing[missing.action.str.startswith("pull")
                       & missing.action.str.contains(f"{cluster}:")]
        ready.append(hits)
        wanted += [f"{r.family}/{r.classifier}/Predictions/{r.dataset}/{name}"
                   for r in hits.itertuples()]
        print(f"{len(hits)} {label} cells ready on {cluster}")
    ready = pd.concat(ready)
    # Every EEG experiment writes to ~/Results/EEG (paths differ only by mount alias).
    remote = ["Results/EEG"]
    wanted = [w for w in wanted if SAFE.match(w) and not (results / w).exists()]
    print(f"asking {cluster} for {len(wanted)} files")
    if dry_run or not wanted:
        for r in ready.head(40).itertuples():
            print(f"   {r.family}/{r.classifier} {r.dataset}")
        return 0
    script = (f"cd {remote[0]} && while IFS= read -r f; do [ -s \"$f\" ] && printf '%s\\n' \"$f\"; "
              "done | tar cf - -T - | gzip -1")
    stream = subprocess.run(SSH + [host, script], input=("\n".join(wanted) + "\n").encode(),
                            capture_output=True)
    if stream.returncode != 0 or not stream.stdout:
        print(stream.stderr.decode(errors="replace").strip() or "transfer failed")
        return 1
    n = 0
    with tarfile.open(fileobj=io.BytesIO(stream.stdout), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = results / m.name
            if not (m.isfile() and SAFE.match(m.name)) or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
            n += 1
    print(f"{datetime.now():%H:%M} fetched {n} files into {results.as_posix()}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="snapshot task states on a cluster (read-only)")
    r.add_argument("cluster", choices=sorted(CLUSTERS))
    b = sub.add_parser("build", help="rebuild the registry from the desktop and snapshots")
    pl = sub.add_parser("pull", help="refresh, fetch cells done on the cluster, rebuild")
    pl.add_argument("cluster", choices=sorted(CLUSTERS))
    pl.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    pl.add_argument("--no-refresh", action="store_true", help="use the saved snapshot")
    for q in (r, b, pl):
        q.add_argument("--results", type=Path, default=DEFAULTS["results"])
        q.add_argument("--datasets", type=Path, default=DEFAULTS["datasets"])
        q.add_argument("--classifiers", type=Path, default=DEFAULTS["classifiers"])
    a = p.parse_args(argv)
    args = (a.results, a.datasets, a.classifiers)
    if a.cmd == "refresh":
        refresh(a.cluster)
        build(*args)
        return 0
    if a.cmd == "build":
        build(*args)
        return 0
    if not a.no_refresh:
        refresh(a.cluster)
    build(*args)
    code = pull(a.cluster, a.results, a.dry_run)
    if code == 0 and not a.dry_run:
        build(*args)
    return code


if __name__ == "__main__":
    sys.exit(main())
