# Neural Interfaces2026 Track2: local classification preparation

Competition: https://www.codabench.org/competitions/17982/ . This is a separate
competition workspace, not an addition to the76-problem archive study.

The user is not registered. Registration is closed; public local training and
evaluation remain possible. No submission, organiser contact or sealed-data
access has been performed. The user paused the preparation, then explicitly
resumed it10November2026. Consult live worker state before starting another job:
`.openeegbench/competition-17982/state.json`, `run.log`, `run.err`.

## Authoritative sources and corrections

- [Current track specification](https://neural-interfaces26.github.io/tracks.html).
- [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/7160f93ae5e86249e7ee5ce0583e34ec7163c521/codabench/pages/scientific_faq_bci.md).
- [Executable objective](https://github.com/neural-interfaces26/2026-competition/blob/7160f93ae5e86249e7ee5ce0583e34ec7163c521/tracks/bci_decoding/objective.py).
- [Official NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html).
- [Rules](https://neural-interfaces26.github.io/rules.html).

Use the current FAQ/executable contract over older descriptive paragraphs.
Earlier guide text describes47signals,20people and a purely within-subject
split; that does not describe the public release or current sealed protocol.
Sealed evaluation includes later sessions of represented participants AND new
participants. Their identities/composition are private. No labels, calibration
examples or IDs are supplied to models through the sealed prediction stream.
The public split described below is a local benchmark, not the sealed split.

## Public data and access

Original public source: [PROTEUS BCI Bordeaux, NEMAR nm000290 v1.0.0](https://nemar.org/dataset/nm000290).
CC-BY-4.0 directly verified in the distributed dataset description and README.
Authors: Dreyer Pauline, Bourdil Manon, Bechon Loic, Kojima Simon, Velut Sebastien,
Rimbert Sebastien, Roy Raphaelle and Lotte Fabien.

The public release has10participants,14sessions,112runs,41EEG channels at500Hz.
Two people have3sessions, eight have1. Runs cover eyes-open/closed rest,
Graz/BrainHero acquisition with sham feedback, and online feedback runs.
Resting runs have no command cues and are not converted into command examples.
Their metadata are retained; the downloader skips their unused EDF signals.

Download via the versioned original manifest:
https://data.nemar.org/nm000290/v1.0.0/manifest.json . It pins original file
sizes and SHA256 or Git-blob SHA1 digests. Manifest SHA256:
`646d0f1347599c69b77ca9cc9b673cf80c2100c82ec77d3c9f1959ff9e2e61bb`.
The formatter refuses a changed manifest, checks every source file and records
its SHA256. Downloads use atomic partial files and verified HTTP Range resume;
if the server ignores Range, the file is restarted rather than concatenated.

Full source is approximately1.59GB; the publisher also offers an approximately
1.25GBcompressed ZIP. Current individual-file throughput was slow. No sealed
objects or external datasets are part of this first preparation.

## Model-facing arrays

Each example is a cue-locked window from1to5seconds after the cue, end-exclusive.
Labels follow the actual submission contract:

| Encoded | Native label/code | Meaning |
|---:|---|---|
|0|mi /33024|Kinesthetic imagery of the non-dominant hand|
|1|sub /33025|Mental subtraction|
|2|word /33026|Word generation from a letter|

Native EDF annotations corroborate BIDS cue labels/times before conversion.
Sensor names and order must agree between EDF and BIDS and across every run.
The required41channels are retained, including any reported flat channels:
changing the channel count would violate this competition contract. Flatness is
reported; the archive's global sensor-removal policy is not applied here.

Preprocessing mirrors inspected NeuralSet0.4 code: calibrated MNE SI decode;
notch50/60Hz plus their harmonics below native Nyquist; zero-phase FIR0.1–75Hz;
whole-recording resampling to120Hz; float32 SI cache boundary followed by float32
microvolt scaling. Final arrays are float64 representations of these exact
float32 model-facing numbers. This intentionally matches the starter's precision
boundary rather than claiming retention of all filtered float64 precision.
Native EDFs remain unchanged. No learned per-recording scaling, clipping,
reference change or baseline correction is added.

Window indices use round((cue_onset+1)*120);480samples each. The output is
`(cases,41,480)`, numeric microvolts. Native source sample/time bounds, cue row,
source file, subject/session/run, interface and encoded/original labels remain
in case sidecars. Native trial IDs are unknown; event-table rows are labelled as
such rather than presented as original trial identifiers. Filtering uses whole
recordings, so native interval bounds describe timing, not finite filter support.

This implementation has been reviewed against the source but has NOT yet been
numerically compared end-to-end against a complete NeuralBench installation.
Environment versions are pinned in `tools/requirements-competition-track2.txt`
and recorded in metadata. Local scores must not be called official scores.

## Output layout and local evaluation

Sources: `D:/Data/EEGSources/Competitions/NeuralInterfaces2026Track2/`.
Outputs: `D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/`:

```text
pilot/                       # one-recording source/pipeline check
prepared/
  X.npy                      # all public command cases, memory-map compatible
  y.npy                      # labels0/1/2, row-aligned
  cases.tsv                  # original strings; keep leading zeros
  channels.tsv               # exact array order and original BIDS sensor fields
  splits.json                # index lists, no copies of cases
  metadata.json              # protocol, original run metadata, checks/versions
  artifacts.json             # prepared-artifact sizes and SHA256
  source/                    # original manifest and global provenance
  subject_holdout/
    ProteusCommands_TRAIN.ts # development train+validation people
    ProteusCommands_TEST.ts  # held-out people
    cases_train.tsv
    cases_test.tsv
  cross_session/
    ProteusCommands_TRAIN.ts # earlier public sessions + other public people
    ProteusCommands_TEST.ts  # latest public session of repeated people only
    cases_train.tsv
    cases_test.tsv
```

Subject holdout follows the starter's two-stage sklearn split, seed33:
60%train/20%validation/20%test by whole person, all their sessions together.
Validation is retained as indices for tuning; TRAIN.ts merges train+validation,
with development membership retained in metadata. Do not tune on TEST.ts.
Only2public people support a chronological session test; this is a limited
check of session drift, not a proxy for the full sealed cohort. All examples
stay in source subject/session/numeric-run/cue order.

Fit classifier transformations on training data only. During evaluation,
`predict` receives ONLY one EEG example `(1,41,480)` at a time. Deep-copy the
fitted model and call optional reset_state at each new subject/session. State
may persist between runs/interfaces within a session; no future windows,
labels or metadata reach the model and no state crosses sessions.

The headline local score matches the objective's grouping: balanced accuracy
within each subject x session x interface cell, then equal mean across cells.
Session-only mean, pooled balanced accuracy and plain accuracy are diagnostics.
Absent-but-predicted classes contribute zero recall, matching the pinned
official helper. An untuned Welch bandpower/standardised logistic-regression
baseline exercises actual fitting and streamed prediction. This is not a
foundation-model result or a competition submission.

## Running and loading

In an isolated scientific Python environment with the pinned dependencies:

```powershell
python -m tools.competition_track2 download --pilot
python -m tools.competition_track2 prepare --pilot
python -m tools.competition_track2 download
python -m tools.competition_track2 prepare
python -m tools.competition_track2 baseline
python -m tools.competition_track2 audit
```

`--cache` and `--output` override the directories. A completed output is never
silently overwritten. On this desktop, the active ignored `run.py` bootstraps
the existing scientific overlays and chains these stages, stopping on any error.
Inspect its live state before resuming or starting another writer. A partial
pilot from a corrected JSON-serialization error is retained separately as
`pilot-incomplete-json-review`; it is not a validated preparation.

```python
import json
from pathlib import Path
import numpy as np
from aeon.datasets import load_from_ts_file

root = Path('D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/prepared')
X = np.load(root / 'X.npy', mmap_mode='r')
y = np.load(root / 'y.npy')
splits = json.loads((root / 'splits.json').read_text())
X_train, y_train = X[splits['train']], y[splits['train']]
X_val, y_val = X[splits['val']], y[splits['val']]
# Or load the combined-training / held-out-subject TS problem:
X_trainval, y_trainval = load_from_ts_file(root / 'subject_holdout/ProteusCommands_TRAIN.ts')
```

Pilot/final reports: `docs/competition-track2-pilot-result.json`,
`docs/competition-track2-preparation-result.json` and
`docs/competition-track2-baseline-results.json` appear only as stages complete.
Source/case duplicates, full numeric TS roundtrip, metadata alignment, sensor
order, finite/flat checks and source annotation correspondence are verified.
Tests cover cue boundaries, split isolation, metric weighting, state reset,
range resume and native numeric metadata. No leaderboard inference should be
made from a public score until its selected phase and metric are checked.

## Completed preparation,10November2026

Full source download, conversion, exact TS readback, case-row/source-sensor
alignment, duplicate/finite/flat checks, artifact audit and aeon classifier
fit/predict smoke passed. All2655cases have41channels x480points at120Hz.
Class counts: motor imagery885, mental subtraction886, word generation884.
No exact duplicates, constant or effectively flat case/channel pairs found.
Nine focused regression tests passed and the local cell metric was checked
against the pinned official helper. No sealed score or submission produced.

| Local protocol | TRAIN | TEST | Untuned bandpower cell-mean balanced accuracy |
|---|---:|---:|---:|
| Subject holdout |1900|755|0.389883|
| Later public sessions |2268|387|0.405247|

The6/2/2development split contains1521train/379validation/755test cases.
Validation-only baseline score0.410725. Subject-holdout TRAIN.ts combines the
first two partitions; their original development memberships remain in cases.tsv.
Repeated-person test sessions belong to P04 and P10. They intentionally share
people with chronological training, but never cases or source recordings.
Prepared output includes portable converter/requirements and a copy of this guide.

### Leave-one-subject-out evaluation

`tools/loso_track2_selected.py` evaluates MRHydra and QUANT with ten folds,
holding out one complete participant (all sessions, runs and interfaces) at a
time. Every fold fits a fresh model on the other nine participants, with seed33
and the same fixed classifier settings as the initial development run. Neither
feature fitting nor classifier fitting uses the held-out participant. This
evaluation includes all public subjects, including the earlier P04/P09 holdout;
that holdout is no longer untouched once these results are examined.

The run started on10November2026. Check
`D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/selected-loso/state.json`
for live status. That folder holds resumable per-fold receipts and compressed
NumPy predictions with original case indices and true labels. Final results
are written to `docs/competition-track2-loso-results.json`. The primary overall
score averages subject/session/interface cell balanced accuracies; a separate
equal-subject average makes the influence of participants with more sessions
explicit. An equal-score MRHydra/QUANT ensemble is evaluated using matched
out-of-fold scores, with no fitted calibration or ensemble weights.

LOSO does not modify the separately fitted all-public submission models. It is
a local public-data evaluation and does not consume a competition submission.
