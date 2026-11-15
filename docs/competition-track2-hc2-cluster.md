# Track2 HC2 components on Iridis6

11November update: all four production components completed with exit0. Their
eight prediction CSVs were retrieved to the separate competition results tree,
and the standard offline builder produced HC2FromFile predictions locally.
See [latest report](iridis-progress-20261111.md). Tony explicitly retained the
18.192GB of fitted checkpoints on Iridis. No model checkpoint was downloaded or
deleted. Diagnostic TEST scores are in-sample, not competition performance.

Requested 10 November 2026: do not submit QUANT; train HC2's four components
separately on the cluster using the tsml-eval experimental structure, retaining
train files for offline assembly. Use **DrCIF-500**, not default DrCIF.

Latest verified status: preflight job4888596 passed, including every case hash
and all four native persistence checks. Full component jobs are submitted:
Arsenal4888747, DrCIF-5004888748, STC-aeon4888749 and TDE4888750, all pending at
the last queue check. Supervisor4888635 is queued. Recorded evidence:
[preflight and queue receipt](competition-track2-hc2-preflight.json).
Three local source/configuration guard tests also passed.

The initial four submissions were replaced only to prevent inheriting the
preflight's four-thread Numba ceiling; production explicitly sets eight threads.
All four saved job scripts were checked for that setting and mail type NONE.

Startup correction: jobs4888654–4888657 failed before training because the payload
manifest mistakenly included a desktop-generated `.pyc`. The source verifier
now checks only the five actual overlay source files, and manifests/bundles
exclude `__pycache__` and `.pyc`. A regression test checks that cache changes are
ignored while source changes still fail. Four local tests passed. Only the
dedicated competition experiment was retried, producing the current IDs above.

Latest startup check: Arsenal4888747 and DrCIF-5004888748 are RUNNING on red6100,
past source verification and inside the standard threaded experiment entry point.
STC-aeon4888749 and TDE4888750 are pending at the per-user node limit.

## Experiment and data

The reproducible local experiment is
[competition17982_hc2_components.toml](../tools/slurm/competition17982_hc2_components.toml).
It uses the existing `tsml_slurm.py` controller and threaded classification entry
point, with `train_files=true`, resample0 and explicit classifier seed33.

| Component | Factory name | Result category |
|---|---|---|
| Arsenal | Arsenal | ConvolutionBased |
| DrCIF with500trees | DrCIF-500 | IntervalBased |
| aeon shapelet transform classifier | STC-aeon | ShapeletBased |
| Temporal Dictionary Ensemble | TDE | DictionaryBased |

No default DrCIF, QUANT, MRHydra, monolithic HC2 or archive-wide experiment is
included. Component settings otherwise follow the existing factory defaults;
verbose progress is enabled.

`ProteusCommandsAllPublic_TRAIN.ts` contains all2655public cases, in original
case order,41channels by480samples. TEST contains one repeated TRAIN case for
each class, solely to exercise standard prediction/results/model loading. It is
an **in-sample diagnostic**, not a holdout, cross-validation or competition score.
The three original case indices are2,1,0 in encoded-class order0,1,2. Original
subject IDs, labels and per-case signal hashes accompany both files. There is
no alteration of the earlier development splits or registered archive datasets.

The train files use each component's native `fit_predict_proba` estimates.
They are not ordinary predictions on the fitted training data. Their specific
internal estimation methods are those of the installed checkpoint-enabled aeon
implementation. These estimates are suitable for standard HC2 weighting but
do not establish unseen-subject generalisation.

## Paths and isolation

Local staging:
`D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/hc2-cluster-stage/`.
The manifest is also recorded in `competition-track2-hc2-stage.json` in this
directory. Payload size is approximately1.023GB; large files stay outside Git.

Remote paths, all inside Tony's own space:

```text
/home/ajb2u23/Competitions/NeuralInterfaces2026Track2/HC2/
    data/ProteusCommandsAllPublic/
    experiment.toml
    manifest.json
    preflight-result.json             # written only after successful checks
    preflight-<jobid>.{out,err}
    preflight-exit-code.txt
    submission.json
    state/{state.json,report.txt,...}   # standard controller state
    results/<Category>/<Component>/Predictions/ProteusCommandsAllPublic/
        trainResample0.csv
        testResample0.csv              # diagnostic only
    results/<Category>/<Component>/Checkpoints/ProteusCommandsAllPublic/
        Resample0.ckpt                 # retained full fitted model

/home/ajb2u23/Code/tsml-eval-proteus-hc2-20261110/
```

The competition checkout is a separate detached local worktree based on
tsml-eval commit `ce0d1131dca7c2f432ae005162284acfe28ce4be`. The shared farm checkout
and existing jobs' source files are not edited. The existing CPU environment
already reports aeon1.6.0 from commit
`a507e2b86fde69086d0560b424f82b0d87a9cd36`, branch
`ajb/hc2-components-checkpoint`. No cluster installation or external fetch is used.

## Small opt-in runner extension

The standard runner deletes completed checkpoints. Retaining only prediction CSVs
would let us combine known-case results but would not let us predict sealed
competition cases. A reviewed source overlay therefore adds `--keep_checkpoint`
to classification experiments. The default remains deletion; the new flag
requires a checkpointable classifier and `-ch`. The experiment checks for a saved
model before writing its final test output.

The threaded entry point is also made to forward the already-existing checkpoint
and interval arguments, which the inspected version omitted. It forwards CV and
train-only-transform arguments consistently too. A fifth overlay file makes
supervisor mail type NONE when email is disabled. `email=""` suppresses controller
report email. This experiment must originate no external calls from the cluster.

The overlay is generated by
[prepare_track2_hc2_cluster.py](../tools/prepare_track2_hc2_cluster.py), with guarded
source substitutions and syntax checks. Every compute job verifies overlay file
hashes against the staged manifest. The isolated checkout intentionally has
reviewed changes, so `require_clean=false`; base-commit pinning and overlay hashes
record the complete intended source instead. No base repository branch is moved.

## Resources and validation before full fits

Four Slurm CPU array tasks, at most four active, eight CPUs per component,
64GB initial RAM with128/256/620GB escalation tiers,60h wall time. Periodic
checkpoint interval60minutes; supported resumptions retry timeouts, at most four
normal attempts. A small Slurm supervisor checks every15minutes. Nothing trains
or performs large data validation on a login node. Existing jobs are not cancelled.

A separate batch preflight requests four CPUs,16GB and45minutes. It:

1. Verifies the uploaded manifest and installed overlay hashes.
2. Loads every TRAIN/TEST case, checking shape, finite values, encoded label and
   exact signal hash against the original row metadata.
3. Confirms the factory gives DrCIF-500 exactly500trees.
4. Runs tiny reduced versions of all four native components through the standard
   experiment function, retains their final checkpoints, reloads them and checks
   their probabilities against saved test files and presence of train files.
5. Runs standard `check`, then `submit --dry-run --no-email`, then `start` only if
   every prior check succeeded.

Preflight failures leave the production components unsubmitted. Local syntax
checks passed and an Arsenal persistence smoke passed. The desktop aeon checkout
does not expose the same DrCIF checkpoint capability as the cluster build, so
full four-component persistence verification is deliberately a compute-node gate.
Do not claim it passed until `preflight-result.json` exists with `passed=true`.

The desktop transfer/submission helper is
[deploy_track2_hc2_cluster.py](../tools/deploy_track2_hc2_cluster.py).
It refuses existing target directories to prevent accidental overwrites. Do not
rerun it blindly after a partial transfer: inspect the dedicated paths first.

## Offline assembly after completion

Fetch only this experiment's prediction files, retained checkpoints and metadata
to a separate competition output directory on the desktop. Verify case counts,
labels, probabilities, component parameters and checkpoint hashes. Keep package
versions with the trusted model artifacts.

The existing `build_hc2_from_files.py` already defaults to Arsenal, DrCIF-500,
STC-aeon and TDE. Standard CAWPE uses train-estimate accuracy raised to power4 to
weight component probabilities. For live competition inference, retain/load the
actual component models and apply the same weights to their new predictions.
Combining the three diagnostic TEST rows alone is not a competition submission.

Native pickle/checkpoint persistence does not establish Codabench compatibility:
the current worker does not provide aeon. Portable HC2 inference/package size,
runtime and official-stream replay remain a later task. No HC2 submission is
authorised or uploaded by this training request.

## Review status

Deployment/job receipt: `competition-track2-hc2-submission.json` once received.
This guide records design and gates; consult the receipt and the dedicated
remote state/logs for actual execution progress. The first scheduling request
does not mean the full components have begun fitting.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
