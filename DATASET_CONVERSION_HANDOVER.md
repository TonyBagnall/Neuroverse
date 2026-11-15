# EEG dataset conversion: agent handover

Entry point for an agent continuing archive assimilation. Read this before downloading or converting. The user authorises continuing through eligible batches without repeatedly asking permission. Read live state before relying on the snapshot below.

## Hali list synced and GPU gaps surveyed,11November2026

User authorised sync dataset lists/push any new data/see GPU gaps. Hali's own
~/DataSetLists/eeg.txt now canonical76 (backup eeg.txt.bak-20261111-084336).
Five previously missing list names added; excluded Blink/EpilepticSeizures/Sleep
removed. All export directories/files already existed with matching names/sizes,
so zero data bytes needed pushing. All88available ID sidecars matched SHA256;
dataset-info84rows cover all76, NBack200Hz correct. No large array rehashing.

See [GPU next-batch report](docs/hali-gpu-next-batch-20261111.md), raw JSON survey
and CSV task matrix. Existing23Hali GPU models have115new-list gaps and21existing
terminal failures (FBLightConvNet12, SleepStagerChambon9). New five datasets are
large: do shape/resource preflights on compute nodes before a restart. In
particular SymbolAttention becomes120points at120Hz, below FBLightConvNet's250
window. No GPU jobs were started; user asked to survey, not restart yet.

Additional draft8architectures unregistered in factory; plain EEGNet/DeepConvNet/
EEGConformer variants have228missing under distinct non-NB identities. DGCNN and
EEGSym need channel metadata. Future GPU restart must disable inherited report
email AND unconditional supervisor FAIL mail. Current live code/envs not modified.
List changes may be picked up by active CPU supervisors; no manual job changes.

## Latest Hali retrieval,11November2026

User switched to UEA VPN and requested Hali progress/results. See
[Hali report](docs/hali-progress-20261111.md). Authenticated as ajb. Slurm path
/opt/slurm/25.11.4/bin/squeue (not on noninteractive PATH). Own queue27CPU jobs
running (24EEG+3Multiverse TDE),3pending entries; no GPU work running. Many GPU
batches settled with some terminal short-input/model compatibility failures.
Fetched/validated806new/newer prediction CSVs (118.6MB) into D:/Results/EEG;
zero invalid files. No remote changes or external calls. Competition stays separate.

Transfer helper now skips older remote versions. Cross-cluster audit caught57
older Iridis copies from the earlier pull; newer local files restored from
backups, older Iridis variants preserved separately. See reconciliation receipt.
Hali list74still has excluded Blink/EpilepticSeizures/Sleep and lacks five of
Iridis's76current problems; no list changes made. Raw snapshots under
.openeegbench/hali-*.json. No queued work restarted and no checkpoints deleted.

## Latest Iridis retrieval,11November2026

Read [progress/retrieval report](docs/iridis-progress-20261111.md). Fetched and
validated1573new/changed archive prediction files (260MB) into D:/Results/EEG,
backing up overwritten local files. Competition results are separate under
D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/hc2-cluster-results/results.
All four competition HC2 components COMPLETED; eight CSVs fetched, standard
tsml-eval offline builder created HC2FromFile predictions there. TEST is three
in-sample diagnostic cases; no valid held-out performance claim. Retained models
are still on Iridis (18.192GB total). User explicitly said keep them.

User requested deleting checkpoints of complete problems. Archive inventory:
21files/807.94GB, zero complete+inactive eligible tasks; no deletion performed.
No local archive ckpts found. Do not delete competition models. Live queues read
only; no cluster jobs/configs changed in this retrieval. MRHydra Codabench upload
reported failed; async request for error/log pending, no resubmission authorised.

Cluster access: read [the confirmed cluster rules](docs/cluster-access.md).
No external calls from clusters, including pip; Tony handles those himself.
Access only his own queues/jobs and file space. Local archive-download authority
does not override these remote restrictions.
Heavy computation must use Slurm compute allocations, never login nodes.
The locally inspected tsml-eval system is explained in
[farm-management reference](docs/tsml-eval-farm-management.md), including the
state-mutating `status` command, farm top-up, retries, registries and automatic
email conflict with the no-external-calls rule. Review only; no cluster changes.

## HC2 competition components on Iridis6,10November2026

User declined QUANT submission and authorised HC2 component builds on the cluster,
explicitly using tsml-eval's experiment structure and DrCIF-500 (not default).
See [full run guide](docs/competition-track2-hc2-cluster.md) and
`docs/competition-track2-hc2-submission.json`. Uploaded all2655public training
cases and metadata; three repeated training cases form diagnostic TEST only.
Preflight Slurm job4888596 was submitted. It checks all data hashes, four native
model reloads, and standard experiment check/dry-run before starting production.
Do not claim production is running until the preflight passes and job IDs appear.

Update: preflight PASSED. Full model jobs submitted: Arsenal4888654,
DrCIF-5004888655, STC-aeon4888656, TDE4888657; pending at latest queue check.
Supervisor4888635 remains queued. First four job IDs4888631-4888634 were replaced
only to apply an explicit8-thread Numba ceiling rather than inherit preflight's4.
All final scripts verified eight-thread export and mail NONE. See
`docs/competition-track2-hc2-preflight.json`; native smoke used aeon1.6.0 and
sklearn1.8.0. All case hashes/labels and all four model reloads passed;3local
configuration/overlay tests passed. Read dedicated queue/state before further
actions. No changes/cancellations to the pre-existing archive farms.

Later startup fix: component jobs4888654–4888657 failed before fitting because a
desktop-generated pyc was mistakenly included as source in the payload manifest.
Verifier now checks only the5 `.py` overlay files; bundler/manifests exclude
`__pycache__`/pyc. Regression test confirms bytecode is ignored and real source
changes fail;4tests passed. Retried only the dedicated experiment: current jobs
Arsenal4888747, DrCIF-5004888748, STC-aeon4888749, TDE4888750. Submission receipt
updated; earlier preflight/queue receipt is historical. Check actual startup.

Remote dedicated stage: /home/ajb2u23/Competitions/NeuralInterfaces2026Track2/HC2;
isolated worktree: /home/ajb2u23/Code/tsml-eval-proteus-hc2-20261110. Existing
farm checkout/running source files untouched. Standard controller and threaded
entry point; opt-in --keep_checkpoint overlay retains final models; threaded
checkpoint forwarding fixed in this copy, email disabled including supervisor
mail. Hashes pin five overlay files, base commit ce0d113. Existing aeon1.6.0
checkpoint branch is used; no external cluster calls or installations.

Four components only: Arsenal, DrCIF-500, STC-aeon, TDE. Seed33; eight CPUs,
64GB initial per component,60h, bounded checkpoint/resume retries. Outputs stay
inside the dedicated competition directory, not the archive results. Native
train estimates support offline CAWPE weights; preserved full checkpoints are
needed for sealed-case inference. No HC2 submission yet. Desktop local four-way
smoke was gated by a different DrCIF checkpoint API; cluster compute preflight
must pass. First desktop Arsenal persistence test passed.

## Downloads resumed,10November2026

User requested continuing downloads. Psychosis1/ds003944 full native download
resumed using the existing pinned original CC0 source and checksum-verifying
downloader. Source total6.606GB, previously cached0.542GB. D had72.62GB free.
Live receipt/log: `.openeegbench/eegdash-diagnosis/psychosis1-download-full.{json,log}`.
Downloader launcher PID46396; inspect live receipt/process rather than assume PID.
No new full dataset has been admitted by this download restart.

Continuation supervisor launcher PID3832, reusable implementation
`tools/resume_eegdash_downloads.py`, local bootstrap
`.openeegbench/eegdash-diagnosis/resume-downloads.py`. State/logs are
`.openeegbench/eegdash-diagnosis/download-continuation.{json,log,err}`.
It waits for the first download, then LOSO completion before any DSP; runs both
mixed-rate pilots in separate processes using current authorised anti-alias
resampling adapter; only then downloads psychosis2/ds003947 (13.466GB total),
requiring source size plus30GB free reserve. Stops on any failed gate. Existing
psychosis1 pilot succeeded at1000Hz but its additional3000Hz participant still
needs the current resampling pilot. Old psychosis2 pilot failure predates the
current resampling implementation; never bypass native/BIDS consistency gates.
No automatic full conversion/registration yet: review full-source recording
overlap and projected .ts footprint. Avoid launching duplicate downloaders or
conversion jobs while this controller is active.

Later update: first full source download completed. Initial continuation stopped
on exact float comparison: native header division produces3000.0003000000297Hz,
one ULP from the reviewed literal3000.00030000003. `harmonise_rate` now recognises
only the reviewed rates with rtol1e-12/atol0, while still resampling with the exact
native frequency. Regression test now uses the real header calculation;8tests
passed in the scientific runtime. LOSO is complete, so continuation restarted
via launcher PID49804; logs `download-continuation-resumed.{log,err}`. Read live
state for actual PID/stage. Both mixed-rate pilots must still pass before the
second full source download. The earlier exact-1000Hz pilot did not cover the
additional3000Hz participant; do not describe the resampled pilot as complete yet.

## Active competition LOSO evaluation,10November2026

Later update: LOSO is complete. Cell-mean BA: MRHydra0.451574,
QUANT0.435845, equal ensemble0.464418 (see exact JSON). User explicitly authorised
submitting MRHydra. Packaging worker `.openeegbench/competition-17982/package-mrhydra.py`
is exporting MRHydra only; live state `mrhydra-package-state.json`, logs
`mrhydra-package.{log,err}`. All-public ZIP path and upload/check instructions:
`docs/competition-track2-mrhydra-submission.md`. Browser control failed during
both available runtime initialisation attempts (kernel assets path error), so
no upload was made. Finish package/isolated/replay checks before presenting as
ready. No extra submission attempts or organiser messages authorised.

MRHydra packaging is now complete: all379development predictions match native
aeon and official objective replay passed; all-public120case sample parity and
isolated12case ZIP checks passed. Final all-public ZIP2201934bytes, SHA256
2f43a9622000aabd6f005bc4e05fde2f28398879dd7b76449ffea32b018164b4.
Browser runtime retry after reset failed too. No submission has been uploaded;
manual upload handover is required. Submission guide records paths and checks.

User requested leave-one-subject-out evaluation for MRHydra and QUANT.
Worker restarted with supervisor PID14496 using `.openeegbench/competition-17982/loso-selected.py`;
check the live process/state rather than relying on that PID. Public reusable
implementation: `tools/loso_track2_selected.py`; split tests: 2 passed.
Current logs: `.openeegbench/competition-17982/loso-selected-isolated.{log,err}`.
First attempt completed MRHydra/P01 (cell-mean BA0.490759), then failed with a
1.03GiB allocation error fitting P02. Launcher now uses a fresh child process
per classifier/fold so native allocators release all memory on exit. Completed
receipts are reused; no changes to classifier settings or split definition.
Live state, resumable per-fold receipts and case-aligned predictions:
`D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/selected-loso/`.
Ten whole-subject folds, fresh seed33 models, 8 CPU threads, all sessions held
out together. MRHydra runs all folds first, then QUANT. Equal-score ensemble
uses matched out-of-fold predictions without fitted calibration. Native hard
predictions are retained separately from ensemble scores. No fold models saved.
Final repo report: `docs/competition-track2-loso-results.json`; includes official
cell-mean metric and equal-subject diagnostic. Public P04/P09 are now part of
LOSO; they must no longer be described as an untouched evaluation set after
reviewing these results. Do not launch duplicate workers. Resume same launcher
only after checking the first worker stopped. No uploads authorised.

Development/all-public fits for both classifiers already complete under
`.../selected-models/`. Previous 6/2 development validation: MRHydra52.64%,
QUANT48.33%, equal ensemble50.75% cell-mean balanced accuracy. These are separate
from LOSO and from the all-public submission models. Portable packaging source
is in `tools/{selected_track2_submission,export_track2_selected}.py`; parity
smoke/export remains to be checked before treating these ZIPs as ready.

## Competition data preparation complete,10November2026

PROTEUS Track2 (Codabench17982) public data downloaded/converted/audited.
2655cases x41EEG x480points at120Hz. Native EDF41ch500Hz retained. Prepared
D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/prepared; source cache under
D:/Data/EEGSources/Competitions/NeuralInterfaces2026Track2. Subject-holdout TS
1900TRAIN/755TEST; later-public-session TS2268TRAIN/387TEST. Full numeric TS,
case alignment, sensors, duplicates, hashes and aeon fit/predict passed;9tests
passed. Untuned bandpower classifier cell-mean BA0.389883subject-heldout,
0.405247cross-session. Worker complete; no required download remains. Read
`docs/competition-track2.md`, preparation/baseline result JSON. No archive
registration, no sealed access, user not registered, no submissions. Current
FAQ takes precedence over stale guide: predict only EEG1x41x480, no IDs;
reset model state per session; metric cells subject/session/interface. Local
preprocessing reviewed against upstream but full NeuralBench numerical parity
not established. Active script was subsequently made resumable using verified
existing artifacts; do not launch duplicate workers or rewrite final TS files.

## Study scope update,10November2026

User excludes all MEG/ECoG from the study: FaceDetection, HandMovementDirection,
MatchWords, MindReading, MotorImagery. eeg.txt now76scalp EEG problems, including
29MI; registry remains85 and all data retained. Both registration scripts keep
these exclusions when rebuilding selections. Current taxonomy/counts:
`docs/problem-taxonomy.json` and `.md`. Older81selected/30MI figures below are
historical archive figures, not the present study selection.

## Current task: twenty additional diagnosis problems from EEGDash

9November2026:83registered/79selected;30MI unchanged. AlzheimersAuditory is the
first completed clinical addition (2020TRAIN/1000TEST,33people). Read
`docs/eegdash-diagnosis-plan.md`, progress/summary JSON and source evidence.
User confirmed: diagnosed groups only initially; INCLUDE UCSD despite source
small-cohort diagnostic-ML caution; anti-alias RESAMPLE mixed-rate psychosis
recordings to1000Hz to retain all people. No pending permission for these.

Iowa full download is running after successful4person pilot. Reviewed queue
`tools/run_eegdash_diagnosis.py` waits for its receipt, converts/validates/registers
Iowa, then runs both mixed-rate psychosis pilots. Local queue state/logs live at
`.openeegbench/eegdash-diagnosis/`; inspect before starting competing writers.
Native sources D:/Data/EEGSources/EEGDash; final TS/sidecars D:/Data/EEG.
Use scientific overlays including eegdash-runtime and escalated native access.
UCSD pilot passed196cases/4people/32scalp channels at512Hz. Separate UCSD queue
`tools.run_eegdash_diagnosis.run_ucsd` converts its completed full download and
waits for Iowa registration before its own registry update. State/logs
`ucsd-queue.*`. Initial strict screen leaves20candidate source rows, not20
validated problems or independent studies; see candidate JSON for deferrals.
24focused clinical/shared-pipeline/registry tests passed.
Next adapter: ds007526 PD rest; TBI ds003522 original Group codes are
0=mTBI,1=Control,2=ChronicTBI (not historical survey guesses). Shared cohorts
must be documented and recordings deduplicated; relabelled copies do not count.
No uploads/pushes; do not modify student BIDS guide or its backup.

## Current completion: thirty motor imagery problems

All five additions passed full conversion, artifact verification and local registration: **82registered /78selected /30MI (29scalp EEG +1ECoG)**. Final arrays and sidecars are under D:/Data/EEG; no uploads or pushes. Machine-readable evidence: docs/mi-to-30-progress.json and docs/mi-to-30-summary.json. Original30paper data/results unchanged. Daly final1698TRAIN/643TEST,18subjects,19EEG x20000points at1000Hz; both ambiguous09/15run10 copies excluded, other recordings retained, no retained duplicate cases or shared subjects. The user confirmed reporting the issue on OpenNeuro on9Nov; comment URL not supplied. The focused final tests passed28checks. No conversion worker remains required for this batch. Earlier snapshots below are historical.

## Historical Daly cleanup snapshot

9Nov: original09/15run10 EDFs and event tables are identical; headers do not identify the true participant. Both ambiguous recordings are excluded by exact pinned path/SHA/size checks; other runs and original source objects retained. User authorised removing duplication after investigation. No existing specific report found, OpenNeuro comments empty, latest public release1.0.1 unchanged. See docs/daly-duplication-investigation.md and ready-to-paste issue draft; no issue posted.28focused tests passed. Cho full conversion completed10520cases, so29MI prepared. Main queue restarted via .openeegbench/mi-to-30/recover-daly-and-resume.py; logs daly-reviewed-batch.log/.err. It restores source metadata/originals, verifies other4full outputs, downloads/converts correctedDaly, then registers all5andtests. Do not start competing writers. State docs/mi-to-30-progress.json. Registry remains25until final registration.

## Earlier 9 November status

Three full outputs validated: Pan23/Pan25/Rozado, giving28prepared MI including25registered. Main queue stopped on Daly: sub09run10/sub15run10 entire original EDF SHA256identical, no owner chosen. Evidence docs/daly-cross-subject-duplicate-audit.json. Daly source directory currently absent on D; repo has immutable original file pins/receipts. Cho all52native MAT files downloaded/verified; independent full conversion `.openeegbench/run_to_30.py cho full` resumed, live state/log cho-full.json/cho-full.log. Do not start a competing main queue while Cho writes. Do not count pilots as final problems.

## Cache recovery correction

The queue stopped during Rozado full conversion with EOFError on an empty derived NPZ left by a previous worker restart. Fixed store_intermediate to rebuild EOF/truncated caches and publish new caches atomically after complete writes; interrupted writes preserve the previous valid cache. Focused native/registry tests:26passed. Two full Pan outputs are validated; registry remains25 MI until the five-problem batch completes. Queue resumed; original recordings and final arrays untouched. Read live docs/mi-to-30-progress.json rather than assuming the last background run finished.

## MI queue completed

All five full preparations passed and were registered locally. Read [current progress](docs/assimilation-progress.json), [batch results](docs/mi-to-30-progress.json) and [summary](docs/mi-to-30-summary.json). Original paper arrays/results unchanged; no uploads or pushes. Active-queue text below is historical.

## Previous active MI queue towards thirty

The next five are running locally on `dataset-assimilation`: Pan2023, Pan2025, Rozado2015, Daly2020 and Cho2017. See [batch progress](docs/mi-to-30-progress.json) and [preparation details](docs/mi-to-30-preparation.md). Current registered counts below remain valid until all five complete; pilots do not count. No push/upload.

Background runner `.openeegbench/mi-to-30/run-batch.py` bootstraps the scientific overlays and calls `tools.run_mi_to_30.run`. Logs `batch.log`/`batch.err`, source and pilot states `<key>-<action>.json` in that directory. It waits for the already running Pan full job, converts Rozado and Daly, waits for the separate Cho full download then converts it. It verifies all sidecar artifact hashes, updates registry/eeg.txt/dataset descriptions/provenance/counts and runs tests only after all full outputs pass. Any exception stops with `stopped_for_review`; inspect the error rather than bypassing a gate. Do not launch competing writers or rerun existing final conversions.

All five primary pilots passed, including Cho. Full Cho download now runs separately while Rozado/Daly convert. Both full Pan sources passed waveform identity audit; original participants across the studies remain unknown. Phone-gesture MI-BMPI is a downloaded/piloted reserve with CC BY-NC4.0, not registered or counted. Cho is slow: resumed original MAT via four normal HTTP ranges per file, retaining existing prefix; verify assembled publisher MD5 before publishing source, SHA256 in receipt. New range code cleans only owned temporary files after digest validation. Public downloader/converter/runner/test tools are `download_mi_to_30.py`, `prepare_mi_to_30.py`, `mi_to_30_native.py`, `run_mi_to_30.py`, `register_mi_to_30.py`, `audit_pan_identity.py`, `lock_daly_release.py`, `download_original_ranges.py`, `test_prepare_mi_to_30.py`. Native source IDs/hash lists under docs/source-evidence/mi-to-30.

Important: Harvard metadata API403 but public numeric datafiles work; DataCite versions Pan23=1.3,Pan25=1.0,Rozado=2.2, exact file IDs/MD5/SHA256 pinned, do not claim unverified V1 release membership. Pan23 lacks native anatomical channel labels; use opaque source columns, secondary MOABB mapping separately, flag. Pan25 native ordered uppercase names and Info.taskpoints verified; original online split pools session1+session2first30 TRAIN, rest TEST (shared people intentional). Cho native MAT lacks labels too; preserve numbered columns, measured coordinates and original trial_sequence ordering; no invented montage. Its actual imagery_event onset is index1023 for first s01 trial, not guessed1024/1536. Rozado Trig1 atindex0 excluded;32EEG indices1..32 explicit native microvolts; common-host raw clocks, no interpolation/dejitter. Daly source includes labelled runs beyond9; all labelled non-calibration native runs audited. Some final20s trials incomplete: excluded with original events/bounds/reason, no truncation/padding. Original README CC BY4 vsdataset_descriptionCC0 both retained.

Scientific overlays: nback-aeon-1.5.0, kalunga-runtime, ofner-runtime, xdf-runtime. Regression suite100passed at8Nov; full final tests run after queue completes. A live process loads code once: refresh sidecar/code copies at final completion as runner does. New batches remain local; do not rewrite origin/main historical commits.

## Previous completion: Batista / Polo / Leeuwis

All three authorised preparations are complete on D, fully validated and registered: **77 registered / 73 selected / 25 MI**, including 24 scalp EEG problems and one original ECoG entry. No remaining conversion workers for this batch. Original 30 paper datasets/results unchanged; no uploads. [Detailed preparation, source identity checks, licences, protocol and metadata completeness](docs/mi-next-three-preparation.md), [machine-readable summary](docs/mi-next-three-summary.json).

| Problem | Source | Train/test | People | EEG x samples | Hz |
|---|---|---:|---:|---:|---:|
| ImaginedHandsHaptics | Batista2022, Zenodo7664069v1 | 3083/1034 | 20 | 32x2495 | 500 |
| ImaginedAnkleRest | PoloHortiguela2025, Zenodo14672334v1 | 264/132 | 6 | 28x1000 | 250 |
| ImaginedHandsFeedback | Leeuwis2021, DataverseNL10.34894/Z7ZVODv1.0 | 6560/2240 | 55 | 16x1250 | 250 |

Finals D:/Data/EEG/<Name>, sources/pilots/NPZ D:/Data/EEGSources/<Source>. Do not rerun existing final converters. Public tools/download_mi_next.py and tools/prepare_mi_next.py take keys batista/polo/leeuwis, --pilot, --cache-root and full-conversion --output. All three pilots passed before bulk retrieval. Final regression suite: 93 passed. Exact .ts readback and case/label/hash alignment, channel identity/order, duplicate and aeon checks passed for every final; fit/predict is a 3cases/class/split compatibility smoke test. No EEG sensors removed, no shared native subjects or duplicate cases across train/test. Final artifact hashes verified after sidecar/code/doc refresh. Existing full results and per-source *-mi-conversion-result.json / *-mi-metadata-completeness.json are authoritative.

Batista: 99 imagery recordings, 4127source cues, ten trials excluded for native gaps or missing/ambiguous S10; retain4117. Common2495sample safe crop from actual native end timings, no interpolation. Original main IDs01-09/12-19 and pilotp01-p03 preserved, even though pilot filename tokens reuse sub-01..03. Fixed15/5 split balances pilots/main. Sub-13 lacks MIMOVRHP; no invented run. ME execution excluded. Every calibrated EEG-only recording hash and all4117retained2s cue fingerprints compared with26Vagaja recordings/source cues: no exact matches, dates disjoint. This does not establish cross-study participant identity. Native32EEG order invariant; two excluded auxiliary schemas retained, generic Aux roles unknown.24headers omit coordinates; no generated montage or missing filter history substituted. Full source evidence docs/batista-full-native-marker-audit.json and batista-vagaja-full-identity.json. An early full attempt stopped before export on a misplaced metadata update; corrected and covered by regression test, later full run passed.

Polo: six release people, S1one visit, STATIC/MOTION conditions pooled.132native repetitions, two rest phases and one MI phase each, first4seconds,396cases. Exclude settling/audio/return phases. Exact uppercaseZsensor spellings from native config, raw data_EEG only. Release B06 internally B07, both IDs retained with bijection across all files. Other release/native IDs match. Original native2Dpositions/preprocessing-step table and sample-wise task boundaries retained. Physical units inferred microvolts, unchanged values; reference/ground and coordinate units/frame unknown. Trigger semantics not guessed for rejection.

Leeuwis:55native non-contiguous IDs7..67,220CSVruns plus4supplementary files. Locate actual t0atindex749, not750;1250samples0..4.996s. Round-trip float64 parsing and original numbers unchanged, microvolt units inferred, exact distributed CSV filtering history unlabelled. Original report/paper reference right earlobe, AFz ground. Questionnaire ppnr maps individual demographics/scores;55rows36Female/19Male, original report's57total inconsistent. Preserve raw BCI_Novice coding without interpreting its question. First41numeric native IDs train, remaining14test; pool calibration+feedback but preserve phases. CSV source offsets epoch-local; continuous intervals unknown. Pinned file IDs handle irregular Subject40names; native filename IDs checked. MOABB secondary subject mappings accompany all three to support provenance matching.

All original data grants CC BY4.0 verified independently of software licensing; source publisher checksums/IDs pinned in docs/{batista,polo,leeuwis}-mi-source-lock.json. Units uncertainty explicit in metadata, sensor table, README, attribution and root descriptions. The active queue at the top supersedes this historical snapshot. Remaining five roadmap candidates Pan2023/Pan2025/Rozado2015/Cho2017/Daly2020 require original access/licence/identity checks before admission. Deferred MIMED/depression/VisualCategories retain their prior unresolved source-quality status.

Earlier snapshots below are historical and superseded by this completion.

## Latest completion:8November2026

ImaginedHandsRobot, ImaginedGrasps and Vagaja2023/ImaginedHandsEmbodiment are fully converted, validated, registered and documented on D.74registered/70selected,22MI including21scalp EEG and one ECoG. No active workers remain for this batch; do not rerun existing finals. [Summary, dimensions, metadata completeness and caveats](docs/mi-completion-batch.md); machine-readable summary docs/mi-completion-batch.json. Original30results unchanged. No uploads.

Robot2880/2880cases32x1000at250Hz; native training/online phases intentionally share people/sessions. Grasps4500/2250cases60x1000at250Hz; official training+validation merged, test day unchanged, shared people intentional. Native float64 Grasp units remain inferred microvolts; no guessed rescaling. Continuous offsets unavailable. All numeric readbacks, labels/metadata/order, duplicates and aeon compatibility checks passed. Every final manifest artifact hash was checked after refreshed sidecars.

Embodiment798/240cases32x4997at500Hz; source1040cues, exclude two left-hand imagery trials crossing lost-sample markers, common safe4997sample window. Native header calibration/coordinates/filter tables recovered; source SUB31 explicitly uses deposited same-stem EEG/VMRK because header references are stale. Whole-subject20/6 split,10/3perpriming group. No flat EEG removals in any of the three. Original source caches D:/Data/EEGSources/Farabbi2020,BCIComp2020UpperLimb,Vagaja2023. Pilot/full ordinal caches separate for Grasps/Vagaja. Preserve old pilot caches; never silently infer sensors/reference or original continuous offsets.

Next candidate is Batista2022, original Zenodo7664069 CC BY4.0 (2.863GB originals); no EEG downloaded in this batch. Before independently counting it, compare native file IDs/dates/waveforms with Embodiment due common laboratory/hardware. Roadmap docs/motor-imagery-road-to-30.md/json has eight further candidates needed for30 under existing counting rule. Do not resume deferred MIMED/depression/VisualCategories without resolving documented source quality issues.

## Objective and boundaries

Prepare EEG classification problems as standard aeon/tsml `.ts` TRAIN/TEST files with comprehensive metadata sidecars in **D:/Data/EEG/**. Original downloads and numeric intermediates belong separately in **D:/Data/EEGSources/**. Maintain source identity, licence evidence, descriptions and source-issue reports in this repository.

- Prioritise verified fully open data, then smaller source/converted sizes. Public availability and software licensing do not establish data redistribution rights.
- No Zenodo or other uploads. No messages/issues to authors without explicit instructions; local issue drafts are fine.
- Do not download SEED-V or reinstate depression without resolving its known source duplicates.
- Preserve the original 30-problem paper subset and existing experiment results. Additions are local preparations until separately released.
- Do not overwrite existing final problems casually. Converters refuse existing final directories. Previously authorised corrections are documented separately.
- Do not commit large EEG/source/array/runtime files. `.openeegbench/` and `tmp/pdfs/` are ignored; large artifacts belong on D.
- Preserve unrelated uncommitted changes in this shared workspace. Read applicable AGENTS.md/current permissions. No subagents unless explicitly authorised by applicable instructions.

## 1. Recover current state

Read, in order:

1. `docs/assimilation-progress.json`: completed/in-progress/deferred work and current counts.
2. `docs/datasets.md`, `docs/dataset-provenance.md`, `eegbench/datasets.py`, `eeg.txt`.
3. `MOABB.md`, `OpenEEGBench.md`, `EEG-Bench.md`, `AdaBrain-Bench.md`, and their machine-readable inventories/mappings.
4. `docs/open-moabb-conversion-results.json` and relevant source-specific preparation/result/lock/completeness files.
5. Relevant converters, downloaders and tests. Inspect running sessions/logs before starting another worker for the same source.

Snapshot on 6 November 2026: **66 registered / 62 selected**. Four excluded from `eeg.txt` remain in the registry/on disk: Blink(4 channels), ButtonPress(32), Sleep(1), EpilepticSeizures(1). Only the last two are single-channel. Counts will change; use live state.

Latest completed sources, already prepared and registered:

| Archive name | Source | Train/test | Subjects | EEG x samples | Hz |
|---|---|---:|---:|---:|---:|
| ParkinsonsPedalling | Singh2020 / PRED+CT d011 | 2123/842 | 39 | 63x2000 | 500 |
| ParkinsonsRest | Cavanagh2018b / PRED+CT d002 | 1216/420 | 55 | 63x2000 | 500 |
| ObsessiveCompulsiveSymptoms | Gruendler2009 / PRED+CT d008 | 7754/2660 | 35 | 52x2000 | 500 |
| WalkingTargets | BCIComp2020WalkingERP / Track5 | 3600/900 | 15 | 42x100 | 100 |
| SymbolAttention | Chailloux2020 / ds003190 v1.0.1 | 521259/175180 | 19 | 8x256 | 256 |
| ArmCoordination | Kueper2024 / Zenodo10229480 | 1498/505 | 8 | 64x1500 | 500 |

Do not reconvert them as new studies.

## 2. Windows runtime and safe execution

Repo: `C:/Code/EEGBench`, PowerShell. **Always read/write repository text with explicit UTF-8.** Elevated Python can default to cp1252 and corrupt prose or fail on smart quotes. Do not embed a PowerShell here-string terminator inside an outer here-string; use apply_patch or an actual runner file for complex content.

Existing scientific environment bootstrap:

```python
import sys
from pathlib import Path
for folder in ['nback-aeon-1.5.0', 'kalunga-runtime', 'ofner-runtime']:
    sys.path.insert(0, str(Path('C:/Code/EEGBench/.openeegbench') / folder))
```

Dependencies are pinned in `tools/requirements-moabb.txt`. Existing overlays provide aeon1.5.0/MNE1.13.2 and scientific dependencies; avoid installing another large environment on C. A Python script can bootstrap those paths then import/call a source converter. For long jobs, use an ignored real runner `.py`, capture stdout/stderr to an ignored log and checkpoint process/session IDs. Use `Start-Process -WindowStyle Hidden` for background helpers. Keep user progress updates concise and frequent; no blocking waits over60 seconds.

D access and some installed runtimes need `require_escalated` under the current sandbox. Dataset work is already authorised; use the tool escalation with a concrete explanation instead of asking again. Check space via `shutil.disk_usage('D:/')`. The sandbox C overlay has sometimes reported ENOSPC while elevated writes worked; large data still belong on D.

Pytest shared temp can be inaccessible. Use an owned basetemp:

```python
import pytest, uuid
pytest.main(['-q', '-p', 'no:cacheprovider',
    '--basetemp=C:/Code/EEGBench/.openeegbench/test-' + uuid.uuid4().hex,
    'tools/test_prepare_open_moabb.py'])
```

The previous midnight-to07:00 pause was a one-off local process guard, not an agent scheduler. The user resumed chat explicitly. Do not promise automatic agent awakening without an actual scheduler. Check a new pause request against current dates/timezone.

## 3. Inspect before download/conversion

- Compare accession, cohort, source file IDs and recording/task membership against our provenance. Relabelled/resampled/filtered views are not new recordings. A shared DOI also does not prove identical recordings: competition tracks and ERP CORE tasks can be separate experiments.
- Inspect pinned loader code, original documentation and host metadata. Current MOABB source snapshot: `bed9f0fa1b9c4d3f71c7f2ddb22e4e44383211b5`; pin/document any replacement.
- Verify the **original data licence** and applicability to actual files, separately from software and paper licences. Save original host evidence, exact version and obligations. Treat NC/ND/research access separately.
- Create `docs/<name>-source-lock.json` with release/revision, original file IDs/URLs, publisher checksums and byte counts. Record paper and data DOI separately; a paper DOI is not an established data DOI.
- Estimate source bytes, float64 arrays `cases*channels*timepoints*8`, text expansion, staging and cache requirements. Source size can mislead: SymbolAttention is about1.36GB source but27.83GB `.ts`/sidecars.
- Inspect actual headers/variables, names/order/types, calibration, labels and native split semantics. Download real organiser test-label sheets, not prediction submission templates.
- Run a small participant pilot through the **entire pipeline** before bulk conversion. A single ZIP may require full download before pilot extraction.

## 4. Reusable implementation and adapter contract

`tools/prepare_open_moabb.py` is the common exporter: native recordings -> compressed float64 NPZ -> staged `.ts` and sidecars. Register a source `SPEC` and reader in `core.SPECS`/`core.READERS`, then call `core.prepare(key, cache_root, output)`.

Reader yields `(X, case_rows, channel_names, recording_info)` per recording:

- X: finite float64 `(cases, channels, timepoints)`, with explicit physical numeric interpretation.
- case_rows: one dictionary per case in exact X order. Start with `core.base_row(...)`, then qualify assigned/unknown IDs and add native source fields.
- channel_names: exact sensor names in X axis order, consistent across all included recordings or safely harmonised by documented names.
- recording_info: native shape/fs/calibration/units, types/positions/reference, source hashes, original markers/headers, participant/run information, source flags and missing metadata.

Examples:

| File | Pattern |
|---|---|
| `prepare_kueper.py` | Explicit BrainVision microvolt resolution, float64 before calibration, repeated Mk keys, demographics/flags |
| `prepare_chailloux.py` | BIDS/native cross-check, missing target markers, copied recording, unknown units, numeric sorting of string IDs |
| `prepare_walking_erp.py` | Native epoched MAT, true XLSX labels, official train+validation merge, shared participants, local epoch times |
| `prepare_predct.py` | Clinical joins, source readability exclusions, continuous windows, flat-case audit |
| `prepare_liu2024.py` | Original trigger/source-product audit and qualified cohort exclusions |
| `prepare_small_moabb.py` | Verified atomic fetch, `.ts` header, exact readback, overlap/duplicate tests, aeon smoke |
| `prepare_wang.py` | Exact float64 case hashes and round-trip decimal formatting |
| `download_openneuro_release.py` | Immutable Git snapshot and git-annex hashes before accepting mutable S3 files |

Keep downloading/conversion separate from experiments/leaderboards. Source-specific `finish_sidecars` enriches metadata and refreshes changed artifact hashes/result size. Do not copy another finisher blindly: assumptions differ.

## 5. Splits, order and original case identity

- Follow official partitions. Archive TRAIN is original training followed by validation; preserve original TEST. No archived validation. Tuning is the fitted classifier's responsibility.
- For unsplit sources, define a deterministic whole-subject split and exact membership. Usually about75% training, but document actual rule rather than calling it an original split.
- Keep IDs as strings. `01`...`09`,`010`...`019` need numeric sorting when appropriate while preserving original tokens.
- Preserve native example/marker order; document file/session order and never shuffle.
- `case_splitter`, `allow_subject_overlap=True` and mandatory `subject_overlap_reason` support official temporal/session protocols. Do not use them to excuse accidental leakage.
- Continuous source: record zero-based start/end-exclusive samples and original event positions; retain one-based source positions separately.
- Epoched source: retain native file/epoch index and local times; unknown continuous offsets stay unknown. The overlap validator recognises the exact string `Native epoch-local samples; original continuous offsets unavailable`. Never present a synthetic concatenated Raw timeline as original time.
- Original session/run/trial IDs that are absent stay unknown. Assigned date groups/ordinals must be labelled assigned/inferred. Repeated Mk keys require literal key plus unique line ordinal/sample position.

## 6. Scale, sensors and authorised flat policy

- Recover true names/order from files or a primary document explicitly identifying rows. Never invent E1...En or montage names from array length. Loader-generated labels need independent evidence.
- Preserve native spelling, e.g. FZ/PZ. Select EEG by recovered roles, not by assuming all inputs are EEG. Preserve source mappings for excluded EOG/ECG/EMG/timestamp/accelerometer inputs.
- Verify every included recording's order. Document any safe named intersection/reordering and excluded sensors; otherwise flag/defer.
- MNE EEG normally arrives in volts; verified volts -> microvolts uses `*1e6` after float64 casting. MAT/EEGLAB numbers may already be microvolts; BrainVision binary may be counts requiring per-channel resolution. Never blanket-rescale all sources.
- User authorises **global sensor removal if any retained case is constant/effectively flat**. Current criterion: `std <= 1e-7` in microvolt numeric scale. Record exact-constant/tiny counts separately. Index6 means the seventh channel.
- Cast to float64 before multiplying/calibrating. Exact round-trip decimal formatting preserves the resulting binary64 values.
- No added demeaning, normalisation, filtering, referencing, resampling or unrelated rejection by default. Large DC offsets are not constant channels.
- Record pre-removal names/indices, removed names/indices, affected cases, scale/calibration and final exported native indices. Final sensor/case tables must reflect removal.
- `prepare_predct.removal_pair_audit` assumes event_index equals NPZ row index; adapt before using it with excluded source ordinals.
- Preserve original positions/montage/reference. Missing geometry/timezone stays unknown. Keep any template distinct from measured individual coordinates.

## 7. Source duplicates and cache integrity

The exporter performs **source-case hash preflight before writing large files**, then full post-selection duplicate checks. Final checks remain necessary: channel removal can make formerly different cases identical.

Exclude an exact copied recording only with verified matching participant, waveform bytes, header/calibration/sensors and native event/label stream. Retain the earlier copy, document paths/hashes/excluded cases. Never repair contradictory labels by guesswork. Identical label schedules do not imply identical EEG.

Cached ZIP extraction checks CRC, not only size. Cached NPZ must exactly match freshly decoded X. Preserve source checksums and native extracted-file hashes. A same-sized file is not proof of validity.

Document bad files and precise exclusions. A validated subset may be possible, but do not hide cohort/protocol changes. Defer unusable labels and move to another eligible source. A loader incompatibility is not proof the original biological signals are corrupt.

## 8. Required output

```text
D:/Data/EEG/ProblemName/
  ProblemName_TRAIN.ts
  ProblemName_TEST.ts
  cases_train.tsv
  cases_test.tsv
  ProblemName_metadata.json
  ProblemName_manifest.json
  ProblemName_validation.json
  channels.tsv
  channels.txt
  metadata-completeness.json
  ATTRIBUTION.txt
  README_ProblemName.md
  source_metadata/
  conversion_code/
```

Intermediate: per-recording `numpy.savez_compressed` float64 X, `allow_pickle=False`, in the source cache. `.ts` embeds y; separate four-NPY files are unnecessary for this archive.

Metadata: subject/session/run/file/recording, native epoch/event/marker index, source time/sample bounds, original/encoded labels, original partition; dataset source/data DOI/paper DOI/licence, participant/session/recording counts, rates/window/overlap, preprocessing, names/types/order/reference/geometry and class counts. Distinguish directly recovered, documented, assigned/inferred and missing information.

## 9. Validate before registering

1. Source pins/checksums, shapes, labels/class counts and finite values.
2. Complete `.ts` numeric readback reproduces every float64 case hash/label and metadata row order.
3. Subject/recording separation or explicit original-protocol overlap. Unknown continuous overlap is not zero.
4. Every array's channel identity/order matches final metadata.
5. Exact duplicates within/across splits and remaining flat/tiny sensors.
6. Explicit missing/inconsistent metadata and source issue evidence.
7. Aeon load/fit/predict smoke. Current smoke uses3 cases/class/split; do not claim full-cohort fitting. Complete arrays are stream-validated separately.
8. Pilot passes before full conversion.
9. Unique staging directory renamed only after validation. Never register Pilot outputs.
10. Refresh hashes/bytes/size after metadata/code/doc enrichment. Untouched already-validated `.ts` need not be reread repeatedly because prose changed.

Stop only an identified owned worker. Before deleting its failed staging directory on Windows, resolve/check the absolute path lies inside the intended D:/Data/EEG staging root and verify the problem header. Never delete broad computed paths or original datasets. Avoid concurrent writers to one cache/output. Shared root results JSON is not a concurrent database: check/merge keys if jobs finish together.

## 10. Register and document

- Add canonical tuple to `_TABLE` before its sort in `eegbench/datasets.py`. Keep `eegbench_paper` derived from the original30 rows before additions.
- Repository uses **all prepared problems**, not only30. New rows are `local_only`; automatic aeon/Zenodo downloads remain the original published open subset until release.
- Add alphabetically to `eeg.txt`, excluding the four user-removed problems.
- Update targeted counts in tests/README/docs; do not globally replace numbers that might be channel counts or results.
- Add measured metadata assertions and meaningful adapter tests for label/timing/identity edge cases.
- Regenerate the table between DATASETS:START/END markers using `metadata_markdown()`. Add task/source/dimensions/split/licence descriptions and original aliases/source identity to provenance.
- Fractional fs such as62.5Hz is supported; never round/resample to fit the registry.
- Save measured result/completeness/summary, update `docs/assimilation-progress.json`, and current comparison-page counts. Preserve explicitly historical surveys and original results.
- Refresh final sidecar copies of source descriptions/code and artifact hashes.
- Run relevant tests plus `git diff --check`. Last completed suite:47 passing tests; expect more with new adapters.

## 11. Known source traps

- **VisualCategories/Kaneshiro2015:** data CC BY3.0; README explicitly maps electrodes1-124. S1 has5188 epochs and4 exact duplicate pairs, all with different exemplar IDs;3262/3263 also conflict in category1/6. X_2D/X_3D agree upstream. S10 is clean,5184 epochs. Only S1/S10 downloaded; other8 unknown. Deferred, not registered. See preparation/audits/local issue draft. Do not blindly choose a label or assume every other subject is affected.
- **BNCI2003_004:** actually Competition III IVa(2005), right hand/right foot,5 subjects118EEG. Original-host CC BY4.0 grant not established; defer fully-open classification.
- **WalkingTargets:** official shared participants, epoch-local offsets only; real True Label sheet differs from submission template. L1/L2/R1/R4 removed, leaving32scalp+10ear. Source fs100Hz, units inferred.
- **SymbolAttention:** copied session recording, missing target block, no-flash placeholders, one-based BIDS sample fields and coarse timestamps; see completed report.27.83GB export versus1.36GB source.
- **ArmCoordination:** repeated Mk keys; explicit EEG calibration differs from accelerometers. Hardware131/software249Hz tables; MNE reports software only. XP01 set4 continues set3. All2003 native onsets retained with flags rather than forcing1920 nominal valid trials. Executed movements, not imagination.
- **ParkinsonsRest:** isolated S3 marker in control896 before real sequence; use validated marker groups. Exclude author-bad803 OFF. Rest import reads *_ODDBALL.vhdr, so rest/oddball may share acquisitions despite distinct MATs. Large DC offsets/author-bad sensors are not automatically flat.
- **ParkinsonsPedalling:** Aa/Bb/Cc are native EEG names; no invented anatomy. Clinical CSV uses literal backslash-t. Workbook/script support first63 EEG; XYZ accelerometers excluded.
- **ObsessiveCompulsiveSymptoms:** high/low student OCI scores, not diagnosed OCD. Use SELECT columnB, not duplicateJ.35 nominal raw CNTs retained;11 processed deferred.12 global flat sensors removed. Reader incompatibility does not prove corrupt source signals.
- **Depression/SEED-V:** remain excluded. MDD identical recordings were non-flat; distinct-subject copies invalidate naive use. Source licences always separate from software.

## 12. Next batch and checkpoint

Read current progress/next-batch inspections; do not select an already-prepared source from an old shortlist. Original-host-confirmed fully open first, then actual download/converted size. Candidates now under inspection include BCIComp2020UpperLimb/Track4 and compact unprepared MOABB sources.

Track4 shares OSF pq7vb with WalkingTargets/ImaginedCommands but has different recordings: three imagined grasps over three recording days. The pinned loader warns native10-second test epochs have the cue section masked; use the actual4-second MI segment consistently. Verify original MAT/PDF and organiser answer sheet; never blindly copy synthetic Raw reconstruction or treat hard-coded loader labels as the sole authority.

Checkpoint each material discovery/completion: exact running IDs/logs/stage path, pins, exclusions, completed files, current blocker and next command. Update live progress and next-batch notes. Another agent should resume from that checkpoint without repeating hours of downloads/export.


Next-batch live checkpoint: [docs/next-batch-progress.json](docs/next-batch-progress.json). Selected candidates are **ImaginedHandsRobot / Farabbi2020** then **ImaginedGrasps / BCIComp2020UpperLimb**. Kodera2023 remains a reserve because its two native channel/rate cohorts require a separate harmonisation/identity decision. Source inspection must precede conversion.


Current new adapters: `tools/prepare_farabbi.py` preserves source training/online file phases (shared people/sessions, explicitly pooled across days); `tools/prepare_grasps.py` preserves official days1+2/day3 and verified source MI slice1500:2500. Both pilots passed. `tools/update_dataset_sidecars.py` safely refreshes sensor/case mappings and manifest hashes; its test checks case labels/hashes/leading-zero IDs remain unchanged. Full conversion runner/log/state lives under `.openeegbench/next-batch/` via `.openeegbench/run_next_conversion.py`. Check worker JSON/logs before starting another conversion.


## Compact imagery batch, 8 November 2026

Read `docs/compact-imagery-progress.json` and `docs/compact-imagery-preparation.md`
for the current task; do not restart an active worker. Original source downloads
MIMED19.526MB, MOVING98.204MB and Zhang1.755GB are publisher-MD5/size verified.
`tools/download_compact_imagery.py` and `tools/prepare_compact_imagery.py` implement
this batch; `tools/read_bci2000_native.py` is a strict direct native reader.

- MOVING/ImaginedHandGestures full conversion passed382TRAIN/143TEST,11people,
  17EEG x3000 at500Hz; missing3 source markers and15 globally flat sensors
  documented. Final folder exists on D; registry integration pending in this
  snapshot. Use live progress to determine whether that has since completed.
- Zhang/ImaginedArmTasks720-case pilot passed; full worker
  `.openeegbench/run_compact_imagery.py zhang`, state/log under
  `.openeegbench/compact-imagery/`. Native3.96second cues determine3960points;
  one40sample initial state fragment excluded. Expected4320cases. Anatomical
  sensor names absent in all180nativeheaders: preserve17source column ordinals,
  explicitly flag unknown anatomy/order correspondence, do not infer EGI1-16/E28.
- MIMED full export stopped on85cross-ID duplicate pairs, one conflicting label.
  `tools/audit_mimed.py` reproduces cropped/full-block hashes; detailed audit and
  unposted issue draft saved. None are flatlines. Complete30person data cannot
  pass admission. User asked defer versus explicitly reduced14person subset;
  no subset conversion without their answer. Pilot is not an admitted dataset.
- Pilot caches use separate `pilot-classification-intermediate` (Zhang suffix
  `-3960`) to avoid conflicting with full recording ordinal caches. Existing
  common exporter supports an optional `intermediate_directory` spec field.
- Relevant63tests passed before final registry edits. Refresh code/doc sidecar
  copies and artifact hashes after final descriptions. Original30/results unchanged.


**Compact batch completion update:** MOVING/ImaginedHandGestures and
Zhang/ImaginedArmTasks are now fully validated, registered and in eeg.txt.
Current68registered/64selected. Zhang3240/1080cases,17columns x3960points at1000Hz,
explicit unresolved anatomical sensor flag; MOVING382/143cases,17EEG x3000 at500Hz.
All sidecar/code/doc hashes refreshed;65relevant tests passed and diffcheck passed.
MIMED full30person problem remains unregistered due to85cross-ID duplicate pairs,
one conflicting class. Optional14person cohort requires the user's choice.
No live compact-batch worker remains. Latest checkpoint:
`docs/compact-imagery-progress.json`; D free approximately17.65GB at finish.
Previous Farabbi conversion is complete on D but previous-batch repository
registration/Grasps continuation is separate pending work; do not redownload.


## Active C:/Temp-only batch (8 November)

User paused D: access while moving the drive, then authorised further candidates
under C:/Temp if C has enough space. **Do not access D: at all until resumed.**
Initial C free8.6GB; enforce4GB reserve before jobs. Native sources go to
C:/Temp/EEGSources; final `.ts`/sidecars C:/Temp/EEG; pilot C:/Temp/EEGPilot.

NeuroTUM2025 is fully converted as two original-participant problems:
ImaginedRightHandFeetRest(P1)150/40cases,5sessions; ImaginedLeftRightHandRest(P2)
480/240cases,3sessions. Both24x750,250Hz,three original classes, one person each.
Last native session test, earlier sessions train; intentional participant sharing.
Native XDF sensor names/XYZ recovered; units presumedmicrovolts unchanged;
reference/coordinatephysicalunits unknown. Use pyxdf1.18.0 with sync/dejitterfalse.
Coincident empty markersP1S3 preserved, no EEG timestamp changes. Both passed full
validation; registry integration pending in this snapshot. Worker state/log
`.openeegbench/neurotum/`, tools prepare_neurotum.py/download_neurotum.py,
requirements-neurotum.txt. Existing overlays plus `.openeegbench/xdf-runtime`.

MILimb source access resolved via original public webpage per-file endpoints:
`/public-api/datasets/x8psbz3f6x/files?folder_id=<id>&version=2&$start=0&$limit=1000`
and `/folders/2`. Full ZIP403 does not require manual retrieval: use per-file
publisherSHA256 URLs. Original v2CCBY4.0,3719imagery CSVs561,848,100bytes,60people,
allR1. Root indexed sensor/encoding figures explicitly map16EEG channels; originals
visually inspected. Every CSV500rows16channels+sampleindex. S22I2trial5absent.
Candidate ImaginedHandFootActions,8classes includingbaseline/rest. PilotS1/S60
passed62/62cases,16x500,125Hz. Full download currently session25644; wait until
receipt `pilot=false` and3719imagery files verified before full conversion.
Scripts download_milimb.py/prepare_milimb.py and focused tests exist. No execution
M files downloaded. Sources/metadata1study; no invented continuous offsets/visits.
Progress `docs/temp-c-mi-progress.json`; main source locks/reports in docs.
Batista/Polo are not started because expanded total footprints exceed C capacity.


**C:/Temp batch completion update:** all three new problems are validated and
registered,71total/67selected. Finalarrays C:/Temp/EEG, sources C:/Temp/EEGSources.
D access remains paused: no read/write until the user resumes it. No live worker.
NeuroP1/P2full150/40and480/240,24x750at250Hz; one person each with intentionally
shared participant/disjoint sessions. MILimb full2727/930cases,60people,11x500at
125Hz,8classes. Raw source3719,62ambiguous copied class8rest files acrossS1/S2
excluded symmetrically without choosing a correct owner.5globalflat sensors
removed:F4,FC1,Cz,CP1,T8. Old raw source/pilots untouched; full input3719=3657retained
+62excluded. Header empty/NaN variants affect only sample-index column.
Read `docs/temp-c-mi-preparations.md`, source audit, exclusions and
`docs/temp-c-mi-progress.json`; do not redownload/reconvert these.74relevant tests
passed and diffcheck passed. New sources are2studies,3classification problems.
Batista/Polo still not started due limited C space; preserve4GB reserve. Exact
remaining C bytes and refreshed sizes in the completion report. No uploads.


## Latest update: D drive restored

The user explicitly resumed D access. All three temporary C: preparations are now
in D:/Data/EEG, their two source caches in D:/Data/EEGSources, and pilots beneath
each source cache's pilot-output directory. Every source/destination file was
byte-count/SHA256 verified before removing only the9explicitly named C:/Temp
copies. Source and numeric .ts/NPZ bytes are unchanged; paths/docs/code defaults
and sidecar hashes refreshed. New adapters now default to D:/Data again.
Current71registered/67selected, no new problems added by relocation. Source
grants/splits/exclusions remain unchanged. Read docs/c-temp-to-d-transfer.json,
docs/temp-c-mi-locations.json and live assimilation progress. Old C-only/D-paused
instructions above are historical and superseded by this user-authorised update.
No live transfer/conversion worker remains; no additional EEG downloaded/uploaded.
