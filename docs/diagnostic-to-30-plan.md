# Diagnostic-to-30 Assimilation Plan (+10 Datasets, 20 -> 30 Clinical)

This specification defines the execution order, source locks, scalp-EEG channel hygiene, per-subject window caps, stratified whole-subject 75/25 splits, disk-streaming safeguards, and registration rules for expanding `by_type("Clinical")` in `neuroverse` from **20** to **30** scalp-EEG problems (supporting the symmetric `30 MI + 30 Diagnostic + 30 Behavioural` benchmark design).

## 0. Baseline Arithmetic (`20 -> 30`)

`neuroverse.datasets.by_type("Clinical")` currently contains **20** scalp-EEG problems (`15` evaluated on the cluster + `5` added in commit `e3d69ed`), which **already include** the two restricted fibromyalgia cohorts (`FibroLiverpool` and `FibroUEA`). Reaching **30** therefore requires **+10** new clinical/diagnostic problems:

- **Phase 1 (Immediate Open Batch, `#1` to `#8` -> `20` to `28` Clinical)**:
  - `#1` to `#6`: The six independent open clinical cohorts from `docs/eegdash-diagnosis-candidates.json` and `EEG-Bench.md`.
  - `#7` and `#8`: Two additional open cohorts (`ds004595` and `PRED+CT d009` / PhysioNet `Siena`) to account for `FibroLiverpool` and `FibroUEA` already sitting in the baseline 20.
- **Phase 2 (Restricted Local-Only Temple Clinical Batch, `#9` and `#10` -> `28` to `30` Clinical)**:
  - `#9` `TempleAbnormalEEG` (`TUAB v3.0.1`) and `#10` `TempleEpilepsyEEG` (`TUEP v2.0.1`), added to `neuroverse.datasets.restricted` alongside `FibroLiverpool` and `FibroUEA` (never redistributed to Zenodo).

---

## 1. Mandatory Engineering, Storage & Repository Rules

1. **Package name is `neuroverse` (not legacy `eegbench`)**:
   - Do **not** run legacy `tools/register_eegdash_diagnosis.py` unmodified (it still imports `eegbench`).
   - Reuse the proven utilities in `tools/prepare_open_moabb.py`, `tools/prepare_cognitive_group_a.py` (`verify_final`, `whole_subject_split`, `harmonise_sampling_rate`, `decode_brainvision`, `decode_eeglab_set`), `tools/prepare_cognitive_group_b.py` (`resolve_storage_destination`), `tools/prepare_predct.py`, and `tools/update_dataset_sidecars.py` (`finish`).
   - Write a clean runner `tools/prepare_diagnostic_to_30.py` with unit/smoke tests in `tools/test_prepare_diagnostic_to_30.py`.

2. **Disk & Filesystem Safety (`C:` vs `D:`)**:
   - Check `Get-Volume D` before writing to `D:`. While `D:` reports `HealthStatus != Healthy` / `Full Repair Needed` or insufficient free space, stage raw downloads in `C:/Data/EEGSources/DiagnosticTo30/<key>` and export final `.ts` + sidecars to `C:/Data/EEG/<DatasetName>` (leaving existing verified folders on `D:/Data/EEG` untouched).
   - **Strictly sequential execution with immediate raw-cache purge**: `C:` has ~90 GB free. Complete `source lock -> pilot check -> full .ts export -> sidecar finish -> verify_final -> purge raw & NPZ intermediates` for one dataset before downloading the next. Never hold two large raw OpenNeuro datasets on `C:` at the same time.
   - **Chunked streaming for multi-gigabyte raw releases (`#5`, `#6`, `#7`, and especially `#6 ZosterNeuralgiaSleep` at 63.5 GB)**: Download raw participant files in bounded batches (e.g. 4 to 8 participants at a time), extract only the capped scalp-EEG windows into compact float64 `.npz` per-recording intermediates, delete those raw EDF/BDF/SET/FDT binaries immediately after recording their SHA-256 hashes in the source lock, then write the final `.ts` files from the `.npz` intermediates, run `verify_final`, and delete the `.npz` cache.

3. **Signal, Label & Split Contract**:
   - **Prediction target is always participant clinical diagnosis** (`participants.tsv` / clinical metadata table), never stimulus or cue identity.
   - **Scalp EEG channels only**: Explicitly exclude EOG, ECG/EKG, EMG, GSR/SCR, respiration, nasal airflow, SpO2, thermistor, Erg/plethysmograph, accelerometers (`X`, `Y`, `Z`), trigger (`Status`), and constant online reference channels (`VREF`, `E129`, mastoids `M1`/`M2`/`A1`/`A2` where typed as reference/auxiliary). Verify identical channel names and order across all retained recordings (aligning named common scalp subsets only when channel permutation is documented).
   - **Physical microvolts (`float64`)**: Cast to `float64` before scaling (`* 1e6` when MNE decodes SI volts; factor `1.0` when EEGLAB `.set/.fdt` stores microvolts, cross-checked on a sample slice against MNE). Apply zero added filtering, baseline subtraction, or rereferencing.
   - **Per-subject window caps on long recordings**: To prevent multi-hour tasks or PSGs from generating >20 GB `.ts` text files and dominating archive runtime, cap continuous or high-trial recordings to **at most 30 consecutive clean non-overlapping windows (or 40 native task trials) per participant**, skipping interior `boundary`/`break` markers, multi-sample all-zero dropouts, and constant/railed windows (`std <= 1e-7 uV`). Document the exact deterministic offset/cap rule in `split_definition` and `source_and_conversion_notes`.
   - **Stratified whole-subject 75/25 split**: Within each diagnostic class, sort participant ID strings deterministically (numeric-aware where appropriate while preserving original ID tokens): first `floor(0.75 * n_class)` participants go to `TRAIN`, remainder to `TEST`. Assert zero shared participants across splits (`shared_subjects == []`) and zero cross-split or within-split duplicate waveforms (`cross_split_duplicate_cases == 0`).

4. **Verification, Provenance & Style Boundaries**:
   - Every completed dataset must write `docs/<source-slug>-diagnosis-source-lock.json` and `docs/<source-slug>-diagnosis-conversion-result.json`, and pass `verify_final` (full `.ts` float64 readback, case/row alignment, manifest SHA-256 verification, and `aeon` `SummaryClassifier` fit/predict smoke test).
   - Use **British English** throughout all notes, docstrings, and markdown updates (`normalise`, `artefact`, `behavioural`, `licence`, `characterised`).
   - **Never output em dashes (U+2014)** anywhere in code, comments, JSON strings, or markdown files; use colons, commas, parentheses, or spaced hyphens (` - `).
   - **No remote pushes or uploads**: Do not run `git push`, do not upload to Zenodo/Hugging Face, and do not touch remote clusters.

---

## 2. Phase 1 Queue: Eight Open Clinical Datasets (`20 -> 28 Clinical`)

Execute sequentially in the order below (compact sources first, large streamed sources after):

### `#1` `SchizophreniaConflict` (`PRED+CT d004` / `Albrecht2019`)
- **Source**: `jalauer/Albrecht2019` pinned revision `53b7c8732740f56325687eca673ae2eed8a2c5f6` (original PRED+CT `d004`, `PDDL-1.0`; verify `LICENSE.txt` as in `docs/predct-preparation.md`).
- **Cohort & Target**: Schizophrenia patients vs healthy controls during reinforcement learning / cognitive conflict (Albrecht et al. 2019).
- **Protocol**: Follow `tools/prepare_predct.py` header preflight, native scalp-EEG channel selection (excluding EOG/ECG/accelerometers), float64 microvolts, and non-overlapping 4.0 s windows (capped at **30** clean non-overlapping 4.0 s windows per participant if full continuous recordings exceed 40 windows/person). Stratified 75/25 whole-subject split.

### `#2` `StrokeMotorImagery` (`PardoGarcia2026` / EEGDash `nm000309`)
- **Source**: Zenodo `19599465` (`doi:10.5281/zenodo.19599465`, `CC-BY-4.0`, ~5.4 GB).
- **Cohort & Target**: 18 participants (`9` chronic MCA stroke vs `9` healthy controls; recover exact participant-to-group mapping from original README / metadata and pin in source lock). Target is **stroke vs control diagnosis** (`1 = chronic_mca_stroke`, `2 = healthy_control`), retaining motor-imagery cue identity in `cases_*.parquet` / `*_rows.tsv`.
- **Protocol**: Native 59 scalp EEG channels at 1,000 Hz (exclude EOG/EMG/trigger channels). Extract non-overlapping task-period windows (e.g. 4.0 s / 4,000 samples, capped at **40** clean trials per participant if needed so `.ts` stays compact). Stratified 75/25 whole-subject split (`6` stroke + `6` control `TRAIN`, `3` stroke + `3` control `TEST`).

### `#3` `AlcoholBandit` (OpenNeuro `ds004595` v1.0.0 / EEGDash `on004595`)
- **Source**: OpenNeuro `ds004595` v1.0.0 (`CC0-1.0`, ~8.5 GB; Campbell et al., Cavanagh lab).
- **Cohort & Target**: 53 participants (`27` `AUD` hazardous drinking / alcohol-use group vs `26` `CTRL` controls from `participants.tsv` `GROUP`). Document clearly in sidecars whether `GROUP == AUD` reflects AUDIT/hazardous-drinking clinical screening criteria per the original README/paper.
- **Protocol**: ~66 native channels at 500 Hz; keep scalp EEG channels only (exclude VEOG/HEOG/ECG/XYZ). Extract up to **30** clean non-overlapping 4.0 s (2,000-sample) task windows per participant from the 3-armed bandit recording. Stratified 75/25 whole-subject split.

### `#4` `MildTBIConflict` or `AdultSeizures` (Dedup-Gated Slot)
- **Primary candidate**: `PRED+CT d009` (`Cavanagh2019`, `jalauer/Cavanagh2019` pinned revision `2b799ac8c319a79838fe64bfe1aea0598c16589e`, `PDDL-1.0`, mTBI vs healthy control auditory oddball).
  - **Mandatory preflight gate**: Before bulk download, inspect `Cavanagh2019` participant IDs, clinical table, and sample counts against already-registered `TBIAuditoryOddball` (`ds003522` v1.1.0, `docs/ds003522-diagnosis-source-lock.json`). Admit `Cavanagh2019` (as `TBIOddballSubacute` or `MildTBIOddball`) **only if** it is a verified distinct cohort/acquisition from `ds003522`.
- **Automatic fallback if `Cavanagh2019` overlaps `ds003522`**: Prepare PhysioNet **Siena Scalp EEG Database** v1.0.0 (`https://physionet.org/content/siena-scalp-eeg/1.0.0/`, `DOI: 10.13026/5d4a-j060`, `CC-BY-4.0`) as **`AdultSeizures`** (ictal seizure vs non-seizure interictal scalp EEG across 14 adult patients; derive seizure labels from exact annotation interval overlap rather than boundary-inside-window heuristics, enforce whole-patient `TRAIN`/`TEST` separation, common named 10-20 scalp EEG channels at native Hz, and cap interictal background windows deterministically per recording so total dataset size stays <= 5 GB `.ts`).

### `#5` `PostCovidVigilance` (OpenNeuro `ds007605` v1.0.0 `EPOC` / EEGDash `on007605`)
- **Source**: OpenNeuro `ds007605` v1.0.0 (`CC0-1.0`, ~23.2 GB; stream in participant batches).
- **Cohort & Target**: 97 participants (`60` `withPCS` post-COVID syndrome vs `37` `withoutPCS` recovered controls without PCS from `participants.tsv` `group`; retain `group_Fatigue` and `neurological_diseases_1` in case metadata). Note in metadata that both cohorts had prior SARS-CoV-2 infection and the contrast is Post-COVID Syndrome diagnosis.
- **Protocol**: 128 scalp EEG channels at 1,000 Hz (exclude EOG/ECG/EMG/reference channels). Because `128 ch x 1,000 Hz` is dense, extract **at most the first 25 clean non-overlapping 4.0 s (4,000-sample) windows (first 100 s of task)** per participant (~2,425 cases total) and stream/purge raw participant files in batches. Stratified 75/25 whole-subject split.

### `#6` `SchizophreniaUltimatum` (OpenNeuro `ds004000` v1.0.0 / EEGDash `on004000`)
- **Source**: OpenNeuro `ds004000` v1.0.0 (`CC0-1.0`, ~24.2 GB; Fribourg Ultimatum Game in Schizophrenia Study; stream in participant batches).
- **Cohort & Target**: 43 participants (`19` `P` schizophrenia patients vs `24` `HC` healthy controls from `participants.tsv` `group`).
- **Protocol**: High-density BioSemi/scalp EEG recorded at `2,048 Hz` (~128 scalp EEG channels; exclude `EXG1`..`EXG8` / `Status` external channels).
  - **Rate & size rule**: At `128 ch x 2,048 Hz`, raw 4 s windows would be 262,144 floats/case (~6 GB `.ts` for just 1,000 cases). Either:
    1. Extract native `1.5 s` (`3,072`-sample) event-locked Ultimatum offer epochs at native `2,048 Hz` capped at **25** clean trials per participant, **or**
    2. Anti-alias decimate `4x` (`2,048 Hz -> 512 Hz`, whole-recording float64 MNE FFT/polyphase anti-alias filter before windowing, recording native sample bounds as in `MoralConflict` / `PsychosisRestFirst`) and extract up to **30** clean non-overlapping `4.0 s` (`2,048`-sample at `512 Hz`) windows per participant. Prefer `(2)` if events are missing/irregular across subjects, or `(1)` if clean BIDS offer triggers exist for all 43 subjects and final `.ts` stays <= 5 GB. Stratified 75/25 whole-subject split.

### `#7` `SchizophreniaMotion` (OpenNeuro `ds004367` v1.0.2 `Meta-rdk` / EEGDash `on004367`)
- **Source**: OpenNeuro `ds004367` v1.0.2 (`CC0-1.0`, ~30.0 GB; stream in participant batches).
- **Cohort & Target**: 40 participants (`17` `Patient` schizophrenia vs `23` `Control` healthy controls from `participants.tsv` `Group`).
- **Protocol**: 64 scalp EEG channels at native `1,200 Hz` (exclude 4 EOG/external channels from the 68 total). Extract up to **30** clean non-overlapping `3.0 s` (`3,600`-sample) random-dot motion trial/task windows per participant (~1,200 cases total). Stratified 75/25 whole-subject split.

### `#8` `ZosterNeuralgiaSleep` (OpenNeuro `ds007181` / EEGDash `on007181`)
- **Source**: OpenNeuro `ds007181` latest pinned release (`CC0-1.0`, ~63.5 GB total across 59 ~8-hour overnight PSG recordings).
- **Cohort & Target**: 59 participants (`27` `ZAN` Zoster-associated neuralgia vs `32` `HC` healthy controls from `participants.tsv` `group`).
- **Strict Streaming & Window Cap Protocol**:
  - **Never download all 59 overnight PSGs simultaneously.** Process in batches of **4 to 6 participants**: download native EEG/PSG + BIDS sidecars, verify git-annex checksums, extract windows to `.npz`, and immediately delete the raw ~1 GB PSG file before fetching the next batch.
  - **Scalp EEG channels only**: Out of the 24 PSG channels, retain **only verified scalp EEG derivations** (explicitly exclude EOG, chin/leg EMG, ECG, snore, airflow, thoracic/abdominal effort, SpO2, body position, and pulse channels, recording all excluded PSG channel names/indices in metadata).
  - **Deterministic window cap**: Skip the initial `300 s` (5 min) settling lead-in (and any discontinuity/flat dropout runs) and extract **the first 30 consecutive clean non-overlapping 4.0 s (`4,096`-sample at `1,024 Hz`) windows (120 s total)** per participant (~1,770 cases total; <= 3.5 GB `.ts`). Stratified 75/25 whole-subject split.

---

## 3. Reserve Open Candidates (Pre-Authorised Drop-In Backups)

If any source in `#1` to `#8` fails pre-flight verification due to upstream HTTP 403 / missing S3 objects (as `ds008892` did in Cognitive Group B), contradictory participant labels, or unrecoverable cross-subject waveform duplication, log the failure in `docs/<source>-diagnosis-excluded.json` and substitute immediately from this pre-approved open reserve list without halting the queue:

1. **`ParkinsonsRestPDEEG`** (OpenNeuro `ds007526` v1.0.2 `task-rest`, `CC0-1.0`, ~2.5 GB): Resting-state recordings of the 144-participant `PD-EEG` cohort (`116` PD vs `28` HC; distinct `task-rest` recordings from `ParkinsonsWalking` `task-walk`, documented shared cohort; reuse `tools/prepare_eegdash_pd_walk.py` pattern, capped at 30 clean 4.0 s windows/person at native 250 Hz).
2. **`ParkinsonsReinforcement`** (OpenNeuro `ds003506` v1.1.0, `CC0-1.0`, ~17.4 GB): 56 participants (`28` PD OFF-medication session vs `28` matched controls, Cavanagh lab RL task, 63 scalp EEG channels @ 500 Hz, capped at 30 clean 4.0 s windows/person).
3. **`AcuteTBIVisualMemory`** (OpenNeuro `ds003523` v1.1.0, `CC0-1.0`): 91 participants (`34` acute mTBI vs `57` controls, visual working memory task; distinct task recordings on the UNM mTBI cohort, capped at 30 clean 4.0 s windows/person).

---

## 4. Phase 2: Temple Clinical Additions (`#9` `TUAB` & `#10` `TUEP`, `28 -> 30 Clinical`)

After Phase 1 reaches **28** registered `Clinical` problems:

1. Check local storage (`D:/Data/EEGSources`, `C:/Data/EEGSources`, environment variables, or local cache) for authorised Temple University Hospital EEG corpus files or credentials for **`TUAB v3.0.1`** (`TempleAbnormalEEG`, normal vs abnormal clinical EEG) and **`TUEP v2.0.1`** (`TempleEpilepsyEEG`, epilepsy diagnosis vs non-epilepsy control), or check whether an authorised local mirror cache is present.
2. **Licence & redistribution rule**: Temple's subscriber agreement prohibits public redistribution. Do **not** scrape unauthorised mirrors or redistribute Temple signals. When `TempleAbnormalEEG` and `TempleEpilepsyEEG` are prepared from authorised local/credentialled files:
   - Add both names to `neuroverse.datasets.restricted` alongside `FibroLiverpool` and `FibroUEA`.
   - Harmonise to the standard common 19-channel 10-20 scalp EEG montage (`Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`, `F7`, `F8`, `T3`/`T7`, `T4`/`T8`, `T5`/`P7`, `T6`/`P8`, `Fz`, `Cz`, `Pz` in consistent reference/derivation) at native/common `250 Hz`, whole-patient `TRAIN`/`TEST` split (preserving official TUAB `train`/`eval` patient separation and auditing patient IDs across `TUAB` and `TUEP`), capped deterministically per patient so each `.ts` export stays <= 6 GB.
3. **If Temple files/credentials are not yet present on disk**: Do **not** block Phase 1 registration. Register all 8 Phase 1 datasets cleanly (`28 Clinical`), write `docs/tuab-tuep-readiness.json` stating the exact path where `TUAB v3.0.1` and `TUEP v2.0.1` archives should be placed (or whether to substitute 2 from Section 3 instead), and report completion of Phase 1.

---

## 5. Registration & Synchronisation Checklist

Once a batch completes `verify_final`:
1. Add `(name, channels, classes, timepoints, subjects, train_cases, test_cases, frequency, "Med")` to `_TABLE` in `neuroverse/datasets.py` and `STUDY_TAXONOMY[name] = ("Clinical", "")` (leaving existing `MI`, `Cognitive`, `ME`, `ERP`, `VEP`, and `Speech` entries intact until the final 90-problem `30/30/30` switch is requested).
2. Update `eeg.txt`, `docs/datasets.md`, `docs/dataset-provenance.md`, `docs/problem-taxonomy.md`, `docs/problem-taxonomy.json`, `docs/eegdash-diagnosis-progress.json`, `docs/eegdash-diagnosis-summary.json`, and `neuroverse/tests/test_datasets.py`.
3. Synchronise `results/neuroverse/datasets.csv`, `Default_*.csv`, `balacc.csv`, `accuracy.csv`, `auroc.csv`, `logloss.csv`, and HTML leaderboards using `neuroverse/experiments` (matching commit `e3d69ed`).
4. Run `pytest` on `neuroverse/tests/test_datasets.py` and `tools/test_prepare_diagnostic_to_30.py` using `--basetemp=C:/Code/EEGBench/.openeegbench/test-<uuid>` and confirm green.
