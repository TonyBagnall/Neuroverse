# Cognitive Group A Preparation Plan (Batch 1: +7 Datasets, 8 -> 15 Cognitive)

This specification locks the conversion rules, channel hygiene, sampling-rate harmonisation, event windowing, and whole-subject 75/25 splits for the seven Group A cognitive and affective scalp-EEG benchmark problems assimilated into `neuroverse_study`.

## General Rules

1. **Scalp EEG only**: Auxiliary `ECG`, `EOG`, `EMG`, `MISC`, `PNS`, and online zero-valued vertex reference (`E129`) channels are excluded prior to export. Any retained channel exhibiting `std <= 1e-7 uV` on any case is removed globally by `tools/prepare_open_moabb.py`.
2. **Whole-subject 75/25 splits**: Sorted participant identifiers are partitioned with `floor(0.75 * N)` participants in `TRAIN` and the remaining participants in `TEST`, with zero shared subjects across splits.
3. **One problem per collection**: Each source collection contributes a single locked classification task. Shared laboratory cohorts (`ds007262` and `ds007169`, Barras & Booth) and sub-archive relationships (`Neuro2021` within Hinss et al. 2021) are documented explicitly in dataset metadata and `docs/dataset-provenance.md`.
4. **Sequential low-disk execution**: Temporary `recording-*.npz` intermediates and raw binary downloads are purged only after full `.ts` float64 exact readback, `aeon` smoke validation, sidecar enrichment, and manifest SHA-256 verification succeed.

## Dataset Specifications

1. **`ArithmeticDifficulty`** (OpenNeuro `ds007262` v1.1.0, `CC0-1.0`):
   - **Cohort**: 18 QC-passed participants (`analysis_included == true`; `sub-002` and `sub-017` excluded by source QC), 13 `TRAIN` / 5 `TEST`.
   - **Signal**: 19 10-20 scalp EEG channels (`Pz`..`FP1`; `D1` ECG and `D2`..`D5` MISC excluded) at 250 Hz in microvolts.
   - **Windowing & classes**: 7 graded Q-value difficulty classes (`0.6_to_1.5` through `6.0_to_6.9`) on non-tutorial trials (`istutorial == false`). Each 6.0 s (1,500-sample) epoch starts at the arithmetic question stimulus marker immediately preceding the paired response/outcome marker (minimum stimulus-to-next-stimulus span 6.785 s, strictly non-overlapping).

2. **`NBackDifficulty`** (OpenNeuro `ds007169` v1.0.5, `CC0-1.0`):
   - **Cohort**: 18 QC-passed participants (`analysis_included == true`; `sub-013` and `sub-017` excluded by source QC), 13 `TRAIN` / 5 `TEST`.
   - **Signal**: 19 10-20 scalp EEG channels (`Pz`..`FP1`; `D1` ECG and `D2`..`D5` MISC excluded) at 250 Hz in microvolts.
   - **Windowing & classes**: 4 working-memory load classes (`1_back`, `2_back`, `3_back`, `4_back`) on non-tutorial trials (`istutorial == false`), 1.5 s (375-sample) post-stimulus epochs within the ~1.70 s SOA (strictly non-overlapping).

3. **`MultitaskWorkload`** (Zenodo `5055046`, Neuroergonomics 2021 Grand Challenge / Hinss et al. 2021, `CC-BY-SA-4.0`):
   - **Cohort**: 15 participants (`P01`..`P15`), 11 `TRAIN` / 4 `TEST`.
   - **Signal**: 61 scalp EEG channels at 250 Hz in microvolts from labelled sessions `S1` and `S2` (unlabelled competition test session `S3` and resting files excluded).
   - **Windowing & classes**: Native released 2.0 s (500-sample) non-overlapping epochs across 3 MATB-II cognitive workload classes (`easy`, `medium`, `difficult`).

4. **`CognitiveLoadTasks`** (Mendeley Data `10.17632/pxysmjn3j2.3`, `CLATEEG26`, `CC-BY-4.0`):
   - **Cohort**: 40 participants (`subject-01`..`subject-40`), 30 `TRAIN` / 10 `TEST`.
   - **Signal**: 8 OpenBCI Cyton 10-20 scalp EEG channels (`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `C4`, columns 1..8) at 250 Hz in microvolts across both `Stroop_Data` and `Arithmetic_Data`.
   - **Windowing & classes**: Up to the first 15 consecutive non-overlapping 4.0 s (1,000-sample) windows (first 60 s) per task-condition recording across 4 protocol-defined cognitive load classes (`neutral`, `low`, `medium`, `high`).

5. **`SternbergMemory`** (OpenNeuro `ds004117` v1.0.1, Onton, Delorme & Makeig 2005, `CC0-1.0`):
   - **Cohort**: 23 participants (`sub-001`..`sub-012`, `sub-014`..`sub-024`), 17 `TRAIN` / 6 `TEST`.
   - **Signal**: 69 scalp EEG channels (`LEYE` and `REYE` EOG excluded), harmonised to 250 Hz in microvolts via whole-recording anti-alias resampling where native runs were recorded at 500 Hz, 500.059 Hz, or 1,000 Hz.
   - **Windowing & classes**: 2.0 s (500-sample) maintenance-delay epochs starting at `task_role == work_memory` (`show_dash`, minimum delay to probe 2.233 s, excluding `bad_trial` and 3 unmarked +/-1,000 uV ADC rail-saturated trials in `sub-004_ses-01_run-2`) across 3 memory-load classes (`load_3`, `load_5`, `load_7`).

6. **`NaturalisticEmotion`** (OpenNeuro `ds003751` v1.0.6, `DENS`, Mishra et al. 2023, `CC0-1.0` in `dataset_description.json` / `ODbL-1.0` in `LICENSE`):
   - **Cohort**: 9 byte-intact participants with released `beh/*_beh.tsv` self-assessment ratings (`6` `TRAIN` / `3` `TEST`; 6 recordings lacking `_beh.tsv` and 25 recordings whose `.fdt` binaries are truncated mid-file across all OpenNeuro `ds003751` releases v1.0.0-v1.0.6 relative to `EEG.nbchan * EEG.pnts * 4` are excluded and logged). Duplicate `sub-mit081` row in `participants.tsv` is deduplicated.
   - **Signal**: 128 active EGI HydroCel scalp EEG channels (`E1`..`E128`; zero-valued vertex reference `E129` and peripheral `ECG`, `EMG`, `EMG_2` excluded) at 250 Hz in microvolts.
   - **Windowing & classes**: Consecutive non-overlapping 4.0 s (1,000-sample) windows within non-neutral emotional video intervals (`[onset_stm, onset_vlnc]`, where `events.tsv` `onset` stores 1-based 250 Hz sample latencies, split around any impedance/boundary markers), labelled into 4 circumplex classes (`HVHA`, `HVLA`, `LVHA`, `LVLA`) from numeric participant self-ratings (`valence >= 5.0` vs `< 5.0`, `arousal >= 5.0` vs `< 5.0`).

7. **`SleepDeprivation`** (OpenNeuro `ds004902` v1.0.8, Xiang et al. 2024, `CC0-1.0`):
   - **Cohort**: 68 byte-intact paired participants with both `ses-1` (`normal_sleep`) and `ses-2` (`sleep_deprived`) `task-eyesopen` recordings (`51` `TRAIN` / `17` `TEST`; `sub-01`, `sub-28`, and `sub-44` excluded and logged due to upstream byte-truncated `ses-2` `.fdt` binaries on OpenNeuro).
   - **Signal**: 61 scalp EEG channels harmonised to 500 Hz in microvolts (oversampled 5,000 Hz recordings - `sub-39_ses-2` and `sub-43_ses-2` among retained subjects - are anti-alias decimated by factor 10 before windowing).
   - **Windowing & classes**: First 30 consecutive non-overlapping 4.0 s (2,000-sample) eyes-open resting windows (first 120 s) per session across 2 vigilance classes (`normal_sleep`, `sleep_deprived`).
