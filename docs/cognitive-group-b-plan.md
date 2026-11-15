# Cognitive Group B Assimilation Plan & Completion Record (+15 Problems -> 30 Cognitive)

This document specifies and records the conversion, verification, provenance, and storage layout for **Cognitive Group B** (`+15` scalp-EEG classification problems), expanding `by_type("Cognitive")` from `15` to `30` problems (`neuroverse_study` `89 -> 104`, `neuroverse` `98 -> 113`).

## 1. Core Conversion Principles

1. **Native sampling rates retained everywhere**: Every dataset is exported at its native acquisition/release `SamplingFrequency` (`200`, `250`, `256`, `500`, `512`, `1,000`, or `1,024 Hz`), with zero added filtering, baseline subtraction, or rereferencing. The sole within-dataset rate harmonisation is `sub-110` in `MoralConflict` (`ds004860`), which was recorded at `2,048 Hz` while the remaining `30` participants were recorded at `512 Hz`; `sub-110` is anti-alias decimated `4x` onto `512 Hz` before epoching.
2. **Scalp EEG channels only**: Documented EOG, ECG/EKG, SCR, EMG, mastoid reference (`A2`, `M1`, `M2`), trigger (`Status`), and non-EEG channels are excluded before export. Where channel order varies across recordings (`ds003458`, `ds003753`), named shared scalp channels are aligned onto a verified canonical sensor order.
3. **Whole-participant 75/25 splits**: Every dataset uses a deterministic whole-participant `TRAIN` / `TEST` split (`floor(0.75 * N)` `TRAIN`, remainder `TEST`; subjects `001..030` `TRAIN` and `031..040` `TEST` on the three `ERP CORE` paradigms to match `WordRelatedness`).
4. **Hardware-railed, duplicate, and truncated file guards**: Individual trials exhibiting constant/railed channels (`std <= 1e-7 uV`), duplicate signal hashes (`sub-149` in `ds003655`, leaving `155` unique retained participants), or upstream byte-truncated binary files are excluded and logged in conversion sidecars rather than corrupting sensor sets globally.
5. **Disk staging and storage placement**: Temporary raw downloads staged in `C:/Data/EEGSources/CognitiveGroupB` and were purged immediately after `verify_final` passed. All 15 datasets reside on `D:/Data/EEG/<Dataset>` and have been verified end-to-end against their SHA-256 manifests (`<Dataset>_manifest.json`).

## 2. Final Verified Dataset Specifications

| # | Dataset | Source & Snapshot | Licence | `Ch` | `Cls` | `Len` | `Hz` | `Subj` | `Train` | `Test` | Path |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `PythonCodeReading` | OpenNeuro `ds004771` v1.0.0 | `CC0-1.0` | 32 | 4 | 334 | 256 | 61 | 7,177 | 2,560 | `D:/Data/EEG/PythonCodeReading` |
| 2 | `FuzzySemantics` | OpenNeuro `ds005383` v1.0.0 (`TMNRED`) | `CC0-1.0` | 30 | 2 | 200 | 200 | 30 | 8,780 | 3,191 | `D:/Data/EEG/FuzzySemantics` |
| 3 | `InternalAttention` | OpenNeuro `ds002691` v1.1.0 | `CC0-1.0` | 32 | 2 | 500 | 250 | 20 | 6,071 | 2,093 | `D:/Data/EEG/InternalAttention` |
| 4 | `VisualSymmetry` | OpenNeuro `ds004347` v1.0.0 (`SPN1`) | `CC0-1.0` | 64 | 2 | 512 | 512 | 24 | 2,879 | 960 | `D:/Data/EEG/VisualSymmetry` |
| 5 | `MoralConflict` | OpenNeuro `ds004860` v1.0.0 | `CC0-1.0` | 32 | 4 | 512 | 512 | 31 | 2,488 | 866 | `D:/Data/EEG/MoralConflict` |
| 6 | `ProbabilisticFeedback` | OpenNeuro `ds003753` v1.1.0 | `CC0-1.0` | 59 | 4 | 500 | 500 | 25 | 2,877 | 1,120 | `D:/Data/EEG/ProbabilisticFeedback` |
| 7 | `SentenceViolations` | OpenNeuro `ds008711` v1.0.0 | `CC0-1.0` | 29 | 5 | 500 | 500 | 64 | 8,640 | 2,880 | `D:/Data/EEG/SentenceViolations` |
| 8 | `NarrativeReward` | OpenNeuro `ds007968` v1.0.0 | `CC0-1.0` | 30 | 4 | 1,000 | 1,000 | 40 | 4,319 | 1,440 | `D:/Data/EEG/NarrativeReward` |
| 9 | `FlankerConflict` | OpenNeuro `ds005866` v1.0.1 (`Flankers-NEAR`) | `CC0-1.0` | 29 | 6 | 500 | 500 | 60 | 12,148 | 4,050 | `D:/Data/EEG/FlankerConflict` |
| 10 | `BanditGambling` | OpenNeuro `ds003458` v1.1.0 | `CC0-1.0` | 63 | 2 | 500 | 500 | 23 | 8,146 | 2,870 | `D:/Data/EEG/BanditGambling` |
| 11 | `AffectiveMusic` | OpenNeuro `ds002721` v1.0.3 | `CC-BY-4.0` / `CC0` | 19 | 4 | 4,000 | 1,000 | 31 | 2,588 | 904 | `D:/Data/EEG/AffectiveMusic` |
| 12 | `VerbalWorkingMemory` | OpenNeuro `ds003655` v1.0.2 | `CC0-1.0` | 19 | 6 | 1,000 | 500 | 155 | 13,910 | 4,676 | `D:/Data/EEG/VerbalWorkingMemory` |
| 13 | `FaceStimuli` | `ErpCore2021_N170` (`ds003061` / Zenodo `14866307`) | `CC-BY-SA-4.0` | 30 | 4 | 1,024 | 1,024 | 40 | 9,600 | 3,200 | `D:/Data/EEG/FaceStimuli` |
| 14 | `AuditoryDeviance` | `ErpCore2021_MMN` (`ds003064` / Zenodo `14866307`) | `CC-BY-SA-4.0` | 30 | 2 | 1,024 | 1,024 | 40 | 11,934 | 4,000 | `D:/Data/EEG/AuditoryDeviance` |
| 15 | `SpatialAttention` | `ErpCore2021_N2pc` (`ds003065` / Zenodo `14866307`) | `CC-BY-SA-4.0` | 30 | 2 | 1,024 | 1,024 | 40 | 9,607 | 3,200 | `D:/Data/EEG/SpatialAttention` |

*(Note: `ds008892` `CuBiAAD` returned `HTTP 403 Forbidden` on OpenNeuro S3/CRN binary downloads during pre-flight verification and was replaced by `BanditGambling` `ds003458` from the pre-audited backup shortlist.)*
