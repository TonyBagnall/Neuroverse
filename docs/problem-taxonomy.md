# Proposed problem taxonomy

Snapshot 15 November 2026, based on `eeg.txt` (primary **90-problem `30 / 30 / 30`** open-access scalp EEG benchmark: `30 MI + 30 Clinical + 30 Cognitive`) and `eeg_114.txt` (extended **114-problem** open-access scalp EEG study including `6 ME`, `8 ERP`, `6 VEP`, and `4 Speech` BCI tasks) alongside documented prediction targets. One primary category/subclass per selected problem. These assignments do not overwrite the existing Med/BCI/Psy domains or source sidecars. They are proposed editorial groupings, not newly verified acquisition facts.

**90 primary (`30 / 30 / 30`) open scalp EEG problems; 114 extended open scalp EEG problems; 125 registered archive problems.** The study excludes FaceDetection, HandMovementDirection, MatchWords and MindReading (MEG), MotorImagery (ECoG), and the two restricted-access fibromyalgia datasets (FibroLiverpool and FibroUEA, replaced in `Clinical` by open-access `LearningDifficulties` and `FamilialAlzheimersRest` so all 90 and 114 selected problems are 100% open-access). Blink, ButtonPress, EpilepticSeizures and Sleep remain excluded for the existing reasons. All registry entries, archived 2026 paper scores, and local files are retained. Planned downloads/pilots are not counted; problems may share studies or source recordings.

| Category | Selected scalp EEG problems |
|---|---:|
| Clinical and physiological states | 32 |
| Brain-computer interfaces | 51 |
| Cognitive and affective states | 28 |
| Behaviour and activity | 3 |
| Individual differences | 0 |
| **Total** | **114** |

*(Note: in `neuroverse.datasets.STUDY_TAXONOMY`, the two physiological eye-state tasks `EyesOpenShut` and `RestingEyeState` sit in `Cognitive` (`30` total) while the `30` clinical disorder, seizure, sleep, and symptom-screening tasks form `by_type("Clinical")` (`30` total), alongside `by_type("BCI")` (`54` total: `30` `MI`, `6` `ME`, `8` `ERP`, `6` `VEP`, `4` `Speech`).)*

The study has **30 strict scalp-EEG motor imagery** problems (`by_type("MI")`, adding `ImaginedFingers` / `Damm2026` to replace the excluded ECoG problem `MotorImagery`), **30 open-access clinical scalp-EEG** problems (`by_type("Clinical")`, adding `MigraineRest`, `ParkinsonsWalking`, `PsychosisRestFirst`, `PsychosisRestSecond`, `TBIAuditoryOddball`, `AcuteTBIVisualMemory`, `AdultSeizures`, `AlcoholBandit`, `FamilialAlzheimersRest`, `LearningDifficulties`, `ParkinsonsReinforcement`, `PostCovidVigilance`, `SchizophreniaConflict`, `SchizophreniaMotion`, `SchizophreniaUltimatum`, `StrokeMotorImagery`, and `ZosterNeuralgiaSleep`, with `FamilialAlzheimersRest` and `LearningDifficulties` replacing restricted `FibroUEA` and `FibroLiverpool`), **30 cognitive and affective scalp-EEG** problems (`by_type("Cognitive")`, adding Cognitive Group A: `ArithmeticDifficulty`, `CognitiveLoadTasks`, `MultitaskWorkload`, `NBackDifficulty`, `NaturalisticEmotion`, `SleepDeprivation`, and `SternbergMemory`; plus Cognitive Group B: `AffectiveMusic`, `AuditoryDeviance`, `BanditGambling`, `FaceStimuli`, `FlankerConflict`, `FuzzySemantics`, `InternalAttention`, `MoralConflict`, `NarrativeReward`, `ProbabilisticFeedback`, `PythonCodeReading`, `SentenceViolations`, `SpatialAttention`, `VerbalWorkingMemory`, and `VisualSymmetry`), and **6 executed / movement-intention motor tasks** (`by_type("ME")`: `ArmCoordination`, `ArmMovements`, `FeetHands`, `FingerMovements`, `MatchingPennies`, `OpenCloseFist`).

## Subclasses and complete assignment

| Category | Subclass | Problems | Names |
|---|---|---:|---|
| Clinical and physiological states | Medical diagnosis / clinical cohort | 26 | AcuteTBIVisualMemory, AlcoholBandit, Alzheimers, AlzheimersAuditory, CovidSurvivors, FamilialAlzheimersRest, LearningDifficulties, LongIntervalTask, MigraineRest, ParkinsonsPedalling, ParkinsonsReinforcement, ParkinsonsRest, ParkinsonsRestIowa, ParkinsonsRestUCSD, ParkinsonsWalking, PhotoStimulation, PostCovidVigilance, PsychosisRestFirst, PsychosisRestSecond, SchizophreniaConflict, SchizophreniaMotion, SchizophreniaUltimatum, ShortIntervalTask, StrokeMotorImagery, TBIAuditoryOddball, ZosterNeuralgiaSleep |
| Clinical and physiological states | Seizure detection | 2 | AdultSeizures, ChildhoodSeizures |
| Clinical and physiological states | Sleep staging | 1 | ClinicalSleepStages |
| Clinical and physiological states | Physiological eye state | 2 | EyesOpenShut, RestingEyeState |
| Clinical and physiological states | Symptom screening (not diagnosis) | 1 | ObsessiveCompulsiveSymptoms |
| Brain-computer interfaces | ERP / P300 target detection | 6 | ALSAttention, GameTargets, SpellerFlashes, SymbolAttention, VisualSubjects, WalkingTargets |
| Brain-computer interfaces | Motor execution | 5 | ArmCoordination, ArmMovements, FeetHands, MatchingPennies, OpenCloseFist |
| Brain-computer interfaces | Code-modulated VEP | 2 | CodedFlickerTargets, CodedSymbols |
| Brain-computer interfaces | SSVEP | 4 | FlickerAttention, FlickerSymbols, FlickerTargets, WirelessFlicker |
| Brain-computer interfaces | Motor imagery | 30 | ImaginedAnkleRest, ImaginedArmMovements, ImaginedArmTasks, ImaginedFeetHands, ImaginedFingers, ImaginedGrasps, ImaginedGripForce, ImaginedHandFeetTongue, ImaginedHandFootActions, ImaginedHandGestures, ImaginedHandMusicTempo, ImaginedHandPupils, ImaginedHandSides, ImaginedHandTraining, ImaginedHandsCrossSession, ImaginedHandsEmbodiment, ImaginedHandsFeedback, ImaginedHandsFeet, ImaginedHandsHaptics, ImaginedHandsOnline, ImaginedHandsPhysiology, ImaginedHandsRobot, ImaginedHandsStroke, ImaginedLeftRightHand, ImaginedLeftRightHandRest, ImaginedOpenCloseFist, ImaginedRightHandFeet, ImaginedRightHandFeetRest, LowCost, SitStand |
| Brain-computer interfaces | Imagined speech | 2 | ImaginedCommands, InnerSpeech |
| Brain-computer interfaces | Slow cortical potential self-regulation | 2 | SelfRegulationSCP1, SelfRegulationSCP2 |
| Cognitive and affective states | Workload / joint task condition | 6 | ArithmeticDifficulty, CognitiveLoadTasks, MentalArithmetic, MultitaskWorkload, NBackAudioCondition, NBackDifficulty |
| Cognitive and affective states | Working memory maintenance | 2 | SternbergMemory, VerbalWorkingMemory |
| Cognitive and affective states | Memory / recognition performance | 1 | SongFamiliarity |
| Cognitive and affective states | Emotion | 3 | AffectiveMusic, NaturalisticEmotion, VideoEmotions |
| Cognitive and affective states | Vigilance / sleep deprivation | 1 | SleepDeprivation |
| Cognitive and affective states | Visual imagery | 1 | VisualSpeech |
| Cognitive and affective states | Semantic processing | 4 | FuzzySemantics, PythonCodeReading, SentenceViolations, WordRelatedness |
| Cognitive and affective states | Cognitive control and conflict | 2 | FlankerConflict, MoralConflict |
| Cognitive and affective states | Reward and feedback valuation | 3 | BanditGambling, NarrativeReward, ProbabilisticFeedback |
| Cognitive and affective states | Attention and sensory deviance | 3 | AuditoryDeviance, InternalAttention, SpatialAttention |
| Cognitive and affective states | Visual perception | 2 | FaceStimuli, VisualSymmetry |
| Behaviour and activity | Movement intention | 2 | FeedbackButton, FingerMovements |
| Behaviour and activity | Overt speech | 1 | PronouncedSpeech |

Subclasses not represented: anaesthesia, consciousness, dedicated error-related-potential decoding, stress, generic human activity recognition, subject identification/authentication and demographic prediction. This does not mean every task has been audited for every possible secondary interpretation. P300 attention targets are counted under BCI rather than counted again as cognitive attention.

## Assignment decisions

Use the **prediction target**, with BCI motor execution following the supplied taxonomy. Clinical cohorts performing a motor task stay clinical when disease/cohort is the label. Motor preparation versus future movement belongs under behaviour/movement intention. Eye-open/closed is a physiological state rather than assumed vigilance. Symptom screening is separated from clinical diagnosis.

ERP is a cross-cutting signal/paradigm family:6 BCI P300 tasks plus1 N400 semantic EEG task (WordRelatedness) gives **7 explicitly documented ERP problems**. Do not turn all7 into P300, or count them twice in the primary-category total. MatchWords is excluded MEG and is not counted.

The previous P300 review missed existing GameTargets (BrainForm/RomaniBF2025ERP) and WalkingTargets. Its baseline4 was incorrect; the correct current baseline is6P300/7explicit ERP. RomaniBF2025ERP must also be removed from proposed new additions because it is already GameTargets.

- **ObsessiveCompulsiveSymptoms:** OCI-threshold symptom grouping in students; not clinical OCD diagnosis.
- **CovidSurvivors:** COVID-survivor versus close-contact cohort membership; not current disease diagnosis.
- **SitStand:** Mixed imagined transitions and seated/standing idle classes; included in the existing30 MI tally.
- **MatchingPennies:** Executed hand raising while imagining the opposite hand; primary assignment motor execution, secondary motor imagery.
- **VisualSpeech:** Visualised circle direction, not spoken or imagined speech.
- **SongFamiliarity:** Correct versus incorrect song identification, not simply familiar versus unfamiliar.
- **NBackAudioCondition:** Joint3 N-back levels x4 audio codes; not a pure workload contrast.
- **WordRelatedness:** ERP/N400 semantic task; ERP is also a cross-cutting response/paradigm tag.
- **FeedbackButton:** Press/no-press target in prompt-period windows ending before press; original acquisition provenance unresolved.
- **ALSAttention:** Attended versus unattended flashes in ALS participants; not ALS diagnosis.
- **ParkinsonsPedalling:** PD versus control during pedalling; diagnosis target, not activity classification.
- **ParkinsonsWalking:** PD versus control during active treadmill walking; diagnosis target, not gait activity classification.
- **PsychosisRestFirst / PsychosisRestSecond:** Disjoint first-episode psychosis versus healthy control cohorts at eyes-open rest (`ds003944` and `ds003947`).
- **TBIAuditoryOddball:** 3-class TBI diagnosis (control, sub-acute mild TBI, chronic TBI) during Session 1 auditory oddball deviant tones (`ds003522`); diagnosis target, not stimulus-type classification.
- **MigraineRest:** Unmedicated interictal migraineurs versus age- and gender-matched controls during central-cross fixation (`Chamanzar2020`).
- **StrokeMotorImagery:** Subacute and chronic post-stroke patients versus healthy controls during hand motor imagery (`PardoGarcia2026`, Zenodo `19599466`); clinical cohort target (`Med` / `Clinical`), not left-versus-right MI decoding.
- **AdultSeizures:** Focal ictal seizure versus matched interictal non-seizure 4-second scalp EEG windows across 14 adult epilepsy patients (`Siena Scalp EEG Database`, PhysioNet `siena-scalp-eeg` v1.0.0; activated after `PRED+CT d009` `Cavanagh2019` was verified to duplicate `TBIAuditoryOddball`).
- **AlcoholBandit:** Alcohol use disorder (`AUD`) versus non-AUD control cohort during a three-armed bandit task (`ds004595` v1.0.0); clinical cohort target, not reward-valence classification.
- **SchizophreniaConflict / SchizophreniaUltimatum / SchizophreniaMotion:** Three independent clinical schizophrenia cohorts versus healthy controls recorded in distinct cognitive/perceptual paradigms: cognitive control (`Albrecht2019`, `PRED+CT d004`), Ultimatum Game social decision-making (`ds004000` v1.0.0), and coherent motion perception (`ds004367` v1.0.2); diagnosis target in all three.
- **PostCovidVigilance:** Post-COVID neurological fatigue cohort versus healthy controls during psychomotor vigilance testing (`ds007605` v1.0.0); clinical cohort target.
- **ZosterNeuralgiaSleep:** Acute herpes zoster neuralgia (`AHZN`) versus chronic postherpetic neuralgia (`PHN`) pain transition stages during nocturnal sleep EEG (`ds007181` v1.0.1).
- **ParkinsonsReinforcement:** OFF-medication Parkinson's disease patients versus matched controls during probabilistic reinforcement learning (`ds003506` v1.1.0; shared UNM cohort with resting-state `ParkinsonsRest` `ds003490`, separate task recordings).
- **AcuteTBIVisualMemory:** Sub-acute mild traumatic brain injury (3-14 days post-injury, with loss of consciousness) versus matched healthy controls during change-detection visual working memory (`ds003523` v1.1.0; shared UNM baseline cohort with `TBIAuditoryOddball` `ds003522`, separate task recordings).
- **LearningDifficulties:** Paediatric mathematical (`sub-M*`) versus reading (`sub-R*`) learning difficulty subtypes during pre-intervention baseline eyes-open resting EEG (`ses-1`, `run-1` of `ds006260` v1.0.1; clinical diagnostic subtype target, replacing restricted `FibroLiverpool`).
- **FamilialAlzheimersRest:** Asymptomatic `PSEN1-E280A` autosomal-dominant familial Alzheimer's mutation carriers (`sub-G1*`) versus matched non-carrier family controls (`sub-G2*`) during eyes-closed resting EEG in the Antioquia kindred (`ds007427` v1.0.2; completely independent cohort and genetic aetiology from Greek hospital sporadic `Alzheimers` / `PhotoStimulation` and `AlzheimersAuditory`, replacing restricted `FibroUEA`).
- **LongIntervalTask:** PD versus control in interval timing; diagnosis target, not timing-task identity.
- **ShortIntervalTask:** PD versus control in interval timing; diagnosis target, not timing-task identity.
- **ArithmeticDifficulty / NBackDifficulty:** Graded mental arithmetic Q-value difficulty (`7` classes, `ds007262`) and visual n-back load (`4` classes, `ds007169`) from the shared University of Hull 20-participant laboratory cohort (Barras & Booth); separate recordings and cognitive paradigms, not independent participant pools.
- **MultitaskWorkload:** 3-level NASA MATB-II multitasking workload across sessions S1 and S2 from the Neuroergonomics 2021 competition (`Zenodo 5055046`), a 15-subject MATB subset of Hinss et al. (`COG-BCI`).
- **CognitiveLoadTasks:** 4-level protocol-defined cognitive load (`neutral`, `low`, `medium`, `high`) pooled across Stroop and mental arithmetic tasks (`CLATEEG26`).
- **SternbergMemory:** 3-level working-memory maintenance load (`3`, `5`, or `7` memorised letters during the 2.0 s post-encoding delay, `ds004117`).
- **NaturalisticEmotion:** 4-quadrant circumplex emotion (`HVHA`, `HVLA`, `LVHA`, `LVLA` from numeric participant ratings `valence >= 5` and `arousal >= 5`) across the 9 byte-intact rated subjects in `DENS` (`ds003751`).
- **SleepDeprivation:** Normal sleep versus acute sleep deprivation during eyes-open resting EEG across 68 byte-intact paired participants (`ds004902`).
- **PythonCodeReading / FuzzySemantics / SentenceViolations:** Code anomaly (`4` classes, `ds004771`), natural-reading semantic target detection (`2` classes, `ds005383`), and RSVP syntactic/semantic/word-order sentence violations (`5` classes, `ds008711`).
- **FlankerConflict / MoralConflict:** Orthographic flanker interference crossed with lexicality (`6` classes, `ds005866`) and moral intentionality crossed with emotional congruency (`4` classes, `ds004860`).
- **ProbabilisticFeedback / NarrativeReward / BanditGambling:** Reinforcement and gambling win/loss feedback valuation across affective cues (`4` classes, `ds003753`), narrative contexts (`4` classes, `ds007968`), and oscillating 3-armed bandits (`2` classes, `ds003458`).
- **InternalAttention / VisualSymmetry:** Internal mental concentration versus relaxation (`2` classes, `ds002691`) and regular reflection/rotation symmetry versus random dot patterns (`2` classes, `ds004347`).
- **AffectiveMusic / VerbalWorkingMemory:** Self-rated music pleasantness by energy circumplex quadrants (`4` classes, `ds002721`) and verbal Sternberg load (`5`/`6`/`7` letters) crossed with retention versus manipulation (`6` classes, `ds003655`).
- **FaceStimuli / AuditoryDeviance / SpatialAttention:** `ERP CORE` `N170` (`4` visual categories, `ds003061`), `MMN` (`2` passive auditory oddball tones, `ds003064`), and `N2pc` (`2` covert visual attention sides, `ds003065`) recorded at native `1,024 Hz` on the same 40-participant cohort (`001..030` `TRAIN`, `031..040` `TEST`) as `WordRelatedness` (`N400`, `ds003068`); distinct paradigms and recordings, not independent cohorts.

Sources: [dataset descriptions](datasets.md), [provenance](dataset-provenance.md), [BrainForm and other conversions](open-moabb-conversions.md), [selected list](../eeg.txt), [registry](../neuroverse/datasets.py). Machine-readable assignments: [problem-taxonomy.json](problem-taxonomy.json).

## Excluded modalities and restricted datasets retained in the archive

- **FaceDetection:** MEG; excluded from this study, data and registry retained.
- **HandMovementDirection:** MEG; excluded from this study, data and registry retained.
- **MatchWords:** MEG; excluded from this study, data and registry retained.
- **MindReading:** MEG; excluded from this study, data and registry retained.
- **MotorImagery:** ECoG; excluded from this study, data and registry retained.
- **FibroLiverpool:** Restricted-access scalp EEG; excluded from the open-access 90/114 study lists and retained in `neuroverse_paper` for Rushbrooke et al. (2026) reproduction.
- **FibroUEA:** Restricted-access scalp EEG; excluded from the open-access 90/114 study lists and retained in `neuroverse_paper` for Rushbrooke et al. (2026) reproduction.
