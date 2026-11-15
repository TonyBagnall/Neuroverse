# Cognitive and affective EEG expansion

Source/metadata review10November2026. Baseline6selected scalp EEG problems in this category. No EEG recordings downloaded for this survey; no candidate admitted. Dimensions are reported metadata, not native-file verification.

## Recommended first batch

Seven candidate problems could increase the category from6to13 if every source, licence and conversion gate passes. Prioritise the first4new workload acquisitions/protocols, then3ERP CORE tasks with the already-documented source terms review. Shared cohorts do not imply duplicate recordings, and separate task recordings do not imply independent studies.

| Source / proposed name | Task | People | EEG / Hz | Licence/access |
|---|---|---:|---|---|
| [OpenNeuro ds007262](https://openneuro.org/datasets/ds007262) / ArithmeticDifficulty | Baseline plus7graded arithmetic levels;8labels total, exact event definitions to verify | 18 | 19 / 250 | CC0 original dataset_description verified |
| [OpenNeuro ds007169](https://openneuro.org/datasets/ds007169) / NBackDifficulty | 1-,2-,3-,4-back; keep tutorial separate/excluded from main task protocol | 18 | 19 / 250 | CC0 original dataset_description verified |
| [Neuro2021 / Hinss2021 competition](https://zenodo.org/records/5055046) / MultitaskWorkload | MATB-II easy/medium/difficult; optional rest class only if original protocol supports | 15 | 61 / 250 | CC-BY-SA-4.0 original Zenodo5055046 verified |
| [CLATEEG26](https://data.mendeley.com/datasets/pxysmjn3j2/1) / CognitiveLoadTasks | Neutral/low/medium/high demand during arithmetic and Stroop; propose one pooled problem with task metadata if compatible | 40 | 8 / 250 | CC-BY-4.0 original Mendeley record verified |
| [ErpCore2021_N170](https://moabb.neurotechx.com/docs/generated/moabb.datasets.ErpCore2021_N170.html) / FaceStimuli | Original face/car/scrambled event labels, exact retained contrast to verify | 40 | 30 / 1024 | Retain original ERP CORE terms discrepancy documented for WordRelatedness |
| [ErpCore2021_MMN](https://moabb.neurotechx.com/docs/generated/moabb.datasets.ErpCore2021_MMN.html) / AuditoryDeviance | Standard versus deviant auditory stimuli | 40 | 30 / 1024 | Original ERP CORE terms review required |
| [ErpCore2021_N2pc](https://moabb.neurotechx.com/docs/generated/moabb.datasets.ErpCore2021_N2pc.html) / SpatialAttention | Left versus right visual target | 40 | 30 / 1024 | Original ERP CORE terms review required |

The two Barras/Booth OpenNeuro sources explicitly describe19EEG at250Hz; catalogues report24total channels. Export EEG based on native channel types/order, not the total-channel headline. Arithmetic has7question-difficulty levels plus baseline; do not read the title as8nonbaseline levels. N-back has1?4back, distinct from the existing Tremmel joint N-back/audio task. Neither is the existing Zyma arithmetic acquisition. Same-age/demographic lists suggest shared cohorts; investigate participant IDs and waveform overlap before claiming independent studies.

## Hinss / COG-BCI provenance trap

The [15person Neuro2021 competition release](https://zenodo.org/records/5055046) is2.716GB and explicitly CC BY-SA4.0. The [full COG-BCI record6874128](https://zenodo.org/records/6874128) is31.688GB,29people/3sessions and explicitly CC BY4.0. The competition description says it uses one task and half the participants from the larger project. Do not count competition MATB and full COG-BCI MATB as independent recordings. Other full-source N-back/PVT/Flanker recordings may offer further tasks after review; PVT reaction time or KSS scores do not automatically define a categorical vigilance target.

The EEGDash/NEMAR Hinss mirror documents an earlier incorrect500Hz declaration for released250Hz,500sample epochs, corrected in October2026. It still contains conflicting generated acquisition/channel/demographic prose. Inspect source SET dimensions/rates and exact sensor names. Our file `source-evidence/cognitive-affective-review/Hinss2021.json` records the **full COG-BCI** host metadata despite its discovery alias; `Neuro2021.json` records the small competition source.

## Further workload, stress and emotion candidates

### STEW

[MultitaskingLoad](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset): Rest versus SIMKAP multitasking, or one prespecified rating-based contrast; choose one problem. Reported48people, 14EEG at128Hz. Original IEEE DataPort data licence needs verification; paper open access does not establish redistribution. Two2.5min conditions/person; original self-report ratings may be missing. Rest/task is experimental condition, not independently measured subjective high/low workload.

### SAM40

[TaskStress](https://doi.org/10.1016/j.dib.2021.107772): Relaxation and Stroop/arithmetic/mirror-image tasks; recover per-trial stress self-reports before defining stress labels. Reported40people, 32EEG at128Hz. Original data record licence and native labels pending; article licence alone insufficient. Do not assume task identity is independently validated stress. One source collection, not3same-recording relabelled stress problems.

### DREAMER

[FilmAffect](https://zenodo.org/records/546113): Valence/arousal/dominance self-ratings; choose one prespecified classification or joint target. Reported23people, 14EEG at128Hz. Original Zenodo546113 access restricted, no licence grant returned; approval/terms required. Do not use unverified mirror terms; do not count valence/arousal/dominance relabellings as3unique-recording datasets.

### DEAP

[MusicVideoAffect](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html): Music-video affect self-ratings. Reported32people, 32EEG at128Hz. Original signed EULA/access route; research-use and redistribution review required. 128Hz author-preprocessed view versus higher-rate raw; do not count both or multiple same-wave labels separately.

### AMIGOS

[IndividualGroupAffect](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/): Affect during individual/group film viewing. Reported40people, 14EEG at128Hz. Original access agreement and terms to verify; not open-redistribution cleared. Group viewing requires participant/game/film grouping; avoid assuming40complete EEGparticipants without QC.

### OpenNeuro ds005416

[VisualFatigue](https://openneuro.org/datasets/ds005416): Visual fatigue during mixed-reality viewing; recover actual fatigue ratings. Reported23people, unresolvedEEG at1000Hz. CC0 catalogue declaration; original verification pending. Metadata describes24selectedelectrodes versus64storedchannels; source22.9GBmirror. Rest/task event codes alone do not establish fatigue; do not invent a binary label from recording time.

## Conversion and count rules

- Scalp EEG only. Preserve subject strings, sessions, source records, exact ordered named sensors/types, original timing and labels, and direct versus inferred evidence. Original revisions and file checksums must be locked before signal downloads.
- Split whole people and linked groups, retaining trials/blocks together. Never split windows from one trial across TRAIN/TEST. For emotion generalisation, explicitly report whether film identities are shared; subject-generalisation and stimulus-generalisation answer different questions.
- Distinguish imposed task difficulty, self-reported workload, stress ratings, target location and clinical diagnoses. Fix label definitions before algorithm evaluation; no post-hoc thresholds selected on test performance. Retain original continuous ratings even if classification uses documented bins.
- One classification view per recording collection. Valence/arousal relabellings, high/low thresholds or different resampling are not additional unique-recording problems. Native data and author-preprocessed data are not separate datasets.
- Neuro2021/COG overlap, existing Tremmel NBack, Zyma arithmetic, FACED/VideoEmotions and ERP CORE N400 must remain explicit source families. None of the recommendations relax the existing SEED restrictions.
- Pilot before full download; check unit calibration and flat sensors, strict duplicates/overlap and group separation, complete numeric .ts/readback and aligned metadata, then aeon load/fit/predict. Estimate disk usage from pilot; actual converted sizes are unknown.

Original metadata-only evidence is retained under [source-evidence/cognitive-affective-review](source-evidence/cognitive-affective-review/). Machine-readable shortlist: [cognitive-affective-candidates.json](cognitive-affective-candidates.json). Current study assignments: [problem-taxonomy.md](problem-taxonomy.md).

## Measured download sizes,10November2026

GB are decimal. These are input downloads, not peak RAM, extracted bytes or
converted TS sizes. OpenNeuro totals select BIDS EEG binary/header/marker and
EEG metadata files, excluding duplicate source XDF and unrelated physiology.
Neuro2021 and CLATEEG26 sizes are published compressed downloads; ERP CORE
sizes are the per-component BIDS files listed in the source manifest.

| Candidate | Download GB |
|---|---:|
| OpenNeuro ds007262 | 0.397 |
| OpenNeuro ds007169 | 0.442 |
| Neuro2021 / Hinss2021 competition | 2.716 |
| CLATEEG26 | 2.315 |
| ErpCore2021_N170 | 3.394 |
| ErpCore2021_MMN | 3.486 |
| ErpCore2021_N2pc | 3.466 |
| **Total** | **16.217** |

Complete OpenNeuro source folders, including XDF/other modalities, are2.062GB
(arithmetic) and2.291GB(N-back). EEG-only selections are0.397GB and0.442GB.
Full COG-BCI is31.688GB and is not the2.716GB Neuro2021 subset proposed here.

Converted TS sizes require pilot measurements and can greatly exceed compressed
source downloads. Retaining sources, extracted files, intermediates and final
text requires more than the16.217GB download total.

The live ERP manifest SHA256 differs from our existing pinned manifest; the three
component byte totals agree with the pinned cached manifest
`16ffd88374e29f121e7fd77cade429d25704526baac11ede027a484c50af5cff`.
This size survey does not replace any conversion source pin. See the saved
size/file-list evidence under source-evidence/cognitive-affective-review.

## Existing-archive overlap review,10November2026

Checked current registry, selected list, dataset/provenance pages and D:/Data/EEG
folder names. None of the seven candidates is already registered/prepared by
these source identities. This is a source-level comparison, not a completed
waveform-level duplication audit.

| Candidate | Closest existing problem | Finding |
|---|---|---|
| ds007262 graded arithmetic | MentalArithmetic / Zyma EEGmat | Different authors/acquisition and graded-difficulty protocol,18released people/19EEG/250Hz versus36people and arithmetic/rest in existing source. No evidence it is an EEGmat re-export. Native waveform comparison not performed. |
| ds007169 four-level N-back | NBackAudioCondition / Christoph Tremmel | Different documented provider/protocol:18released people,1?4back,19EEG/250Hz versus25original people,0?2back x4audio,16channels/200Hz. Likely distinct, but existing Tremmel acquisition DOI/sessions/sensors remain unknown: independent waveform origin cannot be conclusively established from metadata alone. |
| Neuro2021 | No existing MATB workload preparation | New archive source. It is a subset of full COG-BCI, so do not add full COG MATB as another independent-recording problem. |
| CLATEEG26 | MentalArithmetic / NBackAudioCondition | Original record describes IIUM40person OpenBCI8channel2026release with Stroop/arithmetic demand conditions; no source match to existing records found. Native signal integrity and overlap not yet checked. |
| ERP CORE N170/MMN/N2pc | WordRelatedness / ERP CORE N400 | Same40people/study but three different recording paradigms. Pinned publisher manifest has80native SET/FDT files for each component; none has a matching publisher file MD5 against the80N400 signal files. This establishes no byte-identical distributed signal files in that comparison, not absence of copied subarrays or transformed duplicates. |

Preserve shared-study/participant relationships. Before admission, compare actual
signals as well as file hashes, including cross-source duplicates and transformed
views where source histories suggest a relationship. An accession/name difference
alone is not evidence of recording independence. Seven possible problems does not
mean seven independent cohorts or studies. Evidence:
[source-file comparison](source-evidence/cognitive-affective-review/erp-core-existing-source-comparison.json).
