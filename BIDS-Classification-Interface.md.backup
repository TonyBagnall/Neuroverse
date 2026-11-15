# Suggestions for an interactive BIDS EEG classification converter

This was collated by codex that has been doing a lot of conversion for me.

## What the tool should help someone do

Build an interface that lets a researcher turn a recording collection into a
**clearly defined, reproducible classification problem**. The valuable part is
helping them make and inspect the scientific decisions: what a case represents,
where its label comes from, which samples and sensors it contains, and what the
test set is meant to measure.

BIDS organises recordings and metadata. It does not, by itself, determine the
correct prediction task or guarantee independent training and test examples.
Treat structural validation, scientific decisions and numerical validation as
separate parts of the workflow.

The main deliverables should be a saved conversion recipe, classification data,
case-level metadata, and an understandable audit report. A successful file export
alone is not a sufficient definition of success.

## A realistic first version

Start with **local BIDS EEG, event-based epochs, fixed-length cases, EEG channel
selection, an explicit label mapping, and a subject-disjoint train/test split**.
Support one well-tested recording format first, then a second. For example, EDF
and BrainVision would exercise different file structures and calibration paths.

The first release should include duplicate detection, an exclusion log, a pilot
conversion and a reproducible export. These are more useful than supporting many
file formats superficially.

Later additions could support participant-level labels with continuous windows,
sleep-stage annotations, variable-length cases, online evaluation protocols,
channel-layout harmonisation and optional preprocessing. Keep automatic task
inference and a full model-training dashboard outside the initial scope.

## Suggested interaction: five stages

| Stage | What the user sees | What must be saved |
|---|---|---|
| 1. Inspect the source | Subjects, sessions, tasks, recordings, events, channels and metadata problems | Source inventory and exact version/file identities |
| 2. Define the problem | Eligible events, label meanings, excluded conditions and resulting class counts | Inclusion rules and original-to-output label mapping |
| 3. Define each case | A waveform and event timeline with the proposed crop, sensors and units | Window, channel and processing rules |
| 4. Choose and audit the split | Subject/session memberships, counts, duplicates and overlapping windows | Exact memberships and decisions about flagged cases |
| 5. Pilot and export | Pilot report, resource estimate, full progress and final validation | Recipe, data, sidecars, exclusions and manifest |

Let users move back between stages. If a decision changes, mark dependent results
as stale: changing the window invalidates case statistics and duplicate checks;
changing eligibility invalidates class counts and potentially the split.

The final review should be concrete, for example:

> Predict imagined left versus right hand from four seconds after each eligible
> cue, using the selected EEG channels. Evaluate on participants absent from
> training. Preserve native sampling and apply no additional filtering.

Show actual participant memberships and counts beneath that sentence. Do not
generate reassuring text that is inconsistent with the exported recipe.

## 1. Inspect the source before reading all the signals

Begin with filenames, headers and small metadata files. Present a recording table
with subject, session, task, acquisition/run entities, file format, duration,
sampling frequency, channel count and event availability. Keep identifiers as
strings: `01` and `1` must not collapse accidentally.

Count the recordings actually present, and compare that count with the source
description. Do not discard a run solely because its number exceeds a number in
the README. Treat multi-file recordings as bundles: for example, a BrainVision
header, binary signal and marker file must remain associated. Distinguish missing
payloads or unresolved data-storage links from recordings that genuinely do not
exist, and explain what must be materialised before conversion.

Resolve metadata using BIDS rules rather than assuming every recording has a
neighbouring JSON or TSV with a matching filename. Inherited metadata can apply
across recordings. Show the file that supplied each resolved value, including
overrides. See the official [BIDS inheritance rules](https://bids-specification.readthedocs.io/en/stable/common-principles.html#the-inheritance-principle).

Useful metadata sources include `dataset_description.json`, the README,
`participants.tsv`, applicable EEG JSON, channel tables, event tables and their
column descriptions, and electrode/coordinate files where present. The
[BIDS EEG specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html)
distinguishes channels from electrode geometry; the interface should too.

Compare resolved metadata with the actual recording header. A disagreement about
sampling frequency, channel order or physical units should be visible and linked
to both sources. Do not silently choose whichever value is easiest to read.

Provide an evidence status for important fields:

| Evidence status | Example |
|---|---|
| Recovered from files | Ordered channel labels in a native header |
| Documented by the source | Acquisition reference described in a paper |
| Inferred | A likely unit interpretation supported by a secondary loader |
| Assigned by this conversion | A new train/test membership or encoded class |
| Unknown or conflicting | Different licence declarations in README and JSON |

Missing sessions need careful treatment. An absent `ses` entity does not establish
the participant's complete visit history. Preserve the original absence; if the
software needs an internal session key, label that key as assigned.

## 2. Make label construction explicit

Offer distinct task types, initially implementing only the event-based option:

- **Event-based:** one example around an eligible cue or annotation, with a label
  selected from an event-table column or a documented mapping.
- **Participant-based:** windows inherit a documented participant attribute,
  such as a diagnostic group. Many windows still represent one participant.
- **Recording-based:** labels describe a recording condition or session, with
  the relationship between conditions and people shown explicitly.

For events, show the actual column values, their descriptions, missing values and
counts by subject and recording. `trial_type` is optional in BIDS; do not require
it or assume it contains the intended target. Events may also represent responses,
feedback and blocks, and can overlap. These distinctions are described in the
[BIDS events specification](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html).

A mapping editor should show something like:

| Original value | Source description | User interpretation | Output class | Include? |
|---|---|---|---|---|
| `left` | Left-hand imagery cue | Imagined left hand | `1` | Yes |
| `right` | Right-hand imagery cue | Imagined right hand | `2` | Yes |
| `feedback` | Classification feedback | Feedback event, not another trial | — | No |

Retain original values and descriptions alongside the encoded label. Unknown
values should require an explicit mapping or exclusion; they must not silently
become another class. Participant metadata joins need uniqueness checks and a
clear report of unmatched IDs.

Show whether labels are confounded with subject, run, device or site. If every
participant has only one class, a random split of their windows can measure
participant recognition rather than generalisation to new people.

## 3. Preview exactly which samples become a case

Draw the signal, cue and proposed window together. Let users inspect different
subjects and classes, including the first and last eligible events in a recording.
The preview should expose excluded and incomplete examples, not just attractive
examples selected by the application.

Important controls are:

- The anchoring event and any start offset, including a possible negative offset.
- Fixed duration versus a documented variable-duration interval.
- Eligibility filters, such as excluding calibration or executed movement.
- Treatment of annotations, discontinuities and incomplete windows.

Use an explicit sample convention: **start inclusive, end exclusive**. At 250 Hz,
a four-second example has 1,000 samples; its last sample is at 3.996 seconds.
Check the chosen library's endpoint behaviour rather than assuming it matches
this convention. Save the requested times, actual sample indices, rounding rule
and resulting sample count.

BIDS event onsets are relative to the first stored data point. The importer must
also account for the recording reader's time conventions. Preserve the original
event row index and onset even if the application creates another event ID.

An event's duration is not automatically the desired classification window. A
zero-duration cue can anchor a four-second trial. Conversely, a 20-second event
near a recording's end may not have 20 seconds of available EEG.

Default to excluding incomplete windows with a recorded reason. Do not silently
pad, shorten all trials, cross a recording boundary or join disconnected epochs.
If variable-length export is supported later, make that a distinct decision.

Separate original continuous sample indices from indices inside an already
epoched array. A local index of zero in an epoch is not evidence of where that
epoch began in the original acquisition. Never invent that missing history.

## 4. Treat channels and units as first-class decisions

The channel screen should display **array index, exact name, type, unit, source
bad-channel flag and available geometry**. Distinguish EEG from EOG, ECG, EMG,
trigger and auxiliary inputs. A recording containing EEG is not necessarily an
array containing only EEG.

Check channel membership and order across every selected recording. If they
differ, offer a documented choice: restrict the source cohort, select a verified
common set, or stop for review. Reordering by verified names can be legitimate,
but save the source index mapping for every recording. Equal channel counts do
not establish equal layouts.

Preserve spelling and case from the source. Do not automatically normalise names
or invent a standard montage. If anatomical names are unavailable, opaque column
IDs can support inspection, but must be labelled as such. Require acknowledgement
before an export with unresolved sensor identity, and disable automatic anatomical
harmonisation for those columns.

Keep electrode coordinates with their units, coordinate frame and provenance.
A standard template and individually measured positions should remain distinct.
Keep the electrical reference separate from the bibliographic paper reference in
the metadata schema; using one field named `reference` for both caused ambiguity
in our preparation tools.

For units, show three things separately: the declared source unit, the unit
returned by the reader, and the output unit. Inspect reader calibration and
scaling behaviour. A reader may return calibrated SI values even when the file
declares microvolts; multiplying again without checking can introduce a factor
of a million.

Plot robust amplitude and variability summaries by channel and recording. Tiny
values can suggest volts, but amplitude alone is not proof. Never automatically
multiply values by a million just because they are small. Require evidence or an
explicitly recorded inference. Cast to float64 before any required calibration or
rescaling; this avoids further rounding but cannot restore precision already lost
in the source.

## 5. Design the split around the evaluation question

The interface should explain the implications of each split:

| Split | What it evaluates | Main concern |
|---|---|---|
| By participant | Generalisation to unseen people | Keep all windows and visits of each person together |
| By session within participant | Transfer to a later/different visit | Shared people are intentional and must be reported |
| By recording/run | Transfer between runs | People and acquisition conditions may still overlap |
| Original online protocol | The source's calibration-to-test setting | Preserve chronology and source phase definitions |
| Random windows | A narrower within-recording setting, when justified | Overlap, temporal dependence and identity leakage |

Recommend a participant split for an unseen-person question. Do not make random
window splitting the convenient default. Check that group membership and class
availability permit the requested split; a stratification option cannot guarantee
a valid split for every dataset.

Preserve a documented original partition when that is the intended benchmark.
For our archive, pool original training and validation into TRAIN, preserve TEST,
and retain each case's original partition in its sidecar. Do not export a separate
validation partition in this compatibility mode. Classifier tuning and any
validation inside TRAIN remain the fitted classifier's responsibility.

Keep original trial groups together when producing multiple windows per trial.
Detect shared samples and overlapping intervals across partitions, not just
identical arrays. Only compare continuous intervals when the original recording
identity and sample coordinate system are established.

Save exact memberships, source order and any random seed. Prefer deterministic
enumeration by BIDS entities plus original event row, with no hidden shuffle.
If chronology matters, use documented time/run information; lexical filename order
is not automatically acquisition order.

## 6. Give quality checks their own review screen

Use three clear states: **must resolve**, **needs an explicit decision**, and
**information**. Every flag should link to the affected recordings or cases and
explain its consequence. A global warning banner without a way to find the data
is not very useful.

### Duplicates

Check both original recording files and constructed cases. Whole-file hashes can
detect copied EDFs; case hashes catch repeated signal content even when containers
or headers differ. Define a canonical numeric representation for case hashing,
including array shape and byte order. Keep the label out of the signal digest so
identical signals with conflicting labels are detected.

Compare channel identities, units and processing when interpreting a match.
Compute checks before and after transformations that could introduce copies.
Classify matches as within-recording, cross-recording, cross-subject, cross-split
and conflicting-label cases. Near-match/correlation searches are useful later,
but are separate from exact equality and need human review.

Never keep the first copy automatically. If an identical recording is assigned
to two people, retaining either attribution can be scientifically wrong. Offer
quarantining both ambiguous copies, preserving source evidence and retaining
unaffected recordings. Conflicting labels should block export until resolved or
explicitly excluded.

### Flat, tiny and noisy channels

Show constant counts, standard deviations and affected case fractions by sensor.
Distinguish exact constants, effectively flat signals, source bad-channel flags,
large-amplitude noise and unknown units. Large signals are not flat signals.

For EEGBench compatibility, our policy removes a sensor globally if any retained
case has standard deviation at or below `1e-7` in microvolt-valued data. This is
an archive policy, **not a universal EEG cleaning rule**. Make the threshold, unit,
scope and numerical calculation explicit. Offer the researcher a review of
annotating, excluding affected cases or globally excluding a sensor. Recompute
shapes and quality checks after their decision. Removing every sensor must stop
export, not produce an empty classification problem.

### Transformations

Default to no additional filtering, baseline subtraction, rereferencing,
normalisation, interpolation or resampling. Distinguish acquisition/distributed
preprocessing from operations added by this tool. If processing is enabled,
record its order, parameters and software version.

Data-dependent transformations must respect the split. In particular, do not fit
normalisation or feature selection on TEST. Filtering across artificial joins
between epochs can also change signals in ways the user did not intend.

## 7. Use lessons from actual conversions as acceptance scenarios

| Experience in this repository | What the interface should demonstrate |
|---|---|
| Daly / ds002720: subjects09 and15 had identical run-10 EDFs and event tables | Detect copies across participant IDs; show source hashes; exclude both ambiguous recordings without deleting either person's other data |
| Daly: some final 20-second events exceeded the recording end | Show available versus requested samples; log incomplete exclusions rather than padding |
| Pan2025: session2 starts with30 training trials, followed by online test trials | Support the original phase split and explain intentional shared participants |
| MatchWords: scale needed review and a factor-of-a-million conversion | Display declared/reader/output units and save the evidence for rescaling |
| VideoEmotions: F8 was constant in affected cases | Identify both channel name and zero-based index; preview the effect of global sensor removal |
| Some native MAT sources lacked anatomical channel names | Preserve uncertainty instead of manufacturing standard electrode labels |
| A restarted worker left an empty NPZ intermediate | Recover safely and publish caches atomically; never treat file existence as proof of completion |

The detailed [Daly investigation](docs/daly-duplication-investigation.md) is a good
worked example. Its cleanup excluded both ambiguous run-10 copies, retained both
participants' other recordings, and reran the strict duplicate checks. It did not
choose an owner by filename order.

## 8. Export data that another person can audit

For equal-length multivariate classification, use numeric arrays with shape
`(cases, channels, timepoints)` and one label per case. For this archive, export
the standard aeon-compatible `.ts` TRAIN/TEST files with these sidecars:

```text
ProblemName/
  ProblemName_TRAIN.ts
  ProblemName_TEST.ts
  cases_train.tsv
  cases_test.tsv
  channels.tsv
  metadata.json
  manifest.json
  exclusions.tsv
  recipe.json
  validation.json
  README.md
```

This is a proposed product contract; an EEGBench export profile should match the
existing filenames and metadata conventions where they differ.

Each case row should include its output index and stable ID; participant,
session, task and run identifiers; source file and event/epoch row; original and
encoded labels; original partition; source sample bounds and their coordinate
system; output split; selected source channel indices; and a signal digest.
Preserve unknown values explicitly. Metadata row order must exactly match array
order, and every excluded candidate needs a reason and original identity.

Dataset metadata should include ordered sensors and units, available geometry,
the source DOI/version and actual data licence, label definitions, counts,
sampling rates, window/overlap rules, processing history, evidence status and
unresolved limitations. Dataset terms can differ from a loader's software licence;
show conflicting declarations rather than assigning a blanket licence.

The manifest should pin source file identities, the recipe, software versions,
split memberships, exclusions and output checksums. This lets another agent
compare recording origins without loading every array. Dataset names and renamed
labels are insufficient for that comparison.

Keep a compact intermediate representation, such as per-recording compressed NPZ
with numeric arrays and separately typed metadata. Avoid pickle-dependent object
arrays. Document how intermediates map to cases and whether they are disposable.

## 9. Keep the interface separate from the conversion engine

A practical Python design is a small headless package with these components:

```text
inventory -> resolved metadata -> case plan -> split/audit -> materialise -> export
                                    ^                         |
                                    |                         v
                              saved recipe             validation report
```

The case plan can be a table of candidate cases and source bounds before large
arrays are written. The interface displays and edits decisions; the engine
executes the same saved recipe from a command line. Avoid separate implementations
of the conversion in UI callbacks and batch scripts.

A small illustrative recipe might look like this; the team should design and
validate its own complete schema rather than treating these fields as an API:

```yaml
schema_version: 1
source:
  dataset_id: example-study
  revision: immutable-release-identifier
case_definition:
  kind: event_locked
  label_column: trial_type
  label_mapping: {left: "1", right: "2"}
  start_offset_seconds: 0
  duration_seconds: 4
  incomplete_window: exclude_and_log
channels:
  types: [EEG]
  order_policy: verify_native_order_across_recordings
split:
  unit: participant
  train: ["01", "02"]
  test: ["03"]
processing:
  additional_operations: []
export:
  profile: eegbench_ts
```

The complete saved recipe also needs the verified unit decision, source hashes,
all resolved defaults and reviewed exclusions. Saving only the fields a user
changed will not reproduce a conversion after application defaults change.

Use established readers rather than writing an EDF or BrainVision decoder for
this project. [MNE-BIDS `read_raw_bids`](https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html)
can load recordings with associated events and channel information. Its documented
channel mismatch handling is relevant: mismatches should remain visible.
[PyBIDS](https://bids-standard.github.io/pybids/layout/index.html) is another useful
option for querying the dataset inventory. These libraries reduce parsing work;
they do not resolve ambiguous scientific labels for the user. Pin and test the
versions used by the project.

Choose a familiar UI framework after the engine works. A local browser interface
is a sensible first deployment: it can inspect local files without requiring
signal uploads. Long conversions need a background job, progress, cancellation
and a persistent log; a browser refresh should not restart the conversion.

Process recordings or batches rather than concatenating the entire dataset in
RAM. Display downsampled traces if needed for responsiveness, clearly separating
display reduction from exported data. Estimate dense array memory, temporary
storage and text-export size from the pilot. Compressed source size is a poor
estimate of `.ts` size.

Write intermediates and outputs to temporary paths, close and validate them,
then rename them into place. Cache identity should depend on source content and
the applicable recipe/version. Check cached shapes, types and integrity before
reuse; handle zero-byte and truncated caches. Preserve completed destinations
unless the user explicitly selects replacement. Source recordings should be
read-only throughout conversion.

## 10. Pilot, validate, and evaluate the interface

Before bulk export, select a small pilot that exercises both partitions and
important variants: different layouts, sessions, classes or source formats. A
pilot containing only easy examples or only training participants is misleading.
Keep pilot and full caches separate or key them unambiguously.

Full validation should check:

1. Shapes, finite values, class counts and case/metadata alignment.
2. Original label-to-encoded-label correspondence for every case.
3. Exact channel selection/order and physical scaling for every recording.
4. Group memberships, original partition rules and overlapping intervals.
5. Exact duplicates within and across partitions, including conflicting labels.
6. Constant/effectively flat sensors and all documented exclusions.
7. Output readback against the intended numeric representation and labels.
8. Loading and fitting a small aeon classifier, with the smoke-test scope stated.

A successful classifier fit is a compatibility check, not proof that the task is
scientifically valid. For lossless exports, compare every case after readback;
if an explicitly lossy format is supported, state the tolerance and quantify the
error. Do not claim exact recovery of precision that was absent in the source.

Use synthetic fixtures for difficult edge cases: inherited metadata, string IDs
with leading zeros, missing or overlapping events, endpoint rounding, non-integer
sampling rates, mixed sensor types, unit disagreements, channel-order changes,
an ambiguous cross-subject copy, and interrupted cache writes. Use small approved
real examples for format integration. Test the headless recipe and UI together:
the same recipe should produce the same cases, labels, ordering and numerical
digests through either route.

Finally, ask someone outside the team to define and export a problem without
editing Python. Can they explain the labels, trace one output case to its source,
understand why an event was excluded, and recreate the result from the saved
recipe? Those are stronger project outcomes than the number of interface widgets.

## Suggested milestones and division of work

| Milestone | Deliverable |
|---|---|
| 1. Inspect | A reliable inventory with source-linked metadata and signal preview |
| 2. Plan | Saved eligibility, label and window decisions; candidate-case table |
| 3. Audit | Grouped split, unit/channel checks, duplicate report and exclusion decisions |
| 4. Export | Pilot, full `.ts` export, aligned sidecars, deterministic rerun and readback |
| 5. Evaluate | Edge-case tests, usability session and documented limitations |

Possible work streams are the reader/inventory layer, the case/split/audit engine,
and the interface/reporting layer. Agree on the recipe and case-table schemas
early, so these components can be developed independently. Put a working vertical
slice through all layers ahead of adding more source formats.

Relevant examples in this repository include
[`tools/prepare_open_moabb.py`](tools/prepare_open_moabb.py),
[`tools/mi_to_30_native.py`](tools/mi_to_30_native.py),
[`tools/update_dataset_sidecars.py`](tools/update_dataset_sidecars.py), and the
[conversion handover](DATASET_CONVERSION_HANDOVER.md). They contain useful
validation and metadata patterns, but also dataset-specific assumptions. Treat
them as experience to learn from, not a generic BIDS converter to wrap unchanged.
