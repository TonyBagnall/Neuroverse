# Hali progress and result retrieval: 11 November 2026

Connected as ajb over the UEA VPN. Read only Tony's own queue, saved EEG reports,
dataset list and prediction paths. No jobs, environments or remote files were
changed, and no external calls originated from Hali. Slurm was absent from the
non-interactive PATH; the normal login environment resolved squeue to
`/opt/slurm/25.11.4/bin/squeue`.

## Current queue

27CPU jobs running, no GPU jobs running, and three pending compute entries.
Of the running jobs,24are EEG pipelines and three are the separate Multiverse
TDE study. Most remaining EEG work is AUG-TANG-SVM-MOABB; there is also a
LogVar-SVM, FBCSP-SVM and two XdawnCov-TS-SVM tasks. Some have run for several
days, within Hali's configured limits. These are live queue observations;
completion totals below come from the saved controller reports.

## Configured EEG batches

The Hali list has74names. These totals describe those configured names, not a
fully synchronised current scalp-EEG study list.

| Batch | Complete | Other state in latest report |
|---|---:|---|
| eeg-nb-deep |222/222|Finished|
| eeg-baselines2 |296/296|Finished|
| eeg-baselines |74/74|Finished|
| eeg-baselines-mi |74/74|Finished|
| eeg-eegnet-nb |74/74|Finished|
| eeg-moabb-mi |148/148|Finished|
| eeg-ctnet |74/74|Finished|
| eeg-nb-more |873/888|15terminal failures; no remaining queued work|
| eeg-nb-more2 |434/444|10terminal failures; no remaining queued work|
| eeg-filtered |576/592|16terminal failures|
| eeg-moabb-mi-more |685/740|22running,33terminal failures|
| eeg-moabb-erp |293/296|2running,1terminal failure|

Some legacy eeg-cpu/eeg-gpu reports show zero completion and all tasks unsubmitted;
they are not evidence that the active replacement batches have stopped.

Examples of terminal failures:

- FBLightConvNet-NB: input shorter than its temporal window.
- SleepStagerChambon2018-NB: temporal output becomes too small on short epochs.
- TRCSP-LDA-MOABB: intrinsically binary method applied to multiclass problems.
- Filtered CSP variants: too few channels/rank for six CSP components, or
  unsupported single-channel input.
- xDAWN-LDA-MOABB on CodedSymbols: LAPACK integer overflow for a very large matrix.

These were surveyed, not silently retried or changed. Some involve already
excluded problems, so do not sum them as failures on the current selected study.

## Dataset-list drift

Relative to Iridis's76current names, Hali still includes Blink,
EpilepticSeizures and Sleep, which the user excluded. It lacks
ChildhoodSeizures, ClinicalSleepStages, FlickerSymbols,
ObsessiveCompulsiveSymptoms and SymbolAttention. No dataset list or data was
changed during this request. Review source availability and scope before
synchronising lists, because live controllers reread them.

## Results retrieved

Hali contains4032prediction CSVs totalling516.3MB. The pull fetched806new or
newer files,118.6MB, into `D:/Results/EEG`. All806passed structural, label,
probability and stored-accuracy validation; zero files were quarantined.
1589files already matched, and1637older Hali copies were skipped to preserve
newer desktop files. Comparison uses source file size and modification time,
not a claim that model versions or data partitions are equivalent.

Replaced files are backed up under
`D:/Results/EEG-replaced-20261111-081206`. The transfer receipt is
[hali-results-pull-20261111.json](hali-results-pull-20261111.json).
Competition results remain separate under the EEGCompetitions directory.

Cross-cluster reconciliation also caught57older Iridis files that the earlier
pull had placed over newer desktop copies. The newer versions were restored
from backups; both versions are preserved, with the older Iridis copies under
`D:/Results/EEG-source-variants-20261111/iridis-older-copy`.
See [reconciliation receipt](results-newest-copy-reconciliation-20261111.json).
The transfer helper now skips older remote versions; three local tests cover
CSV validation and preventing older-copy replacement.

Raw own-account snapshots are local ignored files:
`.openeegbench/hali-progress-20261111.json` and
`.openeegbench/hali-queue-20261111.json`.
