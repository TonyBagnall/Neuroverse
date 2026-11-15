# Iridis progress and result retrieval: 11 November 2026

Read only Tony's own queues, saved reports, file inventories and job accounting.
No remote jobs, environments or results were changed. Main archive and competition
outputs were transferred into separate local directories. No checkpoint was
deleted: none of the archive candidates was complete and inactive, and Tony
explicitly retained the competition fitted models.

## Archive snapshot

The current remote dataset list contains76selected problems. Counts below are
for testResample0 files from those names, not obsolete/excluded datasets that
also remain in the results tree.

| Estimator | Completed /76 |
|---|---:|
| CNN |76|
| IT |74|
| Arsenal |73|
| STC-aeon |71|
| DrCIF-500 |70|
| MRHydra |69|
| DrCIF |69|
| TDE |63|
| QUANT |54|
| LITETime-MV |40|

Saved controller reports: HC2 components346/380 (includes standalone DrCIF as a
fifth estimator); CPU aeon822/988; GPU aeon190/228. The reports record10CPU
timeouts,3archive MRHydra failures of unknown cause and an IT timeout on
ClinicalSleepStages. The older NeuroBench report records13SSTDPN-NB short-input
failures and is dated10November23:58; its completion count is stale. The file
inventory shows GCNsNet-NB70/71eligible problems and EEGInceptionERP-NB26/71.
These failures were not retried or modified by this inspection.

Live queue snapshot: Iridis6 five running CPU farms and six pending entries
(including delayed supervisors); IridisX five running and five pending GPU
tasks. Farm allocations do not imply all workers are busy. Hali was not accessed
on the Southampton VPN. Iridis6/X share results storage, so only one results
transfer was needed.

## Files fetched

Remote archive inventory:3021prediction CSVs,525.4MB. Incremental comparison
found1573new or changed files,260.3MB. Every fetched CSV passed header, class,
finite-probability, probability-sum and stored-accuracy checks before merging.
The1448unchanged files were not retransferred or revalidated by this run.

- Main destination: `D:/Results/EEG`.
- Replaced local files preserved under `D:/Results/EEG-replaced-20261111-075444`.
- [Detailed transfer receipt](iridis-results-pull-20261111.json).
- [Selected-problem result counts](iridis-progress-20261111.json).

## Competition HC2: all four completed

| Component | Job | Elapsed |
|---|---:|---:|
| Arsenal |4888747_0|31m08s|
| DrCIF-500 |4888748_0|30m23s|
| STC-aeon |4888749_0|5m52s|
| TDE |4888750_0|5h29m32s|

All four accounting states are COMPLETED with exit0. All eight prediction files
were fetched and validated into
`D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/hc2-cluster-results/results`.
They are not mixed into the main archive results.

The existing tsml-eval `build_hc2_from_files.py` built HC2FromFile locally from
these files. Its outputs are in the competition directory under
`results/Hybrid/HC2FromFile/Predictions/ProteusCommandsAllPublic/`.
The TEST files contain three repeated training cases for diagnostics only;
their scores are not generalisation or competition estimates.

Train-estimate accuracies: Arsenal60.49%, DrCIF-50058.31%, STC54.61%, TDE57.93%.
Standard accuracy-to-power4 normalised weights are approximately29.68%,25.62%,
19.73%,24.97%, respectively. No weights were tuned on held-out labels.
See [offline assembly record](competition-track2-hc2-offline-assembly.json).

Fitted checkpoints total18.192GB (TDE15.923GB, STC2.178GB, DrCIF78.2MB,
Arsenal12.0MB). They remain in the dedicated Iridis competition directory at
Tony's explicit request; they were not downloaded or deleted. This offline
prediction-file assembly is not yet a portable sealed-data submission.

## Checkpoint cleanup

The archive checkpoint inventory found21files totalling807.94GB. None had both
complete train/test result files with no corresponding active task. Some are
active fits or predictions, others incomplete tasks; those remain untouched.
No local archive checkpoints were present under `D:/Results/EEG`.
[Cleanup decision receipt](iridis-checkpoint-cleanup-20261111.json).

MRHydra's Codabench submission was reported failed by Tony. Diagnosis remains
pending the execution error/log; no resubmission was made.
