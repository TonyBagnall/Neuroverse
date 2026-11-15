# Hali dataset sync and remaining GPU work

11 November 2026. The user authorised syncing dataset lists, pushing missing
datasets, then surveying GPU gaps. No GPU jobs were submitted by this task.

## Dataset sync completed

`/gpfs/home/ajb/DataSetLists/eeg.txt` now matches the canonical76problem list in
this repo. All Hali EEG experiment configurations examined use that list.
The previous file is preserved as
`eeg.txt.bak-20261111-084336` in the same directory.

Added to the list: ChildhoodSeizures, ClinicalSleepStages, FlickerSymbols,
ObsessiveCompulsiveSymptoms and SymbolAttention. Removed from the list: Blink,
EpilepticSeizures and Sleep. Data and old results for excluded problems were
not deleted.

All76dataset directories and their export files were already on Hali. Local
versus remote top-level export names and byte counts matched, so no dataset or
sidecar upload was necessary. This comparison covered the selected `.ts`, JSON,
TSV, TXT, CSV, Markdown, YAML and licence exports, not raw/intermediate caches.
Large arrays were not rehashed on a login node. All88available subject-ID
sidecars (4.37MB total) additionally matched local SHA256 hashes. The existing
84row dataset-info CSV has entries for all76problems, including the correct
200Hz NBack record and metadata for the five newly listed datasets.

Machine-readable evidence:
[file comparison plan](hali-dataset-sync-plan-20261111.json),
[sync and GPU survey](hali-dataset-sync-gpu-survey-20261111.json).

## Existing Hali GPU recipes: 136 missing tasks

These are the23already-used NeuralBench/CTNet models. Their completion checks
use the standard runner's result-file/registry logic over the updated list.

| Experiment | Models | Completed | Newly listed dataset gaps | Existing terminal failures |
|---|---:|---:|---:|---:|
| eeg-ctnet |1|71/76|5|0|
| eeg-eegnet-nb |1|71/76|5|0|
| eeg-nb-deep |3|213/228|15|0|
| eeg-nb-more |12|840/912|60|12|
| eeg-nb-more2 |6|417/456|30|9|
| Total |23|1612/1748|115|21|

The115new-list gaps are all23models on each of the five added problem names.
They were not missing data transfers: those datasets simply were not selected
by the old list. Settled GPU supervisors do not restart themselves merely
because the file changed.

The21existing failures are FBLightConvNet-NB (12) and
SleepStagerChambon2018-NB (9), predominantly temporal input-size incompatibilities.
Do not blindly retry them with the same settings. The removal of excluded
problems reduced the historical25failure count to21within the current list.

The three eeg-nb-deep models are Deep4Net-NB, EEGConformer-NB and
ShallowFBCSPNet-NB. Other exact estimator names, missing dataset names and failure
status are in [the task table](hali-gpu-missing-tasks-20261111.csv).
CTNet-NB is an explicit factory branch, rather than an entry in the generic NB
mapping; it is recognised by the factory.

## New-problem input sizes matter

The local `.ts` headers and existing Hali sampling metadata give:

| Problem | Channels | Native points / Hz | Duration | Approximate points at the NB recipe's120Hz |
|---|---:|---|---:|---:|
| ChildhoodSeizures |17|2000/200|10s|1200|
| ClinicalSleepStages |6|6000/200|30s|3600|
| FlickerSymbols |64|1500/250|6s|720|
| ObsessiveCompulsiveSymptoms |52|2000/500|4s|480|
| SymbolAttention |8|256/256|1s|120|

115is a raw completion gap, not115demonstrated compatible fits. SymbolAttention
at120points is below FBLightConvNet's current250point window and would reproduce
the known short-input failure without a documented model/protocol change.
SleepStagerChambon2018 also needs a constructor/forward-shape check on short
epochs before scheduling. Do not silently pad, lengthen epochs or change model
windows to turn these into apparently equivalent results. Any shape smoke or
model fitting must run in a compute allocation.

## Other GPU candidates, kept separate

| Candidate | Missing under these identities | Current limitation |
|---|---:|---|
| EEGNet, DeepConvNet, EEGConformer in eeg-gpu |228|Older non-NB recipe variants; check whether these are wanted in addition to the existing NB models, rather than count them as three new architectures |
| Eight models in eeg-nb-extra |608|Not registered in the installed tsml-eval NB factory mapping; wrapper/model compatibility also needs checking |
| DGCNN, EEGSym |Not scheduled here|Require exact channel-name/order metadata and a dataset-info channel_names column, which the current three-column CSV lacks |
| CBraMod, Labram pretrained encoders |Not scheduled here|Need source/weights/licence/input-channel review and desktop-side weight provisioning; no matching default HF cache folders were found, which is not a complete search for weights |

The eight draft names are AttnSleep-NB, DeepSleepNet-NB,
SleepStagerBlanco2020-NB, USleep-NB, TCN-NB, HybridNet-NB, MEDFormer-NB and
SeizureTransformer-NB. Do not submit the draft608tasks before factory support
and installed-wrapper compatibility have been established. A static source
survey recovered the factory mapping, but not a definitive installed wrapper
SUPPORTED_MODELS list; that uncertainty is preserved in the JSON evidence.

## Before an authorised GPU restart

Use the standard tsml-eval experiment structure; retain estimator identities,
preprocessing, metadata and output locations. Preflight the new input shapes on
a compute node. Restart the existing five GPU experiment groups only for missing
eligible tasks; leave terminal failures held until the cause is addressed.

Current GPU configs inherit an email address from the Hali profile, and existing
supervisor generation requests failure mail. A future restart must disable both
report and supervisor email under the no-external-calls rule; `email=""` alone
does not remove unconditional supervisor FAIL mail in the unmodified controller.
No installed packages or running code were changed in this sync. Any needed
package/network operation remains Tony's responsibility, or code/weights must
be prepared on the desktop and explicitly pushed without cluster-side downloads.

The list sync itself may be seen by already-active CPU supervisors on their next
cycle. This task did not manually start/stop/retry/cancel any job or alter model
settings, environments or existing results.
