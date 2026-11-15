# tsml-eval farm management: agent reference

Reviewed locally on 10 November 2026, at `C:/Code/tsml-eval` commit
`ce0d1131dca7c2f432ae005162284acfe28ce4be` (`slurm`). The inspected Slurm
directory had no tracked local changes reported by Git. This describes the local
checkout, not a verified inventory of currently running cluster jobs or deployed
remote revisions. No cluster connections or commands were used for this review.

Read [cluster access rules](cluster-access.md) before using this guide. No external
calls from clusters, including pip/git/downloads/telemetry; Tony handles those.
Only his own queues/jobs and file space may be accessed. Heavy work belongs in
Slurm compute allocations, never on login nodes. This document grants no new
permission to submit, cancel, move files or change environments.

## Entry points and configuration

The standard implementation is `_tsml_research_resources/slurm/` in tsml-eval;
start with its [README](C:/Code/tsml-eval/_tsml_research_resources/slurm/README.md)
and [controller](C:/Code/tsml-eval/_tsml_research_resources/slurm/tsml_slurm.py).
Use this system for new work rather than resurrecting the older ad hoc batch
controllers that Claude's notes say were retired from several branches.

| File | Purpose |
|---|---|
| `tsml_slurm.py` | Main controller; standard-library Python 3.11+ |
| `clusters/{iridis6,iridisx,hali}.toml` | Cluster paths, scheduler resources, environments and supervisor defaults |
| `experiments/*.toml` | Dataset lists, estimators, splits, output paths and experiment-specific overrides |
| `farms.py` | Read-only overview of active task farms and their worker progress |
| `registry.py` | Build/merge the `completed.txt` results registry |
| `run_local.py`, `clusters/hydra.toml` | Separate workstation executor without Slurm |
| `pull_results.sh`, `push_data.sh` | Transfer helpers; pushing creates remote paths/files |
| `tests/test_tsml_slurm.py`, `tests/test_run_local.py` | Offline tests with fake scheduler/local execution fixtures |

Relevant existing guides: `README.md`, `REMOTE_ACCESS_NOTES.md`, `HYDRA_NOTES.md`.
Treat examples involving pip, git network operations or email as subject to Tony's
current restrictions, not as permission to execute them.

An experiment combines a cluster profile with its own `[experiment]`,
`[estimators]`, optional `[estimator_kwargs.<name>]`, `[slurm]` and `[supervisor]`
tables. Paths support `{home}`, `{user}`, `{repo}` and `{name}`. `[slurm]` overrides
profile defaults, so inspect the actual experiment before quoting resource limits.
`repo_dir` can select a separate CPU/GPU checkout and `branch` can constrain it.

## What one unit of work means

Each task is `Estimator|Dataset|Resample`. The desired workload is the Cartesian
product of estimators, dataset names and resample indices, minus exclusions and
tasks lacking required per-dataset metadata. A count of one resample means index0
and normally uses the archive's supplied TRAIN/TEST split.

`dataset_info` provides CSV values such as sampling frequency or training-group
file paths. Placeholders such as `{dataset.sfreq}` and `{dataset.train_groups}`
are substituted per estimator/dataset. Missing required values exclude those
tasks and are reported by `check`; do not interpret them as completed runs.

With `cv_folds`, tasks represent folds instead of ordinary resamples. Subject
grouping depends on `<Dataset>_id_TRAIN.txt` and `_id_TEST.txt` beside the data.
For requested ten-fold CV, up to15subjects gives LOSO; larger cohorts are grouped
using rounded subjects/10, with the remainder in the final fold. The actual fold
count can differ from ten. Without ID files it uses ordinary case-level CV, so
verify sidecars before interpreting an EEG result as subject-independent.

Results normally live at:

```text
<results_dir>/<Category>/<Estimator>/Predictions/<Dataset>/testResample<N>.csv
```

The category is omitted for a flat estimator list. Classification normally needs
the test file; `train_files=true` requires train predictions too. Clustering uses
train assignments as its primary output, plus test when requested.

Completion means every required file is non-empty **or** listed by relative path
in `<results_dir>/completed.txt`. This is an operational completion check, not a
full result-content or scientific-validity audit. `registry.py` scans the master
results copy and monotonically merges entries, allowing space-constrained workers
to retain completion knowledge after results have been moved elsewhere. Never
remove cluster files merely because a registry exists without verifying the
master copy and obtaining authority for the removal.

## Controller cycle and durable state

One `submit` cycle:

1. Reads configuration, dataset list, task settings and existing results/registry.
2. Reads the current user's queue using `squeue --user=...`.
3. Reconciles recorded job IDs with live jobs and accounting/logs for ended jobs.
4. Chooses eligible missing tasks within capacity and groups them into arrays or
   farms; optionally tops up existing farms.
5. Submits through `sbatch`, saves state atomically, and writes a report. Email
   may also be sent by default; see the restrictions below.

The state directory defaults to `~/tsml-slurm/<experiment-name>/`, preserving an
existing legacy `~/.tsml-slurm/<name>/` when applicable. It holds `state.json`,
`report.txt`, submitted scripts, per-task logs/status, farm inboxes and supervisor
logs. State includes attempts, memory tiers, job IDs, configuration fingerprints
and failure history. Keep it: deleting it loses ownership/retry information.

Unix advisory locks prevent simultaneous controller cycles for the same state
directory. They are not a distributed lock across independent experiment state
directories or separate cluster filesystems. Split workloads deliberately so two
experiments do not submit the same estimator/dataset/resample to shared outputs.
Iridis6 and IridisX share home storage according to the existing access notes;
Hali is separate. Recheck paths before assuming two outputs are independent.

## Arrays and Iridis6 task farms

**Array mode** groups tasks by estimator, dataset and memory tier. Resample/fold
indices become array indices. `max_active` limits active tasks for this
experiment; optional `max_user_tasks` also considers the user's partition load.

**Farm mode** reserves one node allocation and runs a Bash pool of independent
Python experiment workers inside it. GPU farm mode is rejected; use arrays.
At a memory tier the parallel slot count is:

```text
min(node_cpus // n_jobs, node_memory_gb // memory_per_task_gb, tasks_available)
```

The allocation requests `slots*n_jobs` CPUs and `slots*memory_per_task_gb` RAM.
The initial task list contains up to `slots*waves` tasks, running at most `slots`
at once. A worker's tier is a packing budget; this code does not create a separate
Slurm allocation for each worker inside the farm. Each task has `.out`, `.err`
and `.status` files recording its start and exit. A missing start is distinct
from a worker that started and failed.

With `top_up=true`, a farm that has started its initial list can claim additional
task files from `farms/inbox/<job-id>/`. The controller fills only free slots,
with tasks whose memory tier fits. The farm closes its inbox when too little
wall time remains (`top_up_hours`) or nothing is running. Late arrivals receive
only the allocation's remaining time, so this threshold matters for long fits.

Defaults in the inspected profiles, before experiment overrides:

| Profile | Work execution | Supervisor |
|---|---|---|
| Iridis6 `cpu` | Batch farms;192CPU/630GiB packing model, up to4farm jobs per experiment, four waves;4–620GB tiers;60h | Small Slurm job every15minutes |
| Iridis6 `cpu-array` | Batch arrays;8–64GB tiers, max100active;60h | Same Slurm supervisor |
| IridisX `gpu` | H200 arrays;32/64/128GB, two CPUs, max8active;60h. A100/swarm profiles also exist | Lightweight login-node loop every30minutes |
| Hali `cpu` | Compute arrays;8–128GB, max2000active;7days | Small Slurm job every15minutes |
| Hali `gpu` | GPU arrays;64/128GB, two CPUs, max16active;60h | Same Slurm supervisor |

These are configuration values, not verified current allocation entitlements.
In particular `max_jobs` is per experiment, not a global cap across all farms.

## Failures, checkpoints and reproducibility

Accounting (`sacct` restricted to recorded job IDs), exit status and logs determine
failure type. Host-RAM OOM advances to the next memory tier without spending a
normal retry; the highest tier becomes terminal. Never-started tasks, node failure
and preemption do not consume normal retries. Other failures, including GPU OOM,
use `max_attempts` (default2). A timeout is terminal by default; checkpointed work
can set `retry_timeouts=true`, still bounded by `max_attempts`. `-ch` enables
supported tsml-eval checkpointing; it is not universal model support.

Per-estimator fingerprints prevent changing parameters/flags/paths in place and
mixing incompatible results. Adding tasks is supported, but a changed scientific
configuration needs an appropriate new result identity. `require_clean=true`
checks the checkout, while `pin_commit=true` additionally fixes the experiment's
commit. Pinning is not the default. Job logs record branch/commit, interpreter,
aeon source/version and execution host. Thread environment variables are capped;
CPU workers hide GPUs and GPU workers perform the configured framework check.

## Commands and their actual effects

Commands below describe the interface, not instructions to execute it now.

| Command | Effect / caution |
|---|---|
| `python farms.py` | Reads current user's queue, farm state and logs; no submissions or state writes in inspected code |
| Read `report.txt`, `state.json`, own task logs | Strict inspection; reports may be stale |
| `tsml_slurm.py status EXP.toml` | No submissions, but reconciles failures and writes state/report/lock files; not strictly read-only |
| `status --running --progress` | Also reads worker progress and checkpoint age; still uses the updating status cycle |
| `status --email` | Explicitly sends email; incompatible with no external calls |
| `check EXP.toml` | Validates paths/settings and constructs estimators; can be nontrivial, and is not a guarantee of network-free constructors |
| `submit EXP.toml --dry-run` | Plans and prints a script without submitting or saving cycle state; the surrounding lock can still create a lock file |
| `start EXP.toml` | Removes STOP, submits work and starts/arms a supervisor |
| `submit EXP.toml` | One submitting cycle, potentially sending email |
| `retry EXP.toml` | Clears terminal failures/attempts, so the next supervisor cycle may submit them automatically |
| `cancel ...` | Cancels matching array tasks or writes farm cancellation markers; running farm process groups are terminated |
| `stop EXP.toml` | Creates STOP and cancels supervisors; worker jobs finish normally |
| `stop EXP.toml --cancel` | Also cancels this experiment's recorded live jobs |

Existing access notes require asking Tony before remote mutations unless already
explicitly authorised. A request for strictly read-only inspection should use
saved reports/logs, `farms.py` and user-filtered scheduler reads rather than the
state-updating `status` command. Heavy checks/fits must use compute allocations.

## Existing EEG experiment examples

- `iridis6_eeg_aeon.toml`: current CPU farm for QUANT, RDST, Riemannian/CSP/XDAWN
  and tabular models. Overrides to one farm,16GB initial tier and top-up until
  36hours remain. Its comments warn not to start overlapping `eeg-cpu` work.
- `iridis6_eeg_mrhydra.toml`: separate MRHydra configuration with larger-memory
  handling; inspect it rather than assuming the generic farm profile suffices.
- `iridis6_eeg_hc2.toml` and `_hc2_components.toml`: full ensemble and components.
- `iridis6_eeg_aeon_cv.toml`: separate EEG-CV output tree and subject-aware CV
  where ID files exist.
- `iridisx_eeg_aeon_gpu.toml`: CNN, IT and LITETime-MV; overrides max-active to20.
- `hali_eeg_nb_more.toml` / `iridisx_eeg_nb_more.toml`: disjoint model families
  across clusters to avoid repeated fits; dataset-specific metadata is required.

The older comments mention obsolete problem totals. Use current dataset lists
and the archive registry instead. This review did not assess live job progress.

## Restrictions and discrepancies to carry forward

**Automatic email conflicts with the no-external-calls rule.** The controller uses
`mail`, `mailx` or `sendmail`. The Slurm supervisor script unconditionally requests
`--mail-type=FAIL`, adding `--mail-user` when an address is configured. `--no-email`
on one submit does not remove email from a later supervisor chain. `email=""`
disables controller-addressed reports but does not by itself remove supervisor
FAIL mail: scheduler defaults may still supply a recipient. Future authorised
runs must disable both mechanisms and inspect generated scripts/setup/model code
for other external activity. Existing jobs/configurations were not changed in
this review. The runner does not enforce a general network sandbox itself.

**Hali documentation is partly stale.** Current `clusters/hali.toml` uses a Slurm
supervisor, while README tables/paragraphs still describe a login-node screen
loop. Trust the inspected deployed profile after verifying its revision. IridisX
still has a lightweight login supervisor; this is scheduler control, not training.
Do not run substantial preprocessing, audits or fits there.

**Setup hooks are executable code.** Profile/experiment `setup` commands and
estimator construction may do more than scheduling. Inspect them before a run;
no pip/git/curl/download step is allowed from a cluster under Tony's rule.

**Hydra is separate.** `run_local.py` shares CPU/RAM/GPU budgets across local
experiments and records `local.json`, rather than submitting Slurm jobs. It is an
office workstation, not a cluster login-node exception. Old Hydra comments quote
different machine capacities; use an authorised machine-specific check before
sizing workloads.

The local source was read and cross-checked against its profiles/examples. No
runner code was modified and no scheduler, model fit or remote test was executed.
