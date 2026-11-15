# Train file registry

`trainResample0.csv` for Arsenal, DrCIF, DrCIF-500, STC-aeon, TDE, TRACE on every dataset. Only experiments
with train_files = true count. The test registry is one level up.

Built 2026-11-15 17:32 from `D:/Results/EEG` by `tools/results_registry.py build`. Regenerate rather than edit.

Cluster snapshots: hali 2026-11-15 17:31, iridis 2026-11-15 08:33

- Expected cells: 540
- On the desktop: 390 (72.2%)
- Missing: 150

## Missing cells by state

`pull`: done on a cluster, not yet copied. `running`/`pending`: in the Slurm queue.
`submitted`: submitted, not in the queue at snapshot time (finishing or lost; the
runner resolves it next cycle). `waiting`: due for resubmission (e.g. a higher memory
tier). `queued`: in an experiment, not yet submitted. `failed`: the runner gave up
(reason in missing.csv). `unscheduled`: no started experiment on a snapshotted
cluster runs it (Hali experiments count only after `refresh hali`).

| state | cells |
| :--- | ---: |
| running | 71 |
| queued | 35 |
| submitted | 16 |
| unscheduled | 14 |
| pending | 10 |
| waiting | 2 |
| failed | 2 |

## Classifiers with missing cells

| classifier | on desktop | expected | running | pending | submitted | waiting | queued | failed | unscheduled |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRACE | 48 | 90 | 27 | 9 | 0 | 0 | 2 | 2 | 2 |
| DrCIF | 49 | 90 | 0 | 0 | 11 | 1 | 25 | 0 | 4 |
| TDE | 61 | 90 | 23 | 1 | 1 | 0 | 2 | 0 | 2 |
| DrCIF-500 | 66 | 90 | 16 | 0 | 4 | 0 | 2 | 0 | 2 |
| STC-aeon | 81 | 90 | 5 | 0 | 0 | 0 | 2 | 0 | 2 |
| Arsenal | 85 | 90 | 0 | 0 | 0 | 1 | 2 | 0 | 2 |

## Use

```
python tools/results_registry.py refresh iridis   # or hali: snapshot task states
python tools/results_registry.py pull iridis      # refresh, fetch ready cells, rebuild
python tools/results_registry.py build            # offline rebuild
```

`pull` fetches only the files of cells the cluster reports as done and the desktop
lacks: test files for this registry, train files for registry/train. It never changes anything remote and never overwrites a local file.
