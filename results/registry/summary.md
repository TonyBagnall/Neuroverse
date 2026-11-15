# Results registry

Built 2026-11-15 17:32 from `D:/Results/EEG` by `tools/results_registry.py build`. Regenerate rather than edit.

Cluster snapshots: hali 2026-11-15 17:31, iridis 2026-11-15 08:33

- Expected cells: 9776
- On the desktop: 7481 (76.5%)
- Missing: 2295

## Missing cells by state

`pull`: done on a cluster, not yet copied. `running`/`pending`: in the Slurm queue.
`submitted`: submitted, not in the queue at snapshot time (finishing or lost; the
runner resolves it next cycle). `waiting`: due for resubmission (e.g. a higher memory
tier). `queued`: in an experiment, not yet submitted. `failed`: the runner gave up
(reason in missing.csv). `unscheduled`: no started experiment on a snapshotted
cluster runs it (Hali experiments count only after `refresh hali`).

| state | cells |
| :--- | ---: |
| queued | 863 |
| unscheduled | 755 |
| pending | 467 |
| running | 127 |
| submitted | 57 |
| failed | 17 |
| waiting | 9 |

## Classifiers with missing cells

| classifier | on desktop | expected | running | pending | submitted | waiting | queued | failed | unscheduled |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MultiRocketHydra | 0 | 88 | 0 | 0 | 0 | 0 | 88 | 0 | 0 |
| ROCKET | 15 | 90 | 0 | 0 | 0 | 0 | 73 | 0 | 2 |
| MiniROCKET | 17 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 73 |
| FreshPRINCE | 21 | 90 | 0 | 67 | 0 | 0 | 0 | 0 | 2 |
| LaBraM | 38 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 52 |
| LaBraM-random | 38 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 52 |
| EEGNet | 0 | 50 | 0 | 0 | 0 | 1 | 49 | 0 | 0 |
| DeepConvNet | 0 | 50 | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| EEGConformer | 0 | 50 | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| LITETime-MV | 42 | 90 | 0 | 0 | 8 | 0 | 38 | 0 | 2 |
| RiemannianMDM | 46 | 90 | 1 | 0 | 0 | 2 | 39 | 0 | 2 |
| CBraMod-random | 46 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 44 |
| CBraMod | 46 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 44 |
| AUG-TANG-SVM-MOABB | 48 | 90 | 15 | 13 | 1 | 0 | 2 | 9 | 2 |
| XDAWN | 48 | 90 | 2 | 0 | 0 | 0 | 38 | 0 | 2 |
| TRACE | 48 | 90 | 27 | 9 | 0 | 0 | 2 | 2 | 2 |
| CSP+SVM | 49 | 90 | 0 | 0 | 0 | 0 | 39 | 0 | 2 |
| CSP+LDA | 49 | 90 | 0 | 0 | 0 | 0 | 39 | 0 | 2 |
| RiemannianKNN | 49 | 90 | 0 | 0 | 0 | 1 | 38 | 0 | 2 |
| SVM | 49 | 90 | 1 | 0 | 0 | 0 | 0 | 0 | 40 |
| DrCIF | 49 | 90 | 0 | 0 | 11 | 1 | 25 | 0 | 4 |
| LogisticRegression | 50 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| RiemannianMDM-OAS | 50 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| RiemannianKNN-OAS | 50 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| RandomForest | 50 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| CNN | 50 | 90 | 0 | 0 | 0 | 0 | 38 | 0 | 2 |
| HC2-500-pc | 52 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 38 |
| TDE | 61 | 90 | 23 | 1 | 1 | 0 | 2 | 0 | 2 |
| TSception-NB | 62 | 90 | 0 | 0 | 13 | 0 | 13 | 0 | 2 |
| HC2-noTDE-500-pc | 62 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
| LaBraM-FT | 63 | 90 | 0 | 5 | 0 | 0 | 2 | 0 | 20 |
| LaBraM-FT-random | 63 | 90 | 0 | 4 | 0 | 0 | 3 | 0 | 20 |
| RotationForest | 63 | 90 | 21 | 0 | 1 | 0 | 2 | 1 | 2 |
| DrCIF-500 | 66 | 90 | 16 | 0 | 4 | 0 | 2 | 0 | 2 |
| IT | 69 | 90 | 4 | 5 | 0 | 0 | 9 | 1 | 2 |
| Catch22 | 74 | 90 | 2 | 0 | 7 | 3 | 2 | 0 | 2 |
| xDAWN-LDA-MOABB | 77 | 90 | 0 | 8 | 0 | 0 | 2 | 1 | 2 |
| LogVar-SVM-MOABB | 77 | 90 | 0 | 8 | 1 | 0 | 2 | 0 | 2 |
| FBCSP-SVM-MOABB | 77 | 90 | 1 | 8 | 0 | 0 | 2 | 0 | 2 |
| EEGNet-NB | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| ERPCov-MDM-MOABB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| CSP+LDA-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| AttentionBaseNet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| BDTCN-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| EEGTCNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| ATCNet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| XdawnCov-MDM-MOABB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| TCFormer-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| SincShallowNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| RiemannianKNN-OAS-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianKNN-LWF | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| MSVTNet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| RiemannianKNN-LWF-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianKNN-OAS-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| MSCFormer-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| EEGConformer-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| EEGInceptionMI-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| EEGInceptionERP-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| EEGSimpleConv-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| EEGNeX-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| FBCNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| FBMSNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| CSP+SVM-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| CSP+SVM-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| CSP-LDA-MOABB | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| EEGITNet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| ContraWR-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| CTNet-NB | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| Deep4Net-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| DLCSP-shLDA-MOABB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| CSP+LDA-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| BrainModule-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| IFNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| RiemannianKNN-LWF-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| GCNsNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| MDM-MOABB | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RDST | 78 | 90 | 8 | 0 | 0 | 0 | 2 | 0 | 2 |
| LogVar-LDA-MOABB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| ShallowFBCSPNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| TSEN-MOABB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| TIDNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| SyncNet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| TMSANet-NB | 78 | 90 | 0 | 8 | 0 | 0 | 2 | 0 | 2 |
| RiemannianMDM-LWF-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianMDM-OAS-BP140 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianMDM-LWF | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianMDM-LWF-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| RiemannianMDM-OAS-BP830 | 78 | 90 | 0 | 10 | 0 | 0 | 0 | 0 | 2 |
| SPARCNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| SCCNet-NB | 78 | 90 | 0 | 0 | 0 | 0 | 10 | 0 | 2 |
| QUANT | 81 | 90 | 0 | 0 | 5 | 0 | 2 | 0 | 2 |
| STC-aeon | 81 | 90 | 5 | 0 | 0 | 0 | 2 | 0 | 2 |
| MRHydra | 81 | 90 | 0 | 0 | 3 | 0 | 2 | 2 | 2 |
| CBraMod-FT-random | 84 | 90 | 0 | 2 | 0 | 0 | 2 | 0 | 2 |
| XdawnTSLR-NB | 84 | 90 | 0 | 3 | 0 | 0 | 0 | 1 | 2 |
| XdawnCov-TS-SVM-MOABB | 84 | 90 | 1 | 0 | 1 | 0 | 2 | 0 | 2 |
| FBTRCA | 84 | 90 | 0 | 4 | 0 | 0 | 0 | 0 | 2 |
| DE-LR | 85 | 90 | 0 | 3 | 0 | 0 | 0 | 0 | 2 |
| DE-SVM | 85 | 90 | 0 | 3 | 0 | 0 | 0 | 0 | 2 |
| CBraMod-FT | 85 | 90 | 0 | 1 | 0 | 0 | 2 | 0 | 2 |
| Arsenal | 85 | 90 | 0 | 0 | 0 | 1 | 2 | 0 | 2 |
| CoSpectraLR-NB | 85 | 90 | 0 | 3 | 0 | 0 | 0 | 0 | 2 |
| TSLR-NB | 85 | 90 | 0 | 3 | 0 | 0 | 0 | 0 | 2 |
| TSLR-MOABB | 85 | 90 | 0 | 3 | 0 | 0 | 0 | 0 | 2 |
| FgMDM-MOABB | 85 | 90 | 0 | 0 | 1 | 0 | 2 | 0 | 2 |
| CSP-SVM-MOABB | 86 | 90 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| TemplateCCA-BP640 | 86 | 90 | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| TRCA-BP640 | 86 | 90 | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| TSSVM-MOABB | 86 | 90 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| EEGMiner-NB | 86 | 88 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

## Use

```
python tools/results_registry.py refresh iridis   # or hali: snapshot task states
python tools/results_registry.py pull iridis      # refresh, fetch ready cells, rebuild
python tools/results_registry.py build            # offline rebuild
```

`pull` fetches only the files of cells the cluster reports as done and the desktop
lacks: test files for this registry, train files for registry/train. It never changes anything remote and never overwrites a local file.
