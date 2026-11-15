# Evaluation

## Protocol

- **Fixed split (`resample 0`)**: Every classifier is fitted once on the supplied `TRAIN` partition and evaluated once on the supplied `TEST` partition (`testResample0.csv` in `tsml-eval`). Because EEG trials from a single recording or subject correlate strongly, random stratified resamples across `TRAIN + TEST` would leak subject/session identity.
- **Default hyperparameters & clean canonical names**: Classifiers run with their standard benchmark configurations and fixed preprocessing tags (for example `-BP830` for 8-30 Hz motor-rhythm band-pass, `-BP140` for 1-40 Hz broadband pass, `-BP640` for 6-40 Hz SSVEP pass, `-MOABB` for canonical MOABB pipelines, and `-NB` for NeuralBench protocol baselines and Braindecode wrappers in `aeon-neuro`). In exported summary tables, raw cluster directory tags for the primary TSML estimators are mapped to clean canonical names (`HC2-noTDE-500-pc` &rarr; `HC2`, `DrCIF-500` &rarr; `DrCIF` with 200-tree `DrCIF` retained as variant `DrCIF-200`, `STC-aeon` &rarr; `STC`, and `IT` &rarr; `InceptionTime`).
- **Explicit foundation controls**: Pretrained EEG foundation models (`CBraMod-FT`, `LaBraM-FT`, `NeuroGPT-FT` and their linear probes) are accompanied by random-weight architectural controls (`*-random`) to separate pretraining transfer from backbone inductive bias.

## Metrics

Primary benchmark ranking uses **test balanced accuracy** (`balacc`) so class-imbalanced paradigms (such as P300 target detection, seizure detection, and sleep staging) cannot be gamed by majority-class prediction.

| File / Column key | Metric | Direction | Notes |
|---|---|---|---|
| `balacc` | Balanced accuracy | Higher is better | Unweighted mean recall across classes (primary ranking metric) |
| `accuracy` | Accuracy | Higher is better | Proportion of exact test matches |
| `auroc` | AUROC | Higher is better | Binary ROC AUC or macro one-vs-rest multi-class ROC AUC |
| `logloss` | Log loss | Lower is better | Cross-entropy with probabilities clipped to $[10^{-15}, 1]$ |

*(In the archived 2026 DMLR Multiverse layout under `results/neuroverse_paper_2026/`, `f1`, `sensitivity`, and `specificity` are also stored.)*

Estimators that emit hard $0/1$ vote proportions (such as `Arsenal` or `MRHydra`) incur larger clipped `logloss` values; this reflects uncalibrated vote probabilities rather than a scoring error. Fit and predict wall-clock times are not ranked because neural models run on GPUs while CPU ensembles run single-threaded across heterogeneous cluster nodes.

## The Default Baseline

`Default` predicts the most frequent class in the **test** split for every test instance, using empirical test class proportions as probability estimates:
- `balacc` equals $1 / C$ for $C$ present classes;
- `auroc` equals $0.5$;
- `accuracy` equals the largest class proportion;
- `logloss` equals the empirical Shannon entropy of the test label distribution.

It serves on `datasets.html` and `matrix.html` as a strict no-skill floor (`Gain over Floor`).

## Where the Classifiers Live

Neuroverse stores metadata and benchmark scores only (never classifier code):

| Paradigm | Home Repository |
|---|---|
| Classical EEG (CSP, xDAWN, CCA, TRCA, Riemannian), Braindecode wrappers (`*-NB`), and EEG foundation models (`CBraMod`, `LaBraM`) | [`aeon-neuro`](https://github.com/aeon-toolkit/aeon-neuro) (`aeon_neuro.classification`) |
| General multivariate TSC ports outside `aeon` | [`multiverse`](https://github.com/aeon-toolkit/multiverse) (`multiverse.classification`) |
| General TSML ensembles, convolutions, intervals, shapelets, dictionaries, and deep learners | [`aeon`](https://github.com/aeon-toolkit/aeon) (`aeon.classification`) |

## Result File Formats

`neuroverse.experiments.tables` and `neuroverse.results` transparently read two layouts:

1. **Wide summary CSVs** (active benchmark in `results/neuroverse/`):
   - `accuracy.csv`, `balacc.csv`, `auroc.csv`, `logloss.csv` (`114` scalp-EEG dataset rows, with the `90` Core `30/30/30` tasks flagged in `datasets.csv`, &times; `109` configuration columns covering **12 Headline** and **81 Distinct Canonical** classifiers plus **28 preprocessing/probe/ablation variants**)
   - `datasets.csv` (`dataset,family,bci_type,core_90`) and `classifiers.csv` (`classifier,family,source_classifier`)
   - `Default/Default_<metric>.csv` for the constant test-majority floor
2. **Multiverse per-estimator CSVs** (archived 2026 paper results in `results/neuroverse_paper_2026/` and `aeon-toolkit/multiverse`):
   - `<results_dir>/<estimator>/<estimator>_<metric>.csv` with `Resamples:` index header.

End users can interactively query or export any slice in Python (`nv.load_results`, `nv.download_results`), via the CLI (`neuroverse-results`), or from the **Download Results** modal in `leaderboard.html`, `matrix.html`, and `datasets.html`. Maintainers can refresh wide snapshots from raw `tsml-eval` predictions with `neuroverse.experiments.ingest`:

```bash
# Export wide summary CSVs directly from a tsml-eval category tree
python -m neuroverse.experiments.ingest --wide-from D:/Results/EEG --out results/neuroverse

# Sync from a dated wide snapshot directory
python -m neuroverse.experiments.ingest --sync-from results/tsml-eval-20261111 --out results/neuroverse

# Export Multiverse per-estimator directories from wide CSVs
python -m neuroverse.experiments.ingest --out results/neuroverse --to-multiverse results/multiverse_export

# Rebuild all HTML pages and Markdown tables
python -m neuroverse.experiments.tables
```

## Registered Benchmark Classifiers (12 Headline; 81 Distinct Classifiers; 109 Total Configurations)

<!-- ESTIMATORS:START -->
| Estimator | Role | Paradigm | tsml-eval Family | Library | Description |
|---|---|---|---|---|---|
| `Arsenal` | Headline (12) | General TSML | ConvolutionBased | aeon | Ensemble of ROCKET convolutional kernel classifiers |
| `ATCNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Attention Temporal Convolutional Network for EEG decoding |
| `AttentionBaseNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Channel-and-temporal attention convolutional baseline network |
| `AUG-TANG-SVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Augmented lag-embedded covariance tangent space + linear/RBF SVM |
| `BDTCN-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Braindecode dilated Temporal Convolutional Network |
| `BrainModule-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Meta AI wav2vec-style dilated convolutional EEG encoder |
| `Catch22` | Headline (12) | General TSML | FeatureBased | aeon | Canonical 22 time-series summary features per channel with Random Forest |
| `CBraMod` | Preprocessing / Probe Variant | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | Criss-cross Brain Foundation Model frozen-encoder linear probe |
| `CBraMod-FT` | Expanded (81) | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | Criss-cross Brain Foundation Model end-to-end fine-tuned patch transformer |
| `CBraMod-FT-random` | Ablation / Control | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | CBraMod fine-tuned from random weights (no pretraining control) |
| `CBraMod-random` | Ablation / Control | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | CBraMod architecture linear probe from random weights (no pretraining control) |
| `CNN` | Expanded (81) | General TSML | DeepLearning | aeon | Time-CNN multivariate 1D convolutional network |
| `ContraWR-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Contrastive world-representation STFT spectrogram 2D ResNet |
| `CoSpectraLR-NB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (NeuralBench) | NeuralBench Welch cross-spectral co-spectra features + LogisticRegressionCV |
| `CSP+LDA` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | Common Spatial Patterns (unfiltered) + Linear Discriminant Analysis |
| `CSP+LDA-BP140` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | 1-40 Hz broadband pass + Common Spatial Patterns + LDA |
| `CSP+LDA-BP830` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | 8-30 Hz mu/beta pass + Common Spatial Patterns + LDA |
| `CSP+SVM` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | Common Spatial Patterns (unfiltered) + Support Vector Machine |
| `CSP+SVM-BP140` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | 1-40 Hz broadband pass + Common Spatial Patterns + SVM |
| `CSP+SVM-BP830` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | 8-30 Hz mu/beta pass + Common Spatial Patterns + SVM |
| `CSP-LDA-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | MOABB standard Common Spatial Patterns + shrinkage LDA pipeline |
| `CSP-SVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | MOABB standard Common Spatial Patterns + SVM pipeline |
| `CTNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Convolutional Transformer Network for motor imagery EEG |
| `DE-LR` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro | Canonical sub-band Differential Entropy spectral features + Logistic Regression |
| `DE-SVM` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro | Canonical sub-band Differential Entropy spectral features + RBF SVM |
| `Deep4Net-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Schirrmeister et al. 4-block spatio-temporal ConvNet (DeepConvNet) |
| `DLCSP-shLDA-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Diagonal-Loading regularised CSP + shrinkage LDA |
| `DrCIF` | Headline (12) | General TSML | IntervalBased | aeon | Diverse Representation Canonical Interval Forest (500 trees) |
| `DrCIF-200` | Preprocessing / Probe Variant | General TSML | IntervalBased | aeon | Diverse Representation Canonical Interval Forest (default 200 trees) |
| `EEGConformer-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Compact spatio-temporal CNN + self-attention Transformer (Song et al.) |
| `EEGInceptionERP-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Multi-scale Inception CNN tailored to event-related potentials |
| `EEGInceptionMI-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Multi-scale Inception CNN tailored to motor imagery rhythms |
| `EEGITNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Inception + dilated Temporal Convolutional Network for EEG |
| `EEGNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Lawhern et al. compact depthwise-separable spatio-temporal CNN |
| `EEGNeX-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Pure ConvNeXt-inspired dilated spatio-temporal ConvNet (Chen et al.) |
| `EEGSimpleConv-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Lightweight 1D batch-normalised ResNet baseline for EEG |
| `EEGTCNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | EEGNet front-end coupled to a dilated Temporal Convolutional Network |
| `ERPCov-MDM-MOABB` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro (MOABB) | Class-prototype super-trial ERP covariance + Riemannian MDM |
| `FBCNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Filter-Bank Convolutional Network with variance temporal pooling |
| `FBCSP-SVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Filter-Bank Common Spatial Patterns + mutual-information selection + SVM |
| `FBLightConvNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Filter-bank lightweight depthwise group-convolutional network |
| `FBMSNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Filter-Bank Multi-Scale spatio-temporal CNN |
| `FBTRCA` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro | Filter-Bank Task-Related Component Analysis spatial filter ensemble |
| `FgMDM-MOABB` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro (MOABB) | Fisher geodesic discriminant filtering in tangent space + Riemannian MDM |
| `FreshPRINCE` | Headline (12) | General TSML | FeatureBased | aeon | TSFresh summary features per channel with Rotation Forest |
| `GCNsNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Spectral Graph Convolutional Network over electrode topology |
| `HC2` | Headline (12) | General TSML | Hybrid | aeon | HIVE-COTE v2.0 hybrid ensemble (Arsenal, DrCIF, and STC with prior correction) |
| `HC2-500-pc` | Preprocessing / Probe Variant | General TSML | Hybrid | aeon | HIVE-COTE v2.0 variant including TDE dictionary component |
| `IFNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Interactive Frequency Convolutional Network (low/high band interplay) |
| `InceptionTime` | Headline (12) | General TSML | DeepLearning | aeon | InceptionTime ensemble of multi-scale 1D Inception networks |
| `LaBraM` | Preprocessing / Probe Variant | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | Large Brain Model frozen-encoder linear probe (10-20 montage) |
| `LaBraM-FT` | Expanded (81) | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | Large Brain Model end-to-end fine-tuned neural tokeniser transformer (10-20 montage) |
| `LaBraM-FT-random` | Ablation / Control | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | LaBraM fine-tuned from random weights (no pretraining control) |
| `LaBraM-random` | Ablation / Control | EEG Foundation | DeepLearning | aeon-neuro (Foundation) | LaBraM architecture linear probe from random weights (no pretraining control) |
| `LITETime-MV` | Expanded (81) | General TSML | DeepLearning | aeon | Multivariate Light Inception with Boosting and depthwise convolutions |
| `LogisticRegression` | Expanded (81) | Tabular | Tabular | scikit-learn | L2-regularised Logistic Regression on concatenated channel values |
| `LogVar-LDA-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Per-channel log-variance features + shrinkage LDA |
| `LogVar-SVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Per-channel log-variance features + SVM |
| `MDM-MOABB` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro (MOABB) | MOABB Riemannian Minimum Distance to Mean over SPD covariances |
| `MiniROCKET` | Expanded (81) | General TSML | ConvolutionBased | aeon | Deterministic MiniROCKET convolutional kernel transform with Ridge classifier |
| `MRHydra` | Headline (12) | General TSML | ConvolutionBased | aeon | MultiROCKET combined with Hydra competing convolutional dictionary |
| `MSCFormer-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Multi-scale convolutional Transformer for EEG classification |
| `MSVTNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Multi-Scale Vision Transformer network for spatio-temporal EEG |
| `QUANT` | Headline (12) | General TSML | IntervalBased | aeon | Dyadic interval quantile features with ExtraTrees |
| `RandomForest` | Expanded (81) | Tabular | Tabular | scikit-learn | Random Forest on concatenated channel time points |
| `RDST` | Headline (12) | General TSML | ShapeletBased | aeon | Random Dilated Shapelet Transform with Ridge classifier |
| `RiemannianKNN` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | k-Nearest Neighbours under affine-invariant Riemannian SPD distance |
| `RiemannianKNN-LWF` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | Riemannian k-NN with Ledoit-Wolf shrinkage covariances |
| `RiemannianKNN-LWF-BP140` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 1-40 Hz pass + Ledoit-Wolf covariance + Riemannian k-NN |
| `RiemannianKNN-LWF-BP830` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 8-30 Hz mu/beta pass + Ledoit-Wolf covariance + Riemannian k-NN |
| `RiemannianKNN-OAS` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | Riemannian k-NN with Oracle Approximating Shrinkage covariances |
| `RiemannianKNN-OAS-BP140` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 1-40 Hz pass + OAS covariance + Riemannian k-NN |
| `RiemannianKNN-OAS-BP830` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 8-30 Hz mu/beta pass + OAS covariance + Riemannian k-NN |
| `RiemannianMDM` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | Riemannian Minimum Distance to Frechet Mean over sample covariances |
| `RiemannianMDM-LWF` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | Riemannian MDM with Ledoit-Wolf shrinkage covariances |
| `RiemannianMDM-LWF-BP140` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 1-40 Hz pass + Ledoit-Wolf covariance + Riemannian MDM |
| `RiemannianMDM-LWF-BP830` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 8-30 Hz mu/beta pass + Ledoit-Wolf covariance + Riemannian MDM |
| `RiemannianMDM-OAS` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | Riemannian MDM with Oracle Approximating Shrinkage covariances |
| `RiemannianMDM-OAS-BP140` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 1-40 Hz pass + OAS covariance + Riemannian MDM |
| `RiemannianMDM-OAS-BP830` | Preprocessing / Probe Variant | EEG Classical | DistanceBased | aeon-neuro (PyRiemann) | 8-30 Hz mu/beta pass + OAS covariance + Riemannian MDM |
| `ROCKET` | Expanded (81) | General TSML | ConvolutionBased | aeon | Random Convolutional Kernel Transform with Ridge classifier |
| `RotationForest` | Headline (12) | Tabular | Tabular | aeon | Rotation Forest (PCA feature-subset rotations + decision trees) |
| `SCCNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Spatial-Component-wise Convolutional Network |
| `ShallowFBCSPNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Schirrmeister et al. shallow ConvNet mimicking log-bandpower FBCSP |
| `SincShallowNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Interpretable SincNet band-pass filter front-end + shallow spatial CNN |
| `SleepStagerChambon2018-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Chambon et al. spatial-temporal ConvNet originally designed for PSG staging |
| `SPARCNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | DenseNet-1D multi-layer dilated convolutional network (Jing et al.) |
| `SSTDPN-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Spatial-Spectral-Temporal Dual-Prototype Network |
| `STC` | Headline (12) | General TSML | ShapeletBased | aeon | Randomised Shapelet Transform Classifier with Rotation Forest |
| `SVM` | Expanded (81) | Tabular | Tabular | scikit-learn | RBF-kernel Support Vector Machine on concatenated channel values |
| `SyncNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Synchronised Gabor-wavelet filter-bank CNN for oscillatory phase/power |
| `TCFormer-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Temporal Convolutional Transformer hybrid network |
| `TDE` | Headline (12) | General TSML | DictionaryBased | aeon | Temporal Dictionary Ensemble over SFA word histograms |
| `TemplateCCA-BP640` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro | 6-40 Hz pass + Individual Template Canonical Correlation Analysis |
| `TIDNet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Thinker-Invariant DenseNet for cross-subject EEG decoding |
| `TMSANet-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Temporal Multi-Scale Attention Network |
| `TRACE` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro | Temporal Riemannian Adaptive Classification Ensemble over weighted intervals |
| `TRCA-BP640` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro | 6-40 Hz pass + Task-Related Component Analysis spatial filter correlation |
| `TRCSP-LDA-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Tikhonov-Regularised Common Spatial Patterns + LDA (binary tasks) |
| `TSception-NB` | Expanded (81) | EEG Supervised DL | DeepLearning | aeon-neuro (Braindecode) | Multi-scale temporal and hemispheric asymmetry CNN for affective/BCI EEG |
| `TSEN-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Riemannian covariance tangent-space projection + ElasticNet logistic regression |
| `TSLR-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | MOABB Riemannian covariance tangent-space projection + L2 Logistic Regression |
| `TSLR-NB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (NeuralBench) | NeuralBench CovTsLR baseline (OAS covariance + tangent space + LogisticRegressionCV) |
| `TSSVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | Riemannian covariance tangent-space projection + linear SVM |
| `XDAWN` | Preprocessing / Probe Variant | EEG Classical | FeatureBased | aeon-neuro (MNE + sklearn) | xDAWN evoked-potential spatial filtering + vectorised LDA |
| `xDAWN-LDA-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | MOABB xDAWN spatial filtering + shrinkage LDA |
| `XdawnCov-MDM-MOABB` | Expanded (81) | EEG Classical | DistanceBased | aeon-neuro (MOABB) | xDAWN template-augmented block covariance + Riemannian MDM |
| `XdawnCov-TS-SVM-MOABB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (MOABB) | xDAWN template-augmented block covariance + tangent-space SVM |
| `XdawnTSLR-NB` | Expanded (81) | EEG Classical | FeatureBased | aeon-neuro (NeuralBench) | NeuralBench xDAWN spatial filtering + covariance tangent-space LogisticRegressionCV |
<!-- ESTIMATORS:END -->

## Dropped Classifiers

Dropped from the benchmark on 13 November 2026. Their results stay on disk, and they
are no longer in `results/neuroverse/classifiers.csv`; `DROPPED` in `tools/results_registry.py`
keeps the registry from expecting them. They are still listed in the table above, which is
generated from `neuroverse/estimators.py`.

| Estimator | Problems with results (of 76) | Why dropped |
|---|---|---|
| `TRCSP-LDA-MOABB` | 46 | Tikhonov-regularised CSP is a binary method: it fails with "TRCSP only supports 2 classes" on every multiclass problem (28 of the 30 failures), and on two more (ChildhoodSeizures, CovidSurvivors) with non-positive-definite covariances. It cannot cover the archive, and plain CSP pipelines (CSP+LDA, CSP-LDA-MOABB) cover the same idea on all problems. Removed from the Hali experiment (`hali_eeg_moabb_mi_more.toml`, tsml-eval c7b5c5d) on 14 November 2026. |
| `SSTDPN-NB` | 63 | On 13 problems a layer raises "Output size is too small" at prediction: the fixed convolution and pooling sizes need longer series than these problems have. It cannot run on them without changing the architecture. |
| `FBLightConvNet-NB` | 63 | Excluded for now (13 November 2026). Its temporal window is fixed at `win_len = 250` samples, and 12 problems have shorter series (30 to 240 samples): "Time dimension (...) is shorter than win_len (250)". Running it there would need a smaller window, which changes the published model. |
| `SleepStagerChambon2018-NB` | 66 | Excluded for now (13 November 2026). On 9 problems a layer raises "Output size is too small" at prediction: like SSTDPN, its fixed convolution and pooling sizes need longer series. |

## Dropped Problems

| Problem | Dropped | Why |
|---|---|---|
| `SymbolAttention` | 13 November 2026 | Not core to the study: a P300 speller problem, and by far the costliest in the archive (521,259 training and 175,180 test cases of 8 channels by 256 points). It held the HC2 components, the KNN and SVM classifiers and TRACE for days, and 26 main classifiers still had no result on it. Removed from `eeg.txt` and the cluster lists; its unfinished Hali jobs were cancelled. Results already on disk are kept. |
