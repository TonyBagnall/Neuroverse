# Leaderboards

`python -m neuroverse.experiments.tables` builds the three-page interactive dashboard in `results/neuroverse/` and updates the summary tables in `README.md` and this document.

## Interactive Dashboard Pages

| Page | Purpose | Scope |
|---|---|---|
| [`results/neuroverse/leaderboard.html`](../results/neuroverse/leaderboard.html) | Classifier leaderboard opening on the **12 Headline Classifiers** across the primary **Core 90 (`30/30/30`)** split, expandable to **81 distinct classifiers** or **109 configurations**, with problem-type tabs, Split-by-Problem-Type columns, coverage switches, critical difference diagrams, and interactive result export | 12 headline &rarr; 81 distinct &rarr; 109 configs across 90 Core (114 Extended) scalp EEG problems |
| [`results/neuroverse/matrix.html`](../results/neuroverse/matrix.html) | Interactive problem-by-classifier score heatmap grouped under paradigm banners, with pool switcher (`12` / `81` / `109`), metric selector, multi-metric hover tooltips, and CSV/JSON export | 90 Core (114 Extended) problems &times; 12 / 81 / 109 classifiers |
| [`results/neuroverse/datasets.html`](../results/neuroverse/datasets.html) | Per-problem explorer comparing the `Default` floor, median, best score, winning paradigm, gain over floor, and spread | 90 Core (`30 MI + 30 Clinical + 30 Cognitive`) &amp; 114 Extended scalp EEG problems |
| [`results/neuroverse_paper_2026/leaderboard.html`](../results/neuroverse_paper_2026/leaderboard.html) | Archived Rushbrooke et al. (2026) DMLR paper leaderboard | 20 estimators across 30 archive problems |

## Benchmark Scope & Three-Tier Classifier Hierarchy

The primary benchmark scope is **`Core 90`** (`neuroverse_study` / `neuroverse_90`, listed in `eeg.txt`): a symmetric **`30 / 30 / 30`** design comprising **30 Motor Imagery**, **30 Clinical**, and **30 Cognitive** scalp-EEG classification problems. The remaining **24** non-MI BCI tasks (`6` Motor Execution, `8` ERP/P300, `6` VEP/SSVEP, and `4` Speech BCI) remain available in the extended **114-problem** tabs (`All 114`, `BCI`, `ME`, `ERP`, `VEP`, `Speech`, listed in `eeg_114.txt`).

The result CSVs contain **109 classifier configurations** organised into three explicit pools:

1. **Headline Leaderboard (`HEADLINE_ESTIMATORS`, 12 classifiers, primary default)**:
   - Hybrid: `HC2`
   - Convolution: `Arsenal`, `MRHydra`
   - Interval: `QUANT`, `DrCIF`
   - Shapelet: `STC`, `RDST`
   - Dictionary: `TDE`
   - Feature: `Catch22`, `FreshPRINCE`
   - Deep Learning: `InceptionTime`
   - Tabular: `RotationForest`
2. **Expanded Distinct Classifiers (`CANONICAL_ESTIMATORS`, 81 classifiers)**:
   - One canonical representative per underlying algorithm across all five paradigms (13 General TSML, 26 EEG Classical, 33 EEG Supervised DL, 3 EEG Foundation, and 6 Tabular/Baseline), avoiding clone bias in Friedman/Wilcoxon rank tests.
3. **All Configurations (`BAKEOFF_ESTIMATORS`, 109 configurations)**:
   - Adds **28 preprocessing, probe, tree-count, and random-weight ablation controls** (`VARIANT_ESTIMATORS`, such as `-BP140`/`-BP830` band-pass variants, `HC2-500-pc`, `DrCIF-200`, frozen linear probes `CBraMod`/`LaBraM`/`NeuroGPT`, and `*-random` architectural controls).

Because extended cluster experiments include computationally intensive hybrids (`HC2`, `FreshPRINCE`) as well as structurally restricted EEG methods (such as `TRCSP-LDA-MOABB` on binary tasks and `LaBraM-FT` on standard 10-20 channel labels), requiring 100% completion across all problems would exclude key methods while newly assimilated datasets finish running.

The leaderboard therefore provides two explicit ranking modes in the UI and Python API:

1. **Common Block (`>=80%` coverage, primary default)**:
   - For a selected problem scope of $N$ problems ($N_{\text{eval}}$ with at least one competitive score while newly assimilated problems await cluster runs), every classifier in the active pool with results on at least $\lceil 0.8 N_{\text{eval}} \rceil$ problems is admitted.
   - Admitted classifiers are then evaluated strictly on their **shared completed problem intersection** (`common_problems`), so mean ranks, mean balanced accuracy, mean accuracy, AUROC, log loss, wins (ties included), and Wilcoxon-Holm critical difference cliques are paired on the exact same problems.
   - Tighter common-block thresholds (`>=90%` and `100%`) can be selected directly in `leaderboard.html`.
2. **All Available (No coverage gate, descriptive mode)**:
   - Admits every classifier in the active pool with at least one result on the selected tab, normalising within-problem ranks to $[1, M]$ so partially completed or restricted pipelines (such as `FreshPRINCE` while cluster runs complete) can be inspected alongside full-archive methods.

## Downloading & Querying Results Interactively

Users can download or slice benchmark results in three ways:

1. **In the browser**: Click **Download Results** in the top bar of `leaderboard.html`, `matrix.html`, or `datasets.html` to pick a problem split, classifier pool, metric set, and export layout (`Wide CSV`, `Tidy Long CSV`, `Ranked League CSV`, or `JSON`), plus a copyable Python/CLI command.
2. **In Python (`neuroverse.load_results` / `neuroverse.download_results`)**:

```python
import neuroverse as nv

# Wide DataFrame (90 Core problems x 12 Headline classifiers)
core_balacc = nv.load_results(metric="balacc", split="core90", estimators="headline")

# Ranked league table across all 81 distinct classifiers on Motor Imagery
mi_league = nv.load_results(metric="balacc", split="MI", estimators="canonical", format="league")

# Export Multiverse per-estimator directories or tidy long CSVs
nv.download_results("export_mv", format="multiverse", split="core90", estimators="headline")
```

3. **From the terminal (`neuroverse-results` / `python -m neuroverse.results`)**:

```bash
neuroverse-results                                                                      # interactive wizard
neuroverse-results --split core90 --estimators headline --format league --out core.csv  # direct export
```

For custom HTML single-table generation, `leaderboard()` also works directly with both wide summary CSVs and Multiverse directories:

```python
from neuroverse.datasets import by_type
from neuroverse.experiments.tables import complete_estimators, leaderboard

erp_problems = by_type("ERP")
candidates = ["Deep4Net-NB", "EEGNeX-NB", "HC2", "MRHydra", "XdawnCov-TS-SVM-MOABB"]
leaderboard(
    datasets=erp_problems,
    estimators=complete_estimators(erp_problems, candidates, metric="balacc"),
    metrics=["balacc", "accuracy", "auroc", "logloss"],
    sort_by="balacc",
    critical_difference=True,
    output_path="erp_leaderboard.html",
)
```

---

## Domain Leaderboards (Top 15 per Problem Type)

<!-- LEADERBOARD_BY_TYPE:START -->
### Headline Classifiers on Core 90 & 30/30/30 Domains

#### Core 90 (30/30/30) - Headline Pool (90 problems; 45 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | Hybrid | 48/90 | 3.30 | 56.23% | 59.06% | 12 | 37 |
| 2 | `Arsenal` | ConvolutionBased | 55/90 | 4.37 | 55.06% | 58.42% | 4 | 32 |
| 3 | `DrCIF` | IntervalBased | 49/90 | 4.66 | 53.57% | 57.49% | 5 | 26 |
| 4 | `QUANT` | IntervalBased | 52/90 | 4.71 | 54.49% | 57.84% | 8 | 27 |
| 5 | `MRHydra` | ConvolutionBased | 50/90 | 4.77 | 53.53% | 56.26% | 8 | 24 |
| 6 | `STC` | ShapeletBased | 52/90 | 5.30 | 53.19% | 56.79% | 1 | 22 |
| 7 | `RDST` | ShapeletBased | 52/90 | 5.73 | 51.98% | 55.36% | 3 | 22 |
| 8 | `TDE` | DictionaryBased | 48/90 | 7.04 | 48.96% | 52.76% | 2 | 10 |
| 9 | `InceptionTime` | DeepLearning | 51/90 | 7.48 | 46.01% | 47.72% | 2 | 11 |
| 10 | `Catch22` | FeatureBased | 48/90 | 7.64 | 47.87% | 51.97% | 0 | 7 |

#### Motor Imagery - Headline Pool (30 problems; 25 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | Hybrid | 28/30 | 3.12 | 58.20% | 60.62% | 7 | 22 |
| 2 | `DrCIF` | IntervalBased | 28/30 | 4.50 | 56.54% | 60.00% | 3 | 16 |
| 3 | `Arsenal` | ConvolutionBased | 29/30 | 4.66 | 56.87% | 59.30% | 3 | 16 |
| 4 | `QUANT` | IntervalBased | 29/30 | 5.34 | 55.72% | 59.08% | 1 | 14 |
| 5 | `MRHydra` | ConvolutionBased | 29/30 | 5.38 | 54.95% | 56.97% | 6 | 10 |
| 6 | `STC` | ShapeletBased | 28/30 | 5.84 | 55.18% | 58.36% | 0 | 10 |
| 7 | `RotationForest` | Tabular | 26/30 | 6.18 | 51.39% | 54.58% | 3 | 10 |
| 8 | `RDST` | ShapeletBased | 29/30 | 6.20 | 54.40% | 57.34% | 1 | 11 |
| 9 | `TDE` | DictionaryBased | 29/30 | 7.90 | 50.25% | 53.21% | 0 | 3 |
| 10 | `InceptionTime` | DeepLearning | 29/30 | 8.34 | 47.58% | 49.62% | 1 | 6 |
| 11 | `Catch22` | FeatureBased | 28/30 | 8.54 | 49.20% | 52.36% | 0 | 4 |

#### Clinical - Headline Pool (30 problems; 14 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `QUANT` | IntervalBased | 15/30 | 1.93 | 55.93% | 61.32% | 8 | 14 |
| 2 | `Arsenal` | ConvolutionBased | 17/30 | 2.36 | 53.57% | 62.54% | 2 | 14 |
| 3 | `STC` | ShapeletBased | 15/30 | 2.57 | 51.76% | 59.87% | 2 | 14 |
| 4 | `RDST` | ShapeletBased | 15/30 | 3.14 | 48.84% | 56.35% | 2 | 14 |

#### Cognitive - Headline Pool (30 problems; 8 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `RDST` | ShapeletBased | 8/30 | 3.25 | 49.28% | 51.86% | 2 | 6 |
| 2 | `Arsenal` | ConvolutionBased | 9/30 | 3.62 | 50.94% | 53.32% | 1 | 7 |
| 3 | `MRHydra` | ConvolutionBased | 8/30 | 3.69 | 49.45% | 50.87% | 1 | 7 |
| 4 | `DrCIF` | IntervalBased | 8/30 | 4.06 | 49.25% | 51.65% | 1 | 7 |
| 5 | `STC` | ShapeletBased | 9/30 | 4.12 | 48.14% | 50.91% | 1 | 6 |
| 6 | `QUANT` | IntervalBased | 8/30 | 4.19 | 49.94% | 51.85% | 3 | 4 |
| 7 | `Catch22` | FeatureBased | 8/30 | 6.31 | 44.54% | 47.86% | 0 | 2 |
| 8 | `InceptionTime` | DeepLearning | 8/30 | 6.75 | 42.41% | 44.06% | 0 | 1 |

### Expanded 81-Classifier Leaderboards (Top 15 per Scope)

#### Core 90 (30/30/30) (90 problems; 52 common; 47 classifiers &ge;54/67 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `Arsenal` | General TSML | 55/90 | 13.54 | 56.80% | 59.86% | 4 | 18 |
| 2 | `EEGNeX-NB` | EEG Supervised DL | 67/90 | 15.00 | 54.72% | 54.71% | 9 | 21 |
| 3 | `TIDNet-NB` | EEG Supervised DL | 67/90 | 17.47 | 53.91% | 53.36% | 0 | 6 |
| 4 | `CoSpectraLR-NB` | EEG Classical | 56/90 | 18.32 | 54.00% | 56.03% | 3 | 5 |
| 5 | `EEGInceptionMI-NB` | EEG Supervised DL | 67/90 | 18.53 | 53.50% | 52.42% | 3 | 10 |
| 6 | `TSLR-MOABB` | EEG Classical | 56/90 | 18.80 | 54.18% | 55.92% | 1 | 7 |
| 7 | `TSSVM-MOABB` | EEG Classical | 67/90 | 19.10 | 54.35% | 56.94% | 1 | 6 |
| 8 | `DE-LR` | EEG Classical | 67/90 | 19.15 | 54.02% | 55.68% | 1 | 9 |
| 9 | `IFNet-NB` | EEG Supervised DL | 67/90 | 19.22 | 53.06% | 52.53% | 0 | 8 |
| 10 | `XdawnTSLR-NB` | EEG Classical | 55/90 | 19.75 | 52.57% | 53.26% | 2 | 10 |
| 11 | `TSEN-MOABB` | EEG Classical | 67/90 | 20.53 | 53.42% | 55.08% | 2 | 5 |
| 12 | `SPARCNet-NB` | EEG Supervised DL | 56/90 | 20.97 | 51.16% | 49.99% | 0 | 5 |
| 13 | `EEGConformer-NB` | EEG Supervised DL | 56/90 | 21.42 | 52.12% | 51.53% | 2 | 9 |
| 14 | `FBMSNet-NB` | EEG Supervised DL | 67/90 | 21.47 | 52.60% | 51.84% | 2 | 5 |
| 15 | `DLCSP-shLDA-MOABB` | EEG Classical | 67/90 | 21.71 | 53.72% | 56.22% | 0 | 6 |

#### Motor Imagery (30 problems; 20 common; 76 classifiers &ge;24/30)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | General TSML | 28/30 | 14.35 | 61.52% | 62.40% | 1 | 6 |
| 2 | `Arsenal` | General TSML | 29/30 | 20.20 | 59.93% | 60.81% | 1 | 6 |
| 3 | `DrCIF` | General TSML | 28/30 | 20.35 | 60.20% | 62.31% | 2 | 7 |
| 4 | `MRHydra` | General TSML | 29/30 | 21.77 | 58.46% | 59.35% | 1 | 2 |
| 5 | `QUANT` | General TSML | 29/30 | 24.20 | 59.07% | 61.07% | 0 | 3 |
| 6 | `TIDNet-NB` | EEG Supervised DL | 30/30 | 24.90 | 55.73% | 55.32% | 0 | 0 |
| 7 | `STC` | General TSML | 28/30 | 25.98 | 58.26% | 60.03% | 0 | 3 |
| 8 | `RDST` | General TSML | 29/30 | 26.02 | 57.63% | 59.17% | 1 | 2 |
| 9 | `CoSpectraLR-NB` | EEG Classical | 29/30 | 26.10 | 56.55% | 57.24% | 1 | 2 |
| 10 | `TSLR-MOABB` | EEG Classical | 29/30 | 26.40 | 57.58% | 58.62% | 0 | 3 |
| 11 | `AUG-TANG-SVM-MOABB` | EEG Classical | 25/30 | 26.73 | 58.05% | 60.15% | 3 | 4 |
| 12 | `DE-LR` | EEG Classical | 30/30 | 27.23 | 56.40% | 56.97% | 0 | 3 |
| 13 | `EEGNeX-NB` | EEG Supervised DL | 30/30 | 27.30 | 53.99% | 53.93% | 2 | 4 |
| 14 | `FgMDM-MOABB` | EEG Classical | 30/30 | 27.40 | 57.03% | 57.87% | 0 | 1 |
| 15 | `TSEN-MOABB` | EEG Classical | 30/30 | 28.43 | 57.36% | 58.47% | 0 | 2 |

#### Clinical (30 problems; 16 common; 47 classifiers &ge;16/20 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `EEGNeX-NB` | EEG Supervised DL | 20/30 | 15.28 | 59.12% | 59.45% | 0 | 5 |
| 2 | `DE-SVM` | EEG Classical | 20/30 | 17.03 | 58.67% | 62.66% | 0 | 4 |
| 3 | `TSLR-MOABB` | EEG Classical | 18/30 | 18.16 | 57.30% | 60.98% | 1 | 2 |
| 4 | `CoSpectraLR-NB` | EEG Classical | 18/30 | 18.44 | 56.22% | 61.20% | 0 | 1 |
| 5 | `CSP-LDA-MOABB` | EEG Classical | 18/30 | 18.53 | 57.72% | 61.28% | 1 | 2 |
| 6 | `TSSVM-MOABB` | EEG Classical | 20/30 | 18.88 | 57.25% | 63.05% | 0 | 1 |
| 7 | `Arsenal` | General TSML | 17/30 | 19.03 | 56.50% | 63.76% | 0 | 3 |
| 8 | `BrainModule-NB` | EEG Supervised DL | 18/30 | 19.56 | 56.29% | 53.02% | 0 | 2 |
| 9 | `ShallowFBCSPNet-NB` | EEG Supervised DL | 18/30 | 19.66 | 55.20% | 51.24% | 0 | 0 |
| 10 | `DLCSP-shLDA-MOABB` | EEG Classical | 20/30 | 19.69 | 57.01% | 60.52% | 0 | 2 |
| 11 | `IFNet-NB` | EEG Supervised DL | 20/30 | 19.88 | 55.61% | 52.92% | 0 | 2 |
| 12 | `EEGInceptionMI-NB` | EEG Supervised DL | 20/30 | 19.94 | 55.28% | 52.30% | 1 | 2 |
| 13 | `SincShallowNet-NB` | EEG Supervised DL | 20/30 | 20.06 | 56.15% | 52.66% | 1 | 4 |
| 14 | `CSP-SVM-MOABB` | EEG Classical | 20/30 | 20.16 | 55.19% | 61.28% | 1 | 2 |
| 15 | `DE-LR` | EEG Classical | 20/30 | 20.62 | 56.74% | 60.35% | 0 | 3 |

#### Cognitive (30 problems; 16 common; 29 classifiers &ge;14/17 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `EEGNeX-NB` | EEG Supervised DL | 17/30 | 8.69 | 44.06% | 43.44% | 3 | 7 |
| 2 | `IFNet-NB` | EEG Supervised DL | 17/30 | 10.41 | 43.27% | 44.53% | 0 | 2 |
| 3 | `TIDNet-NB` | EEG Supervised DL | 17/30 | 10.66 | 43.91% | 43.94% | 1 | 6 |
| 4 | `EEGInceptionMI-NB` | EEG Supervised DL | 17/30 | 10.72 | 46.27% | 46.37% | 2 | 8 |
| 5 | `DE-LR` | EEG Classical | 17/30 | 10.84 | 44.04% | 45.79% | 0 | 4 |
| 6 | `EEGSimpleConv-NB` | EEG Supervised DL | 17/30 | 11.50 | 43.63% | 45.43% | 0 | 6 |
| 7 | `DE-SVM` | EEG Classical | 17/30 | 11.56 | 44.75% | 45.77% | 2 | 5 |
| 8 | `XdawnCov-TS-SVM-MOABB` | EEG Classical | 17/30 | 11.81 | 44.60% | 45.97% | 0 | 3 |
| 9 | `TSSVM-MOABB` | EEG Classical | 17/30 | 12.28 | 45.36% | 45.67% | 2 | 4 |
| 10 | `TSEN-MOABB` | EEG Classical | 17/30 | 13.09 | 42.85% | 43.98% | 0 | 2 |
| 11 | `FBMSNet-NB` | EEG Supervised DL | 17/30 | 13.53 | 41.94% | 41.69% | 0 | 2 |
| 12 | `CBraMod-FT` | EEG Foundation | 17/30 | 14.03 | 43.21% | 43.69% | 2 | 3 |
| 13 | `SincShallowNet-NB` | EEG Supervised DL | 17/30 | 14.41 | 41.63% | 39.93% | 0 | 2 |
| 14 | `FBCNet-NB` | EEG Supervised DL | 17/30 | 14.47 | 41.62% | 42.68% | 0 | 1 |
| 15 | `CSP-SVM-MOABB` | EEG Classical | 17/30 | 14.53 | 43.70% | 45.49% | 1 | 2 |

#### All Scalp EEG (114 problems; 66 common; 63 classifiers &ge;73/91 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `Arsenal` | General TSML | 79/114 | 16.65 | 53.57% | 57.23% | 2 | 19 |
| 2 | `MRHydra` | General TSML | 73/114 | 18.29 | 52.67% | 55.33% | 8 | 11 |
| 3 | `EEGNeX-NB` | EEG Supervised DL | 91/114 | 19.05 | 51.74% | 51.20% | 9 | 25 |
| 4 | `STC` | General TSML | 74/114 | 20.38 | 52.46% | 56.36% | 3 | 12 |
| 5 | `QUANT` | General TSML | 76/114 | 21.58 | 52.85% | 56.65% | 3 | 13 |
| 6 | `EEGInceptionMI-NB` | EEG Supervised DL | 91/114 | 22.73 | 52.36% | 52.57% | 2 | 10 |
| 7 | `TIDNet-NB` | EEG Supervised DL | 91/114 | 22.80 | 50.92% | 50.60% | 0 | 7 |
| 8 | `CoSpectraLR-NB` | EEG Classical | 80/114 | 24.36 | 51.63% | 52.71% | 3 | 7 |
| 9 | `RDST` | General TSML | 75/114 | 24.55 | 50.74% | 54.47% | 2 | 9 |
| 10 | `XdawnTSLR-NB` | EEG Classical | 79/114 | 24.91 | 49.95% | 50.72% | 1 | 10 |
| 11 | `SPARCNet-NB` | EEG Supervised DL | 80/114 | 25.45 | 49.15% | 48.58% | 0 | 4 |
| 12 | `TMSANet-NB` | EEG Supervised DL | 76/114 | 26.02 | 50.26% | 49.22% | 2 | 6 |
| 13 | `IFNet-NB` | EEG Supervised DL | 91/114 | 26.64 | 50.59% | 50.34% | 1 | 5 |
| 14 | `CTNet-NB` | EEG Supervised DL | 80/114 | 27.65 | 49.14% | 49.35% | 0 | 6 |
| 15 | `Deep4Net-NB` | EEG Supervised DL | 80/114 | 27.87 | 49.83% | 48.98% | 1 | 8 |

#### BCI (All) (54 problems; 33 common; 74 classifiers &ge;44/54)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | General TSML | 50/54 | 13.41 | 56.35% | 57.12% | 1 | 13 |
| 2 | `Arsenal` | General TSML | 53/54 | 18.21 | 54.75% | 55.52% | 1 | 9 |
| 3 | `DrCIF` | General TSML | 51/54 | 19.82 | 54.75% | 56.32% | 2 | 9 |
| 4 | `MRHydra` | General TSML | 52/54 | 21.06 | 53.34% | 54.00% | 3 | 3 |
| 5 | `STC` | General TSML | 50/54 | 21.73 | 54.35% | 55.70% | 1 | 7 |
| 6 | `QUANT` | General TSML | 53/54 | 22.53 | 53.72% | 55.19% | 0 | 6 |
| 7 | `RotationForest` | Tabular | 46/54 | 24.44 | 52.16% | 53.50% | 2 | 7 |
| 8 | `RDST` | General TSML | 52/54 | 24.53 | 52.80% | 53.97% | 1 | 6 |
| 9 | `EEGNeX-NB` | EEG Supervised DL | 54/54 | 25.21 | 50.51% | 50.33% | 3 | 8 |
| 10 | `XdawnTSLR-NB` | EEG Classical | 53/54 | 27.94 | 49.97% | 50.28% | 1 | 5 |
| 11 | `CoSpectraLR-NB` | EEG Classical | 53/54 | 28.09 | 52.33% | 52.89% | 3 | 4 |
| 12 | `TIDNet-NB` | EEG Supervised DL | 54/54 | 28.35 | 50.65% | 50.33% | 0 | 1 |
| 13 | `TSEN-MOABB` | EEG Classical | 54/54 | 29.55 | 50.20% | 51.10% | 1 | 3 |
| 14 | `EEGInceptionMI-NB` | EEG Supervised DL | 54/54 | 30.76 | 51.58% | 51.28% | 0 | 3 |
| 15 | `SPARCNet-NB` | EEG Supervised DL | 53/54 | 30.76 | 47.98% | 48.24% | 0 | 3 |

#### Motor Execution (6 problems; 4 common; 75 classifiers &ge;5/6)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | General TSML | 6/6 | 9.25 | 72.05% | 72.05% | 0 | 2 |
| 2 | `Arsenal` | General TSML | 6/6 | 10.50 | 70.28% | 70.20% | 0 | 1 |
| 3 | `TCFormer-NB` | EEG Supervised DL | 6/6 | 11.50 | 70.25% | 70.19% | 2 | 2 |
| 4 | `EEGNeX-NB` | EEG Supervised DL | 6/6 | 12.00 | 71.71% | 71.65% | 1 | 3 |
| 5 | `TSception-NB` | EEG Supervised DL | 5/6 | 13.75 | 69.48% | 69.39% | 0 | 1 |
| 6 | `TIDNet-NB` | EEG Supervised DL | 6/6 | 14.00 | 69.94% | 69.94% | 0 | 1 |
| 7 | `SPARCNet-NB` | EEG Supervised DL | 6/6 | 15.75 | 68.29% | 68.14% | 0 | 0 |
| 8 | `RotationForest` | Tabular | 5/6 | 16.50 | 68.87% | 68.78% | 0 | 1 |
| 9 | `STC` | General TSML | 6/6 | 16.75 | 67.98% | 67.90% | 0 | 0 |
| 10 | `Deep4Net-NB` | EEG Supervised DL | 6/6 | 18.25 | 68.28% | 68.22% | 0 | 0 |
| 11 | `DrCIF` | General TSML | 6/6 | 20.00 | 66.53% | 66.58% | 0 | 0 |
| 12 | `CTNet-NB` | EEG Supervised DL | 6/6 | 20.25 | 67.21% | 67.17% | 0 | 1 |
| 13 | `EEGConformer-NB` | EEG Supervised DL | 6/6 | 20.75 | 66.58% | 66.54% | 0 | 2 |
| 14 | `EEGInceptionMI-NB` | EEG Supervised DL | 6/6 | 20.75 | 66.32% | 66.46% | 0 | 1 |
| 15 | `FBMSNet-NB` | EEG Supervised DL | 6/6 | 22.00 | 65.63% | 65.73% | 0 | 0 |

#### ERP / P300 (8 problems; 7 common; 67 classifiers &ge;7/8)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `EEGNeX-NB` | EEG Supervised DL | 8/8 | 11.07 | 67.42% | 60.98% | 1 | 4 |
| 2 | `Deep4Net-NB` | EEG Supervised DL | 8/8 | 11.57 | 66.72% | 62.50% | 0 | 1 |
| 3 | `CBraMod-FT` | EEG Foundation | 7/8 | 11.79 | 68.03% | 65.64% | 0 | 3 |
| 4 | `XdawnTSLR-NB` | EEG Classical | 8/8 | 11.86 | 66.36% | 66.37% | 0 | 1 |
| 5 | `SPARCNet-NB` | EEG Supervised DL | 8/8 | 13.43 | 64.84% | 60.64% | 0 | 1 |
| 6 | `EEGInceptionMI-NB` | EEG Supervised DL | 8/8 | 13.50 | 67.79% | 70.21% | 1 | 1 |
| 7 | `CTNet-NB` | EEG Supervised DL | 8/8 | 15.50 | 64.28% | 58.48% | 0 | 2 |
| 8 | `EEGSimpleConv-NB` | EEG Supervised DL | 8/8 | 16.21 | 64.27% | 64.76% | 0 | 4 |
| 9 | `MRHydra` | General TSML | 7/8 | 16.57 | 61.36% | 78.38% | 1 | 1 |
| 10 | `BrainModule-NB` | EEG Supervised DL | 8/8 | 18.00 | 63.72% | 64.63% | 0 | 3 |
| 11 | `TCFormer-NB` | EEG Supervised DL | 8/8 | 18.64 | 61.73% | 55.72% | 2 | 3 |
| 12 | `TMSANet-NB` | EEG Supervised DL | 8/8 | 19.43 | 60.81% | 50.15% | 0 | 0 |
| 13 | `IFNet-NB` | EEG Supervised DL | 8/8 | 20.14 | 61.66% | 56.49% | 1 | 1 |
| 14 | `TIDNet-NB` | EEG Supervised DL | 8/8 | 21.43 | 61.38% | 58.92% | 0 | 1 |
| 15 | `EEGNet-NB` | EEG Supervised DL | 8/8 | 22.36 | 62.81% | 51.72% | 0 | 0 |

#### VEP / SSVEP (6 problems; 3 common; 74 classifiers &ge;5/6)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `STC` | General TSML | 5/6 | 9.33 | 37.42% | 38.32% | 0 | 1 |
| 2 | `XdawnCov-MDM-MOABB` | EEG Classical | 6/6 | 11.33 | 53.80% | 53.90% | 0 | 1 |
| 3 | `LaBraM-FT` | EEG Foundation | 5/6 | 12.67 | 62.81% | 62.18% | 0 | 1 |
| 4 | `RDST` | General TSML | 6/6 | 13.00 | 31.01% | 31.64% | 0 | 0 |
| 5 | `xDAWN-LDA-MOABB` | EEG Classical | 5/6 | 13.33 | 47.63% | 47.87% | 0 | 0 |
| 6 | `HC2` | General TSML | 5/6 | 13.33 | 31.15% | 31.92% | 0 | 1 |
| 7 | `DrCIF` | General TSML | 6/6 | 13.33 | 30.50% | 31.59% | 0 | 0 |
| 8 | `AUG-TANG-SVM-MOABB` | EEG Classical | 5/6 | 13.67 | 43.12% | 43.89% | 0 | 1 |
| 9 | `MRHydra` | General TSML | 6/6 | 13.83 | 30.27% | 30.83% | 0 | 0 |
| 10 | `QUANT` | General TSML | 6/6 | 14.00 | 29.36% | 30.33% | 0 | 0 |
| 11 | `TemplateCCA-BP640` | EEG Classical | 6/6 | 14.83 | 50.29% | 50.29% | 0 | 1 |
| 12 | `ERPCov-MDM-MOABB` | EEG Classical | 5/6 | 15.50 | 49.82% | 49.87% | 0 | 1 |
| 13 | `Arsenal` | General TSML | 6/6 | 17.00 | 27.95% | 28.58% | 0 | 1 |
| 14 | `XdawnCov-TS-SVM-MOABB` | EEG Classical | 6/6 | 18.83 | 54.58% | 55.28% | 0 | 0 |
| 15 | `CBraMod-FT` | EEG Foundation | 6/6 | 20.50 | 41.19% | 41.10% | 0 | 1 |

#### Speech BCI (4 problems; 4 common; 75 classifiers &ge;4/4)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `TDE` | General TSML | 4/4 | 3.50 | 35.75% | 35.70% | 3 | 3 |
| 2 | `QUANT` | General TSML | 4/4 | 6.50 | 33.41% | 33.12% | 0 | 3 |
| 3 | `RotationForest` | Tabular | 4/4 | 13.75 | 28.95% | 28.88% | 0 | 0 |
| 4 | `STC` | General TSML | 4/4 | 14.00 | 30.07% | 30.12% | 1 | 2 |
| 5 | `HC2` | General TSML | 4/4 | 16.25 | 31.03% | 31.04% | 0 | 2 |
| 6 | `RiemannianKNN-OAS-BP140` | EEG Classical | 4/4 | 18.00 | 30.17% | 30.19% | 0 | 0 |
| 7 | `CoSpectraLR-NB` | EEG Classical | 4/4 | 22.00 | 26.40% | 26.39% | 0 | 0 |
| 8 | `TSSVM-MOABB` | EEG Classical | 4/4 | 23.75 | 28.22% | 28.27% | 0 | 0 |
| 9 | `DrCIF` | General TSML | 4/4 | 24.00 | 32.07% | 31.91% | 0 | 2 |
| 10 | `MRHydra` | General TSML | 4/4 | 25.75 | 27.59% | 27.41% | 0 | 0 |
| 11 | `TSEN-MOABB` | EEG Classical | 4/4 | 26.75 | 25.48% | 25.36% | 0 | 0 |
| 12 | `TemplateCCA-BP640` | EEG Classical | 4/4 | 28.25 | 24.84% | 24.91% | 0 | 0 |
| 13 | `Arsenal` | General TSML | 4/4 | 28.75 | 26.85% | 26.85% | 0 | 0 |
| 14 | `RandomForest` | Tabular | 4/4 | 29.75 | 25.93% | 26.02% | 0 | 0 |
| 15 | `FgMDM-MOABB` | EEG Classical | 4/4 | 30.25 | 25.28% | 25.25% | 0 | 0 |
<!-- LEADERBOARD_BY_TYPE:END -->
