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

#### Core 90 (30/30/30) - Headline Pool (90 problems; 43 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | Hybrid | 46/90 | 3.26 | 56.32% | 58.97% | 12 | 36 |
| 2 | `Arsenal` | ConvolutionBased | 53/90 | 4.20 | 55.43% | 58.51% | 4 | 32 |
| 3 | `DrCIF` | IntervalBased | 47/90 | 4.66 | 53.47% | 57.11% | 5 | 25 |
| 4 | `MRHydra` | ConvolutionBased | 48/90 | 4.72 | 53.55% | 56.13% | 8 | 23 |
| 5 | `QUANT` | IntervalBased | 50/90 | 4.88 | 53.61% | 57.16% | 6 | 25 |
| 6 | `STC` | ShapeletBased | 50/90 | 5.35 | 53.14% | 56.59% | 1 | 21 |
| 7 | `RDST` | ShapeletBased | 50/90 | 5.58 | 52.39% | 55.84% | 3 | 22 |
| 8 | `TDE` | DictionaryBased | 46/90 | 7.19 | 48.60% | 52.17% | 2 | 8 |
| 9 | `InceptionTime` | DeepLearning | 49/90 | 7.50 | 45.83% | 47.53% | 2 | 10 |
| 10 | `Catch22` | FeatureBased | 46/90 | 7.66 | 47.83% | 51.66% | 0 | 7 |

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

#### Clinical - Headline Pool (30 problems; 12 shared completed)

| # | Classifier | Family | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `QUANT` | IntervalBased | 13/30 | 2.08 | 53.02% | 59.46% | 6 | 12 |
| 2 | `Arsenal` | ConvolutionBased | 15/30 | 2.21 | 54.65% | 63.53% | 2 | 12 |
| 3 | `STC` | ShapeletBased | 13/30 | 2.62 | 51.34% | 59.69% | 2 | 12 |
| 4 | `RDST` | ShapeletBased | 13/30 | 3.08 | 49.78% | 58.22% | 2 | 12 |

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

#### Core 90 (30/30/30) (90 problems; 50 common; 47 classifiers &ge;52/65 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `Arsenal` | General TSML | 53/90 | 13.11 | 57.19% | 59.99% | 4 | 18 |
| 2 | `EEGNeX-NB` | EEG Supervised DL | 65/90 | 14.61 | 54.97% | 54.55% | 9 | 21 |
| 3 | `TIDNet-NB` | EEG Supervised DL | 65/90 | 17.50 | 54.03% | 53.23% | 0 | 6 |
| 4 | `TSLR-MOABB` | EEG Classical | 54/90 | 18.44 | 54.55% | 56.27% | 1 | 7 |
| 5 | `EEGInceptionMI-NB` | EEG Supervised DL | 65/90 | 18.47 | 53.61% | 52.69% | 3 | 10 |
| 6 | `CoSpectraLR-NB` | EEG Classical | 54/90 | 18.70 | 53.95% | 55.92% | 3 | 5 |
| 7 | `TSSVM-MOABB` | EEG Classical | 65/90 | 18.72 | 54.68% | 56.97% | 1 | 6 |
| 8 | `XdawnTSLR-NB` | EEG Classical | 53/90 | 19.05 | 53.01% | 53.49% | 2 | 10 |
| 9 | `DE-LR` | EEG Classical | 65/90 | 19.09 | 54.19% | 55.71% | 1 | 8 |
| 10 | `IFNet-NB` | EEG Supervised DL | 65/90 | 19.16 | 53.19% | 52.19% | 0 | 8 |
| 11 | `TSEN-MOABB` | EEG Classical | 65/90 | 19.56 | 54.22% | 55.91% | 2 | 5 |
| 12 | `SPARCNet-NB` | EEG Supervised DL | 54/90 | 21.23 | 51.11% | 50.34% | 0 | 5 |
| 13 | `TSLR-NB` | EEG Classical | 54/90 | 21.44 | 52.60% | 53.94% | 0 | 5 |
| 14 | `EEGConformer-NB` | EEG Supervised DL | 54/90 | 21.56 | 52.19% | 51.34% | 2 | 9 |
| 15 | `FBMSNet-NB` | EEG Supervised DL | 65/90 | 21.70 | 52.66% | 52.18% | 2 | 5 |

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

#### Clinical (30 problems; 14 common; 47 classifiers &ge;15/18 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `EEGNeX-NB` | EEG Supervised DL | 18/30 | 13.93 | 60.64% | 59.55% | 0 | 5 |
| 2 | `TSLR-MOABB` | EEG Classical | 16/30 | 16.79 | 59.06% | 62.93% | 1 | 2 |
| 3 | `DE-SVM` | EEG Classical | 18/30 | 17.36 | 59.75% | 63.21% | 0 | 4 |
| 4 | `TSSVM-MOABB` | EEG Classical | 18/30 | 17.50 | 58.83% | 64.03% | 0 | 1 |
| 5 | `Arsenal` | General TSML | 15/30 | 18.29 | 57.84% | 64.78% | 0 | 3 |
| 6 | `CSP-LDA-MOABB` | EEG Classical | 16/30 | 19.11 | 57.75% | 62.28% | 0 | 1 |
| 7 | `CSP-SVM-MOABB` | EEG Classical | 18/30 | 19.29 | 56.69% | 62.56% | 1 | 2 |
| 8 | `BrainModule-NB` | EEG Supervised DL | 16/30 | 19.39 | 57.18% | 55.02% | 0 | 2 |
| 9 | `ShallowFBCSPNet-NB` | EEG Supervised DL | 16/30 | 19.50 | 55.95% | 52.99% | 0 | 0 |
| 10 | `TSEN-MOABB` | EEG Classical | 18/30 | 19.75 | 57.63% | 61.25% | 1 | 1 |
| 11 | `SincShallowNet-NB` | EEG Supervised DL | 18/30 | 19.75 | 57.41% | 53.64% | 1 | 4 |
| 12 | `IFNet-NB` | EEG Supervised DL | 18/30 | 19.75 | 56.42% | 51.77% | 0 | 2 |
| 13 | `MDM-MOABB` | EEG Classical | 16/30 | 19.79 | 57.13% | 57.33% | 0 | 1 |
| 14 | `CoSpectraLR-NB` | EEG Classical | 16/30 | 19.82 | 56.34% | 61.55% | 0 | 1 |
| 15 | `DLCSP-shLDA-MOABB` | EEG Classical | 18/30 | 19.86 | 57.51% | 62.06% | 0 | 1 |

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

#### All Scalp EEG (114 problems; 65 common; 62 classifiers &ge;72/89 evaluated)

| # | Classifier | Paradigm | Coverage | Mean Rank | Balanced Acc | Accuracy | Wins | Top-5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `Arsenal` | General TSML | 77/114 | 16.20 | 53.72% | 57.89% | 2 | 19 |
| 2 | `EEGNeX-NB` | EEG Supervised DL | 89/114 | 18.02 | 52.26% | 51.27% | 10 | 26 |
| 3 | `STC` | General TSML | 72/114 | 20.54 | 52.37% | 56.88% | 4 | 13 |
| 4 | `QUANT` | General TSML | 74/114 | 21.49 | 52.70% | 56.82% | 3 | 12 |
| 5 | `TIDNet-NB` | EEG Supervised DL | 89/114 | 22.26 | 51.16% | 50.36% | 0 | 7 |
| 6 | `EEGInceptionMI-NB` | EEG Supervised DL | 89/114 | 22.43 | 52.38% | 51.99% | 3 | 10 |
| 7 | `XdawnTSLR-NB` | EEG Classical | 77/114 | 23.18 | 50.90% | 51.49% | 1 | 11 |
| 8 | `RDST` | General TSML | 73/114 | 23.19 | 51.20% | 55.45% | 3 | 10 |
| 9 | `CoSpectraLR-NB` | EEG Classical | 78/114 | 24.17 | 51.56% | 53.24% | 3 | 7 |
| 10 | `SPARCNet-NB` | EEG Supervised DL | 78/114 | 25.45 | 49.07% | 48.07% | 0 | 5 |
| 11 | `TMSANet-NB` | EEG Supervised DL | 74/114 | 25.91 | 49.80% | 47.89% | 1 | 5 |
| 12 | `IFNet-NB` | EEG Supervised DL | 89/114 | 26.22 | 50.60% | 49.26% | 1 | 5 |
| 13 | `TSLR-MOABB` | EEG Classical | 78/114 | 26.76 | 49.95% | 53.34% | 1 | 7 |
| 14 | `TSEN-MOABB` | EEG Classical | 89/114 | 26.92 | 49.79% | 53.22% | 2 | 6 |
| 15 | `DE-LR` | EEG Classical | 89/114 | 27.12 | 49.81% | 53.18% | 0 | 8 |

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
