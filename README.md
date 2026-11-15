# Neuroverse

**Neuroverse** is an open benchmark and interactive leaderboard for electroencephalogram (EEG) time series classification. Its primary benchmark is a symmetric, 100% open-access **90-problem `30 / 30 / 30` split** (**30 Motor Imagery + 30 Clinical + 30 Cognitive** scalp EEG tasks), within an extended **114-problem open scalp EEG study** (`+ 24` Motor Execution, ERP/P300, VEP/SSVEP, and Speech BCI tasks; `125` prepared datasets in total). A curated **12-classifier Headline Leaderboard** spans every time series machine learning paradigm (`HC2`, `Arsenal`, `MRHydra`, `QUANT`, `DrCIF`, `STC`, `RDST`, `TDE`, `Catch22`, `FreshPRINCE`, `InceptionTime`, `RotationForest`) and expands in one click to **81 distinct classifiers** (across **109 configurations** including classical Riemannian/spatial-filtering EEG pipelines, supervised Braindecode neural networks, pretrained EEG foundation models, and band-pass/random-weight controls).

<p align="center">
  <a href="https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/leaderboard.html"><strong>Algorithm Leaderboard</strong></a>
  &middot;
  <a href="https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/matrix.html"><strong>Overall Score Matrix (90 / 114 &times; 81)</strong></a>
  &middot;
  <a href="https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/datasets.html"><strong>Problem Explorer</strong></a>
  &middot;
  <a href="docs/datasets.md">Datasets</a>
  &middot;
  <a href="docs/evaluation.md">Evaluation</a>
  &middot;
  <a href="docs/leaderboard.md">Domain Tables</a>
  &middot;
  <a href="Repositories.md">External Benchmarks</a>
</p>

The original 30 datasets and initial benchmark study are described in [Time Series Machine Learning for Classifying Electroencephalograms](https://data.mlr.press/assets/pdf/v03-8.pdf) (Rushbrooke, Sami, Middlehurst and Bagnall, *Journal of Data-centric Machine Learning Research*, 2026; archived in [`results/neuroverse_paper_2026/`](results/neuroverse_paper_2026/)). If you use the archive or these results, please cite `CITATION.cff`.

### Part of the Multiverse

Neuroverse is part of the [Multiverse](https://github.com/aeon-toolkit/multiverse), the archive for multivariate time series classification described in [The Multiverse of Time Series Machine Learning: an Archive for Multivariate Time Series Classification](https://arxiv.org/abs/2603.20352) (`arXiv:2603.20352`). The published open subset corresponds to `eeg2026` in `aeon`, kept in a dedicated repository because EEG problems have domain-specific montages, protocols, and classifier families.

This repository stands alone (no dependency on the `multiverse` package) while supporting lossless conversion between wide summary matrices (`results/neuroverse/<metric>.csv`) and Multiverse per-estimator directories (`<estimator>/<estimator>_<metric>.csv`).

---

## Leaderboard

<!-- LEADERBOARD:START -->
#### Headline Leaderboard: Core 90 Scalp EEG Benchmark (30 MI + 30 Clinical + 30 Cognitive)

| # | Classifier | Family | Cov. | Mean Rank | Balanced Acc | Accuracy | AUROC | Log Loss &darr; | Wins |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `HC2` | Hybrid | 46/90 | **3.26** | **56.32%** | **58.97%** | **0.6954** | **0.848** | 12 |
| 2 | `Arsenal` | ConvolutionBased | 53/90 | 4.20 | 55.43% | 58.51% | 0.6812 | 4.741 | 4 |
| 3 | `DrCIF` | IntervalBased | 47/90 | 4.66 | 53.47% | 57.11% | 0.6762 | 0.896 | 5 |
| 4 | `MRHydra` | ConvolutionBased | 48/90 | 4.72 | 53.55% | 56.13% | 0.6069 | 15.177 | 8 |
| 5 | `QUANT` | IntervalBased | 50/90 | 4.88 | 53.61% | 57.16% | 0.6606 | 1.309 | 6 |
| 6 | `STC` | ShapeletBased | 50/90 | 5.35 | 53.14% | 56.59% | 0.6636 | 0.881 | 1 |
| 7 | `RDST` | ShapeletBased | 50/90 | 5.58 | 52.39% | 55.84% | 0.5995 | 15.266 | 3 |
| 8 | `TDE` | DictionaryBased | 46/90 | 7.19 | 48.60% | 52.17% | 0.6038 | 1.751 | 2 |
| 9 | `InceptionTime` | DeepLearning | 49/90 | 7.50 | 45.83% | 47.53% | 0.5734 | 3.541 | 2 |
| 10 | `Catch22` | FeatureBased | 46/90 | 7.66 | 47.83% | 51.66% | 0.5944 | 0.914 | 0 |

*10 of 12 headline classifiers meet &ge;80% coverage (44/54 evaluated problems, 90 in Core 90), evaluated strictly on their 43 shared completed problems (resample 0, ordered by mean balanced-accuracy rank). Pending &ge;80% coverage: `FreshPRINCE` (21/90), `RotationForest` (42/90) (visible under *All Available* mode). Expand the interactive leaderboard to compare all **81 distinct classifiers** (109 total configurations) across TSML, EEG Classical, Braindecode DL, and EEG Foundation models.*

#### 30 / 30 / 30 Core Benchmark & Extended BCI Summary

| Benchmark Scope | Problems (Common) | #1 Headline | #2 Headline | #3 Headline | Best EEG-Specific (of 81) |
|---|---:|---|---|---|---|
| **Core 90 (30/30/30)** | 90 (43) | `HC2` (56.3%) | `Arsenal` (55.4%) | `DrCIF` (53.5%) | `EEGNeX-NB` (#2/47, 55.0%) |
| **Motor Imagery** | 30 (25) | `HC2` (58.2%) | `DrCIF` (56.5%) | `Arsenal` (56.9%) | `TIDNet-NB` (#6/76, 55.7%) |
| **Clinical** | 30 (12) | `QUANT` (53.0%) | `Arsenal` (54.7%) | `STC` (51.3%) | `EEGNeX-NB` (#1/47, 60.6%) |
| **Cognitive** | 30 (8) | `RDST` (49.3%) | `Arsenal` (50.9%) | `MRHydra` (49.4%) | `EEGNeX-NB` (#1/29, 44.1%) |
| **All Scalp EEG** | 114 (64) | `HC2` (54.0%) | `Arsenal` (52.2%) | `MRHydra` (51.2%) | `EEGNeX-NB` (#2/62, 52.3%) |
| **BCI (All)** | 54 (44) | `HC2` (54.8%) | `Arsenal` (53.1%) | `DrCIF` (53.1%) | `EEGNeX-NB` (#9/74, 50.5%) |
| **Motor Execution** | 6 (5) | `HC2` (68.3%) | `Arsenal` (67.1%) | `STC` (66.4%) | `TCFormer-NB` (#3/75, 70.3%) |
| **ERP / P300** | 8 (7) | `MRHydra` (61.4%) | `HC2` (58.7%) | `RDST` (57.3%) | `EEGNeX-NB` (#1/67, 67.4%) |
| **VEP / SSVEP** | 6 (5) | `DrCIF` (37.4%) | `QUANT` (32.6%) | `HC2` (36.2%) | `XdawnCov-MDM-MOABB` (#2/74, 53.8%) |
| **Speech BCI** | 4 (4) | `TDE` (35.8%) | `QUANT` (33.4%) | `HC2` (31.0%) | `RiemannianKNN-OAS-BP140` (#6/75, 30.2%) |
<!-- LEADERBOARD:END -->

### Interactive Views

Rebuilt automatically via `python -m neuroverse.experiments.tables`, which generates three self-contained interactive HTML dashboards in [`results/neuroverse/`](results/neuroverse/) (each including an interactive **Download Results** modal in the top navigation bar):

- **[`results/neuroverse/leaderboard.html`](results/neuroverse/leaderboard.html)** ([live preview](https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/leaderboard.html)): Opens on the **12-classifier Headline Leaderboard** across the primary **Core 90 (`30 MI + 30 Clinical + 30 Cognitive`)** benchmark, with one-click expansion to **81 distinct classifiers** or **all 109 configurations**, tabs for every Core and Extended BCI domain (`All 114`, `BCI`, `Motor Execution`, `ERP`, `VEP`, `Speech`), Standard Metrics vs Split-by-Problem-Type views, and Wilcoxon-Holm critical difference diagrams.
- **[`results/neuroverse/matrix.html`](results/neuroverse/matrix.html)** ([live preview](https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/matrix.html)): Interactive **90 (or 114) Problems &times; 12 / 81 / 109 Classifiers** score matrix grouped by problem type and algorithm paradigm, with metric switching (`Balanced Accuracy`, `Accuracy`, `AUROC`, `Log Loss`), row-normalised or baseline-gain heatmap shading, and multi-metric hover tooltips.
- **[`results/neuroverse/datasets.html`](results/neuroverse/datasets.html)** ([live preview](https://raw.githack.com/TonyBagnall/Neuroverse/main/results/neuroverse/datasets.html)): Per-problem explorer comparing the constant test-majority `Default` floor, median, ceiling, winning algorithm, and spread across the Core 90 and Extended 114 tasks.
- Full Headline and Expanded Top-15 Markdown tables per problem type are also written to [`docs/leaderboard.md`](docs/leaderboard.md).

---

## Installation

```bash
pip install neuroverse
```

Or install from source:

```bash
git clone https://github.com/TonyBagnall/Neuroverse.git
cd Neuroverse
pip install -e .
```

## Quick start

### Download & query benchmark results interactively

Results can be queried directly in Python, exported via an interactive CLI wizard (`neuroverse-results`), or downloaded from the **Download Results** button in any dashboard HTML page. When installed standalone via `pip`, `neuroverse` automatically fetches and caches the published CSVs from GitHub:

```python
import neuroverse as nv

# 1. Load the 12 headline classifiers on the primary 30/30/30 (90-problem) split
balacc = nv.load_results(metric="balacc", split="core90", estimators="headline")

# 2. Load a ranked league table or tidy long-form table across all 81 distinct classifiers
league = nv.load_results(metric="balacc", split="MI", estimators="canonical", format="league")
tidy = nv.load_results(metric="all", split="core90", estimators="canonical", format="long")

# 3. Export a filtered directory of wide CSVs, Multiverse folders, or JSON to disk
nv.download_results("my_eeg_results", format="wide", split="core90", estimators="headline")
```

From the terminal, launch the interactive selector or pass flags directly:

```bash
neuroverse-results                                                                          # interactive wizard
neuroverse-results --list                                                                   # list splits & pools
neuroverse-results --split core90 --estimators headline --preview                           # print ranked table
neuroverse-results --split MI --estimators canonical --format long --out mi_results.csv     # save tidy CSV
```

### Explore datasets by problem type

The repository registers 125 problems in `neuroverse.datasets.neuroverse`:
- **`neuroverse_study` / `neuroverse_90`** (`90` primary open-access scalp EEG problems, listed in [`eeg.txt`](eeg.txt)): symmetric **`30 / 30 / 30`** split across **Motor Imagery** (`30`), **Clinical** (`30`), and **Cognitive** (`30`).
- **`neuroverse_114` / `neuroverse_extended`** (`114` open-access scalp EEG problems, listed in [`eeg_114.txt`](eeg_114.txt)): adds the remaining `24` BCI tasks (`ME` `6`, `ERP` `8`, `VEP` `6`, `Speech` `4`), excluding MEG (`4`), ECoG (`1`), restricted-access files (`FibroLiverpool`, `FibroUEA`), and four legacy tasks (`Blink`, `ButtonPress`, `EpilepticSeizures`, `Sleep`).
- **`neuroverse_paper`** (`30` problems): the Rushbrooke et al. (2026) DMLR benchmark (`28` open on Zenodo as `neuroverse_open` / `aeon`'s `eeg2026`, plus `2` restricted fibromyalgia cohorts archived in [`results/neuroverse_paper_2026/`](results/neuroverse_paper_2026/)).

```python
from aeon.datasets import load_classification
from neuroverse.datasets import by_type, metadata, neuroverse_114, neuroverse_study

print(len(neuroverse_study), len(by_type("MI")), len(by_type("Clinical")), len(by_type("Cognitive")))
print(len(neuroverse_114))
print(metadata().loc["ImaginedFeetHands"])

X_train, y_train = load_classification("ImaginedFeetHands", split="train")
X_test, y_test = load_classification("ImaginedFeetHands", split="test")
```

For local preparations (across `D:/Data/EEG/` and overflow storage `C:/Data/EEG/`), load `.ts` files directly via `aeon.datasets.load_from_ts_file`, e.g. `load_from_ts_file("D:/Data/EEG/ImaginedLeftRightHand/ImaginedLeftRightHand_TRAIN.ts")`. Always use the supplied fixed train/test split (`resample 0`); see [`docs/datasets.md`](docs/datasets.md) and [`docs/dataset-provenance.md`](docs/dataset-provenance.md).

### Train and test a classifier

```python
from aeon.classification.convolution_based import MultiRocketHydraClassifier

clf = MultiRocketHydraClassifier(random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

Neuroverse holds benchmark metadata and results only (no classifier implementation code belongs in this repository):
- EEG-specific classifiers (Riemannian pipelines, CSP/xDAWN/CCA/TRCA, Braindecode wrappers, and pretrained EEG foundation models) live in [`aeon-neuro`](https://github.com/aeon-toolkit/aeon-neuro).
- Ports of general multivariate time series classifiers outside `aeon` live in [`multiverse.classification`](https://github.com/aeon-toolkit/multiverse/tree/main/multiverse/classification).
- General time series classifiers (`HC2`, `MRHydra`, `Arsenal`, `QUANT`, `DrCIF`, `STC`, `RDST`, `TDE`, `Catch22`, `FreshPRINCE`, `InceptionTime`, `RotationForest`) live in [`aeon`](https://github.com/aeon-toolkit/aeon).

### Query individual metric series

```python
from neuroverse.experiments.tables import compute_league_table, load_metric

print(load_metric("HC2", "balacc")["ImaginedFeetHands"])
```

### Refresh results and rebuild the dashboard

Results are generated on fixed `testResample0.csv` splits with [`tsml_eval`](https://github.com/time-series-machine-learning/tsml-eval):

```bash
python -m neuroverse.experiments.run_benchmark                        # run tsml-eval experiments
python -m neuroverse.experiments.ingest --wide-from D:/Results/EEG    # export wide summary CSVs
python -m neuroverse.experiments.tables                               # rebuild HTML + Markdown tables
```

See [`docs/evaluation.md`](docs/evaluation.md) for metric definitions, baseline rules, and Multiverse export options.

## Repository layout

```text
Neuroverse/
├── docs/                        # Dataset provenance, taxonomy, evaluation & domain tables
├── results/
│   ├── neuroverse/              # Wide CSVs (81 distinct classifiers, 109 configs; 12 headline) & HTML suite
│   ├── neuroverse_paper_2026/   # Archived 30-problem DMLR 2026 Multiverse CSVs & pages
│   └── tsml-eval-20261111/      # Dated tsml-eval wide snapshot export
├── neuroverse/                  # Python package (metadata, result downloader & leaderboard tooling)
│   ├── datasets.py              # Archive registry (125 total, 90 open Core 30/30/30, 114 open scalp EEG)
│   ├── estimators.py            # Classifier metadata (12 headline -> 81 canonical -> 109 configs)
│   ├── results.py               # Python API & interactive CLI (`neuroverse-results`) result downloader
│   └── experiments/             # Benchmark runners, wide/Multiverse ingest, HTML builders
└── tools/                       # Dataset import, validation & cluster collation utilities
```

---

### Ongoing Archive References

- Dataset conversion handover & live progress: [`DATASET_CONVERSION_HANDOVER.md`](DATASET_CONVERSION_HANDOVER.md) and [`docs/assimilation-progress.json`](docs/assimilation-progress.json).
- Restored motor imagery preparations: [`docs/temp-c-mi-preparations.md`](docs/temp-c-mi-preparations.md).
- Primary 90-problem (`30/30/30`) & extended 114-problem scalp EEG lists: [`eeg.txt`](eeg.txt), [`eeg_114.txt`](eeg_114.txt), and [`docs/problem-taxonomy.md`](docs/problem-taxonomy.md).
- External dataset & algorithm benchmark surveys: [`Repositories.md`](Repositories.md) and [`docs/benchmark_algorithms_survey.md`](docs/benchmark_algorithms_survey.md).
