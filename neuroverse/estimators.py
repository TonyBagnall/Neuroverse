"""The estimators on the leaderboard, and where each comes from.

This is metadata only. Neuroverse holds no classifier code: classifiers designed
for EEG are in aeon-neuro, ports of general multivariate classifiers are in
multiverse.classification, and the rest are in aeon.

An estimator's name here matches its column in ``results/neuroverse/<metric>.csv``
(or its directory name in Multiverse per-estimator layout). Three nested tiers
organise the active benchmark classifiers:

1. :data:`HEADLINE_ESTIMATORS` (12 classifiers): primary TSML + Tabular
   representatives shown by default on the headline 30/30/30 leaderboard
   (``HC2``, ``Arsenal``, ``MRHydra``, ``QUANT``, ``DrCIF``, ``STC``, ``RDST``,
   ``TDE``, ``Catch22``, ``FreshPRINCE``, ``InceptionTime``, ``RotationForest``).
2. :data:`CANONICAL_ESTIMATORS` (81 distinct classifiers): expanded benchmark
   pool spanning General TSML, EEG Classical, EEG Supervised DL, EEG Foundation,
   and Tabular baselines without counting preprocessing or control clones twice.
3. :data:`BAKEOFF_ESTIMATORS` (109 configurations): all evaluated configurations
   including band-pass, covariance-shrinkage, tree-count, linear-probe, and
   random-weight controls (:data:`VARIANT_ESTIMATORS`).
"""

from __future__ import annotations

__all__ = [
    "ESTIMATORS",
    "ESTIMATOR_METADATA",
    "ESTIMATOR_ALIASES",
    "HEADLINE_ESTIMATORS",
    "BAKEOFF_ESTIMATORS",
    "CANONICAL_ESTIMATORS",
    "VARIANT_ESTIMATORS",
    "FAMILIES",
    "PARADIGMS",
    "TSML_FAMILIES",
    "PAPER_ESTIMATORS",
    "family",
    "paradigm",
    "resolve_estimator_name",
    "estimator_info",
    "estimator_table",
    "estimator_markdown",
]

import pandas as pd

#: Legacy family codes retained for compatibility with the 2026 DMLR paper.
FAMILIES = {
    "TSML": "General purpose time series classifier",
    "Deep": "Deep learning, not specific to EEG",
    "EEG": "Classical spatial-filtering or Riemannian EEG pipeline",
    "EEG-DL": "Supervised deep neural architecture for EEG",
    "EEG-FM": "Pretrained EEG foundation model (and random-encoder controls)",
    "Standard": "Tabular classifier with no EEG or time-series specific step",
}

#: High-level bake-off paradigms shown as filter pills and matrix banners.
PARADIGMS = {
    "TSML": "General TSML",
    "EEG-Classical": "EEG Classical",
    "EEG-DL": "EEG Supervised DL",
    "EEG-FM": "EEG Foundation",
    "Tabular": "Tabular",
}

#: ``tsml-eval`` category directories present in the extended EEG study.
TSML_FAMILIES = {
    "Hybrid": "Hybrid ensembles",
    "ConvolutionBased": "Random convolutional kernels",
    "IntervalBased": "Summary and spectral interval forests / quantiles",
    "ShapeletBased": "Phase-independent subsequences (shapelets)",
    "DictionaryBased": "Symbolic Fourier / bag-of-words representations",
    "FeatureBased": "Catch22/TSFresh features, spatial filters, band-power and tangent-space pipelines",
    "DistanceBased": "Riemannian geometry and template-correlation classifiers",
    "DeepLearning": "End-to-end neural networks and foundation backbones",
    "Tabular": "Flattened-channel standard tabular classifiers",
}

#: The 12 headline classifiers shown on the primary 30/30/30 leaderboard.
HEADLINE_ESTIMATORS: list[str] = [
    "HC2",
    "Arsenal",
    "MRHydra",
    "QUANT",
    "DrCIF",
    "STC",
    "RDST",
    "TDE",
    "Catch22",
    "FreshPRINCE",
    "InceptionTime",
    "RotationForest",
]
_HEADLINE_SET: frozenset[str] = frozenset(HEADLINE_ESTIMATORS)

#: Raw ``tsml-eval`` directory names mapped to clean Neuroverse names.
ESTIMATOR_ALIASES: dict[str, str] = {
    "HC2-noTDE-500-pc": "HC2",
    "DrCIF-500": "DrCIF",
    "STC-aeon": "STC",
    "IT": "InceptionTime",
}

# Non-canonical preprocessing, tree-count, frozen-probe, or ablation variants
# among the 109 active configurations (leaving 81 canonical distinct classifiers).
_NON_CANONICAL_BAKEOFF: frozenset[str] = frozenset(
    {
        "HC2-500-pc",
        "DrCIF-200",
        "CBraMod",
        "CBraMod-random",
        "CBraMod-FT-random",
        "LaBraM",
        "LaBraM-random",
        "LaBraM-FT-random",
        "CSP+LDA",
        "CSP+LDA-BP140",
        "CSP+LDA-BP830",
        "CSP+SVM",
        "CSP+SVM-BP140",
        "CSP+SVM-BP830",
        "XDAWN",
        "RiemannianMDM",
        "RiemannianMDM-OAS",
        "RiemannianMDM-OAS-BP140",
        "RiemannianMDM-OAS-BP830",
        "RiemannianMDM-LWF",
        "RiemannianMDM-LWF-BP140",
        "RiemannianMDM-LWF-BP830",
        "RiemannianKNN",
        "RiemannianKNN-OAS",
        "RiemannianKNN-OAS-BP830",
        "RiemannianKNN-LWF",
        "RiemannianKNN-LWF-BP140",
        "RiemannianKNN-LWF-BP830",
    }
)

# Tuple schema:
#   (paradigm, tsml_family, legacy_family, base_method, library, is_ablation, in_bakeoff, description)
_REGISTRY: dict[str, tuple[str, str, str, str, str, bool, bool, str]] = {
    # --- General TSML (15 distinct classifiers; 17 active configurations) ---
    "HC2": (
        "TSML", "Hybrid", "TSML", "HC2", "aeon", False, True,
        "HIVE-COTE v2.0 hybrid ensemble (Arsenal, DrCIF, and STC with prior correction)",
    ),
    "HC2-500-pc": (
        "TSML", "Hybrid", "TSML", "HC2", "aeon", False, True,
        "HIVE-COTE v2.0 variant including TDE dictionary component",
    ),
    "Arsenal": (
        "TSML", "ConvolutionBased", "TSML", "Arsenal", "aeon", False, True,
        "Ensemble of ROCKET convolutional kernel classifiers",
    ),
    "MRHydra": (
        "TSML", "ConvolutionBased", "TSML", "MRHydra", "aeon", False, True,
        "MultiROCKET combined with Hydra competing convolutional dictionary",
    ),
    "MiniROCKET": (
        "TSML", "ConvolutionBased", "TSML", "MiniROCKET", "aeon", False, True,
        "Deterministic MiniROCKET convolutional kernel transform with Ridge classifier",
    ),
    "ROCKET": (
        "TSML", "ConvolutionBased", "TSML", "ROCKET", "aeon", False, True,
        "Random Convolutional Kernel Transform with Ridge classifier",
    ),
    "DrCIF": (
        "TSML", "IntervalBased", "TSML", "DrCIF", "aeon", False, True,
        "Diverse Representation Canonical Interval Forest (500 trees)",
    ),
    "DrCIF-200": (
        "TSML", "IntervalBased", "TSML", "DrCIF", "aeon", False, True,
        "Diverse Representation Canonical Interval Forest (default 200 trees)",
    ),
    "QUANT": (
        "TSML", "IntervalBased", "TSML", "QUANT", "aeon", False, True,
        "Dyadic interval quantile features with ExtraTrees",
    ),
    "STC": (
        "TSML", "ShapeletBased", "TSML", "STC", "aeon", False, True,
        "Randomised Shapelet Transform Classifier with Rotation Forest",
    ),
    "RDST": (
        "TSML", "ShapeletBased", "TSML", "RDST", "aeon", False, True,
        "Random Dilated Shapelet Transform with Ridge classifier",
    ),
    "TDE": (
        "TSML", "DictionaryBased", "TSML", "TDE", "aeon", False, True,
        "Temporal Dictionary Ensemble over SFA word histograms",
    ),
    "Catch22": (
        "TSML", "FeatureBased", "TSML", "Catch22", "aeon", False, True,
        "Canonical 22 time-series summary features per channel with Random Forest",
    ),
    "FreshPRINCE": (
        "TSML", "FeatureBased", "TSML", "FreshPRINCE", "aeon", False, True,
        "TSFresh summary features per channel with Rotation Forest",
    ),
    "CNN": (
        "TSML", "DeepLearning", "Deep", "CNN", "aeon", False, True,
        "Time-CNN multivariate 1D convolutional network",
    ),
    "InceptionTime": (
        "TSML", "DeepLearning", "Deep", "InceptionTime", "aeon", False, True,
        "InceptionTime ensemble of multi-scale 1D Inception networks",
    ),
    "LITETime-MV": (
        "TSML", "DeepLearning", "Deep", "LITETime", "aeon", False, True,
        "Multivariate Light Inception with Boosting and depthwise convolutions",
    ),

    # --- EEG Foundation Models & Controls (2 canonical; 8 active configurations) ---
    "CBraMod-FT": (
        "EEG-FM", "DeepLearning", "EEG-FM", "CBraMod", "aeon-neuro (Foundation)", False, True,
        "Criss-cross Brain Foundation Model end-to-end fine-tuned patch transformer",
    ),
    "CBraMod": (
        "EEG-FM", "DeepLearning", "EEG-FM", "CBraMod", "aeon-neuro (Foundation)", False, True,
        "Criss-cross Brain Foundation Model frozen-encoder linear probe",
    ),
    "CBraMod-FT-random": (
        "EEG-FM", "DeepLearning", "EEG-FM", "CBraMod", "aeon-neuro (Foundation)", True, True,
        "CBraMod fine-tuned from random weights (no pretraining control)",
    ),
    "CBraMod-random": (
        "EEG-FM", "DeepLearning", "EEG-FM", "CBraMod", "aeon-neuro (Foundation)", True, True,
        "CBraMod architecture linear probe from random weights (no pretraining control)",
    ),
    "LaBraM-FT": (
        "EEG-FM", "DeepLearning", "EEG-FM", "LaBraM", "aeon-neuro (Foundation)", False, True,
        "Large Brain Model end-to-end fine-tuned neural tokeniser transformer (10-20 montage)",
    ),
    "LaBraM": (
        "EEG-FM", "DeepLearning", "EEG-FM", "LaBraM", "aeon-neuro (Foundation)", False, True,
        "Large Brain Model frozen-encoder linear probe (10-20 montage)",
    ),
    "LaBraM-FT-random": (
        "EEG-FM", "DeepLearning", "EEG-FM", "LaBraM", "aeon-neuro (Foundation)", True, True,
        "LaBraM fine-tuned from random weights (no pretraining control)",
    ),
    "LaBraM-random": (
        "EEG-FM", "DeepLearning", "EEG-FM", "LaBraM", "aeon-neuro (Foundation)", True, True,
        "LaBraM architecture linear probe from random weights (no pretraining control)",
    ),

    # --- EEG Supervised Deep Learning: Braindecode / aeon-neuro (33 active) ---
    "ATCNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "ATCNet", "aeon-neuro (Braindecode)", False, True,
        "Attention Temporal Convolutional Network for EEG decoding",
    ),
    "AttentionBaseNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "AttentionBaseNet", "aeon-neuro (Braindecode)", False, True,
        "Channel-and-temporal attention convolutional baseline network",
    ),
    "BDTCN-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "BDTCN", "aeon-neuro (Braindecode)", False, True,
        "Braindecode dilated Temporal Convolutional Network",
    ),
    "BrainModule-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "BrainModule", "aeon-neuro (Braindecode)", False, True,
        "Meta AI wav2vec-style dilated convolutional EEG encoder",
    ),
    "ContraWR-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "ContraWR", "aeon-neuro (Braindecode)", False, True,
        "Contrastive world-representation STFT spectrogram 2D ResNet",
    ),
    "CTNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "CTNet", "aeon-neuro (Braindecode)", False, True,
        "Convolutional Transformer Network for motor imagery EEG",
    ),
    "Deep4Net-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "Deep4Net", "aeon-neuro (Braindecode)", False, True,
        "Schirrmeister et al. 4-block spatio-temporal ConvNet (DeepConvNet)",
    ),
    "EEGConformer-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGConformer", "aeon-neuro (Braindecode)", False, True,
        "Compact spatio-temporal CNN + self-attention Transformer (Song et al.)",
    ),
    "EEGInceptionERP-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGInceptionERP", "aeon-neuro (Braindecode)", False, True,
        "Multi-scale Inception CNN tailored to event-related potentials",
    ),
    "EEGInceptionMI-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGInceptionMI", "aeon-neuro (Braindecode)", False, True,
        "Multi-scale Inception CNN tailored to motor imagery rhythms",
    ),
    "EEGITNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGITNet", "aeon-neuro (Braindecode)", False, True,
        "Inception + dilated Temporal Convolutional Network for EEG",
    ),
    "EEGNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGNet", "aeon-neuro (Braindecode)", False, True,
        "Lawhern et al. compact depthwise-separable spatio-temporal CNN",
    ),
    "EEGNeX-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGNeX", "aeon-neuro (Braindecode)", False, True,
        "Pure ConvNeXt-inspired dilated spatio-temporal ConvNet (Chen et al.)",
    ),
    "EEGSimpleConv-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGSimpleConv", "aeon-neuro (Braindecode)", False, True,
        "Lightweight 1D batch-normalised ResNet baseline for EEG",
    ),
    "EEGTCNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "EEGTCNet", "aeon-neuro (Braindecode)", False, True,
        "EEGNet front-end coupled to a dilated Temporal Convolutional Network",
    ),
    "FBCNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "FBCNet", "aeon-neuro (Braindecode)", False, True,
        "Filter-Bank Convolutional Network with variance temporal pooling",
    ),
    "FBLightConvNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "FBLightConvNet", "aeon-neuro (Braindecode)", False, True,
        "Filter-bank lightweight depthwise group-convolutional network",
    ),
    "FBMSNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "FBMSNet", "aeon-neuro (Braindecode)", False, True,
        "Filter-Bank Multi-Scale spatio-temporal CNN",
    ),
    "GCNsNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "GCNsNet", "aeon-neuro (Braindecode)", False, True,
        "Spectral Graph Convolutional Network over electrode topology",
    ),
    "IFNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "IFNet", "aeon-neuro (Braindecode)", False, True,
        "Interactive Frequency Convolutional Network (low/high band interplay)",
    ),
    "MSCFormer-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "MSCFormer", "aeon-neuro (Braindecode)", False, True,
        "Multi-scale convolutional Transformer for EEG classification",
    ),
    "MSVTNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "MSVTNet", "aeon-neuro (Braindecode)", False, True,
        "Multi-Scale Vision Transformer network for spatio-temporal EEG",
    ),
    "SCCNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SCCNet", "aeon-neuro (Braindecode)", False, True,
        "Spatial-Component-wise Convolutional Network",
    ),
    "ShallowFBCSPNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "ShallowFBCSPNet", "aeon-neuro (Braindecode)", False, True,
        "Schirrmeister et al. shallow ConvNet mimicking log-bandpower FBCSP",
    ),
    "SincShallowNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SincShallowNet", "aeon-neuro (Braindecode)", False, True,
        "Interpretable SincNet band-pass filter front-end + shallow spatial CNN",
    ),
    "SleepStagerChambon2018-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SleepStagerChambon2018", "aeon-neuro (Braindecode)", False, True,
        "Chambon et al. spatial-temporal ConvNet originally designed for PSG staging",
    ),
    "SPARCNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SPARCNet", "aeon-neuro (Braindecode)", False, True,
        "DenseNet-1D multi-layer dilated convolutional network (Jing et al.)",
    ),
    "SSTDPN-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SSTDPN", "aeon-neuro (Braindecode)", False, True,
        "Spatial-Spectral-Temporal Dual-Prototype Network",
    ),
    "SyncNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "SyncNet", "aeon-neuro (Braindecode)", False, True,
        "Synchronised Gabor-wavelet filter-bank CNN for oscillatory phase/power",
    ),
    "TCFormer-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "TCFormer", "aeon-neuro (Braindecode)", False, True,
        "Temporal Convolutional Transformer hybrid network",
    ),
    "TIDNet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "TIDNet", "aeon-neuro (Braindecode)", False, True,
        "Thinker-Invariant DenseNet for cross-subject EEG decoding",
    ),
    "TMSANet-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "TMSANet", "aeon-neuro (Braindecode)", False, True,
        "Temporal Multi-Scale Attention Network",
    ),
    "TSception-NB": (
        "EEG-DL", "DeepLearning", "EEG-DL", "TSception", "aeon-neuro (Braindecode)", False, True,
        "Multi-scale temporal and hemispheric asymmetry CNN for affective/BCI EEG",
    ),

    # --- EEG Classical: FeatureBased (18 canonical; 25 active configurations) ---
    "AUG-TANG-SVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "AugmentedTangentSVM", "aeon-neuro (MOABB)", False, True,
        "Augmented lag-embedded covariance tangent space + linear/RBF SVM",
    ),
    "CoSpectraLR-NB": (
        "EEG-Classical", "FeatureBased", "EEG", "CoSpectraLR", "aeon-neuro (NeuralBench)", False, True,
        "NeuralBench Welch cross-spectral co-spectra features + LogisticRegressionCV",
    ),
    "CSP+LDA": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+LDA", "aeon-neuro (MNE + sklearn)", False, True,
        "Common Spatial Patterns (unfiltered) + Linear Discriminant Analysis",
    ),
    "CSP+LDA-BP140": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+LDA", "aeon-neuro (MNE + sklearn)", False, True,
        "1-40 Hz broadband pass + Common Spatial Patterns + LDA",
    ),
    "CSP+LDA-BP830": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+LDA", "aeon-neuro (MNE + sklearn)", False, True,
        "8-30 Hz mu/beta pass + Common Spatial Patterns + LDA",
    ),
    "CSP+SVM": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+SVM", "aeon-neuro (MNE + sklearn)", False, True,
        "Common Spatial Patterns (unfiltered) + Support Vector Machine",
    ),
    "CSP+SVM-BP140": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+SVM", "aeon-neuro (MNE + sklearn)", False, True,
        "1-40 Hz broadband pass + Common Spatial Patterns + SVM",
    ),
    "CSP+SVM-BP830": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+SVM", "aeon-neuro (MNE + sklearn)", False, True,
        "8-30 Hz mu/beta pass + Common Spatial Patterns + SVM",
    ),
    "CSP-LDA-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+LDA", "aeon-neuro (MOABB)", False, True,
        "MOABB standard Common Spatial Patterns + shrinkage LDA pipeline",
    ),
    "CSP-SVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+SVM", "aeon-neuro (MOABB)", False, True,
        "MOABB standard Common Spatial Patterns + SVM pipeline",
    ),
    "DE-LR": (
        "EEG-Classical", "FeatureBased", "EEG", "DE+LR", "aeon-neuro", False, True,
        "Canonical sub-band Differential Entropy spectral features + Logistic Regression",
    ),
    "DE-SVM": (
        "EEG-Classical", "FeatureBased", "EEG", "DE+SVM", "aeon-neuro", False, True,
        "Canonical sub-band Differential Entropy spectral features + RBF SVM",
    ),
    "DLCSP-shLDA-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "DLCSP+LDA", "aeon-neuro (MOABB)", False, True,
        "Diagonal-Loading regularised CSP + shrinkage LDA",
    ),
    "FBCSP-SVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "FBCSP+SVM", "aeon-neuro (MOABB)", False, True,
        "Filter-Bank Common Spatial Patterns + mutual-information selection + SVM",
    ),
    "LogVar-LDA-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "LogVar+LDA", "aeon-neuro (MOABB)", False, True,
        "Per-channel log-variance features + shrinkage LDA",
    ),
    "LogVar-SVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "LogVar+SVM", "aeon-neuro (MOABB)", False, True,
        "Per-channel log-variance features + SVM",
    ),
    "TRCSP-LDA-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "TRCSP+LDA", "aeon-neuro (MOABB)", False, True,
        "Tikhonov-Regularised Common Spatial Patterns + LDA (binary tasks)",
    ),
    "TSEN-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "TSEN-MOABB", "aeon-neuro (MOABB)", False, True,
        "Riemannian covariance tangent-space projection + ElasticNet logistic regression",
    ),
    "TSLR-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "TSLR-MOABB", "aeon-neuro (MOABB)", False, True,
        "MOABB Riemannian covariance tangent-space projection + L2 Logistic Regression",
    ),
    "TSLR-NB": (
        "EEG-Classical", "FeatureBased", "EEG", "TSLR-NB", "aeon-neuro (NeuralBench)", False, True,
        "NeuralBench CovTsLR baseline (OAS covariance + tangent space + LogisticRegressionCV)",
    ),
    "TSSVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "TSSVM-MOABB", "aeon-neuro (MOABB)", False, True,
        "Riemannian covariance tangent-space projection + linear SVM",
    ),
    "XDAWN": (
        "EEG-Classical", "FeatureBased", "EEG", "xDAWN+LDA", "aeon-neuro (MNE + sklearn)", False, True,
        "xDAWN evoked-potential spatial filtering + vectorised LDA",
    ),
    "xDAWN-LDA-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "xDAWN+LDA", "aeon-neuro (MOABB)", False, True,
        "MOABB xDAWN spatial filtering + shrinkage LDA",
    ),
    "XdawnCov-TS-SVM-MOABB": (
        "EEG-Classical", "FeatureBased", "EEG", "xDAWNCov+TS-SVM", "aeon-neuro (MOABB)", False, True,
        "xDAWN template-augmented block covariance + tangent-space SVM",
    ),
    "XdawnTSLR-NB": (
        "EEG-Classical", "FeatureBased", "EEG", "xDAWN+TSLR", "aeon-neuro (NeuralBench)", False, True,
        "NeuralBench xDAWN spatial filtering + covariance tangent-space LogisticRegressionCV",
    ),

    # --- EEG Classical: DistanceBased (9 canonical; 22 active configurations) ---
    "ERPCov-MDM-MOABB": (
        "EEG-Classical", "DistanceBased", "EEG", "ERPCov+MDM", "aeon-neuro (MOABB)", False, True,
        "Class-prototype super-trial ERP covariance + Riemannian MDM",
    ),
    "FBTRCA": (
        "EEG-Classical", "DistanceBased", "EEG", "FBTRCA", "aeon-neuro", False, True,
        "Filter-Bank Task-Related Component Analysis spatial filter ensemble",
    ),
    "FgMDM-MOABB": (
        "EEG-Classical", "DistanceBased", "EEG", "FgMDM", "aeon-neuro (MOABB)", False, True,
        "Fisher geodesic discriminant filtering in tangent space + Riemannian MDM",
    ),
    "MDM-MOABB": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (MOABB)", False, True,
        "MOABB Riemannian Minimum Distance to Mean over SPD covariances",
    ),
    "RiemannianKNN": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "k-Nearest Neighbours under affine-invariant Riemannian SPD distance",
    ),
    "RiemannianKNN-LWF": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "Riemannian k-NN with Ledoit-Wolf shrinkage covariances",
    ),
    "RiemannianKNN-LWF-BP140": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "1-40 Hz pass + Ledoit-Wolf covariance + Riemannian k-NN",
    ),
    "RiemannianKNN-LWF-BP830": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "8-30 Hz mu/beta pass + Ledoit-Wolf covariance + Riemannian k-NN",
    ),
    "RiemannianKNN-OAS": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "Riemannian k-NN with Oracle Approximating Shrinkage covariances",
    ),
    "RiemannianKNN-OAS-BP140": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "1-40 Hz pass + OAS covariance + Riemannian k-NN",
    ),
    "RiemannianKNN-OAS-BP830": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "aeon-neuro (PyRiemann)", False, True,
        "8-30 Hz mu/beta pass + OAS covariance + Riemannian k-NN",
    ),
    "RiemannianMDM": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "Riemannian Minimum Distance to Frechet Mean over sample covariances",
    ),
    "RiemannianMDM-LWF": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "Riemannian MDM with Ledoit-Wolf shrinkage covariances",
    ),
    "RiemannianMDM-LWF-BP140": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "1-40 Hz pass + Ledoit-Wolf covariance + Riemannian MDM",
    ),
    "RiemannianMDM-LWF-BP830": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "8-30 Hz mu/beta pass + Ledoit-Wolf covariance + Riemannian MDM",
    ),
    "RiemannianMDM-OAS": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "Riemannian MDM with Oracle Approximating Shrinkage covariances",
    ),
    "RiemannianMDM-OAS-BP140": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "1-40 Hz pass + OAS covariance + Riemannian MDM",
    ),
    "RiemannianMDM-OAS-BP830": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "aeon-neuro (PyRiemann)", False, True,
        "8-30 Hz mu/beta pass + OAS covariance + Riemannian MDM",
    ),
    "TRACE": (
        "EEG-Classical", "DistanceBased", "EEG", "TRACE", "aeon-neuro", False, True,
        "Temporal Riemannian Adaptive Classification Ensemble over weighted intervals",
    ),
    "TemplateCCA-BP640": (
        "EEG-Classical", "DistanceBased", "EEG", "TemplateCCA", "aeon-neuro", False, True,
        "6-40 Hz pass + Individual Template Canonical Correlation Analysis",
    ),
    "TRCA-BP640": (
        "EEG-Classical", "DistanceBased", "EEG", "TRCA", "aeon-neuro", False, True,
        "6-40 Hz pass + Task-Related Component Analysis spatial filter correlation",
    ),
    "XdawnCov-MDM-MOABB": (
        "EEG-Classical", "DistanceBased", "EEG", "xDAWNCov+MDM", "aeon-neuro (MOABB)", False, True,
        "xDAWN template-augmented block covariance + Riemannian MDM",
    ),

    # --- Tabular Baselines (4 active) ---
    "LogisticRegression": (
        "Tabular", "Tabular", "Standard", "LogisticRegression", "scikit-learn", False, True,
        "L2-regularised Logistic Regression on concatenated channel values",
    ),
    "RandomForest": (
        "Tabular", "Tabular", "Standard", "RandomForest", "scikit-learn", False, True,
        "Random Forest on concatenated channel time points",
    ),
    "RotationForest": (
        "Tabular", "Tabular", "Standard", "RotationForest", "aeon", False, True,
        "Rotation Forest (PCA feature-subset rotations + decision trees)",
    ),
    "SVM": (
        "Tabular", "Tabular", "Standard", "SVM", "scikit-learn", False, True,
        "RBF-kernel Support Vector Machine on concatenated channel values",
    ),

    # --- Legacy 2026 DMLR Paper Identifiers (retained for paper reproduction) ---
    "IT": (
        "TSML", "DeepLearning", "Deep", "InceptionTime", "aeon", False, False,
        "InceptionTime (2026 DMLR paper identifier; active name is InceptionTime)",
    ),
    "EEGNet": (
        "EEG-DL", "DeepLearning", "EEG", "EEGNet", "Braindecode", False, False,
        "EEGNet (2026 DMLR paper configuration)",
    ),
    "DCN": (
        "EEG-DL", "DeepLearning", "EEG", "Deep4Net", "Braindecode", False, False,
        "DeepConvNet (2026 DMLR paper configuration)",
    ),
    "CSP-SVM": (
        "EEG-Classical", "FeatureBased", "EEG", "CSP+SVM", "MNE + scikit-learn", False, False,
        "Common Spatial Patterns then an SVM (2026 paper)",
    ),
    "R-MDM": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianMDM", "pyriemann", False, False,
        "Riemannian minimum distance to mean (2026 paper)",
    ),
    "R-KNN": (
        "EEG-Classical", "DistanceBased", "EEG", "RiemannianKNN", "pyriemann", False, False,
        "Riemannian nearest neighbour (2026 paper)",
    ),
}

#: Structured metadata dictionary for every registered estimator.
ESTIMATOR_METADATA: dict[str, dict[str, object]] = {
    name: {
        "paradigm": par,
        "paradigm_label": PARADIGMS[par],
        "tsml_family": tsml_fam,
        "family": leg_fam,
        "base_method": base,
        "library": lib,
        "is_ablation": is_abl,
        "is_headline": bool(in_bo and name in _HEADLINE_SET),
        "is_canonical": bool(in_bo and name not in _NON_CANONICAL_BAKEOFF),
        "in_bakeoff": in_bo,
        "description": desc,
    }
    for name, (par, tsml_fam, leg_fam, base, lib, is_abl, in_bo, desc) in _REGISTRY.items()
}

#: Backwards-compatible ``name -> (family, library, description)`` mapping.
ESTIMATORS: dict[str, tuple[str, str, str]] = {
    name: (leg_fam, lib, desc)
    for name, (_, _, leg_fam, _, lib, _, _, desc) in _REGISTRY.items()
}

#: All 109 classifier configurations stored in the wide result CSVs.
BAKEOFF_ESTIMATORS: list[str] = sorted(
    (name for name, meta in ESTIMATOR_METADATA.items() if meta["in_bakeoff"]),
    key=str.lower,
)

#: The 81 distinct canonical classifiers ranked on the expanded leaderboard.
CANONICAL_ESTIMATORS: list[str] = sorted(
    (name for name, meta in ESTIMATOR_METADATA.items() if meta["is_canonical"]),
    key=str.lower,
)

#: The 28 preprocessing, tree-count, probe, and random-weight ablation variants.
VARIANT_ESTIMATORS: list[str] = sorted(
    (
        name
        for name, meta in ESTIMATOR_METADATA.items()
        if meta["in_bakeoff"] and not meta["is_canonical"]
    ),
    key=str.lower,
)

#: The eight classifiers compared in the 2026 DMLR paper, in the order of Table 2.
PAPER_ESTIMATORS = ["R-MDM", "CSP-SVM", "CNN", "IT", "EEGNet", "DCN", "HC2", "MRHydra"]


def resolve_estimator_name(name: str) -> str:
    """Resolve raw ``tsml-eval`` estimator aliases to their canonical Neuroverse name."""
    return ESTIMATOR_ALIASES.get(name, name)


def family(name: str) -> str:
    """Return the legacy family code of an estimator, or ``""`` if unlisted."""
    if name in ESTIMATORS:
        return ESTIMATORS[name][0]
    resolved = resolve_estimator_name(name)
    return ESTIMATORS[resolved][0] if resolved in ESTIMATORS else ""


def paradigm(name: str) -> str:
    """Return the bake-off paradigm code of an estimator, or ``""`` if unlisted."""
    info = ESTIMATOR_METADATA.get(name) or ESTIMATOR_METADATA.get(resolve_estimator_name(name))
    return str(info["paradigm"]) if info is not None else ""


def estimator_info(name: str) -> dict[str, object]:
    """Return structured metadata for ``name``, with sensible fallback for custom runs."""
    if name in ESTIMATOR_METADATA:
        return dict(ESTIMATOR_METADATA[name])
    resolved = resolve_estimator_name(name)
    if resolved in ESTIMATOR_METADATA:
        return dict(ESTIMATOR_METADATA[resolved])
    if name == "Default":
        return {
            "paradigm": "Baseline",
            "paradigm_label": "Baseline",
            "tsml_family": "Baseline",
            "family": "Standard",
            "base_method": "Default",
            "library": "tsml-eval",
            "is_ablation": False,
            "is_headline": False,
            "is_canonical": False,
            "in_bakeoff": False,
            "description": "Constant majority-class test floor",
        }
    return {
        "paradigm": "Other",
        "paradigm_label": "Other",
        "tsml_family": "Custom",
        "family": "",
        "base_method": name,
        "library": "custom",
        "is_ablation": name.endswith("-random"),
        "is_headline": False,
        "is_canonical": not name.endswith("-random"),
        "in_bakeoff": False,
        "description": name,
    }


def estimator_table(active_only: bool = False) -> pd.DataFrame:
    """Return the estimator metadata, one row per estimator."""
    records = {
        name: meta
        for name, meta in ESTIMATOR_METADATA.items()
        if not active_only or bool(meta["in_bakeoff"])
    }
    frame = pd.DataFrame.from_dict(records, orient="index")
    frame.index.name = "estimator"
    return frame


def estimator_markdown(active_only: bool = True) -> str:
    """Return the estimator metadata as a Markdown table, for the documentation."""
    header = ["Estimator", "Role", "Paradigm", "tsml-eval Family", "Library", "Description"]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    names = BAKEOFF_ESTIMATORS if active_only else list(ESTIMATOR_METADATA)
    n_canon = len(CANONICAL_ESTIMATORS)
    n_head = len(HEADLINE_ESTIMATORS)
    for name in names:
        meta = ESTIMATOR_METADATA[name]
        if meta["is_headline"]:
            role = f"Headline ({n_head})"
        elif meta["is_canonical"]:
            role = f"Expanded ({n_canon})"
        elif meta["is_ablation"]:
            role = "Ablation / Control"
        else:
            role = "Preprocessing / Probe Variant"
        rows.append(
            f"| `{name}` | {role} | {meta['paradigm_label']} | {meta['tsml_family']} "
            f"| {meta['library']} | {meta['description']} |"
        )
    return "\n".join(rows)
