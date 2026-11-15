# EEG league tables: 11 November 2026

Primary metric: test balanced accuracy; accuracy is shown alongside it. Metrics are refreshed from raw testResample0 files using tsml-eval local/summarise_results.py.

Each primary table admits classifiers with results on at least80% of that application's problems, then ranks every admitted classifier on the same complete common-problem block. Mean ranks use average ties; problems receive equal weight. Wins include ties. Ranks are comparable within a table, not between application tables.

All76selected scalp EEG problems define the overall scope. MotorImagery has29problems; all evoked potentials has13 (P300/ERP, SSVEP, cVEP and N400); Clinical has17clinical/physiological problems. Other tasks remain in Overall. Competition results, MEG/ECoG and excluded archive problems are outside these tables.

Case-count mismatches against the archive registry and minority true-label row sequences are excluded and listed in the CSV audit. A matching label sequence does not prove identical signals or subject splits. This is a provisional resample0 comparison, not an independent-study analysis, a significance test or a complete benchmark. Foundation-model pretraining overlap and random-encoder ablations remain explicit result identities.

## Overall

77classifiers; 41common problems from 76in scope. Minimum coverage: 61.

| Rank | Classifier | Coverage | Mean rank | Balanced accuracy | Accuracy | Wins |
|---:|---|---:|---:|---:|---:|---:|
|1|ConvolutionBased/Arsenal|73|20.87|51.46%|53.57%|2|
|2|ConvolutionBased/MRHydra|69|20.91|50.91%|52.49%|4|
|3|IntervalBased/DrCIF|69|21.39|50.68%|53.39%|0|
|4|IntervalBased/DrCIF-500|70|22.65|50.95%|53.66%|3|
|5|ShapeletBased/STC-aeon|71|23.12|50.94%|53.34%|2|
|6|FeatureBased/DE-LR|71|27.70|48.53%|49.72%|1|
|7|DeepLearning/EEGNeX-NB|71|27.79|47.18%|47.40%|4|
|8|FeatureBased/TSLR-MOABB|71|30.66|47.58%|48.93%|0|
|9|ShapeletBased/RDST|66|31.24|48.09%|50.22%|1|
|10|DeepLearning/TIDNet-NB|71|31.57|47.09%|47.05%|0|
|11|FeatureBased/TSEN-MOABB|71|31.60|46.88%|48.16%|0|
|12|DeepLearning/EEGInceptionMI-NB|71|31.74|48.51%|48.31%|1|
|13|FeatureBased/TSSVM-MOABB|71|31.89|46.98%|49.45%|1|
|14|DeepLearning/FBMSNet-NB|71|32.37|46.11%|45.72%|1|
|15|FeatureBased/DLCSP-shLDA-MOABB|71|32.55|47.42%|49.59%|0|
|16|DistanceBased/FgMDM-MOABB|71|33.02|46.46%|47.83%|0|
|17|DictionaryBased/TDE|63|33.15|47.55%|50.27%|1|
|18|DeepLearning/SPARCNet-NB|71|33.24|45.04%|44.56%|0|
|19|FeatureBased/CSP-LDA-MOABB|71|33.44|47.35%|49.56%|0|
|20|DeepLearning/SincShallowNet-NB|71|33.51|44.95%|44.01%|0|
|21|FeatureBased/CSP+LDA-BP830|70|33.61|47.29%|48.93%|1|
|22|DistanceBased/RiemannianMDM-OAS-BP830|71|33.65|46.95%|47.41%|1|
|23|DeepLearning/Deep4Net-NB|71|33.91|46.37%|45.64%|1|
|24|FeatureBased/XdawnCov-TS-SVM-MOABB|69|34.35|47.09%|48.89%|0|
|25|FeatureBased/DE-SVM|71|34.73|46.29%|48.35%|0|
|26|DeepLearning/TMSANet-NB|71|34.95|46.39%|46.51%|1|
|27|DistanceBased/MDM-MOABB|71|35.34|46.54%|47.05%|0|
|28|FeatureBased/CSP+SVM-BP830|70|35.39|46.74%|49.29%|1|
|29|DeepLearning/EEGConformer-NB|71|35.68|45.53%|45.82%|1|
|30|FeatureBased/FBCSP-SVM-MOABB|70|35.84|47.31%|49.40%|1|
|31|DistanceBased/ERPCov-MDM-MOABB|71|35.93|47.98%|48.63%|1|
|32|FeatureBased/CSP-SVM-MOABB|71|36.12|46.75%|49.31%|2|
|33|FeatureBased/LogVar-LDA-MOABB|71|37.04|44.14%|45.37%|0|
|34|DeepLearning/IFNet-NB|71|37.17|46.33%|46.68%|0|
|35|DeepLearning/FBCNet-NB|71|37.17|44.43%|46.35%|0|
|36|DeepLearning/IT|74|37.70|45.39%|46.37%|1|
|37|DeepLearning/CTNet-NB|71|38.71|44.16%|45.12%|0|
|38|FeatureBased/TSLR-NB|71|39.01|44.18%|45.54%|0|
|39|FeatureBased/CSP+SVM-BP140|70|39.20|45.75%|48.28%|1|
|40|DeepLearning/ShallowFBCSPNet-NB|71|39.60|43.45%|42.73%|0|
|41|DeepLearning/AttentionBaseNet-NB|71|39.93|44.09%|43.47%|0|
|42|DistanceBased/XdawnCov-MDM-MOABB|71|40.18|47.57%|47.98%|1|
|43|DistanceBased/RiemannianMDM-OAS-BP140|71|40.79|45.80%|46.05%|2|
|44|DeepLearning/ATCNet-NB|71|40.82|43.91%|43.37%|0|
|45|DeepLearning/BrainModule-NB|71|40.82|43.45%|43.23%|0|
|46|DeepLearning/CBraMod|72|41.01|42.63%|44.50%|0|
|47|DeepLearning/TCFormer-NB|71|41.13|44.15%|44.25%|2|
|48|DistanceBased/RiemannianMDM-OAS|67|41.20|45.31%|45.63%|1|
|49|DeepLearning/EEGSimpleConv-NB|71|41.56|43.63%|45.26%|0|
|50|DeepLearning/CBraMod-random|72|41.61|42.77%|44.61%|0|
|51|DeepLearning/SleepStagerChambon2018-NB|62|41.71|42.94%|43.07%|0|
|52|DistanceBased/RiemannianMDM|64|41.77|44.37%|44.89%|0|
|53|DistanceBased/RiemannianKNN-OAS-BP830|71|42.45|44.51%|45.52%|1|
|54|FeatureBased/LogVar-SVM-MOABB|70|42.52|43.86%|46.55%|0|
|55|FeatureBased/CSP+LDA-BP140|70|42.54|44.79%|47.08%|0|
|56|DistanceBased/RiemannianKNN-OAS-BP140|71|42.60|43.86%|44.42%|0|
|57|DeepLearning/EEGNet-NB|71|42.76|43.15%|43.05%|0|
|58|DeepLearning/SCCNet-NB|71|43.84|42.57%|43.01%|0|
|59|Tabular/SVM|63|44.21|42.20%|45.84%|0|
|60|DeepLearning/CNN|76|45.45|42.54%|44.92%|0|
|61|DeepLearning/EEGITNet-NB|71|46.62|41.66%|40.61%|0|
|62|Tabular/RandomForest|67|46.85|41.33%|43.98%|0|
|63|DeepLearning/MSCFormer-NB|71|47.39|41.07%|41.64%|0|
|64|FeatureBased/CSP+LDA|66|47.41|41.89%|44.76%|1|
|65|Tabular/LogisticRegression|67|48.23|41.33%|43.40%|0|
|66|DistanceBased/TemplateCCA-BP640|71|48.74|41.84%|42.09%|0|
|67|DistanceBased/TRCA-BP640|71|49.70|42.89%|42.70%|1|
|68|FeatureBased/xDAWN-LDA-MOABB|70|50.57|41.22%|43.08%|0|
|69|DeepLearning/ContraWR-NB|71|50.63|41.10%|40.93%|1|
|70|DeepLearning/GCNsNet-NB|70|50.77|39.86%|38.66%|0|
|71|DeepLearning/EEGTCNet-NB|71|50.79|40.94%|41.41%|0|
|72|DeepLearning/MSVTNet-NB|71|51.61|41.20%|40.66%|0|
|73|FeatureBased/CSP+SVM|61|51.82|40.96%|43.49%|0|
|74|DistanceBased/RiemannianKNN-OAS|65|52.10|40.34%|41.58%|0|
|75|DeepLearning/BDTCN-NB|71|53.05|39.43%|37.80%|0|
|76|DistanceBased/RiemannianKNN|65|53.06|40.84%|42.73%|1|
|77|DeepLearning/SyncNet-NB|71|55.24|38.58%|39.11%|0|


Common problems: Alzheimers, ArmCoordination, ArmMovements, CodedFlickerTargets, FeetHands, FibroLiverpool, FibroUEA, FlickerAttention, FlickerTargets, ImaginedAnkleRest, ImaginedArmMovements, ImaginedCommands, ImaginedFeetHands, ImaginedGripForce, ImaginedHandFeetTongue, ImaginedHandGestures, ImaginedHandsFeet, ImaginedHandsStroke, ImaginedLeftRightHand, ImaginedLeftRightHandRest, ImaginedOpenCloseFist, ImaginedRightHandFeet, ImaginedRightHandFeetRest, InnerSpeech, LongIntervalTask, LowCost, MatchingPennies, MentalArithmetic, OpenCloseFist, ParkinsonsPedalling, ParkinsonsRest, PhotoStimulation, PronouncedSpeech, RestingEyeState, SelfRegulationSCP1, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech, WirelessFlicker.

## MotorImagery

71classifiers; 18common problems from 29in scope. Minimum coverage: 24.

| Rank | Classifier | Coverage | Mean rank | Balanced accuracy | Accuracy | Wins |
|---:|---|---:|---:|---:|---:|---:|
|1|IntervalBased/DrCIF|24|18.39|58.81%|61.14%|1|
|2|ConvolutionBased/MRHydra|28|18.97|57.39%|58.38%|2|
|3|IntervalBased/DrCIF-500|28|19.64|58.86%|61.20%|5|
|4|ConvolutionBased/Arsenal|28|20.42|58.37%|59.36%|1|
|5|ShapeletBased/STC-aeon|27|23.25|57.12%|59.08%|0|
|6|DeepLearning/TIDNet-NB|29|23.86|53.92%|53.47%|0|
|7|FeatureBased/TSLR-MOABB|29|23.89|56.25%|57.41%|0|
|8|FeatureBased/DLCSP-shLDA-MOABB|29|24.11|56.09%|58.18%|0|
|9|FeatureBased/DE-LR|29|24.72|55.17%|55.81%|0|
|10|DistanceBased/FgMDM-MOABB|29|25.03|55.65%|56.58%|0|
|11|FeatureBased/TSEN-MOABB|29|25.64|56.02%|57.25%|1|
|12|FeatureBased/CSP+LDA-BP830|28|25.94|55.77%|56.88%|0|
|13|FeatureBased/TSSVM-MOABB|29|26.11|55.12%|57.42%|0|
|14|FeatureBased/CSP-LDA-MOABB|29|27.36|55.60%|57.77%|0|
|15|DeepLearning/FBMSNet-NB|29|27.97|53.10%|51.58%|0|
|16|DeepLearning/EEGNeX-NB|29|28.39|51.76%|51.69%|2|
|17|DeepLearning/EEGConformer-NB|29|28.58|52.33%|51.40%|1|
|18|DeepLearning/SPARCNet-NB|29|28.75|50.44%|51.05%|1|
|19|FeatureBased/XdawnCov-TS-SVM-MOABB|28|29.97|53.39%|54.87%|0|
|20|FeatureBased/CSP-SVM-MOABB|29|30.14|55.49%|57.94%|1|
|21|FeatureBased/CSP+SVM-BP830|28|30.42|55.02%|56.99%|1|
|22|DictionaryBased/TDE|25|30.61|52.23%|53.95%|0|
|23|FeatureBased/FBCSP-SVM-MOABB|29|31.31|53.99%|55.12%|0|
|24|FeatureBased/LogVar-LDA-MOABB|29|31.56|52.59%|53.21%|0|
|25|FeatureBased/TSLR-NB|29|32.19|52.28%|53.11%|0|
|26|DeepLearning/FBCNet-NB|29|32.47|51.22%|52.45%|0|
|27|FeatureBased/CSP+SVM-BP140|28|32.67|51.98%|54.24%|0|
|28|DistanceBased/RiemannianMDM-OAS-BP830|29|33.06|54.08%|54.51%|0|
|29|DeepLearning/IFNet-NB|29|33.56|51.47%|51.76%|0|
|30|DeepLearning/LaBraM|24|33.81|48.12%|49.52%|0|
|31|DeepLearning/EEGInceptionMI-NB|29|33.97|51.54%|51.00%|0|
|32|DistanceBased/MDM-MOABB|29|34.83|53.52%|54.00%|0|
|33|DistanceBased/ERPCov-MDM-MOABB|29|35.75|52.53%|53.56%|1|
|34|FeatureBased/CSP+LDA-BP140|28|35.92|52.69%|55.06%|1|
|35|DeepLearning/TMSANet-NB|28|37.06|49.77%|49.91%|0|
|36|FeatureBased/LogVar-SVM-MOABB|28|37.31|52.47%|54.82%|0|
|37|DeepLearning/SincShallowNet-NB|29|37.58|48.92%|47.74%|0|
|38|DeepLearning/SSTDPN-NB|28|37.75|49.25%|49.24%|0|
|39|FeatureBased/DE-SVM|29|37.86|50.82%|53.26%|0|
|40|DeepLearning/BrainModule-NB|29|38.33|49.24%|49.57%|0|
|41|DistanceBased/RiemannianKNN-OAS-BP140|29|38.42|49.49%|50.12%|0|
|42|DeepLearning/CBraMod-random|28|38.89|48.00%|49.18%|0|
|43|DeepLearning/SleepStagerChambon2018-NB|29|39.00|48.05%|48.02%|0|
|44|DistanceBased/RiemannianKNN-OAS-BP830|29|39.11|51.15%|52.86%|0|
|45|DeepLearning/EEGNet-NB|29|39.14|48.68%|48.35%|0|
|46|DeepLearning/TCFormer-NB|28|39.17|49.35%|50.14%|0|
|47|DistanceBased/XdawnCov-MDM-MOABB|29|39.58|52.36%|53.25%|0|
|48|DeepLearning/CTNet-NB|29|39.69|48.04%|48.24%|0|
|49|DeepLearning/IT|29|40.58|47.97%|49.04%|1|
|50|DeepLearning/SCCNet-NB|29|40.67|47.50%|47.94%|1|
|51|DeepLearning/EEGITNet-NB|29|40.89|48.06%|46.94%|0|
|52|DeepLearning/CBraMod|28|40.97|46.94%|47.99%|0|
|53|DeepLearning/ShallowFBCSPNet-NB|29|41.25|46.64%|46.36%|0|
|54|DeepLearning/Deep4Net-NB|29|41.31|48.68%|48.99%|0|
|55|DeepLearning/EEGSimpleConv-NB|29|41.47|49.34%|50.66%|0|
|56|DeepLearning/AttentionBaseNet-NB|28|41.97|47.56%|48.08%|0|
|57|DistanceBased/RiemannianMDM-OAS-BP140|29|42.28|51.88%|52.63%|1|
|58|DeepLearning/LaBraM-random|24|42.33|48.40%|49.57%|0|
|59|DeepLearning/MSCFormer-NB|28|43.39|46.42%|44.85%|0|
|60|DeepLearning/FBLightConvNet-NB|29|45.81|45.88%|44.44%|0|
|61|DeepLearning/CNN|29|45.81|45.74%|46.18%|0|
|62|DeepLearning/ATCNet-NB|28|46.00|47.16%|46.96%|0|
|63|DistanceBased/TRCA-BP640|29|46.31|43.94%|43.73%|0|
|64|DeepLearning/EEGTCNet-NB|29|48.06|45.38%|44.63%|0|
|65|DeepLearning/BDTCN-NB|29|48.25|45.18%|46.01%|0|
|66|FeatureBased/xDAWN-LDA-MOABB|29|49.33|43.22%|44.67%|1|
|67|DeepLearning/GCNsNet-NB|28|50.28|44.92%|44.02%|0|
|68|DeepLearning/ContraWR-NB|29|53.00|42.97%|43.90%|0|
|69|DeepLearning/MSVTNet-NB|28|53.28|44.74%|44.75%|0|
|70|DistanceBased/TemplateCCA-BP640|29|54.17|41.78%|42.17%|0|
|71|DeepLearning/SyncNet-NB|29|56.47|42.32%|41.88%|0|


Common problems: ImaginedAnkleRest, ImaginedFeetHands, ImaginedGripForce, ImaginedHandFeetTongue, ImaginedHandGestures, ImaginedHandPupils, ImaginedHandsEmbodiment, ImaginedHandsFeet, ImaginedHandsOnline, ImaginedHandsRobot, ImaginedHandsStroke, ImaginedLeftRightHand, ImaginedLeftRightHandRest, ImaginedOpenCloseFist, ImaginedRightHandFeet, ImaginedRightHandFeetRest, LowCost, SitStand.

## EvokedPotentials

79classifiers; 8common problems from 13in scope. Minimum coverage: 11.

| Rank | Classifier | Coverage | Mean rank | Balanced accuracy | Accuracy | Wins |
|---:|---|---:|---:|---:|---:|---:|
|1|ConvolutionBased/MRHydra|12|14.25|44.94%|54.81%|0|
|2|Hybrid/HC2-noTDE-pc|11|14.38|45.00%|58.03%|1|
|3|ShapeletBased/RDST|12|17.25|44.32%|57.94%|0|
|4|DeepLearning/EEGSimpleConv-NB|11|17.88|48.24%|47.36%|0|
|5|DeepLearning/TIDNet-NB|11|19.56|45.22%|42.88%|0|
|6|DeepLearning/EEGInceptionMI-NB|11|19.69|56.95%|58.19%|1|
|7|ConvolutionBased/Arsenal|12|20.62|42.74%|56.49%|0|
|8|ShapeletBased/STC-aeon|11|20.81|45.63%|59.67%|0|
|9|DeepLearning/EEGNeX-NB|11|21.75|46.03%|40.83%|1|
|10|DeepLearning/IFNet-NB|11|22.44|53.97%|49.70%|0|
|11|Hybrid/HC2-noTDE|11|24.12|43.67%|57.81%|0|
|12|IntervalBased/DrCIF-500|11|25.06|44.07%|58.49%|0|
|13|DeepLearning/Deep4Net-NB|11|25.19|51.84%|47.23%|1|
|14|IntervalBased/QUANT|12|25.56|43.77%|58.03%|1|
|15|DeepLearning/SPARCNet-NB|11|25.88|44.61%|40.37%|0|
|16|IntervalBased/DrCIF|12|27.50|42.31%|56.65%|0|
|17|DeepLearning/FBMSNet-NB|11|28.12|44.00%|44.03%|0|
|18|FeatureBased/TSLR-NB|11|28.38|38.59%|39.23%|0|
|19|DistanceBased/XdawnCov-MDM-MOABB|11|29.25|52.59%|51.68%|0|
|20|DeepLearning/CTNet-NB|11|30.38|44.46%|39.89%|0|
|21|DeepLearning/BrainModule-NB|11|31.25|46.10%|46.26%|0|
|22|DeepLearning/LaBraM-random|12|32.00|38.62%|52.22%|0|
|23|FeatureBased/XDAWN|11|32.31|39.23%|51.95%|0|
|24|DistanceBased/TRCA-BP640|11|32.62|56.15%|54.98%|1|
|25|DistanceBased/ERPCov-MDM-MOABB|11|32.94|49.78%|53.95%|0|
|26|DeepLearning/EEGNet-NB|11|33.19|42.63%|34.88%|0|
|27|DeepLearning/LaBraM|12|33.44|37.44%|51.46%|0|
|28|DistanceBased/RiemannianMDM-OAS-BP140|11|35.69|36.79%|43.30%|0|
|29|DeepLearning/TCFormer-NB|13|35.81|44.38%|39.70%|2|
|30|DeepLearning/TMSANet-NB|13|37.06|40.62%|32.10%|0|
|31|DistanceBased/TemplateCCA-BP640|11|39.00|48.04%|48.56%|0|
|32|DistanceBased/RiemannianMDM-OAS|13|39.19|36.22%|43.29%|0|
|33|DeepLearning/GCNsNet-NB|13|39.25|36.88%|32.72%|0|
|34|DeepLearning/IT|12|39.31|40.40%|52.79%|0|
|35|DeepLearning/FBCNet-NB|11|39.31|40.07%|35.71%|0|
|36|DeepLearning/SCCNet-NB|11|39.88|40.34%|35.10%|0|
|37|DistanceBased/RiemannianMDM|13|40.56|36.54%|43.40%|0|
|38|DistanceBased/RiemannianKNN-OAS-BP140|11|40.88|36.00%|49.32%|0|
|39|FeatureBased/DE-LR|11|41.62|37.81%|52.07%|0|
|40|DeepLearning/ShallowFBCSPNet-NB|11|42.06|37.94%|30.71%|0|
|41|Tabular/LogisticRegression|13|42.06|37.35%|45.26%|0|
|42|DeepLearning/ATCNet-NB|13|42.12|36.77%|30.70%|0|
|43|DeepLearning/CBraMod|13|42.12|36.66%|50.36%|0|
|44|DistanceBased/FgMDM-MOABB|11|42.94|37.47%|38.37%|0|
|45|DeepLearning/SincShallowNet-NB|11|42.94|36.75%|22.67%|0|
|46|DeepLearning/EEGITNet-NB|11|43.94|36.85%|30.95%|0|
|47|DictionaryBased/TDE|11|44.00|40.82%|54.89%|0|
|48|DistanceBased/RiemannianMDM-OAS-BP830|11|44.25|36.41%|43.07%|0|
|49|DeepLearning/EEGTCNet-NB|11|44.38|36.53%|30.56%|0|
|50|DeepLearning/AttentionBaseNet-NB|13|44.50|38.48%|37.47%|0|
|51|DeepLearning/CBraMod-random|13|45.56|36.02%|50.16%|0|
|52|FeatureBased/LogVar-LDA-MOABB|11|46.31|36.64%|51.03%|0|
|53|DeepLearning/EEGConformer-NB|11|46.56|37.54%|25.92%|0|
|54|FeatureBased/DE-SVM|11|46.75|37.18%|51.61%|0|
|55|DeepLearning/SyncNet-NB|11|46.81|35.80%|35.20%|0|
|56|FeatureBased/CSP+SVM|12|48.62|35.93%|50.50%|0|
|57|FeatureBased/CSP+LDA-BP830|11|49.00|35.49%|49.63%|0|
|58|FeatureBased/CSP-LDA-MOABB|11|49.88|35.61%|50.03%|0|
|59|DistanceBased/MDM-MOABB|11|50.31|36.28%|43.01%|0|
|60|DeepLearning/CNN|13|50.88|37.24%|48.58%|0|
|61|FeatureBased/TSSVM-MOABB|11|50.94|36.04%|50.36%|0|
|62|FeatureBased/CSP+LDA|13|50.94|35.64%|50.14%|0|
|63|FeatureBased/CSP+LDA-BP140|11|51.12|35.52%|49.87%|0|
|64|FeatureBased/CSP+SVM-BP830|11|51.25|35.99%|50.54%|0|
|65|FeatureBased/DLCSP-shLDA-MOABB|11|51.56|35.50%|50.01%|0|
|66|DeepLearning/BDTCN-NB|11|51.62|36.66%|22.98%|0|
|67|DistanceBased/RiemannianKNN-OAS-BP830|11|53.31|35.53%|49.12%|0|
|68|FeatureBased/LogVar-SVM-MOABB|11|53.38|35.40%|49.97%|0|
|69|FeatureBased/TSLR-MOABB|11|54.75|36.03%|50.33%|0|
|70|FeatureBased/TSEN-MOABB|11|55.69|35.99%|50.31%|0|
|71|Tabular/SVM|11|55.69|35.73%|49.50%|0|
|72|Tabular/RandomForest|13|56.56|35.02%|39.88%|0|
|73|FeatureBased/CSP-SVM-MOABB|11|56.62|35.07%|49.49%|0|
|74|DeepLearning/MSVTNet-NB|13|57.75|35.78%|22.43%|0|
|75|DistanceBased/RiemannianKNN-OAS|12|57.94|34.93%|48.48%|0|
|76|DeepLearning/MSCFormer-NB|13|58.31|34.90%|26.44%|0|
|77|DistanceBased/RiemannianKNN|12|59.81|34.96%|48.38%|0|
|78|FeatureBased/CSP+SVM-BP140|11|61.06|34.74%|49.31%|0|
|79|DeepLearning/ContraWR-NB|11|62.31|34.76%|28.55%|0|


Common problems: ALSAttention, CodedFlickerTargets, FlickerAttention, FlickerTargets, GameTargets, SpellerFlashes, WirelessFlicker, WordRelatedness.

## Clinical

69classifiers; 11common problems from 17in scope. Minimum coverage: 14.

| Rank | Classifier | Coverage | Mean rank | Balanced accuracy | Accuracy | Wins |
|---:|---|---:|---:|---:|---:|---:|
|1|ConvolutionBased/Arsenal|16|20.64|58.65%|63.21%|0|
|2|FeatureBased/TSLR-MOABB|14|21.23|57.34%|59.67%|0|
|3|FeatureBased/CSP+SVM-BP140|14|21.73|59.93%|63.78%|1|
|4|FeatureBased/CSP-SVM-MOABB|14|23.82|57.61%|61.61%|1|
|5|FeatureBased/CSP+SVM-BP830|14|24.00|58.18%|62.42%|1|
|6|ShapeletBased/STC-aeon|16|24.32|57.44%|61.18%|0|
|7|FeatureBased/DE-LR|14|25.73|57.71%|59.97%|1|
|8|IntervalBased/DrCIF|16|25.77|54.66%|59.29%|0|
|9|FeatureBased/CSP-LDA-MOABB|14|26.50|57.69%|59.97%|0|
|10|FeatureBased/DE-SVM|14|26.73|56.34%|59.25%|0|
|11|DistanceBased/RiemannianMDM-OAS-BP140|14|27.41|57.93%|57.42%|0|
|12|FeatureBased/CSP+LDA-BP830|14|27.59|57.70%|59.66%|1|
|13|FeatureBased/CSP+LDA-BP140|14|27.68|55.52%|57.73%|0|
|14|DeepLearning/ContraWR-NB|14|28.14|54.97%|52.31%|1|
|15|DeepLearning/TIDNet-NB|14|28.14|54.51%|56.50%|0|
|16|FeatureBased/DLCSP-shLDA-MOABB|14|28.18|56.79%|59.03%|0|
|17|FeatureBased/FBCSP-SVM-MOABB|14|28.18|55.35%|59.11%|0|
|18|DeepLearning/AttentionBaseNet-NB|14|28.55|53.14%|50.49%|0|
|19|IntervalBased/DrCIF-500|15|28.59|54.01%|58.61%|0|
|20|DistanceBased/RiemannianMDM-OAS-BP830|14|29.14|55.22%|55.22%|0|
|21|FeatureBased/TSEN-MOABB|14|29.18|54.69%|56.61%|0|
|22|DeepLearning/EEGNeX-NB|14|29.27|54.47%|55.93%|1|
|23|DistanceBased/ERPCov-MDM-MOABB|14|30.23|55.14%|55.14%|0|
|24|DistanceBased/MDM-MOABB|14|30.68|54.36%|54.48%|0|
|25|FeatureBased/TSSVM-MOABB|14|30.68|54.02%|59.76%|0|
|26|DeepLearning/SincShallowNet-NB|14|30.95|53.69%|52.40%|0|
|27|DeepLearning/IFNet-NB|14|31.23|53.25%|53.36%|0|
|28|DistanceBased/FgMDM-MOABB|14|32.14|53.56%|56.99%|0|
|29|DeepLearning/TMSANet-NB|14|32.27|55.31%|55.57%|1|
|30|DeepLearning/CTNet-NB|14|32.32|52.97%|56.11%|0|
|31|DeepLearning/EEGInceptionMI-NB|14|32.36|53.38%|53.56%|1|
|32|DistanceBased/RiemannianMDM-OAS|14|32.50|55.05%|54.99%|0|
|33|DeepLearning/ATCNet-NB|14|32.86|53.79%|51.87%|0|
|34|DeepLearning/ShallowFBCSPNet-NB|14|33.45|52.85%|50.86%|0|
|35|DeepLearning/Deep4Net-NB|14|34.09|51.63%|49.36%|0|
|36|DeepLearning/BrainModule-NB|14|34.45|52.52%|51.42%|0|
|37|DeepLearning/MSVTNet-NB|14|35.73|52.53%|49.90%|0|
|38|DistanceBased/RiemannianKNN-OAS-BP140|14|35.77|53.52%|55.28%|0|
|39|DeepLearning/FBMSNet-NB|14|36.73|51.42%|52.13%|1|
|40|DistanceBased/RiemannianKNN-OAS-BP830|14|37.27|52.34%|53.70%|1|
|41|DeepLearning/TCFormer-NB|14|37.50|51.13%|49.57%|0|
|42|DeepLearning/EEGSimpleConv-NB|14|37.59|51.53%|53.87%|0|
|43|FeatureBased/TSLR-NB|14|37.68|50.74%|53.65%|0|
|44|DeepLearning/FBCNet-NB|14|37.77|50.76%|55.16%|0|
|45|DeepLearning/EEGNet-NB|14|38.27|50.26%|51.06%|0|
|46|DeepLearning/CBraMod-random|14|38.36|49.57%|52.81%|0|
|47|DeepLearning/SCCNet-NB|14|38.55|50.46%|51.04%|0|
|48|DistanceBased/XdawnCov-MDM-MOABB|14|38.68|51.88%|51.71%|0|
|49|DeepLearning/MSCFormer-NB|14|39.09|51.06%|55.32%|0|
|50|DeepLearning/EEGConformer-NB|14|39.23|50.14%|52.45%|0|
|51|DeepLearning/SPARCNet-NB|14|39.91|50.54%|47.59%|0|
|52|FeatureBased/CSP+SVM|14|40.36|51.49%|55.36%|0|
|53|DeepLearning/CBraMod|14|40.64|50.17%|53.84%|0|
|54|DeepLearning/IT|16|42.32|49.20%|50.19%|0|
|55|DistanceBased/TemplateCCA-BP640|14|42.50|50.42%|50.06%|1|
|56|FeatureBased/LogVar-LDA-MOABB|14|42.50|48.36%|49.96%|0|
|57|DeepLearning/CNN|17|42.59|50.43%|56.88%|0|
|58|DeepLearning/SyncNet-NB|14|43.27|47.92%|48.19%|0|
|59|FeatureBased/LogVar-SVM-MOABB|14|43.86|49.98%|53.50%|0|
|60|FeatureBased/CSP+LDA|14|44.32|49.67%|54.33%|0|
|61|ShapeletBased/RDST|14|45.23|51.12%|54.49%|0|
|62|DeepLearning/EEGTCNet-NB|14|45.77|48.73%|49.77%|0|
|63|DeepLearning/EEGITNet-NB|14|45.77|48.72%|48.02%|0|
|64|DeepLearning/BDTCN-NB|14|45.86|47.81%|43.09%|0|
|65|FeatureBased/XdawnCov-TS-SVM-MOABB|14|45.95|47.56%|50.22%|0|
|66|FeatureBased/xDAWN-LDA-MOABB|14|49.05|47.24%|51.22%|0|
|67|Tabular/LogisticRegression|14|51.73|45.16%|51.42%|0|
|68|Tabular/RandomForest|14|52.00|44.13%|50.26%|0|
|69|DistanceBased/TRCA-BP640|14|52.41|46.47%|46.14%|0|


Common problems: Alzheimers, CovidSurvivors, EyesOpenShut, FibroLiverpool, FibroUEA, LongIntervalTask, ParkinsonsPedalling, ParkinsonsRest, PhotoStimulation, RestingEyeState, ShortIntervalTask.

## Coverage and audit

0result cells excluded for case-count/label-order checks; 0read errors. Full per-problem matrices, coverage, exclusions and broader50-result-panel tables are beside this report. Available-case means in coverage.csv are descriptive and should not be read as a fair league ranking.

The broad-panel appendix preserves the earlier collation's approximately50-result eligibility rule; it can leave a much smaller shared block after restricting to the current76problems. No missing performance was filled with chance, zero, or another classifier's score.
