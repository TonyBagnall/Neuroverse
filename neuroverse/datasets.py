"""The Neuroverse classification archive: dataset lists and metadata.

The archive includes the 30 problems published by Rushbrooke et al. (2026)
and 95 additional local preparations. Use the supplied fixed train/test splits;
some protocols intentionally share subjects and some older identity mappings
remain incomplete. See docs/dataset-provenance.md for dataset-specific evidence.

The lists here are plain lists of dataset names, so they can be passed directly
to the experiment and leaderboard code::

    from neuroverse.datasets import neuroverse_study, by_type

    leaderboard(datasets=by_type("MI"), estimators=["HC2", "MRHydra"])

Published open files are downloaded with ``aeon.datasets.load_classification``;
the newer preparations are loaded from local .ts files. The list is
held here rather than read from ``aeon.datasets.tsc_datasets.eeg2026`` so that
the leaderboard has the metadata beside the names and does not change when aeon
does.
"""

__all__ = [
    "neuroverse",
    "neuroverse_study",
    "neuroverse_core",
    "neuroverse_90",
    "neuroverse_extended",
    "neuroverse_114",
    "neuroverse_104",
    "neuroverse_89",
    "neuroverse_82",
    "neuroverse_77",
    "neuroverse_76",
    "neuroverse_open",
    "neuroverse_paper",
    "excluded_archive",
    "local_only",
    "restricted",
    "DOMAINS",
    "PROBLEM_FAMILIES",
    "BCI_TYPES",
    "CORE_PROBLEM_TYPES",
    "PROBLEM_TYPE_TABS",
    "STUDY_TAXONOMY",
    "metadata",
    "by_domain",
    "by_type",
    "metadata_markdown",
]

import pandas as pd

#: The three application areas the archive covers, as labelled in Table 1.
DOMAINS = {
    "Med": "Medical",
    "BCI": "Brain-computer interface",
    "Psy": "Psychology",
}

#: Top-level problem families for the 114-problem scalp EEG study.
PROBLEM_FAMILIES = {
    "BCI": "Brain-computer interface",
    "Clinical": "Clinical and physiological",
    "Cognitive": "Cognitive and affective",
}

#: Sub-paradigms within the BCI family.
BCI_TYPES = {
    "MI": "Motor imagery",
    "ME": "Motor execution and movement intention",
    "ERP": "Event-related potentials (P300 / SCP)",
    "VEP": "Visual evoked potentials (SSVEP / c-VEP)",
    "Speech": "Decoded and imagined speech",
}

#: The three symmetric 30-problem pillars of the primary 90-problem benchmark.
CORE_PROBLEM_TYPES = {
    "MI": "Motor Imagery",
    "Clinical": "Clinical",
    "Cognitive": "Cognitive",
}

#: Ordered problem-type tabs used on the interactive leaderboard (Core 30/30/30
#: first, followed by the extended 114-problem scalp EEG tabs).
PROBLEM_TYPE_TABS = {
    "Core90": "Core 90 (30/30/30)",
    "MI": "Motor Imagery",
    "Clinical": "Clinical",
    "Cognitive": "Cognitive",
    "All114": "All Scalp EEG",
    "BCI": "BCI (All)",
    "ME": "Motor Execution",
    "ERP": "ERP / P300",
    "VEP": "VEP / SSVEP",
    "Speech": "Speech BCI",
}

_COLUMNS = [
    "dataset",
    "channels",
    "classes",
    "length",
    "subjects",
    "train_size",
    "test_size",
    "frequency",
    "domain",
]

# Original benchmark problems, with documented local corrections to the paper's
# Alzheimers/FingerMovements lengths and FingerMovements sampling rate.
# Subjects is None where unavailable; several problems are MEG rather than EEG.
_TABLE = [
    ("Alzheimers", 19, 3, 15000, 88, 45, 43, 500, "Med"),
    ("Blink", 4, 2, 510, None, 500, 450, 255, "BCI"),
    ("ButtonPress", 32, 2, 1000, 1, 120, 120, 1000, "BCI"),
    ("EpilepticSeizures", 1, 2, 178, None, 80, 11420, 178, "Med"),
    ("EyesOpenShut", 14, 2, 128, None, 56, 42, 128, "BCI"),
    ("FaceDetection", 144, 2, 62, None, 5890, 3524, 250, "Psy"),
    ("FeedbackButton", 61, 2, 450, 57, 1700, 1150, 500, "Psy"),
    ("FeetHands", 64, 2, 640, 105, 2835, 1890, 160, "BCI"),
    ("FibroLiverpool", 64, 2, 15360, 36, 15, 21, 512, "Med"),
    ("FibroUEA", 64, 2, 15000, 41, 17, 23, 500, "Med"),
    ("FingerMovements", 28, 2, 50, None, 316, 100, 100, "BCI"),
    ("HandMovementDirection", 10, 4, 400, None, 160, 74, 400, "BCI"),
    ("ImaginedFeetHands", 64, 2, 640, 105, 2835, 1935, 160, "BCI"),
    ("ImaginedOpenCloseFist", 64, 2, 640, 105, 2835, 1890, 160, "BCI"),
    ("InnerSpeech", 128, 4, 3584, 9, 250, 200, 1024, "BCI"),
    ("LongIntervalTask", 63, 2, 4750, 139, 1660, 1120, 500, "Med"),
    ("LowCost", 15, 2, 375, 10, 600, 600, 125, "BCI"),
    ("MatchingPennies", 10, 2, 10000, 7, 700, 700, 5000, "Psy"),
    ("MindReading", 204, 5, 200, None, 727, 653, 200, "BCI"),
    ("MotorImagery", 64, 2, 3000, None, 278, 100, 1000, "BCI"),
    ("OpenCloseFist", 64, 2, 640, 105, 2835, 1890, 160, "BCI"),
    ("PhotoStimulation", 19, 3, 9000, 73, 37, 36, 500, "Med"),
    ("PronouncedSpeech", 128, 4, 3584, 9, 250, 200, 1024, "BCI"),
    ("SelfRegulationSCP1", 6, 2, 896, None, 268, 293, 256, "BCI"),
    ("SelfRegulationSCP2", 7, 2, 1152, None, 200, 180, 256, "BCI"),
    ("ShortIntervalTask", 63, 2, 2750, 139, 1660, 1120, 500, "Med"),
    ("SitStand", 17, 4, 1250, 32, 3200, 3200, 250, "BCI"),
    ("Sleep", 1, 5, 178, 82, 2394, 1205, 100, "Psy"),
    ("SongFamiliarity", 32, 2, 3000, 29, 700, 750, 1000, "Psy"),
    ("VisualSpeech", 128, 4, 3584, 9, 250, 200, 1024, "BCI"),
]

#: Original published benchmark subset, retained for reproducing the paper.
neuroverse_paper = [row[0] for row in _TABLE]

_TABLE += [
    ("ImaginedHandsStroke", 29, 2, 2000, 46, 1400, 440, 500, "BCI"),
    ("RestingEyeState", 16, 2, 5120, 19, 140, 50, 512, "BCI"),
    ("GameTargets", 8, 2, 225, 22, 20400, 6000, 250, "BCI"),
    ("CodedFlickerTargets", 28, 4, 1100, 12, 2160, 720, 500, "BCI"),
    ("ImaginedHandTraining", 22, 2, 2560, 18, 5568, 1588, 512, "BCI"),
    ("ALSAttention", 8, 2, 256, 8, 14400, 19200, 256, "BCI"),
    ("ArmMovements", 61, 7, 1536, 15, 4620, 1680, 512, "BCI"),
    ("CodedSymbols", 8, 20, 16128, 30, 2200, 800, 512, "BCI"),
    ("FlickerAttention", 8, 4, 512, 12, 576, 352, 256, "BCI"),
    ("FlickerSymbols", 64, 40, 1500, 35, 6240, 2160, 250, "BCI"),
    ("FlickerTargets", 8, 12, 1114, 10, 1440, 360, 256, "BCI"),
    ("ImaginedGripForce", 14, 3, 640, 10, 240, 60, 128, "BCI"),
    ("ImaginedHandSides", 3, 2, 1125, 9, 3680, 2840, 250, "BCI"),
    ("ImaginedHandsFeet", 13, 3, 1250, 4, 1365, 435, 250, "BCI"),
    ("SpellerFlashes", 8, 2, 205, 10, 12600, 12600, 256, "BCI"),
    ("WirelessFlicker", 14, 5, 640, 11, 1000, 375, 128, "BCI"),
    ("ImaginedRightHandFeet", 16, 2, 1536, 8, 240, 80, 512, "BCI"),
    ("ImaginedArmMovements", 61, 7, 1536, 15, 4620, 1680, 512, "BCI"),
    ("ChildhoodSeizures", 17, 2, 2000, 22, 34045, 2140, 200, "Med"),
    ("ClinicalSleepStages", 6, 5, 6000, 82, 64889, 8803, 200, "Med"),
    ("CovidSurvivors", 19, 2, 400, 172, 25319, 11216, 200, "Med"),
    ("ImaginedCommands", 64, 5, 600, 15, 5250, 750, 200, "BCI"),
    ("ImaginedHandFeetTongue", 22, 4, 800, 9, 3456, 1728, 200, "BCI"),
    ("ImaginedLeftRightHand", 60, 2, 800, 10, 1260, 320, 200, "BCI"),
    ("MatchWords", 156, 2, 1500, None, 360, 360, 1000, "Psy"),
    ("MentalArithmetic", 19, 2, 1000, 36, 1515, 192, 200, "BCI"),
    ("NBackAudioCondition", 16, 12, 600, 23, 11520, 5040, 200, "Psy"),
    ("VideoEmotions", 25, 8, 2000, 123, 15629, 3588, 200, "Psy"),
    ("VisualSubjects", 16, 2, 512, 64, 42189, 18826, 512, "BCI"),
    ("WordRelatedness", 30, 2, 1024, 40, 3600, 1200, 1024, "Psy"),
]

_TABLE += [
    ("ParkinsonsPedalling", 63, 2, 2000, 39, 2123, 842, 500, "Med"),
    ("ParkinsonsRest", 63, 2, 2000, 55, 1216, 420, 500, "Med"),
]

_TABLE += [
    ("ObsessiveCompulsiveSymptoms", 52, 2, 2000, 35, 7754, 2660, 500, "Med"),
]

_TABLE += [("WalkingTargets", 42, 2, 100, 15, 3600, 900, 100, "BCI")]


_TABLE += [
    ('SymbolAttention', 8, 2, 256, 19, 521259, 175180, 256, "BCI"),
    ('ArmCoordination', 64, 2, 1500, 8, 1498, 505, 500, "BCI"),
]

_TABLE += [
    ("ImaginedHandGestures", 17, 4, 3000, 11, 382, 143, 500, "BCI"),
    ("ImaginedArmTasks", 17, 10, 3960, 12, 3240, 1080, 1000, "BCI"),
]

_TABLE += [
    ("ImaginedRightHandFeetRest", 24, 3, 750, 1, 150, 40, 250, "BCI"),
    ("ImaginedLeftRightHandRest", 24, 3, 750, 1, 480, 240, 250, "BCI"),
    ("ImaginedHandFootActions", 11, 8, 500, 60, 2727, 930, 125, "BCI"),
]

_TABLE += [("ImaginedHandsRobot", 32, 2, 1000, 12, 2880, 2880, 250, "BCI")]

_TABLE += [('ImaginedGrasps', 60, 3, 1000, 15, 4500, 2250, 250, 'BCI')]

_TABLE += [('ImaginedHandsEmbodiment', 32, 2, 4997, 26, 798, 240, 500, 'BCI')]

_TABLE += [('ImaginedHandsHaptics', 32, 2, 2495, 20, 3083, 1034, 500, 'BCI')]

_TABLE += [('ImaginedAnkleRest', 28, 2, 1000, 6, 264, 132, 250, 'BCI')]

_TABLE += [('ImaginedHandsFeedback', 16, 2, 1250, 55, 6560, 2240, 250, 'BCI')]

_TABLE += [('ImaginedHandsOnline', 28, 2, 1000, 10, 2123, 1519, 250, 'BCI'), ('ImaginedHandsCrossSession', 28, 2, 1000, 14, 2400, 960, 250, 'BCI'), ('ImaginedHandPupils', 32, 2, 3072, 30, 1150, 400, 512, 'BCI'), ('ImaginedHandsPhysiology', 64, 2, 1536, 52, 7880, 2640, 512, 'BCI'), ('ImaginedHandMusicTempo', 19, 2, 20000, 18, 1698, 643, 1000, 'BCI')]

_TABLE += [('AlzheimersAuditory', 19, 3, 1000, 33, 2020, 1000, 250, 'Med')]

_TABLE += [('ParkinsonsRestIowa', 60, 2, 2000, 149, 4598, 1306, 500, 'Med')]

_TABLE += [('ParkinsonsRestUCSD', 32, 2, 2048, 31, 1126, 380, 512, 'Med')]

_TABLE += [('ImaginedFingers', 60, 5, 3072, 20, 4481, 1500, 512, 'BCI')]

_TABLE += [
    ("MigraineRest", 128, 2, 2048, 33, 1920, 750, 512, "Med"),
    ("ParkinsonsWalking", 53, 2, 1000, 133, 6493, 2100, 250, "Med"),
    ("PsychosisRestFirst", 61, 2, 4000, 82, 4657, 1601, 1000, "Med"),
    ("PsychosisRestSecond", 61, 2, 4000, 61, 3472, 1239, 1000, "Med"),
    ("TBIAuditoryOddball", 61, 3, 500, 87, 4848, 1748, 500, "Med"),
]

_TABLE += [
    ("ArithmeticDifficulty", 19, 7, 1500, 18, 819, 315, 250, "Psy"),
    ("CognitiveLoadTasks", 8, 4, 1000, 40, 3600, 1200, 250, "Psy"),
    ("MultitaskWorkload", 61, 3, 500, 15, 9834, 3576, 250, "Psy"),
    ("NBackDifficulty", 19, 4, 375, 18, 5200, 2000, 250, "Psy"),
    ("NaturalisticEmotion", 128, 4, 1000, 9, 780, 405, 250, "Psy"),
    ("SleepDeprivation", 61, 2, 2000, 68, 3060, 1020, 500, "Psy"),
    ("SternbergMemory", 69, 3, 500, 23, 1534, 558, 250, "Psy"),
]

_TABLE += [
    ("AffectiveMusic", 19, 4, 4000, 31, 2588, 904, 1000, "Psy"),
    ("AuditoryDeviance", 30, 2, 1024, 40, 11934, 4000, 1024, "Psy"),
    ("BanditGambling", 63, 2, 500, 23, 8146, 2870, 500, "Psy"),
    ("FaceStimuli", 30, 4, 1024, 40, 9600, 3200, 1024, "Psy"),
    ("FlankerConflict", 29, 6, 500, 60, 12148, 4050, 500, "Psy"),
    ("FuzzySemantics", 30, 2, 200, 30, 8780, 3191, 200, "Psy"),
    ("InternalAttention", 32, 2, 500, 20, 6071, 2093, 250, "Psy"),
    ("MoralConflict", 32, 4, 512, 31, 2488, 866, 512, "Psy"),
    ("NarrativeReward", 30, 4, 1000, 40, 4319, 1440, 1000, "Psy"),
    ("ProbabilisticFeedback", 59, 4, 500, 25, 2877, 1120, 500, "Psy"),
    ("PythonCodeReading", 32, 4, 334, 61, 7177, 2560, 256, "Psy"),
    ("SentenceViolations", 29, 5, 500, 64, 8640, 2880, 500, "Psy"),
    ("SpatialAttention", 30, 2, 1024, 40, 9607, 3200, 1024, "Psy"),
    ("VerbalWorkingMemory", 19, 6, 1000, 155, 13910, 4676, 500, "Psy"),
    ("VisualSymmetry", 64, 2, 512, 24, 2879, 960, 512, "Psy"),
]

_TABLE += [
    ("AcuteTBIVisualMemory", 63, 2, 2000, 86, 1920, 660, 500, "Med"),
    ("AdultSeizures", 29, 2, 2048, 14, 1572, 591, 512, "Med"),
    ("AlcoholBandit", 59, 2, 2000, 53, 1170, 420, 500, "Med"),
    ("ParkinsonsReinforcement", 63, 2, 2000, 56, 1260, 420, 500, "Med"),
    ("PostCovidVigilance", 127, 2, 4000, 97, 1800, 625, 1000, "Med"),
    ("SchizophreniaConflict", 60, 2, 4000, 76, 1680, 600, 1000, "Med"),
    ("SchizophreniaMotion", 61, 2, 3600, 40, 870, 330, 1200, "Med"),
    ("SchizophreniaUltimatum", 128, 2, 3072, 43, 768, 264, 2048, "Med"),
    ("StrokeMotorImagery", 58, 2, 4000, 18, 520, 200, 1000, "Med"),
    ("ZosterNeuralgiaSleep", 19, 2, 1024, 58, 1290, 450, 256, "Med"),
    ("FamilialAlzheimersRest", 58, 2, 4000, 43, 960, 330, 1000, "Med"),
    ("LearningDifficulties", 32, 2, 1024, 75, 1680, 570, 256, "Med"),
]

_TABLE.sort(key=lambda row: row[0])

#: Prepared locally; not registered for aeon's automatic archive downloads.
local_only = [row[0] for row in _TABLE if row[0] not in neuroverse_paper]

#: Datasets that are not fully open access, so are not on Zenodo and cannot be
#: loaded through aeon. Results for them are still reported.
restricted = ["FibroLiverpool", "FibroUEA"]

#: All 125 prepared classification problems used by this repository.
neuroverse = [row[0] for row in _TABLE]

#: The 28 problems anyone can download, the same list as aeon's ``eeg2026``.
neuroverse_open = [name for name in neuroverse_paper if name not in restricted]

#: Problem-type assignment ``(family, bci_type)`` for the 114 selected open-access
#: scalp EEG classification problems. Strict scalp-EEG motor imagery (``MI``) has
#: 30 problems, ``Clinical`` has 30 problems, ``Cognitive`` has 30 problems, and
#: the 6 executed/movement-intention motor tasks sit in ``ME``. MEG
#: (FaceDetection, HandMovementDirection, MatchWords, MindReading), ECoG
#: (MotorImagery), restricted-access files (FibroLiverpool, FibroUEA), and the
#: four older excluded problems (Blink, ButtonPress, EpilepticSeizures, Sleep)
#: remain in the 125-item archive registry above.
STUDY_TAXONOMY: dict[str, tuple[str, str]] = {
    "ALSAttention": ("BCI", "ERP"),
    "AcuteTBIVisualMemory": ("Clinical", ""),
    "AdultSeizures": ("Clinical", ""),
    "AffectiveMusic": ("Cognitive", ""),
    "AlcoholBandit": ("Clinical", ""),
    "Alzheimers": ("Clinical", ""),
    "AlzheimersAuditory": ("Clinical", ""),
    "ArithmeticDifficulty": ("Cognitive", ""),
    "ArmCoordination": ("BCI", "ME"),
    "ArmMovements": ("BCI", "ME"),
    "AuditoryDeviance": ("Cognitive", ""),
    "BanditGambling": ("Cognitive", ""),
    "ChildhoodSeizures": ("Clinical", ""),
    "ClinicalSleepStages": ("Clinical", ""),
    "CodedFlickerTargets": ("BCI", "VEP"),
    "CodedSymbols": ("BCI", "VEP"),
    "CognitiveLoadTasks": ("Cognitive", ""),
    "CovidSurvivors": ("Clinical", ""),
    "EyesOpenShut": ("Cognitive", ""),
    "FaceStimuli": ("Cognitive", ""),
    "FamilialAlzheimersRest": ("Clinical", ""),
    "FeedbackButton": ("Cognitive", ""),
    "FeetHands": ("BCI", "ME"),
    "FingerMovements": ("BCI", "ME"),
    "FlankerConflict": ("Cognitive", ""),
    "FlickerAttention": ("BCI", "VEP"),
    "FlickerSymbols": ("BCI", "VEP"),
    "FlickerTargets": ("BCI", "VEP"),
    "FuzzySemantics": ("Cognitive", ""),
    "GameTargets": ("BCI", "ERP"),
    "ImaginedAnkleRest": ("BCI", "MI"),
    "ImaginedArmMovements": ("BCI", "MI"),
    "ImaginedArmTasks": ("BCI", "MI"),
    "ImaginedCommands": ("BCI", "Speech"),
    "ImaginedFeetHands": ("BCI", "MI"),
    "ImaginedFingers": ("BCI", "MI"),
    "ImaginedGrasps": ("BCI", "MI"),
    "ImaginedGripForce": ("BCI", "MI"),
    "ImaginedHandFeetTongue": ("BCI", "MI"),
    "ImaginedHandFootActions": ("BCI", "MI"),
    "ImaginedHandGestures": ("BCI", "MI"),
    "ImaginedHandMusicTempo": ("BCI", "MI"),
    "ImaginedHandPupils": ("BCI", "MI"),
    "ImaginedHandSides": ("BCI", "MI"),
    "ImaginedHandTraining": ("BCI", "MI"),
    "ImaginedHandsCrossSession": ("BCI", "MI"),
    "ImaginedHandsEmbodiment": ("BCI", "MI"),
    "ImaginedHandsFeedback": ("BCI", "MI"),
    "ImaginedHandsFeet": ("BCI", "MI"),
    "ImaginedHandsHaptics": ("BCI", "MI"),
    "ImaginedHandsOnline": ("BCI", "MI"),
    "ImaginedHandsPhysiology": ("BCI", "MI"),
    "ImaginedHandsRobot": ("BCI", "MI"),
    "ImaginedHandsStroke": ("BCI", "MI"),
    "ImaginedLeftRightHand": ("BCI", "MI"),
    "ImaginedLeftRightHandRest": ("BCI", "MI"),
    "ImaginedOpenCloseFist": ("BCI", "MI"),
    "ImaginedRightHandFeet": ("BCI", "MI"),
    "ImaginedRightHandFeetRest": ("BCI", "MI"),
    "InnerSpeech": ("BCI", "Speech"),
    "InternalAttention": ("Cognitive", ""),
    "LearningDifficulties": ("Clinical", ""),
    "LongIntervalTask": ("Clinical", ""),
    "LowCost": ("BCI", "MI"),
    "MatchingPennies": ("BCI", "ME"),
    "MentalArithmetic": ("Cognitive", ""),
    "MigraineRest": ("Clinical", ""),
    "MoralConflict": ("Cognitive", ""),
    "MultitaskWorkload": ("Cognitive", ""),
    "NBackAudioCondition": ("Cognitive", ""),
    "NBackDifficulty": ("Cognitive", ""),
    "NarrativeReward": ("Cognitive", ""),
    "NaturalisticEmotion": ("Cognitive", ""),
    "ObsessiveCompulsiveSymptoms": ("Clinical", ""),
    "OpenCloseFist": ("BCI", "ME"),
    "ParkinsonsPedalling": ("Clinical", ""),
    "ParkinsonsReinforcement": ("Clinical", ""),
    "ParkinsonsRest": ("Clinical", ""),
    "ParkinsonsRestIowa": ("Clinical", ""),
    "ParkinsonsRestUCSD": ("Clinical", ""),
    "ParkinsonsWalking": ("Clinical", ""),
    "PhotoStimulation": ("Clinical", ""),
    "PostCovidVigilance": ("Clinical", ""),
    "ProbabilisticFeedback": ("Cognitive", ""),
    "PronouncedSpeech": ("BCI", "Speech"),
    "PsychosisRestFirst": ("Clinical", ""),
    "PsychosisRestSecond": ("Clinical", ""),
    "PythonCodeReading": ("Cognitive", ""),
    "RestingEyeState": ("Cognitive", ""),
    "SchizophreniaConflict": ("Clinical", ""),
    "SchizophreniaMotion": ("Clinical", ""),
    "SchizophreniaUltimatum": ("Clinical", ""),
    "SelfRegulationSCP1": ("BCI", "ERP"),
    "SelfRegulationSCP2": ("BCI", "ERP"),
    "SentenceViolations": ("Cognitive", ""),
    "ShortIntervalTask": ("Clinical", ""),
    "SitStand": ("BCI", "MI"),
    "SleepDeprivation": ("Cognitive", ""),
    "SongFamiliarity": ("Cognitive", ""),
    "SpatialAttention": ("Cognitive", ""),
    "SpellerFlashes": ("BCI", "ERP"),
    "SternbergMemory": ("Cognitive", ""),
    "StrokeMotorImagery": ("Clinical", ""),
    "SymbolAttention": ("BCI", "ERP"),
    "TBIAuditoryOddball": ("Clinical", ""),
    "VerbalWorkingMemory": ("Cognitive", ""),
    "VideoEmotions": ("Cognitive", ""),
    "VisualSpeech": ("BCI", "Speech"),
    "VisualSubjects": ("BCI", "ERP"),
    "VisualSymmetry": ("Cognitive", ""),
    "WalkingTargets": ("BCI", "ERP"),
    "WirelessFlicker": ("BCI", "VEP"),
    "WordRelatedness": ("Cognitive", ""),
    "ZosterNeuralgiaSleep": ("Clinical", ""),
}

#: All 114 selected scalp EEG classification problems (30 MI + 24 other BCI +
#: 30 Clinical + 30 Cognitive).
neuroverse_114 = sorted(STUDY_TAXONOMY)
neuroverse_extended = neuroverse_114
neuroverse_104 = neuroverse_114  # backwards-compatible alias
neuroverse_89 = neuroverse_114  # backwards-compatible alias
neuroverse_82 = neuroverse_114  # backwards-compatible alias
neuroverse_77 = neuroverse_114  # backwards-compatible alias
neuroverse_76 = neuroverse_114  # backwards-compatible alias

#: Primary 90-problem benchmark: symmetric 30 Motor Imagery + 30 Clinical +
#: 30 Cognitive scalp EEG problems.
neuroverse_90 = sorted(
    name
    for name, (fam, bci) in STUDY_TAXONOMY.items()
    if fam in ("Clinical", "Cognitive") or bci == "MI"
)
neuroverse_core = neuroverse_90
neuroverse_study = neuroverse_90

#: Registered archive problems excluded from the 114-problem scalp EEG archive.
excluded_archive = [name for name in neuroverse if name not in STUDY_TAXONOMY]


def metadata() -> pd.DataFrame:
    """Return the archive metadata, one row per dataset.

    Returns
    -------
    pd.DataFrame
        Indexed by dataset name, with the columns of Table 1, availability
        flags, and the study taxonomy (``family``, ``bci_type``,
        ``problem_type``, ``study_core_90``, ``study_selected``).
    """
    frame = pd.DataFrame(_TABLE, columns=_COLUMNS).set_index("dataset")
    frame["subjects"] = frame["subjects"].astype("Int64")
    frequency = frame["frequency"].astype("Float64")
    frame["frequency"] = frequency.astype("Int64") if (frequency.dropna() % 1 == 0).all() else frequency
    frame["open"] = frame.index.isin(neuroverse_open)
    frame["published_benchmark"] = frame.index.isin(neuroverse_paper)
    frame["availability"] = [
        "local" if n in local_only else "restricted" if n in restricted else "aeon/Zenodo"
        for n in frame.index
    ]
    frame["study_core_90"] = frame.index.isin(neuroverse_90)
    frame["study_selected"] = frame.index.isin(STUDY_TAXONOMY)
    frame["study_76"] = frame["study_selected"]
    frame["family"] = [STUDY_TAXONOMY.get(n, ("", ""))[0] for n in frame.index]
    frame["bci_type"] = [STUDY_TAXONOMY.get(n, ("", ""))[1] for n in frame.index]
    frame["problem_type"] = [
        bci if bci else (fam if fam else "Excluded")
        for fam, bci in (STUDY_TAXONOMY.get(n, ("", "")) for n in frame.index)
    ]
    return frame


def by_domain(domain: str, open_only: bool = False) -> list[str]:
    """Return the datasets from one application area.

    Parameters
    ----------
    domain : str
        One of the keys of :data:`DOMAINS`: ``"Med"``, ``"BCI"`` or ``"Psy"``.
    open_only : bool, default=False
        Select only datasets registered for automatic aeon/Zenodo downloads.

    Returns
    -------
    list of str
        Dataset names, in archive order.
    """
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {', '.join(DOMAINS)}, not {domain!r}")
    return [
        row[0]
        for row in _TABLE
        if row[-1] == domain and (not open_only or row[0] in neuroverse_open)
    ]


def by_type(problem_type: str = "Core90") -> list[str]:
    """Return scalp EEG benchmark datasets by problem type or benchmark tier.

    Parameters
    ----------
    problem_type : str, default="Core90"
        One of ``"Core90"`` (primary 90-problem ``30 MI + 30 Clinical + 30 Cognitive``
        benchmark; aliases ``"Core"`` and ``"core90"``), ``"All114"`` or ``"All"``
        (all 114 scalp EEG problems), a top-level family (``"BCI"``,
        ``"Clinical"``, ``"Cognitive"``), or a BCI sub-type (``"MI"``, ``"ME"``,
        ``"ERP"``, ``"VEP"``, ``"Speech"``).

    Returns
    -------
    list of str
        Dataset names in alphabetical order.
    """
    if problem_type in {"Core90", "Core", "core90", "90"}:
        return list(neuroverse_90)
    if problem_type in {"All114", "All", "all114", "114"}:
        return list(neuroverse_114)
    if problem_type in PROBLEM_FAMILIES:
        return [
            name for name in neuroverse_114 if STUDY_TAXONOMY[name][0] == problem_type
        ]
    if problem_type in BCI_TYPES:
        return [
            name for name in neuroverse_114 if STUDY_TAXONOMY[name][1] == problem_type
        ]
    valid = ", ".join(PROBLEM_TYPE_TABS)
    raise ValueError(f"problem_type must be one of {valid}, not {problem_type!r}")


def metadata_markdown() -> str:
    """Return the archive metadata as a Markdown table, for the documentation."""
    header = [
        "Dataset", "Domain", "Study type", "Channels", "Length", "Classes",
        "Subjects", "Train", "Test", "Hz", "Availability",
    ]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in metadata().iterrows():
        study_type = "&mdash;" if row["problem_type"] == "Excluded" else row["problem_type"]
        cells = [
            name,
            row["domain"],
            study_type,
            str(row["channels"]),
            str(row["length"]),
            str(row["classes"]),
            "&mdash;" if pd.isna(row["subjects"]) else str(row["subjects"]),
            str(row["train_size"]),
            str(row["test_size"]),
            "&mdash;" if pd.isna(row["frequency"]) else f'{row["frequency"]:g}',
            row["availability"],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# Backwards-compatible aliases for scripts written before the Neuroverse rename.
eegbench = neuroverse
eegbench_76 = neuroverse_study
eegbench_open = neuroverse_open
eegbench_paper = neuroverse_paper
