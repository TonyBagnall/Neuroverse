# MRHydra competition submission

User authorised submitting MRHydra on10November2026. No upload has yet been
made: desktop browser control failed to initialise. Account/registration and
current submission eligibility have not been confirmed.

The intended upload is the single ZIP:

`D:/Data/EEGCompetitions/NeuralInterfaces2026Track2/selected-submissions/all-public/MRHydra.zip`

Size:2201934bytes. SHA256:
`2f43a9622000aabd6f005bc4e05fde2f28398879dd7b76449ffea32b018164b4`.
The isolated ZIP check passed on12public cases with aeon/numba imports blocked.
All-public portable predictions matched native aeon on120evenly spaced cases;
maximum score difference was2.87e-9. Median local CPU inference was0.501seconds
per window. These measurements do not establish performance in the platform's
container or on sealed data.

This is the model trained on all2655public PROTEUS cases, not the1521case
development model or any LOSO fold. Model: aeon1.5.0
MultiRocketHydraClassifier, seed33, default classifier settings, fitted locally.
The ZIP contains only `submission.py`, `weights.npz`, `LICENSE.txt` and
`method.json` at its root. Numeric fitted parameters replace training-time
Python objects; inference uses NumPy, Torch and the organiser's framework.
No runtime installation, aeon or numba import, or network download is required.
The aeon BSD3-Clause licence accompanies the adapted inference implementation.

Both package checks completed successfully on10November2026:

- `.openeegbench/competition-17982/mrhydra-package-state.json`: complete.
- `docs/competition-track2-mrhydra-isolated-check.json`: passed=true.

All379development validation predictions matched native aeon. The organiser's
streaming objective replay completed successfully and reproduced52.6428%
cell-mean balanced accuracy. Browser-control retry after reset failed with the
same kernel-assets path error; upload remains unperformed.

Export/parity details are in
`competition-track2-selected-export-all-public-smoke.json` (120evenly spaced
public cases) and `competition-track2-selected-export-development.json`
(all379development validation cases and the organiser's streaming objective).
These are numerical portability checks; all-public examples are training data
and their prediction agreement is not a generalisation estimate. The isolated
ZIP check blocks aeon/numba imports and verifies unchanged artifact files.
No actual Codabench container run has been verified locally.

Local LOSO results, now complete: MRHydra45.16%, QUANT43.58%, equal-score
ensemble46.44% cell-mean balanced accuracy. See
[full results](competition-track2-loso-results.json). Development's two-subject
MRHydra score was52.64%; it is a different evaluation protocol.

Upload the ZIP intact to the Track2 submission page for
[Codabench17982](https://www.codabench.org/competitions/17982/), if the account
is eligible. Use a description such as "aeon MRHydra; seed33; all public PROTEUS
training data; portable inference". Check the available submission quota and
phase in the UI. Retain the resulting submission ID, execution log and score;
do not retry a failed upload automatically and consume additional attempts.
