# Demonstration and viva preparation

## A simple live demonstration

1. Open notebook 01. State the question and connect EM and PR literature to the method.
2. Open notebook 02. Show the quality counts, exact-repeat assumption, time split and base rates. Explain two EDA charts.
3. Open notebook 03. Point to the 29 feature names and the legitimate training mask. Explain `fit_transform` versus `transform`.
4. Open notebook 04. Describe EM, read the AIC/BIC table and explain why K=8 wins only within this grid.
5. Open notebook 05. Show that validation sets cutoffs before test scoring. Recompute precision and recall from the confusion counts.
6. Compare the two PR curves. Show the policy and capacity tables. End with a limitation and the Phase 2 boundary.

Read 01, then open 02, 03, 04 and 05 in order in VS Code/Jupyter. Restart each notebook kernel and choose **Run All**. The notebook cells save their own outputs. Use `FILE_MAP.md` to trace each result. Large local inputs and binary models stay outside Git. For presentation without running Python, use `output/Study_Notebook.html` as the study index.

## Two-minute explanation

"We learn the density of earlier legitimate transactions with a GMM, then score later transactions by negative log density. We split chronologically, keep the scaler's fit inside legitimate training rows, and select K=8 using minimum training BIC among four candidates. Validation chooses the alert cutoffs. At the illustrative 0.5% validation budget, the test period gives 37 frauds found, 37 missed and 172 false alerts. Precision is 37/209 and recall is 37/74. Reviewing exactly the top 100 scores finds 29 frauds. The data is small and historical, and unusual legitimate activity can trigger alerts. This is Phase 1, so the autoencoder comparison remains future work."

## Each member's final self-check

- Explain the cleaning assumption and why raw and cleaned fraud counts differ.
- Identify the exact rows that fit the scaler and GMM.
- Explain a Gaussian component, soft membership and both EM steps.
- Explain why Class and Time are absent from the model inputs.
- Explain AIC/BIC, score direction, the validation percentile and strict cutoff rule.
- Recompute precision, recall and FPR with the correct denominators.
- Distinguish PR AUC, precision@100 and precision at a fixed threshold.
- Explain what the reference comparison can and cannot prove.
- State at least two limitations and their effect on the conclusion.
- Update the actual contribution record and confirm faculty repository access.

Use the root `STUDY_GUIDE.md` for definitions and short answers. Practicing these points is necessary; generated materials cannot guarantee a viva outcome.
