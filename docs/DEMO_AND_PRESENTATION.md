# Demonstration and presentation guide

## A simple 8-10 minute presentation

| Slides | Explain | Approximate time |
|---|---|---|
| 1-2 | We model legitimate behavior and rank unusual transactions for human review. Phase 1 covers only GMM. | 45 seconds |
| 3 | EM explains fitting, PR literature explains evaluation, scikit-learn documents the implementation. | 35 seconds |
| 4-5 | Explain duplicate removal and chronological splitting. Point to the legitimate-only fitted sample. | 70 seconds |
| 6-7 | Explain the log amount chart and the training-only scaler. State the 29 inputs. | 70 seconds |
| 8-9 | Explain both EM steps, the negative-log-density score, and minimum training BIC. | 90 seconds |
| 10 | Explain the PR curve, AP, the one-Gaussian reference, and rare fraud. | 50 seconds |
| 11-12 | Explain frozen validation thresholds and compute recall/precision from the default counts. | 75 seconds |
| 13 | Explain top-100 batch review and how it differs from a threshold. | 35 seconds |
| 14 | State the missed fraud, false alerts and study limits. Keep the autoencoder in Phase 2. | 35 seconds |

Each slide includes speaker notes with talking points and source locations. The charts and tables remain editable. The deck contains 14 slides and five native charts. The notebook contains the fuller set of 12 analytical charts.

## Opening script

"Our project asks whether a Gaussian Mixture Model can help prioritize card-fraud review. We fit the density of earlier legitimate transactions, give low-density transactions high anomaly scores, and test on a later period containing both classes. This evaluation covers Phase 1 machine learning only."

## Main result script

"We select eight components using minimum training BIC among one, two, four and eight. The test period contains 74 frauds. At the illustrative 0.5% validation false-positive policy, we catch 37 frauds and create 172 false alerts. That means 50% recall and 17.70% precision across 209 alerts. The threshold stays fixed between validation and test."

## Five-minute notebook demonstration

1. Open `output/phase1_notebook.html` if you only need to show the existing run. Open the `.ipynb` in VS Code/Jupyter if you need to run code.
2. Show the data-quality table. Explain why raw and cleaned fraud counts differ.
3. Show the split table. Point out that only legitimate training rows fit the scaler and model.
4. Show `make_features` and `fit_candidates` in `src/project.py`. Identify the 29 inputs, EM settings and K grid.
5. Show AIC/BIC and state that eight wins only within this tested grid.
6. Show the test PR curve and default confusion matrix. Recompute precision and recall aloud.
7. Show the top-100 review row and one limitation.

For a live rerun from the project folder:

```bash
python scripts/run_phase1.py
python -m unittest discover -s tests -v
```

The first command regenerates results and charts. It does not automatically rebuild the report or slides. If any experiment choice changes, rebuild the dependent artifacts and check all numbers again.

## Likely faculty follow-ups

- "Why use labels if it is unsupervised?" We need known legitimate rows for novelty-detection training. The GMM receives no labels during fitting.
- "Why choose this threshold?" It respects a predeclared empirical validation FPR budget, with an illustrative default of 0.5%.
- "Why not achieve a 0.5% test FPR exactly?" A fixed threshold can have a different false-positive rate in later data. Recalibrating on test labels would break the evaluation.
- "Why is precision@100 better than precision@50?" The second block of 50 happens to contain more fraud. Cumulative precision need not be monotonic.
- "Is K=8 optimal?" It is the minimum-BIC candidate within our four-value grid. BIC is still decreasing at that boundary.
- "Where is the autoencoder?" It belongs to Phase 2 under the course's ML/DL phase separation.

## Before the team submits

Every student should rerun or inspect the experiment, explain every default count, and answer the study-guide questions without relying on a script. Update `docs/CONTRIBUTIONS.md` only with actual contributions. Follow any faculty disclosure requirements for AI-assisted preparation. Confirm that the faculty can access the intended GitHub repository.
