# Phase 1 requirements and evidence

This checklist uses page 2 of the supplied course PDF and the Project 26 description supplied in chat. The PDF is assessment reference material. The user's request limits the implementation to Phase 1, so the Phase 2 section does not add DL work to this submission.

The PDF lists nine Phase 1 areas. It gives no numerical grading weights, minimum paper count, report length, or exact evaluation date. We do not invent those requirements or guarantee a grade.

| Required area | Concrete evidence | What the student should explain |
|---|---|---|
| Literature review | Notebook 01, `docs/LITERATURE_REVIEW.md`, report Section II, slide 2 | How EM, PR evaluation and GMM model selection inform our choices, with limits |
| Dataset quality and EDA | Notebook 02, `results/data_quality.json`, `results/splits.csv`, six EDA charts | Source, column meanings, missing/infinite values, exact repeats, time split and base rates |
| Feature engineering | Notebook 03, `results/feature_columns.json`, saved scaler | LogAmount, 29 inputs, exclusion of Class/Time, legitimate-training-only fitting |
| Theoretical rigor | Notebook 04, root study guide, report GMM/EM equations | Mixture weights, memberships, EM updates, covariance, likelihood, AIC/BIC, negative log density |
| Model application | Notebooks 04-05, convergence records, model settings, ranking and policy tables | The fixed K rule, score direction, validation cutoffs, final test results and one-Gaussian reference |
| GitHub repository and code quality | Ordered notebooks, root README, pinned libraries, inline checks, `FILE_MAP.md`, `docs/CODE_GUIDE.md` | Reproduce the run, explain checks, keep code/data separation and avoid leakage |
| Project presentation | `Phase1_Presentation_Final.pptx` (the final deck supplied by Mayank) | Present the same methods and results that the notebooks measured |
| Viva voce | `STUDY_GUIDE.md`, `docs/DEMO_AND_VIVA.md`, notebook checkpoints | Each member explains the full project, metrics, assumptions and limitations individually |
| LaTeX/IEEE report | `report/Phase1_Report.tex`, official `IEEEtran.cls`, compiled report PDF | Follow the written methods, tables, equations, citations and limitations |

## Project 26 requirements

| Assignment detail | Where it is implemented |
|---|---|
| Fit primarily/only on legitimate transactions | Notebooks 03 and 04: exactly the legitimate training subset |
| GMM density anomaly score | Notebook 05: `-model.score_samples(...)` |
| Tune component count with AIC/BIC | Notebook 04: K=1,2,4,8; report both; choose minimum training BIC |
| Chronological split when Time is available | Notebook 02: approximately 60/20/20, equal timestamps kept together |
| Fit preprocessing only on legitimate training rows | Notebook 03: fit scaler once, transform later periods |
| Choose thresholds on validation | Notebook 05: legitimate validation percentiles, fixed before test scoring |
| Preserve a mixed test period | Notebook 02 saves both classes; notebook 05 evaluates them unchanged |
| Report PR AUC | Notebook 05: trapezoidal PR area, same definition for both models |
| Recall at fixed FPR budgets | Notebook 05: validation budgets 0.1%, 0.5%, 1%; report actual test FPR separately |
| Precision at review capacity | Notebook 05: headline precision@100 and supporting 50/200/500 table |
| Document false-positive tradeoffs | Policy table, count chart, confusion matrix and report results |
| Explain limitations and Phase 2 gap | All final materials: density assumptions, data limits and future autoencoder comparison |

## What changed from the earlier versions?

The analysis now has five separate notebooks with visible code and simple explanations. The current package restores a fuller literature review, correlation EDA, model/convergence settings, a one-Gaussian reference, score-overlap diagnostics, several review capacities and inline consistency checks. These support the same experiment rather than introducing another model family.

The project uses one explicit PR-area definition, trapezoidal PR AUC. Average precision is another valid summary but is not required in the supplied assignment; omitting that extra definition does not omit the required PR AUC result.

## Prepared artifacts and remaining team actions

All nine areas have prepared evidence. The team still needs to review the materials, practice the live demonstration, confirm the recorded responsibilities, ensure faculty access to the repository and follow `docs/SUBMISSION.md`. An executed notebook cannot establish a person's understanding or create contribution evidence for them. No faculty submission or invitation is claimed.
