# Which file produces which result?

Read **01**, then run **02 → 03 → 04 → 05** in `notebooks/`. All analysis code is in those notebooks. Paths below start at the main project folder.

| Stage | Read this file | Work done in this notebook | Files passed to the next stage |
|---|---|---|---|
| 01 | Project instructions and cited papers | `01_Project_and_Literature.ipynb`: explain the problem and plan | Reading only |
| 02 | `data/raw/creditcard.csv` | `02_Data_Quality_and_EDA.ipynb`: check, remove exact repeats, split by time, draw EDA | `data/processed/train.csv`, `validation.csv`, `test.csv` |
| 03 | The three CSVs from 02 | `03_Feature_Engineering.ipynb`: create LogAmount, select legitimate training rows, scale | `data/processed/features.npz` |
| 04 | `features.npz`, array `X_train` | `04_GMM_Training.ipynb`: fit K=1,2,4,8; choose minimum training BIC | `models/gmm_models.joblib` |
| 05 | `features.npz` and `gmm_models.joblib` | `05_Evaluation_and_Thresholds.ipynb`: choose validation cutoffs, evaluate later test rows | Final tables in `results/`, charts in `figures/` |

`models/scaler.joblib`, saved by 03, stores the learned training mean and standard deviation. Notebook 03 already applies this scaler before saving `features.npz`; 04 and 05 must not scale again.

Inside `features.npz`, `X_train` is scaled legitimate training inputs; `X_validation` and `X_test` are scaled mixed inputs. `y_validation` and `y_test` are separate correct-label arrays. The X arrays have 29 columns and contain no Class. Inside `gmm_models.joblib`, `selected_gmm` is the chosen model and `one_gaussian` is the K=1 reference. These files store objects/numbers, not additional code.

## Every saved results file

All filenames in this table are inside `results/`.

| File | Produced by | Contents / use |
|---|---|---|
| `data_quality.json` | 02, save section | Raw/clean counts, repeats, missing/infinite counts, time boundaries and input checksum |
| `splits.csv` | 02, split/save sections | Training, validation and test sizes and fraud rates; report/slide split table |
| `feature_columns.json` | 03, save section | Names of the 29 model inputs |
| `gmm_selection.csv` | 04, save section | AIC, BIC, convergence and iterations for each K; choose K=8 |
| `model_settings.json` | 04, save section | Fixed settings, chosen K, training row and feature counts |
| `validation_thresholds.csv` | 05, Section 2 | Three budgets, cutoffs and actual validation false alerts |
| `threshold_results.csv` | 05, Section 5 | Test recall, precision, false alerts and FPR for those fixed cutoffs |
| `ranking_comparison.csv` | 05, Section 6 | Selected-model and one-Gaussian test PR AUC |
| `review_capacities.csv` | 05, Section 8 | Fraud found, recall and precision for 50/100/200/500 reviews |
| `test_scores.csv` | 05, Section 10 | One row per test transaction: true class and both anomaly scores |
| `summary.json` | 05, Section 10 | Default confusion counts and headline measures, plus earlier data/model counts |
| `run_info.json` | 05, Section 10 | Package versions, seed, raw checksum and test counts |

For example, **37 frauds found** comes from 05 Section 3: compare test scores with the validation cutoff, then count actual frauds receiving an alert. It is saved as `TP` in `summary.json`. **29% precision@100** comes from 05 Section 7: count 29 frauds among the 100 highest scores. It is saved in `summary.json` and the 100-review row of `review_capacities.csv`.

## Every chart

All filenames below are inside `figures/`. The notebook saves and displays the same image.

| Image | Produced by | Numbers/data used | What it helps explain |
|---|---|---|---|
| `01_split.png` | 02, Section 3 | Split row counts | Three time periods |
| `02_class_balance.png` | 02, Section 4.1 | Training Class counts | Rare fraud |
| `03_amount.png` | 02, Section 4.2 | Training Amount and log(1+Amount) | Why create LogAmount |
| `04_time.png` | 02, Section 4.3 | Training transaction counts by hour | Changing volume |
| `05_overlap.png` | 02, Section 4.4 | Training V1/V2, sampled for display | Classes can overlap |
| `11_feature_correlation.png` | 02, Section 4.5 | Legitimate training V1–V28 | Linear relationships |
| `gmm_teaching.png` | 04, theory section | Made-up one-dimensional curves | How a mixture combines bell curves; not a measured result |
| `06_aic_bic.png` | 04, Section 2 | Training AIC/BIC table | Component-count selection |
| `07_confusion.png` | 05, Section 3 | Default test confusion counts | Four possible outcomes |
| `08_tradeoff.png` | 05, Section 5 | Three rows in threshold_results.csv | Fraud found versus false alerts |
| `09_precision_recall.png` | 05, Section 6 | Test labels and both model scores | Ranking quality |
| `10_capacity.png` | 05, Section 7 | First 100 positions in score ranking | Review workload |
| `12_review_capacities.png` | 05, Section 8 | review_capacities.csv | Precision at four workloads |
| `13_score_overlap.png` | 05, Section 9 | Test scores, compressed only for drawing | Why mistakes remain |

Numbers in chart filenames preserve the existing figure names; notebook numbers show the order to study.

## Written deliverables

| File/folder | Where its content comes from |
|---|---|
| `report/Phase1_Report.tex` → `output/Phase1_Report.pdf` | Literature from 01; methods from 02–04; measured results from 05. Compile the LaTeX to create the PDF. |
| `output/LaTeX_Report_Source.zip` | A copy of the report source, IEEE class and its four figures, arranged for Overleaf |
| `output/Phase1_Presentation_Complete.pptx` and `.pdf` | The same five notebooks/results; PPTX speaker notes identify sources. Editable deck and reading copy. |
| `output/notebooks/*.html` | Reading exports of the five executed notebooks; supplied copies show the checked run |
| `output/Study_Notebook.html` | An index linking those five reading exports |
| `STUDY_GUIDE.md`, `docs/CODE_GUIDE.md`, `docs/DEMO_AND_VIVA.md` | Explanations and practice questions for the notebook method |
| `docs/LITERATURE_REVIEW.md` | Cited papers and official method/dataset references |
| `docs/PHASE1_CHECKLIST.md` | Each of the nine course assessment areas mapped to evidence |
| `docs/TEAM_CONTRIBUTIONS.md` | Blank actual-contribution record; students fill it after doing their work |

The report and slides are written deliverables prepared from the results. Running notebooks saves the analysis files; it does not automatically rewrite the report or presentation. If an experiment changes, update these documents to match it.
