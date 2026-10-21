"""Build the readable notebook source. The deliverable also includes executed outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))


md("""# Card Fraud Triage: Phase 1
**Team T011, Section A, Project 26**  
Newton School of Technology, Rishihood University

**Question:** Can a Gaussian Mixture Model trained on earlier legitimate transactions help rank later fraud for manual review?

This notebook covers the **ML phase only**. The autoencoder is Phase 2 work.
Read each explanation, run the next cell, then explain its output aloud.

**Route:** inspect data, split by time, explore training data, prepare features, fit GMMs, select K, set validation thresholds, and evaluate the final mixed test period.

Small helper functions live in `src/project.py`. You can open that file to see each calculation. No helper trains a neural network.""")
md("""## 1. Setup and choices
We make a few choices before looking at final test performance: K=1,2,4,8, diagonal covariance, and three validation FPR budgets. The illustrative default is 0.5%, meaning about five false alerts per 1,000 legitimate validation transactions. This is a classroom review policy, not a real bank requirement.""")
code("""from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
if not (ROOT / 'src/project.py').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src/project.py').exists(), 'Open from the project folder or notebooks/.'
sys.path.insert(0, str(ROOT))
from src.project import (
    SEED, COMPONENTS, FPR_BUDGETS, DEFAULT_BUDGET, FEATURES,
    load_data, chronological_split, split_summary, make_features,
    prepare_features, fit_candidates, eda_charts, choose_thresholds,
    ranking_metrics, decision_metrics, review_capacity_table,
    result_charts, save_results,
)
FIGURES = ROOT / 'results/figures'
print('Seed:', SEED)
print('Components tested:', COMPONENTS)
print('Validation FPR budgets:', FPR_BUDGETS)
print('Illustrative default budget:', DEFAULT_BUDGET)""")
md("""## 2. Load the real data and check quality
The recommended ULB/Worldline dataset is available from Kaggle. We use the public CSV mirror documented by TensorFlow. The download script is `python scripts/download_data.py`.

We require the expected numeric columns, finite values, binary labels, and nonnegative Amount/Time. We remove exact repeated rows, keeping the first occurrence. With no transaction ID, this is a cleaning assumption. We do not remove unusual transactions merely because their values are extreme.""")
code("""df, quality = load_data(ROOT / 'data/raw/creditcard.csv')
display(pd.Series(quality, name='Measured value').to_frame())""")
md("""**Read this output:** Raw and cleaned counts differ because of exact repeats. Missing values and infinity are separate checks. The SHA256 checksum identifies the exact CSV used, making the data source reproducible.

**Dictionary:** Time is elapsed seconds, V1-V28 are provider-supplied anonymized PCA features, Amount is in EUR, and Class is 0 for legitimate and 1 for fraud. We cannot assign business meanings to V1-V28.""")
md("""## 3. Chronological train, validation and test sets
We use approximately 60% earlier rows for training, the next 20% for validation, and the latest 20% for test. Equal timestamps stay together. This avoids training on the future.

The table reports base rates for transparency. Feature exploration uses training rows only. Later test labels will never choose features, K or thresholds.""")
code("""train, val, test = chronological_split(df)
splits = split_summary(train, val, test)
display(splits)
assert set(train.index).isdisjoint(val.index)
assert set(train.index).isdisjoint(test.index)
assert set(val.index).isdisjoint(test.index)
assert train.Time.max() < val.Time.min()
assert val.Time.max() < test.Time.min()
print('Time order and row separation checks passed.')""")
md("""## 4. EDA: understand the training period
We save six EDA charts. Class labels help describe training data but never become model inputs. The split chart shows only period boundaries and row counts.""")
code("eda_charts(train, splits, FIGURES)")
for stem, explanation in [
    ("01_class_balance", "Fraud is rare. The vertical axis uses a log scale so both classes are visible. A model can achieve high accuracy while finding no fraud."),
    ("02_amount_transform", "The raw amount distribution has a long right tail. log(1+Amount) compresses the range and handles zero. The right panel normalizes each class separately, so it compares shapes rather than class sizes."),
    ("03_chronological_split", "Training comes first and test comes last. We freeze model choices and validation thresholds before measuring test performance."),
    ("04_training_time_pattern", "Training transaction volume and fraud rate vary over hours. This suggests that a threshold may behave differently later. We do not infer a production seasonality model from this short window."),
    ("05_feature_distributions", "V1 and V2 are illustrative anonymous inputs. Both classes overlap. This chart displays only the central 99% training range, but the model keeps all input values."),
    ("06_feature_correlation", "This is Pearson correlation among legitimate training inputs. Low global correlation does not prove independence within every Gaussian component. We treat diagonal covariance as a simplification."),
]:
    code(f"display(Image(filename=str(FIGURES / '{stem}.png'), width=850))")
    md("**Explanation:** " + explanation)
md("""## 5. Feature engineering without leakage
We use **29 features**: V1-V28 and `LogAmount = log(1 + Amount)`.

Time is for splitting and EDA only. Class is for filtering legitimate training rows and evaluation only. StandardScaler learns the mean and standard deviation from legitimate training rows and applies those same numbers to validation and test.

The GMM fitting is unsupervised, but selecting known legitimate training rows requires labels. Describe this as legitimate-only novelty detection, rather than fully label-free preparation.""")
code("""scaler, x_train, x_val, x_test = prepare_features(train, val, test)
display(make_features(train).head())
print('Legitimate training matrix:', x_train.shape)
print('Validation matrix:', x_val.shape)
print('Test matrix:', x_test.shape)
assert 'Class' not in FEATURES and 'Time' not in FEATURES
assert len(FEATURES) == 29
np.testing.assert_allclose(x_train.mean(axis=0), 0, atol=1e-10)
print('The scaler used only legitimate training rows.')""")
md(r"""## 6. GMM theory before fitting
A Gaussian describes a cloud with a center and spread. A mixture adds several Gaussian densities:

\[p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k).\]

The weights sum to one. We use diagonal covariance, which estimates one variance per feature in each component and ignores its off-diagonal covariance.

**EM:** The E-step estimates soft component memberships. The M-step updates weights, means and variances using those memberships. Repeat until the stopping rule or iteration limit.

**Settings:** diagonal covariance, three initializations, seed 42, up to 200 iterations, tolerance 0.001, and variance regularization 0.000001. Open `fit_candidates` in `src/project.py` to connect these settings to the code.

With 29 inputs, the diagonal parameter count is $q=K(2\cdot29+1)-1$. For K=8, q=471. This is smaller than full covariance, which would require 3,719 parameters at K=8.""")
md(r"""## 7. Fit the candidates and select K
Fit all candidates to the same legitimate training rows. Report both criteria, but use minimum **training BIC among converged candidates** as the fixed selection rule:

\[\mathrm{AIC}=2q-2L,\qquad \mathrm{BIC}=q\log(N)-2L.\]

Here L is total training log likelihood, N is the number of fitted rows, and q is parameter count. Lower is better. Neither criterion directly optimizes fraud detection.""")
code("""model, baseline, selection = fit_candidates(x_train)
display(selection)
print('Selected K:', model.n_components)
print('One-Gaussian reference K:', baseline.n_components)
print('Every reported candidate converged:', selection.converged.all())""")
md("""**Interpretation:** K=8 has the lowest BIC in the tested grid. The criterion still falls at the largest K, so this is a boundary optimum. Eight is the best candidate we tried, not proof of the true number of legitimate patterns. EM convergence does not guarantee a global optimum.""")
md(r"""## 8. Anomaly score and validation thresholds
\[s(x)=-\log p(x).\]

Larger scores mean the row is less typical under the legitimate density. A density is not a fraud probability. `GaussianMixture.predict` returns component IDs, so we use `score_samples` instead.

We choose thresholds using **legitimate validation scores only**. The strict rule `score > threshold` allows at most the chosen fraction of legitimate validation rows to trigger an alert. The order statistic handles ties conservatively. This is an empirical validation control, not a guarantee about future false alerts.""")
code("""val_scores = -model.score_samples(x_val)
validation = choose_thresholds(val.Class, val_scores)
display(validation)
assert (validation.fpr <= validation.fpr_budget).all()
print('Validation false-positive budgets respected.')
print('Model, preprocessing and thresholds are now frozen.')""")
md("""## 9. Final evaluation on the later mixed test set
Only now do we score the test rows. The same scaler and features apply to the selected GMM and the one-Gaussian reference.

We report **average precision (AP)** and **trapezoidal PR AUC** separately because their integration conventions differ. We also report recall/precision at frozen validation thresholds and precision at fixed batch review capacities. Accuracy is not a selection metric.""")
code("""test_scores = -model.score_samples(x_test)
baseline_scores = -baseline.score_samples(x_test)
ranking = pd.DataFrame([
    {'model': name, **ranking_metrics(test.Class, scores)}
    for name, scores in [('selected_gmm', test_scores), ('one_gaussian', baseline_scores)]
])
operating = pd.DataFrame([
    {'fpr_budget': row.fpr_budget,
     **decision_metrics(test.Class, test_scores, row.threshold)}
    for row in validation.itertuples()
])
capacity = review_capacity_table(test.Class, test_scores)
display(ranking)
display(operating)
display(capacity)""")
md("""**Read the tables:** The test period has 74 frauds among 56,746 transactions, a 0.1304% base rate. The selected GMM's AP is about 0.1532 versus 0.0540 for one Gaussian. At the default policy it finds 37 frauds with 172 false alerts. Reviewing the top 100 test scores finds 29 frauds. These are measured results from this split.""")
code("""result_charts(selection, test.Class, test_scores, baseline_scores,
              validation, operating, capacity, FIGURES)""")
for stem, explanation in [
    ("07_aic_bic", "Both information criteria are lower at K=8 than at the smaller tested candidates. Values are divided by one million for readability. They measure legitimate density fit with a penalty."),
    ("08_test_score_distribution", "Fraud and legitimate scores overlap. The display uses log10(1+max(score,0)) only to make the long tail visible. The actual model score and threshold are unchanged."),
    ("09_precision_recall", "The selected mixture improves this test ranking over one Gaussian. The dashed prevalence line gives the baseline precision level associated with uninformed ranking. AP is a summary across cutoffs, not accuracy."),
    ("10_false_positive_tradeoffs", "A looser validation budget catches more test fraud here, with more false alerts. Actual test FPR differs from validation FPR. Do not retune the test threshold to make the rates match."),
    ("11_confusion_matrix", "At the 0.5% validation policy, TP=37, FP=172, FN=37 and TN=56,500. Recall=37/74=50%, precision=37/209=17.70%, and FPR=172/56,672=0.3035%. Colors use log counts and labels give exact counts."),
    ("12_review_capacity", "Precision@100=29/100=29%. Capacity is fixed for the entire test batch. Precision@K need not be monotonic: the next block of ranked transactions can contain a higher or lower fraud fraction."),
]:
    code(f"display(Image(filename=str(FIGURES / '{stem}.png'), width=850))")
    md("**Explanation:** " + explanation)
md("""## 10. Save reproducible outputs
We save measured tables, data checksum, software versions, source-row test scores, and a local model bundle containing the scaler, GMM and thresholds. Raw data and the binary model are excluded from Git. Only load model bundles from sources you trust.""")
code("""save_results(ROOT, quality, splits, selection, validation, operating,
             capacity, ranking, model, scaler, test.Class, test_scores)
print('Saved results, metadata and the fitted local model bundle.')""")
md("""## 11. Conclusion and limits
The selected mixture improves test AP over a single Gaussian in this run. The illustrative default finds half the test fraud, while most review alerts are legitimate. More review can recover more fraud, but increases workload.

The data covers a short historical window, includes anonymous inputs, and has only 74 test frauds. One caught fraud changes recall by about 1.35 percentage points. We cannot audit the provider's original PCA fitting. Duplicate removal is an assumption. Diagonal covariance is restrictive. BIC improves through the largest K tested. These limits prevent a claim of production readiness or a universal best model.

**Phase 2 only:** fit an autoencoder on the same legitimate training rows and compare it under the same frozen evaluation design. No DL or hybrid results are claimed in Phase 1.

**Practice:** Explain why Class is absent from features, describe both EM steps, distinguish AP from precision, and recompute all three default metrics from the confusion matrix.

## References
- [Dempster, Laird and Rubin (1977), EM](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
- [Saito and Rehmsmeier (2015), precision-recall evaluation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).
- [scikit-learn Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html) and [model selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html).
- [ULB/Worldline dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and [TensorFlow's documented data mirror](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
- Supplied course PDF, page 2: Phase 1 is ML only, with individual viva assessment.

See `docs/LITERATURE_REVIEW.md` for how each source informs our design.""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12.4'}}
dest = ROOT / 'notebooks/phase1_gmm.ipynb'
dest.parent.mkdir(exist_ok=True)
nbf.write(nb, dest)
print(f'Created {dest.name}: {len(cells)} cells')
