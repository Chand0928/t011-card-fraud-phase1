# Study guide: explain the whole project simply

## 1. The problem in one minute

There are many legitimate card transactions and very few frauds. An investigator cannot review every transaction. We learn what legitimate transactions usually look like, give unusual transactions a large score, and place them earlier in a review queue.

Our Phase 1 model is a Gaussian Mixture Model. Its job is to describe the legitimate transaction distribution. We test whether fraud tends to receive high anomaly scores. Unusual legitimate transactions also exist, and some fraud looks normal, so this approach makes both types of mistake.

## 2. The data and cleaning

The raw file has 284,807 transactions, including 492 frauds. There are 31 columns: Time, V1-V28, Amount and Class. Class=1 means fraud. V1-V28 are already anonymized PCA features. We cannot say that V14 is a merchant category or invent any other business meaning.

We check numeric columns, missing values, infinity, label values and nonnegative Time/Amount. There are no missing or infinite cells. We remove 1,081 exact repeated rows and keep their first occurrence. That also removes 19 repeated fraud rows, leaving 473 frauds in 283,726 rows. Without transaction IDs we cannot prove that identical rows are duplicate purchases. State this cleaning assumption clearly.

## 3. Why three time periods?

Training comes first, validation next and test last. Approximate sizes are 60%, 20%, 20%. We keep identical Time values within the same split. The split boundaries are 120,393 and 145,234 seconds since the first transaction.

| Set | What we do | What we do not do |
|---|---|---|
| Training | Inspect feature patterns. Filter legitimate rows. Fit scaler and GMM. Compare training AIC/BIC. | Fit on fraud rows |
| Validation | Set anomaly thresholds from legitimate scores. Measure validation tradeoffs. | Refit the scaler or GMM |
| Test | Evaluate frozen choices once. Draw final result diagnostics. | Choose K, features or thresholds from test outcomes |

We use labels to select legitimate training rows, so the setup assumes access to known legitimate history. The GMM fitting itself is unsupervised because it receives no target labels. This is often called novelty detection or a semi-supervised anomaly-detection setup. It is not fully label-free data preparation.

## 4. Feature engineering

We keep V1-V28. We replace Amount with `LogAmount = log(1 + Amount)` and then standardize all 29 inputs. Time controls ordering but does not enter the model. Class never enters the feature matrix.

**Why the log?** Amount can span a very large range. For example, Amount=0 becomes 0, Amount=9 becomes about 2.30, and Amount=999 becomes about 6.91. This compresses a long right tail without deleting high amounts.

**Why scale?** StandardScaler transforms a feature using `z = (x - training_mean) / training_std`. Its mean and standard deviation come only from legitimate training rows. All later rows use the same saved values. Scaling gives a consistent numerical scale and stabilizes regularized fitting. A fully unconstrained, exactly fitted density under an invertible linear transformation has stronger invariance properties, so avoid claiming that scaling alone guarantees better fraud performance.

**Why no new PCA?** V1-V28 already come from the provider's PCA. We keep the setup simple and retain every supplied component. Global PCA decorrelation does not guarantee independence inside every mixture component or after filtering/time splitting.

## 5. One Gaussian, then a mixture

A one-dimensional Gaussian is a bell-shaped curve. Its mean locates the center and its variance controls the spread. In many dimensions it describes a cloud of points.

Legitimate spending can have several different patterns. A single Gaussian puts one cloud around all of them. A GMM adds K Gaussian densities with learned weights:

`p(x) = sum over k of [weight_k * Gaussian(x; mean_k, covariance_k)]`.

The weights are nonnegative and sum to one. Each component has a center and spread. These are mathematical groups, not verified customer types.

**Diagonal covariance** gives every feature its own variance within a component, with zero off-diagonal covariance there. This reduces parameter count. The mixture as a whole can still show dependencies because component membership changes across observations. Diagonal covariance is an approximation, even with PCA inputs.

With d=29, diagonal GMM parameter count is `q = (K-1) + K*d + K*d = K*(2d+1)-1`. K=8 therefore has 471 free parameters. A full-covariance K=8 model would have 3,719. We choose diagonal covariance before final evaluation for a simpler baseline.

## 6. EM in plain language

We do not know which Gaussian generated each transaction. EM repeats two steps:

1. **E-step:** estimate each transaction's probability of belonging to each component. These soft memberships are called responsibilities.
2. **M-step:** update weights, centers and variances using those responsibilities.

For transaction i and component k:

`responsibility_ik = weight_k * Gaussian_k(x_i) / sum_j[weight_j * Gaussian_j(x_i)]`.

Let `N_k` be the sum of responsibilities for component k. Then its weight is `N_k / N`, its mean is a responsibility-weighted average of the inputs, and each diagonal variance is a responsibility-weighted squared distance from that mean. We add `reg_covar=1e-6` to keep variances away from zero.

We allow 200 EM iterations per start, stop at the configured tolerance, and try three initializations with seed 42. scikit-learn retains the best fit across starts. All reported candidates converged. Convergence means the stopping rule was satisfied, not that the global optimum or best fraud model was found.

## 7. Choosing component count

Candidates are K=1,2,4,8, all fitted to the same 169,893 legitimate training rows.

`AIC = 2*q - 2*L`  
`BIC = q*log(N) - 2*L`

L is total training log likelihood, q is the parameter count and N is the number of fitted rows. Lower is better. Higher likelihood helps fit, but more parameters incur a penalty. These are training-set information criteria, not validation fraud metrics.

Minimum training BIC selects K=8. Both AIC and BIC keep improving through K=8, the largest value tried. We do not claim that eight is the true number of spending patterns. Trying larger K would be a new experiment requiring careful validation.

## 8. From density to anomaly score

`score(x) = -log(p(x))`.

A low legitimate density gives a high score. The code uses `-model.score_samples(X)`. `score_samples` returns log density for each row. `predict` would return component membership, which is not a fraud label.

For illustration, density 0.01 gives a score around 4.61 and density 0.0001 gives about 9.21. The second transaction is less typical under this model. A continuous density is not a probability at a single point and can exceed one. Scores can therefore be negative. A score is not a calibrated probability of fraud.

## 9. Choosing thresholds without test leakage

We specify validation false-positive budgets of 0.1%, 0.5% and 1%. A false positive is a legitimate transaction flagged for review.

For the 0.5% policy, validation has 56,688 legitimate rows, so at most `floor(0.005 * 56,688) = 283` may exceed the threshold. We sort their scores and place the threshold at the required order statistic. The rule is strictly `score > threshold`. Ties may make the actual number smaller.

The threshold is 75.8942. Validation has 283 false alerts and catches 29 of 57 frauds. We carry this unchanged threshold into the test period. It then gives 172 false alerts and catches 37 of 74 frauds. We do not force the test FPR to match 0.5% by changing the threshold after seeing test labels.

The default 0.5% is an illustrative classroom policy chosen before test evaluation. It is not a bank-approved operating target, and we did not select it by maximizing test precision or recall.

## 10. The confusion matrix and exact metric calculations

At the 0.5% validation policy:

| Actual / decision | No alert | Review alert |
|---|---:|---:|
| Legitimate | TN = 56,500 | FP = 172 |
| Fraud | FN = 37 | TP = 37 |

- **Recall** = TP / (TP + FN) = 37 / 74 = 50%. We find half the fraud.
- **Precision** = TP / (TP + FP) = 37 / 209 = 17.70%. About one in 5.6 reviewed transactions is fraud.
- **False-positive rate** = FP / (FP + TN) = 172 / 56,672 = 0.3035%. This denominator contains legitimate rows only.
- **Review workload** = TP + FP = 209 transactions.

If we predicted every test row legitimate, the accuracy would be about 99.87% and fraud recall would be zero. This arithmetic explains why accuracy cannot be the headline evaluation measure here.

## 11. Ranking metrics and capacity

The PR curve shows precision against recall across possible score cutoffs. It describes the ranking and does not itself choose our operating threshold.

Average precision (AP) weights each precision by the increase in recall: `AP = sum[(recall_n - recall_(n-1)) * precision_n]` as recall increases. scikit-learn's AP is a step-weighted summary. Trapezoidal PR AUC linearly connects PR points before integration. We report both: 0.1532 AP and 0.1496 trapezoidal PR AUC. AP is not the fraction of all test rows classified correctly.

The one-Gaussian reference scores the same test rows with the same scaler. Its AP is 0.0540. The selected GMM ranks fraud better in this run. This is evidence about this split, not a guarantee that more components always help.

For precision@100, sort the test batch by descending anomaly score and review exactly 100 rows. There are 29 frauds, so precision@100=29% and recall@100=29/74=39.19%. We resolve score ties by existing chronological order. A fixed-capacity queue differs from a fixed threshold: threshold policies may produce any number of alerts.

In our results precision@50 is 16%, precision@100 is 29%, precision@200 is 18.5%, and precision@500 is 10.6%. A larger K does not guarantee monotonically decreasing precision@K, because the next block of scores can contain more or fewer frauds.

## 12. Read the 12 charts

| Figure | What to look at | What you can explain |
|---|---|---|
| 01 Class balance | Logarithmic count axis | Fraud is rare. The log scale makes its count visible. |
| 02 Amount transform | Raw amount and within-class log-amount densities | Log compresses the range. Similar values can occur in both classes. |
| 03 Chronological split | Training, validation, test time spans | Earlier periods determine choices, later data evaluates them. |
| 04 Time pattern | Counts and fraud rate by training hour | The volume and base rate vary. This is descriptive, not a calibrated time model. |
| 05 V1/V2 distributions | Central 99% training range only | Anonymous features can overlap. Plot limits do not clip model inputs. |
| 06 Correlation | Legitimate training feature pairs | Global correlation is descriptive and cannot prove component-wise independence. |
| 07 AIC/BIC | Lower criterion values through K=8 | Eight wins within the small grid. This is a boundary optimum. |
| 08 Score distribution | Within-class overlap | Some legitimate rows look anomalous and some fraud looks normal. Display transformation does not change scoring. |
| 09 PR curve | Precision versus recall and the prevalence line | Ranking performance differs from one operating point. |
| 10 False-positive tradeoffs | Test recall and actual FPR versus validation budget | More alerts can improve recall. Validation and test rates differ. |
| 11 Confusion matrix | The four exact counts | Recompute precision, recall and FPR by hand. Color uses log(1+count), labels show raw counts. |
| 12 Review capacity | Precision for different batch capacities | Review policy affects workload and alert quality. |

## 13. Viva questions with short answers

1. **Why anomaly detection?** We model legitimate behavior and investigate low-density rows instead of learning a direct fraud classifier.
2. **Why legitimate-only training?** It prevents known fraud from becoming part of the normal density we want to learn.
3. **Do we use labels?** Yes, to identify legitimate training examples and evaluate performance. The GMM receives only feature values.
4. **Is Class a feature?** No. Using it would leak the answer.
5. **Why chronological splitting?** We want evaluation on later transactions, matching the direction of future use.
6. **Why no SMOTE?** We do not train a supervised fraud classifier. We preserve rare-fraud rates in validation and test.
7. **What is a component?** One Gaussian density with a weight, mean and covariance. It is not a known fraud or customer category.
8. **Why three initializations?** EM depends on its starting point. We keep the best likelihood among three starts.
9. **Why regularization?** Very small estimated variances can make a Gaussian density unstable. A small positive addition limits that problem.
10. **Why BIC?** It compares training density fit while penalizing parameter count. It does not optimize fraud recall.
11. **Does lower BIC prove better fraud detection?** No. Legitimate density fit and fraud separation are different objectives.
12. **Why K=8?** It has the lowest BIC among the four converged candidates we tested.
13. **Why negative log likelihood?** It makes less typical transactions receive larger anomaly scores and avoids directly working with tiny density products.
14. **Is the score a probability?** No. It is a density-derived ranking score and is not calibrated to fraud probability.
15. **Why not GaussianMixture.predict?** It predicts component IDs, not fraud decisions.
16. **Why not report accuracy?** Rare fraud allows high accuracy with zero fraud detection.
17. **What is AP?** A recall-weighted summary of precision across score cutoffs.
18. **What is the difference between FPR and one minus precision?** FPR divides false alerts by all legitimate rows. One minus precision divides false alerts by all alerts.
19. **Can a validation FPR budget guarantee a future FPR?** No. Score distributions can change over time.
20. **Why is precision@100 different from threshold precision?** The first fixes the number reviewed, while the second fixes a score cutoff.
21. **What is the main result?** At the illustrative default we find 37 of 74 frauds with 172 false alerts.
22. **What is the main weakness?** Half the test fraud is missed at that policy, and most alerts are legitimate.
23. **Would you deploy this?** Deployment evidence is absent. This is an educational baseline requiring newer data, monitoring and operational validation.
24. **Why no autoencoder?** The course specifies ML only for Phase 1. The autoencoder belongs to Phase 2.
25. **What would a fair Phase 2 comparison keep fixed?** The legitimate training rows, preprocessing protocol, mixed test split, validation-only threshold policy and evaluation metrics. No results exist yet.
26. **How certain is test recall?** There are only 74 frauds. One transaction changes recall by about 1.35 percentage points. We do not claim repeated-window statistical certainty.
27. **What does the seed guarantee?** It helps reproduce initialization for the same software setup. It does not make the estimate unbiased or universally stable.
28. **What about duplicate rows?** We remove exact full-row repeats under a stated assumption. Transaction IDs are unavailable to prove identity.

## 14. Three-minute practice

Explain the goal in one sentence. Point to the split table. Explain the 29 features and why Class is absent. Describe the EM steps. Explain K selection using the AIC/BIC plot. Read the PR curve. Recompute the default confusion-matrix metrics. Finish by stating one limitation and the Phase 2 boundary.
