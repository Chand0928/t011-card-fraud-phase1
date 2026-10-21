# Understand the project, one part at a time

Read the notebook first. This guide is the shorter revision sheet. You do not need to memorize a long speech: explain the question that each step answers.

## The Python notation used in the notebook

| Notation | Meaning |
|---|---|
| `name = value` | Store a value under a name |
| `==` | Check whether two values are equal |
| `table["Class"]` | Read one column |
| `table[condition]` | Keep rows where the condition is true |
| `&` | Both conditions must be true |
| `for ... in ...` | Repeat the same steps for each listed value |
| `assert condition` | Stop if a required check fails |
| `[:100]` | Take the first 100 items |
| `:.2f` inside printed text | Show a number with two decimal places |

Chart settings change the picture, not the GMM: `figsize` sets picture size, `bins` sets histogram ranges, `color`/`cmap` sets colors, `s` sets dot size, `alpha` sets dot transparency, and `dpi` sets saved-image resolution. Titles and axis labels explain what the picture shows. `values_format=",d"` prints whole-number counts with commas in the confusion matrix; `colorbar=False` hides an extra color scale.

## 1. What problem are we solving?

Investigators cannot review every card transaction. We want unusual transactions near the front of their queue. This is **triage**: deciding what to inspect first. A review alert does not prove fraud.

**Anomaly detection** means looking for observations that do not fit the usual pattern. We fit a model of legitimate transactions and ask how unusual later transactions look under that model.

## 2. What is in the data?

One row represents one transaction. V1-V28 and Amount describe it. Class gives the correct answer: 0 is legitimate, 1 is fraud. Time tells us its position in the recording.

V1-V28 are numbers transformed by the dataset provider for privacy. Their business meanings are unavailable. Do not call a V column a location, age, or merchant type.

We check missing and repeated records. There are no missing cells. We remove 1,081 exact repeats, leaving 283,726 transactions and 473 frauds. Without transaction IDs, removing identical rows is an assumption. We state it rather than pretending we verified their identity.

## 3. Why are there three sets?

| Set | Question it answers | What happens there |
|---|---|---|
| Training | What does legitimate activity look like? | Fit the scaler and GMM using legitimate rows only |
| Validation | Where should we place the alert cutoff? | Choose cutoffs using legitimate validation scores |
| Test | How do our fixed choices work on later data? | Measure final results without changing the choices |

Earlier transactions come first. Approximately 60% go into training, then 20% validation and 20% test. Equal timestamps stay together. The split is 170,235 / 56,745 / 56,746 rows. Only the 169,893 legitimate training rows fit the model.

**Data leakage** happens when information that should belong to later evaluation influences learning or model choices. Training on test rows or choosing the cutoff from test fraud labels would cause leakage.

## 4. What does EDA tell us?

EDA means looking at counts and charts to understand the training data. Explain each chart with a concrete observation:

| Chart | What you should say |
|---|---|
| Split sizes | We use a larger earlier period for training and separate later periods for validation and test. |
| Class counts | Training contains 169,893 legitimate transactions and only 342 frauds. Fraud is rare. |
| Amount before/after log | A few large amounts stretch the original scale. The log compresses that range. |
| Transactions by hour | Transaction volume changes over the recording. The short recording cannot prove a permanent daily pattern. |
| V1/V2 dots | Some fraud overlaps legitimate activity. The display samples legitimate points for readability, but fitting uses all legitimate training rows. |
| AIC/BIC | Eight components give the smallest BIC among the four choices tested. |
| Confusion matrix | At the default cutoff, 37 frauds are found and 37 missed. There are 172 false alerts. |
| Policy tradeoff | More permissive cutoffs find more fraud here but create more legitimate alerts. |
| PR curve | The balance between precision and recall changes as the cutoff changes. |
| Top 100 | Among exactly 100 reviewed transactions, 29 are fraud and 71 are legitimate. |

## 5. Why transform and scale the inputs?

We replace Amount with **LogAmount = log(1 + Amount)**. It handles zero and compresses large values. For example, 9 becomes about 2.30 and 999 becomes about 6.91. It does not remove large purchases.

We keep V1-V28 and LogAmount: 29 inputs. Class is the answer, so it is never an input. Time is used for splitting, not for anomaly scoring.

StandardScaler centers each input and adjusts its spread:

`scaled value = (value - training mean) / training standard deviation`

The **mean** is the average. The **standard deviation** describes spread around it. Only legitimate training rows determine these numbers. Later rows use those same numbers. `fit_transform` learns and applies; `transform` only applies.

## 6. What exactly does the GMM do?

A Gaussian is a bell-shaped distribution with a center and spread. A mixture combines several Gaussians, allowing more than one center in the legitimate data. **K** is the number of components. Components are mathematical patterns, not verified customer categories.

![Teaching picture of one Gaussian and a two-component mixture](figures/gmm_teaching.png)

The picture uses invented numbers to explain the idea. On the right, the solid curve is the sum of the two weighted curves. Our actual model uses 29 inputs and eight components.

For a transaction x:

`density p(x) = sum of [component weight × that component's Gaussian density at x]`

A component's **weight** says how much it contributes to the mixture. Weights sum to one. The GMM adjusts weights, centers and spreads to make legitimate training data fit its density better.

It uses **EM**. The E-step estimates membership in each component. The M-step updates the component parameters using those estimates. It repeats those steps. A convergence flag means its stopping rule was reached, not that it found the best possible solution.

For example, a transaction can have an estimated 70% membership in one component and 30% in another. Its values then influence both updates, with more weight in the first. This is an illustration of soft membership, not a measured transaction from our run.

**Diagonal covariance** learns a separate spread for each input within a component. This is a simpler model that misses some relationships between inputs. Three initializations try three starting arrangements. The fixed seed helps reproduce them. A tiny variance addition helps avoid numerical failure when a component's spread becomes too small.

We use labels to select legitimate training rows. The GMM fitting receives no target labels. Both statements are true; do not call the full preparation completely label-free.

## 7. Why do AIC and BIC choose eight components?

More components can fit data more closely, but also introduce more adjustable numbers. AIC and BIC balance fit against complexity:

`AIC = 2q - 2L`

`BIC = q × log(N) - 2L`

N is the number of fitted training rows. q is the number of adjustable model parameters. L is the total training log likelihood: add the log densities of the fitted training examples. Higher L helps the fit; the q term penalizes complexity.

**Lower is better. We choose the smallest training BIC.** We try K=1,2,4,8, and eight wins. Because eight is the largest tested value and the scores are still falling, we cannot claim it is the best among all possible values. BIC evaluates legitimate density fit, not fraud recall.

## 8. How do scores and cutoffs work?

`anomaly score = -log p(x)`

Low density means the transaction is less typical of learned legitimate activity. The minus sign makes its anomaly score larger. In scikit-learn, `score_samples` returns log density; we put a minus sign in front. `predict` would return component IDs, which are not fraud labels.

A score is not a fraud probability. Density can exceed one, so a negative score is possible.

A **threshold** is a cutoff. If `score > threshold`, create an alert. We set the cutoff from legitimate validation scores. For a 0.5% validation false-alert budget, use their 99.5th percentile. Most legitimate validation scores fall below it. The `higher` percentile method and strict `>` rule keep the observed rate within budget.

The default cutoff is 75.8942. A larger score triggers review. The 0.5% budget is a classroom example fixed before test evaluation. A future period can produce a different false-alert rate.

## 9. Explain the results by hand

| Actual class | No alert | Alert |
|---|---:|---:|
| Legitimate | TN = 56,500 | FP = 172 |
| Fraud | FN = 37 | TP = 37 |

**Recall:** Of the actual frauds, what fraction did we find? `37 / (37 + 37) = 50%`.

**Precision:** Of the alerts, what fraction were fraud? `37 / (37 + 172) = 17.70%`.

**False-positive rate:** Of the legitimate transactions, what fraction received an alert? `172 / (172 + 56,500) = 0.3035%`.

**Workload:** `37 + 172 = 209` transactions to review. We miss half the fraud, and most alerts are legitimate.

PR AUC summarizes the precision-recall curve using trapezoidal area. Ours is **0.1496**. It does not mean 14.96% accuracy or 14.96% recall. Precision@100 instead asks a concrete capacity question: 29 of the 100 highest scores are fraud, so it is **29%**. The capacity applies to the full test batch.

## 10. Short viva practice

1. **Why use legitimate-only training?** We want to learn normal behavior without teaching known fraud as normal.
2. **Why keep Class out of the inputs?** It is the answer; including it would leak the answer.
3. **Why keep Time for splitting?** We want earlier data to train and later data to evaluate.
4. **Why avoid headline accuracy?** Predicting every test row legitimate gives about 99.87% accuracy and zero fraud recall.
5. **What is a GMM component?** One Gaussian pattern with a weight, center and spread.
6. **What are the two EM steps?** Estimate component memberships, then update parameters using them.
7. **Why choose K=8?** It has the smallest training BIC among K=1,2,4,8.
8. **Does a high anomaly score prove fraud?** No. Legitimate transactions can also look unusual.
9. **Why choose the cutoff on validation?** Test results must evaluate choices already made.
10. **Is precision the same as one minus FPR?** No. Their denominators are alerts and legitimate transactions, respectively.
11. **What is the main result?** Thirty-seven of 74 test frauds found, with 172 false alerts at the default cutoff.
12. **What is one important limit?** Only 74 test frauds, a short old dataset, hidden feature meanings, or a simplified density model. Explain how the chosen limit affects the conclusion.
13. **Where is the autoencoder?** Phase 2. Phase 1 evaluates machine learning only.

Each student should be able to explain the entire notebook. Record actual contributions after studying and reviewing it; the allocation email does not prove contributions.
