# Read the notebook code in simple steps

A **variable** is a name for a value, table, array or model. A **function call** asks a library to do a specific operation. The analysis uses standard library calls directly; there are no custom helper functions or separate analysis scripts.

Read 01, then run 02–05 in order. Run cells from top to bottom. Read the explanation above each cell and the interpretation below it. `FILE_MAP.md` shows the exact source and destination files.

## Names used throughout

| Name | Meaning |
|---|---|
| `df` | Full cleaned transaction table |
| `train`, `validation`, `test` | Earlier, middle and later transaction tables |
| `normal_train` | Only training rows with Class equal to 0 |
| `features` | The 29 column names used as inputs |
| `X_train_scaled` | Legitimate training input numbers after scaling |
| `X_validation_scaled`, `X_test_scaled` | Later input numbers using the same training scaler |
| `y_validation`, `y_test` | Correct answers, kept separate from model inputs |
| `model` | Already fitted GMM selected using training BIC |
| `test_scores` | One anomaly score per test row; larger means less typical |
| `default_threshold` | Validation cutoff for the illustrative 0.5% FPR budget |
| `tp`, `fp`, `fn`, `tn` | Fraud found, false alerts, missed fraud, legitimate rows without an alert |

## The operations you need to know

| Code | Plain meaning | Used in |
|---|---|---|
| `pd.read_csv(path)` | Read a CSV into a table | 02, 03 |
| `df.head()` / `df.shape` | Show the first rows / number of rows and columns | 02 |
| `df.isna().sum().sum()` | Count missing cells across the table | 02 |
| `np.isfinite(values).all()` | Check every number is finite | 02, 03, 05 |
| `df.duplicated().sum()` | Count exact repeated rows | 02 |
| `df.drop_duplicates()` | Keep only the first copy of each exact row | 02 |
| `df.sort_values("Time")` | Order earlier transactions before later ones | 02 |
| `df.iloc[position]` | Read a row by its numbered position, starting at 0 | 02 |
| `table[table["Class"] == 0]` | Keep rows whose correct class is legitimate | 03 |
| `table[features]` | Keep only the input columns | 03 |
| `np.log1p(values)` | Calculate log(1+value), accepting zero | 02, 03 |
| `StandardScaler()` | Create an object that will learn column means and spreads | 03 |
| `scaler.fit_transform(train_inputs)` | Learn from legitimate training inputs and scale them | 03 |
| `scaler.transform(later_inputs)` | Apply the same learned scaling to later inputs | 03 |
| `GaussianMixture(...)` | Create a GMM with the stated settings | 04 |
| `gmm.fit(X_train_scaled)` | Learn normal patterns; receives no fraud targets | 04 |
| `gmm.aic(...)`, `gmm.bic(...)` | Calculate training fit criteria with complexity penalties | 04 |
| `selection["BIC"].idxmin()` | Find the table row with smallest BIC | 04 |
| `-model.score_samples(inputs)` | Return anomaly scores, one per input row | 05 |
| `np.percentile(scores, 99.5, method="higher")` | Take an observed cutoff at the 99.5th percentile | 05 |
| `scores > cutoff` | Return True for alert and False for no alert | 05 |
| `np.mean(condition)` | Fraction of rows where the condition is True | 05 |
| `confusion_matrix(...).ravel()` | Count TN, FP, FN, TP in that order | 05 |
| `precision_recall_curve(labels, scores)` | Calculate precision and recall over cutoffs | 05 |
| `auc(recall, precision)` | Add trapezoid areas under that curve | 05 |
| `np.argsort(-scores, kind="stable")` | Give row positions in highest-score-first order; preserve order for ties | 05 |
| `ranked_positions[:100]` | Take the first 100 ranked positions | 05 |

`for` repeats the same short block for each K, budget or capacity. A list stores values in order. A dictionary stores values under names, such as `models[8]`. `pd.DataFrame(rows)` turns the collected rows into a table. A boolean condition is True or False. Python counts True as 1 and False as 0 when adding or averaging it.

`assert condition` stops if a required check fails. It checks data/method consistency; it does not force the model to achieve a particular result. `random_state=42` fixes the random seed. `round_trip` when reading CSV preserves saved floating-point values. `kind="stable"` keeps the original order when sorting equal values.

## Reading and saving files

| Call | Purpose |
|---|---|
| `Path(...)` | Refer to a file/folder; `.exists()` checks it and `.mkdir()` creates a folder |
| `os.chdir(ROOT)` | Make relative paths start at the project folder |
| `table.to_csv(path, index=False)` | Save a readable table without an extra row-number column |
| `np.savez_compressed(path, name=array, ...)` / `np.load(path)` | Save / read named numerical arrays |
| `joblib.dump(object, path)` / `joblib.load(path)` | Save / read the exact fitted scaler or models |
| `json.dumps(record, indent=2)` | Turn a small named record into readable JSON text |
| `Path(path).write_text(text)` / `.read_text()` | Write / read a text file |
| `hashlib.sha256(file_bytes).hexdigest()` | Make a fingerprint identifying the exact input file |

CSV is for tables, JSON for small named records, NPZ for numerical arrays, and joblib for fitted Python models. None contains another analysis program. Large data and model files are generated locally and excluded from Git.

## Reading the chart code

`plt.figure(figsize=(7,4))` starts a chart of that size. `plt.subplots(1,2)` creates two side-by-side charts. `bar` draws counts, `hist` counts values in ranges, `plot` joins points, `scatter` draws dots and `imshow` displays a matrix as colors. `xlabel`, `ylabel`, `title` and `legend` label the picture. `tight_layout` makes room for labels; `savefig` writes the PNG; `display(Image(...))` shows that saved PNG; `close` finishes the chart. Color and size choices change the picture, not the trained model.

For example, in 05 Section 5, `policy_table` contains the measured counts, `bar(...)` draws those counts and `savefig("figures/08_tradeoff.png")` saves that exact chart. The same counts are in `results/threshold_results.csv`.

A default test calculation is **recall = 37 / (37 + 37) = 50%**. The 37s come from the confusion matrix in 05 Section 3, not from constants inserted to make a result look good.
