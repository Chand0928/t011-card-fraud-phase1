# Run these notebooks in order

01 is a reading notebook. 02-05 contain executable analysis. Each starts with its own setup and loads earlier saved files; variables do not carry across notebooks.

| Notebook | Reads | Saves |
|---|---|---|
| 01 Project and literature | Supplied problem and cited sources | Explanations and evaluation plan |
| 02 Data quality and EDA | `data/raw/creditcard.csv` (downloads if missing) | Three CSV splits, data-quality record, split counts and six EDA charts |
| 03 Feature engineering | The three saved split CSVs | Scaled arrays, labels stored separately, feature names and the fitted scaler |
| 04 GMM training | Legitimate scaled training array | Selected GMM, K=1 reference, AIC/BIC, convergence and settings |
| 05 Evaluation | Scaled validation/test arrays, labels and fitted GMMs | Thresholds, PR comparison, confusion counts, policy and capacity tables, diagnostic charts and run metadata |

Open from this folder or the project root. The first setup cell handles either location. If a prerequisite file is missing, the notebook tells you which earlier notebook to run. Read the markdown before running the next cell.

Read 01. Open 02 in VS Code/Jupyter, select your project Python environment, restart its kernel, and choose **Run All**. When it finishes, repeat for 03, 04 and 05. The cells themselves save all data, model and result files. There is no separate script.

The supplied HTML files in `output/notebooks/` are reading copies of the verified run. Read `../FILE_MAP.md` for each exact filename and `../docs/CODE_GUIDE.md` for function meanings.

All test-based plots and reference comparisons happen after K and validation thresholds are fixed. Reproducing this run is not permission to repeatedly tune against its test outcomes. A changed experiment needs appropriate fresh evaluation data.

## Team responsibilities

- **Notebook 01:** shared project and literature background.
- **Notebook 02:** Keshav, data quality and EDA.
- **Notebook 03:** Vaibhav, feature engineering.
- **Notebook 04:** Mayank, GMM training.
- **Notebook 05:** H Maheshwari (Hardik), evaluation and thresholds.
- **Overall lead:** Mitul Bhatia, management and coordination across the project.

README and presentation: Keshav and Mayank. Report: all five members jointly. All five members are involved; see [the contribution record](../docs/TEAM_CONTRIBUTIONS.md).
