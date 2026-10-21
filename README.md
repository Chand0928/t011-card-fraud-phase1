# Card Fraud Triage - Phase 1

**Team T011, Section A, Project 26**

We learn what legitimate card transactions look like and give unusual transactions higher scores for human review. Phase 1 implements a Gaussian Mixture Model (GMM). The autoencoder belongs to Phase 2.

## Start here

**[Open the study index](index.html)** for readable copies of all five notebooks. Each notebook explains what we do, why we do it, what the code means and how to read the output.

| Order | Notebook | Purpose |
|---|---|---|
| 01 | [Project and literature](notebooks/01_Project_and_Literature.ipynb) | Problem, sources, decisions and Phase 1 scope |
| 02 | [Data quality and EDA](notebooks/02_Data_Quality_and_EDA.ipynb) | Read, validate, clean, split and plot training data |
| 03 | [Feature engineering](notebooks/03_Feature_Engineering.ipynb) | Build 29 inputs and fit the scaler on legitimate training rows |
| 04 | [GMM theory and training](notebooks/04_GMM_Training.ipynb) | Explain EM, fit four candidates, check convergence and choose K |
| 05 | [Thresholds and evaluation](notebooks/05_Evaluation_and_Thresholds.ipynb) | Validation cutoffs, mixed-test results, reference comparison and review tradeoffs |

Notebook 01 is for reading. Run 02-05 in order. Each code notebook loads saved files from the previous stage; no shared Python session is needed. All analysis code remains visible inside the notebooks. [Notebook inputs and outputs](notebooks/README.md) explains the handoffs.

## Share the reading website

`index.html` is the website homepage. It links to all five HTML notebooks, with previous/next navigation, the report and the slides. The page works locally and with a GitHub Pages project URL.

For deployment, use **Settings → Pages → Deploy from a branch → main → /(root) → Save**. See [the short deployment guide](docs/GITHUB_PAGES.md). No build command is needed. You can also upload the contents of the [prepared website ZIP](output/GitHub_Pages_Site.zip) into a separate Pages repository.

## Does this cover every Phase 1 area?

**[The coverage checklist](docs/PHASE1_CHECKLIST.md)** maps all nine course assessment areas and the Project 26 details to specific evidence. It includes literature, quality/EDA, feature engineering, theory, model application, code quality, presentation, viva preparation and the LaTeX/IEEE report.

The prepared materials cover those areas. Students must still study, demonstrate understanding, record actual contributions and complete the faculty's submission process. No grade or individual viva outcome is guaranteed by generated artifacts.

## Run the project

Use Python 3.12. From this project folder:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/` in VS Code or Jupyter. Read 01, then run 02, 03, 04 and 05 in order. For each code notebook, select the Python environment where you installed the packages, restart the kernel and choose **Run All**. Wait until it finishes before proceeding. There is no separate script to run or study. If the raw CSV is missing, notebook 02 downloads the documented public mirror.

**[FILE_MAP.md](FILE_MAP.md)** shows exactly which notebook produces each saved result and chart. **[The code guide](docs/CODE_GUIDE.md)** explains the standard functions and file formats. The supplied HTML copies contain the verified run; notebook outputs update when you rerun.

## The measured result

The test period has 74 frauds among 56,746 transactions. The default cutoff finds **37 frauds**, misses **37**, and creates **172 legitimate false alerts**: 209 reviews, 50% recall and 17.70% precision. The top 100 scores contain 29 frauds.

The selected eight-component GMM has test trapezoidal PR AUC **0.1496**, compared with **0.0527** for a one-Gaussian reference under the same data and scaler. These results describe this split, not production performance.

## Deliverables

- [IEEE report PDF](output/Phase1_Report.pdf) and [editable LaTeX source](report/Phase1_Report.tex), plus an [Overleaf-ready source ZIP](output/LaTeX_Report_Source.zip).
- [Presentation](output/Phase1_Presentation_Complete.pptx) and [PDF copy](output/Phase1_Presentation_Complete.pdf), with source notes in the PPTX.
- [Study and viva guide](STUDY_GUIDE.md), [live demonstration guide](docs/DEMO_AND_VIVA.md) and [literature review](docs/LITERATURE_REVIEW.md).
- [Actual contribution record](docs/TEAM_CONTRIBUTIONS.md); students must fill it honestly after their work.
- `results/`: source checksum, split counts, features, convergence/settings, thresholds, model comparison, capacities, test scores and software versions.
- `figures/`: thirteen data charts and one clearly labeled GMM teaching picture.

Raw data, intermediate split tables, scaled arrays and binary model bundles stay local and are excluded from Git. The notebooks recreate them. The supplied course PDF remains in the root as the assessment reference.

[Private GitHub repository](https://github.com/mitul-bhatia/t011-card-fraud-phase1). Ensure faculty/team access before sharing its link. AI assistance prepared these materials; each student must review and understand them.
