# Card Fraud Triage - Phase 1

**Team T011, Section A, Project 26**

We learn what legitimate card transactions look like, then give unusual transactions a higher score for human review. Our Phase 1 model is a Gaussian Mixture Model (GMM).

## Read these in order

1. **[Open the explained notebook](output/Study_Notebook.html).** Read one step at a time. It includes the code, outputs, 10 data charts and a GMM teaching picture. No installation is needed to read it.
2. **[Read the study guide](STUDY_GUIDE.md).** It explains the terms, the important formulas, and likely viva questions.
3. Use the [report](output/Phase1_Report.pdf) and [slides](output/Phase1_Presentation.pptx) when preparing the submission.

To run the code, open **[Phase1.ipynb](Phase1.ipynb)** in VS Code or Jupyter from this project folder. All analysis code is inside this one notebook. Run its cells from top to bottom.

```bash
python -m pip install -r requirements.txt
```

The notebook downloads the dataset if it is missing. The existing local CSV is in `data/raw/creditcard.csv`.

## The result in plain words

Our test period contains 74 frauds. With the default cutoff, we find **37 frauds**, miss **37 frauds**, and falsely alert on **172 legitimate transactions**. That gives 209 transactions to review. If we review exactly the top 100 scores instead, 29 are fraud.

The autoencoder belongs to Phase 2. This project implements the GMM only.

## What each file is for

| File or folder | Purpose |
|---|---|
| `Phase1.ipynb` | The complete, runnable analysis with explanations |
| `STUDY_GUIDE.md` | Definitions, worked metric calculations and viva practice |
| `output/` | Readable notebook, report PDF and presentation |
| `figures/` | Ten data charts and the clearly labeled GMM teaching picture |
| `results/` | Four small files with the measured results |
| `report/` | Editable LaTeX report and official IEEE class |
| `requirements.txt` | Python libraries needed to run the notebook |

## Phase 1 assessment coverage

| Assessment item | Where to find it |
|---|---|
| Literature review | End of notebook and report references |
| Dataset quality and EDA | Notebook Steps 2-5 |
| Feature engineering | Steps 6-7 |
| Theoretical rigor | Steps 8-11 and study-guide formulas |
| Model application | Steps 9-16 |
| GitHub and code quality | One readable notebook, fixed seed and checks beside the relevant code |
| Presentation | `output/Phase1_Presentation.pptx`, with speaker notes |
| Individual viva | Study guide, plus explaining each notebook step aloud |
| LaTeX/IEEE report | `report/Phase1_Report.tex` and its compiled PDF |

The course PDF says Phase 1 evaluates machine learning and assesses each student's understanding individually. We use it as the assessment reference and follow your request to build Phase 1 only.

## Team and submission

[Private GitHub repository](https://github.com/mitul-bhatia/t011-card-fraud-phase1). Grant team/faculty access before sharing the link. The raw dataset stays outside Git.

The supplied allocation lists H Maheshwari (2401010174), Vaibhav (2401010486), Mayank (2401010267), Keshav (2401010222), and Mitul Bhatia (2401010279). Record each member's actual contribution after reviewing and studying the work. AI assistance prepared these materials; the team must be able to explain them.
