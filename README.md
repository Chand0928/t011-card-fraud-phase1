<div align="center">

# Card Fraud Triage with Gaussian Mixture Models

**Finding the needle in 57,000 transactions, without ever being shown a needle.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-5_notebooks-F37626?logo=jupyter&logoColor=white)
![Model](https://img.shields.io/badge/Model-GMM_(K%3D8)-2E7D32)
![Phase](https://img.shields.io/badge/Phase-1_of_2-6A1B9A)

Team T011 · Section A · Project 26 · ML/DL Course Project

</div>

---

## The problem in one picture

Out of every **1,000** card transactions in this dataset, fewer than **2** are fraud. A bank can't check them all by hand. So we asked a simpler question:

> **If an analyst can only check a few transactions today, which ones should they check first?**

```mermaid
flowchart LR
    A["57,000 transactions<br/>arrive"] --> B["GMM gives each one<br/>an anomaly score"]
    B --> C["Sort by score,<br/>highest first"]
    C --> D["Analyst reviews<br/>the top of the list"]
    D --> E["29 frauds found<br/>in the top 100"]

    classDef step fill:#E3F2FD,stroke:#1565C0,color:#0D2A4A
    classDef win fill:#E8F5E9,stroke:#2E7D32,color:#1B3D1F
    class A,B,C,D step
    class E win
```

The interesting part is that the model **never sees a single fraud during training**. It only learns what *normal* spending looks like, and anything that doesn't fit gets a high score.

---

## Results at a glance

<table>
<tr>
<td align="center" width="25%"><h2>37 / 74</h2>frauds caught<br/>at the default cutoff</td>
<td align="center" width="25%"><h2>29</h2>frauds in the<br/>top 100 alerts</td>
<td align="center" width="25%"><h2>0.30%</h2>of legitimate users<br/>wrongly flagged</td>
<td align="center" width="25%"><h2>~3×</h2>better ranking than<br/>a single Gaussian</td>
</tr>
</table>

| Metric (test period) | Value |
|---|---|
| Transactions tested | 56,746 (74 frauds) |
| Recall at default cutoff | **50.0%** |
| Precision at default cutoff | 17.7% |
| PR AUC: our 8-component GMM | **0.1496** |
| PR AUC: 1-Gaussian baseline | 0.0527 |

Picking 100 transactions at random would find about **0.13** frauds. Our top 100 has **29**.

> These numbers come from our time-based split. They show how the method behaves here, not how it would perform inside a real bank.

---

## How a GMM spots fraud

Normal spending isn't just one pattern. Some people buy coffee, some pay rent, some shop online at midnight. One bell curve can't describe all of that, but a **mixture** of several bell curves can.

<p align="center">
  <img src="figures/gmm_teaching.png" width="680" alt="One Gaussian vs a mixture of Gaussians"/>
</p>

<p align="center"><i>A single Gaussian (blue) puts its peak where almost no data lives. A mixture (orange) can follow several groups at once.</i></p>

Our model uses **8** of these bell curves across 29 features. To score a new transaction:

```mermaid
flowchart LR
    T["New transaction"] --> Q{"How well does it fit<br/>any of the 8 normal<br/>clusters?"}
    Q -- "Fits well" --> L["Low score<br/>probably fine"]
    Q -- "Fits badly" --> H["High score<br/>send for review"]

    classDef ok fill:#E8F5E9,stroke:#2E7D32,color:#1B3D1F
    classDef bad fill:#FFEBEE,stroke:#C62828,color:#4A1414
    classDef neutral fill:#F5F5F5,stroke:#616161,color:#212121
    class L ok
    class H bad
    class T,Q neutral
```

In maths, the score is just the negative log-likelihood:

$$\text{score}(x) = -\log p(x) \quad\text{where}\quad p(x) = \sum_{k=1}^{8} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Each cluster has a weight $\pi_k$, a centre $\mu_k$ and a spread $\Sigma_k$. The **EM algorithm** learns all of them from the legitimate training data.

---

## The full pipeline

```mermaid
flowchart TD
    subgraph S1["1. Data"]
        A["Raw CSV<br/>284,807 rows · 492 frauds"] --> B["Quality checks<br/>0 missing · 0 infinite"]
        B --> C["Remove 1,081<br/>exact duplicates"]
    end

    subgraph S2["2. Split by time"]
        C --> D["Train 60%<br/>earliest"]
        C --> E["Validation 20%<br/>middle"]
        C --> F["Test 20%<br/>latest"]
    end

    subgraph S3["3. Features"]
        D --> G["V1–V28 + log(1 + Amount)<br/>29 inputs"]
        G --> H["StandardScaler<br/>fitted on legit train only"]
    end

    subgraph S4["4. Model"]
        H --> I["Train GMMs<br/>K = 1, 2, 4, 8"]
        I --> J["Pick K = 8<br/>lowest BIC"]
    end

    subgraph S5["5. Decide and evaluate"]
        J --> K["Set cutoff on validation<br/>max 0.5% false alerts"]
        E --> K
        K --> L["Final test<br/>run once"]
        F --> L
    end

    classDef data fill:#E3F2FD,stroke:#1565C0,color:#0D2A4A
    classDef split fill:#FFF8E1,stroke:#F9A825,color:#4A3A00
    classDef feat fill:#F3E5F5,stroke:#6A1B9A,color:#2E0F3D
    classDef model fill:#E8F5E9,stroke:#2E7D32,color:#1B3D1F
    classDef eval fill:#FFEBEE,stroke:#C62828,color:#4A1414
    class A,B,C data
    class D,E,F split
    class G,H feat
    class I,J model
    class K,L eval
```

Here's what happens at each step.

### 1. Data

We use the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from ULB / Worldline: two days of European card transactions.

| Column | Meaning |
|---|---|
| `V1` – `V28` | Anonymised PCA features (the bank hides the original columns) |
| `Amount` | Transaction amount in EUR |
| `Time` | Seconds since the first transaction |
| `Class` | 0 = legitimate, 1 = fraud |

<p align="center">
  <img src="figures/02_class_balance.png" width="560" alt="Class balance"/>
</p>
<p align="center"><i>Fraud makes up only about 0.17% of all transactions. This is why accuracy is useless here: a model that always says "not fraud" would be 99.8% accurate.</i></p>

### 2. Split by time, not randomly

In real life a model is trained on the past and used on the future. A random split would let the model peek at later transactions, so we split in time order.

```mermaid
flowchart LR
    subgraph TL["Day 1  ─────────────────────────────►  Day 2"]
        direction LR
        TR["TRAINING · 60%<br/>170,235 rows · 342 frauds<br/><i>learn what normal looks like</i>"]
        VA["VALIDATION · 20%<br/>56,745 rows · 57 frauds<br/><i>choose the cutoff</i>"]
        TE["TEST · 20%<br/>56,746 rows · 74 frauds<br/><i>final check, run once</i>"]
        TR --> VA --> TE
    end

    classDef tr fill:#E3F2FD,stroke:#1565C0,color:#0D2A4A
    classDef va fill:#FFF8E1,stroke:#F9A825,color:#4A3A00
    classDef te fill:#FFEBEE,stroke:#C62828,color:#4A1414
    class TR tr
    class VA va
    class TE te
```

Only the **legitimate** rows from training (169,893 of them) are used to fit the model.

### 3. Feature engineering

Transaction amounts are heavily skewed: most are small and a few are huge. Taking `log(1 + Amount)` spreads them out into a much friendlier shape.

<p align="center">
  <img src="figures/03_amount.png" width="760" alt="Amount before and after log transform"/>
</p>

Final inputs: **`V1`–`V28` + `LogAmount` = 29 features.** We leave `Time` out on purpose, because "seconds since the start of the file" means nothing for a new transaction.

The `StandardScaler` is fitted **only on legitimate training rows**, then reused as-is for validation and test, so nothing leaks in from the future.

### 4. Training and choosing K

We trained four GMMs with diagonal covariance and 3 random restarts each. All four converged. We picked the one with the lowest **BIC** (Bayesian Information Criterion), which rewards a good fit but punishes extra complexity.

<p align="center">
  <img src="figures/06_aic_bic.png" width="560" alt="AIC and BIC vs K"/>
</p>

| K | BIC | EM iterations |
|---|---|---|
| 1 | 13,982,626 | 2 |
| 2 | 11,936,015 | 9 |
| 4 | 10,896,761 | 32 |
| **8** | **10,144,915** | 35 |

### 5. Picking the cutoff and testing

A score alone doesn't make a decision; you also need a cutoff. We chose it on **validation** data, *before* looking at test, by asking: *"what score keeps false alerts at or below 0.5% of legitimate transactions?"*

Then we ran the test set **once**.

<p align="center">
  <img src="figures/07_confusion.png" width="480" alt="Confusion matrix"/>
</p>

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | 56,500 | 172 (false alert) |
| **Actually fraud** | 37 (missed) | **37 (caught)** |

---

## The trade-off: how many alerts can you handle?

There's no single "right" cutoff. A stricter one means fewer false alerts but more missed fraud, so it depends on how many cases the fraud team can review.

<p align="center">
  <img src="figures/08_tradeoff.png" width="760" alt="Fraud found vs false alerts by budget"/>
</p>

| Alert budget | Frauds found | False alerts | Recall | Precision |
|---|---|---|---|---|
| 0.1% (strict) | 6 | 39 | 8.1% | 13.3% |
| **0.5% (default)** | **37** | **172** | **50.0%** | **17.7%** |
| 1.0% (relaxed) | 52 | 381 | 70.3% | 12.0% |

Another way to look at it: *"if my team can review N cases, how many frauds do they find?"*

| Cases reviewed | Frauds found | Hit rate |
|---|---|---|
| 50 | 8 | 16.0% |
| **100** | **29** | **29.0%** |
| 200 | 37 | 18.5% |
| 500 | 53 | 10.6% |

<p align="center">
  <img src="figures/09_precision_recall.png" width="560" alt="Precision-recall curve"/>
</p>
<p align="center"><i>We use the precision-recall curve instead of ROC because, with this much imbalance, ROC looks good even for weak models.</i></p>

---

## Why not just train a classifier?

| | Supervised classifier | Our GMM approach |
|---|---|---|
| Needs fraud labels to train | Yes, lots of them | **No**, only normal data |
| Fraud examples available | Only 342 in training | Doesn't matter |
| New fraud patterns | Can miss them | Flags anything unusual |
| Easy to explain | Often a black box | "This doesn't look like normal spending" |

## Limitations

- **Only two days of data**, so we can't test how the model ages over weeks or months.
- **Anonymised features** mean we can't explain *why* something looks suspicious in business terms.
- **Scores overlap.** Half of the test frauds score below the default cutoff:

<p align="center">
  <img src="figures/13_score_overlap.png" width="560" alt="Score overlap between fraud and legit"/>
</p>

- **Diagonal covariance** ignores correlations between features. We chose it for speed and stability.

## Coming in Phase 2

```mermaid
flowchart LR
    P1["Phase 1 (done)<br/>GMM<br/>score = −log p(x)"] --> CMP{"Same data,<br/>same splits,<br/>same metrics"}
    P2["Phase 2 (next)<br/>Autoencoder<br/>score = reconstruction error"] --> CMP
    CMP --> R["Which ranks fraud better?<br/>Does combining them help?"]

    classDef done fill:#E8F5E9,stroke:#2E7D32,color:#1B3D1F
    classDef next fill:#F3E5F5,stroke:#6A1B9A,color:#2E0F3D
    classDef neutral fill:#F5F5F5,stroke:#616161,color:#212121
    class P1 done
    class P2 next
    class CMP,R neutral
```

---

## Project structure

```
t011-card-fraud-phase1/
├── notebooks/            # All the code: 5 notebooks, run in order
├── results/              # Saved metrics and settings (CSV / JSON)
├── figures/              # Every chart in this README
├── report/               # IEEE-format LaTeX report source
├── output/               # Final PDF report, slides, HTML notebook copies
├── docs/                 # Literature review, viva guide, checklist, team record
├── data/                 # Dataset notes (CSV downloads automatically)
├── index.html            # Read the notebooks in a browser
├── FILE_MAP.md           # Which notebook creates which file
├── STUDY_GUIDE.md        # Concepts explained for revision
└── requirements.txt
```

| # | Notebook | What happens |
|---|---|---|
| 01 | [Project and Literature](notebooks/01_Project_and_Literature.ipynb) | Problem, related work, design decisions (reading only) |
| 02 | [Data Quality and EDA](notebooks/02_Data_Quality_and_EDA.ipynb) | Load, check, de-duplicate, split by time, explore |
| 03 | [Feature Engineering](notebooks/03_Feature_Engineering.ipynb) | Build 29 features, fit the scaler |
| 04 | [GMM Training](notebooks/04_GMM_Training.ipynb) | Explain EM, train K = 1/2/4/8, choose by BIC |
| 05 | [Evaluation and Thresholds](notebooks/05_Evaluation_and_Thresholds.ipynb) | Cutoffs, test results, PR curve, review capacity |

Each notebook saves its outputs and the next one loads them, so you don't need one long Python session.

---

## Run it yourself

You need **Python 3.12**.

```bash
git clone https://github.com/mitul-bhatia/t011-card-fraud-phase1.git
cd t011-card-fraud-phase1

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then open `notebooks/` in Jupyter or VS Code:

1. Read **01**.
2. Open **02**, pick your Python environment, and click **Restart & Run All**.
3. When it finishes, do the same for **03 → 04 → 05**.

The dataset downloads itself the first time notebook 02 runs. With seed `42` you should get the same numbers shown here.

> **Just want to read?** Open [index.html](index.html) in a browser. All five notebooks are there with outputs included.

---

## Deliverables

| | |
|---|---|
| Report (IEEE format) | [PDF](output/Phase1_Report.pdf) · [LaTeX source](report/Phase1_Report.tex) |
| Presentation | [PPTX](output/Phase1_Presentation_Complete.pptx) · [PDF](output/Phase1_Presentation_Complete.pdf) |
| Literature review | [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) |
| Viva and demo prep | [Demo guide](docs/DEMO_AND_VIVA.md) · [Study guide](STUDY_GUIDE.md) |
| Phase 1 checklist | [docs/PHASE1_CHECKLIST.md](docs/PHASE1_CHECKLIST.md) |

## Team T011

| Enrollment No. | Name |
|---|---|
| 2401010174 | H Maheshwari |
| 2401010486 | Vaibhav |
| 2401010267 | Mayank |
| 2401010222 | Keshav |
| 2401010279 | Mitul Bhatia |

Who did what: [docs/TEAM_CONTRIBUTIONS.md](docs/TEAM_CONTRIBUTIONS.md)

## References

- Dempster, Laird and Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS-B. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
- Saito and Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets*, PLOS ONE. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- scikit-learn: [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- ULB Machine Learning Group / Worldline: [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

More detail is in the [literature review](docs/LITERATURE_REVIEW.md).
