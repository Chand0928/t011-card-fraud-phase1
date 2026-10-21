"""Generate the IEEE LaTeX report using the measured CSV/JSON outputs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
REPORT.mkdir(exist_ok=True)
q = json.loads((ROOT / 'results/data_quality.json').read_text())
s = pd.read_csv(ROOT / 'results/splits.csv')
g = pd.read_csv(ROOT / 'results/gmm_selection.csv')
r = pd.read_csv(ROOT / 'results/ranking_metrics.csv')
v = pd.read_csv(ROOT / 'results/validation_thresholds.csv')
o = pd.read_csv(ROOT / 'results/test_operating_points.csv')
c = pd.read_csv(ROOT / 'results/test_review_capacity.csv')
split_rows = '\n'.join(f"{x.split.title()} & {x.rows:,} & {x.legitimate:,} & {x.fraud} & {100*x.fraud_rate:.4f}\\\\" for x in s.itertuples())
selection_rows = '\n'.join(f"{x.components} & {x.aic/1e6:.4f} & {x.bic/1e6:.4f} & Yes \\\\" for x in g.itertuples())
operation_rows = '\n'.join(f"{100*x.fpr_budget:.1f} & {x.tp} & {x.fp} & {100*x.recall:.2f} & {100*x.precision:.2f} & {100*x.fpr:.4f}\\\\" for x in o.itertuples())
capacity_rows = '\n'.join(f"{x.capacity} & {x.frauds_found} & {100*x.precision_at_k:.1f} & {100*x.recall_at_k:.2f}\\\\" for x in c.itertuples())
text = r'''\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,url}
\renewcommand{\ttdefault}{cmtt}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../results/figures/}}
\newcommand{\fig}[3]{\begin{figure}[!t]\centering\includegraphics[width=\columnwidth]{#1}\caption{#2}\label{#3}\end{figure}}
\title{Card Fraud Triage via Gaussian Mixture Density Scoring\\Phase 1: Machine Learning Evaluation}
\author{\IEEEauthorblockN{H Maheshwari, Vaibhav, Mayank, Keshav, Mitul Bhatia}
\IEEEauthorblockA{Team T011, Section A, Project 26\\Newton School of Technology, Rishihood University\\
Enrollment: 2401010174, 2401010486, 2401010267, 2401010222, 2401010279}}
\begin{document}
\maketitle
\begin{abstract}
We study card-fraud triage as anomaly detection. A Gaussian Mixture Model (GMM) learns a density using only earlier legitimate transactions, and negative log density ranks later transactions for review. We use the recommended ULB/Worldline dataset, remove exact repeated records under a documented assumption, and split the remaining 283,726 transactions chronologically. We fit preprocessing and all candidate models on 169,893 legitimate training rows. Minimum training Bayesian Information Criterion selects eight diagonal Gaussian components from a grid of one, two, four and eight. On a mixed test period containing 74 frauds among 56,746 transactions, the selected model achieves average precision 0.1532 and trapezoidal precision-recall area 0.1496. A threshold fixed using a 0.5\% validation false-positive budget catches 37 frauds and flags 172 legitimate transactions. Reviewing the top 100 test scores finds 29 frauds. These results show the review-workload tradeoff and the limitations of a density baseline. This submission covers Phase 1 only; no autoencoder or hybrid results are claimed.
\end{abstract}
\begin{IEEEkeywords}
fraud triage, Gaussian mixture, novelty detection, precision-recall, false-positive budget
\end{IEEEkeywords}

\section{Problem and Phase 1 Scope}
Card-fraud triage prioritizes transactions for a limited human-review queue. A useful ranking should expose fraud while controlling unnecessary investigation of legitimate activity. An unusual transaction is a candidate for review, not proof of fraud.

The supplied course instructions specify machine learning only for Phase 1. The assigned project proposes a GMM baseline followed by an autoencoder in the later deep-learning phase. We therefore implement the GMM, document its assumptions, and evaluate frozen review policies. The research question is whether a density fitted on legitimate history can rank later fraudulent transactions usefully. Our contribution is a reproducible educational experiment, not a new algorithm or a production fraud service.

\section{Literature Review}
Dempster, Laird and Rubin introduce expectation-maximization (EM) for likelihood estimation with hidden information~\cite{em}. In our model, component membership is hidden, so EM estimates memberships and density parameters iteratively. This connection explains model fitting but does not imply a global optimum.

Saito and Rehmsmeier explain how precision-recall analysis reveals false-alert burden in imbalanced classification~\cite{pr}. We consequently emphasize PR metrics and explicit review policies. Their work concerns evaluation, not performance guarantees for our dataset.

The scikit-learn mixture documentation describes covariance options, EM and information-criterion model selection~\cite{sklearn}. We use diagonal covariance and a small component grid. The data comes from the recommended ULB/Worldline benchmark~\cite{data}; TensorFlow documents the public mirror used to obtain the CSV~\cite{tf}. We use the mirror only as a data source. We do not adopt its neural classification implementation.

\section{Dataset Quality and Exploratory Analysis}
\subsection{Data and Provenance}
The original file contains 284,807 rows, 31 columns and 492 fraud labels. Time records elapsed seconds since the first transaction. V1--V28 are provider-supplied anonymized principal components. Amount is in EUR. Class equals zero for legitimate transactions and one for fraud. Anonymization prevents assigning business meanings to individual V columns. We cannot audit how the provider fitted its original PCA.

The quality checks verify expected columns, numeric finite values, binary labels and nonnegative Time/Amount. We find zero missing or infinite cells. We record a SHA256 checksum in \texttt{results/data\_quality.json} so later runs can identify the exact input file.

\subsection{Exact Repeated Records}
We sort by Time with stable handling of ties, remove 1,081 exact full-row repeats and keep the earliest occurrence. This removes 19 repeated fraud rows, leaving 283,726 rows and 473 frauds, a 0.1667\% cleaned base rate. Transaction IDs are unavailable, so identical records cannot be proven to represent the same real purchase. Duplicate removal is a stated assumption. We do not delete large amounts or extreme features simply because they look anomalous.

\subsection{Chronological Protocol}
We split by approximately 60/20/20 of cleaned rows, keeping equal timestamps within the same period. Training has Time below 120,393 seconds. Validation spans 120,393--145,233 seconds and test spans 145,234--172,792 seconds. Table~\ref{tab:splits} reports the resulting counts. Time ordering better represents learning from earlier records and evaluating later activity than a shuffled split would for this question.

\begin{table}[!t]\centering\caption{Chronological splits after duplicate removal}\label{tab:splits}
\begin{tabular}{lrrrr}\hline
Set & Rows & Legitimate & Fraud & Rate (\%)\\\hline
@@SPLITS@@
\hline\end{tabular}\end{table}

Only legitimate training rows fit the scaler and GMM. Validation determines thresholds; test determines final metrics. Mechanical integrity checks and base-rate reporting cover the whole file, while class-aware feature EDA uses training data only. Test labels never select features, component count or thresholds. Final test score plots are diagnostics after choices are frozen.

\fig{01_class_balance.pdf}{Training class counts use a logarithmic vertical axis so the rare fraud class remains visible.}{fig:imbalance}
\fig{02_amount_transform.pdf}{Training amount analysis. The raw amount has a long right tail. The right panel compares separate within-class densities after the log transform.}{fig:amount}

\subsection{EDA Findings and Their Implications}
Figure~\ref{fig:imbalance} shows the class imbalance. A policy predicting every test row legitimate would achieve about 99.87\% accuracy but zero fraud recall. This arithmetic motivates our evaluation measures rather than a high-accuracy claim.

Figure~\ref{fig:amount} explains the amount transform. The notebook additionally includes training traffic and fraud rate by hour, V1/V2 distributions over their central 99\% training range, a legitimate-feature correlation heatmap, and a chronological split plot. Plot range limits do not clip model inputs. Global correlation does not establish independence within every Gaussian component. The EDA supports understanding and documents assumptions rather than searching test labels for high-performing features.

\section{Feature Engineering}
We keep V1--V28 and replace Amount with
\begin{equation}
\mathrm{LogAmount}=\log(1+\mathrm{Amount}).
\end{equation}
This compresses the long right tail and accepts zero amounts. The resulting input dimension is $d=29$. Time is reserved for splitting and EDA, because absolute elapsed time is not itself our definition of suspiciousness. Class never enters the feature matrix.

We fit StandardScaler on the legitimate training subset. For feature $j$,
\begin{equation}
z_j=\frac{x_j-\mu_{j,\mathrm{train,legit}}}{\sigma_{j,\mathrm{train,legit}}}.
\end{equation}
The saved training parameters transform validation and test rows without refitting. Scaling provides consistent numerical units for regularized fitting. We retain all 29 transformed inputs and add no new PCA, resampling, supervised feature selection or neural processing.

\section{Theoretical Rigor and Model Application}
\subsection{Gaussian Mixture Density}
For a transformed input $z$, a GMM represents the legitimate density as
\begin{equation}
p(z)=\sum_{k=1}^{K}\pi_k\mathcal{N}(z\mid\mu_k,\Sigma_k),\quad
\pi_k\geq0,\quad\sum_k\pi_k=1.
\end{equation}
Each component describes one mathematical cloud of activity. Components are not verified customer groups or fraud classes. We restrict $\Sigma_k$ to a diagonal matrix. This estimates separate feature variances while omitting covariance between features within a component. The mixture as a whole can still express dependencies through component membership.

\subsection{Expectation-Maximization}
Given the current parameters, the E-step computes responsibilities
\begin{equation}
r_{ik}=\frac{\pi_k\mathcal{N}(z_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(z_i\mid\mu_j,\Sigma_j)}.
\end{equation}
Define $N_k=\sum_i r_{ik}$. The M-step updates
\begin{align}
\pi_k&=N_k/N, &
\mu_k&=\frac{1}{N_k}\sum_i r_{ik}z_i,\\
\sigma^2_{kj}&=\frac{1}{N_k}\sum_i r_{ik}(z_{ij}-\mu_{kj})^2+\epsilon.
\end{align}
We use $\epsilon=10^{-6}$ to regularize variances. There are three initializations per candidate, seed 42, tolerance $10^{-3}$ and a maximum of 200 EM iterations. The implementation retains the highest-likelihood initialization. We record convergence and refuse to select from unconverged candidates alone. A convergence flag means the stopping criterion was satisfied, not that a globally optimal density was found.

Although GMM fitting receives no targets, choosing legitimate training examples requires known labels. The full preparation is therefore legitimate-only novelty detection, sometimes described as a semi-supervised anomaly-detection setup. We do not claim fully label-free data preparation.

\subsection{AIC/BIC Model Selection}
All candidates $K\in\{1,2,4,8\}$ see the same legitimate training rows. With diagonal covariance, the parameter count is
\begin{equation}
q=(K-1)+Kd+Kd=K(2d+1)-1.
\end{equation}
For $d=29$, $K=8$ requires 471 free parameters. Full covariance at the same dimension and component count would require 3,719. This motivates the smaller baseline but does not prove its assumptions fit the data.

For total training log likelihood $L$ and $N$ fitted rows,
\begin{equation}
\mathrm{AIC}=2q-2L,\qquad\mathrm{BIC}=q\log N-2L.
\end{equation}
We report both but select minimum training BIC among converged candidates. Table~\ref{tab:selection} and Fig.~\ref{fig:bic} show the result. Both criteria improve through $K=8$, the upper boundary of our small grid. Eight is the best tested value, not a claimed true number of patterns. BIC scores legitimate density fit with a complexity penalty and does not optimize fraud recall.

\begin{table}[!t]\centering\caption{Model selection on legitimate training rows}\label{tab:selection}
\begin{tabular}{rrrr}\hline
K & AIC / $10^6$ & BIC / $10^6$ & Converged\\\hline
@@SELECTION@@
\hline\end{tabular}\end{table}
\fig{07_aic_bic.pdf}{AIC and BIC select eight components within the tested grid. Continued decrease at the boundary limits any optimality claim.}{fig:bic}

\subsection{Anomaly Scoring}
We define the anomaly score as $s(z)=-\log p(z)$. The code uses the negative of \texttt{score\_samples}. Larger values mean a transaction is less typical under the learned legitimate distribution. The GMM's \texttt{predict} method returns component IDs and would not provide fraud labels.

A continuous density is not the probability that a point is fraud. It can exceed one, so negative anomaly scores are possible. No probability calibration is claimed. The shared feature transform makes the selected GMM and one-Gaussian reference comparable on the same test rows.

\subsection{Validation-Only Thresholds}
For an FPR budget $b$, sort the $n_0$ legitimate validation scores in increasing order, allow $m=\lfloor b n_0\rfloor$ scores above the cutoff, and choose threshold $t=s_{(n_0-m)}$ using one-based indexing. Flag a row only if $s>t$. Ties may reduce the number flagged, ensuring empirical validation FPR does not exceed the budget.

We predefine budgets 0.1\%, 0.5\% and 1\%. The illustrative default is 0.5\%, selected before test outcomes and not as a bank requirement. Validation contains 56,688 legitimate rows, allowing 283 false alerts at this budget. The resulting threshold is 75.8942. It gives 283 validation false alerts and catches 29 of 57 validation frauds. We carry it unchanged to test. A validation budget is not a guarantee about later FPR under changing distributions.

\section{Evaluation and Measured Results}
\subsection{Measures and Fair Reference}
For true positives TP, false positives FP, true negatives TN and false negatives FN, we report
\begin{align}
\mathrm{precision}&=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, &
\mathrm{recall}&=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\\
\mathrm{FPR}&=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}.
\end{align}
We report average precision (AP), a recall-weighted step summary, and trapezoidal PR AUC separately. These are different integration conventions. Neither equals accuracy or precision at one operating threshold.

The one-Gaussian reference uses the same legitimate training data, scaler, features and mixed test rows. The selected GMM obtains AP 0.1532 and trapezoidal PR AUC 0.1496, compared with 0.0540 and 0.0527 for one Gaussian. Test prevalence is 0.1304\%. Figure~\ref{fig:pr} compares the rankings. The evidence supports better ranking in this run, not a universal guarantee for larger mixtures.

\fig{09_precision_recall.pdf}{Final test PR curves. AP labels summarize each ranking. The prevalence line provides context for rare fraud.}{fig:pr}

\subsection{Frozen Operating Policies}
Table~\ref{tab:operating} reports test outcomes at thresholds chosen on validation. At the default, TP=37, FP=172, FN=37 and TN=56,500. Therefore recall is $37/74=50\%$, precision is $37/209=17.70\%$, and FPR is $172/56{,}672=0.3035\%$. The workload is 209 review alerts. Most alerts are legitimate, and half the fraud is missed. This is a concrete limitation even though ranking improves over the reference.

\begin{table}[!t]\centering\caption{Test outcomes under fixed validation FPR policies}\label{tab:operating}
\begin{tabular}{rrrrrr}\hline
Budget & TP & FP & Recall & Precision & Test FPR\\
(\%) & & & (\%) & (\%) & (\%)\\\hline
@@OPERATING@@
\hline\end{tabular}\end{table}

\fig{10_false_positive_tradeoffs.pdf}{More permissive validation budgets find more test fraud here, with more false alerts. Actual test FPR differs from the validation budget.}{fig:tradeoff}
\fig{11_confusion_matrix.pdf}{Test confusion matrix at the default 0.5\% validation budget. Labels show exact counts; color uses log counts to display the imbalance.}{fig:cm}

\subsection{Fixed Review Capacity}
Precision@$k$ measures the fraud fraction among the $k$ largest scores. Table~\ref{tab:capacity} uses fixed capacities for the entire test batch, with stable chronological ordering to resolve ties. Reviewing 100 rows finds 29 frauds, precision@100=29\% and recall@100=39.19\%. This does not imply a daily production workload. Precision@$k$ need not decrease monotonically as capacity grows, because the next ranked block may have a different fraud rate.

\begin{table}[!t]\centering\caption{Fixed review capacities on the final test batch}\label{tab:capacity}
\begin{tabular}{rrrr}\hline
Capacity & Fraud found & Precision (\%) & Recall (\%)\\\hline
@@CAPACITY@@
\hline\end{tabular}\end{table}

\section{Reproducibility and Individual Understanding}
The teaching notebook calls small functions in \texttt{src/project.py}; \texttt{scripts/run\_phase1.py} reproduces the same experiment without Jupyter. The repository contains a download script, pinned dependencies, saved numeric tables, software versions, data checksum and twelve explanatory charts. A local model bundle stores the scaler, density and thresholds. Raw data and binary models remain excluded from Git.

Six targeted tests cover strict time boundaries, training-only scaler fitting, exclusion of Class and Time from model inputs, tied-score threshold budgets, future FPR drift and confusion counts. They check protocol errors rather than asserting a desired fraud score. Python 3.12.4, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.4.2 produced this run. Total GMM candidate fit time was about 12 seconds on the preparation machine, excluding imports and charts; timings vary by hardware.

The accompanying study guide explains every chart and includes worked metric calculations and viva questions. The contribution file lists suggested study areas but records no unverified contribution claims. AI assistance prepared code and materials. Each member must review, rerun and understand the experiment and follow any course disclosure policy. The course evaluates individual understanding even for team work.

\section{Limitations and Conclusion}
Only 74 frauds appear in the final test period, so one changed fraud decision shifts recall by about 1.35 percentage points. We do not claim statistical certainty from repeated windows. The short historical period cannot establish performance on current live activity. Anonymous inputs limit explanation, and provider PCA fitting is unaudited. Duplicate removal relies on an assumption. Diagonal covariance misses some within-component structure, and information criteria continue improving at the largest tested K.

Density rarity and fraud are different concepts. Some legitimate transactions are unusual, while fraud can resemble normal activity. The default policy's 172 false alerts and 37 missed frauds make that gap visible. Threshold rates may drift in later data. Operational costs, live monitoring and external validation remain unmeasured.

Within these limits, the eight-component GMM improves test PR ranking over one Gaussian and provides an explainable review-policy baseline. Phase 2 may compare a legitimate-only autoencoder under the same data and evaluation protocol. No such DL or hybrid implementation or performance is included in this Phase 1 submission.

\IEEEtriggeratref{2}
\begin{thebibliography}{9}
\bibitem{em} A. P. Dempster, N. M. Laird, and D. B. Rubin, ``Maximum likelihood from incomplete data via the EM algorithm,'' \emph{J. R. Stat. Soc. B}, vol. 39, no. 1, pp. 1--22, 1977, doi: 10.1111/j.2517-6161.1977.tb01600.x.
\bibitem{pr} T. Saito and M. Rehmsmeier, ``The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,'' \emph{PLOS ONE}, vol. 10, no. 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432.
\bibitem{sklearn} scikit-learn, ``Gaussian mixture models'' and ``Gaussian mixture model selection.'' [Online]. Available: \url{https://scikit-learn.org/stable/modules/mixture.html}. Accessed: Oct. 21, 2026.
\bibitem{data} ULB Machine Learning Group / Worldline, ``Credit Card Fraud Detection.'' [Online]. Available: \url{https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}. Accessed: Oct. 21, 2026.
\bibitem{tf} TensorFlow, ``Classification on imbalanced data,'' dataset mirror documentation. [Online]. Available: \url{https://www.tensorflow.org/tutorials/structured_data/imbalanced_data}. Accessed: Oct. 21, 2026.
\end{thebibliography}
\end{document}
'''
text = text.replace('@@SPLITS@@', split_rows).replace('@@SELECTION@@', selection_rows)
text = text.replace('@@OPERATING@@', operation_rows).replace('@@CAPACITY@@', capacity_rows)
(REPORT / 'phase1_report.tex').write_text(text)
print('Generated report/phase1_report.tex from real results.')
