# Literature review

**Question:** Can a GMM learned from earlier legitimate transactions prioritize later fraud for review?

This is a reproducible educational baseline. We do not claim a new algorithm or copy performance values from other studies.

## 1. EM explains how the model learns

A. P. Dempster, N. M. Laird and D. B. Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*, Journal of the Royal Statistical Society B, 39(1), 1-22. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).

The hidden information in a GMM is component membership. The paper gives the EM framework for alternating membership estimation and parameter updates. This is the reason we explain the E-step and M-step before fitting. EM can reach a local solution, so we use three initializations and record convergence. Convergence alone does not prove the globally best fit or a useful fraud detector.

## 2. PR evaluation explains the measures

T. Saito and M. Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE 10(3), e0118432. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

Their analysis motivates checking alert quality when positives are rare. We therefore show a PR curve, its trapezoidal area, recall, false-positive rates and precision at a fixed review capacity. The paper concerns evaluation, not a promised GMM fraud score. PR results depend on the test base rate, which we report.

## 3. GMM documentation explains implementation choices

scikit-learn, [Gaussian mixture guide](https://scikit-learn.org/stable/modules/mixture.html) and [model selection example](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html).

We use the documented diagonal covariance option to keep the model smaller, try K=1,2,4,8 and report AIC/BIC. Our fixed rule selects minimum training BIC among converged candidates. Diagonal covariance misses some within-component relationships. Information criteria score density fit with a parameter penalty, not fraud detection directly. Eight wins within our tested grid, not among every possible K.

## 4. Dataset documentation explains the evidence limits

ULB Machine Learning Group / Worldline, [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The [TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) documents the public CSV mirror used here.

We use the recommended real transaction benchmark and identify the exact CSV with a checksum. Anonymous inputs limit business explanations. A short historical recording cannot establish present-day production performance. We use TensorFlow's documented data location, not its neural classifier.

## Research gap and our response

GMM density fit does not guarantee that fraud and legitimate activity separate. We test that gap using later mixed transactions, frozen validation policies and a same-data one-Gaussian reference. Score overlap and false alerts are findings to explain, not values to hide.

The assignment suggests an autoencoder to model nonlinear legitimate patterns in Phase 2. Sakurada and Yairi (2014), *Anomaly Detection Using Autoencoders*, is a suggested later reading. No autoencoder result or hybrid improvement is claimed here.

Sources were checked during preparation on 21 October 2026. The report provides formal citations; notebook 01 provides the simple decision-to-source table.
