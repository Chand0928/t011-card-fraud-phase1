# Literature review for Phase 1

These sources explain the design. Their reported experiments are not our measured results.
Links checked on 21 October 2026. Read the linked paper and explain the connection in your own words.

| Source | Main point in plain language | Use here | Limitation |
|---|---|---|---|
| Dempster, Laird and Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS B 39(1), 1-22. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) | EM alternates estimating hidden assignments and updating model parameters. | Gaussian component membership is hidden, so the GMM uses EM. Three initializations reduce dependence on one starting point. | EM can reach a local solution. Convergence does not prove the globally best density or fraud detector. |
| Saito and Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE 10(3), e0118432. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) | Precision exposes false-alert burden when positives are scarce. | Lead with PR metrics, then report recall and false positives under explicit review policies. | The paper is about evaluation, not a card-fraud GMM benchmark. PR metrics change with prevalence. |
| scikit-learn, [Gaussian mixture user guide](https://scikit-learn.org/stable/modules/mixture.html) and [model selection example](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html) | GMMs support several covariance assumptions. Information criteria balance fit and parameter count. | Use diagonal covariance for a small, fast baseline. Report AIC and BIC and select the minimum training BIC among K=1,2,4,8 candidates that converged. | Diagonal covariance misses dependencies within each component. BIC measures density fit with a complexity penalty, not fraud recall. |
| ULB Machine Learning Group / Worldline, [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The [TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) documents a public CSV mirror. | The benchmark contains rare fraud among real card transactions. | Use this recommended dataset, record its checksum, inspect quality, and preserve mixed validation and test distributions. | Anonymized PCA inputs limit business interpretation. A short historical window cannot establish current production performance. |

## The research question

Can a GMM learned only from earlier legitimate transactions rank later fraudulent transactions high enough to support a limited manual-review queue?

## What this implementation adds

This is a reproducible educational baseline, not a new algorithm. Its value is a clear experiment: chronological splits, training-only scaling, explicit threshold selection, and measured review tradeoffs. Compare the selected mixture with one Gaussian under the same preprocessing and test rows.

## Phase 2 reading only

The assignment also suggests Sakurada and Yairi (2014), *Anomaly Detection Using Autoencoders*. That topic is reserved for Phase 2. No autoencoder findings or comparison results are claimed here.
