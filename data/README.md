# Dataset provenance

Original: ULB Machine Learning Group / Worldline, [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Public mirror: `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`, documented in the [TensorFlow imbalanced-data tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data). Only the data source is used from that tutorial. This project implements a GMM, not its neural classifier.

Download with `python scripts/download_data.py`. The expected original dataset has 284,807 rows, 31 columns and 492 fraud labels. `results/data_quality.json` records the checksum and measured counts for the exact file used. A changed checksum requires inspecting the new file before comparing results.

| Column | Meaning | Use |
|---|---|---|
| Time | Seconds since the first transaction in the file | Chronological split and time EDA only |
| V1-V28 | Anonymized principal-component features supplied by the dataset provider | 28 model inputs |
| Amount | Transaction amount, EUR | Transform to `log(1 + Amount)` |
| Class | 0 legitimate, 1 fraud | Filter legitimate training rows and evaluate results |

The model receives 29 features. Class is never an input. The provider already produced V1-V28, so we cannot audit how its original PCA was fitted. We fit our own scaler only on legitimate training rows.

Exact duplicate handling: sort by Time with stable ties and keep the first occurrence of each exact full row. There is no transaction ID, so this is an assumption about repeated records. We document raw and cleaned class counts and preserve all other values. We do not remove large transactions or unusual feature values merely because they look anomalous.

Data quality checks inspect the file mechanically for schema, finite numeric values, binary labels and nonnegative Time/Amount. Class-aware feature EDA uses the training split only. Test labels appear in final metrics and final diagnostic plots.

Raw data stays local and is excluded from Git. Attribute the source and review its current dataset terms before redistribution. The repository includes the download script rather than a second public copy of the dataset.
