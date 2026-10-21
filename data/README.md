# Dataset

Original: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The [TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) documents the public CSV mirror used by notebook 02. This project uses its data source, not its neural-network method.

Notebook 02 loads `data/raw/creditcard.csv`, downloading it if missing. The CSV is excluded from Git. Expected original size: 284,807 rows, 31 columns and 492 frauds.

The exact local CSV used has SHA256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. Class is 0 for legitimate and 1 for fraud. Amount is in EUR; Time is elapsed seconds; V1-V28 are anonymous provider-supplied PCA features.

We remove exact repeated rows under the documented assumption explained in notebook 02, Section 2. There is no transaction ID that proves identical records are the same purchase. Keep source attribution and review dataset terms before redistributing the raw file.
