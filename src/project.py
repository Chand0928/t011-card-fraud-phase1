"""Phase 1: chronological evaluation of a legitimate-only Gaussian mixture.

The notebook calls these small functions in order. No neural network is trained.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, auc, confusion_matrix,
                             precision_recall_curve)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
COMPONENTS = [1, 2, 4, 8]
FPR_BUDGETS = [0.001, 0.005, 0.01]
DEFAULT_BUDGET = 0.005  # Illustrative classroom policy: 5 per 1,000 legitimate rows.
REVIEW_CAPACITIES = [50, 100, 200, 500]
FEATURES = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]
BLUE, RED = "#24668d", "#c44945"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180,
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})


def load_data(path):
    """Validate the actual dataset. Fail clearly rather than silently repair values."""
    path = Path(path)
    df = pd.read_csv(path)
    expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    if list(df.columns) != expected:
        raise ValueError("Expected ULB columns: Time, V1..V28, Amount, Class.")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in expected):
        raise ValueError("All columns must be numeric.")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing or infinite values found. Inspect before modelling.")
    if not set(df.Class.unique()).issubset({0, 1}) or df.Class.nunique() != 2:
        raise ValueError("Class must contain both 0 (legitimate) and 1 (fraud).")
    if (df.Amount < 0).any() or (df.Time < 0).any():
        raise ValueError("Negative Amount or Time found. Inspect the data.")
    quality = {"raw_rows": len(df), "raw_frauds": int(df.Class.sum()),
               "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
               "infinite_cells": int(np.isinf(df.to_numpy()).sum()),
               "exact_duplicate_rows": int(df.duplicated().sum()),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # No transaction ID exists. Treat exact repeats as duplicate records and keep first.
    # This is a documented assumption, not proof that they are the same real purchase.
    df = df.sort_values("Time", kind="stable").drop_duplicates().copy()
    df.index.name = "source_row"
    quality.update(clean_rows=len(df), clean_frauds=int(df.Class.sum()),
                   clean_fraud_rate=float(df.Class.mean()))
    return df, quality


def chronological_split(df):
    """Roughly 60/20/20 by row count. Keep equal timestamps in the later split."""
    t1 = float(df.Time.iloc[int(0.60 * len(df))])
    t2 = float(df.Time.iloc[int(0.80 * len(df))])
    train = df[df.Time < t1].copy()
    val = df[(df.Time >= t1) & (df.Time < t2)].copy()
    test = df[df.Time >= t2].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Too few distinct timestamps to form three chronological splits.")
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if part.Class.nunique() != 2:
            raise ValueError(f"{name} needs both classes for the planned evaluation.")
    assert train.Time.max() < val.Time.min() <= val.Time.max() < test.Time.min()
    return train, val, test


def split_summary(train, val, test):
    return pd.DataFrame([
        {"split": name, "rows": len(d), "legitimate": int((d.Class == 0).sum()),
         "fraud": int(d.Class.sum()), "fraud_rate": float(d.Class.mean()),
         "start_seconds": float(d.Time.min()), "end_seconds": float(d.Time.max())}
        for name, d in [("train", train), ("validation", val), ("test", test)]])


def make_features(df):
    """29 inputs. Time is for ordering only. Class never enters the feature matrix."""
    x = df[[f"V{i}" for i in range(1, 29)]].copy()
    x["LogAmount"] = np.log1p(df.Amount)  # Accepts zero and reduces the long right tail.
    return x


def prepare_features(train, val, test):
    legitimate_train = train[train.Class == 0]
    scaler = StandardScaler().fit(make_features(legitimate_train))
    return (scaler, scaler.transform(make_features(legitimate_train)),
            scaler.transform(make_features(val)), scaler.transform(make_features(test)))


def fit_candidates(x_train):
    """All candidates see the same legitimate training rows; no validation fraud tuning."""
    rows, models = [], {}
    with threadpool_limits(limits=2):
        for k in COMPONENTS:
            start = time.perf_counter()
            model = GaussianMixture(n_components=k, covariance_type="diag",
                                    reg_covar=1e-6, n_init=3, max_iter=200,
                                    tol=1e-3, random_state=SEED).fit(x_train)
            rows.append({"components": k, "aic": model.aic(x_train),
                         "bic": model.bic(x_train), "converged": bool(model.converged_),
                         "iterations": model.n_iter_,
                         "fit_seconds": time.perf_counter() - start})
            models[k] = model
            print(f"K={k}: BIC={rows[-1]['bic']:,.0f}; converged={model.converged_}", flush=True)
    table = pd.DataFrame(rows)
    valid = table[table.converged]
    if valid.empty:
        raise RuntimeError("No candidate converged. Do not report an unconverged model.")
    best_k = int(valid.loc[valid.bic.idxmin(), "components"])
    return models[best_k], models[1], table


def threshold_for_fpr(legitimate_scores, budget):
    """Largest permitted tail, with strict >. Ties can reduce the number flagged.

    If n=1,000 and budget=.005, allow at most 5 scores above the threshold.
    This controls empirical validation FPR, not future test FPR.
    """
    s = np.sort(np.asarray(legitimate_scores, dtype=float))
    if len(s) == 0 or not np.isfinite(s).all() or not 0 <= budget < 1:
        raise ValueError("Need finite legitimate scores and a budget in [0, 1).")
    allowed = int(np.floor(budget * len(s)))
    return float(s[len(s) - allowed - 1])


def decision_metrics(y, scores, threshold):
    flags = np.asarray(scores) > threshold
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "alerts": int(tp + fp),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.,
            "recall": float(tp / (tp + fn)), "fpr": float(fp / (fp + tn))}


def choose_thresholds(y_val, val_scores):
    return pd.DataFrame([
        {"fpr_budget": budget, **decision_metrics(
            y_val, val_scores, threshold_for_fpr(val_scores[np.asarray(y_val) == 0], budget))}
        for budget in FPR_BUDGETS])


def ranking_metrics(y, scores):
    precision, recall, _ = precision_recall_curve(y, scores)
    return {"average_precision": float(average_precision_score(y, scores)),
            "trapezoidal_pr_auc": float(auc(recall, precision)),
            "fraud_rate": float(np.mean(y))}


def review_capacity_table(y, scores):
    y = np.asarray(y)
    # Stable order resolves tied scores by their original chronological row order.
    order = np.argsort(-np.asarray(scores), kind="stable")
    return pd.DataFrame([
        {"capacity": k, "frauds_found": int(y[order[:k]].sum()),
         "precision_at_k": float(y[order[:k]].mean()),
         "recall_at_k": float(y[order[:k]].sum() / y.sum())}
        for k in REVIEW_CAPACITIES if k <= len(y)])


def save_figure(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def eda_charts(train, splits, folder):
    """Class-aware feature exploration uses training rows only."""
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [(train.Class == 0).sum(), (train.Class == 1).sum()]
    ax.bar(["Legitimate", "Fraud"], counts, color=[BLUE, RED])
    ax.set_yscale("log"); ax.set_ylabel("Training transactions (log scale)")
    ax.set_title("Fraud is rare in the training period")
    for i, n in enumerate(counts): ax.text(i, n * 1.10, f"{n:,}", ha="center")
    ax.set_ylim(1, max(counts) * 3)
    save_figure(fig, folder, "01_class_balance")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(train.Amount, bins=60, color=BLUE)
    axes[0].set_xlabel("Amount (EUR)"); axes[0].set_yscale("log")
    axes[0].set_ylabel("Training rows (log count)"); axes[0].set_title("Raw amount has a long tail")
    for label, color in [(0, BLUE), (1, RED)]:
        axes[1].hist(np.log1p(train.loc[train.Class == label, "Amount"]), bins=35,
                     density=True, histtype="step", linewidth=2, color=color,
                     label="Legitimate" if label == 0 else "Fraud")
    axes[1].set_xlabel("log(1 + Amount)"); axes[1].set_ylabel("Density within each class")
    axes[1].set_title("Log amount reduces the range"); axes[1].legend()
    save_figure(fig, folder, "02_amount_transform")

    fig, ax = plt.subplots(figsize=(8, 4))
    for row, color in zip(splits.to_dict("records"), [BLUE, "#d49a37", "#60766b"]):
        ax.barh(row["split"], (row["end_seconds"] - row["start_seconds"]) / 3600,
                left=row["start_seconds"] / 3600, color=color)
        ax.text((row["start_seconds"] + row["end_seconds"]) / 7200, row["split"],
                f"{row['rows']:,} rows", ha="center", va="center", color="white")
    ax.set_xlabel("Hours since first recorded transaction"); ax.set_title("Earlier data trains; later data tests")
    save_figure(fig, folder, "03_chronological_split")

    hourly = train.assign(hour=(train.Time // 3600).astype(int)).groupby("hour").Class.agg(["size", "mean"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].bar(hourly.index, hourly["size"], color=BLUE); axes[0].set_ylabel("Transactions")
    axes[0].set_title("Training traffic and fraud rate vary over time")
    axes[1].plot(hourly.index, hourly["mean"] * 100, color=RED, marker=".")
    axes[1].set_ylabel("Fraud rate (%)"); axes[1].set_xlabel("Hours since first transaction")
    save_figure(fig, folder, "04_training_time_pattern")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for feature, ax in zip(["V1", "V2"], axes):
        # Display central 99% of training values; this does not clip model inputs.
        lo, hi = train[feature].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 45)
        for label, color in [(0, BLUE), (1, RED)]:
            ax.hist(train.loc[train.Class == label, feature], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=str(label))
        ax.set_title(f"{feature}: central 99% training range"); ax.set_xlabel(feature)
        ax.set_ylabel("Density within displayed range"); ax.legend(title="Class")
    save_figure(fig, folder, "05_feature_distributions")

    corr = make_features(train[train.Class == 0]).corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(29), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(29), corr.columns, fontsize=7)
    ax.set_title("Correlations among legitimate training features")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    save_figure(fig, folder, "06_feature_correlation")


def result_charts(selection, y, scores, baseline_scores, validation, operating, capacity, folder):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(selection.components, selection.aic / 1e6, "o-", label="AIC", color=BLUE)
    ax.plot(selection.components, selection.bic / 1e6, "s--", label="BIC", color=RED)
    ax.set_xticks(COMPONENTS); ax.set_xlabel("Number of Gaussian components")
    ax.set_ylabel("Criterion / 1,000,000 (lower is better)"); ax.legend()
    ax.set_title("Choose K using legitimate training data")
    save_figure(fig, folder, "07_aic_bic")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(0, BLUE), (1, RED)]:
        ax.hist(np.log10(1 + np.maximum(scores[np.asarray(y) == label], 0)), bins=40,
                density=True, histtype="step", linewidth=2, color=color,
                label="Legitimate" if label == 0 else "Fraud")
    ax.set_xlabel("log10(1 + max(anomaly score, 0)) for display only")
    ax.set_ylabel("Density within each class"); ax.legend()
    ax.set_title("Score overlap explains false alerts and missed fraud")
    save_figure(fig, folder, "08_test_score_distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s, color in [("Selected GMM", scores, BLUE), ("One Gaussian", baseline_scores, RED)]:
        p, r, _ = precision_recall_curve(y, s)
        ax.step(r, p, where="post", color=color, label=f"{name}, AP={average_precision_score(y, s):.3f}")
    ax.axhline(np.mean(y), color="gray", linestyle="--", label=f"Prevalence={np.mean(y):.4f}")
    ax.set(xlabel="Recall: fraction of fraud found", ylabel="Precision: fraud among alerts",
           xlim=(0, 1), ylim=(0, 1.02), title="Test precision-recall curve")
    ax.legend(fontsize=9); save_figure(fig, folder, "09_precision_recall")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = operating.fpr_budget * 100
    axes[0].plot(x, operating.recall * 100, "o-", color=BLUE)
    axes[0].set(xlabel="Validation FPR budget (%)", ylabel="Test fraud recall (%)",
                title="More review can catch more fraud")
    axes[1].plot(x, validation.fpr * 100, "s--", label="Validation actual", color=BLUE)
    axes[1].plot(x, operating.fpr * 100, "o-", label="Test actual", color=RED)
    axes[1].plot(x, x, ":", color="gray", label="Budget")
    axes[1].set(xlabel="Validation FPR budget (%)", ylabel="Observed FPR (%)",
                title="A validation budget may drift on test"); axes[1].legend(fontsize=9)
    save_figure(fig, folder, "10_false_positive_tradeoffs")

    row = operating.loc[np.isclose(operating.fpr_budget, DEFAULT_BUDGET)].iloc[0]
    cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=18,
                    color="white" if i == 0 and j == 0 else "black")
    ax.set_xticks([0, 1], ["No alert", "Review alert"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set(xlabel="Model decision", ylabel="Actual class",
           title="Test decisions at the 0.5% validation budget")
    save_figure(fig, folder, "11_confusion_matrix")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(capacity.capacity, capacity.precision_at_k * 100, "o-", color=BLUE)
    ax.set(xlabel="Highest-scoring test transactions reviewed (K)",
           ylabel="Precision@K (%)", title="Limited review capacity changes alert quality")
    ax.set_xticks(capacity.capacity)
    save_figure(fig, folder, "12_review_capacity")


def save_results(root, quality, splits, selection, validation, operating, capacity,
                 ranking, model, scaler, y_test, test_scores):
    out = Path(root) / "results"; out.mkdir(exist_ok=True)
    for name, table in [("splits", splits), ("gmm_selection", selection),
                        ("validation_thresholds", validation), ("test_operating_points", operating),
                        ("test_review_capacity", capacity), ("ranking_metrics", ranking)]:
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))
    metadata = {"seed": SEED, "features": FEATURES, "components_tested": COMPONENTS,
                "selected_components": model.n_components, "selection_rule": "minimum training BIC among converged candidates",
                "covariance_type": "diag", "n_init": 3, "max_iter": 200,
                "tol": 1e-3, "reg_covar": 1e-6, "prediction_rule": "score > threshold",
                "default_validation_fpr_budget": DEFAULT_BUDGET,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame({"source_row": y_test.index, "class": y_test.to_numpy(),
                  "anomaly_score": test_scores}).to_csv(out / "test_scores.csv", index=False)
    model_dir = Path(root) / "models"; model_dir.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "gmm": model, "features": FEATURES,
                 "thresholds": validation[["fpr_budget", "threshold"]].to_dict("records")},
                model_dir / "gmm_phase1.joblib")
