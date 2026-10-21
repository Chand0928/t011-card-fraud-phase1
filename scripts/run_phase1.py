"""Run the same analysis as the notebook, without needing Jupyter."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project import (load_data, chronological_split, split_summary, prepare_features,
                         fit_candidates, eda_charts, choose_thresholds, ranking_metrics,
                         decision_metrics, review_capacity_table, result_charts, save_results)


def main():
    df, quality = load_data(ROOT / "data/raw/creditcard.csv")
    train, val, test = chronological_split(df)
    splits = split_summary(train, val, test)
    print(splits.to_string(index=False))
    scaler, x_train, x_val, x_test = prepare_features(train, val, test)
    figures = ROOT / "results/figures"
    eda_charts(train, splits, figures)
    model, baseline, selection = fit_candidates(x_train)
    val_scores = -model.score_samples(x_val)
    validation = choose_thresholds(val.Class, val_scores)
    # Model choice and every threshold are now frozen. Only now score the mixed test.
    scores, baseline_scores = -model.score_samples(x_test), -baseline.score_samples(x_test)
    ranking = pd.DataFrame([
        {"model": name, **ranking_metrics(test.Class, s)}
        for name, s in [("selected_gmm", scores), ("one_gaussian", baseline_scores)]])
    operating = pd.DataFrame([
        {"fpr_budget": row.fpr_budget, **decision_metrics(test.Class, scores, row.threshold)}
        for row in validation.itertuples()])
    capacity = review_capacity_table(test.Class, scores)
    result_charts(selection, test.Class, scores, baseline_scores, validation, operating, capacity, figures)
    save_results(ROOT, quality, splits, selection, validation, operating, capacity,
                 ranking, model, scaler, test.Class, scores)
    assert np.isfinite(scores).all()
    print(ranking.to_string(index=False))
    print(operating.to_string(index=False))
    print("Saved real results and 12 figures to results/.")


if __name__ == "__main__":
    main()
