"""Use real result tables and training distributions for editable slide charts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project import load_data, chronological_split


def records(name):
    return pd.read_csv(ROOT / f'results/{name}.csv').to_dict('records')


if __name__ == '__main__':
    df, quality = load_data(ROOT / 'data/raw/creditcard.csv')
    train, _, _ = chronological_split(df)
    bins = np.linspace(0, 12, 25)
    distributions = []
    for label in [0, 1]:
        values = np.log1p(train.loc[train.Class == label, 'Amount'])
        counts, _ = np.histogram(values, bins=bins)
        distributions.append((counts / len(values)).tolist())
    scores = pd.read_csv(ROOT / 'results/test_scores.csv')
    p, r, _ = precision_recall_curve(scores['class'], scores.anomaly_score, drop_intermediate=True)
    payload = {'quality': quality, 'splits': records('splits'), 'selection': records('gmm_selection'),
               'operating': records('test_operating_points'), 'capacity': records('test_review_capacity'),
               'ranking': records('ranking_metrics'), 'amount': {'bins': ((bins[:-1]+bins[1:])/2).tolist(),
               'legitimate': distributions[0], 'fraud': distributions[1]},
               'pr': {'precision': p.tolist(), 'recall': r.tolist()}}
    (ROOT / 'results/presentation_data.json').write_text(json.dumps(payload, indent=2))
    print('Saved editable-chart data. PR retained points:', len(p))
