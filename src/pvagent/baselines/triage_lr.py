"""TF-IDF + logistic regression seriousness baseline, threshold set for a target recall on dev."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Largest threshold that keeps recall >= target on (y, p)."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def score_cases(dev_ids, dev_text, dev_y, other_ids, other_text, target_recall: float = 0.99, seed: int = 0):
    """Leak-free LR scores for every sample case: dev cases get 5-fold out-of-fold scores, all other cases
    (test) get scores from a model fit on all of dev. Threshold = largest value keeping `target_recall` on dev OOF."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(dev_y, dtype=int)
    model = make_model()
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, list(dev_text), y, cv=cv, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y, oof, target_recall)
    model.fit(list(dev_text), y)
    scores = {int(i): float(p) for i, p in zip(dev_ids, oof)}
    if len(other_ids):
        scores.update({int(i): float(p) for i, p in zip(other_ids, model.predict_proba(list(other_text))[:, 1])})
    return scores, thr


def fit_predict(train_text, train_y, test_text, target_recall: float = 0.99):
    """Fit on dev, pick the threshold from 5-fold out-of-fold probabilities, score test."""
    y = np.asarray(train_y, dtype=int)
    model = make_model()
    oof = cross_val_predict(model, list(train_text), y, cv=5, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y, oof, target_recall)
    model.fit(list(train_text), y)
    p = model.predict_proba(list(test_text))[:, 1]
    return p, thr
