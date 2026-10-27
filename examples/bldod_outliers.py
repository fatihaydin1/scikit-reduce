"""Detect outliers with BLDOD and score new points."""

import numpy as np
from sklearn.metrics import roc_auc_score

from skreduce import BLDOD

rng = np.random.RandomState(0)
inliers = np.vstack([rng.normal(0, 1, size=(300, 2)), rng.normal(6, 0.5, size=(200, 2))])
outliers = rng.uniform(-6, 12, size=(25, 2))
X = np.vstack([inliers, outliers])
y_true = np.r_[np.zeros(len(inliers)), np.ones(len(outliers))]

detector = BLDOD(std_coef=0.5).fit(X)
print(f"radius: {detector.radius_:.3f}")
print(f"ROC AUC of the outlier factor: {roc_auc_score(y_true, detector.outlier_factor_):.3f}")
print(f"points labelled as outliers (factor > 0.9): {(detector.predict(X) == -1).sum()}")

new_points = np.array([[0.0, 0.0], [6.0, 6.0], [3.0, 3.0], [15.0, -5.0]])
print("score_samples of new points (lower = more abnormal):")
print(np.round(detector.score_samples(new_points), 3))
