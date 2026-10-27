"""Reduce a training set with NIS inside an imbalanced-learn pipeline.

The sampler runs only when the pipeline is fitted; prediction uses all
test points. Requires ``pip install imbalanced-learn``.
"""

from imblearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

from skreduce import NIS

X, y = make_classification(
    n_samples=20_000, n_features=6, n_informative=4, random_state=0
)

for name, model in [
    ("1-NN on all data", KNeighborsClassifier(n_neighbors=1)),
    ("NIS + 1-NN", make_pipeline(NIS(alpha=1.0), KNeighborsClassifier(n_neighbors=1))),
]:
    scores = cross_validate(model, X, y, cv=5)
    print(f"{name:18s} accuracy {scores['test_score'].mean():.3f}")

nis = NIS(alpha=1.0).fit(X)
print(f"NIS keeps {nis.sample_indices_.size} of {X.shape[0]} instances "
      f"(reduction rate {nis.reduction_rate_:.1%})")
