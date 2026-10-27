# scikit-reduce

[![Tests](https://github.com/fatihaydin1/scikit-reduce/actions/workflows/tests.yml/badge.svg)](https://github.com/fatihaydin1/scikit-reduce/actions/workflows/tests.yml)

scikit-learn compatible implementations of two data reduction methods:

| Estimator | Task | Interface | Paper |
|---|---|---|---|
| `NIS` | Unsupervised instance selection (Nimble Instance Selection) | `fit_resample`, works in [imbalanced-learn](https://imbalanced-learn.org) pipelines | Aydın, *Neural Computing and Applications*, 2023 |
| `BLDOD` | Outlier detection (Boundary-aware Local Density-based Outlier Detection) | scikit-learn outlier detector: `fit`, `score_samples`, `decision_function`, `predict` | Aydın, *Information Sciences*, 2023 |

Both estimators pass scikit-learn's estimator checks (`check_estimator`).

## Installation

```bash
pip install scikit-reduce
```

The only dependencies are NumPy and scikit-learn (≥ 1.6). To use `NIS` inside a pipeline, install imbalanced-learn as well:

```bash
pip install "scikit-reduce[imblearn]"
```

## NIS: Nimble Instance Selection

NIS shifts each feature by its minimum, divides it by its standard deviation, multiplies it by `alpha` and rounds the result. Each combination of rounded values is a hyperrectangular cell, and NIS keeps the first instance in every occupied cell. It does not use labels, needs no neighbour search and runs in `O(n_samples · n_features · log n_samples)` time.

```python
from skreduce import NIS

X_res, y_res = NIS(alpha=1.0).fit_resample(X, y)   # y is optional
```

`alpha` controls the cell size: smaller values give larger cells and remove more instances.

In a pipeline, NIS reduces only the training data; predictions are made on all test points:

```python
from imblearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(NIS(alpha=1.0), KNeighborsClassifier(n_neighbors=1))
model.fit(X_train, y_train)
model.predict(X_test)
```

After fitting, `sample_indices_` holds the indices of the kept instances and `reduction_rate_` the fraction removed.

## BLDOD: Boundary-aware Local Density-based Outlier Detection

For each point, BLDOD counts the training points within a radius of `std_coef · ‖σ‖`, where `σ` holds the per-feature standard deviations. The counts are min-max normalised and inverted, so the outlier factor is 1 for the point with the fewest neighbours and 0 for the point with the most.

```python
from skreduce import BLDOD

detector = BLDOD(std_coef=1.0).fit(X)
detector.outlier_factor_      # factor of each training point, higher = more abnormal
detector.predict(X)           # -1 for outliers, +1 for inliers
detector.score_samples(X_new) # lower = more abnormal (scikit-learn convention)
```

With `contamination="auto"`, points whose factor is above 0.9 are labelled outliers, as in the paper. A float `contamination` sets the threshold from the expected proportion of outliers instead.

## Agreement with the reference implementations

The outputs were compared with the author's MATLAB code (`NIS.m`, `BLDOD.m`) run in GNU Octave:

- `NIS` selects exactly the same instances on all 55 data sets of the NIS repository, for `alpha` in {0.3, 0.5, 1, 2}.
- `BLDOD` gives the same outlier factors (maximum difference 1.1e-16) on all 17 data sets of the BLDOD repository, for `std_coef` in {0.5, 1, 2} and every `algorithm` option.

The test suite includes fixtures whose expected values were produced by the reference code.

Remaining differences, all deliberate:

- `NIS.sample_indices_` is in increasing order; the MATLAB code returns the same indices in the sort order of the cells.
- If every training point of `BLDOD` has the same neighbour count, all outlier factors are 0; the MATLAB code returns 1 for every point.
- `BLDOD` rejects missing values, as scikit-learn estimators do. `NIS` accepts them and, like the MATLAB code, places them in cell 0.
- `BLDOD` can score new data: it counts the training points within the same radius and normalises with the counts seen during `fit`, so a new point can have a factor above 1.

## Examples

See [`examples/`](https://github.com/fatihaydin1/scikit-reduce/tree/main/examples) for a cross-validated NIS pipeline and a BLDOD example on synthetic data.

## Citing

If you use these methods, please cite the corresponding paper:

```bibtex
@article{aydin2023nis,
  author  = {Aydın, Fatih},
  title   = {Unsupervised instance selection via conjectural hyperrectangles},
  journal = {Neural Computing and Applications},
  volume  = {35},
  pages   = {5335--5349},
  year    = {2023},
  doi     = {10.1007/s00521-022-07974-z}
}

@article{aydin2023bldod,
  author  = {Aydın, Fatih},
  title   = {Boundary-aware local density-based outlier detection},
  journal = {Information Sciences},
  volume  = {647},
  pages   = {119520},
  year    = {2023},
  doi     = {10.1016/j.ins.2023.119520}
}
```

## License

MIT
