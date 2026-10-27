import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier

from skreduce import NIS
from skreduce._utils import round_half_away_from_zero


def _fixture():
    """Data with ties, a constant feature and missing values."""
    rng = np.random.RandomState(42)
    X = np.round(rng.normal(size=(40, 3)), 1)
    X = np.hstack([X, np.full((40, 1), 7.0)])
    X[[3, 11, 25], [0, 2, 1]] = np.nan
    return X


# Indices (0-based) selected by the reference MATLAB implementation, NIS.m,
# run in GNU Octave on the fixture above.
REFERENCE = {
    0.5: [0, 1, 3, 4, 5, 6, 10, 11, 12, 15, 19, 24, 25, 27, 36],
    1.0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 18, 19, 21, 23, 24,
          25, 26, 27, 29, 32, 34, 35, 37, 39],
    2.0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
          38, 39],
}


@pytest.mark.parametrize("alpha", sorted(REFERENCE))
def test_matches_reference_implementation(alpha):
    nis = NIS(alpha=alpha).fit(_fixture())
    np.testing.assert_array_equal(nis.sample_indices_, REFERENCE[alpha])


def test_round_half_away_from_zero():
    z = np.array([0.5, 1.5, 2.5, 0.49999999999999994, 2.4999999, -0.5, -1.5])
    np.testing.assert_array_equal(
        round_half_away_from_zero(z), [1, 2, 3, 0, 2, -1, -2]
    )


def test_keeps_first_instance_of_each_cell():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [0.0, 0.1]])
    nis = NIS(alpha=1.0).fit(X)
    np.testing.assert_array_equal(nis.sample_indices_, [0, 2])
    assert nis.reduction_rate_ == pytest.approx(0.6)


def test_fit_resample_reduces_x_and_y_together():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 2))
    y = np.arange(300)
    X_res, y_res = NIS().fit_resample(X, y)
    np.testing.assert_array_equal(X_res, X[y_res])
    assert X_res.shape[0] < X.shape[0]


def test_fit_resample_without_y():
    X = np.random.RandomState(0).normal(size=(50, 2))
    X_res, y_res = NIS().fit_resample(X)
    assert y_res is None
    assert X_res.shape[1] == 2


def test_pandas_objects_keep_their_type_and_index():
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=range(100, 200))
    y = pd.Series(rng.randint(0, 2, size=100), index=X.index, name="target")
    nis = NIS()
    X_res, y_res = nis.fit_resample(X, y)
    assert isinstance(X_res, pd.DataFrame) and isinstance(y_res, pd.Series)
    assert list(X_res.index) == list(y_res.index)
    np.testing.assert_array_equal(nis.feature_names_in_, ["a", "b"])


def test_constant_and_missing_values_map_to_cell_zero():
    X = np.array([[1.0, 3.0], [np.nan, 3.0], [1.0, 3.0]])
    # Rows 0 and 2 are identical; row 1 has a missing value in a feature
    # whose observed values give cell 0, so all rows share one cell.
    assert NIS().fit(X).sample_indices_.tolist() == [0]


@pytest.mark.parametrize("alpha", [0, -1, np.inf, np.nan, "1", True])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        NIS(alpha=alpha).fit(np.ones((3, 2)))


def test_inconsistent_lengths():
    with pytest.raises(ValueError):
        NIS().fit_resample(np.ones((5, 2)), np.ones(4))


def test_single_sample():
    X_res, _ = NIS().fit_resample(np.array([[1.0, 2.0]]))
    assert X_res.shape == (1, 2)


def test_clone_and_pickle():
    nis = clone(NIS(alpha=2.5))
    assert nis.get_params() == {"alpha": 2.5}
    X = np.random.RandomState(0).normal(size=(30, 2))
    nis.fit(X)
    restored = pickle.loads(pickle.dumps(nis))
    np.testing.assert_array_equal(restored.sample_indices_, nis.sample_indices_)


def test_imblearn_pipeline():
    pipeline = pytest.importorskip("imblearn.pipeline")
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 2))
    y = (X[:, 0] > 0).astype(int)
    pipe = pipeline.make_pipeline(NIS(alpha=1.0), KNeighborsClassifier(n_neighbors=1))
    pipe.fit(X, y)
    knn = pipe[-1]
    assert knn.n_samples_fit_ == pipe[0].sample_indices_.size < X.shape[0]
    assert pipe.predict(X).shape == y.shape
