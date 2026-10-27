import numpy as np
import pytest

from skreduce import BLDOD


def _fixture():
    """Two clusters and two isolated points, rounded to create ties."""
    rng = np.random.RandomState(42)
    rng.normal(size=(40, 3))  # keep the stream aligned with the NIS fixture
    X = np.vstack(
        [
            rng.normal(0, 1, size=(20, 2)),
            rng.normal(5, 0.5, size=(15, 2)),
            [[10, -4], [-6, 6]],
        ]
    )
    return np.round(X, 2)


# Outlier factors computed by the reference MATLAB implementation, BLDOD.m,
# run in GNU Octave on the fixture above.
_A, _B, _C = 1 / 3, 2 / 9, 0.26315789473684215
REFERENCE = {
    0.5: [_A, 11 / 18, 2 / 3, 1 / 6, 1 / 18, 5 / 18, _B, _A, 1 / 9, _A, 5 / 18,
          _B, 1 / 18, 7 / 18, 0, 0, 1 / 9, 1 / 18, 4 / 9] + [_B] * 16 + [1, 1],
    1.0: [0.0] * 20 + [_C] * 15 + [1, 1],
}


def test_fixture_shape():
    assert _fixture().shape == (37, 2)


@pytest.mark.parametrize("std_coef", sorted(REFERENCE))
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree", "auto"])
def test_matches_reference_implementation(std_coef, algorithm):
    det = BLDOD(std_coef=std_coef, algorithm=algorithm).fit(_fixture())
    np.testing.assert_allclose(det.outlier_factor_, REFERENCE[std_coef], atol=1e-12)


def test_algorithms_agree_on_random_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 5))
    counts = [
        BLDOD(algorithm=a).fit(X).n_neighbors_ for a in ["brute", "kd_tree", "ball_tree"]
    ]
    np.testing.assert_array_equal(counts[0], counts[1])
    np.testing.assert_array_equal(counts[0], counts[2])


def test_radius_and_self_count():
    X = np.random.RandomState(0).normal(size=(50, 3))
    det = BLDOD(std_coef=0.7).fit(X)
    assert det.radius_ == pytest.approx(0.7 * np.linalg.norm(X.std(axis=0, ddof=1)))
    assert det.n_neighbors_.min() >= 1


def test_isolated_point_is_the_strongest_outlier():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(200, 2)), [[8.0, 8.0]]])
    det = BLDOD().fit(X)
    assert np.argmax(det.outlier_factor_) == 200
    assert det.outlier_factor_[200] == 1.0
    assert det.predict(X)[200] == -1


def test_score_samples_on_training_data_matches_fit():
    X = np.random.RandomState(1).normal(size=(80, 3))
    det = BLDOD().fit(X)
    np.testing.assert_allclose(det.score_samples(X), -det.outlier_factor_)
    np.testing.assert_array_equal(det.fit_predict(X), det.predict(X))


def test_auto_contamination_uses_paper_threshold():
    X = _fixture()
    det = BLDOD(std_coef=0.5).fit(X)
    expected = np.where(det.outlier_factor_ > 0.9, -1, 1)
    np.testing.assert_array_equal(det.predict(X), expected)


def test_float_contamination():
    X = np.random.RandomState(0).normal(size=(400, 2))
    pred = BLDOD(contamination=0.1).fit(X).predict(X)
    assert 0.05 <= np.mean(pred == -1) <= 0.1


def test_new_points_are_scored_against_training_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2))
    det = BLDOD().fit(X)
    far, near = det.score_samples([[20.0, 20.0], [0.0, 0.0]])
    assert far < near
    assert det.predict([[20.0, 20.0]])[0] == -1


def test_constant_data_has_zero_factors():
    det = BLDOD().fit(np.ones((10, 2)))
    np.testing.assert_array_equal(det.outlier_factor_, np.zeros(10))


@pytest.mark.parametrize(
    "params",
    [
        {"std_coef": 0},
        {"std_coef": -1},
        {"std_coef": np.nan},
        {"contamination": 0.0},
        {"contamination": 0.6},
        {"contamination": "high"},
        {"algorithm": "cosine"},
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        BLDOD(**params).fit(np.random.RandomState(0).normal(size=(10, 2)))


def test_single_sample_raises():
    with pytest.raises(ValueError, match="1 sample"):
        BLDOD().fit([[1.0, 2.0]])
