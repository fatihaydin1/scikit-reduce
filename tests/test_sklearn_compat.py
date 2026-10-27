"""scikit-learn's estimator API checks."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from skreduce import BLDOD, NIS


@parametrize_with_checks(
    [NIS(), BLDOD(), BLDOD(contamination=0.1, algorithm="brute")]
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
