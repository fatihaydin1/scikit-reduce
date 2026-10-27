"""Boundary-aware Local Density-based Outlier Detection (BLDOD)."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import BallTree, KDTree
from sklearn.utils.validation import check_is_fitted, validate_data

_ALGORITHMS = ("auto", "brute", "kd_tree", "ball_tree")
# Upper bound on the number of float64 values held at once by the brute-force
# search (2**24 values = 128 MiB).
_BRUTE_CHUNK_VALUES = 2**24


class BLDOD(OutlierMixin, BaseEstimator):
    """Boundary-aware Local Density-based Outlier Detection (BLDOD).

    For every point, BLDOD counts the training points that lie within a
    radius of ``std_coef * ||sigma||`` from it, where ``sigma`` is the vector
    of per-feature standard deviations of the training data. The counts are
    min-max normalised and inverted, so the outlier factor is 1 for the point
    with the fewest neighbours and 0 for the point with the most. The radius
    is motivated by Chebyshev's inequality in the paper.

    The estimator follows the scikit-learn outlier detection API: lower
    values of :meth:`score_samples` mean more abnormal, and :meth:`predict`
    returns -1 for outliers and +1 for inliers.

    Parameters
    ----------
    std_coef : float, default=1.0
        Coefficient of the standard deviation norm (``lambda`` in the paper)
        that sets the neighbourhood radius.

    contamination : 'auto' or float, default='auto'
        Proportion of outliers in the data, used to set the threshold of
        :meth:`predict`. With ``'auto'``, points whose outlier factor is
        above 0.9 are labelled outliers, the threshold used in the paper.
        A float must be in (0, 0.5].

    algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, default='auto'
        Algorithm used to count the neighbours within the radius. ``'auto'``
        uses a KD-tree for up to 15 features and brute force otherwise. All
        choices give the same counts up to floating-point rounding at the
        boundary.

    Attributes
    ----------
    outlier_factor_ : ndarray of shape (n_samples,)
        Outlier factor of each training point, in [0, 1]. Higher means more
        abnormal. This is the score reported in the paper.

    n_neighbors_ : ndarray of shape (n_samples,)
        Number of training points within the radius of each training point,
        the point itself included.

    radius_ : float
        Neighbourhood radius, ``std_coef * ||sigma||``.

    offset_ : float
        Offset used to define the decision function from the raw scores:
        ``decision_function = score_samples - offset_``.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when ``X``
        has feature names that are all strings.

    Notes
    -----
    On the training data the scores reproduce the reference MATLAB code by
    the author: standard deviations use ``ddof=1``, distances are Euclidean
    and a point counts as a neighbour when its distance is less than or
    equal to the radius. For new data, :meth:`score_samples` counts the
    training points within the same radius and normalises the counts with
    the minimum and maximum seen during :meth:`fit`, so the factor of a new
    point can be above 1. If all training points have the same count, every
    factor is 0. The brute-force search takes ``O(n_samples**2 * n_features)``
    time.

    References
    ----------
    .. [1] F. Aydin, "Boundary-aware local density-based outlier detection,"
       Information Sciences, 647, 119520, 2023.
       https://doi.org/10.1016/j.ins.2023.119520

    Examples
    --------
    >>> import numpy as np
    >>> from skreduce import BLDOD
    >>> rng = np.random.RandomState(0)
    >>> X = np.vstack([rng.normal(size=(100, 2)), [[6.0, 6.0]]])
    >>> detector = BLDOD().fit(X)
    >>> int(np.argmax(detector.outlier_factor_))
    100
    """

    def __init__(self, std_coef=1.0, contamination="auto", algorithm="auto"):
        self.std_coef = std_coef
        self.contamination = contamination
        self.algorithm = algorithm

    def _check_params(self):
        c = self.std_coef
        if (
            not isinstance(c, numbers.Real)
            or isinstance(c, bool)
            or not np.isfinite(c)
            or c <= 0
        ):
            raise ValueError(f"std_coef must be a positive finite number, got {c!r}.")
        cont = self.contamination
        if isinstance(cont, str):
            if cont != "auto":
                raise ValueError(
                    f"contamination must be 'auto' or a float in (0, 0.5], got {cont!r}."
                )
        elif not isinstance(cont, numbers.Real) or isinstance(cont, bool) or not (0 < cont <= 0.5):
            raise ValueError(
                f"contamination must be 'auto' or a float in (0, 0.5], got {cont!r}."
            )
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {_ALGORITHMS}, got {self.algorithm!r}."
            )

    def _count_neighbors(self, Q):
        """Count training points within ``radius_`` of each row of ``Q``."""
        if self._tree is not None:
            return self._tree.query_radius(Q, r=self.radius_, count_only=True).astype(np.intp)
        fit_X = self._fit_X
        counts = np.empty(Q.shape[0], dtype=np.intp)
        step = max(1, _BRUTE_CHUNK_VALUES // max(1, fit_X.size))
        for start in range(0, Q.shape[0], step):
            diff = Q[start : start + step, None, :] - fit_X[None, :, :]
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
            counts[start : start + step] = np.count_nonzero(dist <= self.radius_, axis=1)
        return counts

    def _factor(self, counts):
        spread = self._n_max - self._n_min
        if spread == 0:
            return np.zeros(counts.shape[0], dtype=np.float64)
        return (self._n_max - counts) / spread

    def fit(self, X, y=None):
        """Fit the detector on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_params()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)

        self.radius_ = float(self.std_coef * np.linalg.norm(X.std(axis=0, ddof=1)))
        algorithm = self.algorithm
        if algorithm == "auto":
            algorithm = "kd_tree" if X.shape[1] <= 15 else "brute"
        self._fit_X = X
        if algorithm == "kd_tree":
            self._tree = KDTree(X)
        elif algorithm == "ball_tree":
            self._tree = BallTree(X)
        else:
            self._tree = None

        counts = self._count_neighbors(X)
        self._n_min, self._n_max = int(counts.min()), int(counts.max())
        self.n_neighbors_ = counts
        self.outlier_factor_ = self._factor(counts)

        if self.contamination == "auto":
            self.offset_ = -0.9
        else:
            self.offset_ = float(
                np.percentile(-self.outlier_factor_, 100.0 * self.contamination)
            )
        return self

    def score_samples(self, X):
        """Opposite of the outlier factor of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Opposite of the outlier factor; the lower, the more abnormal.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return -self._factor(self._count_neighbors(X))

    def decision_function(self, X):
        """Shifted opposite of the outlier factor.

        Negative values are outliers and non-negative values are inliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            ``score_samples(X) - offset_``.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """Predict whether each sample is an outlier (-1) or an inlier (+1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            -1 for outliers and +1 for inliers.
        """
        return np.where(self.decision_function(X) < 0, -1, 1)
