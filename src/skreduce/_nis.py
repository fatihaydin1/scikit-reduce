"""Nimble Instance Selection (NIS)."""

import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length, validate_data

from ._utils import round_half_away_from_zero, safe_take


class NIS(BaseEstimator):
    """Nimble Instance Selection (NIS).

    An unsupervised instance selection method based on conjectural
    hyperrectangles. Each feature is shifted by its minimum, divided by its
    standard deviation, multiplied by ``alpha`` and rounded to the nearest
    integer. Every combination of rounded values defines a hyperrectangular
    cell, and only the first instance that falls into each occupied cell is
    kept. Labels are not used, so the method works for classification,
    regression and unlabeled data alike.

    NIS implements ``fit_resample`` and can therefore be used as a sampling
    step in an :class:`imblearn.pipeline.Pipeline`.

    Parameters
    ----------
    alpha : float, default=1.0
        Scaling rate (``alpha`` in the paper). Cell edges are
        ``std / alpha`` long, so smaller values give larger cells and a
        higher reduction rate, and larger values keep more instances.

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_selected,)
        Indices of the selected instances in the training data, in
        increasing order.

    data_min_ : ndarray of shape (n_features,)
        Per-feature minimum of the training data, ignoring missing values.

    scale_ : ndarray of shape (n_features,)
        Per-feature sample standard deviation (``ddof=1``) of the training
        data, ignoring missing values.

    reduction_rate_ : float
        Fraction of the training instances that were removed.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when ``X``
        has feature names that are all strings.

    Notes
    -----
    The implementation reproduces the reference MATLAB code by the author:
    standard deviations use ``ddof=1``, ties are rounded away from zero,
    missing values (and features with zero variance) are mapped to cell 0,
    and the first occurrence in each cell is kept. The time complexity is
    ``O(n_samples * n_features * log(n_samples))``.

    References
    ----------
    .. [1] F. Aydin, "Unsupervised instance selection via conjectural
       hyperrectangles," Neural Computing and Applications, 35,
       5335-5349, 2023. https://doi.org/10.1007/s00521-022-07974-z

    Examples
    --------
    >>> import numpy as np
    >>> from skreduce import NIS
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(1000, 2))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> X_res, y_res = NIS(alpha=1.0).fit_resample(X, y)
    >>> X_res.shape[0] < X.shape[0]
    True
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _check_params(self):
        alpha = self.alpha
        if (
            not isinstance(alpha, numbers.Real)
            or isinstance(alpha, bool)
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError(
                f"alpha must be a positive finite number, got {alpha!r}."
            )

    def _cells(self, X):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.alpha * ((X - self.data_min_) / self.scale_)
        z[np.isnan(z)] = 0.0
        cells = round_half_away_from_zero(z)
        # Integer cells make the row comparison exact (no -0.0 vs 0.0 issue).
        if np.all(np.isfinite(cells)) and np.max(np.abs(cells), initial=0) < 2**62:
            return cells.astype(np.int64)
        return cells + 0.0

    def fit(self, X, y=None):
        """Select the instances to keep.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Missing values are allowed.

        y : array-like of shape (n_samples,), default=None
            Ignored by the selection; checked for consistent length only.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_params()
        X_arr = validate_data(
            self, X, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        if y is not None:
            check_consistent_length(X_arr, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.data_min_ = np.nanmin(X_arr, axis=0)
            self.scale_ = np.nanstd(X_arr, axis=0, ddof=1)

        cells = self._cells(X_arr)
        _, first = np.unique(cells, axis=0, return_index=True)
        self.sample_indices_ = np.sort(first)
        self.reduction_rate_ = 1.0 - self.sample_indices_.size / X_arr.shape[0]
        return self

    def fit_resample(self, X, y=None):
        """Select instances and return the reduced data set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Missing values are allowed.

        y : array-like of shape (n_samples,), default=None
            Targets. They do not influence the selection; they are only
            reduced together with ``X``.

        Returns
        -------
        X_resampled : array-like of shape (n_selected, n_features)
            Selected instances. A pandas DataFrame stays a DataFrame.

        y_resampled : array-like of shape (n_selected,) or None
            Targets of the selected instances, or ``None`` if ``y`` is None.
        """
        self.fit(X, y)
        return safe_take(X, self.sample_indices_), safe_take(y, self.sample_indices_)
