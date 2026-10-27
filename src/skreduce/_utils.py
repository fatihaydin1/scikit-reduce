"""Small helpers shared by the estimators."""

import numpy as np


def round_half_away_from_zero(z):
    """Round to the nearest integer, with ties rounded away from zero.

    This matches MATLAB's ``round`` (used by the reference implementations)
    and differs from :func:`numpy.round`, which rounds ties to the nearest
    even number. Computing the fractional part explicitly avoids the
    floating-point error of ``floor(z + 0.5)`` for values just below a tie.
    """
    a = np.abs(z)
    r = np.floor(a)
    r += (a - r) >= 0.5
    return np.copysign(r, z)


def safe_take(data, indices):
    """Index the rows of an array, a pandas object or a list.

    pandas DataFrames and Series keep their type and index; everything else
    is converted with :func:`numpy.asarray`.
    """
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]
