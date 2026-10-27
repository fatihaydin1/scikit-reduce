"""scikit-reduce: scikit-learn compatible data reduction methods.

- :class:`NIS`: Nimble Instance Selection, an unsupervised instance
  selection method with a ``fit_resample`` interface.
- :class:`BLDOD`: Boundary-aware Local Density-based Outlier Detection,
  an outlier detector with the scikit-learn outlier detection interface.
"""

from ._bldod import BLDOD
from ._nis import NIS

__version__ = "0.1.0"

__all__ = ["BLDOD", "NIS", "__version__"]
