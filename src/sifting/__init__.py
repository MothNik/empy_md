"""
Module :mod:`sifting`

This module implements sifting algorithms for the Empirical Mode Decomposition, i.e.,
the process of extracting an intrinsic mode function (IMF) from a signal.

"""

# === Imports ===

from ._empirical_optimal_envelope import (  # noqa: F401
    empirical_optimal_envelope,
    empirical_optimal_envelope_imf,
)
from ._interpolate import interp_akima, interp_cubic_spline, interp_pchip  # noqa: F401
from ._minmax import (  # noqa: F401
    find_relative_extrema_indices,
    find_relative_minima_and_maxima_indices,
)
